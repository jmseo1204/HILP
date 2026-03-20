"""
Train Downstream Goal-Conditioned Value Function using Frozen Dual Representations.

Phase 2 of arXiv:2510.06714:
  V_down(s, phi(g))  — standard IQL-style value function where phi is FROZEN.

After Phase 1 (train_dual_ogbench.py), load the dual repr checkpoint and train
a new value MLP V(s, phi(g)) = MLP([s, phi(g)]) with the same expectile loss.
Checkpoints saved to --save_dir/params_STEP.pkl.

Usage:
  python train_gcvf_dual_ogbench.py \
      --env_name=pointmaze-large-stitch-v0 \
      --dual_restore_path=exp/dual_repr \
      --dual_restore_epoch=1000000 \
      --save_dir=exp/gcvf_dual
"""

import copy
import os
import sys
import glob
import pickle
import functools
import dataclasses
from pathlib import Path
from typing import Sequence, Any

import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
from absl import app, flags
import tqdm

# ---- Shared utilities from hilp_ogbench.py ----------------------------------
_SCOTS_DIR = Path(__file__).parent.parent.parent / 'scots' / 'scripts' / 'HILP'
sys.path.insert(0, str(_SCOTS_DIR))
from hilp_ogbench import (
    Dataset, GCDataset, ModuleDict, TrainState,
    save_agent, restore_agent, expectile_loss,
    LayerNormRepresentation,
)
# ---- Dual repr agent --------------------------------------------------------
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from train_dual_ogbench import DualHILP

# -----------------------------------------------------------------------------

FLAGS = flags.FLAGS
flags.DEFINE_string ('env_name',           'pointmaze-large-stitch-v0', 'OGBench env.')
flags.DEFINE_string ('dual_restore_path',  'exp/dual_repr',  'Path to Phase-1 checkpoint dir.')
flags.DEFINE_integer('dual_restore_epoch', 1000000,          'Phase-1 checkpoint step to load.')
flags.DEFINE_float  ('lr',                 3e-4,    'Learning rate for downstream GCVF.')
flags.DEFINE_multi_integer('value_hidden_dims', [512, 512, 512], 'Hidden dims.')
flags.DEFINE_integer('skill_dim',          32,      'Must match dual repr skill_dim.')
flags.DEFINE_float  ('discount',           0.99,    'Discount.')
flags.DEFINE_float  ('tau',                0.005,   'Target EMA rate.')
flags.DEFINE_float  ('expectile',          0.95,    'Expectile.')
flags.DEFINE_integer('use_layer_norm',     1,       '1 = LayerNorm.')
flags.DEFINE_integer('batch_size',         1024,    'Batch size.')
flags.DEFINE_integer('train_steps',        500000,  'Training steps.')
flags.DEFINE_integer('save_interval',      100000,  'Checkpoint interval.')
flags.DEFINE_integer('log_interval',       1000,    'Log interval.')
flags.DEFINE_string ('save_dir',           'exp/gcvf_dual', 'Output dir.')
flags.DEFINE_integer('seed',               0,       'Seed.')
flags.DEFINE_float  ('p_currgoal',         0.0,     '')
flags.DEFINE_float  ('p_trajgoal',         0.625,   '')
flags.DEFINE_float  ('p_randomgoal',       0.375,   '')
flags.DEFINE_integer('geom_sample',        1,       '')


# ======================== Network =============================================

class GoalConditionedMLP(nn.Module):
    """Simple MLP value function: V(s, goal_repr) = MLP([s || goal_repr]) -> scalar."""
    hidden_dims:    tuple = (512, 512, 512)
    use_layer_norm: bool  = True
    ensemble:       bool  = True

    def setup(self):
        self.value_net = LayerNormRepresentation(
            (*self.hidden_dims, 1),
            activate_final=False,
            use_layer_norm=self.use_layer_norm,
            ensemble=self.ensemble,
        )

    def __call__(self, observations, goals):
        x = jnp.concatenate([observations, goals], axis=-1)
        v = self.value_net(x)           # (2, B, 1) or (B, 1)
        return v.squeeze(-1)            # (2, B) or (B,)


# ======================== Agent ===============================================

class GCVFDual(flax.struct.PyTreeNode):
    """
    Goal-Conditioned Value Function trained on top of frozen dual phi(g).
    V_down(s, phi(g)) = MLP([s, phi(g)]).
    """
    rng:     Any
    network: TrainState
    config:  dict = flax.struct.field(pytree_node=False)

    # ---- loss ----------------------------------------------------------------
    def value_loss(self, batch, network_params, phi_g):
        """
        phi_g: (B, skill_dim) — pre-computed FROZEN dual goal representations.
        """
        (nv1, nv2) = self.network.select('target_value')(
            batch['next_observations'], phi_g)
        next_v = jnp.minimum(nv1, nv2)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v

        (v1_t, v2_t) = self.network.select('target_value')(
            batch['observations'], phi_g)
        adv = q - (v1_t + v2_t) / 2

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * nv1
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * nv2
        (v1, v2) = self.network.select('value')(
            batch['observations'], phi_g, params=network_params)
        v = (v1 + v2) / 2

        loss = (expectile_loss(adv, q1 - v1, self.config['expectile']).mean() +
                expectile_loss(adv, q2 - v2, self.config['expectile']).mean())
        return loss, {
            'value_loss':  loss,
            'v_mean':      v.mean(),
            'v_max':       v.max(),
            'v_min':       v.min(),
            'adv_mean':    adv.mean(),
            'accept_prob': (adv >= 0).mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, phi_g):
        loss, info = self.value_loss(batch, grad_params, phi_g)
        return loss, {f'gcvf/{k}': v for k, v in info.items()}

    # ---- update --------------------------------------------------------------
    def target_update(self, network, module_name):
        new_tp = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_tp

    @jax.jit
    def update(self, batch, phi_g):
        """phi_g: (B, skill_dim) frozen dual goal repr."""
        new_network, info = self.network.apply_loss_fn(
            loss_fn=lambda p: self.total_loss(batch, p, phi_g))
        self.target_update(new_network, 'value')
        return self.replace(network=new_network), info

    # ---- inference -----------------------------------------------------------
    @jax.jit
    def get_value(self, observations: np.ndarray, phi_g: np.ndarray) -> jnp.ndarray:
        """V(s, phi(g)): scalar value for each (s, phi(g)) pair."""
        v1, v2 = self.network.select('value')(observations, phi_g)
        return (v1 + v2) / 2

    # ---- factory -------------------------------------------------------------
    @classmethod
    def create(cls, seed, ex_observations, skill_dim=32, lr=3e-4,
               value_hidden_dims=(512,512,512), discount=0.99, tau=0.005,
               expectile=0.95, use_layer_norm=1, **kwargs):
        print('GCVFDual.create — extra kwargs:', kwargs)
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        obs_dim  = ex_observations.shape[-1]
        ex_phi_g = np.zeros((1, skill_dim), dtype=np.float32)

        value_def = GoalConditionedMLP(
            hidden_dims    = tuple(value_hidden_dims),
            use_layer_norm = bool(use_layer_norm),
            ensemble       = True,
        )
        network_info = dict(
            value       =(value_def,             (ex_observations, ex_phi_g)),
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_phi_g)),
        )
        networks     = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def    = ModuleDict(networks)
        network_tx     = optax.adam(learning_rate=lr)
        network_params = network_def.init(init_rng, **network_args)['params']
        network        = TrainState.create(network_def, network_params, tx=network_tx)
        network.params['modules_target_value'] = network.params['modules_value']

        return cls(rng, network=network,
                   config=flax.core.FrozenDict(
                       discount=discount, tau=tau, expectile=expectile,
                       skill_dim=skill_dim))


# ======================== Main ================================================

def main(_):
    import ogbench
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    # ---- Load dataset --------------------------------------------------------
    print(f'[GCVFDual] env={FLAGS.env_name}  save_dir={FLAGS.save_dir}')
    _, dataset, _ = ogbench.make_env_and_datasets(
        FLAGS.env_name, compact_dataset=False)

    train_dataset = Dataset.create(
        observations      = dataset['observations'].astype(np.float32),
        actions           = dataset['actions'].astype(np.float32),
        next_observations = dataset['next_observations'].astype(np.float32),
        terminals         = dataset['terminals'].astype(np.float32),
    )
    gc_dataset = GCDataset(
        train_dataset,
        p_randomgoal = FLAGS.p_randomgoal,
        p_trajgoal   = FLAGS.p_trajgoal,
        p_currgoal   = FLAGS.p_currgoal,
        discount     = FLAGS.discount,
        geom_sample  = FLAGS.geom_sample,
        terminal_key = 'terminals',
    )

    ex_obs = train_dataset['observations'][:1]

    # ---- Restore frozen dual agent (Phase 1) ---------------------------------
    dual_agent = DualHILP.create(
        seed              = FLAGS.seed,
        ex_observations   = ex_obs,
        lr                = FLAGS.lr,
        value_hidden_dims = tuple(FLAGS.value_hidden_dims),
        discount          = FLAGS.discount,
        tau               = FLAGS.tau,
        expectile         = FLAGS.expectile,
        use_layer_norm    = FLAGS.use_layer_norm,
        skill_dim         = FLAGS.skill_dim,
    )
    dual_agent = restore_agent(dual_agent, FLAGS.dual_restore_path, FLAGS.dual_restore_epoch)
    print(f'[GCVFDual] Loaded frozen dual repr from {FLAGS.dual_restore_path}')

    # ---- Create downstream GCVF agent (Phase 2) ------------------------------
    gcvf_agent = GCVFDual.create(
        seed              = FLAGS.seed,
        ex_observations   = ex_obs,
        skill_dim         = FLAGS.skill_dim,
        lr                = FLAGS.lr,
        value_hidden_dims = tuple(FLAGS.value_hidden_dims),
        discount          = FLAGS.discount,
        tau               = FLAGS.tau,
        expectile         = FLAGS.expectile,
        use_layer_norm    = FLAGS.use_layer_norm,
    )

    # ---- Training loop -------------------------------------------------------
    for step in tqdm.tqdm(range(1, FLAGS.train_steps + 1),
                          smoothing=0.1, dynamic_ncols=True):
        batch = gc_dataset.sample(FLAGS.batch_size)

        # Compute frozen phi(g) from dual repr
        phi_g = np.array(dual_agent.get_phi_goal(batch['goals']))   # (B, skill_dim)

        gcvf_agent, info = gcvf_agent.update(batch, phi_g)

        if step % FLAGS.log_interval == 0:
            log_str = '  '.join(f'{k}={float(v):.4f}' for k, v in info.items())
            tqdm.tqdm.write(f'[step {step:>8d}] {log_str}')

        if step % FLAGS.save_interval == 0:
            save_agent(gcvf_agent, FLAGS.save_dir, step)

    print('[GCVFDual] Training complete.')


if __name__ == '__main__':
    app.run(main)
