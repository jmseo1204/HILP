"""
Train Dual Goal Representations on OGBench environments.

Phase 1 of arXiv:2510.06714:
  V(s, g) = psi(s)^T phi(g)   (inner product, separate state/goal encoders)
  psi: state encoder, phi: goal encoder (the "dual" representation)

Trained with IQL-style expectile regression on temporal distances.
Checkpoints are saved to --save_dir/params_STEP.pkl.
"""

import copy
import os
import sys
import glob
import pickle
import functools
import dataclasses
from datetime import datetime
from pathlib import Path
from typing import Sequence, Dict, Any, Callable

import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
from flax.core.frozen_dict import FrozenDict
from absl import app, flags
import tqdm

# ---- Pull utility classes from hilp_ogbench.py --------------------------------
_SCOTS_DIR = Path(__file__).parent.parent.parent / 'scots' / 'scripts' / 'HILP'
sys.path.insert(0, str(_SCOTS_DIR))
from hilp_ogbench import (
    Dataset, GCDataset, ModuleDict, TrainState,
    save_agent, restore_agent, expectile_loss,
    ensemblize, default_init, MLP, LayerNormRepresentation,
)
# -------------------------------------------------------------------------------

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name',      'pointmaze-large-stitch-v0', 'OGBench environment name.')
flags.DEFINE_float ('lr',            3e-4,    'Learning rate.')
flags.DEFINE_integer('skill_dim',    32,      'Dimension of psi / phi representations.')
flags.DEFINE_multi_integer('value_hidden_dims', [512, 512, 512], 'Value MLP hidden dims.')
flags.DEFINE_float ('discount',      0.99,    'Discount factor.')
flags.DEFINE_float ('tau',           0.005,   'Target network EMA rate.')
flags.DEFINE_float ('expectile',     0.95,    'IQL expectile.')
flags.DEFINE_integer('use_layer_norm', 1,     '1 = use LayerNorm.')
flags.DEFINE_integer('batch_size',   1024,    'Training batch size.')
flags.DEFINE_integer('train_steps',  1000000, 'Total training steps.')
flags.DEFINE_integer('save_interval', 100000, 'Checkpoint every N steps.')
flags.DEFINE_integer('log_interval',  1000,   'Log every N steps.')
flags.DEFINE_string ('save_dir',     'exp/dual_repr', 'Checkpoint output directory.')
flags.DEFINE_integer('seed',         0,       'Random seed.')
flags.DEFINE_float  ('p_currgoal',   0.0,     'Fraction current-state goals.')
flags.DEFINE_float  ('p_trajgoal',   0.625,   'Fraction trajectory goals.')
flags.DEFINE_float  ('p_randomgoal', 0.375,   'Fraction random goals.')
flags.DEFINE_integer('geom_sample',  1,       '1 = geometric goal sampling.')


# ======================== Network =============================================

class DualGoalPhiValue(nn.Module):
    """
    Dual goal representation value function: V(s, g) = psi(s)^T phi(g).
      psi: state encoder   (s -> R^skill_dim)
      phi: goal encoder    (g -> R^skill_dim) — the "dual" representation
    """
    hidden_dims: tuple = (512, 512, 512)
    skill_dim:   int   = 32
    use_layer_norm: bool = True
    ensemble:    bool  = True

    def setup(self):
        self.psi = LayerNormRepresentation(
            (*self.hidden_dims, self.skill_dim),
            activate_final=False,
            use_layer_norm=self.use_layer_norm,
            ensemble=self.ensemble,
        )
        self.phi = LayerNormRepresentation(
            (*self.hidden_dims, self.skill_dim),
            activate_final=False,
            use_layer_norm=self.use_layer_norm,
            ensemble=self.ensemble,
        )

    def get_phi(self, goals):
        """phi(g): dual goal representation.  Returns shape (B, skill_dim)."""
        out = self.phi(goals)
        return out[0] if self.ensemble else out   # first ensemble member

    def get_psi(self, observations):
        """psi(s): state representation.  Returns shape (B, skill_dim)."""
        out = self.psi(observations)
        return out[0] if self.ensemble else out

    def __call__(self, observations, goals=None):
        psi_s = self.psi(observations)          # (2, B, D)
        phi_g = self.phi(goals)                  # (2, B, D)
        return (psi_s * phi_g).sum(axis=-1)      # (2, B)


# ======================== Agent ===============================================

class DualHILP(flax.struct.PyTreeNode):
    rng:     Any
    network: TrainState
    config:  dict = flax.struct.field(pytree_node=False)

    # ---- loss ----------------------------------------------------------------
    def value_loss(self, batch, network_params):
        (nv1, nv2) = self.network.select('target_value')(
            batch['next_observations'], batch['goals'])
        next_v = jnp.minimum(nv1, nv2)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v

        (v1_t, v2_t) = self.network.select('target_value')(
            batch['observations'], batch['goals'])
        v_t  = (v1_t + v2_t) / 2
        adv  = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * nv1
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * nv2
        (v1, v2) = self.network.select('value')(
            batch['observations'], batch['goals'], params=network_params)
        v = (v1 + v2) / 2

        loss = (expectile_loss(adv, q1 - v1, self.config['expectile']).mean() +
                expectile_loss(adv, q2 - v2, self.config['expectile']).mean())
        return loss, {
            'value_loss':    loss,
            'v_mean':        v.mean(),
            'v_max':         v.max(),
            'v_min':         v.min(),
            'adv_mean':      adv.mean(),
            'accept_prob':   (adv >= 0).mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params):
        loss, info = self.value_loss(batch, grad_params)
        return loss, {f'value/{k}': v for k, v in info.items()}

    # ---- update --------------------------------------------------------------
    def target_update(self, network, module_name):
        new_tp = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_tp

    @jax.jit
    def update(self, batch):
        new_network, info = self.network.apply_loss_fn(
            loss_fn=lambda p: self.total_loss(batch, p))
        self.target_update(new_network, 'value')
        return self.replace(network=new_network), info

    # ---- inference -----------------------------------------------------------
    @jax.jit
    def get_phi_goal(self, goals: np.ndarray) -> jnp.ndarray:
        """phi(g): dual goal representation for direction/skill computation."""
        m  = self.network.model_def.modules['value']
        ps = self.network.params['modules_value']
        return m.apply({'params': ps}, goals, method=m.get_phi)

    @jax.jit
    def get_psi(self, observations: np.ndarray) -> jnp.ndarray:
        """psi(s): state representation."""
        m  = self.network.model_def.modules['value']
        ps = self.network.params['modules_value']
        return m.apply({'params': ps}, observations, method=m.get_psi)

    # For API compatibility with the visualization code that calls get_phi(obs)
    @jax.jit
    def get_phi(self, observations: np.ndarray) -> jnp.ndarray:
        return self.get_psi(observations)

    # ---- factory -------------------------------------------------------------
    @classmethod
    def create(cls, seed, ex_observations, lr=3e-4,
               value_hidden_dims=(512,512,512), discount=0.99, tau=0.005,
               expectile=0.95, use_layer_norm=1, skill_dim=32, **kwargs):
        print('DualHILP.create — extra kwargs:', kwargs)
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        value_def = DualGoalPhiValue(
            hidden_dims=tuple(value_hidden_dims),
            skill_dim=skill_dim,
            use_layer_norm=bool(use_layer_norm),
            ensemble=True,
        )
        network_info = dict(
            value       =(value_def,             (ex_observations, ex_observations)),
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_observations)),
        )
        networks    = {k: v[0] for k, v in network_info.items()}
        network_args= {k: v[1] for k, v in network_info.items()}

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
    print(f'[DualHILP] env={FLAGS.env_name}  save_dir={FLAGS.save_dir}')

    env, dataset, _ = ogbench.make_env_and_datasets(
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
    agent  = DualHILP.create(
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

    for step in tqdm.tqdm(range(1, FLAGS.train_steps + 1),
                          smoothing=0.1, dynamic_ncols=True):
        batch = gc_dataset.sample(FLAGS.batch_size)
        agent, info = agent.update(batch)

        if step % FLAGS.log_interval == 0:
            log_str = '  '.join(f'{k}={float(v):.4f}' for k, v in info.items())
            tqdm.tqdm.write(f'[step {step:>8d}] {log_str}')

        if step % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, step)

    print('[DualHILP] Training complete.')


if __name__ == '__main__':
    app.run(main)
