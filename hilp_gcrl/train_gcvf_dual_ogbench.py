"""
Train Downstream Goal-Conditioned Value Function on Frozen Dual Representations.

Phase 2 of arXiv:2510.06714:
  V_down(s, phi(g)) = MLP([s, phi(g)])  —  phi is FROZEN from Phase 1.

All dependencies are inside hilp_gcrl/.
"""

import copy
import functools
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
from flax.core import unfreeze
from absl import app, flags
import tqdm
import wandb

# ---- hilp_gcrl internal imports ---------------------------------------------
_ROOT = Path(__file__).parent          # hilp_gcrl/
sys.path.insert(0, str(_ROOT))

from jaxrl_m.common import TrainState, shard_batch
from jaxrl_m.dataset import Dataset
from src.dataset_utils import GCDataset
from src.special_networks import GoalConditionedValue   # reuse: takes [obs, phi_g]
from src.agents.hilp import expectile_loss
from train_dual_ogbench import DualHILP, save_agent, restore_agent
# -----------------------------------------------------------------------------

FLAGS = flags.FLAGS


# ======================== Network ============================================

class GCVFNetwork(nn.Module):
    """
    Wraps GoalConditionedValue (value + target_value) for downstream GCVF.
    Input: [obs, phi_g]  where phi_g = frozen phi(g) from Phase 1.
    """
    networks: dict

    def value(self, observations, phi_g=None, **kwargs):
        return self.networks['value'](observations, phi_g, **kwargs)

    def target_value(self, observations, phi_g=None, **kwargs):
        return self.networks['target_value'](observations, phi_g, **kwargs)

    def __call__(self, observations, phi_g):
        return {
            'value':        self.value(observations, phi_g),
            'target_value': self.target_value(observations, phi_g),
        }


# ======================== Agent ==============================================

class GCVFDual(flax.struct.PyTreeNode):
    """
    Downstream GCVF trained on frozen dual repr.
    V_down(s, phi(g)) = GoalConditionedValue([s, phi(g)]).
    """
    network: TrainState
    config:  dict = flax.struct.field(pytree_node=False)

    def value_loss(self, batch, network_params, phi_g):
        """phi_g: (B, skill_dim) frozen dual goal representations."""
        (nv1, nv2) = self.network(
            batch['next_observations'], phi_g, method='target_value')
        next_v = jnp.minimum(nv1, nv2)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v

        (v1_t, v2_t) = self.network(
            batch['observations'], phi_g, method='target_value')
        adv = q - (v1_t + v2_t) / 2

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * nv1
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * nv2
        (v1, v2) = self.network(
            batch['observations'], phi_g,
            method='value', params=network_params)
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

    def total_loss(self, batch, grad_params, phi_g):
        loss, info = self.value_loss(batch, grad_params, phi_g)
        log = {f'gcvf/{k}': v for k, v in info.items()}
        log['loss'] = loss
        return loss, log

    def update(self, batch, phi_g, pmap_axis=None):
        new_network, info = self.network.apply_loss_fn(
            loss_fn=lambda p: self.total_loss(batch, p, phi_g), has_aux=True,
            pmap_axis=pmap_axis)

        new_tp = jax.tree.map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            new_network.params['networks_value'],
            new_network.params['networks_target_value'],
        )
        params = dict(new_network.params)
        params['networks_target_value'] = new_tp
        new_network = new_network.replace(params=params)

        return self.replace(network=new_network), info

    @jax.jit
    def get_value(self, observations: np.ndarray, phi_g: np.ndarray) -> jnp.ndarray:
        v1, v2 = self.network(observations, phi_g, method='value')
        return (v1 + v2) / 2

    @classmethod
    def create(cls, seed, ex_observations, skill_dim=32, lr=3e-4,
               value_hidden_dims=(512, 512, 512), discount=0.99, tau=0.005,
               expectile=0.95, use_layer_norm=1, **kwargs):
        print('GCVFDual.create — extra kwargs:', kwargs)
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        ex_phi_g = np.zeros((1, skill_dim), dtype=np.float32)

        # GoalConditionedValue concatenates [obs, goals] → MLP → scalar
        value_def = GoalConditionedValue(
            hidden_dims=tuple(value_hidden_dims),
            use_layer_norm=bool(use_layer_norm),
            ensemble=True,
        )
        network_def = GCVFNetwork(networks={
            'value':        value_def,
            'target_value': copy.deepcopy(value_def),
        })
        network_tx     = optax.adam(learning_rate=lr)
        network_params = unfreeze(network_def.init(
            init_rng, ex_observations, ex_phi_g)['params'])
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = dict(network.params)
        params['networks_target_value'] = params['networks_value']
        network = network.replace(params=params)

        return cls(network=network,
                   config=flax.core.FrozenDict(
                       discount=discount, tau=tau, expectile=expectile,
                       skill_dim=skill_dim))


# ======================== Main ===============================================

def main(_):
    import ogbench
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    # ---- Multi-GPU setup ----------------------------------------------------
    n_devices = jax.local_device_count()
    print(f'[GCVFDual] env={FLAGS.env_name}  save_dir={FLAGS.save_dir}')
    print(f'[GCVFDual] Using {n_devices} GPU(s): {jax.local_devices()}')
    assert FLAGS.batch_size % n_devices == 0, (
        f'batch_size ({FLAGS.batch_size}) must be divisible by n_devices ({n_devices})')

    # ---- WandB --------------------------------------------------------------
    if FLAGS.wandb_project:
        run_name = FLAGS.wandb_run_name or f'gcvf_dual_{FLAGS.env_name}'
        wandb.init(project=FLAGS.wandb_project, name=run_name,
                   config=FLAGS.flag_values_dict())
        print(f'[GCVFDual] WandB run: {run_name}  project: {FLAGS.wandb_project}')

    _, dataset, _ = ogbench.make_env_and_datasets(
        FLAGS.env_name, compact_dataset=False)

    train_data = Dataset(dict(
        observations      = dataset['observations'].astype(np.float32),
        next_observations = dataset['next_observations'].astype(np.float32),
        actions           = dataset['actions'].astype(np.float32),
        terminals         = dataset['terminals'].astype(np.float32),
    ))
    gc_dataset = GCDataset(
        train_data,
        p_randomgoal = FLAGS.p_randomgoal,
        p_trajgoal   = FLAGS.p_trajgoal,
        p_currgoal   = FLAGS.p_currgoal,
        discount     = FLAGS.discount,
        geom_sample  = FLAGS.geom_sample,
        terminal_key = 'terminals',
        reward_scale = 1.0,
        reward_shift = -1.0,
    )

    ex_obs = train_data['observations'][:1]

    # ---- Restore frozen Phase-1 dual agent ----------------------------------
    ex_act = train_data['actions'][:1]
    dual_agent = DualHILP.create(
        seed=FLAGS.seed, ex_observations=ex_obs, ex_actions=ex_act,
        value_hidden_dims=tuple(FLAGS.value_hidden_dims),
        discount=FLAGS.discount, tau=FLAGS.tau,
        expectile=FLAGS.expectile, use_layer_norm=FLAGS.use_layer_norm,
        skill_dim=FLAGS.skill_dim,
        aggregator=FLAGS.dual_aggregator,
        share_encoder=FLAGS.dual_share_encoder,
    )
    dual_agent = restore_agent(dual_agent, FLAGS.dual_restore_path, FLAGS.dual_restore_epoch)
    print(f'[GCVFDual] Loaded frozen dual repr from {FLAGS.dual_restore_path}')

    # ---- Create Phase-2 GCVF agent ------------------------------------------
    gcvf_agent = GCVFDual.create(
        seed=FLAGS.seed, ex_observations=ex_obs,
        skill_dim=FLAGS.skill_dim, lr=FLAGS.lr,
        value_hidden_dims=tuple(FLAGS.value_hidden_dims),
        discount=FLAGS.discount, tau=FLAGS.tau,
        expectile=FLAGS.expectile, use_layer_norm=FLAGS.use_layer_norm,
    )

    # ---- Resume Phase-2 from checkpoint if requested ------------------------
    start_step = 1
    if FLAGS.resume_step > 0:
        gcvf_agent = restore_agent(gcvf_agent, FLAGS.save_dir, FLAGS.resume_step)
        start_step = FLAGS.resume_step + 1
        print(f'[GCVFDual] Resumed from step {FLAGS.resume_step}, continuing from step {start_step}')

    # ---- Build train_step (pmap for multi-GPU, jit for single GPU) ----------
    get_phi_goal_jit = jax.jit(dual_agent.get_phi_goal)

    if n_devices > 1:
        gcvf_agent = jax.device_put_replicated(gcvf_agent, jax.local_devices())

        @functools.partial(jax.pmap, axis_name='batch')
        def train_step(agent, batch, phi_g):
            return agent.update(batch, phi_g, pmap_axis='batch')
    else:
        @jax.jit
        def train_step(agent, batch, phi_g):
            return agent.update(batch, phi_g)

    # ---- Training loop ------------------------------------------------------
    for step in tqdm.tqdm(range(start_step, FLAGS.train_steps + 1),
                          smoothing=0.1, dynamic_ncols=True):
        batch = gc_dataset.sample(FLAGS.batch_size)
        phi_g = np.array(get_phi_goal_jit(batch['goals']))   # (B, skill_dim)

        if n_devices > 1:
            batch = shard_batch(batch)
            phi_g = phi_g.reshape(n_devices, -1, phi_g.shape[-1])

        gcvf_agent, info = train_step(gcvf_agent, batch, phi_g)

        if step % FLAGS.log_interval == 0:
            if n_devices > 1:
                log_info = {k: float(v[0]) for k, v in info.items()}
            else:
                log_info = {k: float(v) for k, v in info.items()}
            log_str = '  '.join(f'{k}={v:.4f}' for k, v in log_info.items())
            tqdm.tqdm.write(f'[step {step:>8d}] {log_str}')
            if FLAGS.wandb_project:
                wandb.log(log_info, step=step)

        if step % FLAGS.save_interval == 0:
            save_agent(
                jax.tree.map(lambda x: x[0], gcvf_agent) if n_devices > 1 else gcvf_agent,
                FLAGS.save_dir, step, FLAGS.env_name)

    print('[GCVFDual] Training complete.')
    if FLAGS.wandb_project:
        wandb.finish()


if __name__ == '__main__':
    flags.DEFINE_string ('env_name',            'antmaze-giant-navigate-v0', 'OGBench env.')
    flags.DEFINE_string ('dual_restore_path',   'exp/dual_repr',  'Phase-1 checkpoint dir.')
    flags.DEFINE_integer('dual_restore_epoch',  1000000,          'Phase-1 checkpoint step.')
    flags.DEFINE_string ('dual_aggregator',     'inner_prod',     'Phase-1 dual aggregator: inner_prod or neg_l2.')
    flags.DEFINE_bool   ('dual_share_encoder',  False,            'Whether the restored Phase-1 dual checkpoint used a shared encoder.')
    flags.DEFINE_float  ('lr',                  3e-4,    'Learning rate.')
    flags.DEFINE_multi_integer('value_hidden_dims', [512, 512, 512], 'Hidden dims.')
    flags.DEFINE_integer('skill_dim',           32,      'Must match Phase-1 skill_dim.')
    flags.DEFINE_float  ('discount',            0.99,    'Discount.')
    flags.DEFINE_float  ('tau',                 0.005,   'Target EMA rate.')
    flags.DEFINE_float  ('expectile',           0.9,     'Downstream GCIVL expectile (paper Table 9).')
    flags.DEFINE_integer('use_layer_norm',      1,       '1 = LayerNorm.')
    flags.DEFINE_integer('batch_size',          1024,    'Batch size.')
    flags.DEFINE_integer('train_steps',         500000,  'Training steps.')
    flags.DEFINE_integer('save_interval',       100000,  'Checkpoint interval.')
    flags.DEFINE_integer('log_interval',        1000,    'Log interval.')
    flags.DEFINE_string ('save_dir',            'exp/gcvf_dual', 'Output dir.')
    flags.DEFINE_integer('seed',               0,        'Seed.')
    flags.DEFINE_float  ('p_currgoal',          0.2,     'Paper Table 9: downstream value ratio.')
    flags.DEFINE_float  ('p_trajgoal',          0.5,     'Paper Table 9: downstream value ratio.')
    flags.DEFINE_float  ('p_randomgoal',        0.3,     'Paper Table 9: downstream value ratio.')
    flags.DEFINE_integer('geom_sample',         1,       '')
    flags.DEFINE_integer('resume_step',         0,       'Resume Phase-2 from this step. 0 = start from scratch.')
    flags.DEFINE_string ('wandb_project',       '',      'WandB project name. Empty = disabled.')
    flags.DEFINE_string ('wandb_run_name',      '',      'WandB run name. Empty = auto.')
    app.run(main)
