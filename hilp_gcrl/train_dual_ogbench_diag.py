"""
Diagnostic version of train_dual_ogbench.py.

Adds detailed per-step JSONL logging to diagnose value divergence.
Hypotheses tested:
  H1: psi/phi encoder norms grow unboundedly (inner product explosion)
  H2: V > 0 fraction grows over time (violates theoretical upper bound of 0)
  H3: target_V bootstrapping creates positive feedback (target_v_max tracks v_max)
  H4: Certain goal types (curr/traj/random) produce more extreme V values
  H5: Absence of separate Q network makes inline Q = r + γV' unstable
"""

import copy
import functools
import json
import os
import sys
import glob
import pickle
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
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from jaxrl_m.common import TrainState, shard_batch
from jaxrl_m.dataset import Dataset
from src.dataset_utils import GCDataset
from src.special_networks import DualGoalPhiValue
from src.agents.hilp import expectile_loss

FLAGS = flags.FLAGS


# ======================== Checkpoint I/O =====================================

def save_agent(agent, save_dir, step):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'params_{step}.pkl')
    with open(path, 'wb') as f:
        pickle.dump({'agent': flax.serialization.to_state_dict(agent)}, f)
    print(f'Saved → {path}')
    for old in glob.glob(os.path.join(save_dir, 'params_*.pkl')):
        if old != path:
            os.remove(old)
            print(f'Removed → {old}')


# ======================== Network ============================================

class DualValueNetwork(nn.Module):
    networks: dict

    def value(self, observations, goals=None, **kwargs):
        return self.networks['value'](observations, goals, **kwargs)

    def target_value(self, observations, goals=None, **kwargs):
        return self.networks['target_value'](observations, goals, **kwargs)

    def phi(self, observations, **kwargs):
        return self.networks['value'].get_psi(observations)

    def phi_goal(self, goals, **kwargs):
        return self.networks['value'].get_phi(goals)

    def __call__(self, observations, goals):
        return {
            'value':        self.value(observations, goals),
            'target_value': self.target_value(observations, goals),
        }


# ======================== Agent ==============================================

class DualHILP(flax.struct.PyTreeNode):
    network: TrainState
    config:  dict = flax.struct.field(pytree_node=False)

    def value_loss(self, batch, network_params):
        (nv1, nv2) = self.network(
            batch['next_observations'], batch['goals'], method='target_value')
        next_v = jnp.minimum(nv1, nv2)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v

        (v1_t, v2_t) = self.network(
            batch['observations'], batch['goals'], method='target_value')
        adv = q - (v1_t + v2_t) / 2

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * nv1
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * nv2
        (v1, v2) = self.network(
            batch['observations'], batch['goals'],
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

    def total_loss(self, batch, grad_params):
        loss, info = self.value_loss(batch, grad_params)
        log = {f'value/{k}': v for k, v in info.items()}
        log['loss'] = loss
        return loss, log

    def update(self, batch, pmap_axis=None):
        new_network, info = self.network.apply_loss_fn(
            loss_fn=lambda p: self.total_loss(batch, p), has_aux=True,
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

    def diagnose(self, batch):
        """Compute detailed diagnostics without affecting training.
        Returns a dict of scalar JAX values."""
        # --- psi / phi encoder outputs ---
        psi_s = self.network(batch['observations'], method='phi')       # (B, D)
        psi_s_next = self.network(batch['next_observations'], method='phi')
        phi_g = self.network(batch['goals'], method='phi_goal')         # (B, D)

        psi_norm = jnp.linalg.norm(psi_s, axis=-1)     # (B,)
        phi_norm = jnp.linalg.norm(phi_g, axis=-1)      # (B,)

        # --- V from online and target networks ---
        (v1, v2) = self.network(batch['observations'], batch['goals'], method='value')
        v_online = (v1 + v2) / 2

        (tv1, tv2) = self.network(batch['observations'], batch['goals'], method='target_value')
        v_target = (tv1 + tv2) / 2

        # --- target V at next state (for inline Q) ---
        (tnv1, tnv2) = self.network(batch['next_observations'], batch['goals'], method='target_value')
        target_next_v = jnp.minimum(tnv1, tnv2)
        inline_q = batch['rewards'] + self.config['discount'] * batch['masks'] * target_next_v

        # --- cosine similarity between psi(s) and phi(g) ---
        cos_sim = (psi_s * phi_g).sum(axis=-1) / (psi_norm * phi_norm + 1e-8)

        # --- delta psi (transition magnitude) ---
        delta_psi = jnp.linalg.norm(psi_s_next - psi_s, axis=-1)

        # --- reward distribution ---
        reward_zero_frac = (batch['rewards'] == 0.0).mean()  # fraction of success (r=0)
        reward_neg_frac = (batch['rewards'] == -1.0).mean()   # fraction of non-success (r=-1)

        return {
            # H1: encoder norm growth
            'diag/psi_norm_mean':   psi_norm.mean(),
            'diag/psi_norm_max':    psi_norm.max(),
            'diag/psi_norm_std':    psi_norm.std(),
            'diag/phi_norm_mean':   phi_norm.mean(),
            'diag/phi_norm_max':    phi_norm.max(),
            'diag/phi_norm_std':    phi_norm.std(),

            # H2: V > 0 violation
            'diag/v_pos_frac':      (v_online > 0).mean(),
            'diag/v_online_p01':    jnp.percentile(v_online, 1),
            'diag/v_online_p05':    jnp.percentile(v_online, 5),
            'diag/v_online_p25':    jnp.percentile(v_online, 25),
            'diag/v_online_p50':    jnp.percentile(v_online, 50),
            'diag/v_online_p75':    jnp.percentile(v_online, 75),
            'diag/v_online_p95':    jnp.percentile(v_online, 95),
            'diag/v_online_p99':    jnp.percentile(v_online, 99),

            # H3: target V bootstrapping feedback
            'diag/target_v_mean':   v_target.mean(),
            'diag/target_v_max':    v_target.max(),
            'diag/target_v_min':    v_target.min(),
            'diag/target_next_v_mean': target_next_v.mean(),
            'diag/target_next_v_max':  target_next_v.max(),
            'diag/inline_q_mean':   inline_q.mean(),
            'diag/inline_q_max':    inline_q.max(),
            'diag/inline_q_min':    inline_q.min(),
            'diag/v_target_gap_mean': (v_online - v_target).mean(),
            'diag/v_target_gap_max':  (v_online - v_target).max(),

            # H4: cosine similarity (alignment between psi and phi)
            'diag/cos_sim_mean':    cos_sim.mean(),
            'diag/cos_sim_max':     cos_sim.max(),
            'diag/cos_sim_min':     cos_sim.min(),

            # H5: delta psi (representation change per transition)
            'diag/delta_psi_mean':  delta_psi.mean(),
            'diag/delta_psi_max':   delta_psi.max(),

            # Goal type stats
            'diag/reward_zero_frac': reward_zero_frac,
            'diag/reward_neg_frac':  reward_neg_frac,

            # Per-ensemble V spread
            'diag/v1_mean':         v1.mean(),
            'diag/v2_mean':         v2.mean(),
            'diag/v_ensemble_gap':  jnp.abs(v1 - v2).mean(),
        }

    @jax.jit
    def get_phi_goal(self, goals):
        return self.network(goals, method='phi_goal')

    @jax.jit
    def get_psi(self, observations):
        return self.network(observations, method='phi')

    @jax.jit
    def get_phi(self, observations):
        return self.get_psi(observations)

    @classmethod
    def create(cls, seed, ex_observations, lr=3e-4,
               value_hidden_dims=(512, 512, 512), discount=0.99, tau=0.005,
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
        network_def = DualValueNetwork(networks={
            'value':        value_def,
            'target_value': copy.deepcopy(value_def),
        })
        network_tx     = optax.adam(learning_rate=lr)
        network_params = unfreeze(network_def.init(
            init_rng, ex_observations, ex_observations)['params'])
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

    n_devices = jax.local_device_count()
    print(f'[DualHILP-Diag] env={FLAGS.env_name}  save_dir={FLAGS.save_dir}')
    print(f'[DualHILP-Diag] Using {n_devices} GPU(s): {jax.local_devices()}')
    assert FLAGS.batch_size % n_devices == 0

    # ---- WandB --------------------------------------------------------------
    if FLAGS.wandb_project:
        run_name = FLAGS.wandb_run_name or f'dual_repr_diag_{FLAGS.env_name}'
        wandb.init(project=FLAGS.wandb_project, name=run_name,
                   config=FLAGS.flag_values_dict())

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

    # ---- JIT compile diagnose ------------------------------------------------
    diagnose_fn = jax.jit(agent.diagnose)

    # ---- Build train_step ----------------------------------------------------
    if n_devices > 1:
        agent = jax.device_put_replicated(agent, jax.local_devices())
        @functools.partial(jax.pmap, axis_name='batch')
        def train_step(agent, batch):
            return agent.update(batch, pmap_axis='batch')
    else:
        @jax.jit
        def train_step(agent, batch):
            return agent.update(batch)

    # ---- JSONL diagnostic log ------------------------------------------------
    diag_path = os.path.join(FLAGS.save_dir, 'diagnostics.jsonl')
    print(f'[DualHILP-Diag] Diagnostic log → {diag_path}')
    diag_file = open(diag_path, 'w')

    # ---- Training loop ------------------------------------------------------
    for step in tqdm.tqdm(range(1, FLAGS.train_steps + 1),
                          smoothing=0.1, dynamic_ncols=True):
        batch = gc_dataset.sample(FLAGS.batch_size)
        if n_devices > 1:
            batch = shard_batch(batch)
        agent, info = train_step(agent, batch)

        if step % FLAGS.log_interval == 0:
            if n_devices > 1:
                log_info = {k: float(v[0]) for k, v in info.items()}
            else:
                log_info = {k: float(v) for k, v in info.items()}
            log_str = '  '.join(f'{k}={v:.4f}' for k, v in log_info.items())
            tqdm.tqdm.write(f'[step {step:>8d}] {log_str}')
            if FLAGS.wandb_project:
                wandb.log(log_info, step=step)

        # ---- Diagnostic logging (every diag_interval steps) ----
        if step % FLAGS.diag_interval == 0:
            # Sample a fresh batch for diagnostics
            diag_batch = gc_dataset.sample(FLAGS.batch_size)
            if n_devices > 1:
                # Use first replica for diagnostics
                agent_single = jax.tree.map(lambda x: x[0], agent)
            else:
                agent_single = agent
            diag_info = diagnose_fn(agent_single, diag_batch)
            diag_record = {'step': step}
            diag_record.update({k: float(v) for k, v in diag_info.items()})
            # Also include training metrics
            if n_devices > 1:
                diag_record.update({k: float(v[0]) for k, v in info.items()})
            else:
                diag_record.update({k: float(v) for k, v in info.items()})
            diag_file.write(json.dumps(diag_record) + '\n')
            diag_file.flush()

        if step % FLAGS.save_interval == 0:
            save_agent(
                jax.tree.map(lambda x: x[0], agent) if n_devices > 1 else agent,
                FLAGS.save_dir, step)

    diag_file.close()
    print(f'[DualHILP-Diag] Training complete. Diagnostics saved to {diag_path}')
    if FLAGS.wandb_project:
        wandb.finish()


if __name__ == '__main__':
    flags.DEFINE_string ('env_name',       'antmaze-giant-navigate-v0', 'OGBench env.')
    flags.DEFINE_float  ('lr',             3e-4,    'Learning rate.')
    flags.DEFINE_integer('skill_dim',      32,      'Dimension of psi/phi.')
    flags.DEFINE_multi_integer('value_hidden_dims', [512, 512, 512], 'MLP hidden dims.')
    flags.DEFINE_float  ('discount',       0.99,    'Discount.')
    flags.DEFINE_float  ('tau',            0.005,   'Target EMA rate.')
    flags.DEFINE_float  ('expectile',      0.95,    'IQL expectile.')
    flags.DEFINE_integer('use_layer_norm', 1,       '1 = LayerNorm.')
    flags.DEFINE_integer('batch_size',     1024,    'Batch size.')
    flags.DEFINE_integer('train_steps',    1000000, 'Total steps.')
    flags.DEFINE_integer('save_interval',  100000,  'Checkpoint interval.')
    flags.DEFINE_integer('log_interval',   1000,    'Log interval.')
    flags.DEFINE_integer('diag_interval',  1000,    'Diagnostic log interval.')
    flags.DEFINE_string ('save_dir',       'exp/dual_repr', 'Output dir.')
    flags.DEFINE_integer('seed',           0,       'Seed.')
    flags.DEFINE_float  ('p_currgoal',     0.0,     '')
    flags.DEFINE_float  ('p_trajgoal',     0.625,   '')
    flags.DEFINE_float  ('p_randomgoal',   0.375,   '')
    flags.DEFINE_integer('geom_sample',    1,       '')
    flags.DEFINE_string ('wandb_project',  '',      'WandB project name.')
    flags.DEFINE_string ('wandb_run_name', '',      'WandB run name.')
    app.run(main)
