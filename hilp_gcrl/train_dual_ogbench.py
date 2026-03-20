"""
Train Dual Goal Representations on OGBench environments.

Phase 1 of arXiv:2510.06714:
  V(s, g) = psi(s)^T phi(g)  (inner product, separate state/goal encoders)

All dependencies are inside hilp_gcrl/.
"""

import copy
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

# ---- hilp_gcrl internal imports ---------------------------------------------
_ROOT = Path(__file__).parent          # hilp_gcrl/
sys.path.insert(0, str(_ROOT))

from jaxrl_m.common import TrainState
from jaxrl_m.dataset import Dataset
from src.dataset_utils import GCDataset
from src.special_networks import DualGoalPhiValue
from src.agents.hilp import expectile_loss
# -----------------------------------------------------------------------------

FLAGS = flags.FLAGS


# ======================== Checkpoint I/O =====================================

def save_agent(agent, save_dir, step):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'params_{step}.pkl')
    with open(path, 'wb') as f:
        pickle.dump({'agent': flax.serialization.to_state_dict(agent)}, f)
    print(f'Saved → {path}')


def restore_agent(agent, restore_path, restore_epoch):
    candidates = glob.glob(restore_path)
    assert len(candidates) == 1, f'Expected 1 match, got {len(candidates)}: {candidates}'
    path = candidates[0] + f'/params_{restore_epoch}.pkl'
    with open(path, 'rb') as f:
        load_dict = pickle.load(f)
    return flax.serialization.from_state_dict(agent, load_dict['agent'])


# ======================== Network ============================================

class DualValueNetwork(nn.Module):
    """Wraps DualGoalPhiValue (value + target_value) for use with TrainState."""
    networks: dict

    def value(self, observations, goals=None, **kwargs):
        return self.networks['value'](observations, goals, **kwargs)

    def target_value(self, observations, goals=None, **kwargs):
        return self.networks['target_value'](observations, goals, **kwargs)

    def phi(self, observations, **kwargs):
        """psi(s): state representation."""
        return self.networks['value'].get_psi(observations)

    def phi_goal(self, goals, **kwargs):
        """phi(g): dual goal representation."""
        return self.networks['value'].get_phi(goals)

    def __call__(self, observations, goals):
        # Only used for parameter initialization
        return {
            'value':        self.value(observations, goals),
            'target_value': self.target_value(observations, goals),
        }


# ======================== Agent ==============================================

class DualHILP(flax.struct.PyTreeNode):
    network: TrainState
    config:  dict = flax.struct.field(pytree_node=False)

    def value_loss(self, batch, network_params):
        # batch['rewards'] = success - 1  (0 or -1)
        # batch['masks']   = 1 - success  (1 or 0)
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

    @jax.jit
    def total_loss(self, batch, grad_params):
        loss, info = self.value_loss(batch, grad_params)
        return loss, {f'value/{k}': v for k, v in info.items()}

    @jax.jit
    def update(self, batch):
        new_network, info = self.network.apply_loss_fn(
            loss_fn=lambda p: self.total_loss(batch, p), has_aux=True)

        # EMA target update
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
    def get_phi_goal(self, goals: np.ndarray) -> jnp.ndarray:
        """phi(g): dual goal representation."""
        return self.network(goals, method='phi_goal')

    @jax.jit
    def get_psi(self, observations: np.ndarray) -> jnp.ndarray:
        """psi(s): state representation."""
        return self.network(observations, method='phi')

    # Alias so evaluation code can call agent.get_phi(obs)
    @jax.jit
    def get_phi(self, observations: np.ndarray) -> jnp.ndarray:
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

        # Initialize target = value (plain dict — keeps optimizer state consistent)
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
    print(f'[DualHILP] env={FLAGS.env_name}  save_dir={FLAGS.save_dir}')

    _, dataset, _ = ogbench.make_env_and_datasets(
        FLAGS.env_name, compact_dataset=False)

    # Dataset: pass as plain dict so GCDataset can handle terminal_key
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
        reward_shift = -1.0,   # rewards = success - 1
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
    flags.DEFINE_string ('env_name',       'pointmaze-large-stitch-v0', 'OGBench env.')
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
    flags.DEFINE_string ('save_dir',       'exp/dual_repr', 'Output dir.')
    flags.DEFINE_integer('seed',           0,       'Seed.')
    flags.DEFINE_float  ('p_currgoal',     0.0,     '')
    flags.DEFINE_float  ('p_trajgoal',     0.625,   '')
    flags.DEFINE_float  ('p_randomgoal',   0.375,   '')
    flags.DEFINE_integer('geom_sample',    1,       '')
    app.run(main)
