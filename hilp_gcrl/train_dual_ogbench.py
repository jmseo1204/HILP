"""
Train Dual Goal Representations on OGBench environments.

Phase 1 of arXiv:2510.06714:
  V(s, g) = psi(s)^T phi(g)  (inner product, separate state/goal encoders)

Follows Algorithm 1 with separate Q network:
  L(psi, phi) = E[ l2_kappa( V(s,g) - Q_bar(s,a,g) ) ]
  L(Q)        = E[ (Q(s,a,g) - r(s,g) - gamma * V(s',g))^2 ]
  Q_bar       <- EMA of Q

All dependencies are inside hilp_gcrl/.
"""

import copy
import functools
import os
import sys
import glob
import pickle
from pathlib import Path

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
from src.special_networks import DualGoalPhiValue, GoalConditionedCritic
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
    # Remove previous checkpoints, keeping only the latest
    for old in glob.glob(os.path.join(save_dir, 'params_*.pkl')):
        if old != path:
            os.remove(old)
            print(f'Removed → {old}')


def restore_agent(agent, restore_path, restore_epoch):
    candidates = glob.glob(restore_path)
    assert len(candidates) == 1, f'Expected 1 match, got {len(candidates)}: {candidates}'
    path = candidates[0] + f'/params_{restore_epoch}.pkl'
    with open(path, 'rb') as f:
        load_dict = pickle.load(f)
    return flax.serialization.from_state_dict(agent, load_dict['agent'])


# ======================== Network ============================================

class DualValueNetwork(nn.Module):
    """
    Network container for dual goal representation learning (Algorithm 1).

    Contains:
      value:    V(s,g) = psi(s)^T phi(g)   (inner-product, ensemble)
      q_func:   Q(s,a,g) = MLP([s,g,a])    (separate Q network, ensemble)
      target_q: Q_bar = EMA of q_func       (target Q network)
    """
    networks: dict

    def value(self, observations, goals=None, **kwargs):
        return self.networks['value'](observations, goals, **kwargs)

    def q_func(self, observations, goals=None, actions=None, **kwargs):
        return self.networks['q_func'](observations, goals, actions, **kwargs)

    def target_q(self, observations, goals=None, actions=None, **kwargs):
        return self.networks['target_q'](observations, goals, actions, **kwargs)

    def phi(self, observations, **kwargs):
        """psi(s): state representation."""
        return self.networks['value'].get_psi(observations)

    def phi_goal(self, goals, **kwargs):
        """phi(g): dual goal representation."""
        return self.networks['value'].get_phi(goals)

    def __call__(self, observations, goals, actions):
        # Only used for parameter initialization
        return {
            'value':    self.value(observations, goals),
            'q_func':   self.q_func(observations, goals, actions),
            'target_q': self.target_q(observations, goals, actions),
        }


# ======================== Agent ==============================================

class DualHILP(flax.struct.PyTreeNode):
    network: TrainState
    config:  dict = flax.struct.field(pytree_node=False)

    def total_loss(self, batch, network_params):
        """
        Algorithm 1 losses:
          L(psi, phi) = E[ l2_kappa( V(s,g) - Q_bar(s,a,g) ) ]   (Eq. 3)
          L(Q)        = E[ (Q(s,a,g) - r - gamma * V(s',g))^2 ]   (Eq. 4)
        """
        # ---- V loss (Eq. 3): fit V to target Q_bar ----
        # Q_bar(s, a, g) from target Q (stop gradient — uses stored params)
        (tq1, tq2) = self.network(
            batch['observations'], batch['goals'], batch['actions'],
            method='target_q')
        q_bar = jnp.minimum(tq1, tq2)   # pessimistic twin-Q target

        # V(s, g) = psi(s)^T phi(g) — single pair, gradient flows through psi, phi
        v = self.network(
            batch['observations'], batch['goals'],
            method='value', params=network_params)

        # Advantage for expectile weighting
        adv = q_bar - v

        # Expectile loss: push V toward the upper quantile of Q_bar
        loss_v = expectile_loss(adv, q_bar - v, self.config['expectile']).mean()

        # ---- Q loss (Eq. 4): fit Q to r + gamma * V(s') ----
        # V(s', g) from current V (stop gradient — uses stored params, not network_params)
        next_v = self.network(
            batch['next_observations'], batch['goals'],
            method='value')
        target_q_val = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v

        # Q(s, a, g) — gradient flows through Q
        (q1, q2) = self.network(
            batch['observations'], batch['goals'], batch['actions'],
            method='q_func', params=network_params)
        loss_q = ((q1 - target_q_val)**2 + (q2 - target_q_val)**2).mean()

        loss = loss_v + loss_q

        info = {
            'value/value_loss':  loss_v,
            'value/q_loss':      loss_q,
            'value/v_mean':      v.mean(),
            'value/v_max':       v.max(),
            'value/v_min':       v.min(),
            'value/adv_mean':    adv.mean(),
            'value/accept_prob': (adv >= 0).mean(),
            'value/q_bar_mean':  q_bar.mean(),
            'value/q_bar_max':   q_bar.max(),
            'value/q_bar_min':   q_bar.min(),
            'loss':              loss,
        }
        return loss, info

    def update(self, batch, pmap_axis=None):
        new_network, info = self.network.apply_loss_fn(
            loss_fn=lambda p: self.total_loss(batch, p), has_aux=True,
            pmap_axis=pmap_axis)

        # EMA target update: Q_bar <- tau * Q + (1 - tau) * Q_bar
        new_tq = jax.tree.map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            new_network.params['networks_q_func'],
            new_network.params['networks_target_q'],
        )
        params = dict(new_network.params)
        params['networks_target_q'] = new_tq
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
    def create(cls, seed, ex_observations, ex_actions, lr=3e-4,
               value_hidden_dims=(512, 512, 512), discount=0.99, tau=0.005,
               expectile=0.95, use_layer_norm=1, skill_dim=32,
               grad_clip_norm=1.0, **kwargs):
        print('DualHILP.create — extra kwargs:', kwargs)
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        # V(s,g) = psi(s)^T phi(g)
        value_def = DualGoalPhiValue(
            hidden_dims=tuple(value_hidden_dims),
            skill_dim=skill_dim,
            use_layer_norm=bool(use_layer_norm),
        )

        # Q(s,a,g) = MLP([s, g, a]) — separate Q network (Algorithm 1)
        q_def = GoalConditionedCritic(
            hidden_dims=tuple(value_hidden_dims),
            use_layer_norm=bool(use_layer_norm),
            ensemble=True,
        )

        network_def = DualValueNetwork(networks={
            'value':    value_def,
            'q_func':   q_def,
            'target_q': copy.deepcopy(q_def),
        })

        # Gradient clipping + Adam
        network_tx = optax.chain(
            optax.clip_by_global_norm(grad_clip_norm),
            optax.adam(learning_rate=lr),
        )

        network_params = unfreeze(network_def.init(
            init_rng, ex_observations, ex_observations, ex_actions)['params'])
        network = TrainState.create(network_def, network_params, tx=network_tx)

        # Initialize target_q = q_func
        params = dict(network.params)
        params['networks_target_q'] = params['networks_q_func']
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
    print(f'[DualHILP] env={FLAGS.env_name}  save_dir={FLAGS.save_dir}')
    print(f'[DualHILP] Using {n_devices} GPU(s): {jax.local_devices()}')
    assert FLAGS.batch_size % n_devices == 0, (
        f'batch_size ({FLAGS.batch_size}) must be divisible by n_devices ({n_devices})')

    # ---- WandB --------------------------------------------------------------
    if FLAGS.wandb_project:
        run_name = FLAGS.wandb_run_name or f'dual_repr_{FLAGS.env_name}'
        wandb.init(project=FLAGS.wandb_project, name=run_name,
                   config=FLAGS.flag_values_dict())
        print(f'[DualHILP] WandB run: {run_name}  project: {FLAGS.wandb_project}')

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
    ex_act = train_data['actions'][:1]
    agent  = DualHILP.create(
        seed              = FLAGS.seed,
        ex_observations   = ex_obs,
        ex_actions        = ex_act,
        lr                = FLAGS.lr,
        value_hidden_dims = tuple(FLAGS.value_hidden_dims),
        discount          = FLAGS.discount,
        tau               = FLAGS.tau,
        expectile         = FLAGS.expectile,
        use_layer_norm    = FLAGS.use_layer_norm,
        skill_dim         = FLAGS.skill_dim,
        grad_clip_norm    = FLAGS.grad_clip_norm,
    )

    # ---- Resume from checkpoint if requested --------------------------------
    start_step = 1
    if FLAGS.resume_step > 0:
        agent = restore_agent(agent, FLAGS.save_dir, FLAGS.resume_step)
        start_step = FLAGS.resume_step + 1
        print(f'[DualHILP] Resumed from step {FLAGS.resume_step}, continuing from step {start_step}')

    # ---- Build train_step (pmap for multi-GPU, jit for single GPU) ----------
    if n_devices > 1:
        agent = jax.device_put_replicated(agent, jax.local_devices())

        @functools.partial(jax.pmap, axis_name='batch')
        def train_step(agent, batch):
            return agent.update(batch, pmap_axis='batch')
    else:
        @jax.jit
        def train_step(agent, batch):
            return agent.update(batch)

    # ---- Training loop ------------------------------------------------------
    for step in tqdm.tqdm(range(start_step, FLAGS.train_steps + 1),
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

        if step % FLAGS.save_interval == 0:
            save_agent(
                jax.tree.map(lambda x: x[0], agent) if n_devices > 1 else agent,
                FLAGS.save_dir, step)

    print('[DualHILP] Training complete.')
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
    flags.DEFINE_string ('save_dir',       'exp/dual_repr', 'Output dir.')
    flags.DEFINE_integer('seed',           0,       'Seed.')
    flags.DEFINE_float  ('p_currgoal',     0.0,     '')
    flags.DEFINE_float  ('p_trajgoal',     0.625,   '')
    flags.DEFINE_float  ('p_randomgoal',   0.375,   '')
    flags.DEFINE_integer('geom_sample',    1,       '')
    flags.DEFINE_float  ('grad_clip_norm', 1.0,     'Max gradient global norm.')
    flags.DEFINE_integer('resume_step',    0,       'Resume from this step. 0 = start from scratch.')
    flags.DEFINE_string ('wandb_project',  '',      'WandB project name. Empty = disabled.')
    flags.DEFINE_string ('wandb_run_name', '',      'WandB run name. Empty = auto.')
    app.run(main)
