"""
Train Dual Goal Representations on OGBench environments.

Phase 1 of arXiv:2510.06714 — two aggregator modes:

  inner_prod (default):
    V(s, g) = psi(s)^T phi(g)
    Gradient: J_psi(s)^T * phi(g)  (phi(g) acts as goal-direction vector)

  neg_l2:
    V(s, g) = -||psi(s) - phi(g)||
    Gradient direction: unit vector toward phi(g) in latent space,
    independent of temporal distance (no saturation for far goals).

Both modes follow Algorithm 1 with a separate Q network:
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
from datetime import datetime
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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.manifold import TSNE
from shapely.geometry import Point, box
from shapely.ops import unary_union
from scipy.spatial import KDTree

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
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(save_dir, f'params_{step}_{ts}.pkl')
    with open(path, 'wb') as f:
        pickle.dump({'agent': flax.serialization.to_state_dict(agent)}, f)
    print(f'Saved → {path}')


def restore_agent(agent, restore_dir, restore_epoch):
    pattern = os.path.join(restore_dir, f'params_{restore_epoch}_*.pkl')
    candidates = glob.glob(pattern)
    # Fall back to legacy filename without timestamp
    if not candidates:
        legacy = os.path.join(restore_dir, f'params_{restore_epoch}.pkl')
        if os.path.exists(legacy):
            candidates = [legacy]
    assert len(candidates) >= 1, f'No checkpoint found for step {restore_epoch} in {restore_dir}'
    path = sorted(candidates)[-1]   # use latest timestamp if multiple exist
    with open(path, 'rb') as f:
        load_dict = pickle.load(f)
    print(f'Restored ← {path}')
    return flax.serialization.from_state_dict(agent, load_dict['agent'])


# ======================== t-SNE Visualization ================================

def _build_traversable_obs(env, obs_template):
    """
    Compute a list of non-wall (x, y) points and matching full observations
    by sampling a fine grid over the maze.

    Returns:
        obs_array   : (N, obs_dim) float32 numpy array
        xy_array    : (N, 2)       float32 numpy array of world coordinates
    """
    maze_map  = env.unwrapped.maze_map == 1
    UNIT      = float(env.unwrapped._maze_unit)
    ORIGIN_X  = -float(env.unwrapped._offset_x)
    ORIGIN_Y  = -float(env.unwrapped._offset_y)
    half      = UNIT / 2
    num_rows, num_cols = len(maze_map), len(maze_map[0]) if len(maze_map) else 0

    # Build shapely union of wall boxes
    wall_polys = []
    for r in range(num_rows):
        for c in range(num_cols):
            if maze_map[r][c]:
                cx = ORIGIN_X + c * UNIT
                cy = ORIGIN_Y + r * UNIT
                wall_polys.append(box(cx - half, cy - half, cx + half, cy + half))
    poly_union = unary_union(wall_polys) if wall_polys else None

    # Sample a grid at UNIT/4 spacing
    step = UNIT / 4
    x_range = np.arange(ORIGIN_X - half + step / 2, ORIGIN_X + num_cols * UNIT - half, step)
    y_range = np.arange(ORIGIN_Y - half + step / 2, ORIGIN_Y + num_rows * UNIT - half, step)

    rest = np.asarray(obs_template[2:]) if obs_template.shape[0] > 2 else np.array([], dtype=np.float32)

    obs_list, xy_list = [], []
    for x in x_range:
        for y in y_range:
            if poly_union is not None and poly_union.intersects(Point(x, y)):
                continue
            obs_list.append(np.concatenate([[x, y], rest]))
            xy_list.append([x, y])

    if not obs_list:
        return np.zeros((0, obs_template.shape[0]), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    return (np.array(obs_list, dtype=np.float32),
            np.array(xy_list, dtype=np.float32))


def generate_tsne_visualization(agent, obs_array, xy_array, step, seed, aggregator):
    """
    Run t-SNE on psi(s) and phi(g) embeddings for traversable maze states,
    and return an (H, W, 3) uint8 image for WandB logging.

    Left subplot  — psi(s)  colored by X coordinate
    Right subplot — psi(s)  colored by Y coordinate
    """
    if obs_array.shape[0] <= 1:
        print('  [t-SNE] Not enough traversable points — skipping.')
        return np.zeros((800, 1600, 3), dtype=np.uint8)

    # Compute psi(s) embeddings
    chunk = 4096
    psi_list = []
    for i in range(0, obs_array.shape[0], chunk):
        psi_list.append(np.array(agent.get_psi(obs_array[i:i + chunk])))
    psi = np.concatenate(psi_list, axis=0)          # (N, skill_dim)

    perplexity = float(min(30, psi.shape[0] - 1))
    if perplexity <= 0:
        perplexity = 5.0

    print(f'  [t-SNE] Running on {psi.shape[0]} points (dim={psi.shape[1]}) ...')
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity,
                max_iter=300, init='pca', learning_rate='auto', n_jobs=-1)
    try:
        emb = tsne.fit_transform(psi)               # (N, 2)
    except Exception as e:
        print(f'  [t-SNE] Error: {e}')
        return np.zeros((800, 1600, 3), dtype=np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=100)
    fig.suptitle(f'psi(s) t-SNE   [aggregator={aggregator}   step={step:,}]', fontsize=13)

    titles   = ['colored by X position', 'colored by Y position']
    colors   = [xy_array[:, 0], xy_array[:, 1]]
    clabels  = ['X coordinate', 'Y coordinate']
    for ax, title, c, clabel in zip(axes, titles, colors, clabels):
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=c, cmap='viridis', s=6, alpha=0.8)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).set_label(clabel)
        ax.set_title(title)
        ax.set_xlabel('t-SNE dim 1')
        ax.set_ylabel('t-SNE dim 2')
        ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    canvas = FigureCanvas(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    plt.close(fig)
    return image


def _build_prohibited_obs(dataset_obs, obs_template, threshold, grid_step=0.5):
    """
    Build a set of prohibited-zone observations.

    1. Bounding box of dataset xy positions.
    2. Fine grid (grid_step spacing) within that box.
    3. KD-tree query: keep grid points whose nearest dataset point >= threshold.
    4. Return full-dimensional observations (xy replaced, rest from template).

    Returns:
        prohibited_obs : (N_neg, obs_dim) float32,  or empty (0, obs_dim)
    """
    xy = dataset_obs[:, :2].astype(np.float64)
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)

    xs = np.arange(x_min, x_max + grid_step, grid_step)
    ys = np.arange(y_min, y_max + grid_step, grid_step)
    X, Y = np.meshgrid(xs, ys)
    grid_pts = np.stack([X.ravel(), Y.ravel()], axis=1)

    tree = KDTree(xy)
    dist, _ = tree.query(grid_pts, workers=-1)
    mask = dist >= threshold
    neg_xy = grid_pts[mask].astype(np.float32)

    if neg_xy.shape[0] == 0:
        return np.zeros((0, obs_template.shape[0]), dtype=np.float32)

    rest = obs_template[2:].astype(np.float32)
    neg_obs = np.tile(rest, (neg_xy.shape[0], 1))
    neg_obs = np.concatenate([neg_xy, neg_obs], axis=1)
    return neg_obs


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
        }

        # ---- Prohibited-zone negative penalty (V-only) ----------------------
        # Pushes V(s_neg, g) below V_floor = reward_shift / (1 - discount)
        # via hinge loss: penalty only when V(s_neg, g) > V_floor.
        # Q network is NOT touched — no Bellman backup pollution.
        #
        # neg_weight is 0.0 (inactive) or 1.0 (active) — keeps dict structure
        # constant across steps so JIT traces only once.
        v_floor = self.config['v_floor']
        v_neg = self.network(
            batch['neg_states'], batch['neg_goals'],
            method='value', params=network_params)
        hinge = jnp.maximum(v_neg - v_floor, 0.0)
        loss_neg = (hinge ** 2).mean()
        neg_w = jnp.squeeze(batch['neg_weight'])
        loss = loss + self.config['lambda_neg'] * neg_w * loss_neg

        info['neg/loss_weighted'] = loss_neg * neg_w
        info['neg/loss_raw']      = loss_neg          # constraint 충족 여부: 0이면 V_neg < v_floor
        info['neg/v_neg_mean']    = v_neg.mean()      # v_floor 아래여야 정상
        info['neg/v_neg_max']     = v_neg.max()       # 가장 위험한 값; v_floor보다 낮아야 함
        info['neg/active']        = neg_w             # 이 step이 active였는지 (0 or 1)
        info['loss'] = loss
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
               grad_clip_norm=1.0, aggregator='inner_prod',
               lambda_neg=0.1, v_floor=None, **kwargs):
        print(f'DualHILP.create — aggregator={aggregator}  extra kwargs:', kwargs)
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        # V(s,g): aggregator selects inner_prod or neg_l2
        value_def = DualGoalPhiValue(
            hidden_dims=tuple(value_hidden_dims),
            skill_dim=skill_dim,
            use_layer_norm=bool(use_layer_norm),
            aggregator=aggregator,
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

        # V_floor = reward_shift / (1 - discount)  =  -1 / (1 - 0.995)  =  -200
        if v_floor is None:
            v_floor = -1.0 / (1.0 - discount)
        print(f'DualHILP.create — v_floor={v_floor:.2f}  lambda_neg={lambda_neg}')

        return cls(network=network,
                   config=flax.core.FrozenDict(
                       discount=discount, tau=tau, expectile=expectile,
                       skill_dim=skill_dim, aggregator=aggregator,
                       lambda_neg=lambda_neg, v_floor=v_floor))


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

    env, dataset, _ = ogbench.make_env_and_datasets(
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

    # ---- Build traversable maze grid for t-SNE (done once) -----------------
    obs_template = train_data['observations'][0]
    viz_obs, viz_xy = _build_traversable_obs(env, obs_template)
    print(f'[DualHILP] Traversable viz points: {viz_obs.shape[0]}')

    # ---- Build prohibited-zone observations for negative sampling ----------
    prohibited_obs = None
    if FLAGS.p_prohibit > 0 and FLAGS.prohibit_threshold > 0:
        prohibited_obs = _build_prohibited_obs(
            dataset['observations'], obs_template,
            threshold=FLAGS.prohibit_threshold)
        print(f'[DualHILP] Prohibited zone: {prohibited_obs.shape[0]} points  '
              f'(threshold={FLAGS.prohibit_threshold})')
        if prohibited_obs.shape[0] == 0:
            print('[DualHILP] WARNING: no prohibited points found, disabling negative sampling.')
            prohibited_obs = None

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
        aggregator        = FLAGS.aggregator,
        lambda_neg        = FLAGS.lambda_neg,
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

    # ---- Negative sampling helper -------------------------------------------
    all_obs = train_data['observations']

    def _inject_neg_samples(batch):
        """Always inject neg_states/neg_goals/neg_weight into batch.

        When active (probability p_prohibit): real prohibited-zone states,
        neg_weight=1.0.  When inactive: dummy zeros, neg_weight=0.0.
        Dict structure is constant across steps so JIT traces only once.
        """
        n = batch['observations'].shape[0]
        obs_dim = batch['observations'].shape[1]
        active = (prohibited_obs is not None
                  and np.random.rand() < FLAGS.p_prohibit)
        if active:
            neg_idx  = np.random.randint(prohibited_obs.shape[0], size=n)
            goal_idx = np.random.randint(all_obs.shape[0], size=n)
            batch['neg_states'] = prohibited_obs[neg_idx].astype(np.float32)
            batch['neg_goals']  = all_obs[goal_idx].astype(np.float32)
            batch['neg_weight'] = np.ones(n_devices, dtype=np.float32)
        else:
            batch['neg_states'] = np.zeros((n, obs_dim), dtype=np.float32)
            batch['neg_goals']  = np.zeros((n, obs_dim), dtype=np.float32)
            batch['neg_weight'] = np.zeros(n_devices, dtype=np.float32)
        return batch, active

    # ---- Training loop ------------------------------------------------------
    # 모든 지표를 interval mean으로 로깅
    # neg 지표 중 per-step 의미 없는 것들은 active step만 별도 누적
    _NEG_STEP_KEYS = {'neg/active', 'neg/loss_raw', 'neg/loss_weighted',
                      'neg/v_neg_mean', 'neg/v_neg_max'}
    neg_sample_count = 0
    info_acc    = {}   # 일반 지표: 전체 step 누적
    neg_acc     = {}   # neg 지표: active step만 누적
    _extract    = (lambda v: float(v[0])) if n_devices > 1 else float

    for step in tqdm.tqdm(range(start_step, FLAGS.train_steps + 1),
                          smoothing=0.1, dynamic_ncols=True):
        batch = gc_dataset.sample(FLAGS.batch_size)
        batch, is_active = _inject_neg_samples(batch)
        neg_sample_count += int(is_active)
        if n_devices > 1:
            batch = shard_batch(batch)
        agent, info = train_step(agent, batch)

        # 일반 지표 누적 (neg per-step 제외)
        for k, v in info.items():
            if k not in _NEG_STEP_KEYS:
                info_acc.setdefault(k, []).append(_extract(v))

        # neg 지표는 active step일 때만 누적
        if is_active:
            for k in ('neg/loss_raw', 'neg/v_neg_mean', 'neg/v_neg_max'):
                neg_acc.setdefault(k, []).append(_extract(info[k]))

        if step % FLAGS.log_interval == 0:
            log_info = {k: float(np.mean(vs)) for k, vs in info_acc.items()}

            log_info['neg/sample_count']    = neg_sample_count
            log_info['neg/sample_rate']     = neg_sample_count / FLAGS.log_interval
            log_info['neg/prohibited_size'] = prohibited_obs.shape[0] if prohibited_obs is not None else 0
            for k in ('neg/loss_raw', 'neg/v_neg_mean', 'neg/v_neg_max'):
                log_info[k + '_avg'] = float(np.mean(neg_acc[k])) if neg_acc.get(k) else float('nan')

            log_str = '  '.join(
                f'{k}={v:.4f}' for k, v in log_info.items()
                if not (isinstance(v, float) and np.isnan(v))
            )
            tqdm.tqdm.write(f'[step {step:>8d}] {log_str}')
            if FLAGS.wandb_project:
                wandb.log(log_info, step=step)

            neg_sample_count = 0
            info_acc.clear()
            neg_acc.clear()

        if FLAGS.viz_interval > 0 and step % FLAGS.viz_interval == 0:
            if FLAGS.wandb_project and viz_obs.shape[0] > 0:
                single_agent = jax.tree.map(lambda x: x[0], agent) if n_devices > 1 else agent
                img = generate_tsne_visualization(
                    single_agent, viz_obs, viz_xy, step, FLAGS.seed, FLAGS.aggregator)
                wandb.log({'diagnostics/psi_tsne': wandb.Image(img)}, step=step)
                tqdm.tqdm.write(f'[step {step:>8d}] Logged psi(s) t-SNE to WandB.')

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
    flags.DEFINE_integer('viz_interval',   20000,   't-SNE viz interval. 0 = disabled.')
    flags.DEFINE_string ('save_dir',       'exp/dual_repr', 'Output dir.')
    flags.DEFINE_integer('seed',           0,       'Seed.')
    flags.DEFINE_float  ('p_currgoal',     0.0,     '')
    flags.DEFINE_float  ('p_trajgoal',     0.625,   '')
    flags.DEFINE_float  ('p_randomgoal',   0.375,   '')
    flags.DEFINE_integer('geom_sample',    1,       '')
    flags.DEFINE_float  ('grad_clip_norm', 1.0,     'Max gradient global norm.')
    flags.DEFINE_string ('aggregator',     'inner_prod',
                         'Value aggregation: inner_prod (psi^T phi) or neg_l2 (-||psi-phi||).')
    flags.DEFINE_float  ('p_prohibit',     0.0,     'Prob of injecting negative (prohibited) samples per step. 0 = disabled.')
    flags.DEFINE_float  ('lambda_neg',    0.1,     'Weight of prohibited-zone hinge loss.')
    flags.DEFINE_float  ('prohibit_threshold', 1.5, 'KD-tree distance threshold for prohibited zone.')
    flags.DEFINE_integer('resume_step',    0,       'Resume from this step. 0 = start from scratch.')
    flags.DEFINE_string ('wandb_project',  '',      'WandB project name. Empty = disabled.')
    flags.DEFINE_string ('wandb_run_name', '',      'WandB run name. Empty = auto.')
    app.run(main)
