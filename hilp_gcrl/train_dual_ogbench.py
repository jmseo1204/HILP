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
import json
import pickle
import time
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
    # 새 ckpt 저장 후 이전 ckpt 자동 삭제 (최신 1개만 유지)
    old = sorted(glob.glob(os.path.join(save_dir, 'params_*.pkl')))
    for old_path in old:
        if old_path != path:
            os.remove(old_path)
            print(f'Removed old ckpt → {old_path}')


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


def generate_heatmap_visualization(agent, env, obs_template, step, aggregator,
                                   gx=17.0, gy=12.0, grid_res=80, share_encoder=False):
    """
    V(s,g) heatmap + ∇_s V gradient field for WandB logging.
    Returns (H, W, 3) uint8 image.
    """
    # Maze geometry
    try:
        maze_map = env.unwrapped.maze_map
        UNIT  = float(env.unwrapped._maze_unit)
        OFF_X = -float(env.unwrapped._offset_x)
        OFF_Y = -float(env.unwrapped._offset_y)
        mh, mw = len(maze_map), len(maze_map[0])
        xmin = OFF_X - UNIT / 2;  xmax = OFF_X + (mw - 1) * UNIT + UNIT / 2
        ymin = OFF_Y - UNIT / 2;  ymax = OFF_Y + (mh - 1) * UNIT + UNIT / 2
        has_maze = True
    except Exception:
        xy = obs_template[:2]
        xmin, ymin = xy - 6.0;  xmax, ymax = xy + 6.0
        maze_map = None;  has_maze = False

    # Build observation grid
    xs = np.linspace(xmin, xmax, grid_res)
    ys = np.linspace(ymin, ymax, grid_res)
    X, Y = np.meshgrid(xs, ys)
    ref = obs_template.copy().astype(np.float32)
    obs_batch = np.tile(ref, (X.size, 1))
    obs_batch[:, :2] = np.stack([X.ravel(), Y.ravel()], axis=-1)

    # phi(goal)
    goal_obs = ref.copy();  goal_obs[:2] = [gx, gy]
    phi_g = np.array(agent.get_phi_goal(goal_obs[None]))  # (1, D)
    phi_g_jnp = jnp.array(phi_g[0])                       # (D,)

    # Batched value
    @jax.jit
    def _value_batch(obs):
        psi_s = agent.get_psi(obs)
        phi_rep = jnp.tile(phi_g_jnp[None], (psi_s.shape[0], 1))
        if aggregator == 'neg_l2':
            return -jnp.sqrt(jnp.maximum(((psi_s - phi_rep) ** 2).sum(-1), 1e-6))
        return (psi_s * phi_rep).sum(-1)

    chunk = 4096
    values = np.concatenate(
        [np.array(_value_batch(jnp.array(obs_batch[i:i + chunk])))
         for i in range(0, obs_batch.shape[0], chunk)]
    ).reshape(X.shape)

    # Wall mask
    if has_maze:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                c = int(round((X[i, j] - OFF_X) / UNIT))
                r = int(round((Y[i, j] - OFF_Y) / UNIT))
                if not (0 <= r < mh and 0 <= c < mw) or maze_map[r][c] == 1:
                    values[i, j] = np.nan

    # Gradient field ∇_s V(s,g)
    if aggregator == 'neg_l2':
        def _scalar_v(obs):
            psi = agent.network(obs[None], method='phi')[0]
            return -jnp.sqrt(jnp.maximum(((psi - phi_g_jnp) ** 2).sum(), 1e-6))
    else:
        def _scalar_v(obs):
            psi = agent.network(obs[None], method='phi')[0]
            return (psi * phi_g_jnp).sum()

    grad_fn = jax.jit(jax.vmap(jax.grad(_scalar_v)))
    grads = np.concatenate(
        [np.array(grad_fn(jnp.array(obs_batch[i:i + 2000])))[:, :2]
         for i in range(0, obs_batch.shape[0], 2000)]
    ).reshape(*X.shape, 2)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    if has_maze:
        try:
            ax.imshow(1 - np.array(maze_map), cmap='gray',
                      extent=(xmin, xmax, ymin, ymax),
                      origin='lower', alpha=0.3)
        except Exception:
            pass
    im = ax.pcolormesh(X, Y, values, shading='auto', cmap='viridis', alpha=0.80)
    fig.colorbar(im, ax=ax, label='V(s,g)')

    qs = max(1, grid_res // 25)
    Xq = X[::qs, ::qs];  Yq = Y[::qs, ::qs]
    gf_q = grads[::qs, ::qs, :]  # (H', W', 2)
    U_raw = gf_q[:, :, 0];  V_raw = gf_q[:, :, 1]
    mag = np.sqrt(U_raw ** 2 + V_raw ** 2)
    g_mean = float(np.nanmean(mag))
    g_std  = float(np.nanstd(mag)) + 1e-8
    scale  = np.clip((mag - g_mean) / g_std + 1.0, 0.0, None)
    denom  = mag + 1e-8
    U_norm = np.where(np.isnan(U_raw), np.nan, U_raw / denom * scale)
    V_norm = np.where(np.isnan(V_raw), np.nan, V_raw / denom * scale)

    ax.quiver(Xq, Yq, U_norm, V_norm,
              color='crimson', alpha=0.65, angles='xy', pivot='mid',
              scale=None, scale_units='xy', zorder=5,
              label=f'∇V(s,g)  (mean={g_mean:.2e}, std={g_std:.2e})')

    enc_label = 'shared' if share_encoder else 'separate'
    ax.scatter([gx], [gy], c='red', marker='*', s=500,
               edgecolors='white', linewidths=0.8,
               label=f'Goal ({gx:.1f}, {gy:.1f})', zorder=6)
    ax.set(xlabel='X', ylabel='Y',
           title=f'V(s,g)  agg={aggregator}  enc={enc_label}  step={step:,}  goal=({gx},{gy})')
    ax.legend()
    fig.tight_layout()

    canvas = FigureCanvas(fig)
    canvas.draw()
    w_px, h_px = canvas.get_width_height()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h_px, w_px, 4)[:, :, :3]
    plt.close(fig)
    return img


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

        # ---- Prohibited-zone contrastive margin loss (V-only) ---------------
        # use_neg는 Python bool (trace-time static) — False면 이 블록 전체가
        # XLA 프로그램에서 제거됨 (forward pass 3개 없음).
        if self.config['use_neg']:
            # Enforces V(s_neg, g) < V(s_free, g) - margin  (relative ordering).
            # s_free  = nearest feasible neighbor of s_neg (precomputed by xy KD-tree).
            # margin  = spatial xy-distance between s_neg and s_free (per-sample).
            # Gradient flow:
            #   psi(s_neg)    — updated
            #   phi(neg_goal) — stop_gradient
            #   psi(s_free)   — stop_gradient (stored params, reference only)
            psi_neg = self.network(
                batch['neg_states'], method='phi', params=network_params)
            phi_neg_goal = jax.lax.stop_gradient(
                self.network(batch['neg_goals'], method='phi_goal', params=network_params))
            psi_free = jax.lax.stop_gradient(
                self.network(batch['neg_free'], method='phi', params=network_params))

            if self.config['aggregator'] == 'neg_l2':
                sq_neg  = ((psi_neg  - phi_neg_goal) ** 2).sum(axis=-1)
                sq_free = ((psi_free - phi_neg_goal) ** 2).sum(axis=-1)
                v_neg  = -jnp.sqrt(jnp.maximum(sq_neg,  1e-6))
                v_free = -jnp.sqrt(jnp.maximum(sq_free, 1e-6))
            else:
                v_neg  = (psi_neg  * phi_neg_goal).sum(axis=-1)
                v_free = (psi_free * phi_neg_goal).sum(axis=-1)

            margin    = self.config['margin_scale'] * batch['neg_margins']
            violation = jnp.maximum(v_neg - v_free + margin, 0.0)
            loss_neg  = (violation ** 2).mean()
            loss = loss + self.config['lambda_neg'] * loss_neg

            info['neg/loss_raw']       = loss_neg
            info['neg/v_neg_mean']     = v_neg.mean()
            info['neg/v_free_mean']    = v_free.mean()
            info['neg/margin_mean']    = margin.mean()
            info['neg/violation_mean'] = violation.mean()

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
               lambda_neg=0.1, margin_scale=1.0, share_encoder=False, **kwargs):
        print(f'DualHILP.create — aggregator={aggregator}  extra kwargs:', kwargs)
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        # V(s,g): aggregator selects inner_prod or neg_l2
        value_def = DualGoalPhiValue(
            hidden_dims=tuple(value_hidden_dims),
            skill_dim=skill_dim,
            use_layer_norm=bool(use_layer_norm),
            aggregator=aggregator,
            share_encoder=bool(share_encoder),
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

        print(f'DualHILP.create — lambda_neg={lambda_neg}  margin_scale={margin_scale}')

        return cls(network=network,
                   config=flax.core.FrozenDict(
                       discount=discount, tau=tau, expectile=expectile,
                       skill_dim=skill_dim, aggregator=aggregator,
                       lambda_neg=lambda_neg, margin_scale=margin_scale,
                       use_neg=False))   # use_neg은 train_step 선택으로 제어


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
    all_obs             = train_data['observations']   # feasible states (full dataset)
    prohibited_obs      = None
    neg_nearest_free    = None   # (N_prohibited, obs_dim) nearest feasible obs
    neg_spatial_margins = None   # (N_prohibited,)         xy dist to nearest feasible
    if FLAGS.p_prohibit > 0 and FLAGS.prohibit_threshold > 0:
        prohibited_obs = _build_prohibited_obs(
            dataset['observations'], obs_template,
            threshold=FLAGS.prohibit_threshold)
        print(f'[DualHILP] Prohibited zone: {prohibited_obs.shape[0]} points  '
              f'(threshold={FLAGS.prohibit_threshold})')
        if prohibited_obs.shape[0] == 0:
            print('[DualHILP] WARNING: no prohibited points found, disabling negative sampling.')
            prohibited_obs = None
        else:
            # Precompute nearest feasible (dataset) neighbor for each prohibited point.
            # Uses xy only (first 2 dims) to match _build_prohibited_obs logic.
            _t_pre0 = time.perf_counter()
            feasible_xy = all_obs[:, :2].astype(np.float64)
            prohib_xy   = prohibited_obs[:, :2].astype(np.float64)
            _t_pre1 = time.perf_counter()
            tree_feasible = KDTree(feasible_xy)
            _t_pre2 = time.perf_counter()
            dists, idxs = tree_feasible.query(prohib_xy, workers=-1)
            _t_pre3 = time.perf_counter()
            neg_nearest_free    = all_obs[idxs].astype(np.float32)   # (N_prohibited, obs_dim)
            neg_spatial_margins = dists.astype(np.float32)            # (N_prohibited,)
            _t_pre4 = time.perf_counter()
            print(f'[DualHILP] Spatial margins — min={neg_spatial_margins.min():.3f}  '
                  f'mean={neg_spatial_margins.mean():.3f}  '
                  f'max={neg_spatial_margins.max():.3f}')
            print(f'[PROFILE/precompute] xy_cast={_t_pre1-_t_pre0:.3f}s  '
                  f'kdtree_build={_t_pre2-_t_pre1:.3f}s  '
                  f'kdtree_query={_t_pre3-_t_pre2:.3f}s  '
                  f'index_gather={_t_pre4-_t_pre3:.3f}s  '
                  f'total={_t_pre4-_t_pre0:.3f}s  '
                  f'n_feasible={feasible_xy.shape[0]}  n_prohib={prohib_xy.shape[0]}')

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
        margin_scale      = FLAGS.margin_scale,
        share_encoder     = FLAGS.share_encoder,
    )

    # ---- Resume from checkpoint if requested --------------------------------
    start_step = 1
    if FLAGS.resume_step > 0:
        agent = restore_agent(agent, FLAGS.save_dir, FLAGS.resume_step)
        start_step = FLAGS.resume_step + 1
        print(f'[DualHILP] Resumed from step {FLAGS.resume_step}, continuing from step {start_step}')

    # ---- Build train_step (pmap for multi-GPU, jit for single GPU) ----------
    use_neg = FLAGS.p_prohibit > 0.0 and prohibited_obs is not None

    if use_neg:
        agent_neg    = agent.replace(config=flax.core.FrozenDict({**agent.config, 'use_neg': True}))
        agent_no_neg = agent.replace(config=flax.core.FrozenDict({**agent.config, 'use_neg': False}))

    if n_devices > 1:
        if use_neg:
            agent_neg    = jax.device_put_replicated(agent_neg,    jax.local_devices())
            agent_no_neg = jax.device_put_replicated(agent_no_neg, jax.local_devices())

            @functools.partial(jax.pmap, axis_name='batch')
            def train_step_neg(agent, batch):
                return agent.update(batch, pmap_axis='batch')

            @functools.partial(jax.pmap, axis_name='batch')
            def train_step_no_neg(agent, batch):
                return agent.update(batch, pmap_axis='batch')

        agent = jax.device_put_replicated(agent, jax.local_devices()) if not use_neg else agent_no_neg

        @functools.partial(jax.pmap, axis_name='batch')
        def train_step(agent, batch):
            return agent.update(batch, pmap_axis='batch')
    else:
        if use_neg:
            @jax.jit
            def train_step_neg(agent, batch):
                return agent.update(batch)

            @jax.jit
            def train_step_no_neg(agent, batch):
                return agent.update(batch)

        agent = agent_no_neg if use_neg else agent

        @jax.jit
        def train_step(agent, batch):
            return agent.update(batch)

    # ---- Negative sampling helper -------------------------------------------
    def _inject_neg_samples(batch):
        """Always inject neg_states/neg_free/neg_margins/neg_goals/neg_weight.

        When active (probability p_prohibit): real prohibited-zone states with
        their precomputed nearest feasible neighbors and spatial margins.
        neg_weight=1.0.  When inactive: dummy zeros, neg_weight=0.0.
        Dict structure is constant across steps so JIT traces only once.
        """
        n = batch['observations'].shape[0]
        obs_dim = batch['observations'].shape[1]
        active = (prohibited_obs is not None
                  and np.random.rand() < FLAGS.p_prohibit)
        if active and prohibited_obs is not None and neg_nearest_free is not None and neg_spatial_margins is not None:
            neg_idx  = np.random.randint(prohibited_obs.shape[0], size=n)
            goal_idx = np.random.randint(all_obs.shape[0], size=n)
            batch['neg_states']  = prohibited_obs[neg_idx].astype(np.float32)
            batch['neg_free']    = neg_nearest_free[neg_idx].astype(np.float32)
            batch['neg_margins'] = neg_spatial_margins[neg_idx].astype(np.float32)
            batch['neg_goals']   = all_obs[goal_idx].astype(np.float32)
            batch['neg_weight']  = np.ones(n_devices, dtype=np.float32)
        else:
            batch['neg_states']  = np.zeros((n, obs_dim), dtype=np.float32)
            batch['neg_free']    = np.zeros((n, obs_dim), dtype=np.float32)
            batch['neg_margins'] = np.zeros(n, dtype=np.float32)
            batch['neg_goals']   = np.zeros((n, obs_dim), dtype=np.float32)
            batch['neg_weight']  = np.zeros(n_devices, dtype=np.float32)
        return batch, active

    # ---- Training loop ------------------------------------------------------
    # 일반 지표: log step 단일 값 (info_acc 제거 → per-step device sync 없음)
    # neg 지표: active step(~10%)만 누적 → interval 평균
    _NEG_STEP_KEYS = {'neg/loss_raw',
                      'neg/v_neg_mean', 'neg/v_free_mean',
                      'neg/margin_mean', 'neg/violation_mean'}
    neg_sample_count = 0
    neg_acc  = {}   # neg 지표: active step만 누적 (JAX 배열로 보관)
    _extract = (lambda v: float(v[0])) if n_devices > 1 else float

    # ---- Profiling setup ----------------------------------------------------
    _PROFILE_LOG   = os.path.join(FLAGS.save_dir, 'timing_profile.jsonl')
    _PROFILE_STEPS = 200          # 처음 200 step 기록 (JIT 컴파일 포함 구간 커버)
    _timing_acc    = {}           # phase → list[float]
    _profiling     = True         # False 가 되면 기록 중단
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    _prof_f = open(_PROFILE_LOG, 'w')
    print(f'[PROFILE] Timing log → {_PROFILE_LOG}')

    for step in tqdm.tqdm(range(start_step, FLAGS.train_steps + 1),
                          smoothing=0.1, dynamic_ncols=True):
        _t_step0 = time.perf_counter()

        _t0 = time.perf_counter()
        batch = gc_dataset.sample(FLAGS.batch_size)
        _t1 = time.perf_counter()

        if use_neg:
            batch, is_active = _inject_neg_samples(batch)
            if is_active:
                neg_sample_count += 1
        else:
            is_active = False
        _t2 = time.perf_counter()

        if n_devices > 1:
            batch = shard_batch(batch)
        _t3 = time.perf_counter()

        if use_neg:
            if is_active:
                agent_neg        = agent_neg.replace(network=agent.network)
                agent_neg, info  = train_step_neg(agent_neg, batch)
                agent            = agent_no_neg.replace(network=agent_neg.network)
            else:
                agent, info      = train_step_no_neg(agent, batch)
                agent_neg        = agent_neg.replace(network=agent.network)
        else:
            agent, info = train_step(agent, batch)
        if _profiling:
            # 프로파일링 중에만 GPU 동기화 (정확한 t_train 측정용)
            jax.tree.map(lambda x: x.block_until_ready(), info)
        _t4 = time.perf_counter()

        _t_step1 = time.perf_counter()

        if _profiling:
            _rec = {
                'step':        step,
                'is_active':   bool(is_active),
                'n_devices':   n_devices,
                't_sample_ms': round((_t1 - _t0)          * 1000, 3),
                't_inject_ms': round((_t2 - _t1)          * 1000, 3),
                't_shard_ms':  round((_t3 - _t2)          * 1000, 3),
                't_train_ms':  round((_t4 - _t3)          * 1000, 3),
                't_total_ms':  round((_t_step1 - _t_step0)* 1000, 3),
            }
            _prof_f.write(json.dumps(_rec) + '\n')
            _prof_f.flush()

            for k, v in _rec.items():
                if k.startswith('t_'):
                    _timing_acc.setdefault(k, []).append(v)

            if step - start_step >= _PROFILE_STEPS:
                summary = {k: round(sum(vs) / len(vs), 3) for k, vs in _timing_acc.items()}
                _prof_f.write(json.dumps({'step': 'SUMMARY_AVG', **summary}) + '\n')
                _prof_f.flush()
                _prof_f.close()
                _profiling = False
                print(f'[PROFILE] avg over {_PROFILE_STEPS} steps: '
                      + '  '.join(f'{k}={v:.1f}ms' for k, v in summary.items()))

        # neg 지표: JAX 배열 참조만 보관 (sync 없음)
        if is_active:
            for k in ('neg/loss_raw', 'neg/v_neg_mean', 'neg/v_free_mean',
                      'neg/margin_mean', 'neg/violation_mean'):
                if k in info:
                    neg_acc.setdefault(k, []).append(info[k])

        if step % FLAGS.log_interval == 0:
            # 일반 지표 + neg 지표 모두 log 시점에 한꺼번에 sync (이전 구조와 동일)
            log_info = {k: _extract(v) for k, v in info.items()
                        if k not in _NEG_STEP_KEYS}

            log_info['neg/sample_count']    = neg_sample_count
            log_info['neg/sample_rate']     = neg_sample_count / FLAGS.log_interval
            log_info['neg/prohibited_size'] = prohibited_obs.shape[0] if prohibited_obs is not None else 0
            for k in ('neg/loss_raw', 'neg/v_neg_mean', 'neg/v_free_mean',
                      'neg/margin_mean', 'neg/violation_mean'):
                log_info[k + '_avg'] = float(np.mean([_extract(v) for v in neg_acc[k]])) if neg_acc.get(k) else float('nan')

            log_str = '  '.join(
                f'{k}={v:.4f}' for k, v in log_info.items()
                if not (isinstance(v, float) and np.isnan(v))
            )
            tqdm.tqdm.write(f'[step {step:>8d}] {log_str}')
            if FLAGS.wandb_project:
                wandb.log(log_info, step=step)

            neg_sample_count = 0
            neg_acc.clear()

        if FLAGS.viz_interval > 0 and step % FLAGS.viz_interval == 0:
            if FLAGS.wandb_project and viz_obs.shape[0] > 0:
                single_agent = jax.tree.map(lambda x: x[0], agent) if n_devices > 1 else agent
                tsne_img = generate_tsne_visualization(
                    single_agent, viz_obs, viz_xy, step, FLAGS.seed, FLAGS.aggregator)
                heatmap_img = generate_heatmap_visualization(
                    single_agent, env, obs_template, step, FLAGS.aggregator,
                    gx=17.0, gy=12.0, share_encoder=FLAGS.share_encoder)
                wandb.log({
                    'diagnostics/psi_tsne':  wandb.Image(tsne_img),
                    'diagnostics/heatmap':   wandb.Image(heatmap_img),
                }, step=step)
                tqdm.tqdm.write(f'[step {step:>8d}] Logged t-SNE + heatmap to WandB.')

        if step % FLAGS.save_interval == 0:
            save_agent(
                jax.tree.map(lambda x: x[0], agent) if n_devices > 1 else agent,
                FLAGS.save_dir, step)

    if _profiling:   # 학습이 _PROFILE_STEPS 전에 끝났을 경우 파일 닫기
        _prof_f.close()
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
    flags.DEFINE_float  ('lambda_neg',    0.1,     'Weight of prohibited-zone contrastive margin loss.')
    flags.DEFINE_float  ('margin_scale',  1.0,     'Scale applied to spatial xy-distance margin: margin = margin_scale * dist(s_neg, s_free).')
    flags.DEFINE_bool   ('share_encoder', False,   'Share psi/phi encoder (True) or use separate encoders (False).')
    flags.DEFINE_float  ('prohibit_threshold', 1.5, 'KD-tree distance threshold for prohibited zone.')
    flags.DEFINE_integer('resume_step',    0,       'Resume from this step. 0 = start from scratch.')
    flags.DEFINE_string ('wandb_project',  '',      'WandB project name. Empty = disabled.')
    flags.DEFINE_string ('wandb_run_name', '',      'WandB run name. Empty = auto.')
    app.run(main)
