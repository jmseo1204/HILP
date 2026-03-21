"""
Flexible heatmap visualization for Dual Goal Representations (arXiv:2510.06714).

Two modes (--mode):
  dual_repr  —  V(s,g) = psi(s)^T phi(g)      (Phase-1 temporal distance)
  gcvf       —  V_down(s, phi(g))              (Phase-2 downstream GCVF)

For 'dual_repr':
  --restore_path / --restore_epoch  →  DualHILP checkpoint

For 'gcvf':
  --restore_path / --restore_epoch  →  GCVFDual checkpoint
  --dual_restore_path / --dual_restore_epoch  →  DualHILP checkpoint (for phi(g))

All dependencies are inside hilp_gcrl/.
"""

import os
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from absl import app, flags

# ---- hilp_gcrl internal imports ---------------------------------------------
_ROOT = Path(__file__).parent          # hilp_gcrl/
sys.path.insert(0, str(_ROOT))

from train_dual_ogbench import DualHILP, restore_agent
from train_gcvf_dual_ogbench import GCVFDual
# -----------------------------------------------------------------------------

FLAGS = flags.FLAGS
flags.DEFINE_enum   ('mode',               'dual_repr', ['dual_repr', 'gcvf'],
                     'dual_repr = psi(s)^T phi(g),  gcvf = V_down(s, phi(g)).')
flags.DEFINE_string ('env_name',           'antmaze-giant-navigate-v0', 'OGBench env.')
flags.DEFINE_string ('restore_path',       'exp/dual_repr',  'Checkpoint dir (mode-specific).')
flags.DEFINE_integer('restore_epoch',      1000000,          'Checkpoint step.')
flags.DEFINE_string ('dual_restore_path',  None,   '[gcvf] DualHILP checkpoint dir.')
flags.DEFINE_integer('dual_restore_epoch', 1000000,'[gcvf] DualHILP checkpoint step.')
flags.DEFINE_integer('skill_dim',          32,     'Must match training skill_dim.')
flags.DEFINE_multi_integer('value_hidden_dims', [512, 512, 512], 'Must match training.')
flags.DEFINE_float  ('discount',           0.99,   'Must match training.')
flags.DEFINE_float  ('expectile',          0.95,   'Must match training.')
flags.DEFINE_integer('use_layer_norm',     1,      'Must match training.')
flags.DEFINE_string ('aggregator',         'inner_prod',
                     'Value aggregation used at training: inner_prod or neg_l2.')
flags.DEFINE_list   ('goal_pos',           ['12.0', '8.0'], 'Goal (x, y).')
flags.DEFINE_integer('grid_res',           100,    'Grid resolution.')
flags.DEFINE_string ('save_dir',           'visualizations', 'Output dir.')
flags.DEFINE_integer('seed',               0,      'Seed.')


# ======================== Helpers =============================================

def _maze_info(env):
    maze_map = env.unwrapped.maze_map
    UNIT  = float(env.unwrapped._maze_unit)
    OFF_X = -float(env.unwrapped._offset_x)
    OFF_Y = -float(env.unwrapped._offset_y)
    h, w  = maze_map.shape
    return dict(
        maze_map=maze_map, UNIT=UNIT, OFF_X=OFF_X, OFF_Y=OFF_Y,
        maze_h=h, maze_w=w,
        xmin=OFF_X - UNIT/2, xmax=OFF_X + (w-1)*UNIT + UNIT/2,
        ymin=OFF_Y - UNIT/2, ymax=OFF_Y + (h-1)*UNIT + UNIT/2,
    )


def _wall_mask(values, X, Y, mi):
    mask = np.zeros_like(X, dtype=bool)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            c = int(round((X[i, j] - mi['OFF_X']) / mi['UNIT']))
            r = int(round((Y[i, j] - mi['OFF_Y']) / mi['UNIT']))
            if 0 <= r < mi['maze_h'] and 0 <= c < mi['maze_w']:
                if mi['maze_map'][r, c] == 1:
                    mask[i, j] = True
            else:
                mask[i, j] = True
    return np.where(mask, np.nan, values)


def _batched(fn, arr, chunk=5000):
    return np.concatenate([np.array(fn(arr[i:i+chunk]))
                           for i in range(0, arr.shape[0], chunk)])


def _compute_value_grad_field(scalar_v_fn, obs_batch, X, chunk=2000):
    """
    Compute dV/d(x,y) for each grid point using JAX autograd.

    scalar_v_fn : callable (obs_dim,) -> scalar, JAX-differentiable.
    obs_batch   : (N, obs_dim) numpy float32.
    X           : (H, W) meshgrid for shape reference.

    Returns: grad_field (H, W, 2), norms (H, W).
    """
    grad_fn = jax.jit(jax.vmap(jax.grad(scalar_v_fn)))
    N = obs_batch.shape[0]
    print(f'  Computing value gradient field ({N} points) ...')
    grads_list = []
    for i in range(0, N, chunk):
        g = np.array(grad_fn(jnp.array(obs_batch[i:i+chunk])))
        grads_list.append(g[:, :2])   # only x, y components
    grads_xy = np.concatenate(grads_list, axis=0)  # (N, 2)
    grad_field = grads_xy.reshape(*X.shape, 2)
    norms = np.linalg.norm(grad_field, axis=-1)
    return grad_field, norms


def _normalize_grad(grads):
    """
    Normalize gradient field by global mean/std of magnitudes (same as TD_field).
    Returns (U_norm, V_norm, mean_scalar, std_scalar).
    """
    U_raw = grads[:, :, 0]
    V_raw = grads[:, :, 1]
    mag = np.sqrt(U_raw**2 + V_raw**2)
    m = float(np.nanmean(mag))
    s = float(np.nanstd(mag)) + 1e-8
    # Shift so mean-magnitude arrows → length ≈ 1; clip to avoid direction flip
    scale = np.clip((mag - m) / s + 1.0, 0.0, None)
    denom = mag + 1e-8
    U_norm = np.where(np.isnan(U_raw), np.nan, U_raw / denom * scale)
    V_norm = np.where(np.isnan(V_raw), np.nan, V_raw / denom * scale)
    return U_norm, V_norm, m, s


def _plot(X, Y, values, mi, gx, gy, title, cbar_label, path,
          grad_field=None, quiver_step=3):
    fig, ax = plt.subplots(figsize=(12, 10))
    try:
        ax.imshow(1 - mi['maze_map'], cmap='gray',
                  extent=(mi['xmin'], mi['xmax'], mi['ymin'], mi['ymax']),
                  origin='lower', alpha=0.3)
    except Exception as e:
        print(f'  Maze background skipped: {e}')
    im = ax.pcolormesh(X, Y, values, shading='auto', cmap='viridis', alpha=0.75)
    fig.colorbar(im, ax=ax, label=cbar_label)

    # ---- Gradient field arrows (∇_s V(s,g)) ---------------------------------
    if grad_field is not None:
        Xq = X[::quiver_step, ::quiver_step]
        Yq = Y[::quiver_step, ::quiver_step]
        gf_q = grad_field[::quiver_step, ::quiver_step, :]   # (H', W', 2)

        # Mask wall/NaN cells before normalization
        val_q = values[::quiver_step, ::quiver_step]
        gf_q = np.where(np.isnan(val_q)[..., None], np.nan, gf_q)

        Un, Vn, g_mean, g_std = _normalize_grad(gf_q)

        print(f'  [∇V diag] mag mean={g_mean:.4e}  std={g_std:.4e}')

        ax.quiver(Xq, Yq, Un, Vn,
                  color='crimson', alpha=0.70,
                  angles='xy', pivot='mid',
                  scale=None,
                  zorder=5,
                  label=f'∇V(s,g)  (mean={g_mean:.2e}, std={g_std:.2e})')

    ax.scatter([gx], [gy], c='red', marker='*', s=500,
               edgecolors='white', linewidths=0.8,
               label=f'Goal ({gx:.2f}, {gy:.2f})', zorder=6)
    ax.set(xlabel='X', ylabel='Y', title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'  Saved → {path}')


def _make_grid(obs_all, mi):
    xs = np.linspace(mi['xmin'], mi['xmax'], FLAGS.grid_res)
    ys = np.linspace(mi['ymin'], mi['ymax'], FLAGS.grid_res)
    X, Y = np.meshgrid(xs, ys)
    xy   = np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32)
    ref  = obs_all[0].copy()
    n    = xy.shape[0]
    obs_batch = np.tile(ref, (n, 1))
    obs_batch[:, :2] = xy
    return X, Y, obs_batch, ref


# ======================== Mode implementations ================================

def _run_dual_repr(obs_all, ex_act, mi):
    ex_obs = obs_all[:1]
    agent  = DualHILP.create(
        seed=FLAGS.seed, ex_observations=ex_obs, ex_actions=ex_act,
        value_hidden_dims=tuple(FLAGS.value_hidden_dims),
        skill_dim=FLAGS.skill_dim, discount=FLAGS.discount,
        expectile=FLAGS.expectile, use_layer_norm=FLAGS.use_layer_norm,
        aggregator=FLAGS.aggregator,
    )
    agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    gx, gy = float(FLAGS.goal_pos[0]), float(FLAGS.goal_pos[1])
    X, Y, obs_batch, ref = _make_grid(obs_all, mi)

    goal_obs = ref.copy(); goal_obs[:2] = [gx, gy]
    phi_g    = np.array(agent.get_phi_goal(goal_obs[None]))  # (1, D)

    agg = agent.config['aggregator']

    @jax.jit
    def value_fn(obs):
        psi_s     = agent.get_psi(obs)
        phi_g_rep = jnp.tile(phi_g, (psi_s.shape[0], 1))
        if agg == 'neg_l2':
            diff = psi_s - phi_g_rep
            return -jnp.sqrt(jnp.maximum((diff ** 2).sum(axis=-1), 1e-6))
        else:  # inner_prod
            return (psi_s * phi_g_rep).sum(axis=-1)

    print(f'[dual_repr] aggregator={agg}  grid points={obs_batch.shape[0]} ...')
    values = _batched(value_fn, obs_batch).reshape(X.shape)
    values = _wall_mask(values, X, Y, mi)

    # ---- Gradient field ∇_s V(s,g) ------------------------------------------
    phi_g_jnp = jnp.array(phi_g[0])  # (D,)
    if agg == 'neg_l2':
        def scalar_v(single_obs):
            psi_s = agent.network(single_obs[None], method='phi')[0]  # (D,)
            diff  = psi_s - phi_g_jnp
            return -jnp.sqrt(jnp.maximum((diff**2).sum(), 1e-6))
    else:
        def scalar_v(single_obs):
            psi_s = agent.network(single_obs[None], method='phi')[0]  # (D,)
            return (psi_s * phi_g_jnp).sum()

    grad_field, _ = _compute_value_grad_field(scalar_v, obs_batch, X)
    # Mask walls
    grad_field = np.where(np.isnan(values)[..., None], np.nan, grad_field)

    if agg == 'neg_l2':
        v_formula, cbar_label = 'V = -‖ψ(s)−φ(g)‖', 'V(s,g) = -‖ψ(s)−φ(g)‖  [neg L2 dist]'
    else:
        v_formula, cbar_label = 'V = ψ(s)ᵀφ(g)', 'V(s,g) = ψ(s)ᵀφ(g)  [inner product]'
    title = f'Dual Repr  {v_formula}\n{FLAGS.env_name}  |  Goal ({gx:.2f}, {gy:.2f})'
    return X, Y, values, gx, gy, title, cbar_label, grad_field


def _run_gcvf(obs_all, ex_act, mi):
    if FLAGS.dual_restore_path is None:
        raise ValueError('--dual_restore_path is required in gcvf mode.')

    ex_obs = obs_all[:1]

    # Frozen Phase-1 dual agent
    dual_agent = DualHILP.create(
        seed=FLAGS.seed, ex_observations=ex_obs, ex_actions=ex_act,
        value_hidden_dims=tuple(FLAGS.value_hidden_dims),
        skill_dim=FLAGS.skill_dim, discount=FLAGS.discount,
        expectile=FLAGS.expectile, use_layer_norm=FLAGS.use_layer_norm,
        aggregator=FLAGS.aggregator,
    )
    dual_agent = restore_agent(dual_agent, FLAGS.dual_restore_path, FLAGS.dual_restore_epoch)

    # Phase-2 downstream GCVF
    gcvf_agent = GCVFDual.create(
        seed=FLAGS.seed, ex_observations=ex_obs,
        skill_dim=FLAGS.skill_dim, lr=3e-4,
        value_hidden_dims=tuple(FLAGS.value_hidden_dims),
        discount=FLAGS.discount, expectile=FLAGS.expectile,
        use_layer_norm=FLAGS.use_layer_norm,
    )
    gcvf_agent = restore_agent(gcvf_agent, FLAGS.restore_path, FLAGS.restore_epoch)

    gx, gy = float(FLAGS.goal_pos[0]), float(FLAGS.goal_pos[1])
    X, Y, obs_batch, ref = _make_grid(obs_all, mi)

    goal_obs = ref.copy(); goal_obs[:2] = [gx, gy]
    phi_g    = np.array(dual_agent.get_phi_goal(goal_obs[None]))  # (1, D)

    @jax.jit
    def value_fn(obs):
        phi_g_rep = jnp.tile(phi_g, (obs.shape[0], 1))
        return gcvf_agent.get_value(obs, phi_g_rep)

    print(f'[gcvf] Computing values on {obs_batch.shape[0]} grid points ...')
    values = _batched(value_fn, obs_batch).reshape(X.shape)
    values = _wall_mask(values, X, Y, mi)

    # ---- Gradient field ∇_s V_down(s, phi(g)) --------------------------------
    phi_g_jnp = jnp.array(phi_g)   # (1, D) — kept as 2D for network call

    def scalar_v(single_obs):
        v1, v2 = gcvf_agent.network(single_obs[None], phi_g_jnp, method='value')
        return (v1[0] + v2[0]) / 2

    grad_field, _ = _compute_value_grad_field(scalar_v, obs_batch, X)
    grad_field = np.where(np.isnan(values)[..., None], np.nan, grad_field)

    return X, Y, values, gx, gy, \
        f'Downstream GCVF  V(s, φ(g))\n{FLAGS.env_name}  |  Goal ({gx:.2f}, {gy:.2f})', \
        'V_down(s, φ(g))  [goal-conditioned value]', \
        grad_field


# ======================== Main ===============================================

def main(_):
    import ogbench
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    env, dataset, _ = ogbench.make_env_and_datasets(
        FLAGS.env_name, compact_dataset=False)
    obs_all = dataset['observations'].astype(np.float32)
    ex_act  = dataset['actions'][:1].astype(np.float32)

    try:
        mi = _maze_info(env)
        print(f"Maze bounds: X[{mi['xmin']:.2f},{mi['xmax']:.2f}]  "
              f"Y[{mi['ymin']:.2f},{mi['ymax']:.2f}]")
    except Exception as e:
        print(f'Maze geometry unavailable ({e}), using dataset bounds.')
        xmn, ymn = obs_all[:, :2].min(axis=0) - 2.0
        xmx, ymx = obs_all[:, :2].max(axis=0) + 2.0
        mi = dict(maze_map=None, UNIT=1.0, OFF_X=0.0, OFF_Y=0.0,
                  maze_h=0, maze_w=0, xmin=xmn, xmax=xmx, ymin=ymn, ymax=ymx)

    if FLAGS.mode == 'dual_repr':
        X, Y, values, gx, gy, title, cbar, grad_field = _run_dual_repr(obs_all, ex_act, mi)
    else:
        X, Y, values, gx, gy, title, cbar, grad_field = _run_gcvf(obs_all, ex_act, mi)

    gx_s  = f'{gx:.1f}'.replace('.', '_').replace('-', 'm')
    gy_s  = f'{gy:.1f}'.replace('.', '_').replace('-', 'm')
    fname = (f'dual_{FLAGS.mode}_{FLAGS.env_name}'
             f'_x{gx_s}_y{gy_s}_ep{FLAGS.restore_epoch}.png')
    _plot(X, Y, values, mi, gx, gy, title, cbar,
          os.path.join(FLAGS.save_dir, fname),
          grad_field=grad_field)


if __name__ == '__main__':
    app.run(main)
