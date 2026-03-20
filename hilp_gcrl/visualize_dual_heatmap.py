"""
Flexible heatmap visualization for Dual Goal Representations (arXiv:2510.06714).

Two modes selectable via --mode:
  dual_repr  — V(s,g) = psi(s)^T phi(g)     (Phase-1 temporal distance)
  gcvf       — V_down(s, phi(g))             (Phase-2 downstream GCVF)

For 'dual_repr':
  --restore_path / --restore_epoch  →  DualHILP checkpoint

For 'gcvf':
  --restore_path / --restore_epoch  →  GCVFDual  checkpoint
  --dual_restore_path / --dual_restore_epoch  →  DualHILP checkpoint (for phi(g))

Outputs a PNG heatmap to --save_dir.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from absl import app, flags

# ---- Shared utilities & agents -----------------------------------------------
_SCOTS_DIR = Path(__file__).parent.parent.parent / 'scots' / 'scripts' / 'HILP'
_HERE      = Path(__file__).parent
sys.path.insert(0, str(_SCOTS_DIR))
sys.path.insert(0, str(_HERE))

from hilp_ogbench import restore_agent
from train_dual_ogbench import DualHILP
from train_gcvf_dual_ogbench import GCVFDual
# ------------------------------------------------------------------------------

FLAGS = flags.FLAGS

flags.DEFINE_enum  ('mode',         'dual_repr', ['dual_repr', 'gcvf'],
                    'Visualization mode: dual_repr = psi(s)^T phi(g), gcvf = V_down(s,phi(g)).')
flags.DEFINE_string('env_name',     'pointmaze-large-stitch-v0', 'OGBench environment.')
flags.DEFINE_string('restore_path', 'exp/dual_repr',  'Checkpoint directory (mode-specific model).')
flags.DEFINE_integer('restore_epoch', 1000000,         'Checkpoint step to load.')
flags.DEFINE_string('dual_restore_path',  None,        '[gcvf mode] DualHILP checkpoint dir.')
flags.DEFINE_integer('dual_restore_epoch', 1000000,    '[gcvf mode] DualHILP checkpoint step.')
flags.DEFINE_integer('skill_dim',   32,  'Must match training skill_dim.')
flags.DEFINE_multi_integer('value_hidden_dims', [512, 512, 512], 'Must match training hidden dims.')
flags.DEFINE_float ('discount',     0.99, 'Must match training discount.')
flags.DEFINE_float ('expectile',    0.95, 'Must match training expectile.')
flags.DEFINE_integer('use_layer_norm', 1, 'Must match training use_layer_norm.')
flags.DEFINE_list  ('goal_pos',     ['12.0', '8.0'],
                    'Goal (x, y) coordinates for the heatmap.')
flags.DEFINE_integer('grid_res',    100,  'Grid resolution (grid_res x grid_res).')
flags.DEFINE_string ('save_dir',    'visualizations', 'Output directory for PNGs.')
flags.DEFINE_integer('seed',        0,    'Random seed.')


# ======================== Helpers =============================================

def _build_maze_info(env):
    """Extract maze geometry for wall-masking and background rendering."""
    maze_map = env.unwrapped.maze_map
    UNIT  = float(env.unwrapped._maze_unit)
    OFF_X = -float(env.unwrapped._offset_x)
    OFF_Y = -float(env.unwrapped._offset_y)
    maze_h, maze_w = maze_map.shape
    maze_xmin = OFF_X - UNIT / 2
    maze_xmax = OFF_X + (maze_w - 1) * UNIT + UNIT / 2
    maze_ymin = OFF_Y - UNIT / 2
    maze_ymax = OFF_Y + (maze_h - 1) * UNIT + UNIT / 2
    return dict(maze_map=maze_map, UNIT=UNIT, OFF_X=OFF_X, OFF_Y=OFF_Y,
                maze_h=maze_h, maze_w=maze_w,
                xmin=maze_xmin, xmax=maze_xmax, ymin=maze_ymin, ymax=maze_ymax)


def _apply_wall_mask(values, X, Y, maze_info):
    """Set wall / out-of-bounds cells to NaN."""
    maze_map = maze_info['maze_map']
    UNIT = maze_info['UNIT']
    OFF_X, OFF_Y = maze_info['OFF_X'], maze_info['OFF_Y']
    maze_h, maze_w = maze_info['maze_h'], maze_info['maze_w']

    wall_mask = np.zeros_like(X, dtype=bool)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            c = int(round((X[i, j] - OFF_X) / UNIT))
            r = int(round((Y[i, j] - OFF_Y) / UNIT))
            if 0 <= r < maze_h and 0 <= c < maze_w:
                if maze_map[r, c] == 1:
                    wall_mask[i, j] = True
            else:
                wall_mask[i, j] = True
    return np.where(wall_mask, np.nan, values)


def _compute_values_batched(value_fn, obs_batch, batch_size=5000):
    """Call value_fn(obs_batch) in chunks to avoid OOM."""
    results = []
    n = obs_batch.shape[0]
    for start in range(0, n, batch_size):
        results.append(np.array(value_fn(obs_batch[start:start + batch_size])))
    return np.concatenate(results)


def _plot_heatmap(X, Y, values, maze_info, gx, gy, title, cbar_label, save_path):
    fig, ax = plt.subplots(figsize=(12, 10))

    # Maze background
    try:
        mi = maze_info
        ax.imshow(1 - mi['maze_map'], cmap='gray',
                  extent=[mi['xmin'], mi['xmax'], mi['ymin'], mi['ymax']],
                  origin='lower', alpha=0.3)
    except Exception as e:
        print(f'  Warning: maze background failed: {e}')

    im   = ax.pcolormesh(X, Y, values, shading='auto', cmap='viridis', alpha=0.75)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    ax.scatter([gx], [gy], c='red', marker='*', s=500,
               edgecolors='white', linewidths=0.8,
               label=f'Goal ({gx:.2f}, {gy:.2f})', zorder=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'  Saved → {save_path}')


# ======================== Mode implementations ================================

def _run_dual_repr(env, obs_all, gx, gy, maze_info):
    """Visualize V(s,g) = psi(s)^T phi(g) from the dual representation."""
    print('[dual_repr] Loading DualHILP checkpoint ...')
    ex_obs    = obs_all[:1]
    agent     = DualHILP.create(
        seed=FLAGS.seed, ex_observations=ex_obs,
        value_hidden_dims=tuple(FLAGS.value_hidden_dims),
        skill_dim=FLAGS.skill_dim, discount=FLAGS.discount,
        expectile=FLAGS.expectile, use_layer_norm=FLAGS.use_layer_norm,
    )
    agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Build grid
    mi = maze_info
    xs, ys = np.linspace(mi['xmin'], mi['xmax'], FLAGS.grid_res), \
             np.linspace(mi['ymin'], mi['ymax'], FLAGS.grid_res)
    X, Y      = np.meshgrid(xs, ys)
    grid_xy   = np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32)
    ref_obs   = obs_all[0].copy()
    n         = grid_xy.shape[0]
    obs_batch = np.tile(ref_obs, (n, 1))
    obs_batch[:, :2] = grid_xy

    goal_obs = ref_obs.copy()
    goal_obs[:2] = [gx, gy]

    # Precompute phi(g)  — single goal, broadcast
    phi_g = np.array(agent.get_phi_goal(goal_obs[None]))  # (1, D)
    phi_g_batch = np.tile(phi_g, (1, 1))                  # kept as (1, D)

    @jax.jit
    def value_fn(obs):
        psi_s = agent.get_psi(obs)                          # (B, D)
        phi_g_tiled = jnp.tile(phi_g, (psi_s.shape[0], 1)) # (B, D)
        return (psi_s * phi_g_tiled).sum(axis=-1)           # (B,)

    print(f'[dual_repr] Computing values on {n} grid points ...')
    values = _compute_values_batched(value_fn, obs_batch).reshape(X.shape)
    values = _apply_wall_mask(values, X, Y, maze_info)

    return X, Y, values, \
           f'Dual Repr  V(s,g) = ψ(s)ᵀφ(g)\n{FLAGS.env_name}  |  Goal ({gx:.2f}, {gy:.2f})', \
           'V(s,g) = ψ(s)ᵀφ(g)  [temporal distance]'


def _run_gcvf(env, obs_all, gx, gy, maze_info):
    """Visualize V_down(s, phi(g)) — downstream GCVF on frozen dual repr."""
    if FLAGS.dual_restore_path is None:
        raise ValueError('--dual_restore_path must be set in gcvf mode.')

    print('[gcvf] Loading frozen DualHILP checkpoint ...')
    ex_obs     = obs_all[:1]
    dual_agent = DualHILP.create(
        seed=FLAGS.seed, ex_observations=ex_obs,
        value_hidden_dims=tuple(FLAGS.value_hidden_dims),
        skill_dim=FLAGS.skill_dim, discount=FLAGS.discount,
        expectile=FLAGS.expectile, use_layer_norm=FLAGS.use_layer_norm,
    )
    dual_agent = restore_agent(dual_agent, FLAGS.dual_restore_path, FLAGS.dual_restore_epoch)

    print('[gcvf] Loading GCVFDual checkpoint ...')
    gcvf_agent = GCVFDual.create(
        seed=FLAGS.seed, ex_observations=ex_obs,
        skill_dim=FLAGS.skill_dim, lr=3e-4,
        value_hidden_dims=tuple(FLAGS.value_hidden_dims),
        discount=FLAGS.discount, expectile=FLAGS.expectile,
        use_layer_norm=FLAGS.use_layer_norm,
    )
    gcvf_agent = restore_agent(gcvf_agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Build grid
    mi = maze_info
    xs, ys = np.linspace(mi['xmin'], mi['xmax'], FLAGS.grid_res), \
             np.linspace(mi['ymin'], mi['ymax'], FLAGS.grid_res)
    X, Y      = np.meshgrid(xs, ys)
    grid_xy   = np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32)
    ref_obs   = obs_all[0].copy()
    n         = grid_xy.shape[0]
    obs_batch = np.tile(ref_obs, (n, 1))
    obs_batch[:, :2] = grid_xy

    goal_obs = ref_obs.copy()
    goal_obs[:2] = [gx, gy]

    phi_g       = np.array(dual_agent.get_phi_goal(goal_obs[None]))  # (1, D)

    @jax.jit
    def value_fn(obs):
        phi_g_tiled = jnp.tile(phi_g, (obs.shape[0], 1))   # (B, D)
        return gcvf_agent.get_value(obs, phi_g_tiled)       # (B,)

    print(f'[gcvf] Computing values on {n} grid points ...')
    values = _compute_values_batched(value_fn, obs_batch).reshape(X.shape)
    values = _apply_wall_mask(values, X, Y, maze_info)

    return X, Y, values, \
           f'Downstream GCVF  V(s, φ(g))\n{FLAGS.env_name}  |  Goal ({gx:.2f}, {gy:.2f})', \
           'V_down(s, φ(g))  [goal-conditioned value]'


# ======================== Main ================================================

def main(_):
    import ogbench
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    gx = float(FLAGS.goal_pos[0])
    gy = float(FLAGS.goal_pos[1])
    print(f'Mode: {FLAGS.mode}  |  env: {FLAGS.env_name}  |  goal: ({gx}, {gy})')

    # ---- Environment & dataset -----------------------------------------------
    env, dataset, _ = ogbench.make_env_and_datasets(
        FLAGS.env_name, compact_dataset=False)
    obs_all = dataset['observations'].astype(np.float32)

    # ---- Maze geometry -------------------------------------------------------
    try:
        maze_info = _build_maze_info(env)
        print(f"Maze bounds: X[{maze_info['xmin']:.2f},{maze_info['xmax']:.2f}]  "
              f"Y[{maze_info['ymin']:.2f},{maze_info['ymax']:.2f}]")
    except Exception as e:
        print(f'Warning: Could not parse maze geometry ({e}). Using dataset bounds.')
        x_min, y_min = obs_all[:, :2].min(axis=0) - 2.0
        x_max, y_max = obs_all[:, :2].max(axis=0) + 2.0
        maze_info = dict(maze_map=None, UNIT=1.0, OFF_X=0.0, OFF_Y=0.0,
                         maze_h=0, maze_w=0,
                         xmin=x_min, xmax=x_max, ymin=y_min, ymax=y_max)

    # ---- Compute values ------------------------------------------------------
    if FLAGS.mode == 'dual_repr':
        X, Y, values, title, cbar_label = _run_dual_repr(env, obs_all, gx, gy, maze_info)
    else:
        X, Y, values, title, cbar_label = _run_gcvf(env, obs_all, gx, gy, maze_info)

    # ---- Save heatmap --------------------------------------------------------
    gx_str   = f'{gx:.1f}'.replace('.', '_').replace('-', 'm')
    gy_str   = f'{gy:.1f}'.replace('.', '_').replace('-', 'm')
    filename = f'dual_{FLAGS.mode}_{FLAGS.env_name}_x{gx_str}_y{gy_str}_ep{FLAGS.restore_epoch}.png'
    save_path = os.path.join(FLAGS.save_dir, filename)
    _plot_heatmap(X, Y, values, maze_info, gx, gy, title, cbar_label, save_path)


if __name__ == '__main__':
    app.run(main)
