"""
Visualize the 'prohibited zone' of an OGBench maze environment.

Definition
----------
1. Collect all (x, y) positions from the training dataset (obs[:2]).
2. Compute the axis-aligned bounding rectangle of the point cloud.
3. Build a fine grid inside that rectangle.
4. For each grid cell, compute the distance to the nearest dataset point
   (via KD-tree).
5. Cells whose nearest-neighbor distance >= THRESHOLD are 'prohibited zones'
   (wall interiors, inaccessible regions, etc.)
6. Visualise:
     - Prohibited zone  : filled heatmap (distance value, capped at 2×threshold)
     - Bounding box     : dashed rectangle
     - Point cloud      : thin scatter (down-sampled for speed)
     - Maze overlay     : grey wall image if available
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import KDTree
from absl import app, flags

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

FLAGS = flags.FLAGS
flags.DEFINE_string ('env_name',    'antmaze-giant-navigate-v0', 'OGBench env.')
flags.DEFINE_float  ('threshold',   0.5,   'Min distance to dataset → prohibited.')
flags.DEFINE_integer('grid_res',    300,   'Grid resolution (cells per axis).')
flags.DEFINE_integer('pc_subsample',10,    'Keep 1 in N point-cloud points for scatter.')
flags.DEFINE_string ('save_dir',    'visualizations', 'Output directory.')


# ─────────────────────────── helpers ──────────────────────────────────────────

def _maze_info(env):
    maze_map = env.unwrapped.maze_map
    UNIT  = float(env.unwrapped._maze_unit)
    OFF_X = -float(env.unwrapped._offset_x)
    OFF_Y = -float(env.unwrapped._offset_y)
    h, w  = maze_map.shape
    return dict(
        maze_map=maze_map,
        xmin=OFF_X - UNIT/2, xmax=OFF_X + (w-1)*UNIT + UNIT/2,
        ymin=OFF_Y - UNIT/2, ymax=OFF_Y + (h-1)*UNIT + UNIT/2,
    )


# ─────────────────────────── main ─────────────────────────────────────────────

def main(_):
    import ogbench

    os.makedirs(FLAGS.save_dir, exist_ok=True)

    print(f'Loading dataset: {FLAGS.env_name} …')
    env, dataset, _ = ogbench.make_env_and_datasets(
        FLAGS.env_name, compact_dataset=False)

    xy = dataset['observations'][:, :2].astype(np.float64)   # (N, 2)
    print(f'  Dataset size : {xy.shape[0]:,} transitions')

    # ── 1. Bounding rectangle ─────────────────────────────────────────────────
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    print(f'  Bounding box : X[{x_min:.2f}, {x_max:.2f}]  '
          f'Y[{y_min:.2f}, {y_max:.2f}]')

    # ── 2. Grid ───────────────────────────────────────────────────────────────
    xs = np.linspace(x_min, x_max, FLAGS.grid_res)
    ys = np.linspace(y_min, y_max, FLAGS.grid_res)
    X, Y = np.meshgrid(xs, ys)                           # (R, R)
    grid_pts = np.stack([X.ravel(), Y.ravel()], axis=1)  # (R², 2)

    # ── 3. KD-tree nearest-neighbour distances ────────────────────────────────
    print(f'  Building KD-tree from {xy.shape[0]:,} points …')
    tree = KDTree(xy)

    print(f'  Querying {grid_pts.shape[0]:,} grid cells …')
    dist, _ = tree.query(grid_pts, workers=-1)            # parallel query
    dist_map = dist.reshape(X.shape)                      # (R, R)

    # ── 4. Prohibited mask ────────────────────────────────────────────────────
    prohibited = dist_map >= FLAGS.threshold              # True = prohibited

    n_total      = prohibited.size
    n_prohibited = prohibited.sum()
    print(f'  Threshold    : {FLAGS.threshold}')
    print(f'  Prohibited   : {n_prohibited:,} / {n_total:,} cells '
          f'({100 * n_prohibited / n_total:.1f} %)')

    # ── 5. Maze background (optional) ────────────────────────────────────────
    try:
        mi = _maze_info(env)
    except Exception as e:
        print(f'  Maze overlay unavailable: {e}')
        mi = None

    # ── 6. Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    fig.suptitle(
        f'Prohibited Zone  |  {FLAGS.env_name}  |  threshold = {FLAGS.threshold}',
        fontsize=14)

    # ── Left: distance heatmap ────────────────────────────────────────────────
    ax = axes[0]
    ax.set_title('Nearest-neighbour distance to dataset')

    if mi is not None:
        ax.imshow(1 - mi['maze_map'], cmap='gray',
                  extent=(mi['xmin'], mi['xmax'], mi['ymin'], mi['ymax']),
                  origin='lower', alpha=0.25)

    dist_clipped = np.clip(dist_map, 0, 2 * FLAGS.threshold)
    im = ax.pcolormesh(X, Y, dist_clipped,
                       cmap='plasma', shading='auto', alpha=0.85)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f'dist to nearest data point  (capped at {2*FLAGS.threshold})')

    # Contour line at the threshold
    ax.contour(X, Y, dist_map, levels=[FLAGS.threshold],
               colors='cyan', linewidths=1.5, linestyles='--')

    # Bounding rectangle
    ax.add_patch(mpatches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        linewidth=2, edgecolor='white', facecolor='none',
        linestyle='--', label='bounding box'))

    ax.set(xlabel='X', ylabel='Y')
    ax.legend(fontsize=9)

    # ── Right: prohibited zone overlay + point cloud ──────────────────────────
    ax = axes[1]
    ax.set_title(f'Prohibited zone  (dist ≥ {FLAGS.threshold})')

    if mi is not None:
        ax.imshow(1 - mi['maze_map'], cmap='gray',
                  extent=(mi['xmin'], mi['xmax'], mi['ymin'], mi['ymax']),
                  origin='lower', alpha=0.25)

    # Prohibited zone: filled red
    proh_rgba = np.zeros((*X.shape, 4), dtype=np.float32)
    proh_rgba[prohibited, 0] = 0.9   # R
    proh_rgba[prohibited, 1] = 0.1   # G
    proh_rgba[prohibited, 2] = 0.1   # B
    proh_rgba[prohibited, 3] = 0.55  # alpha
    ax.pcolormesh(X, Y, np.zeros_like(X),    # dummy values
                  shading='auto', alpha=0.0)  # invisible base layer
    ax.imshow(proh_rgba,
              extent=(x_min, x_max, y_min, y_max),
              origin='lower', aspect='auto', zorder=2)

    # Point cloud (down-sampled)
    step = max(1, FLAGS.pc_subsample)
    ax.scatter(xy[::step, 0], xy[::step, 1],
               s=0.3, c='dodgerblue', alpha=0.15,
               linewidths=0, zorder=3, label=f'dataset (1/{step})')

    # Bounding rectangle
    ax.add_patch(mpatches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        linewidth=2, edgecolor='yellow', facecolor='none',
        linestyle='--', zorder=4, label='bounding box'))

    # Threshold contour
    ax.contour(X, Y, dist_map, levels=[FLAGS.threshold],
               colors='cyan', linewidths=1.5, linestyles='--', zorder=5)
    ax.plot([], [], color='cyan', linewidth=1.5, linestyle='--',
            label=f'threshold = {FLAGS.threshold}')

    ax.set(xlabel='X', ylabel='Y')
    ax.legend(fontsize=9, markerscale=8)

    # ── Save ──────────────────────────────────────────────────────────────────
    fig.tight_layout()
    fname = (f'prohibited_zone_{FLAGS.env_name}'
             f'_thr{str(FLAGS.threshold).replace(".", "_")}'
             f'_res{FLAGS.grid_res}.png')
    out_path = os.path.join(FLAGS.save_dir, fname)
    fig.savefig(out_path, bbox_inches='tight', dpi=180)
    plt.close(fig)
    print(f'  Saved → {out_path}')


if __name__ == '__main__':
    app.run(main)
