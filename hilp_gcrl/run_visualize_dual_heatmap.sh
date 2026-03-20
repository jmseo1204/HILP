#!/bin/bash
# ============================================================
# Visualize Dual Goal Representation Heatmap
# Mode: dual_repr  →  V(s,g) = psi(s)^T phi(g)
# Shows temporal distance from goal using the Phase-1 model.
# No interactive input required.
# ============================================================

set -e

WORKSPACE_ROOT="/mnt/c/Users/USER/Desktop/test_ogbench"
DOCKER_IMAGE="mctd:0.1"
OGBENCH_DATA_DIR="${WORKSPACE_ROOT}/ogbench_data"

# ---- Parameters -------------------------------------------------------------
ENV_NAME="pointmaze-large-stitch-v0"
SKILL_DIM=32

# Phase 1 checkpoint
RESTORE_PATH="/workspace/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}"
RESTORE_EPOCH=1000000

# Goal position (x, y) — adjust for your environment
GOAL_X=12.0
GOAL_Y=8.0

GRID_RES=100
SAVE_DIR="/workspace/HILP/hilp_gcrl/visualizations"
PYTHON_SCRIPT="/workspace/HILP/hilp_gcrl/visualize_dual_heatmap.py"

echo "============================================"
echo "  Dual Repr Heatmap (mode: dual_repr)"
echo "  env          : ${ENV_NAME}"
echo "  checkpoint   : ${RESTORE_PATH} @ step ${RESTORE_EPOCH}"
echo "  goal         : (${GOAL_X}, ${GOAL_Y})"
echo "  output dir   : ${SAVE_DIR}"
echo "============================================"

docker run --gpus all --rm \
    -v "${WORKSPACE_ROOT}:/workspace" \
    -v "${OGBENCH_DATA_DIR}:/home/jmseo1204/.ogbench/data" \
    -w /workspace/HILP/hilp_gcrl \
    -e MUJOCO_GL=egl \
    -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "${DOCKER_IMAGE}" bash -c "
        pip3 install --quiet pyrallis shapely scikit-learn ogbench distrax &&
        python3 ${PYTHON_SCRIPT} \
            --mode=dual_repr \
            --env_name=${ENV_NAME} \
            --restore_path=${RESTORE_PATH} \
            --restore_epoch=${RESTORE_EPOCH} \
            --skill_dim=${SKILL_DIM} \
            --goal_pos=${GOAL_X},${GOAL_Y} \
            --grid_res=${GRID_RES} \
            --save_dir=${SAVE_DIR}
    "

echo ""
echo "Heatmap saved to: ${WORKSPACE_ROOT}/HILP/hilp_gcrl/visualizations/"
