#!/bin/bash
# ============================================================
# Visualize Downstream GCVF Heatmap
# Mode: gcvf  →  V_down(s, phi(g))
# Shows the goal-conditioned value function trained on top of
# frozen dual representations (Phase 2 of arXiv:2510.06714).
# No interactive input required.
# ============================================================

set -e

WORKSPACE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DOCKER_IMAGE="mctd:0.1"
OGBENCH_DATA_DIR="${WORKSPACE_ROOT}/ogbench_data"
UNAME="junjolp2026spring"
DEVICE='"device=0,1"'

# ---- Parameters -------------------------------------------------------------
ENV_NAME="antmaze-giant-navigate-v0"
SKILL_DIM=32

# Phase 2 (GCVF) checkpoint
RESTORE_PATH="/workspace/HILP/hilp_gcrl/exp/gcvf_dual/${ENV_NAME}"
RESTORE_EPOCH=500000

# Phase 1 (Dual repr) checkpoint — needed to compute phi(g)
DUAL_RESTORE_PATH="/workspace/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}"
DUAL_RESTORE_EPOCH=1000000

# Goal position (x, y)
GOAL_X=12.0
GOAL_Y=8.0

GRID_RES=100
SAVE_DIR="/workspace/HILP/hilp_gcrl/visualizations"
PYTHON_SCRIPT="/workspace/HILP/hilp_gcrl/visualize_dual_heatmap.py"

echo "============================================"
echo "  Downstream GCVF Heatmap (mode: gcvf)"
echo "  env               : ${ENV_NAME}"
echo "  gcvf checkpoint   : ${RESTORE_PATH} @ step ${RESTORE_EPOCH}"
echo "  dual checkpoint   : ${DUAL_RESTORE_PATH} @ step ${DUAL_RESTORE_EPOCH}"
echo "  goal              : (${GOAL_X}, ${GOAL_Y})"
echo "  output dir        : ${SAVE_DIR}"
echo "============================================"

docker run --gpus "${DEVICE}" --rm \
    -v "${WORKSPACE_ROOT}:/workspace" \
    -v "${OGBENCH_DATA_DIR}:/home/${UNAME}/.ogbench/data" \
    -w /workspace/HILP/hilp_gcrl \
    -e MUJOCO_GL=egl \
    -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "${DOCKER_IMAGE}" bash -c "
        pip3 install --quiet pyrallis shapely scikit-learn ogbench distrax &&
        python3 ${PYTHON_SCRIPT} \
            --mode=gcvf \
            --env_name=${ENV_NAME} \
            --restore_path=${RESTORE_PATH} \
            --restore_epoch=${RESTORE_EPOCH} \
            --dual_restore_path=${DUAL_RESTORE_PATH} \
            --dual_restore_epoch=${DUAL_RESTORE_EPOCH} \
            --skill_dim=${SKILL_DIM} \
            --goal_pos=${GOAL_X},${GOAL_Y} \
            --grid_res=${GRID_RES} \
            --save_dir=${SAVE_DIR}
    "

echo ""
echo "Heatmap saved to: ${WORKSPACE_ROOT}/HILP/hilp_gcrl/visualizations/"
