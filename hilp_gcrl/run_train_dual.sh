#!/bin/bash
# ============================================================
# Phase 1: Train Dual Goal Representations (arXiv:2510.06714)
# V(s,g) = psi(s)^T phi(g) on OGBench environments.
# No interactive input required.
# ============================================================

set -e

WORKSPACE_ROOT="/mnt/c/Users/USER/Desktop/test_ogbench"
DOCKER_IMAGE="mctd:0.1"
OGBENCH_DATA_DIR="${WORKSPACE_ROOT}/ogbench_data"

# ---- Parameters (edit here to change runs) ----------------------------------
ENV_NAME="antmaze-giant-navigate-v0"
SKILL_DIM=32
TRAIN_STEPS=500000
BATCH_SIZE=4096
LR=3e-4
DISCOUNT=0.99
EXPECTILE=0.95
SAVE_INTERVAL=50000
SAVE_DIR="/workspace/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}"

PYTHON_SCRIPT="/workspace/HILP/hilp_gcrl/train_dual_ogbench.py"

echo "============================================"
echo "  Dual Goal Repr Training (Phase 1)"
echo "  env        : ${ENV_NAME}"
echo "  skill_dim  : ${SKILL_DIM}"
echo "  train_steps: ${TRAIN_STEPS}"
echo "  save_dir   : ${SAVE_DIR}"
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
            --env_name=${ENV_NAME} \
            --skill_dim=${SKILL_DIM} \
            --train_steps=${TRAIN_STEPS} \
            --batch_size=${BATCH_SIZE} \
            --lr=${LR} \
            --discount=${DISCOUNT} \
            --expectile=${EXPECTILE} \
            --save_interval=${SAVE_INTERVAL} \
            --save_dir=${SAVE_DIR}
    "

echo ""
echo "Phase 1 training complete."
echo "Checkpoint saved to: ${WORKSPACE_ROOT}/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}/"
