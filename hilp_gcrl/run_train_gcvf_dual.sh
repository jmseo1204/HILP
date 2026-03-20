#!/bin/bash
# ============================================================
# Phase 2: Train Downstream GCVF on Frozen Dual Representations
# V_down(s, phi(g)) = MLP([s, phi(g)]) with IQL loss.
# No interactive input required.
# Run AFTER run_train_dual.sh completes.
# ============================================================

set -e

WORKSPACE_ROOT="/mnt/c/Users/USER/Desktop/test_ogbench"
DOCKER_IMAGE="mctd:0.1"
OGBENCH_DATA_DIR="${WORKSPACE_ROOT}/ogbench_data"

# ---- Parameters (must match Phase 1 settings) -------------------------------
ENV_NAME="antmaze-giant-navigate-v0"
SKILL_DIM=32
TRAIN_STEPS=500000
BATCH_SIZE=4096
LR=3e-4
DISCOUNT=0.99
EXPECTILE=0.95
SAVE_INTERVAL=50000

# Phase 1 checkpoint to load (frozen)
DUAL_RESTORE_PATH="/workspace/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}"
DUAL_RESTORE_EPOCH=1000000

# Phase 2 output
SAVE_DIR="/workspace/HILP/hilp_gcrl/exp/gcvf_dual/${ENV_NAME}"

PYTHON_SCRIPT="/workspace/HILP/hilp_gcrl/train_gcvf_dual_ogbench.py"

echo "============================================"
echo "  Downstream GCVF Training (Phase 2)"
echo "  env             : ${ENV_NAME}"
echo "  dual checkpoint : ${DUAL_RESTORE_PATH} @ step ${DUAL_RESTORE_EPOCH}"
echo "  train_steps     : ${TRAIN_STEPS}"
echo "  save_dir        : ${SAVE_DIR}"
echo "============================================"

docker run --gpus all --rm \
    -v "${WORKSPACE_ROOT}:/workspace" \
    -v "${OGBENCH_DATA_DIR}:/home/junjolp2026spring/.ogbench/data" \
    -w /workspace/HILP/hilp_gcrl \
    -e MUJOCO_GL=egl \
    -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "${DOCKER_IMAGE}" bash -c "
        pip3 install --quiet pyrallis shapely scikit-learn ogbench distrax &&
        python3 ${PYTHON_SCRIPT} \
            --env_name=${ENV_NAME} \
            --skill_dim=${SKILL_DIM} \
            --dual_restore_path=${DUAL_RESTORE_PATH} \
            --dual_restore_epoch=${DUAL_RESTORE_EPOCH} \
            --train_steps=${TRAIN_STEPS} \
            --batch_size=${BATCH_SIZE} \
            --lr=${LR} \
            --discount=${DISCOUNT} \
            --expectile=${EXPECTILE} \
            --save_interval=${SAVE_INTERVAL} \
            --save_dir=${SAVE_DIR}
    "

echo ""
echo "Phase 2 training complete."
echo "Checkpoint saved to: ${WORKSPACE_ROOT}/HILP/hilp_gcrl/exp/gcvf_dual/${ENV_NAME}/"
