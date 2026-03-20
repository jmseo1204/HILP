#!/bin/bash
# ============================================================
# Diagnostic version of Phase 1 training.
# Logs detailed diagnostics to JSONL for divergence analysis.
# ============================================================

set -e

WORKSPACE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DOCKER_IMAGE="mctd:0.1"
OGBENCH_DATA_DIR="${WORKSPACE_ROOT}/ogbench_data"
UNAME="jmseo1204"
DEVICE='"device=0"'

# ---- Parameters (same as run_train_dual.sh) ----------------------------------
ENV_NAME="antmaze-giant-navigate-v0"
SKILL_DIM=256
TRAIN_STEPS=60000          # Shortened: divergence happens well before 60k
BATCH_SIZE=2048
LR=3e-4
DISCOUNT=0.995
EXPECTILE=0.9
P_CURRGOAL=0.2
P_TRAJGOAL=0.5
P_RANDOMGOAL=0.3
SAVE_INTERVAL=50000
DIAG_INTERVAL=500          # Log diagnostics every 500 steps
SAVE_DIR="/workspace/HILP/hilp_gcrl/exp/dual_repr_diag/${ENV_NAME}"

WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-dual_repr_diag_${ENV_NAME}}"

PYTHON_SCRIPT="/workspace/HILP/hilp_gcrl/train_dual_ogbench_diag.py"

echo "============================================"
echo "  Dual Goal Repr DIAGNOSTIC Training"
echo "  env        : ${ENV_NAME}"
echo "  skill_dim  : ${SKILL_DIM}"
echo "  train_steps: ${TRAIN_STEPS}"
echo "  diag_intv  : ${DIAG_INTERVAL}"
echo "  save_dir   : ${SAVE_DIR}"
echo "============================================"

docker run --gpus "${DEVICE}" --rm \
    -v "${WORKSPACE_ROOT}:/workspace" \
    -v "${OGBENCH_DATA_DIR}:/home/${UNAME}/.ogbench/data" \
    -w /workspace/HILP/hilp_gcrl \
    -e MUJOCO_GL=egl \
    -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
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
            --p_currgoal=${P_CURRGOAL} \
            --p_trajgoal=${P_TRAJGOAL} \
            --p_randomgoal=${P_RANDOMGOAL} \
            --save_interval=${SAVE_INTERVAL} \
            --diag_interval=${DIAG_INTERVAL} \
            --save_dir=${SAVE_DIR} \
            --wandb_project=${WANDB_PROJECT} \
            --wandb_run_name=${WANDB_RUN_NAME}
    "

echo ""
echo "Diagnostic training complete."
echo "JSONL log: ${WORKSPACE_ROOT}/HILP/hilp_gcrl/exp/dual_repr_diag/${ENV_NAME}/diagnostics.jsonl"
echo ""
echo "Run analysis:"
echo "  python3 hilp_gcrl/analyze_diagnostics.py hilp_gcrl/exp/dual_repr_diag/${ENV_NAME}/diagnostics.jsonl"
