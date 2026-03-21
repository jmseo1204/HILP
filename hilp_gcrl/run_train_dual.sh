#!/bin/bash
# ============================================================
# Phase 1: Train Dual Goal Representations (arXiv:2510.06714)
# V(s,g) = psi(s)^T phi(g) on OGBench environments.
# Interactively prompts to resume from an existing checkpoint or start fresh.
# ============================================================

set -e

WORKSPACE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DOCKER_IMAGE="mctd:0.1"
OGBENCH_DATA_DIR="${WORKSPACE_ROOT}/ogbench_data"
UNAME="jmseo1204"
DEVICE='"device=0"'

# ---- Parameters (edit here to change runs) ----------------------------------
ENV_NAME="antmaze-giant-navigate-v0"
SKILL_DIM=256
TRAIN_STEPS=1000000
BATCH_SIZE=2048
LR=3e-4
DISCOUNT=0.995
EXPECTILE=0.9
P_CURRGOAL=0.2
P_TRAJGOAL=0.5
P_RANDOMGOAL=0.3
SAVE_INTERVAL=50000
SAVE_DIR="/workspace/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}"

# ---- WandB (set WANDB_API_KEY in your shell, or leave WANDB_PROJECT empty to disable) --
WANDB_PROJECT="${WANDB_PROJECT:-hilp_gcrl}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-dual_repr_${ENV_NAME}}"

PYTHON_SCRIPT="/workspace/HILP/hilp_gcrl/train_dual_ogbench.py"

# ---- Checkpoint detection (host-side path) ----------------------------------
HOST_SAVE_DIR="${WORKSPACE_ROOT}/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}"
RESUME_STEP=0

if [ -d "${HOST_SAVE_DIR}" ]; then
    mapfile -t CKPT_FILES < <(ls "${HOST_SAVE_DIR}"/params_*.pkl 2>/dev/null | sort -t_ -k2 -n)

    if [ ${#CKPT_FILES[@]} -gt 0 ]; then
        echo ""
        echo "Existing Phase 1 checkpoints detected:"
        for i in "${!CKPT_FILES[@]}"; do
            STEP=$(basename "${CKPT_FILES[$i]}" .pkl | sed 's/params_//')
            echo "  [$((i+1))] step ${STEP}"
        done
        echo "  [f] Start from scratch (existing checkpoints will be deleted)"
        echo ""
        read -rp "Your choice: " CHOICE

        if [[ "$CHOICE" =~ ^[0-9]+$ ]] && [ "$CHOICE" -ge 1 ] && [ "$CHOICE" -le "${#CKPT_FILES[@]}" ]; then
            IDX=$((CHOICE-1))
            RESUME_STEP=$(basename "${CKPT_FILES[$IDX]}" .pkl | sed 's/params_//')
            echo "→ Resuming from step ${RESUME_STEP}"
        elif [[ "${CHOICE,,}" == "f" ]]; then
            echo "→ Starting from scratch. Deleting existing checkpoints..."
            rm -f "${HOST_SAVE_DIR}"/params_*.pkl
            RESUME_STEP=0
        else
            echo "Invalid choice. Aborting."
            exit 1
        fi
    fi
fi

echo ""
echo "============================================"
echo "  Dual Goal Repr Training (Phase 1)"
echo "  env        : ${ENV_NAME}"
echo "  skill_dim  : ${SKILL_DIM}"
echo "  train_steps: ${TRAIN_STEPS}"
echo "  save_dir   : ${SAVE_DIR}"
if [ "${RESUME_STEP}" -gt 0 ]; then
echo "  resume_step: ${RESUME_STEP}"
fi
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
            --resume_step=${RESUME_STEP} \
            --save_dir=${SAVE_DIR} \
            --wandb_project=${WANDB_PROJECT} \
            --wandb_run_name=${WANDB_RUN_NAME}
    "

echo ""
echo "Phase 1 training complete."
echo "Checkpoint saved to: ${WORKSPACE_ROOT}/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}/"
