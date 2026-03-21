#!/bin/bash
# ============================================================
# Phase 1: Train Dual Goal Representations (arXiv:2510.06714)
# Supports two value aggregators:
#   inner_prod : V(s,g) = psi(s)^T phi(g)
#   neg_l2     : V(s,g) = -||psi(s) - phi(g)||
# Checkpoints are stored in separate subdirectories per aggregator.
# Interactively prompts to select aggregator, then resume or start fresh.
# ============================================================

set -e

WORKSPACE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DOCKER_IMAGE="mctd:0.1"
OGBENCH_DATA_DIR="${WORKSPACE_ROOT}/ogbench_data"
UNAME="junjolp2026spring"
DEVICE='"device=0,1"'


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
VIZ_INTERVAL=20000    # t-SNE logged to WandB every N steps (0 = disabled)

PYTHON_SCRIPT="/workspace/HILP/hilp_gcrl/train_dual_ogbench.py"

# ---- [Step 1] Select aggregator ---------------------------------------------
echo ""
echo "============================================"
echo "  Select value aggregator:"
echo "  [1] inner_prod  — V = psi(s)^T phi(g)"
echo "                    (goal-directed gradient via phi(g) direction)"
echo "  [2] neg_l2      — V = -||psi(s) - phi(g)||"
echo "                    (gradient direction is distance-invariant unit vector)"
echo "============================================"
read -rp "Your choice [1/2]: " AGG_CHOICE

case "${AGG_CHOICE}" in
    1) AGGREGATOR="inner_prod" ;;
    2) AGGREGATOR="neg_l2" ;;
    *)
        echo "Invalid choice. Aborting."
        exit 1
        ;;
esac
echo "→ Aggregator: ${AGGREGATOR}"

# ---- Paths derived from aggregator selection --------------------------------
SAVE_DIR="/workspace/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}/${AGGREGATOR}"
HOST_SAVE_DIR="${WORKSPACE_ROOT}/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}/${AGGREGATOR}"

# ---- WandB ------------------------------------------------------------------
WANDB_PROJECT="${WANDB_PROJECT:-hilp_gcrl}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-dual_repr_${AGGREGATOR}_${ENV_NAME}}"

# ---- [Step 2] Checkpoint detection ------------------------------------------
RESUME_STEP=0

if [ -d "${HOST_SAVE_DIR}" ]; then
    mapfile -t CKPT_FILES < <(ls "${HOST_SAVE_DIR}"/params_*.pkl 2>/dev/null | sort -t_ -k2 -n)

    if [ ${#CKPT_FILES[@]} -gt 0 ]; then
        echo ""
        echo "Existing checkpoints detected (aggregator=${AGGREGATOR}):"
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
echo "  aggregator : ${AGGREGATOR}"
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
            --aggregator=${AGGREGATOR} \
            --resume_step=${RESUME_STEP} \
            --save_dir=${SAVE_DIR} \
            --viz_interval=${VIZ_INTERVAL} \
            --wandb_project=${WANDB_PROJECT} \
            --wandb_run_name=${WANDB_RUN_NAME}
    "

echo ""
echo "Phase 1 training complete."
echo "Checkpoint saved to: ${HOST_SAVE_DIR}/"
