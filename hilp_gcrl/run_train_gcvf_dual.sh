#!/bin/bash
# ============================================================
# Phase 2: Train Downstream GCVF on Frozen Dual Representations
# V_down(s, phi(g)) = MLP([s, phi(g)]) with IQL loss.
# Run AFTER run_train_dual.sh completes.
# ============================================================

set -e

WORKSPACE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DOCKER_IMAGE="mctd:0.1"
OGBENCH_DATA_DIR="${WORKSPACE_ROOT}/ogbench_data"
UNAME="junjolp2026spring"
DEVICE='"device=0,1"'

# ---- Parameters (must match Phase 1 settings) -------------------------------
ENV_NAME="antmaze-giant-navigate-v0"
SKILL_DIM=256
TRAIN_STEPS=500000
BATCH_SIZE=2048
LR=3e-4
DISCOUNT=0.995
EXPECTILE=0.9
P_CURRGOAL=0.2
P_TRAJGOAL=0.5
P_RANDOMGOAL=0.3
SAVE_INTERVAL=50000

# Phase 1 checkpoint to load (frozen)
DUAL_RESTORE_PATH="/workspace/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}"
DUAL_RESTORE_EPOCH=1000000

# Phase 2 output
SAVE_DIR="/workspace/HILP/hilp_gcrl/exp/gcvf_dual/${ENV_NAME}"

# ---- WandB (set WANDB_API_KEY in your shell, or leave WANDB_PROJECT empty to disable) --
WANDB_PROJECT="${WANDB_PROJECT:-hilp_gcrl}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-gcvf_dual_${ENV_NAME}}"

PYTHON_SCRIPT="/workspace/HILP/hilp_gcrl/train_gcvf_dual_ogbench.py"

# ---- Phase 1 checkpoint selection -------------------------------------------
HOST_DUAL_DIR="${WORKSPACE_ROOT}/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}"

if [ ! -d "${HOST_DUAL_DIR}" ]; then
    echo "ERROR: Phase 1 checkpoint directory not found: ${HOST_DUAL_DIR}"
    echo "       Run run_train_dual.sh first."
    exit 1
fi

mapfile -t P1_CKPTS < <(ls "${HOST_DUAL_DIR}"/params_*.pkl 2>/dev/null | sort -t_ -k2 -n)

if [ ${#P1_CKPTS[@]} -eq 0 ]; then
    echo "ERROR: No Phase 1 checkpoints found in ${HOST_DUAL_DIR}"
    echo "       Run run_train_dual.sh first."
    exit 1
fi

echo ""
echo "Phase 1 checkpoints available:"
for i in "${!P1_CKPTS[@]}"; do
    STEP=$(basename "${P1_CKPTS[$i]}" .pkl | sed 's/params_//')
    echo "  [$((i+1))] step ${STEP}"
done

# Default to the latest (last) available checkpoint, not the hardcoded value
DUAL_RESTORE_EPOCH=$(basename "${P1_CKPTS[-1]}" .pkl | sed 's/params_//')
echo ""
read -rp "Select Phase 1 checkpoint to load [default: ${DUAL_RESTORE_EPOCH}]: " P1_CHOICE

if [[ "$P1_CHOICE" =~ ^[0-9]+$ ]] && [ "$P1_CHOICE" -ge 1 ] && [ "$P1_CHOICE" -le "${#P1_CKPTS[@]}" ]; then
    IDX=$((P1_CHOICE-1))
    DUAL_RESTORE_EPOCH=$(basename "${P1_CKPTS[$IDX]}" .pkl | sed 's/params_//')
    echo "→ Using Phase 1 checkpoint at step ${DUAL_RESTORE_EPOCH}"
elif [ -z "$P1_CHOICE" ]; then
    echo "→ Using Phase 1 checkpoint at step ${DUAL_RESTORE_EPOCH} (latest)"
else
    echo "Invalid choice. Aborting."
    exit 1
fi

# Verify the selected checkpoint file actually exists
if [ ! -f "${HOST_DUAL_DIR}/params_${DUAL_RESTORE_EPOCH}.pkl" ]; then
    echo "ERROR: Checkpoint file not found: ${HOST_DUAL_DIR}/params_${DUAL_RESTORE_EPOCH}.pkl"
    exit 1
fi

# ---- Phase 2 checkpoint detection -------------------------------------------
HOST_SAVE_DIR="${WORKSPACE_ROOT}/HILP/hilp_gcrl/exp/gcvf_dual/${ENV_NAME}"
RESUME_STEP=0

if [ -d "${HOST_SAVE_DIR}" ]; then
    mapfile -t P2_CKPTS < <(ls "${HOST_SAVE_DIR}"/params_*.pkl 2>/dev/null | sort -t_ -k2 -n)

    if [ ${#P2_CKPTS[@]} -gt 0 ]; then
        echo ""
        echo "Existing Phase 2 checkpoints detected:"
        for i in "${!P2_CKPTS[@]}"; do
            STEP=$(basename "${P2_CKPTS[$i]}" .pkl | sed 's/params_//')
            echo "  [$((i+1))] step ${STEP}"
        done
        echo "  [f] Start from scratch (existing checkpoints will be deleted)"
        echo ""
        read -rp "Your choice: " P2_CHOICE

        if [[ "$P2_CHOICE" =~ ^[0-9]+$ ]] && [ "$P2_CHOICE" -ge 1 ] && [ "$P2_CHOICE" -le "${#P2_CKPTS[@]}" ]; then
            IDX=$((P2_CHOICE-1))
            RESUME_STEP=$(basename "${P2_CKPTS[$IDX]}" .pkl | sed 's/params_//')
            echo "→ Resuming Phase 2 from step ${RESUME_STEP}"
        elif [[ "${P2_CHOICE,,}" == "f" ]]; then
            echo "→ Starting Phase 2 from scratch. Deleting existing checkpoints..."
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
echo "  Downstream GCVF Training (Phase 2)"
echo "  env             : ${ENV_NAME}"
echo "  dual checkpoint : ${DUAL_RESTORE_PATH} @ step ${DUAL_RESTORE_EPOCH}"
echo "  train_steps     : ${TRAIN_STEPS}"
echo "  save_dir        : ${SAVE_DIR}"
if [ "${RESUME_STEP}" -gt 0 ]; then
echo "  resume_step     : ${RESUME_STEP}"
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
            --dual_restore_path=${DUAL_RESTORE_PATH} \
            --dual_restore_epoch=${DUAL_RESTORE_EPOCH} \
            --train_steps=${TRAIN_STEPS} \
            --batch_size=${BATCH_SIZE} \
            --lr=${LR} \
            --discount=${DISCOUNT} \
            --expectile=${EXPECTILE} \
            --save_interval=${SAVE_INTERVAL} \
            --p_currgoal=${P_CURRGOAL} \
            --p_trajgoal=${P_TRAJGOAL} \
            --p_randomgoal=${P_RANDOMGOAL} \
            --resume_step=${RESUME_STEP} \
            --save_dir=${SAVE_DIR} \
            --wandb_project=${WANDB_PROJECT} \
            --wandb_run_name=${WANDB_RUN_NAME}
    "

echo ""
echo "Phase 2 training complete."
echo "Checkpoint saved to: ${WORKSPACE_ROOT}/HILP/hilp_gcrl/exp/gcvf_dual/${ENV_NAME}/"
