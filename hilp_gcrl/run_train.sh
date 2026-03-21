#!/bin/bash
# ============================================================
# Unified Training Launcher
#   [1] Phase 1: Dual Goal Representations (train_dual_ogbench.py)
#   [2] Phase 2: Downstream GCVF on Frozen Dual Repr (train_gcvf_dual_ogbench.py)
# ============================================================

set -e

WORKSPACE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DOCKER_IMAGE="mctd:0.1"
OGBENCH_DATA_DIR="${WORKSPACE_ROOT}/ogbench_data"
UNAME="$(whoami)"
DEVICE='"device=0,1"'

# ---- Parameters (edit here to change runs) ----------------------------------
ENV_NAME="antmaze-giant-navigate-v0"
SKILL_DIM=256
BATCH_SIZE=2048
LR=3e-4
DISCOUNT=0.995
EXPECTILE=0.9
P_CURRGOAL=0.2
P_TRAJGOAL=0.5
P_RANDOMGOAL=0.3
SAVE_INTERVAL=50000

# Phase 1 specific
TRAIN_STEPS_P1=1000000
VIZ_INTERVAL=20000
P_PROHIBIT=0.1
LAMBDA_NEG=0.1
PROHIBIT_THRESHOLD=1.5

# Phase 2 specific
TRAIN_STEPS_P2=500000

# ---- [Step 0] Select training phase -----------------------------------------
echo ""
echo "============================================"
echo "  Select training phase:"
echo "  [1] Phase 1 — Dual Goal Representations"
echo "                (train_dual_ogbench.py)"
echo "  [2] Phase 2 — Downstream GCVF"
echo "                (train_gcvf_dual_ogbench.py)"
echo "============================================"
read -rp "Your choice [1/2]: " PHASE_CHOICE

case "${PHASE_CHOICE}" in
    1) ;;
    2) ;;
    *)
        echo "Invalid choice. Aborting."
        exit 1
        ;;
esac

# =============================================================================
# PHASE 1
# =============================================================================
if [ "${PHASE_CHOICE}" == "1" ]; then

    PYTHON_SCRIPT="/workspace/HILP/hilp_gcrl/train_dual_ogbench.py"

    # ---- [Step 1] Select aggregator -----------------------------------------
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

    SAVE_DIR="/workspace/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}/${AGGREGATOR}"
    HOST_SAVE_DIR="${WORKSPACE_ROOT}/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}/${AGGREGATOR}"

    WANDB_PROJECT="${WANDB_PROJECT:-hilp_gcrl}"
    WANDB_RUN_NAME="${WANDB_RUN_NAME:-dual_repr_${AGGREGATOR}_${ENV_NAME}}"

    # ---- [Step 2] Checkpoint detection --------------------------------------
    RESUME_STEP=0

    if [ -d "${HOST_SAVE_DIR}" ]; then
        mapfile -t CKPT_FILES < <(ls "${HOST_SAVE_DIR}"/params_*.pkl 2>/dev/null | sort -t_ -k2 -n)

        if [ ${#CKPT_FILES[@]} -gt 0 ]; then
            echo ""
            echo "Existing checkpoints detected (aggregator=${AGGREGATOR}):"
            for i in "${!CKPT_FILES[@]}"; do
                BASE=$(basename "${CKPT_FILES[$i]}" .pkl)
                STEP=$(echo "${BASE}" | cut -d_ -f2)
                TS=$(echo "${BASE}" | cut -d_ -f3-)
                echo "  [$((i+1))] step ${STEP}  (${TS})"
            done
            echo "  [f] Start from scratch (existing checkpoints will be kept)"
            echo ""
            read -rp "Your choice: " CHOICE

            if [[ "$CHOICE" =~ ^[0-9]+$ ]] && [ "$CHOICE" -ge 1 ] && [ "$CHOICE" -le "${#CKPT_FILES[@]}" ]; then
                IDX=$((CHOICE-1))
                RESUME_STEP=$(basename "${CKPT_FILES[$IDX]}" .pkl | cut -d_ -f2)
                echo "→ Resuming from step ${RESUME_STEP}"
            elif [[ "${CHOICE,,}" == "f" ]]; then
                echo "→ Starting from scratch (existing checkpoints preserved)."
                RESUME_STEP=0
            else
                echo "Invalid choice. Aborting."
                exit 1
            fi
        fi
    fi

    echo ""
    echo "============================================"
    echo "  Phase 1: Dual Goal Repr Training"
    echo "  env        : ${ENV_NAME}"
    echo "  aggregator : ${AGGREGATOR}"
    echo "  skill_dim  : ${SKILL_DIM}"
    echo "  train_steps: ${TRAIN_STEPS_P1}"
    echo "  save_dir   : ${SAVE_DIR}"
    [ "${RESUME_STEP}" -gt 0 ] && echo "  resume_step: ${RESUME_STEP}"
    echo "============================================"

    docker run --gpus "${DEVICE}" --rm \
        -v "${WORKSPACE_ROOT}:/workspace" \
        -v "${OGBENCH_DATA_DIR}:/home/${UNAME}/.ogbench/data" \
        -w /workspace/HILP/hilp_gcrl \
        -e MUJOCO_GL=egl \
        -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
        -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
        "${DOCKER_IMAGE}" bash -c "
            pip3 install --quiet pyrallis shapely scikit-learn ogbench distrax scipy &&
            python3 ${PYTHON_SCRIPT} \
                --env_name=${ENV_NAME} \
                --skill_dim=${SKILL_DIM} \
                --train_steps=${TRAIN_STEPS_P1} \
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
                --p_prohibit=${P_PROHIBIT} \
                --lambda_neg=${LAMBDA_NEG} \
                --prohibit_threshold=${PROHIBIT_THRESHOLD} \
                --wandb_project=${WANDB_PROJECT} \
                --wandb_run_name=${WANDB_RUN_NAME}
        "

    echo ""
    echo "Phase 1 training complete."
    echo "Checkpoint saved to: ${HOST_SAVE_DIR}/"
fi

# =============================================================================
# PHASE 2
# =============================================================================
if [ "${PHASE_CHOICE}" == "2" ]; then

    PYTHON_SCRIPT="/workspace/HILP/hilp_gcrl/train_gcvf_dual_ogbench.py"
    DUAL_RESTORE_PATH="/workspace/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}"
    HOST_DUAL_DIR="${WORKSPACE_ROOT}/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}"

    SAVE_DIR="/workspace/HILP/hilp_gcrl/exp/gcvf_dual/${ENV_NAME}"
    HOST_SAVE_DIR="${WORKSPACE_ROOT}/HILP/hilp_gcrl/exp/gcvf_dual/${ENV_NAME}"

    WANDB_PROJECT="${WANDB_PROJECT:-hilp_gcrl}"
    WANDB_RUN_NAME="${WANDB_RUN_NAME:-gcvf_dual_${ENV_NAME}}"

    # ---- Phase 1 checkpoint selection ---------------------------------------
    if [ ! -d "${HOST_DUAL_DIR}" ]; then
        echo "ERROR: Phase 1 checkpoint directory not found: ${HOST_DUAL_DIR}"
        echo "       Run Phase 1 training first."
        exit 1
    fi

    mapfile -t P1_CKPTS < <(ls "${HOST_DUAL_DIR}"/*/params_*.pkl 2>/dev/null | sort -t_ -k2 -n)

    if [ ${#P1_CKPTS[@]} -eq 0 ]; then
        echo "ERROR: No Phase 1 checkpoints found in ${HOST_DUAL_DIR}"
        echo "       Run Phase 1 training first."
        exit 1
    fi

    echo ""
    echo "Phase 1 checkpoints available:"
    for i in "${!P1_CKPTS[@]}"; do
        BASE=$(basename "${P1_CKPTS[$i]}" .pkl)
        STEP=$(echo "${BASE}" | cut -d_ -f2)
        TS=$(echo "${BASE}" | cut -d_ -f3-)
        AGGR=$(basename "$(dirname "${P1_CKPTS[$i]}")")
        echo "  [$((i+1))] step ${STEP}  (${TS})  [aggregator=${AGGR}]"
    done

    DUAL_RESTORE_EPOCH=$(basename "${P1_CKPTS[-1]}" .pkl | cut -d_ -f2)
    DUAL_RESTORE_PATH_FULL="$(dirname "${P1_CKPTS[-1]}")"
    echo ""
    read -rp "Select Phase 1 checkpoint to load [default: ${DUAL_RESTORE_EPOCH}]: " P1_CHOICE

    if [[ "$P1_CHOICE" =~ ^[0-9]+$ ]] && [ "$P1_CHOICE" -ge 1 ] && [ "$P1_CHOICE" -le "${#P1_CKPTS[@]}" ]; then
        IDX=$((P1_CHOICE-1))
        DUAL_RESTORE_EPOCH=$(basename "${P1_CKPTS[$IDX]}" .pkl | cut -d_ -f2)
        DUAL_RESTORE_PATH_FULL="$(dirname "${P1_CKPTS[$IDX]}")"
        echo "→ Using Phase 1 checkpoint at step ${DUAL_RESTORE_EPOCH}"
    elif [ -z "$P1_CHOICE" ]; then
        echo "→ Using Phase 1 checkpoint at step ${DUAL_RESTORE_EPOCH} (latest)"
    else
        echo "Invalid choice. Aborting."
        exit 1
    fi

    # Translate host path to container path
    DUAL_RESTORE_PATH_CONTAINER="${DUAL_RESTORE_PATH_FULL/${WORKSPACE_ROOT}//workspace}"

    # ---- Phase 2 checkpoint detection ---------------------------------------
    RESUME_STEP=0

    if [ -d "${HOST_SAVE_DIR}" ]; then
        mapfile -t P2_CKPTS < <(ls "${HOST_SAVE_DIR}"/params_*.pkl 2>/dev/null | sort -t_ -k2 -n)

        if [ ${#P2_CKPTS[@]} -gt 0 ]; then
            echo ""
            echo "Existing Phase 2 checkpoints detected:"
            for i in "${!P2_CKPTS[@]}"; do
                BASE=$(basename "${P2_CKPTS[$i]}" .pkl)
                STEP=$(echo "${BASE}" | cut -d_ -f2)
                TS=$(echo "${BASE}" | cut -d_ -f3-)
                echo "  [$((i+1))] step ${STEP}  (${TS})"
            done
            echo "  [f] Start from scratch (existing checkpoints will be kept)"
            echo ""
            read -rp "Your choice: " P2_CHOICE

            if [[ "$P2_CHOICE" =~ ^[0-9]+$ ]] && [ "$P2_CHOICE" -ge 1 ] && [ "$P2_CHOICE" -le "${#P2_CKPTS[@]}" ]; then
                IDX=$((P2_CHOICE-1))
                RESUME_STEP=$(basename "${P2_CKPTS[$IDX]}" .pkl | cut -d_ -f2)
                echo "→ Resuming Phase 2 from step ${RESUME_STEP}"
            elif [[ "${P2_CHOICE,,}" == "f" ]]; then
                echo "→ Starting Phase 2 from scratch (existing checkpoints preserved)."
                RESUME_STEP=0
            else
                echo "Invalid choice. Aborting."
                exit 1
            fi
        fi
    fi

    echo ""
    echo "============================================"
    echo "  Phase 2: Downstream GCVF Training"
    echo "  env             : ${ENV_NAME}"
    echo "  dual checkpoint : ${DUAL_RESTORE_PATH_CONTAINER} @ step ${DUAL_RESTORE_EPOCH}"
    echo "  train_steps     : ${TRAIN_STEPS_P2}"
    echo "  save_dir        : ${SAVE_DIR}"
    [ "${RESUME_STEP}" -gt 0 ] && echo "  resume_step     : ${RESUME_STEP}"
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
                --dual_restore_path=${DUAL_RESTORE_PATH_CONTAINER} \
                --dual_restore_epoch=${DUAL_RESTORE_EPOCH} \
                --train_steps=${TRAIN_STEPS_P2} \
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
    echo "Checkpoint saved to: ${HOST_SAVE_DIR}/"
fi
