#!/bin/bash
# ============================================================
# Unified Training Launcher
#   [1] Phase 1: Dual Goal Representations (train_dual_ogbench.py)
#   [2] Phase 2: Downstream GCVF on Frozen Dual Repr (train_gcvf_dual_ogbench.py)
# ============================================================

set -e

WORKSPACE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MCTD_PROJECT_DIR="${WORKSPACE_ROOT}/mctd"
MCTD_GPU_LIB="${MCTD_PROJECT_DIR}/scripts/mctd_ckpt_lib.sh"

ORIGINAL_DOCKER_IMAGE="${DOCKER_IMAGE-}"
ORIGINAL_DOCKER_USER="${DOCKER_USER-}"
ORIGINAL_AVAILABLE_GPUS="${AVAILABLE_GPUS-}"
ORIGINAL_WANDB_PROJECT="${WANDB_PROJECT-}"

if [ -f "${MCTD_GPU_LIB}" ]; then
    # shellcheck source=/dev/null
    source "${MCTD_GPU_LIB}"
else
    echo "[WARN] Shared GPU helper not found: ${MCTD_GPU_LIB}"
fi

[ -n "${ORIGINAL_DOCKER_IMAGE}" ] && DOCKER_IMAGE="${ORIGINAL_DOCKER_IMAGE}"
[ -n "${ORIGINAL_DOCKER_USER}" ] && DOCKER_USER="${ORIGINAL_DOCKER_USER}"
[ -n "${ORIGINAL_AVAILABLE_GPUS}" ] && AVAILABLE_GPUS="${ORIGINAL_AVAILABLE_GPUS}"
if [ -n "${ORIGINAL_WANDB_PROJECT}" ]; then
    WANDB_PROJECT="${ORIGINAL_WANDB_PROJECT}"
else
    unset WANDB_PROJECT 2>/dev/null || true
fi

DOCKER_IMAGE="${DOCKER_IMAGE:-mctd:0.1}"
OGBENCH_DATA_DIR="${WORKSPACE_ROOT}/ogbench_data"
UNAME="${DOCKER_USER:-$(whoami)}"
DEVICE="${DEVICE:-}"
HILP_GPU_IDS_CSV=""

check_docker_ready() {
    echo "Checking Docker availability..."
    if ! command -v docker >/dev/null 2>&1; then
        echo "ERROR: Docker is not installed or not in PATH"
        exit 1
    fi
    if ! docker ps >/dev/null 2>&1; then
        echo "ERROR: Docker daemon is not running"
        exit 1
    fi
    echo "✓ Docker is available and running"
}

prepare_ogbench_data_dir() {
    local container_uid
    local container_gid

    mkdir -p "${OGBENCH_DATA_DIR}"

    echo "Ensuring OGBench data directory ownership matches the container user..."
    container_uid="$(docker run --rm --entrypoint bash "${DOCKER_IMAGE}" -lc 'id -u')"
    container_gid="$(docker run --rm --entrypoint bash "${DOCKER_IMAGE}" -lc 'id -g')"

    docker run --rm \
        --user root \
        --entrypoint bash \
        -v "${OGBENCH_DATA_DIR}:/mnt/ogbench_data" \
        "${DOCKER_IMAGE}" \
        -lc "mkdir -p /mnt/ogbench_data && chown -R ${container_uid}:${container_gid} /mnt/ogbench_data"

    echo "✓ OGBench data directory ready: ${OGBENCH_DATA_DIR} (owner ${container_uid}:${container_gid})"
}

hilp_set_docker_device_from_available_gpus() {
    local entries="${AVAILABLE_GPUS:-localhost:0}"
    local old_ifs="$IFS"
    local entry host gpu
    local -a gpu_ids=()

    IFS=","
    for entry in ${entries}; do
        IFS="$old_ifs"
        host="${entry%%:*}"
        gpu="${entry##*:}"
        if [ -z "${gpu}" ]; then
            continue
        fi
        if [ "${host}" != "localhost" ]; then
            echo "ERROR: Non-local GPU entry is not supported for Docker launch: ${entry}"
            exit 1
        fi
        gpu_ids+=("${gpu}")
        IFS=","
    done
    IFS="$old_ifs"

    if [ ${#gpu_ids[@]} -eq 0 ]; then
        echo "ERROR: No GPU indices found in AVAILABLE_GPUS=${entries}"
        exit 1
    fi

    HILP_GPU_IDS_CSV="$(IFS=,; echo "${gpu_ids[*]}")"
    DEVICE="device=${HILP_GPU_IDS_CSV}"
    export DEVICE
    echo "✓ Docker GPU selection: ${DEVICE}"
}

configure_gpu_selection() {
    echo ""
    if declare -F mctd_select_gpus >/dev/null 2>&1; then
        mctd_select_gpus
    else
        echo "[gpu-select] Shared selector unavailable; keeping AVAILABLE_GPUS=${AVAILABLE_GPUS:-localhost:0}"
    fi

    if declare -F mctd_check_gpu_availability >/dev/null 2>&1; then
        mctd_check_gpu_availability
    fi

    hilp_set_docker_device_from_available_gpus
    echo ""
}

sanitize_container_label() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_.-]/-/g; s/-\{2,\}/-/g; s/^-//; s/-$//'
}

build_container_name() {
    local label
    label="$(sanitize_container_label "$1")"
    local gpu_tag="${HILP_GPU_IDS_CSV//,/-}"
    printf "hilp-%s-gpu%s-%s" "${label}" "${gpu_tag:-na}" "$(date +%Y%m%d_%H%M%S)"
}

launch_training_container() {
    local container_name="$1"
    local inner_cmd="$2"

    docker run -d \
        --name "${container_name}" \
        --gpus "${DEVICE}" \
        -v "${WORKSPACE_ROOT}:/workspace" \
        -v "${OGBENCH_DATA_DIR}:/home/${UNAME}/.ogbench/data" \
        -w /workspace/HILP/hilp_gcrl \
        -e MUJOCO_GL=egl \
        -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
        -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
        "${DOCKER_IMAGE}" bash -lc "${inner_cmd}"
}

DATASET_OPTIONS=(
    "antmaze-medium-navigate-v0"
    "antmaze-large-navigate-v0"
    "antmaze-giant-navigate-v0"
    "antmaze-teleport-navigate-v0"
    "antmaze-medium-stitch-v0"
    "antmaze-large-stitch-v0"
    "antmaze-giant-stitch-v0"
    "antmaze-teleport-stitch-v0"
    "antmaze-medium-explore-v0"
    "antmaze-large-explore-v0"
    "antmaze-teleport-explore-v0"
)

select_dataset() {
    echo ""
    echo "============================================"
    echo "  Select dataset:"
    for i in "${!DATASET_OPTIONS[@]}"; do
        printf "  [%2d] %s\n" "$((i+1))" "${DATASET_OPTIONS[$i]}"
    done
    echo "  [c] Custom OGBench dataset name"
    echo "============================================"
    read -rp "Your choice [default: ${ENV_NAME}]: " DATASET_CHOICE

    if [ -z "${DATASET_CHOICE}" ]; then
        echo "→ Dataset: ${ENV_NAME}"
        return
    fi

    if [[ "${DATASET_CHOICE}" =~ ^[0-9]+$ ]] && [ "${DATASET_CHOICE}" -ge 1 ] && [ "${DATASET_CHOICE}" -le "${#DATASET_OPTIONS[@]}" ]; then
        ENV_NAME="${DATASET_OPTIONS[$((DATASET_CHOICE-1))]}"
    elif [[ "${DATASET_CHOICE,,}" == "c" ]]; then
        read -rp "Enter dataset name: " CUSTOM_ENV_NAME
        if [ -z "${CUSTOM_ENV_NAME}" ]; then
            echo "Dataset name cannot be empty. Aborting."
            exit 1
        fi
        ENV_NAME="${CUSTOM_ENV_NAME}"
    else
        echo "Invalid choice. Aborting."
        exit 1
    fi

    echo "→ Dataset: ${ENV_NAME}"
}

describe_dual_ckpt() {
    local ckpt_path="$1"
    local rel_path="${ckpt_path#${HOST_DUAL_DIR}/}"
    local -a rel_parts
    IFS='/' read -r -a rel_parts <<< "${rel_path}"

    CKPT_AGGREGATOR="${rel_parts[0]}"
    CKPT_ENCODER_MODE="legacy"
    if [ "${#rel_parts[@]}" -ge 3 ]; then
        CKPT_ENCODER_MODE="${rel_parts[1]}"
    fi
}

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
P_PROHIBIT=0.0
LAMBDA_NEG=0.1
PROHIBIT_THRESHOLD=1.5
SHARE_ENCODER=false

# Phase 2 specific
TRAIN_STEPS_P2=500000

check_docker_ready
prepare_ogbench_data_dir
configure_gpu_selection

# ---- [Step 0] Select dataset -------------------------------------------------
select_dataset

# ---- [Step 1] Select training phase -----------------------------------------
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

    # ---- [Step 2] Select aggregator -----------------------------------------
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

    # ---- [Step 3] Encoder sharing -------------------------------------------
    echo ""
    echo "============================================"
    echo "  Select encoder mode:"
    echo "  [1] separate  — psi(s) 와 phi(g) 를 별도 MLP"
    echo "  [2] shared    — psi(s) 와 phi(g) 가 동일 MLP 공유"
    echo "============================================"
    read -rp "Your choice [1/2, default: 1]: " ENC_CHOICE
    ENC_CHOICE="${ENC_CHOICE:-1}"

    case "${ENC_CHOICE}" in
        1) SHARE_ENCODER=false ; ENC_MODE="separate" ;;
        2) SHARE_ENCODER=true  ; ENC_MODE="shared"   ;;
        *)
            echo "Invalid choice. Aborting."
            exit 1
            ;;
    esac
    echo "→ share_encoder: ${SHARE_ENCODER}"

    SAVE_DIR="/workspace/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}/${AGGREGATOR}/${ENC_MODE}"
    HOST_SAVE_DIR="${WORKSPACE_ROOT}/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}/${AGGREGATOR}/${ENC_MODE}"

    WANDB_PROJECT="${WANDB_PROJECT:-hilp_gcrl}"
    WANDB_RUN_NAME="${WANDB_RUN_NAME:-dual_repr_${AGGREGATOR}_${ENV_NAME}}"

    # ---- [Step 4] Checkpoint detection --------------------------------------
    RESUME_STEP=0

    if [ -d "${HOST_SAVE_DIR}" ]; then
        mapfile -t CKPT_FILES < <(ls "${HOST_SAVE_DIR}"/params_*.pkl 2>/dev/null | sort -t_ -k2 -n)

        if [ ${#CKPT_FILES[@]} -gt 0 ]; then
            echo ""
            echo "Existing checkpoints detected (aggregator=${AGGREGATOR}):"
            for i in "${!CKPT_FILES[@]}"; do
                echo "  [$((i+1))] $(basename "${CKPT_FILES[$i]}")"
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
    echo "  encoder    : ${ENC_MODE}"
    echo "  skill_dim  : ${SKILL_DIM}"
    echo "  train_steps: ${TRAIN_STEPS_P1}"
    echo "  save_dir   : ${SAVE_DIR}"
    [ "${RESUME_STEP}" -gt 0 ] && echo "  resume_step: ${RESUME_STEP}"
    echo "============================================"

    PHASE1_CMD=$(cat <<EOF
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
    --share_encoder=${SHARE_ENCODER} \
    --wandb_project=${WANDB_PROJECT} \
    --wandb_run_name=${WANDB_RUN_NAME}
EOF
)

    PHASE1_CONTAINER_NAME="$(build_container_name "p1-${AGGREGATOR}-${ENC_MODE}-${ENV_NAME}")"
    PHASE1_CONTAINER_ID="$(launch_training_container "${PHASE1_CONTAINER_NAME}" "${PHASE1_CMD}")"

    echo ""
    echo "Phase 1 training launched in background."
    echo "  Container : ${PHASE1_CONTAINER_NAME}"
    echo "  ID        : ${PHASE1_CONTAINER_ID}"
    echo "  GPU       : ${DEVICE}"
    echo "  Save dir  : ${HOST_SAVE_DIR}/"
    echo "  Logs      : docker logs -f ${PHASE1_CONTAINER_NAME}"
    echo "  Status    : docker ps --filter name=${PHASE1_CONTAINER_NAME}"
    echo "  Stop      : docker rm -f ${PHASE1_CONTAINER_NAME}"
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

    mapfile -t P1_CKPTS < <(
        while IFS= read -r ckpt; do
            STEP=$(basename "${ckpt}" .pkl | cut -d_ -f2)
            printf '%012d\t%s\n' "${STEP}" "${ckpt}"
        done < <(find "${HOST_DUAL_DIR}" -type f -name 'params_*.pkl' 2>/dev/null) | sort | cut -f2-
    )

    if [ ${#P1_CKPTS[@]} -eq 0 ]; then
        echo "ERROR: No Phase 1 checkpoints found in ${HOST_DUAL_DIR}"
        echo "       Run Phase 1 training first."
        exit 1
    fi

    echo ""
    echo "Phase 1 checkpoints available:"
    for i in "${!P1_CKPTS[@]}"; do
        STEP=$(basename "${P1_CKPTS[$i]}" .pkl | cut -d_ -f2)
        describe_dual_ckpt "${P1_CKPTS[$i]}"
        echo "  [$((i+1))] step ${STEP}  [aggregator=${CKPT_AGGREGATOR}, encoder=${CKPT_ENCODER_MODE}]"
    done

    DUAL_RESTORE_EPOCH=$(basename "${P1_CKPTS[-1]}" .pkl | cut -d_ -f2)
    DUAL_RESTORE_PATH_FULL="$(dirname "${P1_CKPTS[-1]}")"
    describe_dual_ckpt "${P1_CKPTS[-1]}"
    DUAL_AGGREGATOR="${CKPT_AGGREGATOR}"
    DUAL_ENCODER_MODE="${CKPT_ENCODER_MODE}"
    echo ""
    read -rp "Select Phase 1 checkpoint to load [default: ${DUAL_RESTORE_EPOCH}]: " P1_CHOICE

    if [[ "$P1_CHOICE" =~ ^[0-9]+$ ]] && [ "$P1_CHOICE" -ge 1 ] && [ "$P1_CHOICE" -le "${#P1_CKPTS[@]}" ]; then
        IDX=$((P1_CHOICE-1))
        DUAL_RESTORE_EPOCH=$(basename "${P1_CKPTS[$IDX]}" .pkl | cut -d_ -f2)
        DUAL_RESTORE_PATH_FULL="$(dirname "${P1_CKPTS[$IDX]}")"
        describe_dual_ckpt "${P1_CKPTS[$IDX]}"
        DUAL_AGGREGATOR="${CKPT_AGGREGATOR}"
        DUAL_ENCODER_MODE="${CKPT_ENCODER_MODE}"
        echo "→ Using Phase 1 checkpoint at step ${DUAL_RESTORE_EPOCH}"
    elif [ -z "$P1_CHOICE" ]; then
        echo "→ Using Phase 1 checkpoint at step ${DUAL_RESTORE_EPOCH} (latest)"
    else
        echo "Invalid choice. Aborting."
        exit 1
    fi

    if [ "${DUAL_ENCODER_MODE}" == "shared" ]; then
        DUAL_SHARE_ENCODER=true
    elif [ "${DUAL_ENCODER_MODE}" == "separate" ]; then
        DUAL_SHARE_ENCODER=false
    else
        echo ""
        echo "Legacy checkpoint path detected: ${DUAL_RESTORE_PATH_FULL}"
        read -rp "Was this Phase 1 checkpoint trained with shared encoder? [y/N]: " LEGACY_SHARED
        if [[ "${LEGACY_SHARED,,}" == "y" || "${LEGACY_SHARED,,}" == "yes" ]]; then
            DUAL_SHARE_ENCODER=true
            DUAL_ENCODER_MODE="shared"
        else
            DUAL_SHARE_ENCODER=false
            DUAL_ENCODER_MODE="separate"
        fi
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
                STEP=$(basename "${P2_CKPTS[$i]}" .pkl | cut -d_ -f2)
                echo "  [$((i+1))] step ${STEP}"
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
    echo "  aggregator      : ${DUAL_AGGREGATOR}"
    echo "  encoder         : ${DUAL_ENCODER_MODE}"
    echo "  train_steps     : ${TRAIN_STEPS_P2}"
    echo "  save_dir        : ${SAVE_DIR}"
    [ "${RESUME_STEP}" -gt 0 ] && echo "  resume_step     : ${RESUME_STEP}"
    echo "============================================"

    PHASE2_CMD=$(cat <<EOF
pip3 install --quiet pyrallis shapely scikit-learn ogbench distrax &&
python3 ${PYTHON_SCRIPT} \
    --env_name=${ENV_NAME} \
    --skill_dim=${SKILL_DIM} \
    --dual_restore_path=${DUAL_RESTORE_PATH_CONTAINER} \
    --dual_restore_epoch=${DUAL_RESTORE_EPOCH} \
    --dual_aggregator=${DUAL_AGGREGATOR} \
    --dual_share_encoder=${DUAL_SHARE_ENCODER} \
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
EOF
)

    PHASE2_CONTAINER_NAME="$(build_container_name "p2-${ENV_NAME}")"
    PHASE2_CONTAINER_ID="$(launch_training_container "${PHASE2_CONTAINER_NAME}" "${PHASE2_CMD}")"

    echo ""
    echo "Phase 2 training launched in background."
    echo "  Container : ${PHASE2_CONTAINER_NAME}"
    echo "  ID        : ${PHASE2_CONTAINER_ID}"
    echo "  GPU       : ${DEVICE}"
    echo "  Save dir  : ${HOST_SAVE_DIR}/"
    echo "  Logs      : docker logs -f ${PHASE2_CONTAINER_NAME}"
    echo "  Status    : docker ps --filter name=${PHASE2_CONTAINER_NAME}"
    echo "  Stop      : docker rm -f ${PHASE2_CONTAINER_NAME}"
fi
