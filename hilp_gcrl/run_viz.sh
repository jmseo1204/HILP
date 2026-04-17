#!/bin/bash
# ============================================================
# Unified Heatmap + Gradient-Field Visualization
# Supports two modes:
#   [1] dual_repr  - V(s,g) = psi(s)^T phi(g)  or  -||psi(s)-phi(g)||
#   [2] gcvf       - V_down(s, phi(g))
# Arrows show grad_s V(s,g) (value gradient w.r.t. state position).
# ============================================================

set -e

WORKSPACE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DOCKER_IMAGE="mctd:0.1"
OGBENCH_DATA_DIR="${WORKSPACE_ROOT}/ogbench_data"
UNAME="$(whoami)"
HILP_ROOT="${WORKSPACE_ROOT}/HILP/hilp_gcrl"
DUAL_EXP_ROOT="${HILP_ROOT}/exp/dual_repr"
GCVF_EXP_ROOT="${HILP_ROOT}/exp/gcvf_dual"

# ---- Fixed parameters -------------------------------------------------------
SKILL_DIM_DUAL=256
SKILL_DIM_GCVF=32
GRID_RES=100
SAVE_DIR="/workspace/HILP/hilp_gcrl/visualizations"
PYTHON_SCRIPT="/workspace/HILP/hilp_gcrl/visualize_dual_heatmap.py"

CHOSEN_INDEX=-1

normalize_aggregator() {
    case "$1" in
        inner|inner_prod) echo "inner_prod" ;;
        neg_l2) echo "neg_l2" ;;
        *) echo "" ;;
    esac
}

host_path_to_container() {
    local host_path="$1"

    case "${host_path}" in
        "${WORKSPACE_ROOT}"/*) printf '/workspace%s\n' "${host_path#${WORKSPACE_ROOT}}" ;;
        *)
            echo "ERROR: Cannot map host path to container path: ${host_path}"
            exit 1
            ;;
    esac
}

has_ckpt_under_dir() {
    local search_dir="$1"
    local first_ckpt=""

    if [ -d "${search_dir}" ]; then
        first_ckpt="$(find "${search_dir}" -type f -name 'params_*.pkl' -print -quit 2>/dev/null)"
    fi

    [ -n "${first_ckpt}" ]
}

collect_sorted_ckpts() {
    local search_dir="$1"
    local -n out_ref="$2"
    local ckpt_path=""
    local step=""

    out_ref=()

    [ -d "${search_dir}" ] || return 0

    while IFS=$'\t' read -r _sorted_step ckpt_path; do
        [ -n "${ckpt_path}" ] && out_ref+=("${ckpt_path}")
    done < <(
        while IFS= read -r ckpt_path; do
            step="$(basename "${ckpt_path}" .pkl | cut -d_ -f2)"
            if [[ "${step}" =~ ^[0-9]+$ ]]; then
                printf '%012d\t%s\n' "${step}" "${ckpt_path}"
            else
                printf '%012d\t%s\n' 0 "${ckpt_path}"
            fi
        done < <(find "${search_dir}" -type f -name 'params_*.pkl' 2>/dev/null) | sort -t $'\t' -k1,1n -k2,2
    )
}

prompt_index_choice() {
    local prompt="$1"
    local max_choice="$2"
    local user_choice=""

    read -rp "${prompt}" user_choice

    if [[ "${user_choice}" =~ ^[0-9]+$ ]] && [ "${user_choice}" -ge 1 ] && [ "${user_choice}" -le "${max_choice}" ]; then
        CHOSEN_INDEX=$((user_choice - 1))
        return 0
    fi

    echo "Invalid choice. Aborting."
    exit 1
}

discover_datasets() {
    local dataset_file=""
    local dataset_name=""

    DATASET_OPTIONS=()

    if [ ! -d "${OGBENCH_DATA_DIR}" ]; then
        echo "Dataset directory not found: ${OGBENCH_DATA_DIR}"
        exit 1
    fi

    while IFS= read -r dataset_file; do
        dataset_name="$(basename "${dataset_file}" .npz)"
        DATASET_OPTIONS+=("${dataset_name}")
    done < <(find "${OGBENCH_DATA_DIR}" -maxdepth 1 -type f -name '*.npz' ! -name '*-val.npz' | sort)

    if [ ${#DATASET_OPTIONS[@]} -eq 0 ]; then
        echo "No datasets found in: ${OGBENCH_DATA_DIR}"
        exit 1
    fi
}

select_dataset() {
    discover_datasets

    echo ""
    echo "============================================"
    echo "  Select dataset from ../ogbench_data:"
    for i in "${!DATASET_OPTIONS[@]}"; do
        printf '  [%2d] %s\n' "$((i + 1))" "${DATASET_OPTIONS[$i]}"
    done
    echo "============================================"
    prompt_index_choice "Your choice: " "${#DATASET_OPTIONS[@]}"

    ENV_NAME="${DATASET_OPTIONS[$CHOSEN_INDEX]}"
    echo "→ Dataset: ${ENV_NAME}"
}

ensure_dataset_has_any_ckpt() {
    local dual_dir="${DUAL_EXP_ROOT}/${ENV_NAME}"
    local gcvf_dir="${GCVF_EXP_ROOT}/${ENV_NAME}"

    if has_ckpt_under_dir "${dual_dir}" || has_ckpt_under_dir "${gcvf_dir}"; then
        return 0
    fi

    echo "No checkpoints found for dataset: ${ENV_NAME}"
    echo "  checked:"
    echo "    ${dual_dir}"
    echo "    ${gcvf_dir}"
    exit 1
}

infer_dual_ckpt_metadata() {
    local ckpt_path="$1"
    local rel_path="${ckpt_path#${HOST_DUAL_ENV_DIR}/}"
    local -a rel_parts=()
    local raw_aggregator=""
    local raw_encoder="separate"

    IFS='/' read -r -a rel_parts <<< "${rel_path}"

    case "${#rel_parts[@]}" in
        2)
            raw_aggregator="${rel_parts[0]}"
            ;;
        3)
            raw_aggregator="${rel_parts[0]}"
            raw_encoder="${rel_parts[1]}"
            ;;
        *)
            return 1
            ;;
    esac

    CKPT_AGGREGATOR="$(normalize_aggregator "${raw_aggregator}")"
    [ -n "${CKPT_AGGREGATOR}" ] || return 1

    case "${raw_encoder}" in
        shared) CKPT_ENCODER_MODE="shared" ;;
        separate|"") CKPT_ENCODER_MODE="separate" ;;
        *) return 1 ;;
    esac

    CKPT_STEP="$(basename "${ckpt_path}" .pkl | cut -d_ -f2)"
    [[ "${CKPT_STEP}" =~ ^[0-9]+$ ]] || return 1
}

build_dual_ckpt_options() {
    local -a raw_ckpts=()
    local ckpt_path=""
    local rel_path=""

    DUAL_FILES=()
    DUAL_LABELS=()
    DUAL_STEPS=()
    DUAL_AGGREGATORS=()
    DUAL_ENCODER_MODES=()
    DUAL_RESTORE_DIRS=()

    collect_sorted_ckpts "${HOST_DUAL_ENV_DIR}" raw_ckpts

    for ckpt_path in "${raw_ckpts[@]}"; do
        if infer_dual_ckpt_metadata "${ckpt_path}"; then
            rel_path="${ckpt_path#${HOST_DUAL_ENV_DIR}/}"
            DUAL_FILES+=("${ckpt_path}")
            DUAL_LABELS+=("step ${CKPT_STEP} | agg=${CKPT_AGGREGATOR} | encoder=${CKPT_ENCODER_MODE} | ${rel_path}")
            DUAL_STEPS+=("${CKPT_STEP}")
            DUAL_AGGREGATORS+=("${CKPT_AGGREGATOR}")
            DUAL_ENCODER_MODES+=("${CKPT_ENCODER_MODE}")
            DUAL_RESTORE_DIRS+=("$(host_path_to_container "$(dirname "${ckpt_path}")")")
        fi
    done
}

select_dual_ckpt() {
    local header="$1"
    local choice_label="$2"
    local selected_ckpt=""

    HOST_DUAL_ENV_DIR="${DUAL_EXP_ROOT}/${ENV_NAME}"
    build_dual_ckpt_options

    if [ ${#DUAL_FILES[@]} -eq 0 ]; then
        echo "No usable dual_repr checkpoints found for dataset: ${ENV_NAME}"
        echo "  checked: ${HOST_DUAL_ENV_DIR}"
        exit 1
    fi

    echo ""
    echo "============================================"
    echo "  ${header}"
    for i in "${!DUAL_FILES[@]}"; do
        printf '  [%2d] %s\n' "$((i + 1))" "${DUAL_LABELS[$i]}"
    done
    echo "============================================"
    prompt_index_choice "${choice_label}: " "${#DUAL_FILES[@]}"

    selected_ckpt="${DUAL_FILES[$CHOSEN_INDEX]}"
    RESTORE_EPOCH="${DUAL_STEPS[$CHOSEN_INDEX]}"
    AGGREGATOR="${DUAL_AGGREGATORS[$CHOSEN_INDEX]}"
    ENC_MODE="${DUAL_ENCODER_MODES[$CHOSEN_INDEX]}"
    RESTORE_PATH="${DUAL_RESTORE_DIRS[$CHOSEN_INDEX]}"

    echo "→ Dual repr checkpoint: $(basename "${selected_ckpt}")"
    echo "  aggregator: ${AGGREGATOR}"
    echo "  encoder   : ${ENC_MODE}"
}

build_gcvf_ckpt_options() {
    local -a raw_ckpts=()
    local ckpt_path=""
    local rel_path=""
    local step=""

    GCVF_FILES=()
    GCVF_LABELS=()
    GCVF_STEPS=()
    GCVF_RESTORE_DIRS=()

    collect_sorted_ckpts "${HOST_GCVF_ENV_DIR}" raw_ckpts

    for ckpt_path in "${raw_ckpts[@]}"; do
        step="$(basename "${ckpt_path}" .pkl | cut -d_ -f2)"
        if [[ "${step}" =~ ^[0-9]+$ ]]; then
            rel_path="${ckpt_path#${HOST_GCVF_ENV_DIR}/}"
            GCVF_FILES+=("${ckpt_path}")
            GCVF_LABELS+=("step ${step} | ${rel_path}")
            GCVF_STEPS+=("${step}")
            GCVF_RESTORE_DIRS+=("$(host_path_to_container "$(dirname "${ckpt_path}")")")
        fi
    done
}

select_gcvf_ckpt() {
    local selected_ckpt=""

    HOST_GCVF_ENV_DIR="${GCVF_EXP_ROOT}/${ENV_NAME}"
    build_gcvf_ckpt_options

    if [ ${#GCVF_FILES[@]} -eq 0 ]; then
        echo "No GCVF checkpoints found for dataset: ${ENV_NAME}"
        echo "  checked: ${HOST_GCVF_ENV_DIR}"
        exit 1
    fi

    echo ""
    echo "============================================"
    echo "  Select GCVF checkpoint:"
    for i in "${!GCVF_FILES[@]}"; do
        printf '  [%2d] %s\n' "$((i + 1))" "${GCVF_LABELS[$i]}"
    done
    echo "============================================"
    prompt_index_choice "Your choice: " "${#GCVF_FILES[@]}"

    selected_ckpt="${GCVF_FILES[$CHOSEN_INDEX]}"
    RESTORE_EPOCH="${GCVF_STEPS[$CHOSEN_INDEX]}"
    RESTORE_PATH="${GCVF_RESTORE_DIRS[$CHOSEN_INDEX]}"

    echo "→ GCVF checkpoint: $(basename "${selected_ckpt}")"
}

# ============================================================
# [Step 0] Dataset selection
# ============================================================
select_dataset
ensure_dataset_has_any_ckpt

# ============================================================
# [Step 1] Goal position
# ============================================================
echo ""
echo "============================================"
echo "  Goal position"
echo "============================================"
read -rp "Goal X [default: 12.0]: " GOAL_X
GOAL_X="${GOAL_X:-12.0}"
read -rp "Goal Y [default:  8.0]: " GOAL_Y
GOAL_Y="${GOAL_Y:-8.0}"
echo "→ Goal: (${GOAL_X}, ${GOAL_Y})"

# ============================================================
# [Step 2] Visualization mode
# ============================================================
echo ""
echo "============================================"
echo "  Select visualization mode:"
echo "  [1] dual_repr  - V(s,g) = psi(s)^T phi(g)"
echo "  [2] gcvf       - V_down(s, phi(g))"
echo "============================================"
prompt_index_choice "Your choice [1/2]: " 2

case "$((CHOSEN_INDEX + 1))" in
    1) VIZ_MODE="dual_repr" ;;
    2) VIZ_MODE="gcvf" ;;
    *)
        echo "Invalid choice. Aborting."
        exit 1
        ;;
esac
echo "→ Mode: ${VIZ_MODE}"

# ============================================================
# Mode-specific checkpoint selection
# ============================================================

if [ "${VIZ_MODE}" = "dual_repr" ]; then
    SKILL_DIM="${SKILL_DIM_DUAL}"

    select_dual_ckpt "Select dual_repr checkpoint:" "Your choice"

    SHARE_ENCODER_FLAG="false"
    [ "${ENC_MODE}" = "shared" ] && SHARE_ENCODER_FLAG="true"

    echo ""
    echo "============================================"
    echo "  Dual Repr Heatmap + grad V arrows"
    echo "  env        : ${ENV_NAME}"
    echo "  aggregator : ${AGGREGATOR}"
    echo "  encoder    : ${ENC_MODE}"
    echo "  checkpoint : ${RESTORE_PATH} @ step ${RESTORE_EPOCH}"
    echo "  goal       : (${GOAL_X}, ${GOAL_Y})"
    echo "  output dir : ${SAVE_DIR}"
    echo "============================================"

    docker run --rm \
        -v "${WORKSPACE_ROOT}:/workspace" \
        -v "${OGBENCH_DATA_DIR}:/home/${UNAME}/.ogbench/data" \
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
                --aggregator=${AGGREGATOR} \
                --share_encoder=${SHARE_ENCODER_FLAG} \
                --goal_pos=${GOAL_X},${GOAL_Y} \
                --grid_res=${GRID_RES} \
                --save_dir=${SAVE_DIR}
        "
else
    SKILL_DIM="${SKILL_DIM_GCVF}"

    select_gcvf_ckpt

    GCVF_RESTORE_EPOCH="${RESTORE_EPOCH}"
    GCVF_RESTORE_PATH="${RESTORE_PATH}"

    select_dual_ckpt "Select dual_repr checkpoint for phi(g):" "Your choice"

    DUAL_RESTORE_EPOCH="${RESTORE_EPOCH}"
    DUAL_RESTORE_PATH="${RESTORE_PATH}"

    SHARE_ENCODER_FLAG="false"
    [ "${ENC_MODE}" = "shared" ] && SHARE_ENCODER_FLAG="true"

    echo ""
    echo "============================================"
    echo "  Downstream GCVF Heatmap + grad V arrows"
    echo "  env            : ${ENV_NAME}"
    echo "  aggregator     : ${AGGREGATOR}"
    echo "  dual encoder   : ${ENC_MODE}"
    echo "  gcvf ckpt      : ${GCVF_RESTORE_PATH} @ step ${GCVF_RESTORE_EPOCH}"
    echo "  dual ckpt      : ${DUAL_RESTORE_PATH} @ step ${DUAL_RESTORE_EPOCH}"
    echo "  goal           : (${GOAL_X}, ${GOAL_Y})"
    echo "  output dir     : ${SAVE_DIR}"
    echo "============================================"

    docker run --rm \
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
                --restore_path=${GCVF_RESTORE_PATH} \
                --restore_epoch=${GCVF_RESTORE_EPOCH} \
                --dual_restore_path=${DUAL_RESTORE_PATH} \
                --dual_restore_epoch=${DUAL_RESTORE_EPOCH} \
                --skill_dim=${SKILL_DIM} \
                --aggregator=${AGGREGATOR} \
                --share_encoder=${SHARE_ENCODER_FLAG} \
                --goal_pos=${GOAL_X},${GOAL_Y} \
                --grid_res=${GRID_RES} \
                --save_dir=${SAVE_DIR}
        "
fi

echo ""
echo "Heatmap saved to: ${WORKSPACE_ROOT}/HILP/hilp_gcrl/visualizations/"
