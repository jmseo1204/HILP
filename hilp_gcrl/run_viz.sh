#!/bin/bash
# ============================================================
# Unified Heatmap + Gradient-Field Visualization
# Supports two modes:
#   [1] dual_repr  —  V(s,g) = psi(s)^T phi(g)  or  -||psi(s)-phi(g)||
#   [2] gcvf       —  V_down(s, phi(g))
# Arrows show ∇_s V(s,g) (value gradient w.r.t. state position).
# ============================================================

set -e

WORKSPACE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DOCKER_IMAGE="mctd:0.1"
OGBENCH_DATA_DIR="${WORKSPACE_ROOT}/ogbench_data"
UNAME="junjolp2026spring"

# ---- Fixed parameters -------------------------------------------------------
ENV_NAME="antmaze-giant-navigate-v0"
SKILL_DIM_DUAL=256   # skill_dim for dual_repr mode
SKILL_DIM_GCVF=32    # skill_dim for gcvf mode (dual agent used in phase-2)

GRID_RES=100
SAVE_DIR="/workspace/HILP/hilp_gcrl/visualizations"
PYTHON_SCRIPT="/workspace/HILP/hilp_gcrl/visualize_dual_heatmap.py"

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
echo "  [1] dual_repr  — V(s,g) = psi(s)^T phi(g)"
echo "  [2] gcvf       — V_down(s, phi(g))"
echo "============================================"
read -rp "Your choice [1/2]: " MODE_CHOICE

case "${MODE_CHOICE}" in
    1) VIZ_MODE="dual_repr" ;;
    2) VIZ_MODE="gcvf" ;;
    *)
        echo "Invalid choice. Aborting."
        exit 1
        ;;
esac
echo "→ Mode: ${VIZ_MODE}"

# ============================================================
# [Step 3] Aggregator (needed for dual_repr; also for dual
#          checkpoint reconstruction in gcvf mode)
# ============================================================
echo ""
echo "============================================"
echo "  Select aggregator (must match training):"
echo "  [1] inner_prod  — V = psi(s)^T phi(g)"
echo "  [2] neg_l2      — V = -||psi(s) - phi(g)||"
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

# ============================================================
# Mode-specific checkpoint selection
# ============================================================

if [ "${VIZ_MODE}" = "dual_repr" ]; then
    SKILL_DIM=${SKILL_DIM_DUAL}

    # ---- [Step 4a] Dual repr checkpoint ------------------------------------
    HOST_CKPT_DIR="${WORKSPACE_ROOT}/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}/${AGGREGATOR}"
    RESTORE_PATH="/workspace/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}/${AGGREGATOR}"

    if [ ! -d "${HOST_CKPT_DIR}" ]; then
        HOST_CKPT_DIR="${WORKSPACE_ROOT}/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}"
        RESTORE_PATH="/workspace/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}"
        echo "  (using legacy checkpoint path without aggregator subdir)"
    fi

    mapfile -t CKPT_FILES < <(ls "${HOST_CKPT_DIR}"/params_*.pkl 2>/dev/null | sort -t_ -k2 -n)
    if [ ${#CKPT_FILES[@]} -eq 0 ]; then
        echo "No checkpoints found in: ${HOST_CKPT_DIR}"
        exit 1
    fi

    echo ""
    echo "Available dual_repr checkpoints:"
    for i in "${!CKPT_FILES[@]}"; do
        STEP=$(basename "${CKPT_FILES[$i]}" .pkl | sed 's/params_//')
        echo "  [$((i+1))] step ${STEP}"
    done
    read -rp "Your choice: " CKPT_CHOICE

    if [[ "$CKPT_CHOICE" =~ ^[0-9]+$ ]] && \
       [ "$CKPT_CHOICE" -ge 1 ] && [ "$CKPT_CHOICE" -le "${#CKPT_FILES[@]}" ]; then
        IDX=$((CKPT_CHOICE-1))
        RESTORE_EPOCH=$(basename "${CKPT_FILES[$IDX]}" .pkl | sed 's/params_//')
        echo "→ Dual repr checkpoint: step ${RESTORE_EPOCH}"
    else
        echo "Invalid choice. Aborting."
        exit 1
    fi

    echo ""
    echo "============================================"
    echo "  Dual Repr Heatmap + ∇V arrows"
    echo "  env        : ${ENV_NAME}"
    echo "  aggregator : ${AGGREGATOR}"
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
                --goal_pos=${GOAL_X},${GOAL_Y} \
                --grid_res=${GRID_RES} \
                --save_dir=${SAVE_DIR}
        "

else
    # ---- gcvf mode -----------------------------------------------------------
    SKILL_DIM=${SKILL_DIM_GCVF}

    # ---- [Step 4b] GCVF checkpoint ------------------------------------------
    HOST_GCVF_DIR="${WORKSPACE_ROOT}/HILP/hilp_gcrl/exp/gcvf_dual/${ENV_NAME}"
    RESTORE_PATH="/workspace/HILP/hilp_gcrl/exp/gcvf_dual/${ENV_NAME}"

    mapfile -t GCVF_FILES < <(ls "${HOST_GCVF_DIR}"/params_*.pkl 2>/dev/null | sort -t_ -k2 -n)
    if [ ${#GCVF_FILES[@]} -eq 0 ]; then
        echo "No GCVF checkpoints found in: ${HOST_GCVF_DIR}"
        exit 1
    fi

    echo ""
    echo "Available GCVF (phase-2) checkpoints:"
    for i in "${!GCVF_FILES[@]}"; do
        STEP=$(basename "${GCVF_FILES[$i]}" .pkl | sed 's/params_//')
        echo "  [$((i+1))] step ${STEP}"
    done
    read -rp "Your choice: " GCVF_CHOICE

    if [[ "$GCVF_CHOICE" =~ ^[0-9]+$ ]] && \
       [ "$GCVF_CHOICE" -ge 1 ] && [ "$GCVF_CHOICE" -le "${#GCVF_FILES[@]}" ]; then
        IDX=$((GCVF_CHOICE-1))
        RESTORE_EPOCH=$(basename "${GCVF_FILES[$IDX]}" .pkl | sed 's/params_//')
        echo "→ GCVF checkpoint: step ${RESTORE_EPOCH}"
    else
        echo "Invalid choice. Aborting."
        exit 1
    fi

    # ---- [Step 5] Dual (phase-1) checkpoint for phi(g) ----------------------
    HOST_DUAL_DIR="${WORKSPACE_ROOT}/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}/${AGGREGATOR}"
    DUAL_RESTORE_PATH="/workspace/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}/${AGGREGATOR}"

    if [ ! -d "${HOST_DUAL_DIR}" ]; then
        HOST_DUAL_DIR="${WORKSPACE_ROOT}/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}"
        DUAL_RESTORE_PATH="/workspace/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}"
        echo "  (using legacy dual checkpoint path without aggregator subdir)"
    fi

    mapfile -t DUAL_FILES < <(ls "${HOST_DUAL_DIR}"/params_*.pkl 2>/dev/null | sort -t_ -k2 -n)
    if [ ${#DUAL_FILES[@]} -eq 0 ]; then
        echo "No dual repr checkpoints found in: ${HOST_DUAL_DIR}"
        exit 1
    fi

    echo ""
    echo "Available dual_repr (phase-1) checkpoints for phi(g):"
    for i in "${!DUAL_FILES[@]}"; do
        STEP=$(basename "${DUAL_FILES[$i]}" .pkl | sed 's/params_//')
        echo "  [$((i+1))] step ${STEP}"
    done
    read -rp "Your choice: " DUAL_CHOICE

    if [[ "$DUAL_CHOICE" =~ ^[0-9]+$ ]] && \
       [ "$DUAL_CHOICE" -ge 1 ] && [ "$DUAL_CHOICE" -le "${#DUAL_FILES[@]}" ]; then
        IDX=$((DUAL_CHOICE-1))
        DUAL_RESTORE_EPOCH=$(basename "${DUAL_FILES[$IDX]}" .pkl | sed 's/params_//')
        echo "→ Dual repr checkpoint: step ${DUAL_RESTORE_EPOCH}"
    else
        echo "Invalid choice. Aborting."
        exit 1
    fi

    echo ""
    echo "============================================"
    echo "  Downstream GCVF Heatmap + ∇V arrows"
    echo "  env            : ${ENV_NAME}"
    echo "  aggregator     : ${AGGREGATOR}"
    echo "  gcvf ckpt      : ${RESTORE_PATH} @ step ${RESTORE_EPOCH}"
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
                --restore_path=${RESTORE_PATH} \
                --restore_epoch=${RESTORE_EPOCH} \
                --dual_restore_path=${DUAL_RESTORE_PATH} \
                --dual_restore_epoch=${DUAL_RESTORE_EPOCH} \
                --skill_dim=${SKILL_DIM} \
                --aggregator=${AGGREGATOR} \
                --goal_pos=${GOAL_X},${GOAL_Y} \
                --grid_res=${GRID_RES} \
                --save_dir=${SAVE_DIR}
        "
fi

echo ""
echo "Heatmap saved to: ${WORKSPACE_ROOT}/HILP/hilp_gcrl/visualizations/"
