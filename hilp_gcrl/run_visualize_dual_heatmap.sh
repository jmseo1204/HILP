#!/bin/bash
# ============================================================
# Visualize Dual Goal Representation Heatmap
# Mode: dual_repr  →  V(s,g) = psi(s)^T phi(g)  or  -||psi(s)-phi(g)||
# Shows temporal distance from goal using the Phase-1 model.
# Interactively prompts for aggregator, then finds matching checkpoint.
# ============================================================

set -e

WORKSPACE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DOCKER_IMAGE="mctd:0.1"
OGBENCH_DATA_DIR="${WORKSPACE_ROOT}/ogbench_data"
UNAME="junjolp2026spring"

# ---- Parameters -------------------------------------------------------------
ENV_NAME="antmaze-giant-navigate-v0"
SKILL_DIM=256

# Goal position (x, y) — adjust for your environment
GOAL_X=12.0
GOAL_Y=8.0

GRID_RES=100
SAVE_DIR="/workspace/HILP/hilp_gcrl/visualizations"
PYTHON_SCRIPT="/workspace/HILP/hilp_gcrl/visualize_dual_heatmap.py"

# ---- [Step 1] Select aggregator ---------------------------------------------
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

# ---- Checkpoint path (new structure: .../ENV_NAME/AGGREGATOR/) --------------
HOST_CKPT_DIR="${WORKSPACE_ROOT}/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}/${AGGREGATOR}"
RESTORE_PATH="/workspace/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}/${AGGREGATOR}"

# Fallback: old structure without aggregator subdir (for checkpoints trained before this change)
if [ ! -d "${HOST_CKPT_DIR}" ]; then
    HOST_CKPT_DIR="${WORKSPACE_ROOT}/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}"
    RESTORE_PATH="/workspace/HILP/hilp_gcrl/exp/dual_repr/${ENV_NAME}"
    echo "  (using legacy checkpoint path without aggregator subdir)"
fi

# ---- [Step 2] Select checkpoint epoch ---------------------------------------
mapfile -t CKPT_FILES < <(ls "${HOST_CKPT_DIR}"/params_*.pkl 2>/dev/null | sort -t_ -k2 -n)

if [ ${#CKPT_FILES[@]} -eq 0 ]; then
    echo "No checkpoints found in: ${HOST_CKPT_DIR}"
    exit 1
fi

echo ""
echo "Available checkpoints:"
for i in "${!CKPT_FILES[@]}"; do
    STEP=$(basename "${CKPT_FILES[$i]}" .pkl | sed 's/params_//')
    echo "  [$((i+1))] step ${STEP}"
done
read -rp "Your choice: " CKPT_CHOICE

if [[ "$CKPT_CHOICE" =~ ^[0-9]+$ ]] && [ "$CKPT_CHOICE" -ge 1 ] && [ "$CKPT_CHOICE" -le "${#CKPT_FILES[@]}" ]; then
    IDX=$((CKPT_CHOICE-1))
    RESTORE_EPOCH=$(basename "${CKPT_FILES[$IDX]}" .pkl | sed 's/params_//')
    echo "→ Checkpoint: step ${RESTORE_EPOCH}"
else
    echo "Invalid choice. Aborting."
    exit 1
fi

echo ""
echo "============================================"
echo "  Dual Repr Heatmap (mode: dual_repr)"
echo "  env          : ${ENV_NAME}"
echo "  aggregator   : ${AGGREGATOR}"
echo "  checkpoint   : ${RESTORE_PATH} @ step ${RESTORE_EPOCH}"
echo "  goal         : (${GOAL_X}, ${GOAL_Y})"
echo "  output dir   : ${SAVE_DIR}"
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

echo ""
echo "Heatmap saved to: ${WORKSPACE_ROOT}/HILP/hilp_gcrl/visualizations/"
