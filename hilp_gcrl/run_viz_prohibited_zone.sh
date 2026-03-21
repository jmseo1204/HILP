#!/bin/bash
# ============================================================
# Visualize the Prohibited Zone of an OGBench maze.
#
# 'Prohibited zone' = region inside the dataset bounding box
# whose nearest-neighbour distance to any training state >= THRESHOLD.
# ============================================================

set -e

WORKSPACE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DOCKER_IMAGE="mctd:0.1"
OGBENCH_DATA_DIR="${WORKSPACE_ROOT}/ogbench_data"
UNAME="$(whoami)"

# ---- Parameters -------------------------------------------------------------
ENV_NAME="antmaze-giant-navigate-v0"
THRESHOLD=1.5        # min distance to dataset → prohibited
GRID_RES=300         # grid resolution per axis
PC_SUBSAMPLE=10      # scatter 1 in N dataset points (display only)

SAVE_DIR="/workspace/HILP/hilp_gcrl/visualizations"
PYTHON_SCRIPT="/workspace/HILP/hilp_gcrl/visualize_prohibited_zone.py"

echo ""
echo "============================================"
echo "  Prohibited Zone Visualization"
echo "  env        : ${ENV_NAME}"
echo "  threshold  : ${THRESHOLD}"
echo "  grid_res   : ${GRID_RES}"
echo "  output dir : ${SAVE_DIR}"
echo "============================================"

docker run --rm \
    -v "${WORKSPACE_ROOT}:/workspace" \
    -v "${OGBENCH_DATA_DIR}:/home/${UNAME}/.ogbench/data" \
    -w /workspace/HILP/hilp_gcrl \
    -e MUJOCO_GL=egl \
    -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "${DOCKER_IMAGE}" bash -c "
        pip3 install --quiet pyrallis shapely scikit-learn ogbench distrax scipy &&
        python3 ${PYTHON_SCRIPT} \
            --env_name=${ENV_NAME} \
            --threshold=${THRESHOLD} \
            --grid_res=${GRID_RES} \
            --pc_subsample=${PC_SUBSAMPLE} \
            --save_dir=${SAVE_DIR}
    "

echo ""
echo "Saved to: ${WORKSPACE_ROOT}/HILP/hilp_gcrl/visualizations/"
