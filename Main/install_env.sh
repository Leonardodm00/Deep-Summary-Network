#!/usr/bin/env bash
# =============================================================================
# install_env.sh -- meacnn environment installer for davinci-1 (HPC)
# =============================================================================
#
# What this script does (in order)
# ---------------------------------
#   1. Checks the execution context (warns if on a login node).
#   2. Detects the system CUDA version from nvidia-smi.
#   3. Selects the matching PyTorch wheel URL (cu118 / cu121 / cu124).
#   4. Creates (or recreates) the conda environment from environment.yml.
#   5. Installs PyTorch with the correct CUDA wheel.
#   6. pip-installs pytorch-metric-learning and the rest of the pip block.
#   7. Runs verify_env.py as a final sanity check.
#
# Prerequisites on davinci-1
# --------------------------
#   (a) Load the CUDA module BEFORE running this script:
#           module load cuda/12.1          # or 11.8, 12.4 -- check with:
#           module avail cuda
#
#   (b) Make conda available:
#           module load miniconda3         # or whatever the site module is
#           # check with: module avail conda | module avail miniconda
#
#   (c) Run from a compute / build node, NOT the login node:
#           qsub -I -l select=1:ncpus=4:ngpus=1 -l walltime=01:00:00
#
# Usage
# -----
#   bash install_env.sh               # standard install
#   bash install_env.sh --rebuild     # destroy existing env and rebuild
#
# After installation
# ------------------
#   conda activate meacnn
#   python verify_env.py              # should print 0 failures
# =============================================================================

set -euo pipefail          # exit on error, undefined vars, pipeline failures
IFS=$'\n\t'

# -- configurable variables ----------------------------------------------------
ENV_NAME="meacnn"
PYTHON_VERSION="3.11"
TORCH_VERSION="2.3.1"      # pinned; bump here when updating
ENV_FILE="environment.yml" # must live in the same directory as this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -- colour helpers (no-op if terminal lacks colour) ---------------------------
RED='\033[0;31m';  GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'        # no colour
info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERR]${NC}   $*" >&2; }

# -- banner --------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "  meacnn environment installer -- davinci-1 HPC"
echo "======================================================================"
echo ""

# -- 0. parse flags ------------------------------------------------------------
REBUILD=false
for arg in "$@"; do
    [[ "$arg" == "--rebuild" ]] && REBUILD=true
done

# -- 1. login-node guard -------------------------------------------------------
# On PBS systems PBS_JOBID is only set when running inside a job (batch or interactive).
if [[ -z "${PBS_JOBID:-}" ]]; then
    warn "PBS_JOBID is not set -- you may be on a login node."
    warn "Heavy installs on login nodes can get killed and may be"
    warn "against site policy. To get an interactive compute node run:"
    warn "    qsub -I -l select=1:ncpus=4:ngpus=1 -l walltime=01:00:00"
    echo ""
    read -rp "  Continue anyway? [y/N] " yn
    [[ "$yn" =~ ^[Yy]$ ]] || { info "Aborted."; exit 0; }
fi

# -- 2. CUDA version detection -------------------------------------------------
info "Detecting CUDA version..."

CUDA_VER=""
if command -v nvidia-smi &>/dev/null; then
    # nvidia-smi is the most reliable source on HPC (reads driver, not toolkit)
    CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
fi
if [[ -z "$CUDA_VER" ]] && command -v nvcc &>/dev/null; then
    # fall back to nvcc (toolkit version, may differ from driver version)
    CUDA_VER=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+" | head -1)
fi
if [[ -z "$CUDA_VER" ]]; then
    warn "No CUDA found (nvidia-smi and nvcc both unavailable)."
    warn "Have you run:  module load cuda/<version>  ?"
    warn "Falling back to CPU-only PyTorch (no GPU training)."
    CUDA_VER="0.0"
fi

CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d'.' -f1)
CUDA_MINOR=$(echo "$CUDA_VER" | cut -d'.' -f2)
info "Detected CUDA ${CUDA_VER}"

# -- 3. select PyTorch CUDA wheel tag -----------------------------------------
#
#    PyTorch wheel tags vs system CUDA:
#       cu118  -> works with CUDA 11.8 and any 11.x >= 11.8
#       cu121  -> works with CUDA 12.0 / 12.1 / 12.2 / 12.3
#       cu124  -> works with CUDA 12.4 and above
#
#    Rule: use the HIGHEST cu-tag that does NOT exceed your system CUDA.
if   (( CUDA_MAJOR >= 12 && CUDA_MINOR >= 4 )); then
    TORCH_CUDA_TAG="cu124"
elif (( CUDA_MAJOR >= 12 )); then
    TORCH_CUDA_TAG="cu121"
elif (( CUDA_MAJOR == 11 && CUDA_MINOR >= 8 )); then
    TORCH_CUDA_TAG="cu118"
elif (( CUDA_MAJOR == 0 )); then
    TORCH_CUDA_TAG="cpu"
else
    warn "CUDA ${CUDA_VER} < 11.8: no matching GPU wheel."
    warn "Installing CPU-only PyTorch."
    TORCH_CUDA_TAG="cpu"
fi

if [[ "$TORCH_CUDA_TAG" == "cpu" ]]; then
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
else
    TORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"
fi
info "PyTorch wheel: torch==${TORCH_VERSION}+${TORCH_CUDA_TAG}"
info "Index URL    : ${TORCH_INDEX_URL}"
echo ""

# -- 4. check environment.yml exists ------------------------------------------
ENV_FILE_PATH="${SCRIPT_DIR}/${ENV_FILE}"
if [[ ! -f "$ENV_FILE_PATH" ]]; then
    err "environment.yml not found at ${ENV_FILE_PATH}"
    err "Make sure install_env.sh and environment.yml are in the same directory."
    exit 1
fi

# -- 5. create / rebuild the conda environment ---------------------------------
#
#    Prefer mamba over conda if available: the solver is 10-20x faster
#    for complex envs with numpy/scipy/matplotlib.
if command -v mamba &>/dev/null; then
    CONDA_CMD="mamba"
    info "Using mamba (faster solver)"
else
    CONDA_CMD="conda"
    warn "mamba not found -- using conda (may be slow). Consider:"
    warn "    conda install -n base -c conda-forge mamba"
fi

if conda env list | grep -q "^${ENV_NAME} "; then
    if [[ "$REBUILD" == true ]]; then
        warn "Removing existing environment '${ENV_NAME}' (--rebuild)..."
        conda env remove -n "$ENV_NAME" -y
    else
        warn "Environment '${ENV_NAME}' already exists."
        warn "To rebuild from scratch:  bash install_env.sh --rebuild"
        warn "Skipping conda create; going straight to PyTorch install."
    fi
fi

if ! conda env list | grep -q "^${ENV_NAME} "; then
    info "Creating conda environment '${ENV_NAME}' from ${ENV_FILE}..."
    $CONDA_CMD env create -f "$ENV_FILE_PATH" -n "$ENV_NAME"
    ok "Conda environment created."
fi

# -- 6. activate environment inside the script ---------------------------------
#
#    `conda activate` modifies the current shell; inside a bash script we must
#    source the conda init script first, then call activate.
CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
info "Activated: $(which python)  ($(python --version))"
echo ""

# -- 7. install PyTorch (CUDA-specific wheel) ----------------------------------
info "Installing PyTorch ${TORCH_VERSION}+${TORCH_CUDA_TAG}..."
pip install \
    "torch==${TORCH_VERSION}" \
    --index-url "$TORCH_INDEX_URL" \
    --no-deps          # environment.yml already covered numpy etc.
pip install \
    "torchvision" \    # optional but commonly needed
    --index-url "$TORCH_INDEX_URL" \
    --quiet
ok "PyTorch installed."
echo ""

# -- 8. install remaining pip packages -----------------------------------------
#
#    pytorch-metric-learning must come AFTER torch so its setup.py can find it.
info "Installing pip packages (PML, skopt, optuna)..."
pip install \
    "pytorch-metric-learning>=2.3" \
    "scikit-optimize>=0.9,<0.11" \
    "optuna>=3.4" \
    --quiet
ok "pip packages installed."
echo ""

# -- 9. freeze the environment for reproducibility ----------------------------
FREEZE_FILE="${SCRIPT_DIR}/environment_frozen_$(date +%Y%m%d).txt"
pip list --format=freeze > "$FREEZE_FILE"
info "Frozen pip list written to: ${FREEZE_FILE}"
info "(Keep this file: it lets you reproduce the exact install later.)"
echo ""

# -- 10. final verification ---------------------------------------------------
VERIFY_SCRIPT="${SCRIPT_DIR}/verify_env.py"
if [[ -f "$VERIFY_SCRIPT" ]]; then
    info "Running verify_env.py..."
    echo ""
    python "$VERIFY_SCRIPT"
    echo ""
else
    warn "verify_env.py not found -- skipping verification."
    warn "Run it manually once you have it."
fi

echo "======================================================================"
echo "  Done. Activate with:  conda activate ${ENV_NAME}"
echo "======================================================================"
