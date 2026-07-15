#!/bin/bash
# =============================================================================
# setup_env_davinci.sh -- one-shot environment setup for the 1D-CNN MEA
#                         pipeline on davinci-1 (Leonardo S.p.A., PBS).
# =============================================================================
#
# What it does, in order:
#   1. Parses --env-name (required) and --gpu / --rebuild (optional).
#   2. Creates the conda env from environment_hpc.yml (base scientific stack
#      + libstdc++ fix + skopt + pytorch-metric-learning, but NOT torch).
#   3. Installs PyTorch:
#        - CPU wheel by default (correct for a CPU queue);
#        - CUDA 12.1 wheel if --gpu is given (davinci-1 default CUDA is 12.1).
#   4. Writes an LD_LIBRARY_PATH activation hook so the env's newer libstdc++
#      wins over davinci-1's system /lib64 one (the GLIBCXX_3.4.26 fix) on
#      every future `conda activate`, no manual export needed.
#   5. Runs verify_env_hpc.py.
#
# RUN THIS ON AN INTERACTIVE COMPUTE NODE, not the login node:
#     qsub -I -l select=1:ncpus=4 -l walltime=01:00:00      # CPU
#     # (add :ngpus=1 to the select line if you are setting up for GPU)
#     module load miniconda3          # or whatever `module avail` shows
#     bash setup_env_davinci.sh --env-name meacnn_cpu
#
# To set up for GPU later (separate env recommended so CPU stays intact):
#     module load cuda/12.1
#     module load miniconda3
#     bash setup_env_davinci.sh --env-name meacnn_gpu --gpu
#
# HPC note (hpc-python-compat): this script and every .py it touches are pure
# ASCII, so no cp1252/UTF-8 transfer corruption can break them.
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------- #
# tiny logging helpers
# --------------------------------------------------------------------------- #
info() { printf '[setup] %s\n' "$*"; }
ok()   { printf '[setup] OK: %s\n' "$*"; }
warn() { printf '[setup] WARNING: %s\n' "$*" >&2; }
err()  { printf '[setup] ERROR: %s\n' "$*" >&2; }

# --------------------------------------------------------------------------- #
# 1. parse arguments
# --------------------------------------------------------------------------- #
ENV_NAME=""
USE_GPU=false
REBUILD=false
TORCH_VERSION="2.3.1"          # pinned; matches numpy 1.26 / python 3.11
CUDA_TAG_FOR_GPU="cu121"       # davinci-1 default CUDA module is 12.1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/environment_hpc.yml"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name) ENV_NAME="$2"; shift 2 ;;
        --gpu)      USE_GPU=true; shift ;;
        --rebuild)  REBUILD=true; shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) err "unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$ENV_NAME" ]]; then
    err "--env-name is required. Example: bash setup_env_davinci.sh --env-name meacnn_cpu"
    exit 1
fi
if [[ ! -f "$ENV_FILE" ]]; then
    err "environment_hpc.yml not found next to this script (looked at ${ENV_FILE})."
    exit 1
fi

info "env name : ${ENV_NAME}"
info "device   : $([[ "$USE_GPU" == true ]] && echo GPU || echo CPU)"
info "rebuild  : ${REBUILD}"
echo ""

# --------------------------------------------------------------------------- #
# 2. pick the torch wheel index URL
# --------------------------------------------------------------------------- #
if [[ "$USE_GPU" == true ]]; then
    # Sanity check: warn (do not fail) if no CUDA is visible, since --gpu on a
    # CPU node would install a GPU wheel that then finds no device at runtime.
    if ! command -v nvidia-smi &>/dev/null; then
        warn "--gpu was given but nvidia-smi is not on PATH. If this is a CPU"
        warn "node, re-run without --gpu. Continuing with a ${CUDA_TAG_FOR_GPU} wheel anyway."
    fi
    TORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_TAG_FOR_GPU}"
    info "PyTorch wheel: torch==${TORCH_VERSION}+${CUDA_TAG_FOR_GPU}"
else
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
    info "PyTorch wheel: torch==${TORCH_VERSION}+cpu (CPU-only)"
fi
info "index URL : ${TORCH_INDEX_URL}"
echo ""

# --------------------------------------------------------------------------- #
# 3. create (or rebuild) the conda env
# --------------------------------------------------------------------------- #
if ! command -v conda &>/dev/null; then
    err "conda not found. Load the conda module first (e.g. module load miniconda3)."
    exit 1
fi

# mamba is much faster if present
if command -v mamba &>/dev/null; then
    CONDA_CMD="mamba"; info "using mamba (faster solver)"
else
    CONDA_CMD="conda"; warn "mamba not found; using conda (slower solve)"
fi

if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    if [[ "$REBUILD" == true ]]; then
        warn "removing existing env '${ENV_NAME}' (--rebuild)"
        conda env remove -n "$ENV_NAME" -y
    else
        warn "env '${ENV_NAME}' already exists; skipping create (use --rebuild to recreate)"
    fi
fi

if ! conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    info "creating conda env '${ENV_NAME}' from environment_hpc.yml ..."
    $CONDA_CMD env create -f "$ENV_FILE" -n "$ENV_NAME"
    ok "conda env created"
fi
echo ""

# --------------------------------------------------------------------------- #
# 4. activate the env inside this script
# --------------------------------------------------------------------------- #
CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
info "activated: $(which python)  ($(python --version 2>&1))"
echo ""

# --------------------------------------------------------------------------- #
# 5. install PyTorch (CPU or CUDA wheel)
# --------------------------------------------------------------------------- #
info "installing torch ${TORCH_VERSION} ..."
pip install "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX_URL"
ok "torch installed"
echo ""

# --------------------------------------------------------------------------- #
# 6. libstdc++ activation hook (davinci-1 GLIBCXX_3.4.26 fix)
#    libstdcxx-ng/libgcc-ng were installed into the env by environment_hpc.yml;
#    this hook makes the env's copy win over the system /lib64 one on every
#    future `conda activate <env>`.
# --------------------------------------------------------------------------- #
info "writing LD_LIBRARY_PATH activation hook (libstdc++ fix) ..."
HOOK_DIR="${CONDA_PREFIX}/etc/conda/activate.d"
mkdir -p "$HOOK_DIR"
cat > "${HOOK_DIR}/zz_libstdcxx.sh" <<'HOOK'
# Make this conda env's newer libstdc++ take precedence over davinci-1's
# system /lib64/libstdc++.so.6 (which lacks GLIBCXX_3.4.26). Written by
# setup_env_davinci.sh. Prefixed zz_ so it runs after conda's own hooks.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
HOOK
# apply it to the CURRENT shell too, so the verify step below already benefits
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
ok "activation hook written to ${HOOK_DIR}/zz_libstdcxx.sh"
echo ""

# --------------------------------------------------------------------------- #
# 7. verify
# --------------------------------------------------------------------------- #
VERIFY_PY="${SCRIPT_DIR}/verify_env_hpc.py"
if [[ -f "$VERIFY_PY" ]]; then
    info "running verify_env_hpc.py ..."
    python "$VERIFY_PY"
else
    warn "verify_env_hpc.py not found next to this script; skipping verification."
    warn "Run it manually once you have it: python verify_env_hpc.py"
fi

echo ""
ok "setup complete for env '${ENV_NAME}'."
info "In every future session (and in your PBS job script) do:"
info "    module load miniconda3          # + 'module load cuda/12.1' if GPU"
info "    conda activate ${ENV_NAME}"
info "    # LD_LIBRARY_PATH is set automatically by the activation hook"
