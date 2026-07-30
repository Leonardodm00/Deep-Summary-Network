#!/bin/bash
#
# launch_l3c_factorial.sh
# =======================
# Submit the l3c 3 x 2 factorial (head geometry x triplet miner) as ONE PBS
# array job. Run from the directory that holds run_optimization.py, on a LOGIN
# node: this script only checks and submits, it does not train.
#
# It runs five gates in order and stops at the first failure, because every one
# of them costs seconds here and hours if it is discovered inside a running job:
#
#   1. DESIGN   audit_factorial.py -- the six configs differ ONLY in the factor
#               fields. A stray edit anywhere else turns the ablation into an
#               uncontrolled comparison that still runs and still looks fine.
#   2. ENV      the conda env activates and torch imports.
#   3. GEOMETRY preflight_config.py on all six -- derived batch geometry, caps
#               and gates, without generating a single trace.
#   4. CACHE    populate the SHARED trace cache ONCE, via a --dry-run.
#               All six configs carry the SAME data block and the SAME
#               runtime.cache_dir, so six concurrent subjobs would otherwise
#               each try to synthesise and write the same traces at the same
#               time. cache_traces() skips traces already on disk, so
#               populating up front makes all six subjobs read-only against
#               the cache. This is the same race the 4-class launcher guards
#               against, and it is cheaper to avoid than to debug.
#   5. CONFIRM  print the budget and ask before committing 6 x 42 h of queue.
#
# Usage
#   bash launch_l3c_factorial.sh                 # all five gates, then submit
#   bash launch_l3c_factorial.sh --check-only    # gates 1-3, no cache, no qsub
#   bash launch_l3c_factorial.sh --skip-cache    # skip gate 4 (cache is warm)
#   bash launch_l3c_factorial.sh --yes           # skip the confirmation prompt
#   bash launch_l3c_factorial.sh --cells 0-2     # submit a contiguous range
#   bash launch_l3c_factorial.sh --cells 0,3     # submit these cells (one job each)
#
# Environment overrides:
#   DSN_CONDA_ENV       conda env name          (default: meacnn_cpu)
#   DSN_CONFIG_SUBDIR   where the configs live  (default: hpc/Config)
#   DSN_PBS             the PBS script to submit (default: hpc/dsn_l3c_factorial.pbs)

set -e
set -o pipefail

CONDA_ENV="${DSN_CONDA_ENV:-meacnn_cpu}"
CONFIG_SUBDIR="${DSN_CONFIG_SUBDIR:-hpc/Config}"
PBS_SCRIPT="${DSN_PBS:-hpc/dsn_l3c_factorial.pbs}"

CHECK_ONLY=0
SKIP_CACHE=0
ASSUME_YES=0
CELLS=""

while [ $# -gt 0 ]; do
    case "$1" in
        --check-only) CHECK_ONLY=1; shift ;;
        --skip-cache) SKIP_CACHE=1; shift ;;
        --yes|-y)     ASSUME_YES=1; shift ;;
        --cells)      CELLS="$2"; shift 2 ;;
        -h|--help)    sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

CONFIGS=(
    "config_l3c_h_singlemean_005.json"
    "config_l3c_h_multimean_005.json"
    "config_l3c_h_multiall_005.json"
    "config_l3c_epsh_singlemean_005.json"
    "config_l3c_epsh_multimean_005.json"
    "config_l3c_epsh_multiall_005.json"
)

hr() { echo "----------------------------------------------------------------------"; }
die() { echo ""; echo "ABORTED: $1" >&2; exit 1; }

# ---- locate everything -------------------------------------------------
if [ ! -f "run_optimization.py" ]; then
    die "run_optimization.py not found in $(pwd). cd to the pipeline directory first."
fi
if [ ! -d "$CONFIG_SUBDIR" ]; then
    die "config directory '$CONFIG_SUBDIR' not found. Set DSN_CONFIG_SUBDIR."
fi
for name in "${CONFIGS[@]}"; do
    [ -f "$CONFIG_SUBDIR/$name" ] || die "missing config: $CONFIG_SUBDIR/$name"
done
if [ "$CHECK_ONLY" -eq 0 ] && [ ! -f "$PBS_SCRIPT" ]; then
    die "PBS script '$PBS_SCRIPT' not found. Set DSN_PBS."
fi

FIRST_CFG="$CONFIG_SUBDIR/${CONFIGS[0]}"

# ---- gate 1: design audit (stdlib only, no env needed) -----------------
hr
echo "GATE 1 -- factorial design audit"
hr
AUDIT=""
for cand in "hpc/audit_factorial.py" "audit_factorial.py" \
            "$CONFIG_SUBDIR/audit_factorial.py"; do
    [ -f "$cand" ] && { AUDIT="$cand"; break; }
done
if [ -z "$AUDIT" ]; then
    die "audit_factorial.py not found. It is the only check that catches a
         stray edit in a non-factor field; do not skip it."
fi
python3 "$AUDIT" "${CONFIGS[@]/#/$CONFIG_SUBDIR/}" \
    || die "the six configs are NOT a clean factorial (see errors above)."

# ---- gate 2: environment ----------------------------------------------
hr
echo "GATE 2 -- conda environment '$CONDA_ENV'"
hr
if command -v module >/dev/null 2>&1; then
    module load miniconda3 || echo "WARN: 'module load miniconda3' failed."
fi
command -v conda >/dev/null 2>&1 || die "conda not found on PATH."
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV" || die "cannot activate conda env '$CONDA_ENV'."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
python3 -c "import torch, numpy, sklearn, skopt, pytorch_metric_learning; \
print('  torch %s | numpy %s' % (torch.__version__, numpy.__version__))" \
    || die "the environment is incomplete. Run: python3 hpc/verify_env_hpc.py"

# ---- gate 3: per-config pre-flight ------------------------------------
hr
echo "GATE 3 -- pre-flight geometry and gates (all six, no traces generated)"
hr
if [ -f "hpc/preflight_config.py" ]; then
    for name in "${CONFIGS[@]}"; do
        echo ""
        python3 hpc/preflight_config.py "$CONFIG_SUBDIR/$name" \
            || die "pre-flight failed for $name."
    done
else
    echo "WARN: hpc/preflight_config.py not found; skipping."
fi

if [ "$CHECK_ONLY" -eq 1 ]; then
    hr
    echo "--check-only: gates 1-3 passed. Nothing submitted."
    exit 0
fi

# ---- gate 4: populate the shared trace cache ONCE ----------------------
hr
echo "GATE 4 -- shared trace cache"
hr
CACHE_DIR=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['runtime']['cache_dir'])" "$FIRST_CFG")
echo "  cache_dir : $CACHE_DIR"
if [ "$SKIP_CACHE" -eq 1 ]; then
    echo "  --skip-cache: assuming the cache is already populated."
    echo "  (If it is not, all six subjobs will race to write it.)"
else
    echo "  populating via a --dry-run on ${CONFIGS[0]} (no training)."
    echo "  45 latent traces x 600 s at 50 Hz; this can take a few minutes."
    python3 -u run_optimization.py --config "$FIRST_CFG" --dry-run --verbose \
        || die "the dry run failed. Do not submit until it passes."
fi

# ---- gate 5: confirm ---------------------------------------------------
hr
echo "GATE 5 -- confirm"
hr
echo "  design    : 3 (head geometry) x 2 (miner) = 6 cells"
echo "  budget    : 140 search trials x 2 seeds + final = 282 train() runs/cell"
echo "  walltime  : $(grep -m1 'PBS -l walltime' "$PBS_SCRIPT" | sed 's/.*walltime=//')"
echo "  resources : $(grep -m1 'PBS -l select' "$PBS_SCRIPT" | sed 's/#PBS -l //')"
echo "  script    : $PBS_SCRIPT"
if [ -n "$CELLS" ]; then
    echo "  cells     : $CELLS (subset)"
else
    echo "  cells     : 0-5 (all)"
fi
echo ""
echo "  Reminder: the SEARCH PHASES DO NOT RESUME. If a subjob exceeds its"
echo "  walltime, that cell is lost entirely, not merely truncated."
echo ""
if [ "$ASSUME_YES" -eq 0 ]; then
    printf "Submit? [y/N] "
    read -r reply
    case "$reply" in
        [Yy]*) ;;
        *) echo "aborted; nothing submitted."; exit 0 ;;
    esac
fi

# ---- submit ------------------------------------------------------------
hr
if [ -n "$CELLS" ]; then
    # PBS Pro's -J takes a RANGE (X-Y, optionally :step) -- NOT a comma
    # separated list (that is Slurm's --array syntax). So a comma list is
    # submitted as one single-subjob array per cell, which also means each
    # cell gets its own job id and can be qdel'd independently.
    case "$CELLS" in
        *,*)
            OLDIFS="$IFS"; IFS=','
            for cell in $CELLS; do
                IFS="$OLDIFS"
                echo "submitting cell $cell"
                qsub -J "${cell}-${cell}" "$PBS_SCRIPT"
                IFS=','
            done
            IFS="$OLDIFS"
            ;;
        *)
            qsub -J "$CELLS" "$PBS_SCRIPT"
            ;;
    esac
else
    qsub "$PBS_SCRIPT"
fi
hr
echo "submitted."
echo "  watch queue : qstat -tu \$USER"
echo "  follow cell : tail -f dsn_l3c_fact_0.out"
echo "  one cell    : qsub -J 2-2 $PBS_SCRIPT      # re-run cell 2 only"
echo "  results     : <out_dir>/l3c_<miner>_<head>_005/results.json"
