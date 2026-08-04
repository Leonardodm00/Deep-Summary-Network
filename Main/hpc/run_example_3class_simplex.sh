#!/bin/bash
#PBS -N dsn_l3c_simplex
#PBS -l select=1:ncpus=16
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o dsn_l3c_simplex.out
#
# run_example_3class_simplex.sh
# =============================
#
# ONE diagnostic run of the 3-class LATENT surrogate benchmark under the new
# paradigm, with every plot saved. Runs either way:
#
#     qsub Main/hpc/run_example_3class_simplex.sh        # as a batch job
#     bash Main/hpc/run_example_3class_simplex.sh        # on an interactive node
#
# WHAT THIS RUN IS FOR
# --------------------
# It is a MECHANISM check, not a performance measurement. --skip-search means no
# hyper-parameter search happens: the config's fixed values are trained directly,
# 3 seeds, early stopping OFF. The point is to read the per-epoch census and
# answer four questions that no smoke test can answer:
#
#     1. does the objective keep producing gradient, or does it freeze?
#     2. does the strict semi-hard filter starve it?
#     3. does the silhouette gate ever latch, and when?
#     4. do the class centroids actually move toward the simplex target?
#
# Do NOT read the ARI or the silhouette of this run as evidence that the new
# objective is better or worse than the old one. A single un-searched
# configuration on a 3-class benchmark is not that evidence.
#
# WHAT IS NEW HERE vs the archived l3c cells
# ------------------------------------------
#     data.latent.class_center_mode  "interior" -> "simplex"
#         class centres were ONE SCALAR replicated across every label axis, so
#         all three lay on the diagonal (rank Cov = 1). Now regular-simplex
#         vertices: rank min(L, C-1) = 2, pairwise cosine -0.5.
#     train.loss_type                (new) "joint_sep"
#         margin hinge + angular hinge + gated centroid separation.
#     train.mining_strategy          "hard"
#         the collapse-SEEKING miner. Easy-positive mining resists within-class
#         collapse by design and would fight the angular floor.
#     train.selection_primary        "silhouette", floor-calibrated threshold.
#
# ----------------------------------------------------------------------------

set -uo pipefail

# --- resolve the repo whether qsub'd or run by hand -------------------------
if [ -n "${PBS_O_WORKDIR:-}" ]; then
    cd "$PBS_O_WORKDIR"
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if   [ -f "$HERE/../config.py" ];      then MAIN="$(cd "$HERE/.." && pwd)"
elif [ -f "$HERE/config.py" ];         then MAIN="$HERE"
elif [ -f "$HERE/Main/config.py" ];    then MAIN="$(cd "$HERE/Main" && pwd)"
else echo "ABORT: cannot locate Main/ from $HERE"; exit 1; fi
cd "$MAIN"
export PYTHONPATH="$MAIN"

# --- environment ------------------------------------------------------------
# Adjust these two lines to your site if they differ.
module load anaconda3 2>/dev/null || true
# shellcheck disable=SC1091
source activate brian_env 2>/dev/null || conda activate brian_env 2>/dev/null || true

python3 -c "import torch, pytorch_metric_learning, skopt, sklearn" || {
    echo "ABORT: dependencies not importable. Activate the environment first."
    exit 1
}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

# --- paths ------------------------------------------------------------------
CONFIG="hpc/Config/config_l3c_simplex_jointsep.json"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${OUT_DIR:-$MAIN/../out_l3c_simplex_$STAMP}"
CACHE_DIR="${CACHE_DIR:-$MAIN/../cache_l3c_simplex}"

if [ ! -f "$CONFIG" ]; then
    echo "ABORT: $CONFIG not found (run from the branch that adds it)."
    exit 1
fi

echo "=============================================================="
echo "repo   : $MAIN"
echo "config : $CONFIG"
echo "out    : $OUT_DIR"
echo "cache  : $CACHE_DIR"
echo "threads: $OMP_NUM_THREADS"
echo "=============================================================="

# --- 1. preflight: never spend hours on a config that cannot work -----------
echo ""
echo "== preflight =="
python3 hpc/preflight_config.py "$CONFIG" || {
    echo "ABORT: preflight rejected the config."
    exit 1
}

# --- 2. the run -------------------------------------------------------------
# NOTE the cache: changing class_center_mode changes EVERY generated trace, so
# the latent cache fingerprint changes too. --overwrite-cache forces
# regeneration; without it the run would either refuse to start or silently
# train on traces built with the OLD collinear class centres.
echo ""
echo "== training (--skip-search: fixed config, 3 seeds, ES off) =="
python3 run_optimization.py \
    --config "$CONFIG" \
    --skip-search \
    --overwrite-cache \
    --out-dir "$OUT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --device cpu \
    --verbose
RC=$?

# --- 3. what came out -------------------------------------------------------
echo ""
echo "== artifacts =="
if [ -d "$OUT_DIR" ]; then
    echo "  figures:"
    find "$OUT_DIR" -name "*.png" | sed 's/^/    /'
    echo "  json / npz:"
    find "$OUT_DIR" -maxdepth 2 \( -name "*.json" -o -name "*.npz" \) | sed 's/^/    /'
fi

# --- 4. the census: the four questions this run exists to answer -------------
echo ""
echo "== per-epoch census =="
python3 - "$OUT_DIR" <<'PYEOF'
import glob, os, sys

import torch

out = sys.argv[1]
# The per-epoch history is NOT in results.json -- it is carried in the
# checkpoint under extra["history"], written every epoch by train.py. Verified
# on a real run; reading results.json finds nothing and looks like a bug.
cks = sorted(glob.glob(os.path.join(out, "**", "checkpoints", "seed_*", "last.pt"),
                       recursive=True))
if not cks:
    print("  no seed checkpoint found under %s" % out)
    raise SystemExit(0)

for ck_path in cks:
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    h = ck.get("extra", {}).get("history")
    seed = os.path.basename(os.path.dirname(ck_path))
    print("")
    print("  --- %s (%s) ---" % (seed, ck_path))
    if not h:
        print("  no history in this checkpoint.")
        continue
    hdr = ("epoch", "loss", "sil", "ari", "mined", "strict", "active",
           "sep_cos", "run_sil", "gate")
    print("  " + "".join("%9s" % c for c in hdr))
    for r in h:
        print("  " + "".join([
            "%9d" % r["epoch"],
            "%9.4f" % r["train_loss"],
            "%9.4f" % r["silhouette"],
            "%9.4f" % r["ari"],
            "%9d" % r.get("n_mined", -1),
            "%9d" % r.get("n_strict", -1),
            "%9d" % r.get("n_active", -1),
            "%9.4f" % r.get("sep_mean_cos", float("nan")),
            "%9.4f" % r.get("sil_running", float("nan")),
            "%9.0f" % r.get("sep_active", -1),
        ]))

    print("")
    C = int(r.get("sep_n_classes", 3) or 3)
    target = -1.0 / (C - 1) if C > 1 else float("nan")
    zero = sum(1 for x in h if x.get("n_active", 1) == 0)
    print("  [1] epochs with n_active == 0 : %d of %d  %s"
          % (zero, len(h),
             "FROZEN -- the objective is satisfied, tighten alpha"
             if zero else "ok, the objective keeps producing gradient"))
    fr = [x.get("n_strict", 0) / max(1.0, x.get("n_mined", 1)) for x in h]
    print("  [2] min n_strict / n_mined    : %.3f  %s"
          % (min(fr),
             "STARVED -- set strict_semihard=false"
             if min(fr) < 0.2 else "ok"))
    fired = [x["epoch"] for x in h if x.get("sep_active", 0) >= 1]
    print("  [3] separation gate           : %s"
          % ("latched at epoch %d" % fired[0] if fired else
             "NEVER LATCHED -- joint_sep behaved exactly like joint. "
             "Lower train.sep_gate_threshold (running silhouette reached "
             "%.4f)." % max(x.get("sil_running", float("-inf")) for x in h)))
    print("  [4] sep_mean_cos              : %+.4f -> %+.4f  (target %+.4f)"
          % (float(h[0].get("sep_mean_cos", float("nan"))),
             float(h[-1].get("sep_mean_cos", float("nan"))), target))
    print("  [5] train_loss                : %.4f -> %.4f  %s"
          % (h[0]["train_loss"], h[-1]["train_loss"],
             "(IDENTICALLY ZERO -- nothing trained)"
             if max(x["train_loss"] for x in h) == 0.0 else ""))
PYEOF

echo ""
if [ "$RC" -eq 0 ]; then
    echo "DONE. Plots are under $OUT_DIR/figures/"
else
    echo "run_optimization.py exited with $RC -- see the log above."
fi
exit $RC
