#!/bin/sh
# run_wiring_checks.sh -- the verification ladder for the C1-C6 wiring.
#
# NOTE: the C5 rung (the latent-factor retention metric) was DELETED in
# Change 2 of the v3 handoff. Its module and its suite are gone; what runs
# in its place is the deletion guard, which asserts they stay gone. The
# latent ground-truth artefact is still written (see Rung 2) -- only its
# consumer was removed.
#
# Run from the repository's Main/ directory:
#     cd Main && sh run_wiring_checks.sh
#
# Rung 1 (no torch needed) runs first and is cheap. Rung 2 is the ACCEPTANCE
# GATE for this wiring and needs torch + pytorch_metric_learning installed.
# Rung 3 (a real search, and the C6 two-miner ablation) is cluster-only and is
# NOT run here -- the archived comparable run took about 39 h on CPU.
#
# The script stops at the first failure so a later pass cannot hide an earlier
# break. Pure ASCII (hpc-python-compat).

set -e

PY=${PY:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
export PYTHONPATH="$HERE:$PYTHONPATH"

echo "=================================================================="
echo " RUNG 0 -- HPC safety gate (byte scan + compile)"
echo "=================================================================="
$PY - <<'EOF'
import pathlib, sys
files = sorted(list(pathlib.Path(".").glob("*.py"))
               + list(pathlib.Path("Smoke_Tests").glob("*.py"))
               + list(pathlib.Path("hpc").glob("*.json")))
bad_any = False
for f in files:
    bad = [(i + 1, hex(b)) for i, b in enumerate(f.read_bytes()) if b > 127]
    if bad:
        bad_any = True
        print("NON-ASCII  %-50s %s" % (f, bad[:6]))
print("scanned %d files" % len(files))
if bad_any:
    print("FAIL: non-ASCII bytes found -- these crash on an ASCII-locale node")
    sys.exit(1)
print("OK -- every file is pure ASCII")
EOF
$PY -m py_compile *.py Smoke_Tests/*.py
echo "OK -- every module compiles"
echo ""

echo "=================================================================="
echo " RUNG 1 -- module checks (no torch required)"
echo "=================================================================="
echo "--- the handoff's own generator + objective checks (expect 21/21) ---"
$PY smoke_test_latent_and_objective.py
echo ""
echo "--- C2 + C3: epoch selection, epsilon guarantee, budget split ---"
$PY Smoke_Tests/smoke_test_objective_wiring.py
echo ""
echo "--- the deletion guard: removed modules must stay removed ---"
$PY Smoke_Tests/smoke_test_removed_modules.py
echo ""

echo "=================================================================="
echo " RUNG 2 -- ACCEPTANCE GATE (requires torch)"
echo "=================================================================="
$PY -c "import torch, pytorch_metric_learning; print('torch', torch.__version__)"
echo ""
echo "--- pre-existing suite: nothing here may regress ---"
for t in smoke_test_config smoke_test_synthetic_config smoke_test_data_pipeline \
         smoke_test_data_splits smoke_test_backbone smoke_test_metrics \
         smoke_test_augmentation smoke_test_checkpoint smoke_test_train \
         smoke_test_search smoke_test_evaluate smoke_test_inference \
         smoke_test_run_optimization smoke_test_end_to_end; do
    echo "  >>> $t"
    $PY "Smoke_Tests/$t.py" > "/tmp/${t}.log" 2>&1 \
        && echo "      PASS" \
        || { echo "      FAIL -- see /tmp/${t}.log"; tail -20 "/tmp/${t}.log"; exit 1; }
done
echo ""
echo "--- C1 wiring: dispatch, fingerprint, stale-cache refusal, artifacts ---"
$PY Smoke_Tests/smoke_test_latent_wiring.py
echo ""
echo "--- C2 drift test: recomputed e* vs train.py's own best_epoch ---"
$PY Smoke_Tests/smoke_test_selected_epoch.py
echo ""
echo "--- driver dry run on the new latent benchmark ---"
$PY run_optimization.py --config hpc/config_latent_3class_hard.json \
    --dry-run --verbose \
    --out-dir /tmp/dsn_dryrun --cache-dir /tmp/dsn_dryrun_cache
echo ""
echo "--- the dry run must have written the latent ground truth ---"
ls -l /tmp/dsn_dryrun/latent_3class_hard/latent_ground_truth.json
echo ""

echo "=================================================================="
echo " ALL RUNGS PASSED"
echo "=================================================================="
echo "Rung 3 (cluster only, NOT run here):"
echo "  qsub the two C6 ablation configs, which differ in exactly one field:"
echo "    hpc/config_latent_3class_hard.json      (mining_strategy = hard)"
echo "    hpc/config_latent_3class_easypos.json   (mining_strategy = easy_positive)"
echo "  Then compare ARI and eff_rank on the held-out split."
