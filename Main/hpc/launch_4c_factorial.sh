#!/bin/sh
# Launch the 2x2 factorial: {hard, easy_positive} x {single-stage, multi-stage head}.
# Run from Main/ on a LOGIN node (it only submits; it does not train).
#
# Step 1 exists to avoid a RACE: all four jobs read the SAME benchmark and would
# otherwise each try to synthesize and write the trace cache at the same time.
# cache_traces() skips traces already on disk, so populating the cache ONCE up
# front makes all four jobs read-only against it. manifest.json is written
# atomically, but four concurrent writers is a risk not worth taking for the
# ~10 s this costs.

set -e
CFG=hpc/config_4c_hard_single.json

echo "=== 1. populate the shared trace cache (dry run, no training) ==="
python3 run_optimization.py --config "$CFG" --dry-run --verbose

echo ""
echo "=== 2. confirm the geometry before committing 4 jobs ==="
echo "    expect: 16 traces, 4 phenotypes, W = 3000 samples (60 s)"
echo "            windows: train=192 val=48 test=48"
echo "            TOTAL_train_runs: 282"
echo ""
printf "Submit all four jobs? [y/N] "
read yn
case "$yn" in
    [Yy]*) ;;
    *) echo "aborted."; exit 0 ;;
esac

for tag in hard_single hard_multi easypos_single easypos_multi; do
    qsub "hpc/dsn_4c_${tag}.pbs"
done

echo ""
echo "submitted. monitor with:  qstat -u $USER"
echo "follow one job with:      tail -f dsn_4c_hard_single.out"
