#!/bin/sh
# Empirical thread-count sweep, using the d5 (20M param) corner: the
# architecture most likely to benefit from intra-op parallelism, so this is
# an UPPER BOUND on what threading buys you -- the d2 corner (35k params)
# will benefit less, possibly not at all.
#
# Each config runs 4 train() calls x 3 epochs = 12 epochs. Reuses the trace
# cache from earlier timing runs, so each is ~30-90 s depending on threads.
#
# Run from Main/, inside a job with the cores you're testing allocated
# (an interactive qsub -I with enough ncpus, or inside the PBS script itself).

set -e
CACHE=/tmp/timing_cache

for n in 4 8 16 32 48; do
    export OMP_NUM_THREADS=$n
    export MKL_NUM_THREADS=$n
    echo "=================================================================="
    echo " torch_threads = $n   (OMP_NUM_THREADS = MKL_NUM_THREADS = $n)"
    echo "=================================================================="
    /usr/bin/time -p python3 run_optimization.py \
        --config "hpc/config_timing_threads_${n}.json" \
        --out-dir "/tmp/tt_${n}" --cache-dir "$CACHE" \
        2>&1 | grep -E "^\[train\] epoch|^real|^user|^sys"
    echo ""
done
