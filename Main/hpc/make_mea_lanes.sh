#!/bin/bash
# ======================================================================
# make_mea_lanes.sh -- expand the base MEA config into N independent lanes
# ======================================================================
#
#   bash Main/hpc/make_mea_lanes.sh [N_LANES] [THREADS_PER_LANE...]
#
# THREADS_PER_LANE is optional and positional: lane i takes the i-th value,
# and any lane past the end of the list keeps the base config's
# runtime.torch_threads. So
#
#     bash make_mea_lanes.sh 5 48 48 48 48 192
#
# makes five lanes where lane 4 is a 192-thread lane and the rest stay at 48.
# torch_threads MUST match the ncpus the job actually requests --
# run_mea_joint_search.pbs aborts on a mismatch rather than oversubscribing.
#
# Changing threads does NOT change the science: runtime.seed is fixed across
# lanes, so the data, splits and class geometry are identical; only
# gp_random_state differs. A wider lane explores the SAME space faster, and
# its trials pool with the others at the end exactly as a 48-thread lane's do.
#
# There is no --gp-random-state CLI override: the GP seed lives only in the
# config file. Lanes that differ only by --experiment-name would run a
# bit-identical study and you would pay Nx to learn nothing. So BOTH knobs
# move together here: gp_random_state AND experiment_name.
#
# runtime.seed stays fixed across lanes on purpose -- same data, same
# splits, same class geometry; only the GP's trajectory differs. That is
# what makes the lanes poolable at the end.
#
# Pure ASCII, LF only (hpc-python-compat).
# ======================================================================
set -uo pipefail

N="${1:-4}"
shift || true
THREADS=("$@")          # positional: THREADS[i] is lane i's torch_threads

cd "$(dirname "$0")/Config" || { echo "ABORT: cannot find hpc/Config"; exit 2; }
BASE="config_mea_joint_full.json"
[ -f "$BASE" ] || { echo "ABORT: $BASE not found in $PWD"; exit 2; }

for L in $(seq 0 $((N - 1))); do
    # empty string -> keep the base config's torch_threads for this lane
    T="${THREADS[$L]:-}"
    python3 - "$BASE" "$L" "$T" <<'EOF'
import json, sys
base, lane, threads = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d = json.load(open(base))
d["search"]["gp_random_state"] = lane
d["runtime"]["experiment_name"] = "mea_joint_full_lane%d" % lane
if threads:
    d["runtime"]["torch_threads"] = int(threads)
out = "config_mea_joint_full_lane%d.json" % lane
with open(out, "w", encoding="ascii") as fh:
    json.dump(d, fh, indent=2); fh.write("\n")
print("  lane %d -> gp_random_state=%d  threads=%d  experiment_name=%s"
      % (lane, lane, d["runtime"]["torch_threads"],
         d["runtime"]["experiment_name"]))
EOF
done

echo ""
ls -1 config_mea_joint_full_lane*.json