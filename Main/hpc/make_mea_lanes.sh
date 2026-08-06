#!/bin/bash
# ======================================================================
# make_mea_lanes.sh -- expand the base MEA config into N independent lanes
# ======================================================================
#
#   bash Main/hpc/make_mea_lanes.sh [N_LANES]      (default 4)
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
cd "$(dirname "$0")/Config" || { echo "ABORT: cannot find hpc/Config"; exit 2; }
BASE="config_mea_joint_full.json"
[ -f "$BASE" ] || { echo "ABORT: $BASE not found in $PWD"; exit 2; }

for L in $(seq 0 $((N - 1))); do
    python3 - "$BASE" "$L" <<'EOF'
import json, sys
base, lane = sys.argv[1], int(sys.argv[2])
d = json.load(open(base))
d["search"]["gp_random_state"] = lane
d["runtime"]["experiment_name"] = "mea_joint_full_lane%d" % lane
out = "config_mea_joint_full_lane%d.json" % lane
with open(out, "w", encoding="ascii") as fh:
    json.dump(d, fh, indent=2); fh.write("\n")
print("  lane %d -> gp_random_state=%d  experiment_name=%s"
      % (lane, lane, d["runtime"]["experiment_name"]))
EOF
done

echo ""
ls -1 config_mea_joint_full_lane*.json
