#!/usr/bin/env bash
#
# launch_factorial52.sh
# =====================
#
# Submit the 52-cell factorial with a HARD CAP on how many run at once, so the
# rest of the node/queue stays free for your other work.
#
#     bash Main/hpc/launch_factorial52.sh -n 4                # 4 at a time
#     bash Main/hpc/launch_factorial52.sh -n 4 -c 8           # 4 x 8 cpus = 32
#     bash Main/hpc/launch_factorial52.sh -n 4 --dry-run      # print, submit nothing
#     bash Main/hpc/launch_factorial52.sh -n 4 --mode array   # single array job
#
# TWO MODES, and the default is the one that always works
# -------------------------------------------------------
#   lanes (DEFAULT)
#       Submits N independent CHAINS. Each chain runs its share of the configs
#       one after another, using -W depend=afterany on the previous job in that
#       chain. At most N jobs are ever running, on ANY scheduler that supports
#       job dependencies -- which is all of PBS Pro, OpenPBS and Torque.
#       "afterany" (not "afterok") is deliberate: one cell failing must not
#       strand the remaining cells in its lane.
#
#   array
#       One array job throttled with -W max_run_subjobs=N. Tidier -- one job ID
#       instead of 52 -- but max_run_subjobs only exists on PBS Pro 2021.1 and
#       later. On Torque the equivalent spelling is -t 0-51%N, and on older PBS
#       Pro there is no equivalent at all and the throttle is SILENTLY IGNORED,
#       which is why this is not the default: you would discover it by finding
#       52 jobs running at once.
#
# CPU FOOTPRINT
# -------------
# Total cpus in use = n x ncpus. With the defaults (-n 4 -c 8) that is 32.
# Choose it against what you want to leave free, not against what the node has.
#
# Pure ASCII.

set -uo pipefail

N_PAR=4
NCPUS=8
WALLTIME="08:00:00"
TIER="screen"
MODE="lanes"
DRY=0
JOBSCRIPT=""

usage() {
    sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        -n|--n-parallel)  N_PAR="$2"; shift 2 ;;
        -c|--ncpus)       NCPUS="$2"; shift 2 ;;
        -w|--walltime)    WALLTIME="$2"; shift 2 ;;
        -t|--tier)        TIER="$2"; shift 2 ;;
        --mode)           MODE="$2"; shift 2 ;;
        --dry-run)        DRY=1; shift ;;
        -h|--help)        usage ;;
        *) echo "unknown option: $1"; exit 1 ;;
    esac
done

case "$TIER" in
    screen) CFG_DIR="hpc/Config/factorial";      JOBSCRIPT="hpc/run_factorial52_screen.pbs" ;;
    full)   CFG_DIR="hpc/Config/factorial_full"; JOBSCRIPT="hpc/run_factorial52_screen.pbs" ;;
    *) echo "ABORT: --tier must be 'screen' or 'full'"; exit 1 ;;
esac

# --- locate Main/ -----------------------------------------------------------
if   [ -f "Main/config.py" ]; then MAIN="$PWD/Main"
elif [ -f "config.py" ];      then MAIN="$PWD"
else echo "ABORT: run from the repo root or from Main/"; exit 1; fi
cd "$MAIN"

if [ ! -d "$CFG_DIR" ]; then
    echo "ABORT: $CFG_DIR not found. Generate it first:"
    echo "  python3 hpc/make_factorial_configs.py \\"
    echo "      --base hpc/Config/config_l3c_h_multimean_005.json \\"
    echo "      --out-dir $CFG_DIR --tier $TIER"
    exit 1
fi
N_CFG=$(find "$CFG_DIR" -name "*.json" | wc -l | tr -d ' ')
if [ "$N_CFG" -eq 0 ]; then echo "ABORT: no configs in $CFG_DIR"; exit 1; fi

mkdir -p logs

echo "=============================================================="
echo "tier        : $TIER  ($N_CFG configs)"
echo "mode        : $MODE"
echo "concurrency : $N_PAR job(s) at a time"
echo "resources   : 1 node x $NCPUS cpus each  ->  $((N_PAR * NCPUS)) cpus in use at peak"
echo "walltime    : $WALLTIME per job"
[ "$DRY" = "1" ] && echo "DRY RUN     : nothing will be submitted"
echo "=============================================================="

SELECT="select=1:ncpus=${NCPUS}"

# --------------------------------------------------------------------------- #
if [ "$MODE" = "array" ]; then
    LAST=$((N_CFG - 1))
    CMD="qsub -N l3c_${TIER} -J 0-${LAST} -l ${SELECT} -l walltime=${WALLTIME}"
    CMD="$CMD -W max_run_subjobs=${N_PAR} -j oe -o logs/ -v TIER=${TIER} $JOBSCRIPT"
    echo ""
    echo "  $CMD"
    if [ "$DRY" = "1" ]; then exit 0; fi
    out=$($CMD 2>&1); rc=$?
    echo "  -> $out"
    if [ $rc -ne 0 ]; then
        echo ""
        echo "  Array throttling was REJECTED by this scheduler."
        echo "  max_run_subjobs needs PBS Pro 2021.1+. Re-run with the default:"
        echo "      bash $0 -n $N_PAR -c $NCPUS --tier $TIER"
        exit 1
    fi
    echo ""
    echo "  VERIFY the throttle actually took effect -- on older schedulers it"
    echo "  is silently ignored and all $N_CFG subjobs run at once:"
    echo "      qstat -tJ | head"
    exit 0
fi

# --------------------------------------------------------------------------- #
# lanes: N chains, each running its configs sequentially via depend=afterany
# --------------------------------------------------------------------------- #
echo ""
lane=0
declare -a LAST_ID
while [ "$lane" -lt "$N_PAR" ]; do
    LAST_ID[$lane]=""
    lane=$((lane + 1))
done

idx=0
submitted=0
while [ "$idx" -lt "$N_CFG" ]; do
    lane=$((idx % N_PAR))
    dep=""
    if [ -n "${LAST_ID[$lane]}" ]; then
        dep="-W depend=afterany:${LAST_ID[$lane]}"
    fi
    CMD="qsub -N l3c_${TIER}_${idx} -l ${SELECT} -l walltime=${WALLTIME}"
    CMD="$CMD -j oe -o logs/l3c_${TIER}_${idx}.out -v PBS_ARRAY_INDEX=${idx}"
    CMD="$CMD $dep $JOBSCRIPT"

    if [ "$DRY" = "1" ]; then
        printf "  lane %-2d idx %-3d %s\n" "$lane" "$idx" \
            "$([ -n "$dep" ] && echo "waits for ${LAST_ID[$lane]}" || echo "(head of lane -- starts immediately)")"
        # simulate the job id so the DEPENDENCY CHAIN is visible in a dry run.
        # Without this every job prints "head of lane" and the dry run proves
        # nothing about the throttling.
        LAST_ID[$lane]="DRY.${idx}"
    else
        jid=$(eval "$CMD" 2>&1)
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "  SUBMIT FAILED at idx $idx: $jid"
            echo "  ($submitted job(s) already queued; they will still run.)"
            exit 1
        fi
        jid=$(echo "$jid" | tr -d '[:space:]')
        LAST_ID[$lane]="$jid"
        printf "  lane %-2d idx %-3d -> %s\n" "$lane" "$idx" "$jid"
        submitted=$((submitted + 1))
    fi
    idx=$((idx + 1))
done

echo ""
if [ "$DRY" = "1" ]; then
    echo "  DRY RUN: would submit $N_CFG jobs across $N_PAR lanes"
    echo "  ($(( (N_CFG + N_PAR - 1) / N_PAR )) jobs deep per lane, so at most"
    echo "   $N_PAR run concurrently)"
else
    echo "  submitted $submitted job(s) in $N_PAR lanes"
    echo "  at most $N_PAR run at once; each lane is"
    echo "  $(( (N_CFG + N_PAR - 1) / N_PAR )) job(s) deep"
    echo ""
    echo "  watch:   qstat -u \$USER"
    echo "  cancel:  qselect -u \$USER -N l3c_${TIER} | xargs qdel"
fi
