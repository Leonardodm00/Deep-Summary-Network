#!/usr/bin/env python3
"""
make_refit_batch.py -- build N re-fit configs + a submit script, with COST.

Why this exists rather than a loop over config_from_trial.py: raising n_seeds
and max_epochs together multiplies cost, and the per-epoch cost of these
candidates spans more than an order of magnitude (depth_exponent 2 vs 5 was
38 s/epoch vs 508 s/epoch on the MEA study). Submitting five deep candidates at
5 seeds x 200 epochs with a default walltime wastes five queue slots and
returns nothing. So every candidate's walltime is PREDICTED from its own
logged timing before anything is submitted.

The trial -> config reconstruction is NOT reimplemented: it calls the same
search.config_from_joint_condition_point chain best_from_trials.py uses.

USAGE
  python3 make_refit_batch.py \\
      --run-root out --experiment mea_joint_full \\
      --top-k 5 --n-seeds 5 --max-epochs 200 \\
      --include lane0:85 \\
      --config-dir hpc/Config --submit submit_refit_mea.sh

  # see the cost table without writing anything
  python3 make_refit_batch.py --run-root out --experiment mea_joint_full \\
      --top-k 5 --n-seeds 5 --max-epochs 200 --dry-run

COST MODEL (stated, because it drives the walltime you will queue on)
  For each candidate trial t, from its own log entry:
      dt_t          wall seconds that trial took (diff of wall_elapsed_s,
                    with resume boundaries handled)
      epochs_t      epochs actually TRAINED = min(e*_t + P_old, E_old)
      s_per_epoch   = dt_t / epochs_t
  Predicted worst case for the re-fit:
      seconds = n_seeds * E_new * s_per_epoch * margin
  E_new, not the observed e*, because early stopping may not trigger before the
  new cap -- that is the whole point of raising it. This is deliberately
  PESSIMISTIC: a job that finishes early costs nothing, a job killed at the
  walltime costs everything.

DEDUPE
  --dedupe-by cell (default) keeps the best trial per condition cell, so the
  batch spans different conditions rather than five neighbours in one basin.
  --dedupe-by none takes the raw top-k.

Pure ASCII (hpc-python-compat).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np


def _ensure_repo(main_dir):
    main_dir = os.path.abspath(main_dir)
    if not os.path.isfile(os.path.join(main_dir, "config.py")):
        raise SystemExit("not a repository Main/ (no config.py): %s" % main_dir)
    if main_dir not in sys.path:
        sys.path.insert(0, main_dir)
    return main_dir


def load_lane(run_root, experiment, lane, read_trials, TRIALS_FILENAME):
    d = os.path.join(run_root, "%s_lane%d" % (experiment, lane))
    p = os.path.join(d, TRIALS_FILENAME)
    if not os.path.exists(p):
        return d, []
    recs, n_torn = read_trials(p)
    if n_torn:
        print("  lane%d: %d torn line(s) skipped" % (lane, n_torn))
    for r in recs:
        r["_lane"] = lane
        r["_dir"] = d
    return d, recs


def add_timing(recs, e_max_old, patience_old):
    """Attach s_per_epoch to each record, from its own wall-clock delta.

    wall_elapsed_s restarts near 0 at every resume, so a negative step marks a
    segment boundary and the trial at that boundary took its OWN wall value.
    Dropping it instead would lose the first trial of every later segment.
    """
    recs = sorted(recs, key=lambda r: int(r.get("trial", 0)))
    wall = np.array([float(r.get("wall_elapsed_s") or 0.0) for r in recs])
    steps = np.diff(np.r_[0.0, wall])
    boundaries = np.flatnonzero(steps < 0)
    steps[boundaries] = wall[boundaries]
    for r, dt in zip(recs, steps):
        se = r.get("selected_epochs") or []
        estar = int(se[0]) if se else None
        if estar is None or dt <= 0:
            r["_s_per_epoch"] = None
            continue
        epochs = min(estar + int(patience_old), int(e_max_old))
        r["_s_per_epoch"] = float(dt) / max(1, epochs)
    return recs


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default="out")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--main-dir", default=".")
    ap.add_argument("--lanes", default="0,1,2,3")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--dedupe-by", choices=("cell", "none"), default="cell")
    ap.add_argument("--include", action="append", default=[],
                    metavar="laneL:TRIAL",
                    help="force-include a candidate, e.g. lane0:85; repeatable")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=None,
                    help="override train.patience (default: keep the study's)")
    ap.add_argument("--margin", type=float, default=1.35,
                    help="walltime safety factor on the predicted cost")
    ap.add_argument("--max-walltime-h", type=float, default=168.0,
                    help="refuse to write a job needing more than this")
    ap.add_argument("--ncpus", type=int, default=48)
    ap.add_argument("--config-dir", default="hpc/Config")
    ap.add_argument("--submit", default="submit_refit_batch.sh")
    ap.add_argument("--pbs", default="hpc/run_refit.pbs")
    ap.add_argument("--tag-prefix", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    main_dir = _ensure_repo(args.main_dir)
    from best_from_trials import load_base_config
    from search import (_JOINT_CONDITION_NAMES, config_from_joint_condition_point,
                        joint_condition_space)
    from search_persistence import TRIALS_FILENAME, named_to_point, read_trials

    lanes = [int(x) for x in args.lanes.split(",") if x.strip() != ""]
    all_recs, lane_dirs = [], {}
    for L in lanes:
        d, recs = load_lane(args.run_root, args.experiment, L, read_trials,
                            TRIALS_FILENAME)
        lane_dirs[L] = d
        all_recs += recs
    if not all_recs:
        raise SystemExit("no trials found for %s under %s"
                         % (args.experiment, args.run_root))

    cfg0 = load_base_config(lane_dirs[lanes[0]], None)
    e_max_old = int(cfg0.train.max_epochs)
    patience_old = int(cfg0.train.patience)
    patience_new = args.patience if args.patience is not None else patience_old
    print("study: E_max=%d patience=%d -> re-fit E_max=%d patience=%d n_seeds=%d"
          % (e_max_old, patience_old, args.max_epochs, patience_new, args.n_seeds))

    by_lane = {}
    for r in all_recs:
        by_lane.setdefault(r["_lane"], []).append(r)
    timed = []
    for L, recs in by_lane.items():
        timed += add_timing(recs, e_max_old, patience_old)
    ok = [r for r in timed if not r.get("failed")]
    print("pooled non-failed trials: %d" % len(ok))

    ranked = sorted(ok, key=lambda r: float(r["objective"]))
    chosen, seen_cells = [], set()
    for r in ranked:
        if len(chosen) >= args.top_k:
            break
        if args.dedupe_by == "cell":
            c = r.get("cell")
            if c in seen_cells:
                continue
            seen_cells.add(c)
        chosen.append(r)

    for spec in args.include:
        try:
            lane_s, trial_s = spec.split(":")
            L, tr = int(lane_s.replace("lane", "")), int(trial_s)
        except ValueError:
            raise SystemExit("--include must look like lane0:85; got %r" % spec)
        hit = [r for r in ok if r["_lane"] == L and int(r["trial"]) == tr]
        if not hit:
            raise SystemExit("--include %s: no such non-failed trial" % spec)
        if not any(r["_lane"] == L and int(r["trial"]) == tr for r in chosen):
            chosen.append(hit[0])
            print("force-included lane%d trial %d" % (L, tr))

    # median s_per_epoch as a fallback for any candidate whose own timing is
    # missing; stated rather than silently substituted.
    spes = [r["_s_per_epoch"] for r in ok if r["_s_per_epoch"]]
    median_spe = float(np.median(spes)) if spes else None

    prefix = args.tag_prefix or ("refit_%s" % args.experiment)
    rows, jobs, total_h = [], [], 0.0
    for rank, r in enumerate(chosen, 1):
        L, tr = r["_lane"], int(r["trial"])
        spe = r["_s_per_epoch"]
        est_note = ""
        if not spe:
            spe = median_spe
            est_note = " (timing missing; pooled median used)"
        secs = args.n_seeds * args.max_epochs * spe * args.margin
        hours = secs / 3600.0
        tag = "%s_r%d_l%d_t%d" % (prefix, rank, L, tr)
        rows.append({
            "rank": rank, "lane": L, "trial": tr,
            "objective": float(r["objective"]),
            "ari": r.get("ari_mean"), "eff_rank": r.get("eff_rank"),
            "cell": r.get("cell"),
            "depth": (r.get("point_raw") or {}).get("depth_exponent"),
            "s_per_epoch": spe, "est_h": hours, "tag": tag, "note": est_note,
        })
        total_h += hours

    print("\n%-4s %-5s %-6s %-11s %-8s %-9s %-28s %-6s %-9s %s"
          % ("rank", "lane", "trial", "objective", "ari", "eff_rank", "cell",
             "depth", "s/epoch", "est walltime"))
    for d in rows:
        print("%-4d %-5d %-6d %-11.6f %-8s %-9s %-28s %-6s %-9.1f %.1f h%s"
              % (d["rank"], d["lane"], d["trial"], d["objective"],
                 ("%.4f" % d["ari"]) if d["ari"] is not None else "-",
                 ("%.3f" % d["eff_rank"]) if d["eff_rank"] is not None else "-",
                 str(d["cell"])[:28], str(d["depth"]), d["s_per_epoch"],
                 d["est_h"], d["note"]))
    print("\ntotal predicted compute: %.1f h across %d job(s) "
          "(they run in PARALLEL, so wall-clock is the max: %.1f h)"
          % (total_h, len(rows), max(d["est_h"] for d in rows)))

    over = [d for d in rows if d["est_h"] > args.max_walltime_h]
    if over:
        print("\nWARNING: %d candidate(s) exceed --max-walltime-h=%.0f:"
              % (len(over), args.max_walltime_h))
        for d in over:
            print("  rank %d (lane%d trial %d, depth %s): %.1f h"
                  % (d["rank"], d["lane"], d["trial"], d["depth"], d["est_h"]))
        print("  Options: lower --max-epochs, lower --n-seeds, or drop these\n"
              "  candidates. They are written anyway, with their own walltime;\n"
              "  the queue may refuse them.")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    os.makedirs(args.config_dir, exist_ok=True)
    space = joint_condition_space(cfg0.search, cfg0.regularization, cfg0.train)
    for d, r in zip(rows, chosen):
        point = named_to_point(r["point_raw"], space, list(_JOINT_CONDITION_NAMES))
        cfg_out = config_from_joint_condition_point(
            load_base_config(lane_dirs[r["_lane"]], None), point)
        cd = cfg_out.to_dict()
        cd["train"]["n_seeds"] = int(args.n_seeds)
        cd["train"]["max_epochs"] = int(args.max_epochs)
        cd["train"]["patience"] = int(patience_new)
        cd["runtime"]["experiment_name"] = d["tag"]
        cd["_provenance"] = {
            "run_dir": r["_dir"], "lane": r["_lane"], "trial": r["trial"],
            "search_objective": float(r["objective"]),
            "search_ari": r.get("ari_mean"), "search_eff_rank": r.get("eff_rank"),
            "cell": r.get("cell"), "rank_in_batch": d["rank"],
            "predicted_walltime_h": round(d["est_h"], 2),
            "s_per_epoch_observed": round(d["s_per_epoch"], 2),
        }
        p = os.path.join(args.config_dir, d["tag"] + ".json")
        with open(p, "w") as fh:
            json.dump(cd, fh, indent=2)
        d["config"] = p
        jobs.append(d)
        print("wrote %s" % p)

    lines = ["#!/bin/bash",
             "# Generated by make_refit_batch.py -- do not hand-edit; regenerate.",
             "# Walltime per job is PREDICTED from that candidate's own logged",
             "# seconds-per-epoch: n_seeds * max_epochs * s_per_epoch * %.2f."
             % args.margin,
             "set -euo pipefail",
             'cd "$(dirname "$0")"',
             ""]
    for d in jobs:
        wh = int(math.ceil(d["est_h"]))
        lines += [
            "# rank %d: lane%d trial %d | cell %s | depth %s | %.0f s/epoch"
            % (d["rank"], d["lane"], d["trial"], d["cell"], d["depth"],
               d["s_per_epoch"]),
            "qsub -l select=1:ncpus=%d -l walltime=%d:00:00 \\" % (args.ncpus, wh),
            "     -v CFG=%s -o %s.out %s" % (d["config"], d["tag"], args.pbs),
            ""]
    with open(args.submit, "w") as fh:
        fh.write("\n".join(lines))
    os.chmod(args.submit, 0o755)
    print("\nwrote %s  (%d job(s)) -- review it, then: bash %s"
          % (args.submit, len(jobs), args.submit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
