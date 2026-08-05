"""
best_from_trials.py
===================

Recover a usable result from a joint condition search that did NOT finish.

This is the tool for the case the walltime arithmetic makes likely: the search
is killed at trial k of n_calls, so run_optimization never reaches the line that
writes config_best.json, and the run leaves a directory with no answer in it.
Given the trials.jsonl that search_persistence flushed, this script writes the
config_best.json that run would have written, so the final training can proceed
against the best configuration actually found.

    python3 best_from_trials.py --run-dir out/l3c_joint_search_e100

    python3 run_optimization.py --config .../config_best.json --skip-search

WHY THE SHORTLIST MATTERS MORE THAN THE ARGMIN
-----------------------------------------------
At n_seeds = 1 the reported winner is a top-k candidate, not the best
configuration: with a within-cell seed sd of s, two configurations differing by
d in the objective are misranked by a single seed with probability
Phi(-d / (s * sqrt(2))). The generator that built these configs says as much,
and says the confirmatory re-fit of the top few at 5 seeds is part of the plan
rather than optional. So this script prints a RANKED TABLE by default and
writes config_best.json for the argmin; --top-k n additionally writes
config_top<i>.json for each of the first n, which are the inputs to that
confirmatory re-fit.

A partial search is therefore not a wasted one. A study killed at k = 200 of
300 still yields a shortlist, which is what the next stage consumes.

WHAT THIS SCRIPT DOES NOT DO
-----------------------------
It does not train, does not re-score, and does not re-rank. It reads the
objectives exactly as the search recorded them. If the trial log mixes two
values of the tie-break weight epsilon the objectives are not comparable and it
REFUSES rather than ranking them anyway.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ExperimentConfig
from search import (
    _JOINT_CONDITION_NAMES, annotate_joint_condition_point,
    config_from_joint_condition_point, joint_condition_space,
)
from search_persistence import (
    STATE_FILENAME, TRIALS_FILENAME, ResumeError, named_to_point, read_trials,
)


def load_base_config(run_dir, explicit=None):
    """The config the interrupted run was launched with.

    run_optimization writes config_input.json into the run directory before any
    training starts, precisely so the inputs of a run survive the run. That is
    the correct base: rebuilding from the ORIGINAL config file risks picking up
    an edit made after the job was submitted.
    """
    if explicit:
        return ExperimentConfig.from_json(explicit)
    candidate = os.path.join(run_dir, "config_input.json")
    if not os.path.exists(candidate):
        raise SystemExit(
            "no config_input.json in %s and no --config given. The trial log "
            "records the sampled COORDINATES, not the base configuration they "
            "were applied to, so one or the other is required." % run_dir)
    return ExperimentConfig.from_json(candidate)


def rank_trials(records):
    """(ranked, n_failed) -- completed trials sorted by objective, best first.

    Refuses on a mixed-epsilon log for the reason in the module docstring.
    FAILED trials are kept in the count but excluded from the ranking: their
    objective is the FAILED sentinel, not a measurement, and promoting a
    sentinel to "best" because every real trial scored worse would be absurd.
    """
    seen_eps = None
    for rec in records:
        eps = rec.get("epsilon", None)
        if eps is None:
            continue
        eps = float(eps)
        if seen_eps is None:
            seen_eps = eps
        elif abs(eps - seen_eps) > 1e-12 * max(1.0, abs(seen_eps)):
            raise ResumeError(
                "the trial log contains two different tie-break weights "
                "epsilon (%r and %r). The recorded objectives are not "
                "comparable and ranking them would be meaningless."
                % (seen_eps, eps))
    n_failed = sum(1 for r in records if bool(r.get("failed", False)))
    ok = [r for r in records
          if not bool(r.get("failed", False))
          and r.get("objective") is not None
          and float(r["objective"]) == float(r["objective"])]
    ok.sort(key=lambda r: float(r["objective"]))
    return ok, n_failed


def describe(rec, cfg):
    """One table row for a trial, using the recorded annotation where present."""
    cell = rec.get("cell")
    if cell is None and "point_raw" in rec:
        try:
            space = joint_condition_space(cfg.search, cfg.regularization,
                                          cfg.train)
            pt = named_to_point(rec["point_raw"], space,
                                list(_JOINT_CONDITION_NAMES))
            cell = annotate_joint_condition_point(pt, cfg.train).get("cell")
        except Exception:
            cell = "?"
    return {"trial": rec.get("trial"),
            "objective": float(rec["objective"]),
            "mean": rec.get("mean"),
            "std": rec.get("std"),
            "cell": cell,
            "projected": bool(rec.get("projected", False)),
            "selected_epochs": rec.get("selected_epochs")}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Recover config_best.json and a ranked shortlist from an "
                    "interrupted joint condition search.")
    ap.add_argument("--run-dir", required=True,
                    help="the run directory, i.e. <out_dir>/<experiment_name>, "
                         "containing trials.jsonl")
    ap.add_argument("--config", default=None,
                    help="base config to apply the winning point to. Defaults "
                         "to <run-dir>/config_input.json.")
    ap.add_argument("--top-k", type=int, default=0,
                    help="also write config_top1.json .. config_top<k>.json, "
                         "the inputs to the confirmatory re-fit at n_seeds > 1")
    ap.add_argument("--show", type=int, default=10,
                    help="how many rows of the ranked table to print")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the table, write nothing")
    args = ap.parse_args(argv)

    run_dir = str(args.run_dir)
    trials_path = os.path.join(run_dir, TRIALS_FILENAME)
    records, n_torn = read_trials(trials_path)
    if not records:
        raise SystemExit("no completed trials found in %s" % trials_path)

    cfg = load_base_config(run_dir, args.config)
    ranked, n_failed = rank_trials(records)
    if not ranked:
        raise SystemExit(
            "%d trial(s) recorded but every one of them FAILED. There is no "
            "configuration to recover." % len(records))

    n_calls_total = None
    state_path = os.path.join(run_dir, STATE_FILENAME)
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="ascii", errors="replace") as fh:
                n_calls_total = (json.load(fh).get("study") or {}).get(
                    "n_calls_total")
        except Exception:
            n_calls_total = None

    print("=" * 72)
    print("recovered from %s" % trials_path)
    print("  %d completed trial(s)%s, %d FAILED, %d torn line(s)"
          % (len(records),
             "" if n_calls_total is None else " of %d planned" % n_calls_total,
             n_failed, n_torn))
    if n_calls_total:
        print("  the search covered %.0f%% of its planned budget"
              % (100.0 * len(records) / float(n_calls_total)))
    print("=" * 72)
    print("%5s  %10s  %8s  %8s  %5s  %s"
          % ("rank", "objective", "mean", "std", "proj", "cell"))
    for i, rec in enumerate(ranked[:max(0, int(args.show))]):
        row = describe(rec, cfg)
        mean = row["mean"]
        std = row["std"]
        print("%5d  %+10.4f  %8s  %8s  %5s  %s"
              % (i + 1, row["objective"],
                 "n/a" if mean is None else "%.4f" % float(mean),
                 "n/a" if std is None else "%.4f" % float(std),
                 "yes" if row["projected"] else "no", row["cell"]))
    print("=" * 72)
    print("REMINDER: at n_seeds = 1 this ranking is a SHORTLIST, not a verdict. "
          "Re-fit the\ntop few at n_seeds > 1 before treating any one of them "
          "as the winner.")

    if args.dry_run:
        return 0

    space = joint_condition_space(cfg.search, cfg.regularization, cfg.train)
    names = list(_JOINT_CONDITION_NAMES)

    def write_for(rec, path):
        if "point_raw" not in rec:
            raise SystemExit(
                "trial %r carries no point_raw, so its configuration cannot be "
                "rebuilt. This log predates per-trial point recording."
                % rec.get("trial"))
        point = named_to_point(rec["point_raw"], space, names)
        cfg_best = config_from_joint_condition_point(cfg, point)
        cfg_best.to_json(path)
        return path

    out = write_for(ranked[0], os.path.join(run_dir, "config_best.json"))
    print("\nwrote %s  (trial %s, objective %+.4f)"
          % (out, ranked[0].get("trial"), float(ranked[0]["objective"])))
    for i in range(min(int(args.top_k), len(ranked))):
        p = write_for(ranked[i], os.path.join(run_dir,
                                              "config_top%d.json" % (i + 1)))
        print("wrote %s  (trial %s, objective %+.4f)"
              % (p, ranked[i].get("trial"), float(ranked[i]["objective"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
