#!/usr/bin/env python3
"""
config_from_trial.py -- rebuild the configuration of ANY logged trial.

best_from_trials.py writes configs for the top-k trials RANKED BY OBJECTIVE.
That is the wrong tool when the trial you want was not selected on objective --
for example the highest-eff_rank trial, which is the one that tests whether a
NON-collapsed embedding can also reach ARI = 1.0. This script selects by trial
index, or by any recorded field, and rebuilds that trial's config.

Directive 1: the trial -> config reconstruction itself is NOT reimplemented.
It calls the same search.joint_condition_space / named_to_point /
config_from_joint_condition_point chain best_from_trials.py uses, so a config
produced here is byte-identical to one produced there for the same trial.

USAGE
  # by trial index
  python3 config_from_trial.py --run-dir out/mea_joint_full_lane2 \\
      --trial 65 --out hpc/Config/refit_A.json \\
      --set train.n_seeds=5 --set runtime.experiment_name=refit_A

  # by "best of a field", e.g. the healthiest embedding among perfect-ARI trials
  python3 config_from_trial.py --run-dir out/mea_joint_full_lane0 \\
      --max-field eff_rank --where "ari_mean>=0.999999" \\
      --out hpc/Config/refit_B.json \\
      --set train.n_seeds=5 --set runtime.experiment_name=refit_B

  # just look, write nothing
  python3 config_from_trial.py --run-dir out/mea_joint_full_lane0 \\
      --max-field eff_rank --where "ari_mean>=0.999999" --dry-run

--set takes dotted paths into the config dict and is applied AFTER the
reconstruction, so it can override anything (n_seeds, experiment_name, ...).
Values are parsed as JSON when possible, else kept as strings.

Pure ASCII (hpc-python-compat).
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _ensure_repo(main_dir):
    main_dir = os.path.abspath(main_dir)
    if not os.path.isfile(os.path.join(main_dir, "config.py")):
        raise SystemExit("not a repository Main/ directory (no config.py): %s"
                         % main_dir)
    if main_dir not in sys.path:
        sys.path.insert(0, main_dir)
    return main_dir


def _passes(rec, where):
    """Evaluate a simple `field<op>value` predicate against one record."""
    if not where:
        return True
    for op in (">=", "<=", "==", ">", "<"):
        if op in where:
            field, val = where.split(op, 1)
            field, val = field.strip(), val.strip()
            if field not in rec or rec[field] is None:
                return False
            try:
                lhs, rhs = float(rec[field]), float(val)
            except (TypeError, ValueError):
                lhs, rhs = str(rec[field]), val.strip("'\"")
            return {">=": lhs >= rhs, "<=": lhs <= rhs, "==": lhs == rhs,
                    ">": lhs > rhs, "<": lhs < rhs}[op]
    raise SystemExit("--where must look like field>=value; got %r" % where)


def _set_path(d, dotted, raw):
    try:
        val = json.loads(raw)
    except (TypeError, ValueError):
        val = raw
    parts = dotted.split(".")
    node = d
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = val
    return val


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--main-dir", default=None,
                    help="repository Main/ (default: two levels above run-dir)")
    ap.add_argument("--trial", type=int, help="select this trial index")
    ap.add_argument("--max-field", help="select the trial maximising this field")
    ap.add_argument("--min-field", help="select the trial minimising this field")
    ap.add_argument("--where", help="filter, e.g. 'ari_mean>=0.999999'")
    ap.add_argument("--out", help="config path to write")
    ap.add_argument("--set", action="append", default=[], metavar="a.b=value",
                    help="override a config field; repeatable")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    main_dir = args.main_dir or os.path.dirname(os.path.dirname(run_dir))
    _ensure_repo(main_dir)

    from best_from_trials import load_base_config
    from search import (_JOINT_CONDITION_NAMES, config_from_joint_condition_point,
                        joint_condition_space)
    from search_persistence import TRIALS_FILENAME, named_to_point, read_trials

    # read_trials takes the LOG PATH and returns (records, n_torn) -- not the
    # run directory, and not a bare list. A torn final line is normal after a
    # walltime kill, so it is reported rather than silently dropped.
    log = os.path.join(run_dir, TRIALS_FILENAME)
    if not os.path.exists(log):
        raise SystemExit("no %s in %s" % (TRIALS_FILENAME, run_dir))
    all_recs, n_torn = read_trials(log)
    if n_torn:
        print("NOTE: %d torn line(s) in the log were skipped (normal after a "
              "walltime kill)." % n_torn)
    recs = [r for r in all_recs if not r.get("failed")]
    if not recs:
        raise SystemExit("no non-failed trials in %s" % run_dir)

    chosen = None
    if args.trial is not None:
        for r in recs:
            if int(r.get("trial", -1)) == int(args.trial):
                chosen = r
                break
        if chosen is None:
            raise SystemExit("trial %d not found (or it failed) in %s"
                             % (args.trial, run_dir))
    elif args.max_field or args.min_field:
        field = args.max_field or args.min_field
        pool = [r for r in recs if _passes(r, args.where)
                and r.get(field) is not None]
        if not pool:
            raise SystemExit("no trial matches --where %r with a %r value"
                             % (args.where, field))
        chosen = (max(pool, key=lambda r: float(r[field])) if args.max_field
                  else min(pool, key=lambda r: float(r[field])))
        print("%d of %d trials matched %r"
              % (len(pool), len(recs), args.where or "(no filter)"))
    else:
        raise SystemExit("give --trial, --max-field or --min-field")

    print("selected trial %s: objective=%+.6f ari=%.6f sil=%.6f eff_rank=%.3f "
          "cell=%s" % (chosen.get("trial"), float(chosen["objective"]),
                       float(chosen.get("ari_mean", float("nan"))),
                       float(chosen.get("sil_mean", float("nan"))),
                       float(chosen.get("eff_rank", float("nan"))),
                       chosen.get("cell")))

    if "point_raw" not in chosen:
        raise SystemExit("trial %s carries no point_raw; its configuration "
                         "cannot be rebuilt." % chosen.get("trial"))

    if args.dry_run:
        print("--dry-run: nothing written.")
        return 0
    if not args.out:
        raise SystemExit("--out is required unless --dry-run")

    cfg = load_base_config(run_dir, None)
    space = joint_condition_space(cfg.search, cfg.regularization, cfg.train)
    point = named_to_point(chosen["point_raw"], space,
                           list(_JOINT_CONDITION_NAMES))
    cfg_out = config_from_joint_condition_point(cfg, point)

    d = cfg_out.to_dict()
    d.setdefault("_provenance", {})
    d["_provenance"].update({"run_dir": run_dir, "trial": chosen.get("trial"),
                             "objective": float(chosen["objective"]),
                             "ari_mean": chosen.get("ari_mean"),
                             "eff_rank": chosen.get("eff_rank"),
                             "cell": chosen.get("cell")})
    for kv in args.set:
        if "=" not in kv:
            raise SystemExit("--set needs a.b=value; got %r" % kv)
        k, v = kv.split("=", 1)
        print("  set %s = %r" % (k, _set_path(d, k, v)))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(d, fh, indent=2)
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
