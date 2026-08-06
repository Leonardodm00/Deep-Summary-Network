"""
search_dry_run.py
=================

Stage 7: sample the joint condition space, build every config WITHOUT training,
report coverage, MEASURE the architecture size mix, and gate submission on the
extrapolated wall clock.

    # coverage only -- needs no data, runs on a login node in seconds
    python3 search_dry_run.py --config hpc/Config/config_l3c_joint_search.json \\
                              --n 300

    # coverage + the size mix + timed extrapolation (needs the data pipeline)
    python3 search_dry_run.py --config hpc/Config/config_l3c_joint_search.json \\
                              --n 300 --time-points 10 --walltime 144

WHY THIS STAGE EXISTS
---------------------
Two failures it is designed to catch, one of which has already happened once:

  1. A WIRING BUG that only shows up on the cluster. The dry run builds each
     sampled point through the SAME config_from_joint_condition_point the real
     search uses, and preflights the result. It is not a reimplementation, so a
     coordinate that decodes wrongly fails HERE, on a login node, instead of
     after a job has been queued.

  2. AN UNMEASURED WALL CLOCK. The design document's 157 h estimate is the
     depth-4 rate. The search samples depth_exponent over its whole range, and
     parameter count is roughly EXPONENTIAL in depth: measured elsewhere in this
     repo, depth 5 is about 18-20 M parameters against 2.6 M at depth 4. The
     multiplier that mix implies was explicitly recorded as unmeasured. This
     module measures it, two ways that cross-check each other:

       (a) the SIZE MIX, free: build every sampled backbone on the meta device
           and count parameters. No allocation, no training, no data. This
           gives the DISTRIBUTION of model sizes the search will actually draw,
           which is not the same thing as its corners.
       (b) the COST MODEL, cheap: time a handful of sampled points for a couple
           of epochs each and regress seconds/epoch on parameter count. Combined
           with (a) this yields E[seconds/epoch] over the sampling distribution,
           and the multiplier is that expectation divided by the cost at the
           reference architecture the 157 h figure assumed.

     Neither alone is enough: (a) knows the mix but not what it costs, (b) knows
     the cost of a few points but not how often they are drawn.

THE GATE
--------
gate() returns (ok, lines). It is False, and the CLI exits non-zero, when
either coverage is incomplete or the extrapolated wall clock does not fit the
requested walltime with margin. A PBS script should therefore run this and
check the exit code before qsub, rather than trusting a number in a document.

HPC note (hpc-python-compat): pure ASCII.
"""

import argparse
import collections
import json
import os
import sys
import time
import warnings

import numpy as np
from skopt.space import Space

import condition_space as CS
import search as S
from backbone import build_backbone
from config import ExperimentConfig

__all__ = [
    "sample_points",
    "dry_run",
    "measure_cost_model",
    "extrapolate_hours",
    "gate",
]

# the reference architecture the design document's 157 h figure assumed
_REF_DEPTH = 4
# the screening averaged this fraction of its epoch cap before early stopping
_MEAN_EPOCH_FRACTION = 0.55


def _count_params(bcfg):
    """Parameter count without allocating the weights, if that is available."""
    try:
        from run_optimization import _count_params_meta
        return int(_count_params_meta(bcfg))
    except Exception:
        model = build_backbone(bcfg)
        return int(sum(p.numel() for p in model.parameters()))


def sample_points(cfg, n, random_state=0):
    """n raw points from the joint condition space, as plain lists.

    Raw, i.e. BEFORE Pi: the projection is part of what the dry run reports.
    """
    space = S.joint_condition_space(cfg.search, cfg.regularization, cfg.train)
    pts = Space(space).rvs(n_samples=int(n), random_state=int(random_state))
    return [list(p) for p in pts]


def dry_run(cfg, n, random_state=0, count_params=True, verbose=True):
    """Sample n points, project, build, and report. TRAINS NOTHING.

    Returns a report dict. Every field is JSON-serialisable, so the report can
    be archived beside the study it gated.
    """
    points = sample_points(cfg, n, random_state)
    names = S.joint_condition_names()

    conditions = collections.Counter()
    cells = collections.Counter()
    heads = collections.Counter()
    loss_types = collections.Counter()
    minings = collections.Counter()
    depths = collections.Counter()
    params = []
    n_projected = 0
    failures = []

    for i, pt in enumerate(points):
        note = S.annotate_joint_condition_point(pt)
        n_projected += int(note["projected"])
        m, l, s = note["condition"]
        conditions[(m, l, bool(s))] += 1
        cells[note["cell"]] += 1
        heads[(bool(note["head_fusion"]), tuple(note["head_pool_ops"]))] += 1
        loss_types[l] += 1
        minings[m] += 1
        # THE SAME builder the real search uses -- not a reimplementation
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                built = S.config_from_joint_condition_point(cfg, pt)
        except Exception as ex:
            failures.append({"i": i, "cell": note["cell"],
                             "error": "%s: %s" % (type(ex).__name__, ex)})
            continue
        depths[int(built.backbone.depth_exponent)] += 1
        if count_params:
            params.append(_count_params(built.backbone))
        # the built config must agree with what was recorded
        if (built.train.mining_strategy, built.train.loss_type,
                bool(built.train.strict_semihard)) != (m, l, bool(s)):
            failures.append({"i": i, "cell": note["cell"],
                             "error": "built config disagrees with the "
                                      "annotation -- a coordinate decodes wrongly"})

    legal = [tuple(c) for c in CS.legal_conditions()]
    missing_conditions = [c for c in legal if c not in conditions]
    all_heads = [(f, ops) for f in (False, True)
                 for ops in CS.HEAD_POOL_OPS_LEVELS]
    missing_heads = [h for h in all_heads if h not in heads]
    all_cells = set(CS.cell_name(m, l, s, f, ops)
                    for (m, l, s) in legal for (f, ops) in all_heads)
    missing_cells = sorted(all_cells - set(cells))

    report = {
        "n_points": int(n),
        "random_state": int(random_state),
        "n_projected": int(n_projected),
        "n_build_failed": len(failures),
        "failures": failures[:20],
        "n_conditions_seen": len(conditions),
        "n_conditions_total": len(legal),
        "missing_conditions": [list(c) for c in missing_conditions],
        "n_heads_seen": len(heads),
        "n_heads_total": len(all_heads),
        "n_cells_seen": len(cells),
        "n_cells_total": len(all_cells),
        "missing_cells": missing_cells,
        "cell_counts_min": int(min(cells.values())) if cells else 0,
        "cell_counts_median": float(np.median(list(cells.values()))) if cells else 0.0,
        "loss_type_counts": dict(loss_types),
        "mining_counts": dict(minings),
        "depth_counts": dict((str(k), v) for k, v in sorted(depths.items())),
        "coverage_complete": (not missing_conditions and not missing_heads
                              and not failures),
    }
    if params:
        arr = np.asarray(params, dtype=float)
        report["params"] = {
            "min": int(arr.min()), "p25": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)), "p75": float(np.percentile(arr, 75)),
            "max": int(arr.max()), "mean": float(arr.mean()),
        }
        report["params_all"] = [int(v) for v in params]

    if verbose:
        print(format_report(report))
    return report


def format_report(r):
    out = []
    out.append("DRY RUN: %d sampled points, built WITHOUT training"
               % r["n_points"])
    out.append("  projected by Pi   : %d (%.0f%%)"
               % (r["n_projected"], 100.0 * r["n_projected"] / max(1, r["n_points"])))
    out.append("  build failures    : %d" % r["n_build_failed"])
    for f in r["failures"]:
        out.append("      point %d (%s): %s" % (f["i"], f["cell"], f["error"]))
    out.append("")
    out.append("COVERAGE")
    out.append("  conditions : %d of %d" % (r["n_conditions_seen"],
                                            r["n_conditions_total"]))
    for c in r["missing_conditions"]:
        out.append("      MISSING: %s" % (c,))
    out.append("  heads      : %d of %d" % (r["n_heads_seen"], r["n_heads_total"]))
    out.append("  cells      : %d of %d (thinnest cell drawn %d times, median %.1f)"
               % (r["n_cells_seen"], r["n_cells_total"], r["cell_counts_min"],
                  r["cell_counts_median"]))
    if r["missing_cells"]:
        out.append("      MISSING: %s%s"
                   % (", ".join(r["missing_cells"][:8]),
                      " ..." if len(r["missing_cells"]) > 8 else ""))
    out.append("  loss_type  : %s" % (r["loss_type_counts"],))
    out.append("  mining     : %s" % (r["mining_counts"],))
    out.append("  depth      : %s" % (r["depth_counts"],))
    if "params" in r:
        p = r["params"]
        out.append("")
        out.append("SIZE MIX (parameter count over the SAMPLED distribution)")
        out.append("  min %.2f M | p25 %.2f M | median %.2f M | p75 %.2f M | "
                   "max %.2f M | mean %.2f M"
                   % (p["min"] / 1e6, p["p25"] / 1e6, p["median"] / 1e6,
                      p["p75"] / 1e6, p["max"] / 1e6, p["mean"] / 1e6))
        out.append("  the mean is what the wall clock scales with, NOT the "
                   "median: cost is roughly linear in parameters and the")
        out.append("  distribution is heavily right-skewed by the deep corner.")
    out.append("")
    out.append("COVERAGE COMPLETE: %s" % ("YES" if r["coverage_complete"] else "NO"))
    return "\n".join(out)


def measure_cost_model(cfg, splits, device, k=10, random_state=0, epochs=2,
                       verbose=True):
    """Time k sampled points for a few epochs each; regress s/epoch on params.

    Returns {"intercept", "slope_per_param", "points": [...], "r2"}, the cost
    model seconds_per_epoch ~= a + b * n_params.

    epochs is deliberately small: this measures the PER-EPOCH cost, and a
    2-epoch run measures that as well as a 60-epoch one while costing 30x less.
    The first epoch is discarded where more than one is run, because it carries
    one-off costs (cache warm-up, lazy allocation) the later epochs do not.
    """
    from train import train
    from dataclasses import replace as _replace

    points = sample_points(cfg, k, random_state)
    rows = []
    for i, pt in enumerate(points):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trial = S.config_from_joint_condition_point(cfg, pt)
        trial.train = _replace(trial.train, max_epochs=int(epochs),
                               patience=int(epochs), n_seeds=1)
        n_params = _count_params(trial.backbone)
        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _m, hist = train(trial, splits.train, splits.val, device, seed=0)
        except Exception as ex:
            if verbose:
                print("  point %d (%.2f M params) FAILED: %s: %s"
                      % (i, n_params / 1e6, type(ex).__name__, ex))
            continue
        elapsed = time.time() - t0
        # prefer the per-epoch times train() already records
        secs = [float(h["seconds"]) for h in hist if "seconds" in h]
        if len(secs) > 1:
            per_epoch = float(np.mean(secs[1:]))          # drop the first
        elif secs:
            per_epoch = float(secs[0])
        else:
            per_epoch = elapsed / max(1, len(hist))
        rows.append({"i": i, "n_params": int(n_params),
                     "depth_exponent": int(trial.backbone.depth_exponent),
                     "seconds_per_epoch": per_epoch})
        if verbose:
            print("  point %2d: depth %d, %7.2f M params -> %7.2f s/epoch"
                  % (i, trial.backbone.depth_exponent, n_params / 1e6,
                     per_epoch))

    if len(rows) < 2:
        raise RuntimeError(
            "only %d of %d timing points completed; cannot fit a cost model. "
            "Re-run with more --time-points, or investigate the failures."
            % (len(rows), k))
    x = np.asarray([r["n_params"] for r in rows], dtype=float)
    y = np.asarray([r["seconds_per_epoch"] for r in rows], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"intercept": float(intercept), "slope_per_param": float(slope),
            "r2": float(r2), "points": rows}


def extrapolate_hours(cfg, report, cost_model):
    """E[wall clock] over the SAMPLED architecture distribution.

    The size-mix multiplier is E[cost] over that distribution divided by the
    cost at the reference architecture, which is what the design document's
    157 h figure implicitly assumed.
    """
    a = cost_model["intercept"]
    b = cost_model["slope_per_param"]
    params = np.asarray(report["params_all"], dtype=float)
    per_epoch = a + b * params
    mean_per_epoch = float(per_epoch.mean())

    # the reference: the measured cost at the reference depth, or the cost
    # model at the median size when no timed point landed on that depth
    depth_ref = [r for r in cost_model["points"]
                 if r["depth_exponent"] == _REF_DEPTH]
    if depth_ref:
        ref_cost = float(np.mean([r["seconds_per_epoch"] for r in depth_ref]))
    else:
        ref_cost = float(a + b * np.median(params))
    multiplier = mean_per_epoch / ref_cost if ref_cost > 0 else float("nan")

    n_calls = int(S.resolve_n_calls_joint(cfg.search, cfg.regularization))
    n_seeds = int(cfg.train.n_seeds)
    mean_epochs = _MEAN_EPOCH_FRACTION * int(cfg.train.max_epochs)
    hours = n_calls * n_seeds * mean_epochs * mean_per_epoch / 3600.0
    return {"mean_seconds_per_epoch": mean_per_epoch,
            "reference_seconds_per_epoch": ref_cost,
            "size_mix_multiplier": float(multiplier),
            "n_runs": n_calls * n_seeds,
            "mean_epochs_per_run": float(mean_epochs),
            "hours": float(hours)}


# below this, the linear-in-parameters cost model does not describe the timed
# points and the extrapolation must not be trusted. MEASURED failure mode: on
# small models the per-epoch cost is dominated by fixed overhead (data loading,
# miner, Python) rather than by parameter count, the slope term explains almost
# nothing, and the fit still returns a confident-looking number.
_MIN_COST_MODEL_R2 = 0.70


def gate(report, walltime_h=None, extrapolation=None, margin=0.15,
         cost_model=None):
    """(ok, lines). False when coverage is incomplete or the clock does not fit.

    margin is the fraction of the walltime that must remain unused. A study
    that fits with 0% margin does not fit: the size mix is an estimate, the
    epoch fraction is a historical average, and a job killed at the wall loses
    everything after the last checkpoint.

    cost_model, when given, is checked for GOODNESS OF FIT. A low R^2 means the
    timed points are not described by "cost is linear in parameters", so the
    expectation taken over the size mix is not meaningful however precise it
    looks. That is a FAIL, not a footnote: an unreliable estimate used as a
    submission gate is worse than no estimate, because it carries authority.
    """
    ok = True
    lines = ["GATE"]
    if report["coverage_complete"]:
        lines.append("  [PASS] coverage complete: %d/%d conditions, %d/%d heads, "
                     "0 build failures"
                     % (report["n_conditions_seen"], report["n_conditions_total"],
                        report["n_heads_seen"], report["n_heads_total"]))
    else:
        ok = False
        lines.append("  [FAIL] coverage INCOMPLETE (%d/%d conditions, %d/%d "
                     "heads, %d build failures)"
                     % (report["n_conditions_seen"], report["n_conditions_total"],
                        report["n_heads_seen"], report["n_heads_total"],
                        report["n_build_failed"]))
    if extrapolation is None or walltime_h is None:
        ok = False
        lines.append("  [FAIL] wall clock NOT MEASURED. Run with --time-points "
                     "and --walltime before submitting; the design document's "
                     "figure is the depth-4 rate and excludes the size mix.")
    else:
        h = extrapolation["hours"]
        budget = float(walltime_h) * (1.0 - margin)
        lines.append("  size-mix multiplier %.2fx (mean %.1f s/epoch over the "
                     "sampled mix, reference %.1f s/epoch)"
                     % (extrapolation["size_mix_multiplier"],
                        extrapolation["mean_seconds_per_epoch"],
                        extrapolation["reference_seconds_per_epoch"]))
        lines.append("  extrapolated %.0f h for %d runs x %.0f mean epochs"
                     % (h, extrapolation["n_runs"],
                        extrapolation["mean_epochs_per_run"]))
        r2 = None if cost_model is None else float(cost_model.get("r2", float("nan")))
        if r2 is not None and not (r2 >= _MIN_COST_MODEL_R2):
            ok = False
            lines.append("  [FAIL] the cost model does not fit: R^2 = %.3f < "
                         "%.2f. 'Cost is linear in parameters' does not"
                         % (r2, _MIN_COST_MODEL_R2))
            lines.append("         describe the timed points, so the "
                         "expectation over the size mix is not meaningful. "
                         "Time more points,")
            lines.append("         spanning a WIDER range of depths, or model "
                         "the cost on something other than parameter count.")
        if h <= budget:
            lines.append("  [PASS] fits %g h with %.0f%% margin (%.0f h spare)"
                         % (walltime_h, 100 * margin, walltime_h - h))
        else:
            ok = False
            lines.append("  [FAIL] %.0f h exceeds the %.0f h budget "
                         "(%g h walltime less %.0f%% margin)"
                         % (h, budget, walltime_h, 100 * margin))
            lines.append("         gp_minimize is SEQUENTIAL: this cannot be "
                         "split across lanes. Reduce max_epochs, patience, or "
                         "n_calls_joint.")
    lines.append("  SUBMIT: %s" % ("YES" if ok else "NO"))
    return ok, lines


def _splits_for(cfg):
    """The data splits, built exactly as the driver builds them.

    Imported lazily and by NAME rather than reimplemented: if the driver's
    split construction changes, the timing must change with it or the measured
    cost is not the cost of the study.
    """
    import run_optimization as RO
    traces, conditions, fs = RO.build_traces(cfg)
    # [K3] same grouping the real run uses, so the dry run's reported batch
    # geometry is the geometry that will actually be trained on.
    return RO.build_splits(cfg, traces, conditions, fs,
                           cultures=RO.build_cultures(cfg))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 7 dry run and walltime gate")
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=300,
                    help="points to sample; use the real n_calls_joint")
    ap.add_argument("--random-state", type=int, default=0)
    ap.add_argument("--time-points", type=int, default=0,
                    help="time this many sampled points end to end (needs data)")
    ap.add_argument("--time-epochs", type=int, default=2)
    ap.add_argument("--walltime", type=float, default=None,
                    help="requested walltime in hours")
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--no-params", action="store_true",
                    help="skip the parameter counts (faster, no size mix)")
    ap.add_argument("--out", default=None, help="write the report as JSON")
    args = ap.parse_args(argv)

    cfg = ExperimentConfig.from_json(args.config)
    if str(getattr(cfg.search, "search_mode", "staged")) != "joint_conditions":
        print("WARNING: search_mode = %r, not 'joint_conditions'. The dry run "
              "samples the joint condition space regardless, so this reports a "
              "space the study will not actually search."
              % (cfg.search.search_mode,))

    # --no-params skips the parameter counts, but the wall-clock extrapolation
    # regresses seconds-per-epoch on those counts and averages over them. The
    # two flags are mutually exclusive. Refuse HERE, before the timing runs:
    # the old behaviour raised KeyError('params_all') in extrapolate_hours only
    # AFTER every timed point had been trained, throwing the whole measurement
    # away at the last step.
    if args.no_params and args.time_points > 0:
        print("ABORT: --no-params cannot be combined with --time-points. The "
              "wall-clock model is a regression on parameter counts, so the "
              "counts are required. Drop --no-params.")
        return 2

    report = dry_run(cfg, args.n, args.random_state,
                     count_params=not args.no_params)

    extrap = None
    if args.time_points > 0:
        print("")
        print("TIMING %d sampled points at %d epochs each"
              % (args.time_points, args.time_epochs))
        import torch
        splits = _splits_for(cfg)
        cm = measure_cost_model(cfg, splits, torch.device("cpu"),
                                k=args.time_points,
                                random_state=args.random_state + 1000,
                                epochs=args.time_epochs)
        print("  cost model: s/epoch ~= %.3f + %.3e * n_params   (R^2 = %.3f)"
              % (cm["intercept"], cm["slope_per_param"], cm["r2"]))
        extrap = extrapolate_hours(cfg, report, cm)
        report["cost_model"] = cm
        report["extrapolation"] = extrap

    print("")
    ok, lines = gate(report, args.walltime, extrap, args.margin,
                     cost_model=report.get("cost_model"))
    print("\n".join(lines))
    report["gate_ok"] = bool(ok)

    if args.out:
        with open(args.out, "w", encoding="ascii") as fh:
            json.dump(report, fh, indent=2)
        print("\nwrote %s" % args.out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
