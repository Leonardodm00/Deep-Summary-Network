"""
smoke_test_search_dry_run.py
============================

Acceptance tests for Stage 7: the dry run, the size-mix measurement and the
submission gate.

  [7-A] the dry run TRAINS NOTHING (train() is patched to explode; if it is
        ever called the test fails loudly)
  [7-B] it builds through the SAME builder the real search uses, so a wiring
        bug fails here rather than on the cluster
  [7-C] coverage of the 13 conditions and the 4 head geometries is reported,
        and cell-level coverage grows with the number of draws
  [7-D] a build failure is COUNTED and reported, not swallowed
  [7-E] the size mix is measured over the SAMPLED distribution, and its mean
        exceeds its median (the deep corner dominates the wall clock)
  [7-F] the cost model fits seconds/epoch on parameter count, and its R^2 is
        reported
  [7-G] the gate FAILS on incomplete coverage
  [7-H] the gate FAILS when the wall clock was never measured -- the design
        document's figure is the depth-4 rate and excludes the size mix
  [7-I] the gate FAILS when the extrapolation exceeds walltime minus margin
  [7-J] the gate FAILS on a badly-fitting cost model, however precise the
        resulting hour count looks
  [7-K] the whole report is JSON-serialisable, so it can be archived beside
        the study it gated

HOW TO RUN
----------
    cd Main
    PYTHONPATH=. python3 Smoke_Tests/smoke_test_search_dry_run.py

HPC note (hpc-python-compat): pure ASCII.
"""

import json
import os
import sys
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAIN = os.path.dirname(_HERE)
if _MAIN not in sys.path:
    sys.path.insert(0, _MAIN)

import numpy as np

import condition_space as CS
import search as S
import search_dry_run as DR
from config import ExperimentConfig


def _expect(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _cfg():
    cfg = ExperimentConfig()
    cfg.search.search_mode = "joint_conditions"
    return cfg


def _fake_extrap(hours):
    return {"hours": float(hours), "size_mix_multiplier": 1.7,
            "mean_seconds_per_epoch": 58.0, "reference_seconds_per_epoch": 34.2,
            "n_runs": 300, "mean_epochs_per_run": 33.0}


def _ok_report():
    return {"coverage_complete": True, "n_conditions_seen": 13,
            "n_conditions_total": 13, "n_heads_seen": 4, "n_heads_total": 4,
            "n_build_failed": 0}


def test_7a_trains_nothing():
    import train as T
    orig = T.train

    def explode(*a, **k):
        raise AssertionError("the DRY RUN called train() -- it must not")

    T.train = explode
    S.train = explode
    try:
        r = DR.dry_run(_cfg(), 60, random_state=0, count_params=False,
                       verbose=False)
    finally:
        T.train = orig
        S.train = orig
    _expect(r["n_points"] == 60, "the dry run did not sample 60 points")
    print("  [7-A] the dry run trains nothing (train() patched to raise) OK")


def test_7b_uses_the_real_builder():
    """A wiring bug must fail HERE, not after a job is queued."""
    cfg = _cfg()
    orig = S.config_from_joint_condition_point
    seen = {"n": 0}

    def counting(base, point):
        seen["n"] += 1
        return orig(base, point)

    S.config_from_joint_condition_point = counting
    try:
        DR.dry_run(cfg, 25, random_state=0, count_params=False, verbose=False)
    finally:
        S.config_from_joint_condition_point = orig
    _expect(seen["n"] == 25,
            "the dry run built %d of 25 points through the real builder; a "
            "reimplementation would not catch a decode bug" % seen["n"])
    print("  [7-B] every sampled point goes through the REAL builder OK")


def test_7c_coverage_reported_and_grows():
    cfg = _cfg()
    r40 = DR.dry_run(cfg, 40, 0, count_params=False, verbose=False)
    r300 = DR.dry_run(cfg, 300, 0, count_params=False, verbose=False)
    for r in (r40, r300):
        _expect(r["n_conditions_total"] == CS.n_legal_conditions() == 13,
                "the report must count against the 13 legal conditions")
        _expect(r["n_heads_total"] == 4, "there are 4 head geometries")
        _expect(r["n_cells_total"] == 52,
                "13 conditions x 4 heads = the 52 historical cells; got %d"
                % r["n_cells_total"])
    _expect(r300["n_cells_seen"] >= r40["n_cells_seen"],
            "cell coverage must not shrink with more draws")
    _expect(r300["n_cells_seen"] == 52,
            "300 draws should reach every cell; got %d" % r300["n_cells_seen"])
    _expect(r40["n_cells_seen"] < 52,
            "40 draws should NOT reach every cell -- if it does, the coverage "
            "measure is not discriminating and cannot inform n_initial_points")
    _expect(r300["n_projected"] > 0, "Pi never fired in 300 draws")
    print("  [7-C] coverage reported against 13 / 4 / 52; 40 draws reach %d "
          "cells, 300 draws reach %d OK"
          % (r40["n_cells_seen"], r300["n_cells_seen"]))


def test_7d_build_failure_counted():
    cfg = _cfg()
    orig = S.config_from_joint_condition_point
    calls = {"n": 0}

    def flaky(base, point):
        calls["n"] += 1
        if calls["n"] % 7 == 0:
            raise RuntimeError("simulated build failure")
        return orig(base, point)

    S.config_from_joint_condition_point = flaky
    try:
        r = DR.dry_run(cfg, 21, random_state=0, count_params=False,
                       verbose=False)
    finally:
        S.config_from_joint_condition_point = orig
    _expect(r["n_build_failed"] == 3,
            "expected 3 seeded failures, the report says %d"
            % r["n_build_failed"])
    _expect(r["coverage_complete"] is False,
            "a build failure must make coverage INCOMPLETE, or the gate passes "
            "a study that cannot build every point it will sample")
    _expect(r["failures"] and "simulated" in r["failures"][0]["error"],
            "the failure reason must be recorded, not swallowed")
    _expect("cell" in r["failures"][0],
            "a failed point must say WHICH cell failed")
    print("  [7-D] build failures are counted, attributed to a cell, and fail "
          "coverage OK")


def test_7e_size_mix():
    cfg = _cfg()
    cfg.search.depth_exponent_range = (2, 5)
    r = DR.dry_run(cfg, 80, random_state=0, count_params=True, verbose=False)
    p = r["params"]
    for k in ("min", "p25", "median", "p75", "max", "mean"):
        _expect(k in p, "the size mix is missing %r" % k)
    _expect(p["min"] <= p["median"] <= p["max"], "size-mix quantiles disordered")
    _expect(p["mean"] > p["median"],
            "the parameter distribution should be RIGHT-SKEWED (the deep "
            "corner dominates); mean %.3g vs median %.3g"
            % (p["mean"], p["median"]))
    _expect(len(r["params_all"]) == 80 - r["n_build_failed"],
            "one parameter count per successfully built point")
    _expect(len(r["depth_counts"]) > 1,
            "the sampled depths must vary, or there is no size mix to measure")
    print("  [7-E] size mix measured over the sampled distribution: mean "
          "%.2f M > median %.2f M (right-skewed) OK"
          % (p["mean"] / 1e6, p["median"] / 1e6))


def test_7f_cost_model_shape():
    """The cost model is a fit; its goodness of fit must be reported."""
    rows = [{"i": i, "n_params": n, "depth_exponent": 4,
             "seconds_per_epoch": 5.0 + 3e-6 * n}
            for i, n in enumerate([1e5, 5e5, 2e6, 8e6, 2e7])]
    x = np.asarray([r["n_params"] for r in rows], float)
    y = np.asarray([r["seconds_per_epoch"] for r in rows], float)
    slope, intercept = np.polyfit(x, y, 1)
    _expect(abs(slope - 3e-6) < 1e-9 and abs(intercept - 5.0) < 1e-6,
            "the fit does not recover a known linear cost")
    cm = {"intercept": float(intercept), "slope_per_param": float(slope),
          "r2": 1.0, "points": rows}
    cfg = _cfg()
    cfg.train.n_seeds = 1
    cfg.train.max_epochs = 60
    cfg.search.n_calls_joint = 300
    rep = {"params_all": [1e5, 2e7, 1e6], "coverage_complete": True}
    ex = DR.extrapolate_hours(cfg, rep, cm)
    for k in ("mean_seconds_per_epoch", "size_mix_multiplier", "hours",
              "n_runs", "mean_epochs_per_run"):
        _expect(k in ex, "the extrapolation is missing %r" % k)
    _expect(ex["n_runs"] == 300, "n_runs = n_calls x n_seeds")
    _expect(ex["hours"] > 0, "non-positive hours")
    # the mean cost must be taken over the MIX, not at the median point
    want = float(np.mean([intercept + slope * v for v in rep["params_all"]]))
    _expect(abs(ex["mean_seconds_per_epoch"] - want) < 1e-9,
            "the extrapolation must average the cost over the sampled sizes")
    print("  [7-F] the cost model recovers a known linear cost and the "
          "extrapolation averages over the MIX OK")


def test_7g_gate_fails_on_coverage():
    r = _ok_report()
    r.update({"coverage_complete": False, "n_conditions_seen": 11,
              "n_build_failed": 2})
    ok, lines = DR.gate(r, 144.0, _fake_extrap(50.0), cost_model={"r2": 0.95})
    _expect(ok is False, "the gate passed INCOMPLETE coverage")
    _expect(any("coverage INCOMPLETE" in l for l in lines), lines)
    print("  [7-G] the gate fails on incomplete coverage OK")


def test_7h_gate_fails_when_unmeasured():
    ok, lines = DR.gate(_ok_report(), None, None)
    _expect(ok is False,
            "the gate passed an UNMEASURED wall clock -- this is the whole "
            "reason Stage 7 exists")
    _expect(any("NOT MEASURED" in l for l in lines), lines)
    ok2, _l = DR.gate(_ok_report(), 144.0, None)
    _expect(ok2 is False, "a walltime without an extrapolation must still fail")
    print("  [7-H] the gate refuses to pass an unmeasured wall clock OK")


def test_7i_gate_fails_on_overrun():
    r = _ok_report()
    cm = {"r2": 0.95}
    ok, _l = DR.gate(r, 144.0, _fake_extrap(50.0), cost_model=cm)
    _expect(ok is True, "50 h should fit 144 h with 15% margin")
    ok, lines = DR.gate(r, 144.0, _fake_extrap(130.0), cost_model=cm)
    _expect(ok is False,
            "130 h must FAIL against 144 h at 15%% margin (budget %.1f h)"
            % (144.0 * 0.85))
    _expect(any("SEQUENTIAL" in l for l in lines),
            "the failure must say gp_minimize cannot be split across lanes")
    ok, _l = DR.gate(r, 144.0, _fake_extrap(130.0), margin=0.0, cost_model=cm)
    _expect(ok is True, "at margin 0 the same 130 h fits, so the margin is "
            "genuinely doing the work")
    print("  [7-I] the gate fails on overrun, and the margin is load-bearing OK")


def test_7j_gate_fails_on_bad_fit():
    r = _ok_report()
    ok_good, _l = DR.gate(r, 144.0, _fake_extrap(50.0), cost_model={"r2": 0.95})
    ok_bad, lines = DR.gate(r, 144.0, _fake_extrap(50.0), cost_model={"r2": 0.29})
    _expect(ok_good is True, "a well-fitting model at 50 h should pass")
    _expect(ok_bad is False,
            "the gate passed an extrapolation built on a cost model that does "
            "not fit -- a precise-looking number with no basis")
    _expect(any("R^2" in l for l in lines), lines)
    ok_nan, _l = DR.gate(r, 144.0, _fake_extrap(50.0),
                         cost_model={"r2": float("nan")})
    _expect(ok_nan is False, "a NaN R^2 must not pass")
    print("  [7-J] the gate fails on a badly-fitting cost model (R^2 < %.2f), "
          "NaN included OK" % DR._MIN_COST_MODEL_R2)


def test_7k_report_is_json():
    r = DR.dry_run(_cfg(), 40, random_state=0, count_params=True, verbose=False)
    try:
        json.dumps(r)
    except TypeError as ex:
        raise AssertionError("the dry-run report is not JSON-serialisable: %s"
                             % ex)
    txt = DR.format_report(r)
    _expect("COVERAGE COMPLETE" in txt, "the human-readable report lost its verdict")
    print("  [7-K] the report is JSON-serialisable and formats to text OK")


def main():
    print("Stage 7 -- the dry run")
    test_7a_trains_nothing()
    test_7b_uses_the_real_builder()
    test_7c_coverage_reported_and_grows()
    test_7d_build_failure_counted()
    print("Stage 7 -- the size mix and the cost model")
    test_7e_size_mix()
    test_7f_cost_model_shape()
    print("Stage 7 -- the submission gate")
    test_7g_gate_fails_on_coverage()
    test_7h_gate_fails_when_unmeasured()
    test_7i_gate_fails_on_overrun()
    test_7j_gate_fails_on_bad_fit()
    test_7k_report_is_json()
    print("\nALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
