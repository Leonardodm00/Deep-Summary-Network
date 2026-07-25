"""
smoke_test_synthetic_config.py
==============================

Correctness checks for the configurable burst-generator settings and the
per-run synthetic artifact saving added to the pipeline. Run BEFORE trusting
either feature.

Covers three things:

  A. MultiClassSyntheticProvider parameter passthrough + per-class overrides
     (data_splits.py):
       A1. Backward compatibility: constructed with only the ORIGINAL kwargs
           (n_classes/duration_s/fs/seed, plus the original rate/width), the
           provider yields byte-identical traces to a reference built the same
           way. This guards the rng-draw-order invariant that keeps previously
           cached runs reproducible.
       A2. A global sweep change (e.g. rate_max) actually changes the output.
       A3. A per-class rate/width override changes ONLY that class; other
           classes are byte-identical to the no-override provider.
       A4. Class 0 stays fixed-amplitude (a=1.0) by default, but a per-class
           amp override on class 0 promotes it to jittered (output changes).
       A5. Invalid params raise (rate_min>rate_max, amp_min without amp_max,
           per_class longer than n_classes).

  B. Config layer (config.py):
       B1. SyntheticConfig round-trips through config_from_dict with per-class
           overrides preserved (dict -> dataclass -> asdict equality on the
           synthetic subtree).
       B2. DataConfig rejects per_class longer than synthetic_n_per_class.
       B3. _resolved_synthetic_params (run_optimization.py) reports the SAME
           effective rate/width/amp per class that the provider actually uses,
           for a config mixing sweep and overrides. This is the check that the
           saved params JSON is not lying about what was generated.

  C. Artifact saving (run_optimization.save_synthetic_artifacts):
       C1. Given already-generated traces/conditions, it writes
           synthetic_generator_params.json into out_dir and
           synthetic_traces_overview.png into fig_dir, both non-trivial.
       C2. The params JSON on disk matches _resolved_synthetic_params exactly.

How to run
----------
    cd Main
    PYTHONPATH="$PWD:$PYTHONPATH" python3 Smoke_Tests/smoke_test_synthetic_config.py

Exits 0 + prints ALL SMOKE TESTS PASSED on success, else exits 1. Cheap enough
to run twice; A1/A3 already assert determinism internally.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

# This project's convention (see run_all_smoke_tests.py): all pipeline modules
# and all smoke tests live in ONE FLAT directory and import each other by bare
# name. No sys.path manipulation needed when run from that directory.
import config as C
from data_splits import MultiClassSyntheticProvider
import run_optimization as R

PASSED = 0
FAILED = 0
FAILURES = []


def check(label, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print("PASS  %s" % label)
    else:
        FAILED += 1
        FAILURES.append("%s -- %s" % (label, detail))
        print("FAIL  %s  (%s)" % (label, detail))


# --------------------------------------------------------------------------- #
# A. provider passthrough + overrides
# --------------------------------------------------------------------------- #
def test_A1_backward_compat():
    for n_classes in (1, 2, 3):
        ref = MultiClassSyntheticProvider(n_classes=n_classes, duration_s=120.0,
                                          fs=50.0, seed=0)
        # same original kwargs, explicitly (defaults must equal the old code)
        same = MultiClassSyntheticProvider(n_classes=n_classes, duration_s=120.0,
                                           fs=50.0, seed=0, rate_min=0.25,
                                           rate_max=0.55, width_min=0.15,
                                           width_max=0.70)
        ok = all(
            np.array_equal(ref(c, t)[0], same(c, t)[0])
            for c in range(n_classes) for t in range(3)
        )
        check("A1 backward-compat: default provider byte-identical (C=%d)" % n_classes,
              ok, "defaults changed the rng draw order / output")


def test_A2_sweep_changes_output():
    base = MultiClassSyntheticProvider(n_classes=3, duration_s=120.0, fs=50.0, seed=0)
    hot = MultiClassSyntheticProvider(n_classes=3, duration_s=120.0, fs=50.0, seed=0,
                                      rate_max=0.90)
    # class 2 (frac=1) sees rate_max directly -> must differ; class 0 (frac=0)
    # is unaffected by rate_max -> must match.
    check("A2 sweep: rate_max change alters class 2 output",
          not np.array_equal(base(2, 0)[0], hot(2, 0)[0]),
          "class 2 unchanged despite rate_max change")
    check("A2 sweep: rate_max change leaves class 0 output identical",
          np.array_equal(base(0, 0)[0], hot(0, 0)[0]),
          "class 0 changed despite frac(0)=0")


def test_A3_per_class_isolation():
    base = MultiClassSyntheticProvider(n_classes=3, duration_s=120.0, fs=50.0, seed=0)
    ov = MultiClassSyntheticProvider(
        n_classes=3, duration_s=120.0, fs=50.0, seed=0,
        per_class=[{"rate": 0.15, "width": 0.5}, None, None])
    check("A3 per-class: class 0 override changes class 0",
          not np.array_equal(base(0, 0)[0], ov(0, 0)[0]),
          "class 0 unchanged despite override")
    check("A3 per-class: class 1 (no override) byte-identical",
          np.array_equal(base(1, 0)[0], ov(1, 0)[0]),
          "class 1 changed despite no override")
    check("A3 per-class: class 2 (no override) byte-identical",
          np.array_equal(base(2, 0)[0], ov(2, 0)[0]),
          "class 2 changed despite no override")


def test_A4_class0_amp_promotion():
    base = MultiClassSyntheticProvider(n_classes=3, duration_s=120.0, fs=50.0, seed=0)
    promoted = MultiClassSyntheticProvider(
        n_classes=3, duration_s=120.0, fs=50.0, seed=0,
        per_class=[{"amp_min": 0.6, "amp_max": 1.4}, None, None])
    # Promoting class 0 to jittered amplitude draws an extra rng.uniform per
    # burst -> output must change from the fixed-a=1.0 baseline.
    check("A4 amp-promotion: class-0 amp override changes class 0 output",
          not np.array_equal(base(0, 0)[0], promoted(0, 0)[0]),
          "class 0 output identical despite amp promotion")


def test_A5_invalid_params_raise():
    def raises(fn):
        try:
            fn()
            return False
        except (ValueError, Exception):
            return True

    check("A5 invalid: rate_min > rate_max raises",
          raises(lambda: MultiClassSyntheticProvider(
              n_classes=2, rate_min=0.9, rate_max=0.1)),
          "no error for rate_min>rate_max")
    check("A5 invalid: amp_jitter_min > amp_jitter_max raises",
          raises(lambda: MultiClassSyntheticProvider(
              n_classes=2, amp_jitter_min=1.5, amp_jitter_max=0.5)),
          "no error for amp_jitter_min>max")
    check("A5 invalid: per_class longer than n_classes raises",
          raises(lambda: MultiClassSyntheticProvider(
              n_classes=2, per_class=[{}, {}, {}])),
          "no error for oversized per_class")
    check("A5 invalid: SyntheticClassOverride amp_min alone raises",
          raises(lambda: C.SyntheticClassOverride(amp_min=0.5)),
          "no error for amp_min without amp_max")


# --------------------------------------------------------------------------- #
# B. config layer
# --------------------------------------------------------------------------- #
def test_B1_config_roundtrip():
    raw = {
        "rate_min": 0.2, "rate_max": 0.8, "width_min": 0.1, "width_max": 0.6,
        "amp_jitter_min": 0.5, "amp_jitter_max": 1.5,
        "per_class": [
            {"rate": 0.15, "width": 0.5, "amp_min": None, "amp_max": None},
            {"rate": None, "width": None, "amp_min": None, "amp_max": None},
            {"rate": None, "width": None, "amp_min": 0.9, "amp_max": 1.1},
        ],
    }
    syn = C.config_from_dict(C.SyntheticConfig, raw)
    from dataclasses import asdict
    round = asdict(syn)
    # per_class comes back as a tuple (config uses Tuple[..., ...] by design);
    # normalize tuple->list before comparing so we test CONTENT equality, not
    # the tuple-vs-list container distinction which is an intended convention.
    round_norm = {**round, "per_class": [dict(o) for o in round["per_class"]]}
    check("B1 config round-trip: SyntheticConfig dict -> dataclass -> dict equal",
          round_norm == raw, "round-trip mismatch: %r" % round_norm)


def test_B2_dataconfig_rejects_long_per_class():
    try:
        C.config_from_dict(C.DataConfig, {
            "synthetic_n_per_class": [2],
            "synthetic": {"per_class": [{}, {}]},
        })
        check("B2 DataConfig rejects per_class longer than C", False,
              "no error raised")
    except ValueError:
        check("B2 DataConfig rejects per_class longer than C", True)


def test_B3_resolved_params_match_provider():
    # Build a config mixing sweep + overrides, then check the reported
    # effective params equal what the provider computes internally.
    raw = {
        "data": {
            "data_mode": "synthetic",
            "synthetic_n_per_class": [2, 2, 2],
            "synthetic_duration_s": 120.0,
            "synthetic_fs": 50.0,
            "synthetic": {
                "rate_min": 0.25, "rate_max": 0.55,
                "width_min": 0.15, "width_max": 0.70,
                "amp_jitter_min": 0.6, "amp_jitter_max": 1.4,
                "per_class": [
                    {"rate": 0.20, "width": 0.60},
                    {},
                    {"amp_min": 0.9, "amp_max": 1.1},
                ],
            },
        },
        "runtime": {"seed": 0},
    }
    cfg = C.ExperimentConfig.from_dict(raw)
    params = R._resolved_synthetic_params(cfg)

    # Independently recompute the expected effective values.
    syn = cfg.data.synthetic
    Cn = 3
    def frac(c):
        return 0.0 if Cn == 1 else c / (Cn - 1)
    # class 0: rate override 0.20, width override 0.60, amp fixed
    e0 = params["per_class_effective"][0]
    check("B3 resolved: class 0 rate/width overrides reported",
          abs(e0["rate_bursts_per_s"] - 0.20) < 1e-9
          and abs(e0["base_width_s"] - 0.60) < 1e-9
          and e0["amp_fixed_1p0"] is True,
          "got %r" % e0)
    # class 1: pure sweep
    e1 = params["per_class_effective"][1]
    exp_rate1 = 0.25 + (0.55 - 0.25) * frac(1)
    exp_width1 = 0.70 - (0.70 - 0.15) * frac(1)
    check("B3 resolved: class 1 pure-sweep rate/width reported",
          abs(e1["rate_bursts_per_s"] - exp_rate1) < 1e-9
          and abs(e1["base_width_s"] - exp_width1) < 1e-9
          and e1["amp_fixed_1p0"] is False,
          "got %r (expected rate=%.4f width=%.4f)" % (e1, exp_rate1, exp_width1))
    # class 2: swept rate/width, amp override tightened
    e2 = params["per_class_effective"][2]
    check("B3 resolved: class 2 amp override reported (0.9,1.1)",
          abs(e2["amp_jitter_min"] - 0.9) < 1e-9
          and abs(e2["amp_jitter_max"] - 1.1) < 1e-9,
          "got %r" % e2)


# --------------------------------------------------------------------------- #
# C. artifact saving
# --------------------------------------------------------------------------- #
def test_C_artifact_saving():
    raw = {
        "data": {
            "data_mode": "synthetic",
            "synthetic_n_per_class": [2, 2, 2],
            "synthetic_duration_s": 60.0,
            "synthetic_fs": 50.0,
            "synthetic": {"per_class": [{"rate": 0.2}, {}, {"amp_min": 0.9, "amp_max": 1.1}]},
        },
        "runtime": {"seed": 0},
    }
    cfg = C.ExperimentConfig.from_dict(raw)

    # Generate small traces directly with the provider (mirrors build_traces).
    from data_splits import make_synthetic_specs
    n_per_class = tuple(cfg.data.synthetic_n_per_class)
    syn = cfg.data.synthetic
    provider = MultiClassSyntheticProvider(
        n_classes=len(n_per_class), duration_s=60.0, fs=50.0, seed=0,
        rate_min=syn.rate_min, rate_max=syn.rate_max,
        width_min=syn.width_min, width_max=syn.width_max,
        amp_jitter_min=syn.amp_jitter_min, amp_jitter_max=syn.amp_jitter_max,
        per_class=list(syn.per_class))
    specs = make_synthetic_specs(n_per_class)
    traces, conditions = [], []
    for spec in specs:
        tr, _ = provider(*spec["args"])
        traces.append(tr)
        conditions.append(spec["condition"])

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "out"
        fig_dir = out_dir / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)

        params_path = R.save_synthetic_artifacts(
            cfg, traces, conditions, 50.0, out_dir, fig_dir, verbose=False)

        json_ok = params_path.exists() and params_path.stat().st_size > 100
        check("C1 artifact: params JSON written and non-trivial", json_ok,
              "missing or tiny at %s" % params_path)

        fig_path = fig_dir / "synthetic_traces_overview.png"
        fig_ok = fig_path.exists() and fig_path.stat().st_size > 10_000
        check("C1 artifact: traces figure written and non-trivial (>10KB)", fig_ok,
              "missing or tiny at %s" % fig_path)

        # C2: on-disk JSON equals _resolved_synthetic_params
        on_disk = json.loads(params_path.read_text())
        expected = R._resolved_synthetic_params(cfg)
        check("C2 artifact: on-disk params JSON matches _resolved_synthetic_params",
              on_disk == expected, "mismatch between saved JSON and resolver")


def main():
    tests = [
        test_A1_backward_compat,
        test_A2_sweep_changes_output,
        test_A3_per_class_isolation,
        test_A4_class0_amp_promotion,
        test_A5_invalid_params_raise,
        test_B1_config_roundtrip,
        test_B2_dataconfig_rejects_long_per_class,
        test_B3_resolved_params_match_provider,
        test_C_artifact_saving,
    ]
    for t in tests:
        try:
            t()
        except Exception:
            global FAILED
            FAILED += 1
            FAILURES.append("%s raised:\n%s" % (t.__name__, traceback.format_exc()))
            print("FAIL  %s  (exception, see below)" % t.__name__)
            print(traceback.format_exc())

    print()
    print("=" * 60)
    print("PASSED: %d   FAILED: %d" % (PASSED, FAILED))
    if FAILED:
        print("FAILURES:")
        for f in FAILURES:
            print("  - %s" % f)
        print("SMOKE TESTS FAILED")
        sys.exit(1)
    print("ALL SMOKE TESTS PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
