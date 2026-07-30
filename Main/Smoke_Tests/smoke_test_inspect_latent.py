"""
smoke_test_inspect_latent.py

Correctness checks for inspect_latent_benchmark.py.

The check that matters
----------------------
inspect_latent_benchmark.latent_spec_from_config_dict re-implements the
config -> LatentSpec unpacking that run_optimization.latent_spec_from_config
performs, so that the inspection tool needs no torch. Two copies of one mapping
is a DRIFT risk of exactly the kind the C2 selected-epoch rule has: edit one and
the other silently disagrees, and the figure you inspected would then describe a
different benchmark from the one the cluster trains on. Check [A] compares the
two, field by field, on several configs. It REQUIRES torch, because it imports
run_optimization; every other check here is torch-free.

Run:
    cd Main/Smoke_Tests && python3 smoke_test_inspect_latent.py

Checks:
  A. The two config -> LatentSpec mappings agree field by field: axes (name,
     target, lo, hi, orientation), label_axes, n_classes, n_per_class,
     class_overlap, duration_s, n_neurons, w_size, gaussian_window, seed.
     Tested on the shipped hard config, the easypos config, a config with an
     axis range override, and an empty config (defaults).
  B. Overrides are honoured: --duration-s, --n-neurons, --seed reach the spec
     and do NOT disturb any other field.
  C. synthesize_dataset returns the right count, shape, ordering and labels;
     traces are finite and non-negative; every trace shares one f_s.
  D. The latent coordinates it reports are the SAME ones latent_ground_truth_table
     records for those traces -- i.e. the figure and the latent ground truth
     cannot disagree.
  E. The label axes track the class and the free axes do not, computed on the
     returned Phi. This is the property the right-hand panel of
     latent_factors.png exists to show, asserted numerically so a broken axis
     assignment fails here rather than being missed by eye.
  F. main() runs end to end and writes both figures plus the JSON; the figures
     are non-trivial in size and the JSON is pure ASCII and parses.
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inspect_latent_benchmark as I                             # noqa: E402
from latent_burst_generator import latent_ground_truth_table     # noqa: E402

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_SPEC_FIELDS = ("label_axes", "n_classes", "n_per_class", "class_overlap",
                "class_center_mode", "duration_s", "n_neurons", "w_size",
                "gaussian_window", "seed")


def _assert_specs_equal(a, b, label):
    assert len(a.axes) == len(b.axes), (
        "%s: n differs (%d vs %d)" % (label, len(a.axes), len(b.axes)))
    for k, (ax_a, ax_b) in enumerate(zip(a.axes, b.axes)):
        for f in ("name", "target", "lo", "hi", "orientation"):
            va, vb = getattr(ax_a, f), getattr(ax_b, f)
            assert va == vb, ("%s: axis %d field %r differs: %r vs %r"
                              % (label, k, f, va, vb))
    for f in _SPEC_FIELDS:
        va, vb = getattr(a, f), getattr(b, f)
        if isinstance(va, float):
            assert abs(va - vb) < 1e-12, ("%s: %s differs: %r vs %r"
                                          % (label, f, va, vb))
        else:
            assert va == vb, "%s: %s differs: %r vs %r" % (label, f, va, vb)


def check_mappings_agree():
    """[A] the two config -> LatentSpec mappings must not drift apart."""
    try:
        import run_optimization as R
    except ImportError as ex:
        print("  [A] SKIPPED: run_optimization not importable (%s: %s). This "
              "check needs torch." % (type(ex).__name__, ex))
        return
    from config import ExperimentConfig, LatentAxisOverride

    cases = []
    for name in ("config_latent_3class_hard.json",
                 "config_latent_3class_easypos.json"):
        p = os.path.join(_HERE, "hpc", name)
        if os.path.exists(p):
            cases.append((name, json.load(open(p, encoding="ascii"))))
    if not cases:
        raise AssertionError("neither shipped hpc/ config was found")

    # an override case, built from the hard config
    ov = json.loads(json.dumps(cases[0][1]))
    ov["data"]["latent"]["axis_overrides"] = [
        {"name": "burst_rate", "lo": 0.05, "hi": 0.9, "orientation": -1}]
    ov["data"]["latent"]["class_overlap"] = 0.25
    ov["data"]["latent"]["label_axes"] = [0, 2]
    cases.append(("axis-override variant", ov))

    for label, cfg_dict in cases:
        spec_inspect = I.latent_spec_from_config_dict(cfg_dict)
        spec_driver = R.latent_spec_from_config(ExperimentConfig.from_dict(cfg_dict))
        _assert_specs_equal(spec_inspect, spec_driver, label)
        print("      %-24s agrees on all %d axes + %d scalar fields"
              % (label, len(spec_inspect.axes), len(_SPEC_FIELDS)))

    # and with no config at all, the inspection defaults must match the
    # DataConfig/LatentConfig defaults the driver would use for a latent run
    cfg_default = ExperimentConfig()
    cfg_default.data.data_mode = "latent"
    cfg_default.data.synthetic_n_per_class = (3, 3, 3)
    spec_driver = R.latent_spec_from_config(cfg_default)
    spec_inspect = I.latent_spec_from_config_dict(
        {"data": {"synthetic_n_per_class": [3, 3, 3],
                  "synthetic_duration_s": cfg_default.data.synthetic_duration_s,
                  "synthetic_fs": cfg_default.data.synthetic_fs}})
    _assert_specs_equal(spec_inspect, spec_driver, "defaults")
    print("      %-24s agrees" % "defaults (no config)")
    print("  [A] the inspection tool and the driver build the SAME LatentSpec OK")


def check_overrides():
    """[B] the CLI overrides reach the spec and disturb nothing else."""
    p = os.path.join(_HERE, "hpc", "config_latent_3class_hard.json")
    cfg = json.load(open(p, encoding="ascii"))
    base = I.latent_spec_from_config_dict(cfg)
    got = I.latent_spec_from_config_dict(cfg, duration_s=45.0, n_neurons=17,
                                         seed=99)
    assert abs(got.duration_s - 45.0) < 1e-12, got.duration_s
    assert got.n_neurons == 17, got.n_neurons
    assert got.seed == 99, got.seed
    for f in ("label_axes", "n_classes", "n_per_class", "class_overlap",
              "w_size", "gaussian_window"):
        assert getattr(got, f) == getattr(base, f), f
    assert [a.name for a in got.axes] == [a.name for a in base.axes]
    print("  [B] --duration-s / --n-neurons / --seed reach the spec; every "
          "other field is untouched OK")

    # --n-per-class sets C, and --tau overrides the overlap
    four = I.latent_spec_from_config_dict(cfg, n_per_class=[3, 3, 3, 3])
    assert four.n_classes == 4, four.n_classes
    assert four.n_per_class == (3, 3, 3, 3), four.n_per_class
    from latent_burst_generator import _class_mean
    means = [_class_mean(c, four.n_classes, four.class_center_mode)
             for c in range(four.n_classes)]
    assert means == [0.2, 0.4, 0.6, 0.8], means
    tuned = I.latent_spec_from_config_dict(cfg, n_per_class=[2, 2, 2],
                                           class_overlap=0.07)
    assert abs(tuned.class_overlap - 0.07) < 1e-12
    assert tuned.n_per_class == (2, 2, 2)
    print("  [B] --n-per-class sets C (4 classes -> m_c = %r) and --tau "
          "overrides the overlap OK" % (means,))


def check_synthesis():
    """[C] + [D] + [E] the generated dataset and its reported coordinates."""
    p = os.path.join(_HERE, "hpc", "config_latent_3class_hard.json")
    cfg = json.load(open(p, encoding="ascii"))
    spec = I.latent_spec_from_config_dict(cfg, duration_s=40.0, n_neurons=25)
    traces, conditions, trace_ids, Phi, fs = I.synthesize_dataset(spec)

    n_total = sum(spec.n_per_class)
    K = int(round(spec.duration_s * spec.fs))
    assert len(traces) == n_total, (len(traces), n_total)
    assert Phi.shape == (n_total, spec.n_latent), Phi.shape
    assert abs(fs - spec.fs) < 1e-9
    assert conditions == [0, 0, 0, 1, 1, 1, 2, 2, 2], conditions
    assert trace_ids == [0, 1, 2, 0, 1, 2, 0, 1, 2], trace_ids
    for x in traces:
        assert x.shape[0] == K, (x.shape, K)
        assert np.all(np.isfinite(x)) and np.all(x >= 0.0)
    assert np.all((Phi >= 0.0) & (Phi <= 1.0))
    print("  [C] %d traces, K = %d samples, f_s = %.4g Hz, class-major order, "
          "all finite and non-negative, phi in [0,1]^%d OK"
          % (len(traces), K, fs, spec.n_latent))

    # [D] the coordinates must match the latent ground-truth table exactly
    table = latent_ground_truth_table(spec)
    by_key = {(int(r["condition"]), int(r["trace_id"])): np.asarray(r["phi"])
              for r in table["rows"]}
    for t, (c, r) in enumerate(zip(conditions, trace_ids)):
        assert np.allclose(Phi[t], by_key[(c, r)]), (
            "trace (c=%d, r=%d): the inspected phi differs from the ground-truth "
            "table written as latent_ground_truth.json" % (c, r))
    print("  [D] every reported phi matches latent_ground_truth_table exactly -- "
          "the figure and the latent ground truth cannot disagree OK")

    # [E] label axes track the class; free axes do not
    y = np.asarray(conditions, dtype=float)
    corr = []
    for k in range(spec.n_latent):
        col = Phi[:, k]
        c = 0.0 if np.std(col) < 1e-12 else float(np.corrcoef(col, y)[0, 1])
        corr.append(abs(c))
    lab = [corr[k] for k in spec.label_axes]
    free = [corr[k] for k in spec.free_axes]
    assert min(lab) > 0.8, "label axes should track the class; got %r" % (lab,)
    assert max(free) < 0.6, (
        "a FREE axis tracks the class (|rho| = %.3f): the axis assignment is "
        "wrong, so the axis is not label-irrelevant and nothing computed on "
        "the free axes would mean what it claims" % max(free))
    print("  [E] |rho| with the class: label axes %s vs free axes max %.3f -- "
          "separated, as the right-hand figure panel should show OK"
          % (["%.3f" % v for v in lab], max(free)))


def check_end_to_end():
    """[F] main() writes both figures and the JSON."""
    tmp = tempfile.mkdtemp(prefix="inspect_latent_")
    try:
        rc = I.main(["--config", os.path.join(_HERE, "hpc",
                                              "config_latent_3class_hard.json"),
                     "--out-dir", tmp, "--duration-s", "40", "--n-neurons", "25",
                     "--show-seconds", "20"])
        assert rc == 0, rc
        for name, min_bytes in (("latent_traces.png", 10000),
                                ("latent_factors.png", 10000),
                                ("latent_inspection.json", 500)):
            p = os.path.join(tmp, name)
            assert os.path.exists(p), "missing output %s" % name
            size = os.path.getsize(p)
            assert size >= min_bytes, "%s is only %d bytes" % (name, size)
        raw = open(os.path.join(tmp, "latent_inspection.json"), "rb").read()
        assert all(b < 128 for b in raw), "the JSON is not pure ASCII"
        table = json.loads(raw.decode("ascii"))
        assert len(table["rows"]) == 9, len(table["rows"])
        print("  [F] main() wrote latent_traces.png (%d B), latent_factors.png "
              "(%d B) and a pure-ASCII latent_inspection.json with %d rows OK"
              % (os.path.getsize(os.path.join(tmp, "latent_traces.png")),
                 os.path.getsize(os.path.join(tmp, "latent_factors.png")),
                 len(table["rows"])))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("smoke_test_inspect_latent.py")
    check_mappings_agree()
    check_overrides()
    check_synthesis()
    check_end_to_end()
    print("ALL INSPECT-LATENT CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
