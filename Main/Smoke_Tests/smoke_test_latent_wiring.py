"""
smoke_test_latent_wiring.py

Correctness checks for the C1 WIRING -- the connection of the (already verified)
latent_burst_generator to the pipeline. The generator's own behaviour is checked
by Main/smoke_test_latent_and_objective.py; this file checks only that the
driver reaches it, records it, and refuses to reuse stale traces.

REQUIRES torch, because it imports run_optimization (which imports train ->
torch). It does NOT train anything and runs in a few seconds on CPU.

Run:
    cd Main/Smoke_Tests && python3 smoke_test_latent_wiring.py

Checks:
  A. data_mode = "latent" round-trips through the config, and the latent block
     survives serialization as TUPLES (the coercion contract config.py asserts).
  B. latent_spec_from_config maps the config onto a LatentSpec correctly:
     w_size = 1 / synthetic_fs, C and n_c from synthetic_n_per_class, T_rec from
     synthetic_duration_s -- i.e. the shared fields are genuinely shared and not
     silently re-declared.
  C. Axis range overrides reach the spec.
  D. THE CACHE FINGERPRINT IS SENSITIVE TO THE LATENT PARAMETERS. This is the
     check that matters most: the spec list for latent mode is
     make_synthetic_specs(n_per_class), which is IDENTICAL for every tau and
     every choice of label axes. Without the latent block in the fingerprint,
     changing tau would silently reuse the old traces and the run would report
     the new config while training on the old benchmark. Asserted for tau,
     label_axes, axis_names, axis range overrides, n_neurons and seed --
     and asserted STABLE when nothing changes.
  E. build_traces(data_mode="latent") produces C * n_c traces of the right
     length at the right f_s, with the right labels, through the SAME
     cache_traces path the synthetic branch uses.
  F. Re-running with a changed tau against the SAME cache directory RAISES the
     stale-cache error rather than silently reusing traces. This is D's payoff
     and the actual protection.
  G. save_latent_artifacts writes latent_ground_truth.json, pure ASCII, with one
     row per trace and the label/free axis split recorded.
  H. The provider call signature is identical to MultiClassSyntheticProvider's,
     which is what lets make_synthetic_specs be reused unchanged.
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C                                              # noqa: E402
from config import DataConfig, ExperimentConfig, LatentAxisOverride, LatentConfig  # noqa: E402
from data_splits import MultiClassSyntheticProvider, make_synthetic_specs  # noqa: E402
from latent_burst_generator import LatentBurstProvider          # noqa: E402
import run_optimization as R                                    # noqa: E402


def _cfg(tmp, **latent_kw):
    """A deliberately tiny latent config: 3 traces, 60 s, 20 neurons."""
    lat = LatentConfig(**latent_kw) if latent_kw else LatentConfig()
    cfg = ExperimentConfig()
    cfg.data = DataConfig(
        data_mode="latent",
        synthetic_n_per_class=(1, 1, 1),
        synthetic_duration_s=60.0,
        synthetic_fs=50.0,
        window_s=10.0, train_stride_s=5.0, eval_stride_s=10.0,
        latent=lat,
    )
    cfg.runtime.seed = 0
    cfg.runtime.cache_dir = os.path.join(tmp, "cache")
    cfg.runtime.out_dir = os.path.join(tmp, "out")
    return cfg


def check_config_roundtrip(tmp):
    cfg = _cfg(tmp, axis_names=("irregularity", "burst_rate", "burst_duration"),
               label_axes=(0, 1), class_overlap=0.15, n_neurons=20)
    p = os.path.join(tmp, "cfg.json")
    cfg.to_json(p)
    back = ExperimentConfig.from_json(p)
    assert back == cfg, "latent config did not round-trip"
    assert isinstance(back.data.latent.axis_names, tuple)
    assert isinstance(back.data.latent.label_axes, tuple)
    assert back.data.data_mode == "latent"
    raw = open(p, "rb").read()
    assert all(b < 128 for b in raw), "config JSON is not pure ASCII"
    print("  [A] data_mode='latent' round-trips exactly; nested tuples stay "
          "tuples; JSON on disk is pure ASCII OK")


def check_spec_mapping(tmp):
    cfg = _cfg(tmp, axis_names=("irregularity", "burst_rate", "background"),
               label_axes=(0,), class_overlap=0.2, n_neurons=17,
               gaussian_window=0.06)
    spec = R.latent_spec_from_config(cfg)
    assert spec.n_latent == 3, spec.n_latent
    assert spec.n_classes == 3, spec.n_classes
    assert spec.n_per_class == (1, 1, 1), spec.n_per_class
    assert abs(spec.duration_s - 60.0) < 1e-12
    assert abs(spec.w_size - 1.0 / 50.0) < 1e-15, spec.w_size
    assert abs(spec.fs - 50.0) < 1e-12, spec.fs
    assert spec.n_neurons == 17
    assert abs(spec.gaussian_window - 0.06) < 1e-12
    assert abs(spec.class_overlap - 0.2) < 1e-12
    assert spec.label_axes == (0,)
    assert spec.free_axes == (1, 2), spec.free_axes
    assert spec.seed == cfg.runtime.seed
    print("  [B] config -> LatentSpec: w_size = 1/f_s = %.4f s, C = %d and "
          "n_c = %r from synthetic_n_per_class, T_rec = %.1f s from "
          "synthetic_duration_s -- shared, not re-declared OK"
          % (spec.w_size, spec.n_classes, list(spec.n_per_class), spec.duration_s))

    cfg2 = _cfg(tmp, axis_names=("irregularity", "burst_rate", "background"),
                label_axes=(0, 1),
                axis_overrides=(LatentAxisOverride(name="burst_rate", lo=0.05,
                                                   hi=0.90, orientation=-1),))
    spec2 = R.latent_spec_from_config(cfg2)
    ax = {a.name: a for a in spec2.axes}["burst_rate"]
    assert (ax.lo, ax.hi, ax.orientation) == (0.05, 0.90, -1), ax
    assert abs(ax.value(0.0) - 0.90) < 1e-12, "orientation -1 not applied"
    print("  [C] axis range override reaches the spec: burst_rate -> [%.2f, %.2f], "
          "orientation %+d (phi = 0 maps to %.2f) OK"
          % (ax.lo, ax.hi, ax.orientation, ax.value(0.0)))


def check_fingerprint_sensitivity(tmp):
    base = _cfg(tmp)
    specs = make_synthetic_specs(base.data.synthetic_n_per_class)
    fp0 = R._data_fingerprint(base, specs)

    # stable when nothing changes
    assert R._data_fingerprint(_cfg(tmp), specs) == fp0, "fingerprint is not stable"

    variants = {
        "class_overlap (tau)": _cfg(tmp, class_overlap=0.30),
        "label_axes S": _cfg(tmp, label_axes=(0, 2)),
        "axis_names (which axes)": _cfg(tmp, axis_names=(
            "irregularity", "burst_rate", "burst_duration")),
        "n_neurons": _cfg(tmp, n_neurons=50),
        "gaussian_window": _cfg(tmp, gaussian_window=0.05),
        "class_center_mode": _cfg(tmp, class_center_mode="endpoints"),
        "axis range override": _cfg(tmp, axis_overrides=(
            LatentAxisOverride(name="burst_rate", lo=0.11, hi=0.41),)),
    }
    for label, cfg_v in variants.items():
        fp = R._data_fingerprint(cfg_v, specs)
        assert fp != fp0, (
            "CHANGING %s DID NOT CHANGE THE FINGERPRINT. The spec list is "
            "identical for every latent parameter, so the trace cache would be "
            "silently reused and the run would train on the OLD benchmark."
            % label)
    seeded = _cfg(tmp)
    seeded.runtime.seed = 1
    assert R._data_fingerprint(seeded, specs) != fp0, "seed must change it too"
    print("  [D] fingerprint changes for every latent parameter: %s, seed -- and "
          "is stable when nothing changes OK" % ", ".join(variants))

    # and the specs themselves really ARE identical, which is why D is necessary
    s_a = make_synthetic_specs(_cfg(tmp).data.synthetic_n_per_class)
    s_b = make_synthetic_specs(_cfg(tmp, class_overlap=0.30).data.synthetic_n_per_class)
    assert s_a == s_b
    print("  [D] confirmed the spec lists ARE identical across tau, so the "
          "fingerprint is the ONLY thing standing between you and a stale "
          "cache OK")


def check_build_traces(tmp):
    cfg = _cfg(tmp, n_neurons=20)
    traces, conditions, fs = R.build_traces(cfg, verbose=False)
    assert len(traces) == 3, len(traces)
    assert conditions == [0, 1, 2], conditions
    assert abs(fs - 50.0) < 1e-9, fs
    for t in traces:
        assert t.shape[0] == 3000, t.shape        # 60 s * 50 Hz
        assert np.all(np.isfinite(t)) and np.all(t >= 0.0)
    assert os.path.exists(os.path.join(cfg.runtime.cache_dir,
                                       "data_fingerprint.json"))
    print("  [E] build_traces(latent): %d traces, labels %r, f_s = %.1f Hz, "
          "K = %d samples, cached with a fingerprint OK"
          % (len(traces), conditions, fs, traces[0].shape[0]))
    return cfg


def check_stale_cache_refusal(tmp, cfg_used):
    """The payoff of D: a changed tau against the same cache dir must REFUSE."""
    cfg2 = _cfg(tmp, class_overlap=0.30)
    cfg2.runtime.cache_dir = cfg_used.runtime.cache_dir      # SAME cache
    try:
        R.build_traces(cfg2, verbose=False)
    except ValueError as ex:
        assert "STALE TRACE CACHE" in str(ex), str(ex)
        print("  [F] changing tau against the same cache dir REFUSES with the "
              "stale-cache error instead of silently reusing traces OK")
    else:
        raise AssertionError(
            "changing tau did NOT trip the stale-cache guard -- this is the "
            "silent-corruption bug the fingerprint exists to prevent")

    # ... and --overwrite-cache is the documented escape hatch
    traces, conditions, fs = R.build_traces(cfg2, overwrite_cache=True)
    assert len(traces) == 3
    print("  [F] --overwrite-cache regenerates them, as documented OK")


def check_ground_truth_artifact(tmp):
    cfg = _cfg(tmp, class_overlap=0.10)
    out = os.path.join(tmp, "gt_out")
    os.makedirs(out, exist_ok=True)
    path = R.save_latent_artifacts(cfg, out, verbose=False)
    raw = open(path, "rb").read()
    assert all(b < 128 for b in raw), "ground-truth JSON is not pure ASCII"
    table = json.loads(raw.decode("ascii"))
    assert len(table["rows"]) == 3, len(table["rows"])
    assert table["n_latent"] == 6, table["n_latent"]
    assert table["label_axes"] == [0, 1], table["label_axes"]
    assert table["free_axes"] == [2, 3, 4, 5], table["free_axes"]
    assert abs(table["fs"] - 50.0) < 1e-9
    for row in table["rows"]:
        assert len(row["phi"]) == 6
        assert all(0.0 <= v <= 1.0 for v in row["phi"]), row["phi"]
        assert set(row["physical"]) >= {"lambda_b", "sigma_d", "median_duration_s"}
    print("  [G] latent_ground_truth.json: %d rows, n = %d axes, S = %r, "
          "free = %r, pure ASCII OK"
          % (len(table["rows"]), table["n_latent"], table["label_axes"],
             table["free_axes"]))


def check_provider_signature(tmp):
    cfg = _cfg(tmp, n_neurons=10)
    spec = R.latent_spec_from_config(cfg)
    latent_p = LatentBurstProvider(spec)
    syn_p = MultiClassSyntheticProvider(n_classes=3, duration_s=60.0, fs=50.0,
                                        seed=0)
    x_l, fs_l = latent_p(1, 0)
    x_s, fs_s = syn_p(1, 0)
    assert isinstance(x_l, np.ndarray) and isinstance(x_s, np.ndarray)
    assert np.isscalar(fs_l) or isinstance(fs_l, float)
    assert abs(fs_l - fs_s) < 1e-9
    assert (1, 0) in latent_p.latents and latent_p.latents[(1, 0)].shape == (6,)
    print("  [H] provider(condition, trace_id) -> (x, f_s) matches "
          "MultiClassSyntheticProvider exactly, and phi is recorded per trace, "
          "so make_synthetic_specs / cache_traces are reused unchanged OK")


def main():
    print("smoke_test_latent_wiring.py [C1 wiring]")
    tmp = tempfile.mkdtemp(prefix="latent_wiring_")
    try:
        check_config_roundtrip(tmp)
        check_spec_mapping(tmp)
        check_fingerprint_sensitivity(tmp)
        cfg_used = check_build_traces(tmp)
        check_stale_cache_refusal(tmp, cfg_used)
        check_ground_truth_artifact(tmp)
        check_provider_signature(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL LATENT-WIRING CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
