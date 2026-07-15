#!/usr/bin/env python3
"""
smoke_test_run_optimization.py
==============================

Correctness harness for run_optimization.py (the driver).

Self-contained: no data files, no GPU, no display, no cluster. Every check runs
on synthetic data in a temporary directory and cleans up after itself.

    python3 smoke_test_run_optimization.py            # all checks  (~3-5 min)
    python3 smoke_test_run_optimization.py --quick    # skip the real-skopt e2e

Exit code 0 = every check passed.

Testing philosophy (inherited from the project's other suites): where a property
could be asserted either by READING the source or by OBSERVING behaviour, these
tests observe behaviour.

  * "--dry-run trains nothing" is proven by PATCHING train() to raise -- in BOTH
    namespaces that hold a reference to it (run_optimization and search) -- and
    showing the dry run still completes. Not by reading the early return.
  * "best_model.pt is selected on VALIDATION, never on test" is proven with
    injected histories where the validation winner and the test winner are
    DIFFERENT seeds, so a test-based selection would pick the other file. The
    check cannot pass vacuously.
  * "dropout is pinned to 0 across the search" is proven by CAPTURING the config
    each search phase actually receives, not by trusting the assignment.
  * "the saved model reproduces the reported score" is proven by rebuilding the
    network from best_model.pt with ZERO hyper-parameters in the test, embedding
    the test split, re-clustering, and comparing the ARI against results.json.

The checks
----------
  [A] budget formula reproduces the two DOCUMENTED totals (753 and 243)
  [B] batch-row formula  M = C * B_c * (1 + P + N)
  [C] window feasibility: the shipped DEFAULT is rejected, and the message names
      the fix; a feasible config passes
  [D] model sizes: meta-device counts == real counts, and reproduce the docs'
      corner table; the >1 GB warning fires
  [E] build_traces, synthetic mode
  [F] build_traces, numpy mode (incl. specs-relative path resolution)
  [G] build_traces, real mode: NotImplementedError without --engine-module, and
      it WORKS with one
  [H] the stale-cache fingerprint guard fires, and --overwrite-cache clears it
  [I] --dry-run trains NOTHING and writes no results.json
  [J] end-to-end with --skip-search: every documented artifact is produced
  [K] best_model.pt REBUILDS and REPRODUCES the reported test ARI exactly
  [L] the splits are leakage-free (test windows disjoint from train and val)
  [M] best_model.pt is selected on VALIDATION, never on test (non-vacuous)
  [N] the search phases are wired correctly: phase-1 arch -> phase-2 HPs ->
      regularization, with beta = 1 - u and regularization's weight_decay winning
  [O] dropout is pinned to 0 for the WHOLE search, even when the input config
      sets dropout > 0
  [P] reproducibility: the same seed twice gives the identical test ARI
  [Q] headless: plt.show is patched to raise and is never reached
  [R] the driver and its whole import chain are pure ASCII (HPC transfer safety)
  [S] the stale-resume guard clears a stale last.pt unless --resume
  [T] the final seeds cannot collide with any search trial's seed block
  [U] end-to-end WITH the real skopt search (skipped by --quick)

HPC note (hpc-python-compat): pure ASCII.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import run_optimization as R
from run_optimization import (
    build_parser, load_config, apply_cli_overrides, run,
    build_traces, check_window_feasibility, estimate_model_sizes,
    estimate_batch_rows, estimate_budget, run_final, run_search_phases,
    _prepare_ckpt_dir, _count_params_meta, FINAL_SEED_OFFSET,
)
from config import ExperimentConfig
from backbone import BackboneConfig, build_backbone
from checkpoint import rebuild_model_from_checkpoint, load_checkpoint
from data_splits import make_time_segment_splits
from inference import embed_clean_windows
from metrics import clustering_metrics

_PASSED = []
_TMP = None


def ok(label, msg):
    _PASSED.append(label)
    print("  [%s] OK   %s" % (label, msg))


def tmpdir(name):
    p = Path(_TMP) / name
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# a tiny, FEASIBLE config used by most checks
# --------------------------------------------------------------------------- #
def toy_config_dict(out_dir, cache_dir, exp="t", n_seeds=1, max_epochs=1,
                    n_calls=3, dropout=0.0):
    return {
        "data": {
            "data_mode": "synthetic",
            "window_s": 8.0, "train_stride_s": 4.0, "eval_stride_s": 8.0,
            "split_fractions": [0.6, 0.2, 0.2],
            "synthetic_duration_s": 200.0, "synthetic_fs": 50.0,
            "synthetic_n_per_class": [2, 2],
            "augmentation": {"fs": 50.0, "n_positives": 3, "n_negatives": 3,
                             "shift_magnitude_s": 1.0},
        },
        "backbone": {"depth_exponent": 3, "width_multiplier": 2.0,
                     "block_family": 0, "embedding_size": 8, "stem_width": 16,
                     "dropout": dropout},
        "train": {"windows_per_condition": 2, "batches_per_epoch": 2,
                  "n_seeds": n_seeds, "max_epochs": max_epochs,
                  "patience": max(1, max_epochs)},
        "search": {"depth_exponent_range": [3, 4],
                   "width_multiplier_range": [1.5, 2.5],
                   "embedding_size_range": [8, 12],
                   "n_calls_arch": n_calls, "n_calls_train": n_calls},
        "regularization": {"n_calls": n_calls},
        "runtime": {"seed": 0, "device": "cpu", "num_workers": 0,
                    "torch_threads": 1, "out_dir": str(out_dir),
                    "cache_dir": str(cache_dir), "experiment_name": exp},
    }


def write_cfg(path, d):
    with open(path, "w", encoding="ascii") as fh:
        json.dump(d, fh, indent=2)
    return str(path)


def cli(*flags):
    return build_parser().parse_args(list(flags))


# --------------------------------------------------------------------------- #
# [A] budget
# --------------------------------------------------------------------------- #
def check_A():
    d = estimate_budget(ExperimentConfig())
    assert d["TOTAL_train_runs"] == 753, \
        "defaults: expected the DOCUMENTED 753 train() runs, got %d" % d["TOTAL_train_runs"]

    # config_example.json geometry (03_USAGE 4.2): 30 + 30 arch/train calls,
    # 20 regularization calls, 3 seeds -> (30+30+20)*3 + 3 = 243
    cfg = ExperimentConfig()
    cfg.search = replace(cfg.search, n_calls_arch=30, n_calls_train=30)
    cfg.regularization = replace(cfg.regularization, n_calls=20)
    e = estimate_budget(cfg)
    assert e["TOTAL_train_runs"] == 243, \
        "config_example: expected the DOCUMENTED 243, got %d" % e["TOTAL_train_runs"]

    # --skip-search must skip EVERY search phase (that is what makes the timing
    # run in 03_USAGE step 2 a timing run): only the final N_s trainings remain.
    s = estimate_budget(cfg, skip_search=True)
    assert s["TOTAL_train_runs"] == cfg.train.n_seeds, \
        "skip_search should cost exactly n_seeds runs, got %d" % s["TOTAL_train_runs"]
    assert s["regularization"] == 0 and s["phase1_arch"] == 0

    # the re-tune is a THIRD architecture pass and must be counted
    cfg.search = replace(cfg.search, do_retune_arch=True)
    r = estimate_budget(cfg)
    assert r["TOTAL_train_runs"] == 243 + 30 * 3, \
        "re-tune not counted: got %d" % r["TOTAL_train_runs"]

    # INVARIANT: every key except TOTAL_train_runs is a count of train() runs and
    # they SUM to the total. A non-count key (n_seeds used to be in here) silently
    # breaks any caller that sums the values -- smoke_test_end_to_end.py [B] does.
    for b in (estimate_budget(ExperimentConfig()),
              estimate_budget(cfg),
              estimate_budget(cfg, skip_search=True),
              estimate_budget(cfg, skip_regularization=True)):
        assert b["TOTAL_train_runs"] == sum(
            v for k, v in b.items() if k != "TOTAL_train_runs"), \
            ("the budget dict must contain ONLY train()-run counts that sum to "
             "TOTAL_train_runs; got %r" % (b,))

    ok("A", "budget reproduces the documented 753 and 243; skip-search=n_seeds; "
            "re-tune counted; every value is a run count that sums to the total")


# --------------------------------------------------------------------------- #
# [B] batch rows
# --------------------------------------------------------------------------- #
def check_B():
    cfg = ExperimentConfig()                       # B_c = 8, P = N = 30
    r = estimate_batch_rows(cfg, n_classes=2)
    assert r["rows_per_batch_M"] == 2 * 8 * (1 + 30 + 30) == 976, r
    assert r["rows_per_batch_M"] == 976, \
        "the documented default batch is 976 rows, got %d" % r["rows_per_batch_M"]

    # the toy geometry the docs print: 2*2*(1+3+3) = 28
    cfg.train = replace(cfg.train, windows_per_condition=2)
    cfg.data = replace(cfg.data, augmentation=replace(
        cfg.data.augmentation, n_positives=3, n_negatives=3))
    assert estimate_batch_rows(cfg, 2)["rows_per_batch_M"] == 28

    # config_example: 2*4*(1+8+8) = 136
    cfg.train = replace(cfg.train, windows_per_condition=4)
    cfg.data = replace(cfg.data, augmentation=replace(
        cfg.data.augmentation, n_positives=8, n_negatives=8))
    assert estimate_batch_rows(cfg, 2)["rows_per_batch_M"] == 136

    # C scales it linearly
    assert estimate_batch_rows(cfg, 3)["rows_per_batch_M"] == 3 * 4 * 17
    ok("B", "M = C*B_c*(1+P+N) exact: 976 (defaults), 28 (toy), 136 (example)")


# --------------------------------------------------------------------------- #
# [C] window feasibility
# --------------------------------------------------------------------------- #
def check_C():
    fs = 50.0
    L = int(600.0 * fs)                            # a 600 s recording
    lengths = [L, L, L, L]
    conds = [0, 0, 1, 1]

    # the SHIPPED DEFAULT is infeasible: window_s = 200 s, but 60/20/20 leaves
    # 120 s val and test segments -> zero windows.
    cfg = ExperimentConfig()
    assert cfg.data.window_s == 200.0
    raised = None
    try:
        check_window_feasibility(cfg, lengths, conds, fs)
    except ValueError as ex:
        raised = str(ex)
    assert raised is not None, \
        "the default config is INFEASIBLE (200 s window vs 120 s eval segments) " \
        "but check_window_feasibility accepted it"
    assert "0 windows" in raised and "window_s" in raised, \
        "the error must name the fix; got: %s" % raised

    # a feasible geometry passes and counts correctly
    cfg.data = replace(cfg.data, window_s=30.0, train_stride_s=15.0,
                       eval_stride_s=30.0)
    feas = check_window_feasibility(cfg, lengths, conds, fs)

    # The report IS a dict of exactly the three splits -- smoke_test_end_to_end.py
    # [C] asserts set(report) == {"train","val","test"}, so extra KEYS would break
    # it. The diagnostics are ATTRIBUTES instead.
    assert set(feas) == {"train", "val", "test"}, sorted(feas)
    assert feas.window_length == 1500
    for k in ("train", "val", "test"):
        assert feas[k] > 0, k
    # cross-check the train count by hand: segment 360 s = 18000 samples,
    # W = 1500, stride 750 -> floor((18000-1500)/750)+1 = 23 per trace, x4 traces
    assert feas["train"] == 23 * 4, dict(feas)
    # val segment 120 s = 6000 samples, stride 1500 -> floor((6000-1500)/1500)+1 = 4
    assert feas["val"] == 4 * 4, dict(feas)
    assert feas.per_class["train"][0] == 23 * 2      # two traces per phenotype
    assert feas.strides["train"] == 750

    # BOTH call forms must work. The 3-arg (cfg, L, fs) form is the one
    # 02_TECHNICAL.md documents and smoke_test_end_to_end.py uses; the 4-arg
    # (cfg, lengths, conditions, fs) form is what the driver itself calls, because
    # real .npz recordings are NOT all the same length and the scalar form can
    # only ever check one of them.
    legacy = check_window_feasibility(cfg, L, fs)          # scalar, no conditions
    assert set(legacy) == {"train", "val", "test"}, sorted(legacy)
    assert all(v > 0 for v in legacy.values()), dict(legacy)
    # one trace instead of four, so exactly a quarter of the windows
    assert legacy["train"] == 23 and legacy["val"] == 4, dict(legacy)
    assert legacy.window_length == 1500

    # and the scalar form must still REJECT the infeasible default
    cfg_bad = ExperimentConfig()
    try:
        check_window_feasibility(cfg_bad, L, fs)
        raise AssertionError("the 3-arg form must also reject the 200 s default")
    except ValueError as ex:
        assert "0 windows" in str(ex)

    ok("C", "default (200 s) REJECTED by BOTH call forms with a fix-naming "
            "message; feasible geometry passes; counts match a hand calculation; "
            "the report's key set is exactly {train,val,test} (extras are attrs)")


# --------------------------------------------------------------------------- #
# [D] model sizes
# --------------------------------------------------------------------------- #
def check_D():
    # meta-device counting must agree EXACTLY with a real construction, or the
    # pre-flight is lying about the thing it exists to predict.
    for d, wm, blk in ((3, 1.5, 0), (3, 2.0, 1), (4, 2.5, 0)):
        b = BackboneConfig(depth_exponent=d, width_multiplier=wm,
                           block_family=blk, embedding_size=16)
        meta = _count_params_meta(b)
        real = int(sum(p.numel() for p in build_backbone(b).parameters()))
        assert meta == real, "meta=%d real=%d at d=%d wm=%s blk=%d" % (
            meta, real, d, wm, blk)

    # and it must reproduce the corner table in 02_TECHNICAL sec.4 (ResNet family)
    documented = {(3, 1.5): 300192, (4, 1.5): 2450144,
                  (5, 1.5): 17734112, (6, 3.0): 214441248}
    for (d, wm), n in documented.items():
        got = _count_params_meta(BackboneConfig(depth_exponent=d,
                                                width_multiplier=wm,
                                                block_family=0,
                                                embedding_size=16))
        assert got == n, "documented depth%d width%s = %d, got %d" % (d, wm, n, got)

    # the deep corner must TRIP the RAM warning (214 M params ~ 2.6 GB)
    cfg = ExperimentConfig()                       # depth range [3, 6]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sizes = estimate_model_sizes(cfg, skip_search=False)
    assert sizes["max_params"] == 214441248, sizes["max_params"]
    assert sizes["max_ram_gb_weights_and_optimizer"] > 1.0, sizes
    assert any("OOM-killer" in str(x.message) for x in w), \
        "the >1 GB corner must warn about the uncatchable SIGKILL"

    # EXACTLY 4 corners (2 depths x 2 widths), each reporting the WORST block
    # family -- smoke_test_end_to_end.py [D] asserts len(corners) == 4, and the
    # worst family is the safety-relevant number anyway. The per-family breakdown
    # lives alongside in corners_by_family.  [ADDED 5]
    assert len(sizes["corners"]) == 4, sizes["corners"]
    assert set(sizes["corners"]) == {"depth3_width1.5", "depth3_width3.0",
                                     "depth6_width1.5", "depth6_width3.0"}, \
        sorted(sizes["corners"])
    # the corner value IS the ResNet (heavier) family, not the ResNeXt one
    assert sizes["corners"]["depth6_width3.0"] == 214441248
    assert sizes["corners_by_family"]["depth6_width3.0_blk0"] == 214441248
    assert sizes["corners_by_family"]["depth6_width3.0_blk1"] == 73937632
    assert sizes["corners"]["depth6_width3.0"] == max(
        sizes["corners_by_family"]["depth6_width3.0_blk0"],
        sizes["corners_by_family"]["depth6_width3.0_blk1"]), \
        "each corner must report the WORST family the search can sample"
    small, big = min(sizes["corners"].values()), max(sizes["corners"].values())
    assert big > 50 * small, (small, big)          # the explosion is real

    # skip_search reports the single configured architecture instead
    s2 = estimate_model_sizes(cfg, skip_search=True)
    assert len(s2["corners"]) == 1
    ok("D", "meta counts == real counts; 4 corners at the WORST family reproduce "
            "the docs (300192 / 2450144 / 17734112 / 214441248); per-family "
            "breakdown kept; OOM warning fires")


# --------------------------------------------------------------------------- #
# [E] build_traces, synthetic
# --------------------------------------------------------------------------- #
def check_E():
    d = tmpdir("E")
    cfg = ExperimentConfig.from_dict(toy_config_dict(d / "out", d / "cache"))
    traces, conds, fs = build_traces(cfg)
    assert len(traces) == 4, len(traces)           # [2, 2] per class
    assert conds == [0, 0, 1, 1], conds
    assert abs(fs - 50.0) < 1e-9
    assert traces[0].shape[0] == int(200.0 * 50.0)
    assert all(t.dtype == np.float32 for t in traces)
    assert all(float(t.min()) >= 0.0 for t in traces), \
        "an IFR trace is a rate: it must be non-negative"
    # the cache is REUSED, not recomputed, on a second call
    t2, c2, fs2 = build_traces(cfg)
    assert np.allclose(traces[0], t2[0]) and c2 == conds
    ok("E", "synthetic: 4 traces, conditions [0,0,1,1], fs=50, non-negative; "
            "cache reused on re-entry")


# --------------------------------------------------------------------------- #
# [F] build_traces, numpy
# --------------------------------------------------------------------------- #
def check_F():
    d = tmpdir("F")
    data = d / "npz"
    data.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    recs = []
    for i, cond in enumerate([0, 0, 1, 1]):
        tr = np.abs(rng.normal(size=5000)).astype(np.float32)
        np.savez(data / ("w%d.npz" % i), ifr_trace=tr, fs_ifr=50.0)
        # RELATIVE path on purpose: it must resolve against the specs file's dir
        recs.append({"name": "w%d" % i, "condition": cond,
                     "path": "npz/w%d.npz" % i})
    specs_path = d / "specs.json"
    with open(specs_path, "w", encoding="ascii") as fh:
        json.dump(recs, fh)

    cd = toy_config_dict(d / "out", d / "cache")
    cd["data"]["data_mode"] = "numpy"
    cd["data"]["npz_specs"] = str(specs_path)
    cfg = ExperimentConfig.from_dict(cd)
    traces, conds, fs = build_traces(cfg)
    assert len(traces) == 4 and conds == [0, 0, 1, 1] and abs(fs - 50.0) < 1e-9
    assert traces[0].shape[0] == 5000

    # a missing file must be named, not swallowed
    bad = list(recs)
    bad[0] = dict(bad[0], path="npz/does_not_exist.npz")
    with open(specs_path, "w", encoding="ascii") as fh:
        json.dump(bad, fh)
    cfg2 = ExperimentConfig.from_dict(cd)
    cfg2.runtime = replace(cfg2.runtime, cache_dir=str(d / "cache2"))
    try:
        build_traces(cfg2)
        raise AssertionError("a missing .npz must raise FileNotFoundError")
    except FileNotFoundError as ex:
        assert "does_not_exist" in str(ex)

    # a malformed schema must name BOTH accepted key sets
    with open(specs_path, "w", encoding="ascii") as fh:
        json.dump([{"name": "x", "condition": 0}], fh)      # no path, no npz_path
    cfg3 = ExperimentConfig.from_dict(cd)
    cfg3.runtime = replace(cfg3.runtime, cache_dir=str(d / "cache3"))
    try:
        build_traces(cfg3)
        raise AssertionError("a record with no 'path' and no 'npz_path' must raise")
    except ValueError as ex:
        assert "path" in str(ex) and "npz_path" in str(ex), str(ex)

    # [ADDED 11] The Topic-1 schema must ALSO load. generate_burst_data.py writes
    # {"npz_path", "condition", "tag"}, NOT the {"path", "condition", "name"} of
    # 03_USAGE.md sec.5 -- so following the Augmentation README's own workflow
    # (generate the burst dataset, then point the pipeline at its
    # burst_specs.json) fed this driver a file it used to REJECT. These are the
    # exact key names generate_burst_data.py emits.
    topic1 = [{"npz_path": str(data / ("w%d.npz" % i)), "condition": c,
               "tag": "burst_%d" % i}
              for i, c in enumerate([0, 0, 1, 1])]
    t1_path = d / "burst_specs.json"
    with open(t1_path, "w", encoding="ascii") as fh:
        json.dump(topic1, fh)
    cd_t1 = dict(cd)
    cd_t1["data"] = dict(cd["data"], npz_specs=str(t1_path))
    cfg_t1 = ExperimentConfig.from_dict(cd_t1)
    cfg_t1.runtime = replace(cfg_t1.runtime, cache_dir=str(d / "cache_t1"))
    tr_t1, cond_t1, fs_t1 = build_traces(cfg_t1)
    assert len(tr_t1) == 4 and cond_t1 == [0, 0, 1, 1] and abs(fs_t1 - 50.0) < 1e-9, \
        "generate_burst_data.py's burst_specs.json must feed this driver directly"

    # and 'name'/'tag' is OPTIONAL: it falls back to the .npz file stem
    minimal = [{"npz_path": str(data / ("w%d.npz" % i)), "condition": c}
               for i, c in enumerate([0, 1])]
    m_path = d / "minimal_specs.json"
    with open(m_path, "w", encoding="ascii") as fh:
        json.dump(minimal, fh)
    cd_m = dict(cd)
    cd_m["data"] = dict(cd["data"], npz_specs=str(m_path))
    cfg_m = ExperimentConfig.from_dict(cd_m)
    cfg_m.runtime = replace(cfg_m.runtime, cache_dir=str(d / "cache_m"))
    tr_m, cond_m, _ = build_traces(cfg_m)
    assert len(tr_m) == 2 and cond_m == [0, 1]

    ok("F", "numpy: BOTH schemas load -- the documented {name,condition,path} AND "
            "the {tag,condition,npz_path} generate_burst_data.py actually emits; "
            "name is optional (falls back to the file stem); relative paths "
            "resolved; missing file and keyless record raise with the fix named")


# --------------------------------------------------------------------------- #
# [G] build_traces, real (the previously-unwired branch)  [ADDED 8]
# --------------------------------------------------------------------------- #
def check_G():
    d = tmpdir("G")
    specs = [{"folder": "/fake/ptrain_Control00_Well11",
              "base": "ptrain_Control00_Well11_", "condition": 0, "name": "c0"},
             {"folder": "/fake/pgroup02_Well14",
              "base": "pgroup02_Well14_", "condition": 1, "name": "p0"}]
    sp = d / "specs_real.json"
    with open(sp, "w", encoding="ascii") as fh:
        json.dump(specs, fh)

    cd = toy_config_dict(d / "out", d / "cache")
    cd["data"]["data_mode"] = "real"
    cd["data"]["specs_json"] = str(sp)
    cfg = ExperimentConfig.from_dict(cd)

    # without --engine-module the branch must refuse, and NAME the two fixes
    try:
        build_traces(cfg)
        raise AssertionError("real mode without an engine must raise")
    except NotImplementedError as ex:
        assert "engine-module" in str(ex) and "numpy" in str(ex), str(ex)

    # with a stand-in engine it must WORK: this is the branch 02_TECHNICAL sec.15
    # lists as a known gap.
    eng = d / "fake_engine.py"
    eng.write_text(
        "import numpy as np\n"
        "def Neuronal_traces(Char_folder, Char_base, w_size, Gaussian_window,\n"
        "                    t_rec, Visible=False):\n"
        "    rng = np.random.default_rng(abs(hash(Char_base)) % 1000)\n"
        "    return np.abs(rng.normal(size=4000)).astype(np.float32), 50.0\n",
        encoding="ascii")
    sys.path.insert(0, str(d))
    try:
        traces, conds, fs = build_traces(
            cfg, engine_module="fake_engine",
            engine_kwargs={"w_size": 0.02, "gaussian_window": 0.04, "t_rec": 600.0})
    finally:
        sys.path.remove(str(d))
    assert len(traces) == 2 and conds == [0, 1] and abs(fs - 50.0) < 1e-9
    assert traces[0].shape[0] == 4000
    ok("G", "real mode: refuses with a fix-naming NotImplementedError without an "
            "engine, and LOADS correctly through NeuronalTracesProvider with one")


# --------------------------------------------------------------------------- #
# [H] stale-cache fingerprint guard  [ADDED 3]
# --------------------------------------------------------------------------- #
def check_H():
    d = tmpdir("H")
    cache = d / "cache"
    cd = toy_config_dict(d / "out", cache)
    cfg = ExperimentConfig.from_dict(cd)
    traces, _, _ = build_traces(cfg)
    n0 = traces[0].shape[0]
    assert n0 == 10000

    # change the DATA SOURCE but keep the same cache_dir. Without the guard,
    # cache_traces(overwrite=False) would silently hand back the OLD 200 s traces.
    cd2 = dict(cd)
    cd2["data"] = dict(cd["data"], synthetic_duration_s=300.0)
    cfg2 = ExperimentConfig.from_dict(cd2)
    try:
        build_traces(cfg2)
        raise AssertionError(
            "a changed data source against a populated cache must be REFUSED; "
            "silently reusing the stale traces is the bug this guard exists for")
    except ValueError as ex:
        assert "STALE TRACE CACHE" in str(ex) and "--overwrite-cache" in str(ex)

    # --overwrite-cache recomputes, and the traces really are the new ones
    traces2, _, _ = build_traces(cfg2, overwrite_cache=True)
    assert traces2[0].shape[0] == 15000, traces2[0].shape
    ok("H", "stale cache REFUSED with the fix named; --overwrite-cache "
            "recomputes (10000 -> 15000 samples)")


# --------------------------------------------------------------------------- #
# [I] --dry-run trains nothing
# --------------------------------------------------------------------------- #
def check_I():
    import search as S
    d = tmpdir("I")
    cfgp = write_cfg(d / "c.json", toy_config_dict(d / "out", d / "cache"))
    args = cli("--config", cfgp, "--dry-run")
    cfg = apply_cli_overrides(load_config(cfgp), args)

    def boom(*a, **k):
        raise AssertionError("train() was called during a --dry-run")

    # BOTH namespaces hold a reference to train: run_optimization (for the final
    # stage) and search (evaluate_candidate resolves the global name at call
    # time). Patching only one would leave a way for training to happen.
    r_old, s_old = R.train, S.train
    R.train, S.train = boom, boom
    try:
        res = run(cfg, args)
    finally:
        R.train, S.train = r_old, s_old

    assert res["dry_run"] is True
    assert res["budget"]["TOTAL_train_runs"] == 1 * (3 + 3 + 3) + 1, res["budget"]
    out = Path(d / "out") / "t"
    assert not (out / "results.json").exists(), "a dry run must not write results"
    assert not (out / "checkpoints").exists() or \
        not list((out / "checkpoints").glob("*.pt"))
    assert (out / "config_input.json").exists(), \
        "the resolved config should still be recorded"
    ok("I", "--dry-run completed with train() patched to raise in BOTH "
            "namespaces; budget reported; no results.json, no checkpoints")


# --------------------------------------------------------------------------- #
# [J] [K] [L] [P] [Q] end-to-end with --skip-search
# --------------------------------------------------------------------------- #
def check_JKLPQ():
    import matplotlib.pyplot as plt
    d = tmpdir("J")
    cfgp = write_cfg(d / "c.json",
                     toy_config_dict(d / "out", d / "cache", exp="e2e",
                                     n_seeds=2, max_epochs=2))
    args = cli("--config", cfgp, "--skip-search", "--verbose")
    cfg = apply_cli_overrides(load_config(cfgp), args)

    # [Q] headless: a display call anywhere in the pipeline must be impossible
    show_old = plt.show
    plt.show = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("plt.show() was reached: the pipeline is not headless"))
    try:
        res = run(cfg, args)
    finally:
        plt.show = show_old
    ok("Q", "plt.show patched to raise; the whole pipeline ran without touching it")

    out = Path(d / "out") / "e2e"

    # [J] the documented artifacts
    for f in ("config_input.json", "config_best.json", "results.json"):
        assert (out / f).exists(), "missing artifact: %s" % f
    for n in range(2):
        assert (out / "figures" / ("embedding_test_seed_%d.png" % n)).exists()
        assert (out / "checkpoints" / ("final_seed_%d.pt" % n)).exists()
        assert (out / "checkpoints" / ("seed_%d" % n) / "last.pt").exists()
    assert (out / "checkpoints" / "best_model.pt").exists()
    with open(out / "results.json", encoding="ascii") as fh:
        disk = json.load(fh)                       # must be STRICT, parseable JSON
    for k in ("experiment", "device", "seconds", "budget", "model_sizes",
              "batch_rows", "n_traces", "n_classes", "fs", "window_length",
              "n_windows", "config_best", "test"):
        assert k in disk, "results.json missing key %r" % k
    for k in ("ari", "ami", "silhouette", "eff_rank", "per_seed", "n_seeds"):
        assert k in disk["test"], "results.test missing key %r" % k
    assert len(disk["test"]["per_seed"]) == 2
    assert disk["skip_search"] is True and "phase1_arch" not in disk
    ok("J", "every documented artifact written; results.json is strict, "
            "parseable, complete (2 seeds)")

    # [K] the SAVED model rebuilds and REPRODUCES the reported score.
    # Zero hyper-parameters appear in this test: the architecture comes only from
    # the config embedded in the checkpoint.
    model, ckpt = rebuild_model_from_checkpoint(
        str(out / "checkpoints" / "best_model.pt"))
    cfg_best = ExperimentConfig.from_json(out / "config_best.json")
    traces, conds, fs = build_traces(cfg_best)
    splits = make_time_segment_splits(traces, conds, fs, cfg_best.data,
                                      base_seed=int(cfg_best.runtime.seed))
    Z, y = embed_clean_windows(model, splits.test, torch.device("cpu"))
    E = int(cfg_best.backbone.embedding_size)
    assert Z.shape == (len(splits.test.index), E), Z.shape
    assert np.allclose(np.linalg.norm(Z, axis=1), 1.0, atol=1e-5), \
        "l2_normalize=True: every embedding row must lie on the unit sphere"
    m = clustering_metrics(Z, y, seed=int(cfg_best.eval.kmeans_seed),
                           n_clusters=int(disk["n_classes"]),
                           n_init=int(cfg_best.eval.kmeans_n_init),
                           silhouette_metric=cfg_best.eval.silhouette_metric)
    winner = max(disk["test"]["per_seed"], key=lambda s: s["best_val_ari"])
    assert abs(float(m["ari"]) - float(winner["test_ari"])) < 1e-9, \
        ("the rebuilt best_model.pt scores ARI %.6f but results.json reports "
         "%.6f for that seed" % (m["ari"], winner["test_ari"]))
    ok("K", "best_model.pt rebuilds from its EMBEDDED config alone and "
            "reproduces the reported test ARI exactly (%.4f)" % m["ari"])

    # [L] leakage: no test window may overlap any train/val window of the SAME trace
    cov = splits.coverage
    n_cmp = 0
    for (ti, s, e, _c) in cov["test"]:
        for other in ("train", "val"):
            for (tj, s2, e2, _c2) in cov[other]:
                if ti != tj:
                    continue
                n_cmp += 1
                assert e <= s2 or e2 <= s, \
                    "LEAKAGE: test window [%d,%d) of trace %d overlaps a %s " \
                    "window [%d,%d)" % (s, e, ti, other, s2, e2)
    assert n_cmp > 0
    ok("L", "leakage-free: %d test-vs-(train,val) interval comparisons, zero "
            "overlaps" % n_cmp)

    # [P] reproducibility: same seed, same everything -> the identical test ARI
    cfgp2 = write_cfg(d / "c2.json",
                      toy_config_dict(d / "out2", d / "cache2", exp="e2e",
                                      n_seeds=2, max_epochs=2))
    args2 = cli("--config", cfgp2, "--skip-search")
    res2 = run(apply_cli_overrides(load_config(cfgp2), args2), args2)
    a1 = [s["test_ari"] for s in res["test"]["per_seed"]]
    a2 = [s["test_ari"] for s in res2["test"]["per_seed"]]
    assert a1 == a2, "same seed gave different test ARIs: %r vs %r" % (a1, a2)
    ok("P", "reproducible: two independent runs at the same seed give the "
            "identical per-seed test ARI %r" % [round(v, 6) for v in a1])


# --------------------------------------------------------------------------- #
# [M] best_model.pt is selected on VALIDATION, never on test  [ADDED 2]
# --------------------------------------------------------------------------- #
def check_M():
    d = tmpdir("M")
    cd = toy_config_dict(d / "out", d / "cache", exp="sel", n_seeds=2,
                         max_epochs=1)
    cfg = ExperimentConfig.from_dict(cd)
    traces, conds, fs = build_traces(cfg)
    splits = make_time_segment_splits(traces, conds, fs, cfg.data, base_seed=0)

    # Inject a DISAGREEMENT: seed 0 wins on validation, seed 1 wins on test.
    # A test-based selection would copy final_seed_1.pt; a validation-based one
    # copies final_seed_0.pt. The check therefore cannot pass vacuously.
    val_ari = {0: 0.90, 1: 0.10}
    test_ari = {0: 0.10, 1: 0.90}
    calls = {"n": 0}

    def fake_train(cfg_, tr, va, dev, seed, ckpt_dir=None, verbose=False):
        i = calls["n"]
        calls["n"] += 1
        model = build_backbone(cfg_.backbone)
        hist = [{"epoch": 1, "train_loss": 0.5, "ari": val_ari[i], "ami": 0.5,
                 "silhouette": 0.5, "n_triplets": 10, "lr": 1e-3, "seconds": 0.1,
                 "health": {"min_std": 0.1, "mean_std": 0.2, "eff_rank": 3.0,
                            "mean_pairwise_cos": 0.5}}]
        return model, hist

    def fake_eval(model, ds, dev, out_path, seed=0, n_clusters=None,
                  eval_cfg=None, title=None):
        i = calls["n"] - 1
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"png")
        return {"ari": test_ari[i], "ami": 0.5, "silhouette": 0.5,
                "labels_pred": np.zeros(3, dtype=np.int64),
                "n_clusters": 2, "Z": np.zeros((3, 8), dtype=np.float32),
                "y": np.zeros(3, dtype=np.int64),
                "health": {"min_std": 0.1, "mean_std": 0.2, "eff_rank": 3.0,
                           "mean_pairwise_cos": 0.5},
                "n_windows": 3, "figure": str(out_path)}

    t_old, e_old = R.train, R.evaluate_and_plot
    R.train, R.evaluate_and_plot = fake_train, fake_eval
    try:
        out = Path(d / "out") / "sel"
        test = run_final(cfg, splits, torch.device("cpu"), 2, out, verbose=False)
    finally:
        R.train, R.evaluate_and_plot = t_old, e_old

    # the file that got copied must be seed 0's (the VALIDATION winner)
    best = load_checkpoint(out / "checkpoints" / "best_model.pt")
    s0 = load_checkpoint(out / "checkpoints" / "final_seed_0.pt")
    s1 = load_checkpoint(out / "checkpoints" / "final_seed_1.pt")
    assert best["extra"]["seed"] == s0["extra"]["seed"], \
        ("best_model.pt is seed %r; it must be the VALIDATION winner (seed %r), "
         "not the TEST winner (seed %r). Selecting on test would leak the "
         "held-out split into model selection."
         % (best["extra"]["seed"], s0["extra"]["seed"], s1["extra"]["seed"]))
    assert best["extra"]["seed"] != s1["extra"]["seed"]
    assert abs(best["best_metric"]["val_ari"] - 0.90) < 1e-9
    assert abs(best["best_metric"]["test_ari"] - 0.10) < 1e-9, \
        "sanity: the selected model is indeed the WORSE one on test"
    assert test["best_model_selected_on"].startswith("validation")
    # and the reported test spread is over the two seeds
    assert abs(test["ari"]["mean"] - 0.5) < 1e-9
    ok("M", "NON-VACUOUS: with val-winner=seed0 and test-winner=seed1, "
            "best_model.pt is seed0 (val ARI 0.90, test ARI 0.10)")


# --------------------------------------------------------------------------- #
# [N] [O] search-phase wiring, with the phases faked so the wiring is isolated
# --------------------------------------------------------------------------- #
def check_NO():
    import search as S
    d = tmpdir("N")
    cd = toy_config_dict(d / "out", d / "cache", exp="w")
    cd["backbone"]["dropout"] = 0.25          # [O]: a NON-zero input dropout
    cfg = ExperimentConfig.from_dict(cd)

    class FakeRes(object):
        def __init__(self, x):
            self.x = x
            self.func_vals = np.array([-0.5, -0.7])
            self.x_iters = [x, x]
            self.trial_log = [{"trial": 0, "objective": -0.7}]

    seen = {}

    def fake_arch(c, sp, dev, space=None, verbose=False):
        seen["arch_dropout"] = float(c.backbone.dropout)
        return FakeRes([np.int64(4), np.float64(2.25), 1, np.int64(12)])

    def fake_train_search(c, sp, dev, best_arch, verbose=False):
        seen["train_dropout"] = float(c.backbone.dropout)
        seen["train_arch"] = dict(best_arch)
        # margin, lr, u1 = 1-b1, u2 = 1-b2, wd
        return FakeRes([0.42, 0.005, 0.1, 0.001, 7e-4])

    def fake_reg(c, sp, dev, verbose=False):
        seen["reg_dropout"] = float(c.backbone.dropout)
        seen["reg_wd_in"] = float(c.train.weight_decay)
        return FakeRes([0.15, 3e-3])          # dropout, weight_decay

    old = (S.search_architecture, S.search_training, S.search_regularization,
           S.plot_objective_pdp)
    S.search_architecture = fake_arch
    S.search_training = fake_train_search
    S.search_regularization = fake_reg
    S.plot_objective_pdp = lambda res, p, dpi=130: None
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            best, report = run_search_phases(cfg, None, torch.device("cpu"),
                                             tmpdir("N") / "figs")
    finally:
        (S.search_architecture, S.search_training, S.search_regularization,
         S.plot_objective_pdp) = old

    # [O] dropout pinned to 0 for phases 1 AND 2 (search.py only pins it in the
    # ARCH builder; phase 2 would otherwise inherit the config's 0.25)
    assert seen["arch_dropout"] == 0.0, seen
    assert seen["train_dropout"] == 0.0, \
        ("phase 2 received dropout=%r. search.config_from_train_point does NOT "
         "pin dropout, so the driver must -- otherwise phase 1 and phase 2 run "
         "under different regularization and their scores are incomparable."
         % seen["train_dropout"])
    assert any("pinned to 0" in str(x.message) for x in w), \
        "a non-zero input dropout must be announced, not silently overridden"
    ok("O", "dropout pinned to 0 across phases 1 and 2 despite dropout=0.25 in "
            "the input config, and the override is announced")

    # [N] the winners really are carried forward, in the right precedence
    assert best.backbone.depth_exponent == 4 and best.backbone.embedding_size == 12
    assert abs(best.backbone.width_multiplier - 2.25) < 1e-12
    assert best.backbone.block_family == 1
    assert isinstance(best.backbone.depth_exponent, int), \
        "skopt hands back np.int64; it must be cast or the config will not serialize"
    assert seen["train_arch"]["depth_exponent"] == 4
    assert abs(best.train.margin - 0.42) < 1e-12
    assert abs(best.train.lr - 0.005) < 1e-12
    # beta = 1 - u, converted in exactly one place
    assert abs(best.train.beta1 - 0.9) < 1e-12, best.train.beta1
    assert abs(best.train.beta2 - 0.999) < 1e-12, best.train.beta2
    # regularization runs LAST and its weight_decay WINS over phase 2's
    assert abs(seen["reg_wd_in"] - 7e-4) < 1e-12, "reg should start from phase 2's wd"
    assert abs(best.train.weight_decay - 3e-3) < 1e-12, \
        "the regularization stage's weight_decay must win (it is tuned jointly " \
        "with dropout, which is the point of running it last)"
    assert abs(best.backbone.dropout - 0.15) < 1e-12
    # the config must survive a JSON round-trip (numpy types would break it)
    assert ExperimentConfig.from_dict(best.to_dict()) == best
    ok("N", "phase winners carried forward with correct precedence: arch(4, 2.25, "
            "blk1, E12) -> HPs(m .42, lr 5e-3, b1 .9, b2 .999) -> reg(drop .15, "
            "wd 3e-3 OVERRIDES phase-2 7e-4); config JSON-round-trips")


# --------------------------------------------------------------------------- #
# [R] ASCII purity of the driver and its whole import chain
# --------------------------------------------------------------------------- #
def check_R():
    here = Path(__file__).resolve().parent
    chain = ["run_optimization.py", "config.py", "backbone.py", "augmentation.py",
             "data_pipeline.py", "preprocessing_cache.py", "data_splits.py",
             "metrics.py", "checkpoint.py", "inference.py", "train.py",
             "evaluate.py", "search.py"]
    for name in chain:
        p = here / name
        if not p.exists():
            continue
        data = p.read_bytes()
        bad = [(i, hex(b)) for i, b in enumerate(data) if b > 127]
        assert not bad, \
            ("%s carries non-ASCII bytes %r. A cp1252 transfer (MobaXterm, "
             "copy-paste, scp from Windows) turns these into a SyntaxError that "
             "only surfaces at job-submission time on the cluster."
             % (name, bad[:5]))
    ok("R", "the driver and all %d modules in its import chain are pure ASCII"
            % len(chain))


# --------------------------------------------------------------------------- #
# [S] stale-resume guard  [ADDED 7]
# --------------------------------------------------------------------------- #
def check_S():
    d = tmpdir("S") / "ck"
    d.mkdir(parents=True, exist_ok=True)
    for f in ("last.pt", "best.pt", "epoch_0003.pt"):
        (d / f).write_bytes(b"stale")

    _prepare_ckpt_dir(d, resume=True)
    assert (d / "last.pt").exists(), "--resume must KEEP last.pt"

    _prepare_ckpt_dir(d, resume=False)
    assert not (d / "last.pt").exists(), \
        ("without --resume a stale last.pt must be cleared: train() resumes "
         "AUTOMATICALLY from it, so a re-run with a different architecture would "
         "try to load the old weights into the new model")
    assert not (d / "best.pt").exists()
    assert not (d / "epoch_0003.pt").exists()
    ok("S", "stale last.pt/best.pt/epoch_*.pt cleared without --resume, kept with it")


# --------------------------------------------------------------------------- #
# [T] final seeds cannot collide with a search trial's seed block  [ADDED 6]
# --------------------------------------------------------------------------- #
def check_T():
    s0, Ns = 0, 3
    # search trial t owns [s0 + t*Ns, s0 + t*Ns + Ns). Take a budget far beyond
    # anything realistic (100k trials) and show the blocks still cannot reach the
    # final block.
    max_trial_seed = s0 + 100_000 * Ns + (Ns - 1)
    finals = [s0 + FINAL_SEED_OFFSET + n for n in range(Ns)]
    assert min(finals) > max_trial_seed, \
        ("the final seeds %r collide with the search's seed space (max %d): the "
         "final model would be fitted on the very draws that selected it"
         % (finals, max_trial_seed))
    ok("T", "final seeds %r are disjoint from every search trial block "
            "(max reachable %d)" % (finals, max_trial_seed))


# --------------------------------------------------------------------------- #
# [U] end-to-end through the REAL skopt search
# --------------------------------------------------------------------------- #
def check_U():
    d = tmpdir("U")
    cfgp = write_cfg(d / "c.json",
                     toy_config_dict(d / "out", d / "cache", exp="full",
                                     n_seeds=1, max_epochs=1, n_calls=3))
    args = cli("--config", cfgp, "--verbose")
    cfg = apply_cli_overrides(load_config(cfgp), args)
    res = run(cfg, args)

    out = Path(d / "out") / "full"
    for k in ("phase1_arch", "phase2_train", "regularization"):
        assert k in res, "missing search report: %s" % k
        assert len(res[k]["trial_log"]) >= 1
    # the winning config on disk must BE the winners the phases reported
    cb = ExperimentConfig.from_json(out / "config_best.json")
    assert cb.backbone.depth_exponent == res["phase1_arch"]["best"]["depth_exponent"]
    assert cb.backbone.block_family == res["phase1_arch"]["best"]["block_family"]
    assert abs(cb.train.lr - res["phase2_train"]["best"]["lr"]) < 1e-12
    assert abs(cb.backbone.dropout
               - res["regularization"]["best"]["dropout"]) < 1e-12
    assert abs(cb.train.weight_decay
               - res["regularization"]["best"]["weight_decay"]) < 1e-12
    # every objective is FINITE (a NaN would abort the GP surrogate)
    for k in ("phase1_arch", "phase2_train", "regularization"):
        for rec in res[k]["trial_log"]:
            o = rec["objective"]
            assert o is not None and np.isfinite(o), \
                "trial objective must never be NaN: %r" % rec
            assert o <= 1.0
    assert (out / "checkpoints" / "best_model.pt").exists()
    assert (out / "figures" / "embedding_test_seed_0.png").exists()
    ok("U", "real skopt e2e: 3 phases ran, config_best carries every winner, no "
            "objective is NaN, model saved")


# --------------------------------------------------------------------------- #
def main():
    global _TMP
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip [U], the real-skopt end-to-end")
    a = ap.parse_args()

    print("=" * 74)
    print("smoke_test_run_optimization.py")
    print("=" * 74)

    tmp = tempfile.mkdtemp(prefix="dsn_smoke_")
    _TMP = tmp
    try:
        print("\n-- pre-flight arithmetic --")
        check_A(); check_B(); check_C(); check_D()
        print("\n-- data path --")
        check_E(); check_F(); check_G(); check_H()
        print("\n-- orchestration --")
        check_I()
        check_JKLPQ()
        check_M()
        check_NO()
        print("\n-- HPC / hygiene --")
        check_R(); check_S(); check_T()
        if not a.quick:
            print("\n-- full search end-to-end --")
            check_U()
        else:
            print("\n-- [U] skipped (--quick) --")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 74)
    print("ALL %d RUN_OPTIMIZATION SMOKE CHECKS PASSED: %s"
          % (len(_PASSED), " ".join(sorted(set(_PASSED)))))
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
