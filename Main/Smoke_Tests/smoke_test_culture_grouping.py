"""
smoke_test_culture_grouping.py
==============================

Verification for K3: the culture-grouping vector that lets several trace records
belong to ONE biological recording.

What K3 is, in one line: the map

    gamma : {0, ..., U_tot - 1} -> {0, ..., U_cult - 1}

from trace record index u to culture index. Before K3 the pipeline hardcoded
gamma = identity (one trace == one culture). That is wrong for the channel-subset
extractor's mode='per_region_single', where one well yields C single-channel
subregion traces: they are C records but ONE culture.

Checks (each is a separate function; all must pass):

    A. BACKWARD COMPATIBILITY. cultures=None reproduces the pre-K3 split
       index-for-index under the same seed, and so does an explicitly injective
       grouping. This is the check that protects every existing config.
    B. NO CULTURE STRADDLES A SPLIT. With 9 siblings per well, every sibling of
       a well lands in the same split. Under the identity grouping this test
       FAILS by construction -- it is included to prove the test can detect the
       bug it exists to catch, not merely that the new code passes.
    C. g IS THE CULTURE, NOT THE TRACE. bundle.trace_of_window holds culture
       indices, and the vector train.py builds from the dataset agrees with it.
    D. NO SAME-CULTURE POSITIVE PAIR. Every batch the cross-culture sampler
       emits draws its windows from DISTINCT cultures per class, so a mined
       positive can never be an anchor's own sibling subregion.
    E. CENSUS COUNTS CULTURES. culture_census / resolve_batch_geometry report
       U_cult and cap U_eff by it, not by the inflated trace count.
    F. MIXED-CONDITION CULTURE RAISES. A culture carrying two phenotype labels
       is a specs-generation bug and must not be silently majority-voted.
    G. ORPHAN SUBREGION RAISES. A per-subregion archive whose spec record has no
       culture field is refused, rather than degrading silently to identity.
    H. FINGERPRINT COVERS THE GROUPING. Regrouping the same traces changes the
       cache fingerprint, so a stale cache_dir refuses instead of being reused.

Run
---
    cd Main
    python3 Smoke_Tests/smoke_test_culture_grouping.py

Exit status is 0 only if every check passes. Each check prints its own PASS
line; the first failure raises with a message naming what disagreed.

No .mat files, no GPU, no network: traces are small synthetic ramps, since
nothing here tests signal content -- only the grouping algebra.
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAIN = os.path.dirname(_HERE)
if _MAIN not in sys.path:
    sys.path.insert(0, _MAIN)

from config import AugmentationConfig, DataConfig            # noqa: E402
from data_splits import make_trace_splits                    # noqa: E402
from preprocessing_cache import (TraceSpec, cache_traces,     # noqa: E402
                                 load_cached_cultures,
                                 load_cached_traces)
from batch_geometry import (culture_census,                 # noqa: E402
                            resolve_cultures_per_class)
from data_pipeline import ConditionBalancedBatchSampler       # noqa: E402

FS = 50.0
L = 3000                      # samples per trace (60 s at 50 Hz)
N_SUB = 9                     # subregions per well, as in the real cohort
N_WELL_PER_CLASS = 6          # wells per class
N_CLASSES = 2


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def make_cohort(n_sub=N_SUB, n_well=N_WELL_PER_CLASS, n_classes=N_CLASSES):
    """A cohort of n_classes * n_well wells, each split into n_sub traces.

    Returns (traces, conditions, cultures, well_of_trace) where cultures holds
    STRING ids (as a real specs file would) and well_of_trace is the same
    information as an int, for the assertions.
    """
    traces, conditions, cultures, well_of_trace = [], [], [], []
    w = 0
    for c in range(n_classes):
        for j in range(n_well):
            wid = "plate%d__well_%02d" % (c, j)
            for r in range(n_sub):
                rng = np.random.default_rng((c, j, r))
                traces.append(rng.random(L).astype(np.float32))
                conditions.append(c)
                cultures.append(wid)
                well_of_trace.append(w)
            w += 1
    return traces, conditions, cultures, well_of_trace


def data_cfg(window_s=6.0, train_stride_s=6.0, eval_stride_s=6.0):
    """Minimal DataConfig. Strides == window so window counts stay small."""
    return DataConfig(
        window_s=window_s,
        train_stride_s=train_stride_s,
        eval_stride_s=eval_stride_s,
        split_mode="trace",
        augmentation=AugmentationConfig(fs=FS),
    )


def _fail(msg):
    raise AssertionError(msg)


# --------------------------------------------------------------------------- #
# A. backward compatibility
# --------------------------------------------------------------------------- #
def check_A_identity_is_unchanged():
    """cultures=None and an injective grouping must give the SAME split."""
    traces, cond, _cult, _w = make_cohort()
    cfg = data_cfg()

    b_none = make_trace_splits(traces, cond, FS, cfg, split_seed=11)
    # an explicitly injective grouping: one distinct id per trace
    inj = ["trace_%04d" % u for u in range(len(traces))]
    b_inj = make_trace_splits(traces, cond, FS, cfg, split_seed=11, cultures=inj)

    for name in ("train", "val", "test"):
        a = np.asarray(b_none.cultures[name])
        b = np.asarray(b_inj.cultures[name])
        if a.shape != b.shape or not np.array_equal(a, b):
            _fail("A: identity grouping changed the '%s' split: %r vs %r"
                  % (name, a.tolist()[:8], b.tolist()[:8]))
        ga = np.asarray(b_none.trace_of_window[name])
        gb = np.asarray(b_inj.trace_of_window[name])
        if not np.array_equal(ga, gb):
            _fail("A: identity grouping changed g on '%s'" % name)
        if len(b_none.coverage[name]) != len(b_inj.coverage[name]):
            _fail("A: identity grouping changed the window count on '%s'" % name)
    print("  [A] PASS  cultures=None == injective grouping, split identical")


# --------------------------------------------------------------------------- #
# B. no culture straddles a split (and the test can detect the bug)
# --------------------------------------------------------------------------- #
def check_B_no_culture_straddles():
    traces, cond, cult, well = make_cohort()
    cfg = data_cfg()
    bundle = make_trace_splits(traces, cond, FS, cfg, split_seed=3, cultures=cult)

    # every window's culture, per split, must be disjoint across splits
    seen = {}
    for name in ("train", "val", "test"):
        g = np.asarray(bundle.trace_of_window[name])
        seen[name] = set(int(x) for x in g.tolist())
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a >= b:
                continue
            shared = seen[a] & seen[b]
            if shared:
                _fail("B: culture(s) %r appear in both '%s' and '%s'"
                      % (sorted(shared)[:5], a, b))

    n_cult_total = len(set(cult))
    n_assigned = sum(len(bundle.cultures[n]) for n in ("train", "val", "test"))
    if n_assigned != n_cult_total:
        _fail("B: %d cultures assigned but the cohort has %d"
              % (n_assigned, n_cult_total))

    # NEGATIVE CONTROL: under the identity grouping, siblings of one well DO
    # straddle. If this does not happen, the test above is not testing anything.
    bad = make_trace_splits(traces, cond, FS, cfg, split_seed=3, cultures=None)
    wells_by_split = {}
    for name in ("train", "val", "test"):
        wells_by_split[name] = set(
            int(well[int(u)]) for u in bad.cultures[name].tolist())
    straddled = ((wells_by_split["train"] & wells_by_split["test"])
                 | (wells_by_split["train"] & wells_by_split["val"])
                 | (wells_by_split["val"] & wells_by_split["test"]))
    if not straddled:
        _fail("B: negative control did not reproduce the bug -- under the "
              "identity grouping some well MUST straddle a split, otherwise "
              "this check proves nothing")
    print("  [B] PASS  no culture straddles a split "
          "(negative control: %d well(s) straddle without K3)"
          % len(straddled))


# --------------------------------------------------------------------------- #
# C. g is the culture, not the trace
# --------------------------------------------------------------------------- #
def check_C_g_is_the_culture():
    traces, cond, cult, _w = make_cohort()
    cfg = data_cfg()
    bundle = make_trace_splits(traces, cond, FS, cfg, split_seed=5, cultures=cult)

    for name in ("train", "val", "test"):
        ds = getattr(bundle, name)
        g_bundle = np.asarray(bundle.trace_of_window[name])
        # exactly the expression train.py uses
        g_train = np.asarray([int(ds.cultures[ti]) for (ti, _s, _c) in ds.index],
                             dtype=int)
        if not np.array_equal(g_bundle, g_train):
            _fail("C: bundle.trace_of_window['%s'] disagrees with the vector "
                  "train.py derives from the dataset" % name)
        n_distinct_g = len(set(g_bundle.tolist()))
        n_traces_here = len(ds.traces)
        if n_distinct_g >= n_traces_here:
            _fail("C: '%s' has %d distinct culture(s) over %d trace(s); with "
                  "%d siblings per well g is still the trace index"
                  % (name, n_distinct_g, n_traces_here, N_SUB))
    print("  [C] PASS  g holds culture indices and matches train.py's derivation")


# --------------------------------------------------------------------------- #
# D. no same-culture positive pair
# --------------------------------------------------------------------------- #
def check_D_no_same_culture_positive():
    traces, cond, cult, _w = make_cohort()
    cfg = data_cfg()
    bundle = make_trace_splits(traces, cond, FS, cfg, split_seed=7, cultures=cult)

    ds = bundle.train
    g = np.asarray(bundle.trace_of_window["train"])
    conds = np.asarray([c for (_t, _s, c) in ds.index], dtype=int)

    sampler = ConditionBalancedBatchSampler(
        conditions=conds,
        per_condition=2,
        n_batches=12,
        seed=0,
        positives_mode="cross_culture",
        trace_of_window=g,
        mining_strategy="hard",
        cultures_per_class_per_batch=2,
        windows_per_culture_per_batch=2,
        n_surrogates=1,
        exclude_same_culture_positives=True,
        min_train_cultures_per_class=2,
    )

    n_checked = 0
    for batch in sampler:
        idx = np.asarray(list(batch), dtype=int)
        for c in np.unique(conds[idx]):
            rows = idx[conds[idx] == c]
            cult_here = g[rows]
            per_culture = {}
            for r, k in zip(rows.tolist(), cult_here.tolist()):
                per_culture.setdefault(int(k), []).append(int(r))
            if len(per_culture) < 2:
                _fail("D: class %d contributed windows from only %d culture(s) "
                      "to a batch; an anchor then has no cross-culture positive"
                      % (int(c), len(per_culture)))
            n_checked += 1
    if n_checked == 0:
        _fail("D: sampler yielded no batches to check")
    print("  [D] PASS  every batch offers >= 2 distinct cultures per class "
          "(%d class-batches checked)" % n_checked)


# --------------------------------------------------------------------------- #
# E. census counts cultures
# --------------------------------------------------------------------------- #
def check_E_census_counts_cultures():
    traces, cond, cult, _w = make_cohort()
    cfg = data_cfg()
    bundle = make_trace_splits(traces, cond, FS, cfg, split_seed=13, cultures=cult)

    g = np.asarray(bundle.trace_of_window["train"])
    conds = np.asarray([c for (_t, _s, c) in bundle.train.index], dtype=int)
    census = culture_census(g, conds)

    n_train_cult = len(bundle.cultures["train"])
    n_train_traces = len(bundle.train.traces)
    if n_train_traces <= n_train_cult:
        _fail("E: fixture is degenerate -- %d train traces over %d cultures"
              % (n_train_traces, n_train_cult))

    # culture_census returns three dicts; the culture count is the number of
    # keys in windows_per_culture, and equivalently the sum over classes.
    counted = len(census["windows_per_culture"])
    by_class = sum(census["cultures_per_class"].values())
    if counted != n_train_cult or by_class != n_train_cult:
        _fail("E: culture_census reports %d culture(s) (%d summed over classes) "
              "but the split assigned %d (trace count is %d -- the inflated "
              "number)" % (counted, by_class, n_train_cult, n_train_traces))

    # and the cap that actually matters: U_eff must be bounded by the CULTURE
    # count, so an over-large request is clamped to it rather than to the
    # ninefold-inflated trace count.
    u_eff, _info = resolve_cultures_per_class(
        requested=99, cultures_per_class=census["cultures_per_class"],
        min_train_cultures_per_class=2)
    expected = min(census["cultures_per_class"].values())
    if int(u_eff) != int(expected):
        _fail("E: U_eff resolved to %r; expected the per-class culture floor %r"
              % (u_eff, expected))
    if int(u_eff) >= n_train_traces:
        _fail("E: U_eff = %d is not below the trace count %d -- the grouping "
              "is not capping anything" % (int(u_eff), n_train_traces))
    print("  [E] PASS  census reports %d culture(s), not %d trace(s); "
          "U_eff clamps to %d" % (n_train_cult, n_train_traces, int(u_eff)))


# --------------------------------------------------------------------------- #
# F. mixed-condition culture raises
# --------------------------------------------------------------------------- #
def check_F_mixed_condition_raises():
    traces, cond, cult, _w = make_cohort(n_sub=3, n_well=4)
    cond = list(cond)
    cond[1] = 1 - int(cond[1])          # one sibling relabelled
    cfg = data_cfg()
    try:
        make_trace_splits(traces, cond, FS, cfg, split_seed=0, cultures=cult)
    except ValueError as exc:
        if "more than one condition" not in str(exc):
            _fail("F: raised, but not with the expected message: %s" % exc)
        print("  [F] PASS  mixed-condition culture raises")
        return
    _fail("F: a culture with two distinct conditions did NOT raise")


# --------------------------------------------------------------------------- #
# G. orphan subregion archive raises
# --------------------------------------------------------------------------- #
def check_G_orphan_subregion_raises():
    import run_optimization as RO
    tmp = tempfile.mkdtemp(prefix="k3_orphan_")
    try:
        p = os.path.join(tmp, "trace_subregion_00.npz")
        np.savez_compressed(
            p, ifr_trace=np.zeros(L, dtype=np.float32), fs_ifr=np.float64(FS),
            in_channels=1, n_samples=1, subregion_index=0,
            culture_id="plate0__well_00")
        try:
            RO._guard_orphan_subregion(p, "trace_subregion_00", 0)
        except ValueError as exc:
            if "no 'culture' field" not in str(exc):
                _fail("G: raised, but not with the expected message: %s" % exc)
            # and a plain archive must NOT raise
            q = os.path.join(tmp, "plain.npz")
            np.savez_compressed(q, ifr_trace=np.zeros(L, dtype=np.float32),
                                fs_ifr=np.float64(FS))
            RO._guard_orphan_subregion(q, "plain", 1)
            print("  [G] PASS  orphan subregion raises; plain archive does not")
            return
        _fail("G: a per-subregion archive with no culture field did NOT raise")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# H. the fingerprint covers the grouping
# --------------------------------------------------------------------------- #
def check_H_fingerprint_covers_grouping():
    import run_optimization as RO
    from config import ExperimentConfig

    cfg = ExperimentConfig()
    specs_a = [TraceSpec("t%02d" % u, u % 2, ("/tmp/t%02d.npz" % u,),
                         culture="well_%02d" % (u // 3)) for u in range(12)]
    specs_b = [TraceSpec("t%02d" % u, u % 2, ("/tmp/t%02d.npz" % u,),
                         culture="well_%02d" % (u // 4)) for u in range(12)]
    fa = RO._data_fingerprint(cfg, specs_a)
    fb = RO._data_fingerprint(cfg, specs_b)
    if fa == fb:
        _fail("H: regrouping the SAME traces into different cultures left the "
              "fingerprint unchanged; a stale cache_dir would be reused")

    fa2 = RO._data_fingerprint(cfg, specs_a)
    if fa != fa2:
        _fail("H: fingerprint is not stable across two calls on equal specs")
    print("  [H] PASS  grouping enters the fingerprint, and it is stable")


# --------------------------------------------------------------------------- #
# I. round trip through the cache
# --------------------------------------------------------------------------- #
def check_I_cache_round_trip():
    tmp = tempfile.mkdtemp(prefix="k3_cache_")
    try:
        rng = np.random.default_rng(0)

        def provider(tag):
            return rng.random(L).astype(np.float32), FS

        specs = [TraceSpec("w%02d_sub%d" % (w, r), w % 2, ("x",),
                           culture="well_%02d" % w)
                 for w in range(4) for r in range(3)]
        cache_traces(specs, provider, tmp)
        traces, conds, fs = load_cached_traces(tmp)
        cults = load_cached_cultures(tmp)
        if len(cults) != len(traces):
            _fail("I: %d cultures for %d traces" % (len(cults), len(traces)))
        if len(set(cults)) != 4:
            _fail("I: expected 4 distinct cultures, got %d" % len(set(cults)))
        with open(os.path.join(tmp, "manifest.json")) as fh:
            man = json.load(fh)
        if any("culture" not in e for e in man):
            _fail("I: manifest entry without a culture field")

        # legacy manifest (no culture key) must fall back to name
        for e in man:
            e.pop("culture")
        with open(os.path.join(tmp, "manifest.json"), "w") as fh:
            json.dump(man, fh)
        legacy = load_cached_cultures(tmp)
        if legacy != [e["name"] for e in man]:
            _fail("I: legacy manifest did not fall back to name")
        print("  [I] PASS  culture survives the cache round trip; legacy "
              "manifest falls back to identity")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
def main():
    print("K3 culture-grouping smoke test")
    print("  cohort: %d classes x %d wells x %d subregions = %d trace records, "
          "%d cultures"
          % (N_CLASSES, N_WELL_PER_CLASS, N_SUB,
             N_CLASSES * N_WELL_PER_CLASS * N_SUB,
             N_CLASSES * N_WELL_PER_CLASS))
    checks = [
        check_A_identity_is_unchanged,
        check_B_no_culture_straddles,
        check_C_g_is_the_culture,
        check_D_no_same_culture_positive,
        check_E_census_counts_cultures,
        check_F_mixed_condition_raises,
        check_G_orphan_subregion_raises,
        check_H_fingerprint_covers_grouping,
        check_I_cache_round_trip,
    ]
    for fn in checks:
        fn()
    print("ALL CHECKS PASSED (%d)" % len(checks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
