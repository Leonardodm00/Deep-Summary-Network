"""
smoke_test_data_splits.py

Standalone correctness checks for preprocessing_cache.py + data_splits.py
(Stage 2). Synthetic data only, CPU only.

Run:
    python3 smoke_test_data_splits.py

Checks (looped over C in {2, 3, 4} phenotypes unless noted):
  [A] segment_bounds: 3 half-open segments, disjoint, exactly covering [0, L).
  [B] window_starts mirrors MEAWindowDataset's tiling rule (cross-check vs the
      Dataset's own .index -> proves the provenance has not drifted).
  [C] Cache round-trip: cached traces reload bit-for-bit; fs / condition / name
      preserved; caching is idempotent under overwrite=False.
  [D] Leakage: per original trace, train / val / test window intervals are
      pairwise DISJOINT (no shared sample).
  [E] No boundary straddle: every window lies inside its own time segment.
  [F] Every split contains every phenotype with >= 2 windows (silhouette-safe).
  [G] Stride semantics: train windows OVERLAP; val / test windows are DISJOINT.
  [H] Multi-class labels 0..C-1 propagate through TripletCollator; destroyed
      surrogates get labels >= unique_label_base.
  [I] Helpful error when window_s exceeds every segment (0 windows).
"""

import os
import sys
import tempfile
import warnings
from dataclasses import replace

import numpy as np

from config import DataConfig, AugmentationConfig
from preprocessing_cache import cache_traces, load_cached_traces, manifest_path
from data_splits import (
    MultiClassSyntheticProvider, make_synthetic_specs,
    segment_bounds, window_starts, make_time_segment_splits,
)
from data_pipeline import (
    ConditionBalancedBatchSampler, TripletCollator,
)

_SPLIT_INDEX = {"train": 0, "val": 1, "test": 2}
_UNIQUE_BASE = 1_000_000            # TripletCollator default unique_label_base

# small, fast configuration (kept comfortably above k_min for the spline knots)
_DURATION_S = 120.0
_FS = 50.0
_WINDOW_S = 8.0
_TRAIN_STRIDE_S = 4.0               # < window -> overlap
_EVAL_STRIDE_S = 8.0               # == window -> disjoint
_FRACTIONS = (0.6, 0.2, 0.2)


def _make_cfg(n_per_class):
    aug = replace(AugmentationConfig(fs=_FS),
                  n_positives=3, n_negatives=3, shift_magnitude_s=2.0)
    return DataConfig(
        data_mode="synthetic", synthetic_n_per_class=tuple(n_per_class),
        synthetic_duration_s=_DURATION_S, synthetic_fs=_FS,
        window_s=_WINDOW_S, train_stride_s=_TRAIN_STRIDE_S,
        eval_stride_s=_EVAL_STRIDE_S, split_fractions=_FRACTIONS,
        augmentation=aug,
    )


def _intervals_overlap(a, b):
    """Half-open [s, e) overlap test."""
    return a[0] < b[1] and b[0] < a[1]


# --------------------------------------------------------------------------- #
def check_segment_bounds():
    for L in (10, 6000, 6001, 999):
        b = segment_bounds(L, _FRACTIONS)
        assert b[0][0] == 0 and b[-1][1] == L, b
        # contiguous + ordered + covering
        for k in range(3):
            assert b[k][0] <= b[k][1], b
        assert b[0][1] == b[1][0] and b[1][1] == b[2][0], b
        covered = sum(e - s for (s, e) in b)
        assert covered == L, (covered, L)
    print("  [A] segment_bounds: disjoint, ordered, covering [0, L) OK")


def check_cache_roundtrip(C, n_per_class, cache_dir):
    provider = MultiClassSyntheticProvider(
        n_classes=C, duration_s=_DURATION_S, fs=_FS, seed=0)
    specs = make_synthetic_specs(n_per_class)

    # reference traces generated directly (provider is deterministic)
    ref = {}
    for spec in specs:
        tr, fs = provider(*spec["args"])
        ref[spec["name"]] = (np.asarray(tr, dtype=np.float32), float(fs), spec["condition"])

    man1 = cache_traces(specs, provider, cache_dir)
    man2 = cache_traces(specs, provider, cache_dir)          # idempotent (skip recompute)
    assert man1 == man2, "cache not idempotent under overwrite=False"
    assert manifest_path(cache_dir).exists()

    traces, conditions, fs = load_cached_traces(cache_dir)
    assert fs == _FS, fs
    assert len(traces) == len(specs)
    # manifest order == specs order; verify bit-for-bit + metadata
    for i, spec in enumerate(specs):
        rtr, rfs, rcond = ref[spec["name"]]
        assert np.array_equal(traces[i], rtr), "trace %s not bit-identical" % spec["name"]
        assert conditions[i] == rcond
    print("  [C] C=%d cache round-trip bit-for-bit + idempotent OK" % C)
    return traces, conditions, fs


def check_splits(C, n_per_class, traces, conditions, fs, cfg):
    bundle = make_time_segment_splits(traces, conditions, fs, cfg, base_seed=0)
    W = bundle.window_length
    cov = bundle.coverage
    ds_by_split = {"train": bundle.train, "val": bundle.val, "test": bundle.test}

    # [B] provenance vs the Dataset's own .index (no drift): reconstruct
    #     (orig_trace_idx, orig_start) from each dataset's index using seg_bounds.
    for name in ("train", "val", "test"):
        ds = ds_by_split[name]
        si = _SPLIT_INDEX[name]
        from_index = []
        for (seg_list_idx, seg_rel_start, cond) in ds.index:
            seg_start = bundle.seg_bounds[seg_list_idx][si][0]
            from_index.append((seg_list_idx, seg_start + seg_rel_start))
        from_cov = [(ti, s) for (ti, s, e, c) in cov[name]]
        assert from_index == from_cov, "provenance drift in split '%s'" % name
        # also confirm the segment-relative starts equal window_starts()
        for seg_list_idx in range(len(traces)):
            s, e = bundle.seg_bounds[seg_list_idx][si]
            expected = window_starts(e - s, W, ds.stride)
            got = [rel for (ti, rel, c) in ds.index if ti == seg_list_idx]
            assert got == expected, (name, seg_list_idx, got, expected)

    # [E] no boundary straddle: every window inside its own segment
    for name in ("train", "val", "test"):
        si = _SPLIT_INDEX[name]
        for (ti, s, e, c) in cov[name]:
            seg_s, seg_e = bundle.seg_bounds[ti][si]
            assert seg_s <= s and e <= seg_e, (name, ti, s, e, seg_s, seg_e)

    # [D] leakage: per trace, cross-split window intervals are pairwise disjoint
    n_traces = len(traces)
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for ti in range(n_traces):
        per = {nm: [(s, e) for (t, s, e, c) in cov[nm] if t == ti]
               for nm in ("train", "val", "test")}
        for a, b in pairs:
            for ia in per[a]:
                for ib in per[b]:
                    assert not _intervals_overlap(ia, ib), \
                        "leakage: %s window %s overlaps %s window %s (trace %d)" % (a, ia, b, ib, ti)

    # [F] every split has every phenotype with >= 2 windows
    for name in ("train", "val", "test"):
        counts = {}
        for (ti, s, e, c) in cov[name]:
            counts[c] = counts.get(c, 0) + 1
        for cls in range(C):
            assert counts.get(cls, 0) >= 2, \
                "split '%s' has %d windows for class %d (need >= 2)" % (name, counts.get(cls, 0), cls)

    # [G] stride semantics: train overlaps within a trace; val/test disjoint
    def _consecutive(name, ti):
        return sorted([(s, e) for (t, s, e, c) in cov[name] if t == ti])
    overlap_seen = False
    for ti in range(n_traces):
        tr = _consecutive("train", ti)
        for j in range(len(tr) - 1):
            if tr[j + 1][0] < tr[j][1]:
                overlap_seen = True
        for name in ("val", "test"):
            seq = _consecutive(name, ti)
            for j in range(len(seq) - 1):
                assert seq[j + 1][0] >= seq[j][1], \
                    "%s windows overlap (trace %d): %s, %s" % (name, ti, seq[j], seq[j + 1])
    assert overlap_seen, "expected overlapping train windows but found none"
    print("  [B,D,E,F,G] C=%d splits: no-drift, leakage-free, per-class coverage, "
          "stride semantics OK" % C)
    return bundle


def check_label_propagation(C, bundle):
    ds = bundle.train
    sampler = ConditionBalancedBatchSampler(
        ds.conditions_per_item, per_condition=2, n_batches=1, seed=0)
    collate = TripletCollator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # ignore augmentation redraws
        batch_indices = next(iter(sampler))
        items = [ds[i] for i in batch_indices]
        X, y, metas = collate(items)
    y_np = y.numpy()
    pos_labels = set(int(v) for v in y_np if v < _UNIQUE_BASE)
    assert pos_labels == set(range(C)), (pos_labels, set(range(C)))
    assert (y_np >= _UNIQUE_BASE).any(), "no unique negative labels present"
    assert X.shape[1] == bundle.window_length, (X.shape, bundle.window_length)
    print("  [H] C=%d labels 0..C-1 + unique negatives propagate through collator OK" % C)


def check_empty_split_error():
    cfg = _make_cfg((2, 2))
    cfg.window_s = 200.0            # 200 s * 50 Hz = 10000 samples > any segment
    provider = MultiClassSyntheticProvider(n_classes=2, duration_s=_DURATION_S, fs=_FS, seed=0)
    specs = make_synthetic_specs((2, 2))
    with tempfile.TemporaryDirectory() as d:
        cache_traces(specs, provider, d)
        traces, conditions, fs = load_cached_traces(d)
        raised = False
        try:
            make_time_segment_splits(traces, conditions, fs, cfg)
        except ValueError as ex:
            raised = True
            assert "0 windows" in str(ex), str(ex)
    assert raised, "expected ValueError for oversized window_s"
    print("  [I] oversized window_s raises a helpful 0-windows error OK")


def main():
    print("Running data_splits + preprocessing_cache smoke tests...")
    check_segment_bounds()
    n_per_class_by_C = {2: (2, 2), 3: (2, 2, 2), 4: (2, 2, 2, 2)}
    for C in (2, 3, 4):
        n_per_class = n_per_class_by_C[C]
        cfg = _make_cfg(n_per_class)
        with tempfile.TemporaryDirectory() as d:
            traces, conditions, fs = check_cache_roundtrip(C, n_per_class, d)
            bundle = check_splits(C, n_per_class, traces, conditions, fs, cfg)
            check_label_propagation(C, bundle)
    check_empty_split_error()
    print("ALL DATA-SPLIT SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
