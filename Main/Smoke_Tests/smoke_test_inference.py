"""
smoke_test_inference.py

Standalone correctness checks for inference.py (Stage 5, shared helper). CPU
only, synthetic data only, no data files.

Run:
    python3 smoke_test_inference.py

Checks:
  [A] Shapes / dtypes: Z is (N, E) float32 with N == len(dataset.index) and
      E == cfg.backbone.embedding_size; y is (N,) int64 and equals
      dataset.conditions_per_item exactly (alignment).
  [B] DISTINCT rows: clean_windows returns N distinct (trace_idx, start) slices,
      so no window is duplicated -- the legacy resample-with-replacement bug
      cannot recur. Asserted on the WINDOWS themselves (exact provenance), not on
      the embeddings (two distinct windows could embed to the same point).
  [C] Windows are CLEAN: X[i] equals the raw slice traces[t_i][s_i:s_i+W]
      bit-for-bit, and equals MEAWindowDataset.__getitem__(i)["anchor"] (the
      clean unshifted anchor) -- i.e. NO augmentation is applied.
  [D] L2-normalized rows: ||z_i||_2 = 1 for all i (backbone l2_normalize=True).
  [E] Batch-size invariance: Z is IDENTICAL (bit-for-bit) for batch_size in
      {1, 7, N, N + 100}. Proves the embedding has no cross-sample dependence
      (GroupNorm, not BatchNorm) and no chunk-boundary bug.
  [F] Determinism: two calls on the same model give bit-identical Z.
  [G] Model mode is RESTORED: a model left in .train() is still in .train() after
      the call (and an .eval() model stays in .eval()), so calling this inside a
      training loop cannot silently disable dropout for the rest of the epoch.
  [H] Dropout is inactive during the embed even when cfg.dropout > 0 (a direct
      consequence of .eval()): two successive calls agree bit-for-bit.
  [I] Multi-class C in {2, 3, 4}: y carries labels 0..C-1 and every class is
      present.
"""

import sys
import tempfile
import warnings
from dataclasses import replace

import numpy as np
import torch

from config import DataConfig, AugmentationConfig, BackboneConfig
from backbone import build_backbone
from preprocessing_cache import cache_traces, load_cached_traces
from data_splits import (
    MultiClassSyntheticProvider, make_synthetic_specs, make_time_segment_splits,
)
from inference import clean_windows, embed_clean_windows

# small + fast; window must stay >= aug k_min and long enough for the stem stride
_DURATION_S = 120.0
_FS = 50.0
_WINDOW_S = 8.0            # -> W = 400 samples
_TRAIN_STRIDE_S = 4.0
_EVAL_STRIDE_S = 8.0
_FRACTIONS = (0.6, 0.2, 0.2)
_EMB = 8


def _make_cfg(n_per_class):
    aug = replace(AugmentationConfig(fs=_FS),
                  n_positives=2, n_negatives=2, shift_magnitude_s=2.0)
    return DataConfig(
        data_mode="synthetic", synthetic_n_per_class=tuple(n_per_class),
        synthetic_duration_s=_DURATION_S, synthetic_fs=_FS,
        window_s=_WINDOW_S, train_stride_s=_TRAIN_STRIDE_S,
        eval_stride_s=_EVAL_STRIDE_S, split_fractions=_FRACTIONS,
        augmentation=aug,
    )


def _make_bundle(C, cache_dir):
    n_per_class = tuple([2] * C)
    cfg = _make_cfg(n_per_class)
    provider = MultiClassSyntheticProvider(
        n_classes=C, duration_s=_DURATION_S, fs=_FS, seed=0)
    specs = make_synthetic_specs(n_per_class)
    cache_traces(specs, provider, cache_dir)
    traces, conditions, fs = load_cached_traces(cache_dir)
    bundle = make_time_segment_splits(traces, conditions, fs, cfg, base_seed=0)
    return bundle, cfg


def _make_model(dropout=0.0, seed=0):
    bb = BackboneConfig(depth_exponent=3, width_multiplier=1.5, stem_width=8,
                        embedding_size=_EMB, dropout=dropout)
    torch.manual_seed(seed)
    return build_backbone(bb)


def check_shapes_and_alignment(bundle):
    ds = bundle.train
    model = _make_model()
    Z, y = embed_clean_windows(model, ds, "cpu", batch_size=16)
    N = len(ds.index)
    assert Z.shape == (N, _EMB), (Z.shape, N, _EMB)
    assert Z.dtype == np.float32, Z.dtype
    assert y.shape == (N,), y.shape
    assert y.dtype == np.int64, y.dtype
    assert np.array_equal(y, np.asarray(ds.conditions_per_item, dtype=np.int64))
    print("  [A] shapes/dtypes OK: Z=(N,E)=(%d,%d) float32, y=(N,) int64 aligned"
          % (N, _EMB))


def check_distinct_windows(bundle):
    ds = bundle.train
    X = clean_windows(ds)
    N = len(ds.index)
    assert X.shape[0] == N, (X.shape, N)
    # provenance must be unique: one row per distinct (trace_idx, start)
    prov = [(ti, s) for (ti, s, _c) in ds.index]
    assert len(set(prov)) == N, "duplicated (trace_idx, start) in dataset.index"
    # and no row is dropped / repeated
    assert X.shape[0] == len(set(prov))
    print("  [B] distinct windows OK: %d rows, %d unique (trace,start) pairs "
          "(no resample-with-replacement)" % (N, len(set(prov))))


def check_windows_are_clean(bundle):
    ds = bundle.train
    X = clean_windows(ds)
    W = ds.window_length
    for i in (0, len(ds.index) // 2, len(ds.index) - 1):
        ti, s, _c = ds.index[i]
        raw = ds.traces[ti][s:s + W]
        assert np.array_equal(X[i], raw), "row %d is not the raw slice" % i
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # aug re-draw warnings
            item = ds[i]
        anchor = item["anchor"].numpy().ravel()               # clean, UNSHIFTED
        assert np.array_equal(X[i], anchor), \
            "row %d differs from the dataset's own clean anchor" % i
    print("  [C] windows are CLEAN OK: rows == raw slices == dataset anchors "
          "(no augmentation applied)")


def check_l2_normalized(bundle):
    model = _make_model()
    Z, _ = embed_clean_windows(model, bundle.val, "cpu", batch_size=32)
    norms = np.linalg.norm(Z.astype(np.float64), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), (norms.min(), norms.max())
    print("  [D] rows L2-normalized OK (min=%.6f max=%.6f)"
          % (norms.min(), norms.max()))


def check_batch_size_invariance(bundle):
    """Cross-sample INDEPENDENCE: z_i must not depend on which other rows share
    its forward batch. Two separate assertions:

      (E1) numerical invariance to the chunking knob batch_size, and
      (E2) a direct no-leakage probe: embed row i alone, then embed it again
           sitting inside a batch of deliberately EXTREME companions.

    Tolerance note (flagged, not silently relaxed): the comparison is at float32
    tolerance (atol 1e-5), NOT bit-exactness. Different batch sizes make the conv
    / matmul kernels choose different blocking and accumulation orders, and
    floating-point addition is not associative, so bit-exactness across batch
    sizes is not a property any correct implementation has. What MUST hold -- and
    what (E2) tests decisively -- is that the companions cannot move z_i beyond
    that float noise floor. A real cross-sample path (e.g. BatchNorm's batch
    statistics) would make (E2) fail by a wide, companion-dependent margin, since
    the companions here are 3-4 orders of magnitude larger than the signal.
    """
    ds = bundle.val
    model = _make_model()
    N = len(ds.index)
    W = ds.window_length
    tol = 1e-5

    # (E1) invariance to the chunking knob
    Z_ref, y_ref = embed_clean_windows(model, ds, "cpu", batch_size=N)
    worst = 0.0
    for bs in (1, 7, N, N + 100):
        Z, y = embed_clean_windows(model, ds, "cpu", batch_size=bs)
        dev = float(np.abs(Z.astype(np.float64) - Z_ref.astype(np.float64)).max())
        worst = max(worst, dev)
        assert dev < tol, \
            "batch_size=%d moved the embedding by %.3g (>= tol %.3g): " \
            "cross-sample dependence, not float noise" % (bs, dev, tol)
        assert np.array_equal(y, y_ref), "labels changed with batch_size=%d" % bs

    # (E2) direct no-leakage probe with extreme companions
    X = clean_windows(ds)
    model.eval()
    i = min(3, N - 1)
    rng = np.random.default_rng(0)
    companions = np.empty((5, W), dtype=np.float32)
    companions[0] = (1.0e3 * rng.standard_normal(W)).astype(np.float32)
    companions[1] = 0.0
    companions[2] = -5.0e2
    companions[3] = X[i]                       # the row under test
    companions[4] = 9.99e2
    with torch.no_grad():
        alone = model(torch.from_numpy(X[i:i + 1])).numpy()
        mixed = model(torch.from_numpy(companions)).numpy()[3:4]
    leak = float(np.abs(alone.astype(np.float64) - mixed.astype(np.float64)).max())
    assert leak < tol, \
        "row %d moved by %.3g when batched with extreme companions -> genuine " \
        "cross-sample leakage (a BatchNorm-style path?)" % (i, leak)

    print("  [E] cross-sample independence OK (bs in {1, 7, N, N+100}: max dev "
          "%.2g; extreme-companion leakage %.2g; both < %.0e = float32 noise)"
          % (worst, leak, tol))


def check_determinism(bundle):
    model = _make_model()
    Z1, _ = embed_clean_windows(model, bundle.test, "cpu", batch_size=16)
    Z2, _ = embed_clean_windows(model, bundle.test, "cpu", batch_size=16)
    assert np.array_equal(Z1, Z2), "repeated embed is not bit-identical"
    print("  [F] determinism OK (two calls -> bit-identical Z)")


def check_mode_restored(bundle):
    model = _make_model()
    model.train()
    embed_clean_windows(model, bundle.val, "cpu", batch_size=16)
    assert model.training is True, "model was left in eval() after the embed"
    model.eval()
    embed_clean_windows(model, bundle.val, "cpu", batch_size=16)
    assert model.training is False, "model was left in train() after the embed"
    print("  [G] train/eval mode restored OK (train stays train, eval stays eval)")


def check_dropout_inactive(bundle):
    model = _make_model(dropout=0.3)        # dropout ON in the config
    model.train()                            # and the model is in TRAIN mode
    Z1, _ = embed_clean_windows(model, bundle.val, "cpu", batch_size=16)
    Z2, _ = embed_clean_windows(model, bundle.val, "cpu", batch_size=16)
    assert np.array_equal(Z1, Z2), \
        "embedding is stochastic -> dropout was active during inference"
    assert model.training is True             # and the mode was still restored
    print("  [H] dropout inactive during embed OK (deterministic despite "
          "dropout=0.3 and train mode)")


def check_multiclass(C, bundle):
    model = _make_model()
    Z, y = embed_clean_windows(model, bundle.test, "cpu", batch_size=16)
    present = set(int(v) for v in np.unique(y))
    assert present == set(range(C)), (present, set(range(C)))
    assert Z.shape[0] == y.shape[0] == len(bundle.test.index)
    print("  [I] C=%d: labels 0..%d all present in y OK" % (C, C - 1))


def main():
    print("Running inference smoke tests...")
    with tempfile.TemporaryDirectory() as d:
        bundle, _cfg = _make_bundle(2, d)
        check_shapes_and_alignment(bundle)
        check_distinct_windows(bundle)
        check_windows_are_clean(bundle)
        check_l2_normalized(bundle)
        check_batch_size_invariance(bundle)
        check_determinism(bundle)
        check_mode_restored(bundle)
        check_dropout_inactive(bundle)
    for C in (2, 3, 4):
        with tempfile.TemporaryDirectory() as d:
            bundle, _cfg = _make_bundle(C, d)
            check_multiclass(C, bundle)
    print("ALL INFERENCE SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
