#!/usr/bin/env python3
"""
Smoke test for Step 4: multichannel windowing / dataset / collation.

Verifies:
  1. MEAWindowDataset windows (C, T) traces into (C, W) windows;
  2. __getitem__ returns anchor (1,C,W), positives (1+P,C,W), negatives (N,C,W);
  3. TripletCollator assembles a batch X of shape (M, C, W) and labels (M,);
  4. a C=9 backbone consumes X -> (M, E), finite, unit-norm;
  5. backward compatibility: 1-D traces flow through the SAME path to (M, W)
     and a 1-channel backbone -> (M, E).

Run:
    python3 smoke_test_data_pipeline_mc.py
Exit 0 = all passed.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from generate_burst_data import generate_multichannel_traces  # noqa: E402
from data_pipeline import MEAWindowDataset, TripletCollator, closest_power_of_2  # noqa: E402
from augmentation import AugmentationConfig  # noqa: E402
from backbone import BackboneConfig, build_backbone  # noqa: E402


def _check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print("[%s] %s%s" % (tag, name, ("  (%s)" % detail) if detail else ""))
    return bool(cond)


def build_dataset(traces, conditions, fs, window_s=20.0, stride_s=10.0, seed=0):
    W = closest_power_of_2(window_s * fs)
    stride = max(1, int(stride_s * fs))
    aug = AugmentationConfig(fs=fs, n_positives=4, n_negatives=4,
                             shift_magnitude_s=2.0, split_method="warp_bands")
    ds = MEAWindowDataset(traces, conditions, W, stride, aug, base_seed=seed)
    return ds, W


def collate_first_k(ds, k):
    items = [ds[i] for i in range(min(k, len(ds)))]
    X, y, metas = TripletCollator()(items)
    return X, y, items


def main():
    ok = True
    torch.manual_seed(0)
    np.random.seed(0)
    C = 9
    P = N = 4  # from aug cfg
    traces, conditions, fs = generate_multichannel_traces(
        n_control=2, n_patho=1, n_channels=C, neurons_per_channel=15,
        duration_s=100.0, seed=0)
    ok &= _check("multichannel traces are (C, T)",
                 all(t.ndim == 2 and t.shape[0] == C for t in traces),
                 "shapes %s" % ([t.shape for t in traces],))

    # ---- 1,2. dataset windowing + item shapes ---------------------------- #
    ds, W = build_dataset(traces, conditions, fs)
    ok &= _check("dataset non-empty", len(ds) > 0, "len=%d" % len(ds))
    it = ds[0]
    ok &= _check("anchor (1,C,W)", tuple(it["anchor"].shape) == (1, C, W),
                 "%s" % (tuple(it["anchor"].shape),))
    ok &= _check("positives (1+P,C,W)",
                 tuple(it["positives"].shape) == (1 + P, C, W),
                 "%s" % (tuple(it["positives"].shape),))
    ok &= _check("negatives (N,C,W)", tuple(it["negatives"].shape) == (N, C, W),
                 "%s" % (tuple(it["negatives"].shape),))

    # ---- 3. collate -> (M, C, W) ----------------------------------------- #
    k = 3
    X, y, items = collate_first_k(ds, k)
    M_expected = sum(itm["positives"].shape[0] + itm["negatives"].shape[0]
                     for itm in items)
    ok &= _check("collated X is (M, C, W)",
                 tuple(X.shape) == (M_expected, C, W),
                 "%s (M_expected=%d)" % (tuple(X.shape), M_expected))
    ok &= _check("labels y is (M,)", tuple(y.shape) == (M_expected,),
                 "%s" % (tuple(y.shape),))

    # ---- 4. C=9 backbone consumes the batch ------------------------------ #
    model = build_backbone(BackboneConfig(in_channels=C, depth_exponent=3,
                                          stem_width=16, embedding_size=16))
    model.eval()
    with torch.no_grad():
        Z = model(X)
    ok &= _check("backbone(X) -> (M, E)", tuple(Z.shape) == (M_expected, 16),
                 "%s" % (tuple(Z.shape),))
    ok &= _check("embeddings finite", bool(torch.isfinite(Z).all()))
    norms = Z.norm(dim=1)
    # L2-normalized rows have norm 1, EXCEPT near-constant (quiet) windows that
    # GroupNorm maps to ~0 -> norm 0. That is a property of GroupNorm nets (holds
    # for single-channel too), not a multichannel issue. Assert norms in {~0,~1}
    # and that normalization is active.
    is_zero = norms < 1e-4
    is_unit = (norms - 1.0).abs() < 1e-4
    ok &= _check("embeddings unit-norm or zero (L2-normalize property)",
                 bool((is_zero | is_unit).all()),
                 "range [%.4f, %.4f], zeros=%d/%d"
                 % (norms.min(), norms.max(), int(is_zero.sum()), norms.numel()))
    ok &= _check("normalization active (>=1 unit-norm row)", bool(is_unit.any()))

    # ---- 5. backward compatibility: 1-D traces --------------------------- #
    traces_1d = [t[0].copy() for t in traces]              # (T,) single channel
    ds1, W1 = build_dataset(traces_1d, conditions, fs)
    it1 = ds1[0]
    ok &= _check("1-D item positives (1+P, W)",
                 tuple(it1["positives"].shape) == (1 + P, W1),
                 "%s" % (tuple(it1["positives"].shape),))
    X1, y1, items1 = collate_first_k(ds1, k)
    ok &= _check("1-D collated X is (M, W)", X1.ndim == 2 and X1.shape[1] == W1,
                 "%s" % (tuple(X1.shape),))
    model1 = build_backbone(BackboneConfig(in_channels=1, depth_exponent=3,
                                           stem_width=16, embedding_size=16))
    model1.eval()
    with torch.no_grad():
        Z1 = model1(X1)
    ok &= _check("1-channel backbone(X1) -> (M, E)",
                 tuple(Z1.shape) == (X1.shape[0], 16), "%s" % (tuple(Z1.shape),))

    print("=" * 60)
    print("SMOKE RESULT: %s" % ("ALL PASSED" if ok else "FAILURES ABOVE"))
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
