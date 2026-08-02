#!/usr/bin/env python3
"""
Smoke test for the multichannel (C, T) shared-warp augmentation.

Verifies, against the edited augmentation.py in THIS folder:
  1. shared-field invariant : identical input channels -> identical output
     channels (one warp field is applied to every channel);
  2. ratio invariant        : channels that are scalar multiples stay scalar
     multiples after the surrogate (a consequence of a single shared linear
     warp; clamp_min commutes with positive scaling);
  3. shape contract         : (C, T) -> anchor (1,C,T), positives (1+P,C,T),
     negatives (N,C,T), and pre-shift variants;
  4. shift sharing          : the circular shift is shared across a surrogate's
     channels (identical channels remain identical AFTER the shift);
  5. percentile_mse path    : the alternative split also works for (C, T);
  6. non-negativity         : enforce_nonneg holds across channels;
  7. C=1 REGRESSION         : for a (T,) window, the edited module reproduces the
     ORIGINAL augmentation.py output bit-for-bit under the same seed.

Run:
    python3 smoke_test_augmentation_mc.py [--orig /path/to/original/augmentation.py]
Exit code 0 = all passed.
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# edited (multichannel) module from this folder
from augmentation import AugmentationConfig, build_triplet_instance  # noqa: E402


def load_module_from_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # register BEFORE exec (dataclass introspection)
    spec.loader.exec_module(mod)
    return mod


def _check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print("[%s] %s%s" % (tag, name, ("  (%s)" % detail) if detail else ""))
    return bool(cond)


def make_base(T, rng):
    """A simple non-negative bump trace (numpy float32, shape (T,))."""
    t = np.arange(T, dtype=np.float64)
    x = 0.1 * np.ones(T)
    for c in rng.uniform(0.15 * T, 0.85 * T, size=4):
        x += (0.5 + rng.random()) * np.exp(-0.5 * ((t - c) / (0.02 * T)) ** 2)
    return x.astype(np.float32)


def cfg_for(fs, C_note, **over):
    base = dict(fs=fs, n_positives=6, n_negatives=6, shift_magnitude_s=2.0,
                split_method="warp_bands", intra_knot_dist=0.2, k_min=4)
    base.update(over)
    return AugmentationConfig(**base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default=None,
                    help="path to the ORIGINAL (single-channel) augmentation.py")
    args = ap.parse_args()

    ok = True
    fs = 50.0
    T = 256
    C = 9
    rng_data = np.random.default_rng(7)
    base = make_base(T, rng_data)                         # (T,)

    # ---- 1. shared-field invariant: identical channels stay identical ----- #
    Xident = np.tile(base[None, :], (C, 1))              # (C, T) all rows equal
    cfg = cfg_for(fs, "ident")
    a, pos, neg, pos_pre, neg_pre = build_triplet_instance(
        Xident, cfg, np.random.default_rng(0), return_pre_shift=True)
    ok &= _check("shape anchor (1,C,T)", tuple(a.shape) == (1, C, T),
                 "%s" % (tuple(a.shape),))
    ok &= _check("shape positives (1+P,C,T)",
                 tuple(pos.shape) == (1 + cfg.n_positives, C, T),
                 "%s" % (tuple(pos.shape),))
    ok &= _check("shape negatives (N,C,T)",
                 tuple(neg.shape) == (cfg.n_negatives, C, T),
                 "%s" % (tuple(neg.shape),))

    def channels_identical(block):
        # block: (B, C, T) -> max abs deviation of each channel from channel 0
        return float((block - block[:, :1, :]).abs().max())

    ok &= _check("pre-shift positives: channels identical (shared field)",
                 channels_identical(pos_pre) < 1e-5,
                 "max dev %.2e" % channels_identical(pos_pre))
    ok &= _check("pre-shift negatives: channels identical (shared field)",
                 channels_identical(neg_pre) < 1e-5,
                 "max dev %.2e" % channels_identical(neg_pre))
    # 4. shift sharing: identical channels remain identical AFTER the shift
    ok &= _check("post-shift positives: channels identical (shift shared)",
                 channels_identical(pos) < 1e-5,
                 "max dev %.2e" % channels_identical(pos))
    ok &= _check("post-shift negatives: channels identical (shift shared)",
                 channels_identical(neg) < 1e-5,
                 "max dev %.2e" % channels_identical(neg))

    # ---- 2. ratio invariant ---------------------------------------------- #
    scales = np.arange(1, C + 1, dtype=np.float64)      # a_c = 1..9
    Xscaled = (scales[:, None] * base[None, :]).astype(np.float32)  # (C, T)
    _, _, _, pos_pre_s, _ = build_triplet_instance(
        Xscaled, cfg, np.random.default_rng(1), return_pre_shift=True)
    # channel c should equal a_c * channel 0 for every surrogate row
    ch0 = pos_pre_s[:, :1, :]                            # (B,1,T)
    expected = ch0 * torch.as_tensor(scales, dtype=ch0.dtype).view(1, C, 1)
    ratio_dev = float((pos_pre_s - expected).abs().max())
    ok &= _check("ratio invariant: channel c == a_c * channel 0",
                 ratio_dev < 1e-3, "max dev %.2e" % ratio_dev)

    # ---- 5. percentile_mse path for (C, T) ------------------------------- #
    cfg_pct = cfg_for(fs, "pct", split_method="percentile_mse", percentile_q=0.30)
    ap_, posp, negp, _, _ = build_triplet_instance(
        Xscaled, cfg_pct, np.random.default_rng(2), return_pre_shift=True)
    ok &= _check("percentile_mse: shapes (C,T) and non-empty classes",
                 posp.shape[0] >= 2 and negp.shape[0] >= 1
                 and posp.shape[1] == C and negp.shape[1] == C,
                 "pos %s neg %s" % (tuple(posp.shape), tuple(negp.shape)))

    # ---- 6. non-negativity ----------------------------------------------- #
    rng_r = np.random.default_rng(3)
    Xreal = np.stack([make_base(T, rng_r) for _ in range(C)], axis=0)  # distinct channels
    _, posr, negr, _, _ = build_triplet_instance(
        Xreal, cfg, np.random.default_rng(4), return_pre_shift=True)
    mn = float(min(posr.min(), negr.min()))
    ok &= _check("enforce_nonneg: min surrogate value >= -1e-4", mn >= -1e-4,
                 "min %.4g" % mn)
    ok &= _check("realistic C=9 channels are NOT identical (true multichannel)",
                 channels_identical(negr) > 1e-3,
                 "cross-channel spread %.3g" % channels_identical(negr))
    ok &= _check("all surrogates finite",
                 bool(torch.isfinite(posr).all() and torch.isfinite(negr).all()))

    # ---- 7. C=1 REGRESSION vs original ----------------------------------- #
    orig_path = args.orig
    if orig_path is None:
        # default guess: sibling pipeline_unzip
        guess = os.path.normpath(os.path.join(
            HERE, "..", "hpc", "pipeline_unzip", "augmentation.py"))
        orig_path = guess if os.path.exists(guess) else None
    if orig_path and os.path.exists(orig_path):
        aug_orig = load_module_from_path(orig_path, "aug_orig")
        cfg1 = cfg_for(fs, "c1")
        seed = 123
        out_new = build_triplet_instance(base, cfg1, np.random.default_rng(seed),
                                         return_pre_shift=True)
        cfg1o = aug_orig.AugmentationConfig(
            fs=fs, n_positives=6, n_negatives=6, shift_magnitude_s=2.0,
            split_method="warp_bands", intra_knot_dist=0.2, k_min=4)
        out_old = aug_orig.build_triplet_instance(
            base, cfg1o, np.random.default_rng(seed), return_pre_shift=True)
        names = ["anchor", "positives", "negatives", "pos_pre", "neg_pre"]
        all_eq = True
        for nm, a_new, a_old in zip(names, out_new, out_old):
            eq = a_new.shape == a_old.shape and torch.allclose(a_new, a_old, atol=0.0)
            all_eq &= eq
            if not eq:
                print("    regression mismatch in %s: shapes %s vs %s, maxdiff %s"
                      % (nm, tuple(a_new.shape), tuple(a_old.shape),
                         (a_new - a_old).abs().max() if a_new.shape == a_old.shape
                         else "n/a"))
        ok &= _check("C=1 reproduces ORIGINAL augmentation bit-for-bit", all_eq)
    else:
        print("[SKIP] C=1 regression (original augmentation.py not found; "
              "pass --orig PATH)")

    print("=" * 60)
    print("SMOKE RESULT: %s" % ("ALL PASSED" if ok else "FAILURES ABOVE"))
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
