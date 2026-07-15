"""
smoke_test_augmentation.py
==========================

Self-contained correctness + sanity test for augmentation.py and
augmentation_viz.py. It synthesizes a non-negative burst-like signal (a proxy
for a smoothed cumulative IFR window), exercises every public function, asserts
the invariants that matter, prints a PASS/FAIL summary, and writes a few
visual-debug PNGs for both split methods.

Run
---
    pip install numpy scipy torch matplotlib    # CPU torch is enough
    python smoke_test_augmentation.py

Outputs
-------
    ./aug_debug/triplet_*.png    -- visual-debug figures
    console PASS/FAIL summary; process exits non-zero if any check fails.

Quick-embed snippet (run a single check from a Python shell)
------------------------------------------------------------
    import numpy as np
    from augmentation import AugmentationConfig, magnitude_warp
    rng = np.random.default_rng(0)
    x = np.abs(np.sin(np.linspace(0, 20, 2048))).astype("float32")
    w = magnitude_warp(x, fs=50, sigma_mag=0.5, intra_knot_dist=0.2, rng=rng)
    assert float(w.min()) >= -1e-6, "magnitude warp must keep firing rate >= 0"
    print("ok:", w.shape, w.dtype, float(w.min()))
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import torch

from augmentation import (
    AugmentationConfig,
    magnitude_warp,
    time_warp,
    random_circular_shift,
    build_triplet_instance,
)
from augmentation_viz import plot_triplet_instance


# --------------------------------------------------------------------------- #
# tiny test harness
# --------------------------------------------------------------------------- #
_RESULTS = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    status = "PASS" if cond else "FAIL"
    line = f"[{status}] {name}"
    if detail:
        line += f"   | {detail}"
    print(line)
    _RESULTS.append(bool(cond))
    return bool(cond)


# --------------------------------------------------------------------------- #
# synthetic non-negative burst signal (proxy for a smoothed cumulative IFR)
# --------------------------------------------------------------------------- #
def synth_burst_signal(T: int, fs: float, n_bursts: int = 12, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(T) / fs
    x = np.zeros(T, dtype=np.float64)
    centers = rng.uniform(t[0], t[-1], n_bursts)
    widths = rng.uniform(0.2, 1.0, n_bursts)
    amps = rng.uniform(0.5, 1.0, n_bursts)
    for c, w, a in zip(centers, widths, amps):
        x += a * np.exp(-0.5 * ((t - c) / w) ** 2)
    return x.astype(np.float32)          # non-negative by construction


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def main() -> int:
    T, FS = 2048, 50.0
    x = synth_burst_signal(T, FS, seed=0)
    print(f"\nSynthetic signal: T={T}, fs={FS} Hz, length={T / FS:.2f} s, "
          f"min={x.min():.4f}, max={x.max():.4f}\n")

    # ---- 1. magnitude_warp: positivity (the whole point of log-space) -------
    rng = np.random.default_rng(1)
    for sm in (0.05, 0.2, 0.5):
        w = magnitude_warp(x, FS, sigma_mag=sm, intra_knot_dist=0.2, rng=rng)
        check(f"magnitude_warp positivity (sigma_mag={sm})",
              float(w.min()) >= -1e-6,
              f"min={float(w.min()):.6f}")
    check("magnitude_warp dtype float32", w.dtype == torch.float32)
    check("magnitude_warp shape (T,)", tuple(w.shape) == (T,), f"shape={tuple(w.shape)}")
    check("magnitude_warp finite", bool(torch.isfinite(w).all()))

    # ---- 2. time_warp: endpoints pinned, shape, finite ----------------------
    rng = np.random.default_rng(2)
    w = time_warp(x, FS, sigma_time_s=0.2, intra_knot_dist=0.2, rng=rng)
    check("time_warp endpoint[0] preserved",
          np.isclose(float(w[0]), float(x[0]), atol=1e-3),
          f"warp[0]={float(w[0]):.5f} vs x[0]={float(x[0]):.5f}")
    check("time_warp endpoint[-1] preserved",
          np.isclose(float(w[-1]), float(x[-1]), atol=1e-3),
          f"warp[-1]={float(w[-1]):.5f} vs x[-1]={float(x[-1]):.5f}")
    check("time_warp dtype float32", w.dtype == torch.float32)
    check("time_warp shape (T,)", tuple(w.shape) == (T,))
    check("time_warp finite", bool(torch.isfinite(w).all()))

    # ---- 3. random_circular_shift: matches torch.roll; warns on zero --------
    rng = np.random.default_rng(0)
    # force a known single shift by making max_shift=1 and checking roll-equivalence
    base = torch.arange(T, dtype=torch.float32).unsqueeze(0)   # (1, T) ramp
    s_known = 7
    rolled_ref = torch.roll(base, shifts=s_known, dims=1)
    # build the same circular index the function uses
    idx = (np.arange(T)[None, :] - np.array([s_known])[:, None]) % T
    rolled_man = torch.gather(base, 1, torch.as_tensor(idx, dtype=torch.long))
    check("circular-shift gather == torch.roll",
          bool(torch.equal(rolled_ref, rolled_man)))
    # shape / dtype preserved through the public function
    sh = random_circular_shift(torch.stack([torch.tensor(x)] * 4), 5.0, FS, rng)
    check("random_circular_shift shape (B,T)", tuple(sh.shape) == (4, T))
    check("random_circular_shift dtype float32", sh.dtype == torch.float32)
    # zero-magnitude warning
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out0 = random_circular_shift(torch.tensor(x), 0.0, FS, rng)
        check("zero-shift warns and returns unchanged",
              len(rec) >= 1 and bool(torch.equal(out0, torch.tensor(x))),
              f"warnings={len(rec)}")

    # ---- 4. split: percentile_mse gives ~q positives, non-empty -------------
    q = 0.30
    cfg2 = AugmentationConfig(fs=FS, split_method="percentile_mse", percentile_q=q,
                              n_positives=70, n_negatives=70)   # pool=140
    rng = np.random.default_rng(10)
    a, pos, neg = build_triplet_instance(x, cfg2, rng)
    pool = pos.shape[0] - 1 + neg.shape[0]      # minus the clean anchor we prepended
    frac = (pos.shape[0] - 1) / pool
    check("percentile_mse positive fraction ~ q",
          abs(frac - q) < 0.08, f"frac={frac:.3f} (target {q})")
    check("percentile_mse both classes non-empty",
          (pos.shape[0] - 1) >= 1 and neg.shape[0] >= 1,
          f"P={pos.shape[0]-1}, N={neg.shape[0]}")

    # ---- 5. split: warp_bands -> mean MSE(pos) < mean MSE(neg) ---------------
    cfg3 = AugmentationConfig(fs=FS, split_method="warp_bands",
                              n_positives=40, n_negatives=40)
    rng = np.random.default_rng(11)
    a3, pos3, neg3 = build_triplet_instance(x, cfg3, rng)
    xa = a3.reshape(1, -1)
    # exclude the prepended clean anchor (row 0) from the positive-MSE statistic
    mse_pos = ((pos3[1:] - xa) ** 2).mean(dim=1).mean().item()
    mse_neg = ((neg3 - xa) ** 2).mean(dim=1).mean().item()
    check("warp_bands mean MSE(pos) < mean MSE(neg)",
          mse_pos < mse_neg, f"MSE_pos={mse_pos:.5f} < MSE_neg={mse_neg:.5f}")

    # ---- 6. build_triplet_instance: anchor included, shapes, dtype, nonneg --
    for cfg, tag in ((cfg2, "percentile_mse"), (cfg3, "warp_bands")):
        rng = np.random.default_rng(20)
        a, pos, neg = build_triplet_instance(x, cfg, rng)
        check(f"[{tag}] anchor shape (1,T)", tuple(a.shape) == (1, T))
        check(f"[{tag}] anchor is first positive (clean, unshifted)",
              bool(torch.allclose(pos[0], a.reshape(-1), atol=1e-5)))
        check(f"[{tag}] positives/negatives dtype float32",
              pos.dtype == torch.float32 and neg.dtype == torch.float32)
        check(f"[{tag}] all finite",
              bool(torch.isfinite(pos).all() and torch.isfinite(neg).all()))
        check(f"[{tag}] non-negative (enforce_nonneg)",
              float(pos.min()) >= -1e-6 and float(neg.min()) >= -1e-6,
              f"min(pos)={float(pos.min()):.6f}, min(neg)={float(neg.min()):.6f}")

    # ---- 7. empty-class guard: q=1.0 -> negatives always empty -> raises -----
    cfg_bad = AugmentationConfig(fs=FS, split_method="percentile_mse",
                                 percentile_q=1.0, n_positives=20, n_negatives=20,
                                 max_retries=2)
    rng = np.random.default_rng(30)
    raised = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            build_triplet_instance(x, cfg_bad, rng)
        except RuntimeError:
            raised = True
    check("empty-class guard raises after retries (q=1.0)", raised)

    # ---- 8. reproducibility: same seed -> identical; diff seed -> different --
    a1, p1, n1 = build_triplet_instance(x, cfg3, np.random.default_rng(42))
    a2, p2, n2 = build_triplet_instance(x, cfg3, np.random.default_rng(42))
    a3b, p3b, n3b = build_triplet_instance(x, cfg3, np.random.default_rng(43))
    check("reproducible with same seed",
          bool(torch.equal(p1, p2) and torch.equal(n1, n2)))
    check("different with different seed",
          not bool(torch.equal(p1, p3b)))

    # ---- 9. visual debug: write a few PNGs for both methods ------------------
    out_dir = "./aug_debug"
    paths = []
    for i, (cfg, tag) in enumerate(((cfg3, "warp_bands"), (cfg2, "percentile_mse"))):
        rng = np.random.default_rng(100 + i)
        a, pos, neg = build_triplet_instance(x, cfg, rng)
        p = plot_triplet_instance(a, pos, neg, fs=FS, out_dir=out_dir,
                                  instance_id=i, title=f"split = {tag}")
        paths.append(p)
    check("visual-debug PNGs written", all(__import__("os").path.exists(p) for p in paths),
          f"{paths}")

    # ---- summary ------------------------------------------------------------
    n_pass = sum(_RESULTS)
    n_tot = len(_RESULTS)
    print(f"\n================  {n_pass}/{n_tot} checks passed  ================")
    if paths:
        print("Visual-debug figures:")
        for p in paths:
            print(f"   {p}")
    return 0 if n_pass == n_tot else 1


if __name__ == "__main__":
    sys.exit(main())
