"""
smoke_test_data_pipeline.py
===========================

Correctness + sanity test for data_pipeline.py (Dataset, condition-balanced
batch sampler, option-(b) collator) using synthetic data -- no .mat files needed.

Run
---
    pip install numpy scipy torch matplotlib
    python smoke_test_data_pipeline.py

Checks
------
    * windowing produces windows for every trace (overlap honored)
    * Dataset item: anchor is the first (clean, unshifted) positive
    * collator label scheme (OPTION b):
        - positives carry condition labels {CONTROL, PATHO}
        - destroyed negatives have UNIQUE labels, all >= base, all distinct,
          disjoint from {CONTROL, PATHO}
        - M = sum_b (1 + P_b + N_b)
    * every batch is condition-balanced (both conditions present)
    * determinism for fixed (seed, num_workers); changes with seed
    * num_workers > 0 path runs and is reproducible
    * both split methods ("warp_bands", "percentile_mse") work
    * a debug plot is written

Quick-embed snippet
-------------------
    from data_pipeline import MEAWindowDataset, TripletCollator, ConditionBalancedBatchSampler
    from augmentation import AugmentationConfig
    # ... build a dataset, then:
    # loader = DataLoader(ds, batch_sampler=bs, collate_fn=TripletCollator(), num_workers=0)
"""

from __future__ import annotations

import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

from augmentation import AugmentationConfig
from data_pipeline import (
    CONTROL, PATHO, closest_power_of_2,
    SyntheticTraceProvider, MEAWindowDataset,
    ConditionBalancedBatchSampler, TripletCollator, seed_worker,
)
from augmentation_viz import plot_triplet_instance

_RESULTS = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"   | {detail}" if detail else ""))
    _RESULTS.append(bool(cond))
    return bool(cond)


def build_dataset(split_method="warp_bands", seed=0, fs=50.0):
    provider = SyntheticTraceProvider(duration_s=120.0, fs=fs, seed=seed)
    traces, conditions = [], []
    for tid in range(2):
        tr, _ = provider(CONTROL, tid); traces.append(tr); conditions.append(CONTROL)
    for tid in range(1):
        tr, _ = provider(PATHO, tid); traces.append(tr); conditions.append(PATHO)
    window_length = closest_power_of_2(20.0 * fs)        # ~1024 samples
    stride = int(10.0 * fs)                               # overlap
    cfg = AugmentationConfig(fs=fs, split_method=split_method,
                             n_positives=15, n_negatives=15, percentile_q=0.30)
    ds = MEAWindowDataset(traces, conditions, window_length, stride, cfg, base_seed=seed)
    return ds, window_length


def build_loader(ds, num_workers=0, seed=0, mode="unique", n_batches=4, per_condition=2):
    bs = ConditionBalancedBatchSampler(ds.conditions_per_item, per_condition, n_batches, seed=seed)
    col = TripletCollator(destroyed_label_mode=mode)
    return DataLoader(ds, batch_sampler=bs, collate_fn=col, num_workers=num_workers,
                      worker_init_fn=seed_worker, persistent_workers=(num_workers > 0)), col


def main() -> int:
    fs = 50.0

    # ---- 1. windowing + dataset item structure ------------------------------
    ds, T = build_dataset("warp_bands", seed=0, fs=fs)
    check("dataset produced windows", len(ds) > 0, f"n_windows={len(ds)}")
    check("both conditions windowed",
          (ds.conditions_per_item == CONTROL).any() and (ds.conditions_per_item == PATHO).any())
    item = ds[0]
    check("anchor == first positive (clean, unshifted)",
          bool(torch.allclose(item["positives"][0], item["anchor"].reshape(-1), atol=1e-5)))
    check("item dtypes float32",
          item["positives"].dtype == torch.float32 and item["negatives"].dtype == torch.float32)

    # ---- 2. collator label scheme (OPTION b), num_workers=0 -----------------
    loader, col = build_loader(ds, num_workers=0, seed=0, mode="unique")
    X, y, metas = next(iter(loader))
    check("X is (M,T) float32", X.ndim == 2 and X.shape[1] == T and X.dtype == torch.float32,
          f"X={tuple(X.shape)}")
    check("y is (M,) long", y.ndim == 1 and y.shape[0] == X.shape[0] and y.dtype == torch.long)

    pos_mask = (y == CONTROL) | (y == PATHO)
    neg_mask = y >= col.unique_label_base
    check("positives carry condition labels {0,1}",
          bool(((y[pos_mask] == CONTROL) | (y[pos_mask] == PATHO)).all()) and int(pos_mask.sum()) > 0)
    check("both conditions present among positives",
          bool((y == CONTROL).any() and (y == PATHO).any()))
    neg_labels = y[neg_mask]
    check("destroyed negatives have UNIQUE labels",
          neg_labels.numel() == torch.unique(neg_labels).numel(),
          f"n_neg={neg_labels.numel()}, unique={torch.unique(neg_labels).numel()}")
    check("negative labels disjoint from {0,1} (>= base)",
          bool((neg_labels >= col.unique_label_base).all()))
    # M == sum_b (1 + n_pos + n_neg) for warp_bands (fixed counts)
    B = len(metas)
    expected_M = B * (1 + ds.aug_cfg.n_positives + ds.aug_cfg.n_negatives)
    check("M == sum_b (1 + P + N)  [warp_bands fixed counts]",
          X.shape[0] == expected_M, f"M={X.shape[0]} expected={expected_M}")

    # ---- 3. every batch condition-balanced ----------------------------------
    all_balanced = True
    for X_b, y_b, _ in loader:
        if not (bool((y_b == CONTROL).any()) and bool((y_b == PATHO).any())):
            all_balanced = False
            break
    check("every batch contains both conditions", all_balanced)

    # ---- 4. determinism (FRESH construction, fixed seed, same num_workers) ---
    # The dataset RNG is stateful (advances per item -> augmentation diversity
    # across epochs), so reproducibility is guaranteed for a FRESH pipeline with
    # a fixed (seed, num_workers) -- not for two loaders over one used dataset.
    dsA, _ = build_dataset("warp_bands", seed=0, fs=fs)
    dsB, _ = build_dataset("warp_bands", seed=0, fs=fs)
    lA, _ = build_loader(dsA, num_workers=0, seed=123)
    lB, _ = build_loader(dsB, num_workers=0, seed=123)
    X1, y1, _ = next(iter(lA)); X2, y2, _ = next(iter(lB))
    check("deterministic for fixed seed (fresh ds, nw=0)",
          bool(torch.equal(X1, X2) and torch.equal(y1, y2)))
    dsC, _ = build_dataset("warp_bands", seed=0, fs=fs)
    lC, _ = build_loader(dsC, num_workers=0, seed=999)   # different sampler order
    X3, _, _ = next(iter(lC))
    check("changes with sampler seed", not bool(torch.equal(X1, X3)))

    # ---- 5. num_workers > 0 path runs and is reproducible -------------------
    try:
        lw1, _ = build_loader(ds, num_workers=2, seed=7)
        lw2, _ = build_loader(ds, num_workers=2, seed=7)
        Xw1, yw1, _ = next(iter(lw1)); Xw2, yw2, _ = next(iter(lw2))
        check("num_workers=2 runs and is reproducible",
              bool(torch.equal(Xw1, Xw2) and torch.equal(yw1, yw2)))
    except Exception as exc:  # pragma: no cover
        check("num_workers=2 runs and is reproducible", False, f"raised {type(exc).__name__}: {exc}")

    # ---- 6. percentile_mse split also flows through -------------------------
    ds2, T2 = build_dataset("percentile_mse", seed=1, fs=fs)
    lp, colp = build_loader(ds2, num_workers=0, seed=1, mode="unique")
    Xp, yp, mp = next(iter(lp))
    neg_p = yp[yp >= colp.unique_label_base]
    check("percentile_mse: non-empty pos & uniquely-labelled neg",
          int(((yp == CONTROL) | (yp == PATHO)).sum()) > 0
          and neg_p.numel() == torch.unique(neg_p).numel() and neg_p.numel() > 0)

    # ---- 7. shared label mode ----------------------------------------------
    ls, cols = build_loader(ds, num_workers=0, seed=0, mode="shared")
    Xs, ys, _ = next(iter(ls))
    check("shared mode: destroyed share one label",
          int((ys == cols.shared_destroyed_label).sum()) > 0)

    # ---- 8. debug plot ------------------------------------------------------
    import os
    out_dir = "./pipeline_out/aug_debug_test"
    p = plot_triplet_instance(item["anchor"], item["positives"], item["negatives"],
                              fs=fs, out_dir=out_dir, instance_id=0, title="smoke-test item")
    check("debug plot written", os.path.exists(p), p)

    n_pass, n_tot = sum(_RESULTS), len(_RESULTS)
    print(f"\n================  {n_pass}/{n_tot} checks passed  ================")
    return 0 if n_pass == n_tot else 1


if __name__ == "__main__":
    sys.exit(main())
