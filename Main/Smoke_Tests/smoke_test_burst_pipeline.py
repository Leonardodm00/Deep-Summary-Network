"""
smoke_test_burst_pipeline.py
============================

End-to-end smoke test that:
    1. generates synthetic burst data (100 neurons, 200 s) via
       generate_burst_data.generate_all,
    2. loads the resulting .npz files through the full Topic-1 pipeline
       (NumpyTraceProvider -> MEAWindowDataset -> ConditionBalancedBatchSampler
       -> TripletCollator),
    3. validates the batch structure (shapes, dtypes, option-b label scheme,
       condition balance, non-negativity) and the IFR -> window path, and
    4. saves debug raster plots.

Run
---
    python smoke_test_burst_pipeline.py

All checks are labelled [PASS] / [FAIL]; the process exits 0 only if every
check passes.  Controlled twice (second run re-uses the cached .npz files).
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

# Topic-1 modules
from generate_burst_data import (
    generate_all,
    load_burst_npz,
    CONTROL_PARAMS,
    PATHO_PARAMS,
)
from augmentation import AugmentationConfig
from augmentation_viz import plot_triplet_instance
from data_pipeline import (
    CONTROL, PATHO,
    closest_power_of_2,
    NumpyTraceProvider,
    MEAWindowDataset,
    ConditionBalancedBatchSampler,
    TripletCollator,
    seed_worker,
)

# --------------------------------------------------------------------------- #
# tiny harness
# --------------------------------------------------------------------------- #
_RESULTS: List[bool] = []


def _check(name: str, cond: bool, detail: str = "") -> bool:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"   | {detail}" if detail else ""))
    _RESULTS.append(bool(cond))
    return bool(cond)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _build_loader(traces, conditions, fs, window_s, stride_s,
                  split_method, seed, n_batches, per_condition, num_workers=0):
    """Build a DataLoader from a list of IFR traces."""
    window_length = closest_power_of_2(window_s * fs)
    stride        = max(1, int(stride_s * fs))
    aug_cfg       = AugmentationConfig(
        fs=fs, split_method=split_method,
        n_positives=12, n_negatives=12,
        shift_magnitude_s=min(15.0, window_s * 0.3),
    )
    ds      = MEAWindowDataset(traces, conditions, window_length, stride,
                               aug_cfg, base_seed=seed)
    sampler = ConditionBalancedBatchSampler(
        ds.conditions_per_item, per_condition, n_batches, seed=seed)
    col     = TripletCollator(destroyed_label_mode="unique")
    loader  = DataLoader(
        ds, batch_sampler=sampler, collate_fn=col,
        num_workers=num_workers, worker_init_fn=seed_worker,
        persistent_workers=(num_workers > 0),
    )
    return loader, ds, col, window_length


# --------------------------------------------------------------------------- #
# main test body
# --------------------------------------------------------------------------- #
def run_smoke_test(out_dir: str) -> int:
    _RESULTS.clear()
    print(f"\n{'='*60}")
    print(f"  smoke_test_burst_pipeline   (out_dir={out_dir!r})")
    print(f"{'='*60}")

    # ------------------------------------------------------------------ #
    # STAGE 1 -- data generation
    # ------------------------------------------------------------------ #
    print("\n-- STAGE 1: generate_burst_data --")
    specs_path = generate_all(out_dir=out_dir, seed=42)

    _check("burst_specs.json written", os.path.exists(specs_path))

    import json
    with open(specs_path) as fh:
        specs = json.load(fh)

    _check("specs contains 3 traces", len(specs) == 3)
    n_ctrl = sum(1 for s in specs if s["condition"] == CONTROL)
    n_path = sum(1 for s in specs if s["condition"] == PATHO)
    _check("2 control + 1 patho traces", n_ctrl == 2 and n_path == 1,
           f"control={n_ctrl}, patho={n_path}")
    _check("all .npz files exist",
           all(os.path.exists(s["npz_path"]) for s in specs))
    _check("all raster PNGs exist",
           all(os.path.exists(s["npz_path"].replace(".npz", "_raster.png"))
               for s in specs))

    # ------------------------------------------------------------------ #
    # STAGE 2 -- NumpyTraceProvider load + IFR checks
    # ------------------------------------------------------------------ #
    print("\n-- STAGE 2: NumpyTraceProvider + IFR properties --")
    provider = NumpyTraceProvider()
    traces, conditions = [], []
    fs_common = None
    for rec in specs:
        tr, fs = provider(rec["npz_path"])
        if fs_common is None:
            fs_common = fs
        traces.append(tr); conditions.append(int(rec["condition"]))

    _check("all traces loaded", len(traces) == 3)
    _check("fs consistent across traces",
           all(np.isclose(provider(s["npz_path"])[1], fs_common) for s in specs),
           f"fs={fs_common} Hz")

    # cross-check against load_burst_npz
    ref = load_burst_npz(specs[0]["npz_path"])
    _check("NumpyTraceProvider == load_burst_npz IFR",
           np.array_equal(traces[0], ref["ifr_trace"]))

    for i, tr in enumerate(traces):
        _check(f"trace[{i}] float32",      tr.dtype == np.float32)
        _check(f"trace[{i}] non-negative", bool(np.all(tr >= 0.0)),
               f"min={tr.min():.6f}")
        K_expected = int(CONTROL_PARAMS.duration_s / CONTROL_PARAMS.w_size)
        _check(f"trace[{i}] length K={K_expected}",
               len(tr) == K_expected, f"got {len(tr)}")

    # patho IFR statistics should differ from control
    ifr_c = traces[0]          # control_0
    ifr_p = traces[2]          # patho_0
    _check("patho mean IFR > control mean IFR",
           float(ifr_p.mean()) > float(ifr_c.mean()),
           f"patho={ifr_p.mean():.4f}, ctrl={ifr_c.mean():.4f}")

    # ------------------------------------------------------------------ #
    # STAGE 3 -- Dataset windowing
    # ------------------------------------------------------------------ #
    print("\n-- STAGE 3: MEAWindowDataset --")
    window_s = 30.0
    stride_s = 15.0
    _, ds, col, T = _build_loader(
        traces, conditions, fs_common,
        window_s=window_s, stride_s=stride_s,
        split_method="warp_bands", seed=0,
        n_batches=4, per_condition=2,
    )

    _check("dataset has windows", len(ds) > 0, f"n_windows={len(ds)}")
    _check("both conditions windowed",
           bool((ds.conditions_per_item == CONTROL).any()) and
           bool((ds.conditions_per_item == PATHO).any()))

    # single item check
    item = ds[0]
    _check("anchor == first positive (clean, unshifted)",
           bool(torch.allclose(item["anchor"].reshape(-1),
                               item["positives"][0], atol=1e-5)))
    _check("item float32",
           item["positives"].dtype == torch.float32 and
           item["negatives"].dtype == torch.float32)
    _check("item non-negative",
           float(item["positives"].min()) >= -1e-6 and
           float(item["negatives"].min()) >= -1e-6,
           f"min(pos)={item['positives'].min():.2e}, "
           f"min(neg)={item['negatives'].min():.2e}")

    # ------------------------------------------------------------------ #
    # STAGE 4 -- Collator / option-b label scheme
    # ------------------------------------------------------------------ #
    print("\n-- STAGE 4: TripletCollator label scheme (option b) --")
    loader, _, col, T = _build_loader(
        traces, conditions, fs_common,
        window_s=window_s, stride_s=stride_s,
        split_method="warp_bands", seed=1,
        n_batches=4, per_condition=2,
    )

    X, y, metas = next(iter(loader))

    _check("X is (M, T) float32",
           X.ndim == 2 and X.shape[1] == T and X.dtype == torch.float32,
           f"X={tuple(X.shape)}")
    _check("y is (M,) long",
           y.ndim == 1 and y.shape[0] == X.shape[0] and
           y.dtype == torch.long)

    pos_mask = (y == CONTROL) | (y == PATHO)
    neg_mask = y >= col.unique_label_base
    _check("positives carry condition labels {0,1}",
           int(pos_mask.sum()) > 0 and
           bool(((y[pos_mask] == CONTROL) | (y[pos_mask] == PATHO)).all()))
    _check("both conditions present among positives",
           bool((y == CONTROL).any() and (y == PATHO).any()))
    neg_labels = y[neg_mask]
    _check("negatives have unique labels",
           neg_labels.numel() == torch.unique(neg_labels).numel(),
           f"n_neg={neg_labels.numel()}")
    _check("negative labels >= base (disjoint from {0,1})",
           bool((neg_labels >= col.unique_label_base).all()))
    _check("all embeddings finite",
           bool(torch.isfinite(X).all()))

    # ------------------------------------------------------------------ #
    # STAGE 5 -- Condition balance across all batches
    # ------------------------------------------------------------------ #
    print("\n-- STAGE 5: condition balance --")
    all_balanced = True
    for X_b, y_b, _ in loader:
        if not (bool((y_b == CONTROL).any()) and bool((y_b == PATHO).any())):
            all_balanced = False
            break
    _check("every batch has both conditions", all_balanced)

    # ------------------------------------------------------------------ #
    # STAGE 6 -- percentile_mse split path
    # ------------------------------------------------------------------ #
    print("\n-- STAGE 6: percentile_mse split --")
    loader_p, _, col_p, _ = _build_loader(
        traces, conditions, fs_common,
        window_s=window_s, stride_s=stride_s,
        split_method="percentile_mse", seed=2,
        n_batches=2, per_condition=2,
    )
    Xp, yp, _ = next(iter(loader_p))
    neg_p = yp[yp >= col_p.unique_label_base]
    _check("percentile_mse: non-empty positives + unique negatives",
           int(((yp == CONTROL) | (yp == PATHO)).sum()) > 0 and
           neg_p.numel() == torch.unique(neg_p).numel() and neg_p.numel() > 0)

    # ------------------------------------------------------------------ #
    # STAGE 7 -- debug plots
    # ------------------------------------------------------------------ #
    print("\n-- STAGE 7: debug plots --")
    dbg_dir = os.path.join(out_dir, "aug_debug_burst")
    for j in range(min(3, len(ds))):
        item = ds[j]
        cond_name = "control" if item["condition"] == CONTROL else "patho"
        plot_triplet_instance(
            item["anchor"], item["positives"], item["negatives"],
            fs=fs_common, out_dir=dbg_dir, instance_id=j,
            title=f"{cond_name} | meta={item['meta']} | warp_bands",
        )
    n_plots = len([f for f in os.listdir(dbg_dir) if f.endswith(".png")])
    _check("debug PNGs written", n_plots >= 3, f"found {n_plots}")

    # ------------------------------------------------------------------ #
    # summary
    # ------------------------------------------------------------------ #
    n_pass = sum(_RESULTS)
    n_tot  = len(_RESULTS)
    print(f"\n{'='*60}")
    print(f"  {n_pass}/{n_tot} checks passed")
    return 0 if n_pass == n_tot else 1


def main() -> int:
    out_dir = "./burst_data"
    print("=== CONTROL RUN 1 ===")
    rc1 = run_smoke_test(out_dir)
    print("\n=== CONTROL RUN 2 ===")
    rc2 = run_smoke_test(out_dir)          # re-uses the cached .npz files
    return max(rc1, rc2)


if __name__ == "__main__":
    sys.exit(main())
