"""CLI driver -- extract channel-subset IFR traces from a ptrain folder and save.

Runs the full extractor (load -> geometry/MFR -> partition -> IFR) on ONE folder
of ptrain_<idx>.mat files and writes a self-describing .npz plus (for the
partition modes) the electrode-map and per-subregion IFR PNGs.

Usage
-----
    python3 run_channel_subset_extraction.py FOLDER --out-dir OUT \
        [--mode multichannel|per_region_single|whole_culture] \
        [--n-subsets 9] [--electrodes-per-subset 9] [--mfr-threshold 0.1] \
        [--fs-raw 10110.09] [--base 0] [--no-plots]

Output (in OUT)
---------------
    traces.npz : arrays
        X            : (rows, K) float32 IFR traces
        row_meaning  : "channels" (multichannel: rows are channels of ONE sample)
                       or "samples" (per_region_single / whole_culture: rows are
                       independent single-channel samples)
        in_channels  : C for multichannel, else 1
        n_samples    : number of training samples this recording yields
        fs_ifr, mode, index_base, grid_width, T_rec, n_samples_raw
        centers, center_mfr, discarded  (empty for whole_culture)
    subregion_map.png, subregion_ifrs.png  (unless --no-plots / whole_culture)

Only numpy / scipy / matplotlib are required (no torch): this is data extraction,
not training.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import numpy as np

from channel_subset_extraction import DEFAULT_FS_RAW, extract_channel_subsets


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Extract channel-subset IFR traces from a ptrain folder.")
    p.add_argument("folder", help="directory of ptrain_<idx>.mat files (one recording)")
    p.add_argument("--out-dir", required=True, help="output directory")
    p.add_argument("--mode", default="multichannel",
                   choices=["multichannel", "per_region_single", "whole_culture"])
    p.add_argument("--n-subsets", type=int, default=9)
    p.add_argument("--electrodes-per-subset", type=int, default=9)
    p.add_argument("--mfr-threshold", type=float, default=0.1)
    p.add_argument("--fs-raw", type=float, default=DEFAULT_FS_RAW)
    p.add_argument("--base", type=int, default=0, choices=[0, 1])
    p.add_argument("--grid-width", type=int, default=48)
    p.add_argument("--w-size", type=float, default=0.02)
    p.add_argument("--gaussian-window", type=float, default=0.04)
    p.add_argument("--no-plots", action="store_true", help="skip PNG rendering")
    args = p.parse_args(argv)

    traces, fs_ifr, diag = extract_channel_subsets(
        args.folder, mode=args.mode, n_subsets=args.n_subsets,
        electrodes_per_subset=args.electrodes_per_subset,
        mfr_threshold=args.mfr_threshold, fs_raw=args.fs_raw, index_base=args.base,
        grid_width=args.grid_width, w_size=args.w_size,
        gaussian_window=args.gaussian_window, return_diagnostics=True)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "multichannel":
        X = np.asarray(traces[0], dtype=np.float32)          # (C, K), rows = channels
        row_meaning = "channels"
        in_channels = int(X.shape[0])
        n_samples = 1
    else:
        X = np.stack([np.asarray(t, dtype=np.float32).reshape(-1) for t in traces],
                     axis=0)                                  # (n_samples, K), rows = samples
        row_meaning = "samples"
        in_channels = 1
        n_samples = int(X.shape[0])

    centers = np.array([s.center for s in diag.subregions], dtype=np.int64)
    center_mfr = np.array([s.center_mfr for s in diag.subregions], dtype=np.float64)
    discarded = np.array(diag.discarded, dtype=np.int64)

    npz_path = os.path.join(args.out_dir, "traces.npz")
    np.savez_compressed(
        npz_path,
        X=X, row_meaning=row_meaning, in_channels=in_channels, n_samples=n_samples,
        fs_ifr=float(fs_ifr), mode=args.mode, index_base=int(diag.index_base),
        grid_width=int(diag.grid_width), T_rec=float(diag.T_rec),
        n_samples_raw=int(diag.n_samples), n_present=int(diag.n_present),
        centers=centers, center_mfr=center_mfr, discarded=discarded)

    if (not args.no_plots) and diag.subregions:
        from channel_subset_viz import plot_subregion_ifrs, plot_subregion_map
        plot_subregion_map(diag, os.path.join(args.out_dir, "subregion_map.png"))
        arr = traces[0] if args.mode == "multichannel" else X
        plot_subregion_ifrs(arr, fs_ifr,
                            os.path.join(args.out_dir, "subregion_ifrs.png"),
                            centers=[s.center for s in diag.subregions])

    print("mode=%s  X.shape=%s  row_meaning=%s  in_channels=%d  n_samples=%d  fs_ifr=%.3f"
          % (args.mode, X.shape, row_meaning, in_channels, n_samples, fs_ifr))
    print("present=%d  discarded(<theta)=%d  centres=%s"
          % (diag.n_present, discarded.size, centers.tolist()))
    print("wrote", npz_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
