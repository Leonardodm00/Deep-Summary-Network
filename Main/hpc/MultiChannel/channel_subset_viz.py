"""Stage 6 -- headless visualization for the channel-subset extractor.

Two PNG renderers, both driven by the Stage-5 ExtractionDiagnostics:

    plot_subregion_map(diag, out_path)   -- the 48x48 electrode map: centres,
        per-channel members, valid-but-unassigned, and discarded (< theta)
        electrodes. THIS is the plot to eyeball to confirm the index base and
        the row/col orientation (both currently unconfirmed).

    plot_subregion_ifrs(traces, fs_ifr, out_path)  -- the per-subregion IFR
        traces stacked vertically (one panel per channel).

Design notes
------------
* Separation of concerns (directive 2): this module ONLY draws. It imports no
  extraction logic; it consumes an ExtractionDiagnostics and/or trace arrays that
  the caller already produced via extract_channel_subsets. Nothing here recomputes
  IFRs, MFRs or partitions.
* Matplotlib runs on the Agg backend (headless / HPC nodes, no display).
* ORIENTATION FLAG: the map is drawn with col on x and row on y, and the y-axis
  is inverted so row 0 is at the TOP (numpy/imshow convention). Whether this
  matches the physical chip is exactly what you confirm by eye; flip base or
  orientation upstream if the hot region lands in the wrong place.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")                     # headless: set BEFORE importing pyplot
import matplotlib.pyplot as plt
import numpy as np


def _channel_colormap(n_channels: int):
    name = "tab10" if n_channels <= 10 else "tab20"
    return plt.get_cmap(name, max(int(n_channels), 1))


def plot_subregion_map(diag, out_path: str, title: Optional[str] = None,
                       dpi: int = 130) -> str:
    """Render the electrode map from an ExtractionDiagnostics.

    Parameters
    ----------
    diag : ExtractionDiagnostics
        Must carry a partition (subregions non-empty); whole_culture diagnostics
        have no map and raise ValueError.
    out_path : str
        Destination PNG path.
    title : str, optional
        Plot title; a default with base and grid size is used if omitted.
    dpi : int
        Figure resolution.

    Returns
    -------
    out_path : str
    """
    if not diag.subregions:
        raise ValueError(
            "plot_subregion_map needs partition diagnostics; the given diag has "
            "no subregions (whole_culture mode has nothing to map)")

    width = int(diag.grid_width)
    coords: Dict[int, Tuple[int, int]] = diag.coords
    discarded = set(diag.discarded)

    member_channel: Dict[int, int] = {}
    center_channel: Dict[int, int] = {}
    for ch, s in enumerate(diag.subregions):
        for m in s.members:
            member_channel[m] = ch
        center_channel[s.center] = ch

    assigned = set(member_channel)
    present = list(coords.keys())
    n_ch = len(diag.subregions)
    cmap = _channel_colormap(n_ch)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.set_xlim(-1, width)
    ax.set_ylim(-1, width)
    ax.set_aspect("equal")

    # discarded (< theta): grey squares
    if discarded:
        xs = [coords[e][1] for e in discarded]
        ys = [coords[e][0] for e in discarded]
        ax.scatter(xs, ys, c="lightgrey", s=16, marker="s", label="discarded (< theta)")

    # valid but unassigned: open circles
    unassigned = [e for e in present if e not in assigned and e not in discarded]
    if unassigned:
        xs = [coords[e][1] for e in unassigned]
        ys = [coords[e][0] for e in unassigned]
        ax.scatter(xs, ys, facecolors="none", edgecolors="steelblue", s=24,
                   linewidths=0.8, label="valid, unassigned")

    # members (non-centre) coloured by channel
    mem = [m for m in member_channel if m not in center_channel]
    if mem:
        xs = [coords[m][1] for m in mem]
        ys = [coords[m][0] for m in mem]
        cc = [member_channel[m] for m in mem]
        ax.scatter(xs, ys, c=cc, cmap=cmap, vmin=0, vmax=max(n_ch - 1, 1),
                   s=42, label="subregion members")

    # centres: stars coloured by channel, annotated with channel index
    cxs = [coords[c][1] for c in center_channel]
    cys = [coords[c][0] for c in center_channel]
    ccc = [center_channel[c] for c in center_channel]
    ax.scatter(cxs, cys, c=ccc, cmap=cmap, vmin=0, vmax=max(n_ch - 1, 1),
               s=200, marker="*", edgecolors="black", linewidths=1.0,
               zorder=5, label="centre (channel #)")
    for c, ch in center_channel.items():
        ax.annotate(str(ch), (coords[c][1], coords[c][0]),
                    textcoords="offset points", xytext=(5, 4),
                    fontsize=8, weight="bold")

    ax.set_xlabel("col (x index)   [x_um = col * pitch]")
    ax.set_ylabel("row (y index)   [y_um = row * pitch]")
    ax.invert_yaxis()  # row 0 at TOP (numpy convention) -- CONFIRM against chip
    ax.set_title(title or ("Subregion electrode map  (base=%d, grid %dx%d, C=%d)"
                           % (diag.index_base, width, width, n_ch)))
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_subregion_ifrs(traces, fs_ifr: float, out_path: str,
                        centers: Optional[Sequence[int]] = None,
                        title: Optional[str] = None, dpi: int = 130) -> str:
    """Stack per-subregion IFR traces, one panel per channel.

    Parameters
    ----------
    traces : (C, K) array OR list of (K,) arrays OR a single (K,) / (1, K) array.
    fs_ifr : float
        IFR sampling rate [Hz] (sets the time axis).
    out_path : str
        Destination PNG path.
    centers : sequence of int, optional
        Centre electrode index per channel (annotated in the panel labels).
    title : str, optional
    dpi : int

    Returns
    -------
    out_path : str
    """
    if isinstance(traces, np.ndarray):
        arr = traces if traces.ndim == 2 else traces[None, :]
    else:
        arr = np.stack([np.asarray(t, dtype=np.float32).reshape(-1) for t in traces], axis=0)
    n_ch, K = arr.shape
    t = np.arange(K, dtype=np.float64) / float(fs_ifr)

    fig, axes = plt.subplots(n_ch, 1, figsize=(9.0, 1.35 * n_ch + 0.6),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ch in range(n_ch):
        axes[ch].plot(t, arr[ch], lw=0.7, color="C%d" % (ch % 10))
        lbl = "ch %d" % ch
        if centers is not None and ch < len(centers):
            lbl += "\n(c=%d)" % int(centers[ch])
        axes[ch].set_ylabel(lbl, fontsize=8)
        axes[ch].margins(x=0)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(title or ("Per-subregion IFR  (C=%d, fs_ifr=%.1f Hz)" % (n_ch, fs_ifr)))
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _cli(argv: Optional[List[str]] = None) -> int:
    """Render both plots for a real ptrain folder so the geometry can be eyeballed."""
    from channel_subset_extraction import extract_channel_subsets, DEFAULT_FS_RAW

    p = argparse.ArgumentParser(description="Render subregion map + IFR PNGs from a ptrain folder.")
    p.add_argument("folder", help="directory of ptrain_<idx>.mat files")
    p.add_argument("--mode", default="multichannel",
                   choices=["multichannel", "per_region_single", "whole_culture"])
    p.add_argument("--n-subsets", type=int, default=9)
    p.add_argument("--electrodes-per-subset", type=int, default=9)
    p.add_argument("--mfr-threshold", type=float, default=0.1)
    p.add_argument("--fs-raw", type=float, default=DEFAULT_FS_RAW)
    p.add_argument("--base", type=int, default=0, choices=[0, 1])
    p.add_argument("--out-dir", default=".")
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    traces, fs_ifr, diag = extract_channel_subsets(
        args.folder, mode=args.mode, n_subsets=args.n_subsets,
        electrodes_per_subset=args.electrodes_per_subset,
        mfr_threshold=args.mfr_threshold, fs_raw=args.fs_raw, index_base=args.base,
        return_diagnostics=True)

    ifr_png = os.path.join(args.out_dir, "subregion_ifrs.png")
    plot_subregion_ifrs(traces if args.mode != "multichannel" else traces[0],
                        fs_ifr, ifr_png,
                        centers=[s.center for s in diag.subregions] or None)
    print("wrote", ifr_png)
    if diag.subregions:
        map_png = os.path.join(args.out_dir, "subregion_map.png")
        plot_subregion_map(diag, map_png)
        print("wrote", map_png)
    else:
        print("(whole_culture mode: no electrode map)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
