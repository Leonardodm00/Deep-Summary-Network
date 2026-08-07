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
                        title: Optional[str] = None, dpi: int = 130,
                        zoom_s: Optional[float] = 2.0,
                        zoom_ref_channel: Optional[int] = None,
                        zoom_center_s: Optional[float] = None) -> str:
    """Stack per-subregion IFR traces, one panel per channel.

    With `zoom_s` set (the default), each row gains a second panel: a close-up
    of one network burst, so the shape of the event and the participation of
    each subregion in it are visible at all. At 1200 s per recording the
    full-trace panel compresses a ~0.5 s burst into well under a pixel.

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
    zoom_s : float or None
        TOTAL width [s] of the close-up window, centred on the burst. None
        disables the close-up and restores the single-column figure exactly.
        Ignored (with the full trace still drawn) if the window would cover the
        whole recording.
    zoom_ref_channel : int or None
        Which channel's IFR locates the burst. None (default) uses the MEAN
        across channels, which is the right reference for a NETWORK burst: the
        event is defined by subregions rising together, and a single channel's
        argmax may sit on a local, non-network transient. Pass an index to
        centre on one subregion's own maximum instead.
    zoom_center_s : float or None
        Centre the window at this time [s] instead of at the detected peak.
        Use it to inspect a specific burst rather than the largest one.

    Returns
    -------
    out_path : str

    Notes
    -----
    The same window is used for EVERY channel: a network burst is one event, so
    a shared window is what makes the panels comparable. Per-channel windows
    would put nine different moments side by side.
    """
    if isinstance(traces, np.ndarray):
        arr = traces if traces.ndim == 2 else traces[None, :]
    else:
        arr = np.stack([np.asarray(t, dtype=np.float32).reshape(-1) for t in traces], axis=0)
    n_ch, K = arr.shape
    t = np.arange(K, dtype=np.float64) / float(fs_ifr)

    # ---- locate the burst window -------------------------------------------
    i0 = i1 = None
    if zoom_s is not None and float(zoom_s) > 0.0:
        w = max(1, int(round(float(zoom_s) * float(fs_ifr))))
        if w < K:                       # a window covering everything is no zoom
            if zoom_center_s is not None:
                i_c = int(round(float(zoom_center_s) * float(fs_ifr)))
            elif zoom_ref_channel is not None:
                i_c = int(np.argmax(arr[int(zoom_ref_channel)]))
            else:
                i_c = int(np.argmax(arr.mean(axis=0)))
            i_c = int(np.clip(i_c, 0, K - 1))
            # centre the window, then SHIFT (not truncate) at the edges so the
            # close-up keeps its requested width wherever the peak lands
            i0 = i_c - w // 2
            if i0 < 0:
                i0 = 0
            elif i0 + w > K:
                i0 = K - w
            i1 = i0 + w

    zoomed = i0 is not None
    n_col = 2 if zoomed else 1
    fig, axes = plt.subplots(n_ch, n_col,
                             figsize=(9.0 if not zoomed else 13.0,
                                      1.35 * n_ch + 0.6),
                             sharex="col", sharey="row", squeeze=False)

    for ch in range(n_ch):
        ax_full = axes[ch, 0]
        ax_full.plot(t, arr[ch], lw=0.7, color="C%d" % (ch % 10))
        lbl = "ch %d" % ch
        if centers is not None and ch < len(centers):
            lbl += "\n(c=%d)" % int(centers[ch])
        ax_full.set_ylabel(lbl, fontsize=8)
        ax_full.margins(x=0)

        if zoomed:
            # Mark where the close-up came from. The span alone is not enough:
            # 2 s inside 1200 s is ~2 px, so it reads as nothing. The line is
            # always visible; the span becomes meaningful for wide windows.
            ax_full.axvspan(t[i0], t[i1 - 1], color="0.6", alpha=0.35, lw=0)
            ax_full.axvline(t[i0] + 0.5 * (t[i1 - 1] - t[i0]),
                            color="0.25", lw=0.8, ls="--", alpha=0.9)
            ax_zoom = axes[ch, 1]
            ax_zoom.plot(t[i0:i1], arr[ch, i0:i1], lw=0.9,
                         color="C%d" % (ch % 10))
            ax_zoom.margins(x=0)

    axes[-1, 0].set_xlabel("time [s]")
    if zoomed:
        axes[-1, 1].set_xlabel("time [s]")
        axes[0, 0].set_title("full recording", fontsize=9)
        axes[0, 1].set_title("burst close-up  (%.3g s window @ %.2f s)"
                             % (float(zoom_s), t[i0] + 0.5 * (t[i1 - 1] - t[i0])),
                             fontsize=9)

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
