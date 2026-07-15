"""
augmentation_viz.py
===================

Visual-debug plotting for the augmentation module. Kept SEPARATE from the
transform logic (separation of concerns -- directive 2): this file imports
nothing from the training/model path and only reads tensors that the
augmentation module produces.

Headless-safe (Agg backend) for HPC batch nodes: figures are written to disk,
never shown interactively.

Layout
------
When pre-shift tensors are supplied (positives_pre, negatives_pre), the figure
uses a 2-row x 2-column layout that isolates the two effects:

    col 0  "warp only (pre-shift)"   : the pure distortion each transform adds
    col 1  "warp + shift (post-shift)": what the network actually receives

When pre-shift tensors are omitted, the figure falls back to the original
2-row x 1-column layout (backward compatible).
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")          # headless backend -- no display on compute nodes
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import find_peaks


# --------------------------------------------------------------------------- #
# internal helpers
# --------------------------------------------------------------------------- #
def _as_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    return arr[None, :] if arr.ndim == 1 else arr


def _subsample(arr: np.ndarray, max_curves: int, rng: np.random.Generator) -> np.ndarray:
    """Return at most `max_curves` rows, drawn without replacement (stable sort)."""
    n = arr.shape[0]
    if n <= max_curves:
        return arr
    return arr[np.sort(rng.choice(n, size=max_curves, replace=False))]


def _draw_panel(ax, t, anchor, surrogates, color, label_pre, n_shown, n_total):
    """Draw one panel: surrogates in `color`, anchor in black."""
    for row in surrogates:
        ax.plot(t, row, color=color, alpha=0.40, linewidth=0.85)
    ax.plot(t, anchor, color="black", linewidth=2.0, label="anchor (clean)")
    ax.set_title(
        f"{label_pre}  (showing {n_shown} / {n_total})",
        fontsize=9,
    )
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.22)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def plot_triplet_instance(
    anchor: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    fs: float,
    out_dir: str,
    instance_id: int = 0,
    max_curves: int = 8,
    title: Optional[str] = None,
    seed: int = 0,
    positives_pre: Optional[torch.Tensor] = None,
    negatives_pre: Optional[torch.Tensor] = None,
) -> str:
    """Save a visual-debug figure for one triplet instance.

    Layout depends on whether pre-shift tensors are supplied:

    WITH pre-shift (4 panels, 2 x 2)
    ---------------------------------
    +----------------------+----------------------+
    |  POSITIVES           |  POSITIVES           |
    |  warp only           |  warp + shift        |
    |  (pre-shift)         |  (post-shift)        |
    +----------------------+----------------------+
    |  NEGATIVES           |  NEGATIVES           |
    |  warp only           |  warp + shift        |
    |  (pre-shift)         |  (post-shift)        |
    +----------------------+----------------------+

    WITHOUT pre-shift (2 panels, 2 x 1)  -- backward-compatible fallback
    ---------------------------------
    +----------------------------------------------+
    |  POSITIVES  warp + shift                     |
    +----------------------------------------------+
    |  NEGATIVES  warp + shift                     |
    +----------------------------------------------+

    Parameters
    ----------
    anchor          : (1, T) or (T,) tensor -- clean unshifted window.
    positives       : (P, T) tensor -- profile-preserving surrogates, POST-shift.
    negatives       : (N, T) tensor -- profile-destroying surrogates, POST-shift.
    fs              : sampling rate f_s  [Hz] -- x-axis plotted in seconds.
    out_dir         : directory for the PNG (created if missing).
    instance_id     : integer used in the filename  triplet_{id:03d}.png.
    max_curves      : max surrogates drawn per panel (random subsample).
    title           : optional suptitle string.
    seed            : RNG seed for the subsampling (reproducible).
    positives_pre   : (P, T) tensor -- same positives BEFORE the circular shift.
                      When supplied (together with negatives_pre), the 4-panel
                      layout is used so the pure warp effect is visible.
    negatives_pre   : (N, T) tensor -- same negatives BEFORE the circular shift.

    Returns
    -------
    out_path : str -- absolute path of the written PNG.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    a   = _as_np(anchor).reshape(-1)
    P   = _ensure_2d(_as_np(positives))
    N   = _ensure_2d(_as_np(negatives))
    t   = np.arange(a.shape[0]) / float(fs)

    have_pre = (positives_pre is not None) and (negatives_pre is not None)

    if have_pre:
        # -- 4-panel layout ------------------------------------------------
        Pp  = _ensure_2d(_as_np(positives_pre))
        Np  = _ensure_2d(_as_np(negatives_pre))

        Pp_s = _subsample(Pp, max_curves, rng)
        P_s  = _subsample(P,  max_curves, rng)
        Np_s = _subsample(Np, max_curves, rng)
        N_s  = _subsample(N,  max_curves, rng)

        fig, axes = plt.subplots(
            2, 2, figsize=(16, 7),
            sharex=True, sharey=True,
            gridspec_kw={"wspace": 0.06, "hspace": 0.35},
        )

        # column headings (drawn as centered text above each column)
        fig.text(0.30, 0.97, "warp only  --  pre-shift",
                 ha="center", va="top", fontsize=10, fontweight="bold",
                 color="dimgray")
        fig.text(0.73, 0.97, "warp + shift  --  post-shift",
                 ha="center", va="top", fontsize=10, fontweight="bold",
                 color="dimgray")

        # row 0: positives
        _draw_panel(axes[0, 0], t, a, Pp_s, "tab:green",
                    f"POSITIVES  pre-shift", Pp_s.shape[0], Pp.shape[0])
        _draw_panel(axes[0, 1], t, a, P_s,  "tab:green",
                    f"POSITIVES  post-shift", P_s.shape[0], P.shape[0])

        # row 1: negatives
        _draw_panel(axes[1, 0], t, a, Np_s, "tab:red",
                    f"NEGATIVES  pre-shift", Np_s.shape[0], Np.shape[0])
        _draw_panel(axes[1, 1], t, a, N_s,  "tab:red",
                    f"NEGATIVES  post-shift", N_s.shape[0], N.shape[0])

        for ax in axes[:, 0]:
            ax.set_ylabel("amplitude")
        for ax in axes[1, :]:
            ax.set_xlabel("time  t  [s]")

    else:
        # -- 2-panel fallback (backward compatible) ------------------------
        P_s = _subsample(P, max_curves, rng)
        N_s = _subsample(N, max_curves, rng)

        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, sharey=True)
        axes = axes.reshape(2, 1)   # unify indexing

        _draw_panel(axes[0, 0], t, a, P_s, "tab:green",
                    f"POSITIVES  profile-preserving + shift",
                    P_s.shape[0], P.shape[0])
        _draw_panel(axes[1, 0], t, a, N_s, "tab:red",
                    f"NEGATIVES  profile-destroying + shift",
                    N_s.shape[0], N.shape[0])

        axes[0, 0].set_ylabel("amplitude")
        axes[1, 0].set_ylabel("amplitude")
        axes[1, 0].set_xlabel("time  t  [s]")

    if title:
        fig.suptitle(title, fontsize=11, y=1.01 if have_pre else 1.02)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"triplet_{instance_id:03d}.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_triplet_bursts(
    anchor: torch.Tensor,
    positives_pre: torch.Tensor,
    negatives_pre: torch.Tensor,
    fs: float,
    out_dir: str,
    instance_id: int = 0,
    max_bursts: int = 3,
    zoom_half_s: float = 2.0,
    max_curves: int = 8,
    title: Optional[str] = None,
    seed: int = 0,
    peak_height_frac: float = 0.30,
    peak_distance_s: float = 1.0,
) -> str:
    """Save a burst-zoomed visual-debug figure for one triplet instance.

    Detects the N strongest burst peaks in the anchor trace and produces one
    figure with N_bursts rows and 2 columns:

        col 0  anchor (black) + POSITIVES (green)  zoomed to the burst
        col 1  anchor (black) + NEGATIVES (red)    zoomed to the burst

    All surrogates shown are PRE-SHIFT so the figure isolates the pure warp
    effect on individual burst shapes, with no circular-shift confound.

    Burst detection uses scipy.signal.find_peaks on the anchor with:
        height  = peak_height_frac * max(anchor)
        distance = peak_distance_s * fs  [samples]
    If no peaks are found the global maximum is used as a single pseudo-burst.
    The top max_bursts peaks (by amplitude) are shown in temporal order.

    Parameters
    ----------
    anchor          : (1, T) or (T,) tensor  clean unshifted window.
    positives_pre   : (P, T) tensor          positives BEFORE circular shift.
    negatives_pre   : (N, T) tensor          negatives BEFORE circular shift.
    fs              : float                  sampling rate [Hz].
    out_dir         : str                    output directory (created if missing).
    instance_id     : int                    index used in filename
                                             triplet_{id:03d}_bursts.png
    max_bursts      : int                    max burst panels to draw.
    zoom_half_s     : float                  half-window around each peak [s].
                                             Capped at half the trace length.
    max_curves      : int                    max surrogates per panel.
    title           : str or None            figure suptitle.
    seed            : int                    RNG seed for subsampling.
    peak_height_frac: float                  burst detection height threshold
                                             as a fraction of max(anchor).
    peak_distance_s : float                  min inter-peak distance [s].

    Returns
    -------
    out_path : str  path of the written PNG.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    a  = _as_np(anchor).reshape(-1)          # (T,)
    T  = a.shape[0]
    t  = np.arange(T) / float(fs)

    P  = _ensure_2d(_as_np(positives_pre))   # (P, T)
    N  = _ensure_2d(_as_np(negatives_pre))   # (N, T)
    Ps = _subsample(P, max_curves, rng)
    Ns = _subsample(N, max_curves, rng)

    # ------------------------------------------------------------------
    # burst detection on the anchor
    # ------------------------------------------------------------------
    height_thr  = float(peak_height_frac * float(a.max()))
    dist_samp   = max(1, int(peak_distance_s * fs))
    peaks, _    = find_peaks(a, height=height_thr, distance=dist_samp)

    if len(peaks) == 0:
        # fallback: treat the global maximum as one burst
        peaks = np.array([int(np.argmax(a))], dtype=int)

    # keep the tallest max_bursts peaks, displayed in temporal order
    if len(peaks) > max_bursts:
        top_idx = np.argsort(a[peaks])[::-1][:max_bursts]
        peaks   = np.sort(peaks[top_idx])

    n_bursts = len(peaks)
    # zoom half-window in samples, capped at half the trace
    half = min(int(zoom_half_s * fs), T // 2)

    # ------------------------------------------------------------------
    # figure: n_bursts rows x 2 columns
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        n_bursts, 2,
        figsize=(14, 3.5 * n_bursts),
        sharey=False,
        gridspec_kw={"hspace": 0.50, "wspace": 0.08},
    )
    # always make axes 2-D for uniform indexing
    if n_bursts == 1:
        axes = axes[None, :]

    for row, pk in enumerate(peaks):
        s      = max(0, pk - half)
        e      = min(T, pk + half)
        sl     = slice(s, e)
        t_zoom = t[sl]
        t_peak = float(t[pk])

        # --- left: positives ---
        ax = axes[row, 0]
        for curve in Ps:
            ax.plot(t_zoom, curve[sl], color="tab:green", alpha=0.40, linewidth=0.85)
        ax.plot(t_zoom, a[sl], color="black", linewidth=2.0, label="anchor (clean)")
        ax.axvline(t_peak, color="gray", linestyle="--", linewidth=0.8, alpha=0.55)
        ax.set_title(
            f"Burst {row + 1}  t = {t_peak:.2f} s  |  POSITIVES  pre-shift",
            fontsize=9,
        )
        ax.set_ylabel("amplitude")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.22)

        # --- right: negatives ---
        ax = axes[row, 1]
        for curve in Ns:
            ax.plot(t_zoom, curve[sl], color="tab:red", alpha=0.40, linewidth=0.85)
        ax.plot(t_zoom, a[sl], color="black", linewidth=2.0, label="anchor (clean)")
        ax.axvline(t_peak, color="gray", linestyle="--", linewidth=0.8, alpha=0.55)
        ax.set_title(
            f"Burst {row + 1}  t = {t_peak:.2f} s  |  NEGATIVES  pre-shift",
            fontsize=9,
        )
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.22)

    for ax in axes[-1, :]:
        ax.set_xlabel("time  t  [s]")

    # column headers
    fig.text(0.28, 0.995,
             f"anchor + POSITIVES  (warp only, pre-shift)  "
             f"-- showing {Ps.shape[0]}/{P.shape[0]}",
             ha="center", va="top", fontsize=10,
             fontweight="bold", color="tab:green")
    fig.text(0.73, 0.995,
             f"anchor + NEGATIVES  (warp only, pre-shift)  "
             f"-- showing {Ns.shape[0]}/{N.shape[0]}",
             ha="center", va="top", fontsize=10,
             fontweight="bold", color="tab:red")

    if title:
        fig.suptitle(title, fontsize=11, y=1.03)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"triplet_{instance_id:03d}_bursts.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path
