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
uses a 2-row × 2-column layout that isolates the two effects:

    col 0  "warp only (pre-shift)"   : the pure distortion each transform adds
    col 1  "warp + shift (post-shift)": what the network actually receives

When pre-shift tensors are omitted, the figure falls back to the original
2-row × 1-column layout (backward compatible).
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")          # headless backend — no display on compute nodes
import matplotlib.pyplot as plt
import numpy as np
import torch


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

    WITH pre-shift (4 panels, 2 × 2)
    ---------------------------------
    +---------------------------------------------+
    ¦  POSITIVES           ¦  POSITIVES           ¦
    ¦  warp only           ¦  warp + shift        ¦
    ¦  (pre-shift)         ¦  (post-shift)        ¦
    +----------------------+----------------------¦
    ¦  NEGATIVES           ¦  NEGATIVES           ¦
    ¦  warp only           ¦  warp + shift        ¦
    ¦  (pre-shift)         ¦  (post-shift)        ¦
    +---------------------------------------------+

    WITHOUT pre-shift (2 panels, 2 × 1)  — backward-compatible fallback
    ---------------------------------
    +----------------------------------------------+
    ¦  POSITIVES  warp + shift                     ¦
    +----------------------------------------------¦
    ¦  NEGATIVES  warp + shift                     ¦
    +----------------------------------------------+

    Parameters
    ----------
    anchor          : (1, T) or (T,) tensor — clean unshifted window.
    positives       : (P, T) tensor — profile-preserving surrogates, POST-shift.
    negatives       : (N, T) tensor — profile-destroying surrogates, POST-shift.
    fs              : sampling rate f_s  [Hz] — x-axis plotted in seconds.
    out_dir         : directory for the PNG (created if missing).
    instance_id     : integer used in the filename  triplet_{id:03d}.png.
    max_curves      : max surrogates drawn per panel (random subsample).
    title           : optional suptitle string.
    seed            : RNG seed for the subsampling (reproducible).
    positives_pre   : (P, T) tensor — same positives BEFORE the circular shift.
                      When supplied (together with negatives_pre), the 4-panel
                      layout is used so the pure warp effect is visible.
    negatives_pre   : (N, T) tensor — same negatives BEFORE the circular shift.

    Returns
    -------
    out_path : str — absolute path of the written PNG.
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
        fig.text(0.30, 0.97, "warp only  —  pre-shift",
                 ha="center", va="top", fontsize=10, fontweight="bold",
                 color="dimgray")
        fig.text(0.73, 0.97, "warp + shift  —  post-shift",
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
