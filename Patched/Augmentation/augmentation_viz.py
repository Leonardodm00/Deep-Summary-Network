"""
augmentation_viz.py
===================

Visual-debug plotting for the augmentation module. Kept SEPARATE from the
transform logic (separation of concerns -- directive 2): this file imports
nothing from the training/model path and only reads tensors that the
augmentation module produces.

Headless-safe (Agg backend) for HPC batch nodes: figures are written to disk,
never shown interactively.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")          # headless backend (no display on compute nodes)
import matplotlib.pyplot as plt  # noqa: E402


def _as_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _subsample(arr: np.ndarray, max_curves: int, rng: np.random.Generator) -> np.ndarray:
    """Return at most `max_curves` rows of a (N, T) array (random subset, stable)."""
    n = arr.shape[0]
    if n <= max_curves:
        return arr
    idx = np.sort(rng.choice(n, size=max_curves, replace=False))
    return arr[idx]


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
) -> str:
    """
    Save a 2-row visual-debug figure for one triplet instance.

        top    : anchor (bold black) overlaid with a sample of POSITIVES (green)
        bottom : anchor (bold black) overlaid with a sample of NEGATIVES (red)

    Parameters
    ----------
    anchor     : (1, T) or (T,) tensor   -- clean window
    positives  : (P, T) tensor           -- profile-preserving (incl. anchor copy)
    negatives  : (N, T) tensor           -- profile-destroying
    fs         : sampling rate [Hz]      -- x-axis is plotted in seconds
    out_dir    : directory to write the PNG into (created if missing)
    instance_id: integer used in the filename (triplet_{id:03d}.png)
    max_curves : max positives/negatives drawn per panel (subsampled)

    Returns
    -------
    The path of the written PNG.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    a = _as_np(anchor).reshape(-1)
    P = _as_np(positives)
    N = _as_np(negatives)
    if P.ndim == 1:
        P = P[None, :]
    if N.ndim == 1:
        N = N[None, :]

    T = a.shape[0]
    t = np.arange(T) / float(fs)

    P_s = _subsample(P, max_curves, rng)
    N_s = _subsample(N, max_curves, rng)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, sharey=True)

    # --- positives ---
    ax = axes[0]
    for row in P_s:
        ax.plot(t, row, color="tab:green", alpha=0.45, linewidth=0.9)
    ax.plot(t, a, color="black", linewidth=2.0, label="anchor (clean)")
    ax.set_ylabel("amplitude")
    ax.set_title(
        f"POSITIVES  (showing {P_s.shape[0]} / {P.shape[0]})  -- profile-preserving + shift",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    # --- negatives ---
    ax = axes[1]
    for row in N_s:
        ax.plot(t, row, color="tab:red", alpha=0.45, linewidth=0.9)
    ax.plot(t, a, color="black", linewidth=2.0, label="anchor (clean)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("amplitude")
    ax.set_title(
        f"NEGATIVES  (showing {N_s.shape[0]} / {N.shape[0]})  -- profile-destroying + shift",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    if title:
        fig.suptitle(title, fontsize=12)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"triplet_{instance_id:03d}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
