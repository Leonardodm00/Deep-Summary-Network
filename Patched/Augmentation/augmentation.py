"""
augmentation.py
===============

Decoupled, CPU-side data-augmentation transforms for 1-D neuronal-activity
windows (e.g. smoothed cumulative IFR traces), for the 1D-CNN summary network.

Pipeline role (separation of concerns -- directive 2)
-----------------------------------------------------
    * THIS module : pure transforms + the positive/negative split + the
                    per-anchor triplet builder. No model, no device logic,
                    no plotting, no data loading.
    * Visualization : augmentation_viz.py
    * Smoke test    : smoke_test_augmentation.py

Design decisions locked with the user
-------------------------------------
    * magnitude warp is done in LOG-SPACE  -> multiplier strictly positive
      (firing rate stays non-negative for any sigma_mag).
    * time warp pins the endpoints (phi(0)=0, phi(T-1)=T-1) -> no clip-induced
      edge plateaus; folds (non-monotonic phi) are allowed and fall in negatives.
    * magnitude and time warps use SEPARATE strengths: sigma_mag (dimensionless)
      and sigma_time_s (seconds).
    * positive/negative split is selectable:
        - "warp_bands"      (option 3): label by the strength band sampled from.
        - "percentile_mse"  (option 2): split the UNSHIFTED surrogates by a
                                        per-anchor MSE quantile.
      In BOTH cases the split is computed BEFORE the shift; the circular shift
      is then applied to both classes as a label-preserving augmentation so the
      network learns translation-invariant features.
    * the clean (unshifted) anchor is included among the positives -> the
      embedding is calibrated on the exact distribution seen at inference.

Notation (consistent with the design notes)
-------------------------------------------
    x(t)        : input window, samples t = 0, ..., T-1  (x(t) >= 0)
    g_k         : magnitude knot log-gains, k = 1, ..., K,  g_k ~ N(0, sigma_mag^2)
    s(t)        : cubic spline through {(t_k, g_k)}
    c(t)        : magnitude scaling curve, c(t) = exp(s(t)) > 0
    delta_k     : time knot offsets (samples), delta_k ~ N(0, (sigma_time_s*fs)^2),
                  with delta_1 = delta_K = 0 (pinned endpoints)
    phi(t)      : warped index map, phi(t) = t + CubicSpline({(t_k, delta_k)})(t)

Note (flagged abuse): c(t) = exp(s(t)) is log-normal with MEDIAN 1 but
MEAN exp(sigma_mag^2 / 2) > 1 (a slight upward amplitude bias, ~2% at
sigma_mag = 0.2). Median-1 + strict positivity is what we want for augmentation,
so the bias is kept and documented rather than corrected.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from scipy.interpolate import CubicSpline  # established library (directive 1)

__all__ = [
    "AugmentationConfig",
    "magnitude_warp",
    "time_warp",
    "random_circular_shift",
    "build_triplet_instance",
]

# --------------------------------------------------------------------------- #
# dtype policy (single, robust guarantee applied at every tensor boundary)
#   * every tensor that leaves this module is CPU torch.float32
#   * scipy works transiently in float64 (numerical conditioning of the spline)
# --------------------------------------------------------------------------- #
_TORCH_DTYPE = torch.float32
_NUMPY_WORK_DTYPE = np.float64


def _to_work_array(x) -> np.ndarray:
    """torch/np -> contiguous float64 numpy array (for scipy)."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.ascontiguousarray(x, dtype=_NUMPY_WORK_DTYPE)


def _to_tensor(x) -> torch.Tensor:
    """np/torch -> contiguous CPU float32 tensor."""
    if isinstance(x, torch.Tensor):
        return x.detach().to(_TORCH_DTYPE).cpu().contiguous()
    return torch.as_tensor(np.ascontiguousarray(x), dtype=_TORCH_DTYPE)


def _n_knots(T: int, fs: float, intra_knot_dist: float, k_min: int) -> int:
    """
    Number of spline knots for a window of length T at sampling rate fs.

        knots_per_second = int(1 / intra_knot_dist)        (integer by design)
        K                = max(k_min, int(knots_per_second * T / fs))

    Guard: CubicSpline needs >= 2 knots; we enforce K >= k_min (>= 4 recommended
    for a well-posed not-a-knot cubic). Raises if the window is too short.
    """
    if T < k_min:
        raise ValueError(f"Window length T={T} < k_min={k_min}; cannot place knots.")
    knots_per_second = int(1.0 / intra_knot_dist)
    K = int(knots_per_second * T / fs)
    return max(k_min, K)


# --------------------------------------------------------------------------- #
# Configuration (all strengths exposed and TUNABLE -- the user will tune them)
# --------------------------------------------------------------------------- #
@dataclass
class AugmentationConfig:
    """Per-anchor augmentation settings. The sigma_* bands are PLACEHOLDERS to be
    tuned against the network-burst time scale (calculate_mean_burst_duration)."""

    fs: float                                  # sampling rate [Hz]
    intra_knot_dist: float = 0.2               # seconds between spline knots

    # --- magnitude warp: log-amplitude std (dimensionless) -------------------
    sigma_mag_pos: Tuple[float, float] = (0.01, 0.10)   # positive band  [TUNE]
    sigma_mag_neg: Tuple[float, float] = (0.20, 0.50)   # negative band  [TUNE]

    # --- time warp: temporal std in SECONDS ----------------------------------
    sigma_time_pos_s: Tuple[float, float] = (0.005, 0.050)  # positive band [TUNE]
    sigma_time_neg_s: Tuple[float, float] = (0.100, 0.400)  # negative band [TUNE]

    # --- circular shift ------------------------------------------------------
    shift_magnitude_s: float = 30.0            # max |shift| in seconds

    # --- counts --------------------------------------------------------------
    n_positives: int = 30                      # exact for "warp_bands"; pool=pos+neg for "percentile_mse"
    n_negatives: int = 30

    # --- split method --------------------------------------------------------
    split_method: str = "warp_bands"           # "warp_bands" (opt 3) | "percentile_mse" (opt 2)
    percentile_q: float = 0.30                 # fraction labelled positive (used iff "percentile_mse")

    # --- safeguards ----------------------------------------------------------
    k_min: int = 4                             # min spline knots
    max_retries: int = 5                       # empty-class re-draws before giving up
    enforce_nonneg: bool = True                # clamp surrogates to >= 0 (physical firing rate)


# --------------------------------------------------------------------------- #
# Pure transforms (single 1-D window in, single 1-D window out)
# --------------------------------------------------------------------------- #
def magnitude_warp(
    window,
    fs: float,
    sigma_mag: float,
    intra_knot_dist: float,
    rng: np.random.Generator,
    k_min: int = 4,
) -> torch.Tensor:
    """
    Log-space magnitude warp (strictly positive multiplier).

        g_k ~ N(0, sigma_mag^2),   k = 1..K
        s(t) = CubicSpline({(t_k, g_k)})(t)
        c(t) = exp(s(t)) > 0
        x~(t) = x(t) * c(t),   t = 0..T-1

    Returns a (T,) CPU float32 tensor.
    """
    x = _to_work_array(window).ravel()
    T = x.shape[0]
    K = _n_knots(T, fs, intra_knot_dist, k_min)
    t = np.arange(T, dtype=_NUMPY_WORK_DTYPE)
    t_knots = np.linspace(0.0, T - 1.0, K)
    g = rng.normal(loc=0.0, scale=sigma_mag, size=K)
    s = CubicSpline(t_knots, g)(t)
    c = np.exp(s)                       # strictly positive by construction
    return _to_tensor(x * c)


def time_warp(
    window,
    fs: float,
    sigma_time_s: float,
    intra_knot_dist: float,
    rng: np.random.Generator,
    k_min: int = 4,
) -> torch.Tensor:
    """
    Time warp with PINNED endpoints (edges untouched -> no clip plateaus).

        delta_k ~ N(0, (sigma_time_s * fs)^2)  [samples],  delta_1 = delta_K = 0
        phi(t)  = t + CubicSpline({(t_k, delta_k)})(t)
        x~(t)   = CubicSpline(t, x)(phi(t)),   t = 0..T-1

    phi is NOT constrained monotonic (folds allowed -> negatives by design).
    Out-of-range interior phi is handled by the resampling spline's own
    extrapolation, never by edge clipping. Endpoints are exact because
    phi(0)=0 and phi(T-1)=T-1.

    Returns a (T,) CPU float32 tensor.
    """
    x = _to_work_array(window).ravel()
    T = x.shape[0]
    K = _n_knots(T, fs, intra_knot_dist, k_min)
    t = np.arange(T, dtype=_NUMPY_WORK_DTYPE)
    t_knots = np.linspace(0.0, T - 1.0, K)

    sigma_samples = sigma_time_s * fs
    delta = rng.normal(loc=0.0, scale=sigma_samples, size=K)
    delta[0] = 0.0
    delta[-1] = 0.0                     # pin endpoints
    phi = t + CubicSpline(t_knots, delta)(t)

    # resample the signal at the warped indices (extrapolate=True by default)
    warped = CubicSpline(t, x)(phi)
    return _to_tensor(warped)


def random_circular_shift(
    windows: torch.Tensor,
    shift_magnitude_s: float,
    fs: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    """
    Vectorized circular shift: each row is rolled by its own random integer
    shift in [-S, S], S = int(shift_magnitude_s * fs). Equivalent to a per-row
    torch.roll(row, shift) (out[i] = row[i - shift]).

    windows : (T,) or (B, T) float32 tensor.
    Returns : same shape and dtype.
    """
    single = windows.ndim == 1
    w = windows.unsqueeze(0) if single else windows
    B, T = w.shape

    max_shift = int(shift_magnitude_s * fs)
    if max_shift <= 0:
        warnings.warn(
            f"random_circular_shift: shift_magnitude_s*fs = {shift_magnitude_s * fs:.4f} "
            f"rounds to 0 -> no shift applied.",
            RuntimeWarning,
        )
        return windows

    shifts = rng.integers(-max_shift, max_shift + 1, size=B)          # (B,)
    base = np.arange(T)[None, :]                                      # (1, T)
    idx = (base - shifts[:, None]) % T                               # (B, T) circular
    idx_t = torch.as_tensor(idx, dtype=torch.long)
    out = torch.gather(w, dim=1, index=idx_t)
    return out.squeeze(0) if single else out


# --------------------------------------------------------------------------- #
# Surrogate generation + split + triplet builder
# --------------------------------------------------------------------------- #
def _make_surrogate(window, cfg, rng, sigma_mag_range, sigma_time_range) -> torch.Tensor:
    """One surrogate = time_warp(magnitude_warp(x)) with per-surrogate strengths."""
    sm = float(rng.uniform(*sigma_mag_range))
    st = float(rng.uniform(*sigma_time_range))
    w = magnitude_warp(window, cfg.fs, sm, cfg.intra_knot_dist, rng, cfg.k_min)
    w = time_warp(w, cfg.fs, st, cfg.intra_knot_dist, rng, cfg.k_min)
    if cfg.enforce_nonneg:
        w = torch.clamp_min(w, 0.0)    # cubic-resample overshoot can dip slightly < 0
    return w


def _generate_pool(window, cfg, rng, n, sigma_mag_range, sigma_time_range) -> torch.Tensor:
    rows = [_make_surrogate(window, cfg, rng, sigma_mag_range, sigma_time_range) for _ in range(n)]
    return torch.stack(rows, dim=0)    # (n, T)


def _split_percentile_mse(pool: torch.Tensor, anchor: torch.Tensor, q: float):
    """
    Option 2: per-anchor MSE quantile split (on UNSHIFTED surrogates).

        d_m  = mean_t ( pool_m(t) - x(t) )^2,   m = 1..n
        tau  = Q_q({d_m})                       (per-anchor q-quantile)
        pos  = {m : d_m <= tau},  neg = {m : d_m > tau}
    """
    a = anchor.reshape(1, -1)
    d = ((pool - a) ** 2).mean(dim=1)              # (n,)
    tau = torch.quantile(d, q)
    pos_mask = d <= tau
    return pool[pos_mask], pool[~pos_mask]


def build_triplet_instance(window, cfg: AugmentationConfig, rng: np.random.Generator):
    """
    Build one anchor's contrastive instance.

    Returns
    -------
    anchor    : (1, T)  clean, UNSHIFTED window (also embedded at inference)
    positives : (1+P, T) clean anchor + profile-preserving surrogates, shifted
    negatives : (N, T)   profile-destroying surrogates, shifted

    The split (per cfg.split_method) is computed BEFORE the shift; the circular
    shift is then applied to both classes (label-preserving). Empty classes
    trigger a re-draw (with a warning); persistent failure raises.

    This function is condition-AGNOSTIC: the condition-level label (control vs
    pathological) is attached downstream by the batch sampler.
    """
    window = _to_tensor(window).reshape(-1)        # (T,)
    pos = neg = None

    for attempt in range(cfg.max_retries):
        if cfg.split_method == "warp_bands":               # option 3
            pos = _generate_pool(window, cfg, rng, cfg.n_positives,
                                 cfg.sigma_mag_pos, cfg.sigma_time_pos_s)
            neg = _generate_pool(window, cfg, rng, cfg.n_negatives,
                                 cfg.sigma_mag_neg, cfg.sigma_time_neg_s)

        elif cfg.split_method == "percentile_mse":         # option 2
            n_pool = cfg.n_positives + cfg.n_negatives
            broad_mag = (cfg.sigma_mag_pos[0], cfg.sigma_mag_neg[1])
            broad_time = (cfg.sigma_time_pos_s[0], cfg.sigma_time_neg_s[1])
            pool = _generate_pool(window, cfg, rng, n_pool, broad_mag, broad_time)
            pos, neg = _split_percentile_mse(pool, window, cfg.percentile_q)

        else:
            raise ValueError(f"Unknown split_method: {cfg.split_method!r}")

        if pos.shape[0] >= 1 and neg.shape[0] >= 1:
            break
        warnings.warn(
            f"build_triplet_instance: empty positive/negative class "
            f"(attempt {attempt + 1}/{cfg.max_retries}) -> re-drawing.",
            RuntimeWarning,
        )
    else:
        raise RuntimeError(
            "build_triplet_instance: could not obtain non-empty positive AND "
            "negative classes after retries; check sigma bands / percentile_q."
        )

    # split-before-shift: shift both classes (label-preserving translation aug)
    pos = random_circular_shift(pos, cfg.shift_magnitude_s, cfg.fs, rng)
    neg = random_circular_shift(neg, cfg.shift_magnitude_s, cfg.fs, rng)

    # include the clean (unshifted) anchor among the positives
    anchor = window.reshape(1, -1)
    positives = torch.cat([anchor, pos], dim=0)
    return anchor, positives, neg
