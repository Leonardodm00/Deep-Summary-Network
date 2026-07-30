"""
silhouette_floor.py
===================

The label-shuffled silhouette FLOOR and the improvement threshold it calibrates.
Two torch-free functions (directive 2: testable without a training run):

  1. silhouette_floor(): the label-shuffled NULL distribution of the mean
     silhouette s_bar on a given embedding. Section 12.3 of the v3 handoff
     records that s_bar's reference levels are unmeasured and that its practical
     range is compressed (the best completed cell scored 0.0572), so no absolute
     improvement threshold can be chosen honestly without measuring this first.

  2. resolve_min_delta_sil(): turn a measured floor into the early-stopping
     improvement threshold delta = kappa * (a chosen statistic of the floor).

Change 6 reverted the growing-patience state machine that used to live here
(AdaptivePatience) back to the FIXED integer patience P, whose counter now lives
inline in train.py. This module therefore no longer decides WHEN to stop; it only
measures the floor and forms the threshold that feeds train.py's improvement test.

Choosing the improvement threshold delta
----------------------------------------
"The silhouette did not change beyond a certain threshold, namely 2 times or
more the evaluated floor" fixes delta = kappa * (floor), kappa = 2. The floor
has to be measured, and WHICH statistic of the floor is used matters:

    mode = "floor_location"   delta = kappa * mu_floor
    mode = "floor_scale"      delta = kappa * sigma_floor      (DEFAULT)
    mode = "absolute"         delta = a constant (current behaviour)

mu_floor is the MEAN of s_bar under label permutation and sigma_floor its
standard deviation. The default is "floor_scale", which is a deliberate
deviation from the literal reading, for a reason that is a property of the
silhouette and not a matter of taste:

    Under a random labelling, a(i) and b(i) estimate the same underlying mean
    distance, so E[s_bar] sits AT ZERO rather than at some positive level.
    Finite samples push it slightly below zero, because b(i) takes a MINIMUM
    over the C - 1 other classes and a minimum of noisy quantities is biased
    downward, whereas a(i) involves no minimum. The bias grows with C and is
    weakest at C = 2, where there is no minimum to take.

    So mu_floor is approximately 0 and frequently negative. delta =
    2 * mu_floor is then approximately 0 (no threshold at all) or negative -- and
    a negative threshold declares a DECREASE in s_bar to be an improvement,
    which silently disables early stopping in the worst possible way.

sigma_floor is strictly positive, is what "beyond noise" actually means, and
still honours the "2 times the floor" intent: delta = 2 * sigma_floor asks for a
gain twice the size of the run-to-run wobble the null can produce on this
evaluation set. mode = "floor_location" is provided for the literal reading and
REFUSES to return a non-positive threshold rather than returning one that
disables the mechanism.

HPC note (hpc-python-compat): pure ASCII. Imports numpy and sklearn only.
"""

import math
import warnings
from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import silhouette_score

__all__ = [
    "silhouette_floor",
    "resolve_min_delta_sil",
]

_MODES = ("absolute", "floor_location", "floor_scale")


# --------------------------------------------------------------------------- #
# 1. the label-shuffled floor
# --------------------------------------------------------------------------- #
def silhouette_floor(Z,
                     y: Sequence[int],
                     n_permutations: int = 200,
                     metric: str = "cosine",
                     seed: int = 0) -> Dict[str, float]:
    """Null distribution of the mean silhouette s_bar under label permutation.

    Parameters
    ----------
    Z              : (N, E) embedding matrix, the SAME one the real s_bar was
                     computed on. The geometry is held fixed; only the labels
                     move, which is what isolates "is this s_bar better than
                     chance" from "is this embedding spread out".
    y              : (N,) true labels. Permuting y preserves the class SIZES, so
                     the null keeps the class-imbalance structure of the real
                     evaluation set instead of replacing it with a balanced one.
    n_permutations : R, the number of permutations. sigma_floor is estimated
                     with a relative standard error of about 1/sqrt(2(R-1)),
                     i.e. roughly 5 percent at R = 200.
    metric         : distance passed to sklearn; must match
                     cfg.eval.silhouette_metric ("cosine") or the floor is
                     measured in a different geometry from the metric it is
                     meant to calibrate.
    seed           : RNG seed for the permutations.

    Returns
    -------
    dict with mu, sigma, q05, q50, q95, minimum, maximum, n_permutations,
    n_valid, n_eval, n_classes, metric.

    sigma is the SAMPLE standard deviation (ddof = 1). Permutations that leave
    fewer than 2 distinct labels, or that sklearn rejects, are skipped and
    reported in n_valid; the estimate is formed from the valid ones only.
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError("Z must be 2-D (N, E); got shape %r" % (Z.shape,))
    y = np.asarray(y).astype(np.int64).ravel()
    if y.shape[0] != Z.shape[0]:
        raise ValueError("Z has %d rows but y has %d labels"
                         % (Z.shape[0], y.shape[0]))
    R = int(n_permutations)
    if R < 2:
        raise ValueError("n_permutations must be >= 2 to estimate a spread; "
                         "got %d" % (R,))
    n_classes = int(np.unique(y).shape[0])
    if n_classes < 2:
        raise ValueError("silhouette_floor needs >= 2 distinct labels; got %d"
                         % (n_classes,))

    rng = np.random.default_rng(int(seed))
    values = []
    for _ in range(R):
        y_perm = rng.permutation(y)
        if np.unique(y_perm).shape[0] < 2:         # cannot happen for a permutation
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                values.append(float(silhouette_score(Z, y_perm, metric=metric)))
        except ValueError:
            continue

    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if v.shape[0] < 2:
        raise ValueError(
            "silhouette_floor: only %d valid permutation(s) out of %d; the "
            "embedding is probably degenerate (all rows identical?)"
            % (v.shape[0], R))

    return {
        "mu": float(v.mean()),
        "sigma": float(v.std(ddof=1)),
        "q05": float(np.quantile(v, 0.05)),
        "q50": float(np.quantile(v, 0.50)),
        "q95": float(np.quantile(v, 0.95)),
        "minimum": float(v.min()),
        "maximum": float(v.max()),
        "n_permutations": R,
        "n_valid": int(v.shape[0]),
        "n_eval": int(Z.shape[0]),
        "n_classes": n_classes,
        "metric": str(metric),
    }


def resolve_min_delta_sil(floor: Optional[Dict[str, float]] = None,
                          kappa: float = 2.0,
                          mode: str = "floor_scale",
                          absolute: float = 0.0) -> float:
    """Turn a measured floor into the early-stopping improvement threshold delta.

    See the module docstring for why "floor_scale" is the default and why
    "floor_location" refuses a non-positive result instead of returning one.
    Always returns a float >= 0.
    """
    if mode not in _MODES:
        raise ValueError("mode must be one of %r; got %r" % (_MODES, mode))
    if mode == "absolute":
        d = float(absolute)
        if d < 0.0:
            raise ValueError("absolute delta must be >= 0; got %r" % (d,))
        return d

    if floor is None:
        raise ValueError("mode=%r needs a floor dict from silhouette_floor()"
                         % (mode,))
    k = float(kappa)
    if k <= 0.0:
        raise ValueError("kappa must be > 0; got %r" % (k,))

    if mode == "floor_scale":
        sigma = float(floor["sigma"])
        if not (sigma > 0.0 and math.isfinite(sigma)):
            raise ValueError(
                "floor sigma is %r, so no scale-based threshold can be formed. "
                "The permutation null produced no spread, which usually means a "
                "collapsed embedding." % (sigma,))
        return k * sigma

    # mode == "floor_location"
    mu = float(floor["mu"])
    if not (mu > 0.0 and math.isfinite(mu)):
        raise ValueError(
            "mode='floor_location' asks for delta = %.3g * mu_floor, but the "
            "measured mu_floor is %.6g (<= 0). A non-positive threshold makes a "
            "DECREASE in silhouette count as an improvement, which disables "
            "early stopping silently. mu_floor near or below zero is the NORMAL "
            "case for a permutation null (see the module docstring); use "
            "mode='floor_scale', which gives delta = %.3g * sigma_floor = %.6g "
            "here." % (k, mu, k, k * float(floor["sigma"])))
    return k * mu
