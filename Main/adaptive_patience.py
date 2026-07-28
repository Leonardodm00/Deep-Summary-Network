"""
adaptive_patience.py
====================

Two decoupled pieces, both torch-free so they can be tested without a training
run (directive 2):

  1. silhouette_floor(): the label-shuffled NULL distribution of the mean
     silhouette s_bar on a given embedding. Section 12.3 of the v3 handoff
     records that s_bar's reference levels are unmeasured and that its practical
     range is compressed (the best completed cell scored 0.0572), so no absolute
     improvement threshold can be chosen honestly without measuring this first.

  2. AdaptivePatience: the early-stopping state machine with a patience budget
     that GROWS while the primary metric sits on a plateau, instead of being the
     fixed integer P.

Why growing patience
--------------------
With selection_primary = "silhouette" the primary signal is continuous, so
train.min_delta_sil = 0.0 means essentially every epoch counts as an improvement
and `patience` never fires. Raising the threshold to a constant instead makes
the opposite error: on a compressed metric, a constant large enough to reject
noise also rejects real progress. The resolution taken here is to make
non-improvement WEAK evidence rather than decisive evidence: a plateau epoch
still advances the wait counter, but it also buys a little more budget, so the
run is stopped only by a plateau that persists.

    Let w be the wait counter and P the budget, updated per epoch as

        improvement :  w <- 0 ,          P <- P_0  (if reset_on_improvement)
        plateau     :  w <- w + 1 ,      P <- min(P + g, P_max)
        stop iff       w >= P .                                          (1)

    On a run of n CONSECUTIVE plateau epochs starting from (w_0, P_0),
    w = w_0 + n and P = min(P_0 + n g, P_max), so with P_max = inf the stop
    condition w_0 + n >= P_0 + n g first holds at

        n* = ceil( (P_0 - w_0) / (1 - g) ) ,     valid for 0 <= g < 1 .   (2)

    Two consequences worth stating explicitly:

      * g < 1 is REQUIRED for termination when P_max is infinite. At g >= 1 the
        budget grows at least as fast as the counter and the run never stops on
        its own; only max_epochs would end it. __init__ rejects that
        combination rather than letting it surface as a wasted cluster job.
      * g is interpretable: it multiplies the effective patience by 1/(1-g).
        g = 0.0 reproduces fixed patience EXACTLY (n* = P_0); g = 0.5 doubles
        it; g = 0.75 quadruples it. Pick g from how long a plateau you are
        willing to sit through, not by feel.

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
    "effective_patience_bound",
    "AdaptivePatience",
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


# --------------------------------------------------------------------------- #
# 2. the growing-patience state machine
# --------------------------------------------------------------------------- #
_TOL = 1e-9


def _smallest_plateau_count(patience: int,
                            growth: float,
                            max_patience: Optional[int],
                            wait: int,
                            counter: int) -> float:
    """Smallest n >= 1 with  wait + n >= min(P_0 + g (counter + n), P_max).

    This solves EXACTLY the inequality AdaptivePatience.should_stop tests, which
    is why the two cannot drift apart. Two roots, and the answer is the smaller:

        uncapped :  (wait + n) - g (counter + n) >= P_0
                    n (1 - g) >= P_0 + g * counter - wait
                    n >= (P_0 + g * counter - wait) / (1 - g)       [g < 1 only]
        capped   :  n >= P_max - wait

    _TOL absorbs binary-representation error. Without it, P_0 = 2, g = 0.9 gives
    2 / (1 - 0.9) = 20.000000000000004 and ceil() reports 21 for an answer that
    is exactly 20 in real arithmetic.
    """
    P0 = int(patience)
    g = float(growth)
    w = int(wait)
    c = int(counter)
    candidates = []
    if g < 1.0:
        numerator = P0 + g * c - w
        candidates.append(math.ceil(numerator / (1.0 - g) - _TOL))
    if max_patience is not None:
        candidates.append(int(max_patience) - w)
    if not candidates:
        return float("inf")
    return float(max(1, min(candidates)))


def effective_patience_bound(patience: int,
                             growth: float = 0.0,
                             max_patience: Optional[int] = None,
                             wait: int = 0) -> float:
    """Number of CONSECUTIVE plateau epochs survivable from a consistent state.

    "Consistent" means the budget counter equals `wait`, which is the invariant
    the default reset_on_improvement = True maintains: an improvement zeroes both
    at once, so after w plateau epochs the budget is P_0 + g w, never P_0. Under
    that invariant the stop condition (wait + n) >= P_0 + g (wait + n) collapses
    to a statement about the TOTAL wait, giving

        n* = min( ceil(P_0 / (1 - g)) - wait   [g < 1],
                  P_max - wait                 [P_max set] ),   at least 1.

    At wait = 0 this is the headline result n* = ceil(P_0 / (1 - g)): the growth
    rate multiplies the effective patience by 1 / (1 - g). Returns float("inf")
    when g >= 1 with no cap, the configuration __init__ rejects.

    This is the quantity config.validate() should compare max_epochs against;
    comparing against the raw `patience` understates it by exactly that factor.
    """
    P0 = int(patience)
    g = float(growth)
    w = int(wait)
    if P0 < 1:
        raise ValueError("patience must be >= 1; got %d" % (P0,))
    if g < 0.0:
        raise ValueError("growth must be >= 0; got %r" % (g,))
    if w < 0:
        raise ValueError("wait must be >= 0; got %d" % (w,))
    return _smallest_plateau_count(P0, g, max_patience, wait=w, counter=w)


class AdaptivePatience:
    """Early-stopping counter whose budget grows while the primary metric is flat.

    Drop-in for the two lines `patience_counter += 1` / `if patience_counter >=
    P` in train.py. With the defaults (growth = 0.0, reset_on_improvement =
    True) it is EXACTLY the fixed-patience rule, so wiring it in changes no
    existing result; the behaviour only changes once growth > 0.

    Parameters
    ----------
    patience   : P_0, the starting budget (>= 1).
    growth     : g, budget added per plateau epoch (>= 0). Must be < 1 unless
                 max_patience is set, or the run can never stop itself.
    max_patience : P_max, hard cap on the budget. None means uncapped.
    reset_on_improvement : if True (default) an improvement returns the budget to
                 P_0; if False the budget earned so far is kept, so a run that
                 has already shown itself to be a slow improver stays patient.
    delta      : the improvement threshold used by is_improvement(). May be left
                 at None and supplied later by arm() once the floor has been
                 measured on a real embedding.

    The class only counts; it does not decide what "improvement" means beyond
    the delta comparison, so train.py's locked lexicographic rule
    (_is_improvement) stays the single source of truth and is passed in as a
    bool.
    """

    def __init__(self,
                 patience: int,
                 growth: float = 0.0,
                 max_patience: Optional[int] = None,
                 reset_on_improvement: bool = True,
                 delta: Optional[float] = None):
        P0 = int(patience)
        g = float(growth)
        if P0 < 1:
            raise ValueError("patience must be >= 1; got %d" % (P0,))
        if g < 0.0:
            raise ValueError("growth must be >= 0; got %r" % (g,))
        if max_patience is not None:
            if int(max_patience) < P0:
                raise ValueError(
                    "max_patience (%d) must be >= patience (%d), or the budget "
                    "starts above its own cap"
                    % (int(max_patience), P0))
        if g >= 1.0 and max_patience is None:
            raise ValueError(
                "growth = %r with no max_patience never terminates: the budget "
                "grows by %r per plateau epoch while the wait counter grows by "
                "1, so w >= P can only be reached by max_epochs. Use growth < 1 "
                "(effective patience = patience / (1 - growth)) or set "
                "max_patience." % (g, g))
        if delta is not None and float(delta) < 0.0:
            raise ValueError("delta must be >= 0; got %r" % (delta,))

        self.patience_0 = P0
        self.growth = g
        self.max_patience = None if max_patience is None else int(max_patience)
        self.reset_on_improvement = bool(reset_on_improvement)
        self.delta = None if delta is None else float(delta)

        self.wait = 0
        # The budget is DERIVED from a counter rather than accumulated by
        # repeated addition. Repeated addition drifts (2.0 + 0.9 twenty times is
        # not 20.0 in binary floating point) and, worse, lets the object hold an
        # inconsistent state in which wait and budget disagree about how many
        # plateau epochs have happened. One multiplication cannot do either.
        self._counter = 0
        self.n_plateau = 0
        self.n_improvement = 0
        self.history = []                          # (wait, budget) per update

    # -- threshold ---------------------------------------------------------- #
    @property
    def is_armed(self) -> bool:
        """True once a delta is available (either passed in or set by arm())."""
        return self.delta is not None

    def arm(self, delta: float) -> float:
        """Supply the improvement threshold once the floor has been measured.

        Called at the first epoch that yields a finite silhouette, because the
        floor is a property of the EMBEDDING and does not exist before one.
        Returns the delta actually set.
        """
        d = float(delta)
        if d < 0.0:
            raise ValueError("delta must be >= 0; got %r" % (d,))
        self.delta = d
        return d

    def is_improvement(self, value: float, best: float) -> bool:
        """value > best + delta, NaN-safe (a non-finite value never improves).

        Provided for standalone use. train.py already owns the composite
        lexicographic rule; there, pass that rule's bool straight to update().
        """
        if self.delta is None:
            raise RuntimeError(
                "AdaptivePatience is not armed: no delta has been set. Call "
                "arm(delta) once the silhouette floor has been measured, or "
                "construct with delta=...")
        if not np.isfinite(value):
            return False
        if not np.isfinite(best):
            return True
        return float(value) > float(best) + self.delta

    # -- the counter -------------------------------------------------------- #
    @property
    def budget(self) -> float:
        """P = min(P_0 + g * counter, P_max), computed, never accumulated."""
        b = float(self.patience_0) + self.growth * float(self._counter)
        if self.max_patience is not None:
            b = min(b, float(self.max_patience))
        return b

    def update(self, improved: bool) -> bool:
        """Advance one epoch. Returns True when training should stop."""
        if improved:
            self.wait = 0
            self.n_improvement += 1
            if self.reset_on_improvement:
                self._counter = 0
        else:
            self.wait += 1
            self.n_plateau += 1
            self._counter += 1
        self.history.append((self.wait, self.budget))
        return self.should_stop

    @property
    def should_stop(self) -> bool:
        # _TOL: wait is an exact integer but budget carries one rounding of
        # g * counter, so an exact tie must not be lost to representation error.
        return float(self.wait) >= self.budget - _TOL

    def remaining(self) -> float:
        """Plateau epochs still survivable from the CURRENT state."""
        return _smallest_plateau_count(
            patience=self.patience_0,
            growth=self.growth,
            max_patience=self.max_patience,
            wait=self.wait,
            counter=self._counter,
        )

    def state_dict(self) -> Dict[str, float]:
        """Everything needed to resume mid-run from a checkpoint."""
        return {
            "wait": int(self.wait),
            "counter": int(self._counter),
            "budget": float(self.budget),          # derived; for logs only
            "n_plateau": int(self.n_plateau),
            "n_improvement": int(self.n_improvement),
            "delta": (None if self.delta is None else float(self.delta)),
            "patience_0": int(self.patience_0),
            "growth": float(self.growth),
            "max_patience": self.max_patience,
            "reset_on_improvement": bool(self.reset_on_improvement),
        }

    def load_state_dict(self, state: Dict[str, float]) -> None:
        self.wait = int(state["wait"])
        if "counter" in state:
            self._counter = int(state["counter"])
        else:
            # checkpoint written before the budget became derived: invert
            # P = P_0 + g * counter. g = 0 means the counter is irrelevant.
            if self.growth > 0.0:
                self._counter = int(round(
                    (float(state["budget"]) - self.patience_0) / self.growth))
            else:
                self._counter = 0
        self.n_plateau = int(state.get("n_plateau", 0))
        self.n_improvement = int(state.get("n_improvement", 0))
        d = state.get("delta", None)
        self.delta = None if d is None else float(d)

    def __repr__(self) -> str:
        return ("AdaptivePatience(wait=%d, budget=%.3f, P_0=%d, g=%.3g, "
                "P_max=%r, delta=%r)"
                % (self.wait, self.budget, self.patience_0, self.growth,
                   self.max_patience, self.delta))
