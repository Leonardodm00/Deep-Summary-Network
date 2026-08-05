"""
objective_utils.py
==================

Two search-objective utilities, both pure functions of things already available
to search.py. No training, no model, no plotting, no I/O.

  (A) adaptive tie-break weight epsilon, DERIVED from the evaluation set rather
      than hard-coded, so the guarantee "the secondary metric can only reorder
      configurations that the primary metric cannot separate" holds for
      whatever eval set the run actually uses;

  (B) explicit control of the Bayesian-optimization budget split between the
      random initial design and the surrogate-guided trials.

Notation (symbols introduced at first use; carried in full)
-----------------------------------------------------------
    N_eval        : number of evaluation windows, N_eval >= 2
    y             : true label vector, y = (y_1, ..., y_{N_eval}),
                    y_i in {0, ..., C-1}
    C             : number of distinct labels present in y, C >= 2
    ARI(y, y')    : adjusted Rand index between the true labels y and a
                    candidate labelling y', ARI in [-0.5, 1], ARI(y, y) = 1
    Delta_min     : the RESOLUTION of the primary metric on this evaluation
                    set: the smallest strictly-positive gap below 1 that ARI
                    can take, i.e.
                        Delta_min(y) = 1 - max { ARI(y, y') : y' in R_1(y) },
                    where R_1(y) is the set of labellings obtained from y by
                    reassigning EXACTLY ONE element to a different existing
                    label. Single-element reassignment is the minimal
                    perturbation of a partition, so this is the finest
                    distinction the evaluation set can register at all.
    Sil           : mean silhouette (cosine, against true labels), the
                    secondary metric. Sil in [-1, 1] in general.
    [s_lo, s_hi]  : the range the secondary metric is assumed to occupy,
                    s_lo < s_hi. Its width is W = s_hi - s_lo.
    gamma         : safety factor, gamma in (0, 1]. gamma = 1 saturates the
                    lexicographic condition; gamma < 1 leaves margin.
    epsilon       : tie-break weight, epsilon > 0.

    Lexicographic condition. The composite objective
        J_epsilon = - ( ARI + epsilon * Sil )
    orders configurations lexicographically (primary first, secondary only
    within a primary tie) provided
        epsilon * W  <  Delta_min(y).                                     (*)
    adaptive_epsilon returns epsilon = gamma * Delta_min(y) / W, which
    satisfies (*) strictly for every gamma in (0, 1).

Why this must be adaptive
-------------------------
Delta_min(y) shrinks as N_eval grows (one misassigned window out of many moves
ARI less than one out of few), so any hard-coded epsilon that is safe on a
small evaluation set becomes unnecessarily conservative on a larger one, and
any epsilon tuned on a large one silently VIOLATES (*) on a smaller one --
letting the secondary metric overturn a genuine primary difference. Deriving
epsilon from y removes the failure mode entirely.

HPC note (hpc-python-compat): pure ASCII. Imports only numpy and scikit-learn.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import adjusted_rand_score

__all__ = [
    "min_ari_gap",
    "adaptive_epsilon",
    "tie_break_applicable",
    "composite_objective",
    "selected_epoch_index",
    "selected_epoch_scores",
    "primary_secondary_scores",
    "resolve_n_initial_points",
]


# --------------------------------------------------------------------------- #
# (A) adaptive tie-break weight
# --------------------------------------------------------------------------- #
def min_ari_gap(y: Sequence[int]) -> Dict[str, float]:
    """Delta_min(y): the ARI resolution of this evaluation set.

    Computes, exactly, the largest ARI strictly below 1 that is reachable by
    reassigning exactly one element of y to a different existing label, and
    returns the gap 1 - that value.

    Cost is O(N_eval * C) ARI evaluations. For the sizes involved in model
    selection (tens to a few thousand windows) this is milliseconds, and it is
    computed ONCE per study, not per trial.

    Parameters
    ----------
    y : (N_eval,) integer labels with at least 2 distinct values.

    Returns
    -------
    dict with keys:
        delta_min      : Delta_min(y), in (0, 1.5]
        best_ari_below1: the maximizing ARI value
        n_eval         : N_eval
        n_classes      : C
    """
    y = np.asarray(y, dtype=int).ravel()
    n_eval = int(y.shape[0])
    if n_eval < 2:
        raise ValueError("need at least 2 evaluation points; got %d" % n_eval)
    labels = np.unique(y)
    n_classes = int(labels.shape[0])
    if n_classes < 2:
        raise ValueError("need at least 2 distinct labels; got %d" % n_classes)

    best = -np.inf
    for i in range(n_eval):
        original = y[i]
        for lab in labels:
            if lab == original:
                continue
            y_perturbed = y.copy()
            y_perturbed[i] = lab
            v = float(adjusted_rand_score(y, y_perturbed))
            if v > best:
                best = v
    if not np.isfinite(best):
        raise RuntimeError("no valid single-element reassignment found")
    return {
        "delta_min": float(1.0 - best),
        "best_ari_below1": float(best),
        "n_eval": n_eval,
        "n_classes": n_classes,
    }


def adaptive_epsilon(y: Sequence[int],
                     sil_lo: float = -1.0,
                     sil_hi: float = 1.0,
                     gamma: float = 0.5) -> Dict[str, float]:
    """epsilon = gamma * Delta_min(y) / (s_hi - s_lo), guaranteeing condition (*).

    Parameters
    ----------
    y      : (N_eval,) true labels of the evaluation split.
    sil_lo : s_lo, assumed lower bound of the secondary metric. Default -1.0
             (the theoretical silhouette minimum) -- the SAFE choice. Pass the
             empirically observed minimum only if you are willing to have (*)
             hold empirically rather than universally.
    sil_hi : s_hi, assumed upper bound. Default +1.0.
    gamma  : safety factor in (0, 1]. Default 0.5 leaves a factor-2 margin.

    Returns
    -------
    dict with keys: epsilon, delta_min, sil_range, gamma, n_eval, n_classes,
                    max_secondary_influence (= epsilon * sil_range, which is
                    strictly less than delta_min for gamma < 1).
    """
    if not (sil_lo < sil_hi):
        raise ValueError("require sil_lo < sil_hi; got (%r, %r)" % (sil_lo, sil_hi))
    if not (0.0 < gamma <= 1.0):
        raise ValueError("gamma must lie in (0, 1]; got %r" % (gamma,))
    info = min_ari_gap(y)
    width = float(sil_hi) - float(sil_lo)
    epsilon = float(gamma) * info["delta_min"] / width
    return {
        "epsilon": float(epsilon),
        "delta_min": info["delta_min"],
        "sil_range": width,
        "gamma": float(gamma),
        "n_eval": info["n_eval"],
        "n_classes": info["n_classes"],
        "max_secondary_influence": float(epsilon * width),
    }


def tie_break_applicable(selection_primary="ari", gamma=0.0):
    """[C3] Whether the Eq. (4) tie-break can be FORMED for this configuration.

    Pure policy, deliberately separated from resolve_tie_break_epsilon in
    search.py: that module imports train, hence torch and skopt, so anything
    living there cannot be tested without the full environment. The decision
    itself needs neither, so it belongs here beside adaptive_epsilon.

    Returns
    -------
    (applicable, reason) where reason is "" when applicable is True, and
    otherwise names WHICH condition failed:
        "gamma <= 0"          the tie-break is switched off by configuration;
        "continuous primary"  epsilon = gamma * Delta_min(y) / (s_hi - s_lo) is
                              premised on a primary with a smallest expressible
                              gap. Delta_min(y) exists for ARI because ARI on a
                              fixed y of N_eval points takes finitely many
                              values. The mean silhouette is continuous in Z, so
                              it has no Delta_min, exact ties between trials have
                              probability zero, and a weight whose entire
                              justification is "it acts only inside an exact tie"
                              has nothing to act on. Swapping the two halves of
                              Eq. (4) would preserve its form and discard its
                              premise, so the tie-break is disabled instead and
                              the search minimises the primary alone.

    The gamma test is applied FIRST, so a study that had already set
    tie_break_gamma = 0 reports the reason it actually chose rather than being
    told about a premise it never relied on.
    """
    g = float(gamma)
    if (not np.isfinite(g)) or g <= 0.0:
        return False, "gamma <= 0"
    if selection_primary == "ari":
        return True, ""
    if selection_primary == "silhouette":
        return False, "continuous primary"
    raise ValueError("selection_primary must be 'ari' or 'silhouette'; got %r"
                     % (selection_primary,))


def composite_objective(ari: float, silhouette: float, epsilon: float) -> float:
    """J_epsilon = -(ARI + epsilon * Sil), the value gp_minimize MINIMIZES.

    A non-finite ARI is treated as -inf (a degenerate embedding must never win
    selection), matching the existing NaN convention in train.py. A non-finite
    silhouette contributes 0 rather than poisoning an otherwise valid ARI.
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be > 0; got %r" % (epsilon,))
    a = float(ari)
    if not np.isfinite(a):
        return float("inf")            # -(-inf) -> worst possible objective
    s = float(silhouette)
    if not np.isfinite(s):
        s = 0.0
    return float(-(a + float(epsilon) * s))


# --------------------------------------------------------------------------- #
# (A2) epoch selection: the epoch the search must READ
# --------------------------------------------------------------------------- #
def _finite_or_neg_inf(x):
    """NaN-safe metric read. Mirrors train._finite_or_neg_inf EXACTLY: a
    non-finite metric (degenerate embedding) becomes -inf, so it can never win a
    lexicographic argmax."""
    v = float(x)
    return v if np.isfinite(v) else float("-inf")


def selected_epoch_index(history, selection_primary="ari"):
    """[C2] i*, the INDEX INTO history of the epoch e*(t, sigma) that train() itself
    selected, or None if no epoch produced a finite selection metric.

    This RECOMPUTES train.py's rule rather than reading it back, because train()
    returns only (model, history) and not best_epoch. The rule is mirrored
    line-for-line from train.py's epoch loop (decision 17):

        u_best = v_best = -inf
        for each epoch e, with (u_e, v_e) the (primary, secondary) metrics:
            if (u_e, v_e) > (u_best, v_best):   e* <- e     [lexicographic argmax]
            u_best <- max(u_best, u_e)                      [component-wise ...]
            v_best <- max(v_best, v_e)                      [... running maxima]

    Two properties of that rule are load-bearing and easy to get wrong:
      * the comparison is against the COMPONENT-WISE running maxima (u*, v*),
        which need not be the pair observed at any single epoch;
      * it is a strict >, so the FIRST epoch attaining a tied pair wins.

    DRIFT WARNING. If train.py's rule ever changes, this function silently
    disagrees with it. Main/Smoke_Tests/smoke_test_selected_epoch.py converts
    that risk into a test failure by asserting equality against the best_epoch
    train() itself wrote into its checkpoint on a real toy run.

    Parameters
    ----------
    history           : list of per-epoch dicts, each with "ari" and "silhouette"
                        (train.py writes both at every epoch).
    selection_primary : "ari" (u = ARI, v = Sil) or "silhouette" (u = Sil, v = ARI).

    Returns
    -------
    i* : int index into history, or None if every epoch was non-finite.
    """
    if selection_primary == "ari":
        u_key, v_key = "ari", "silhouette"
    elif selection_primary == "silhouette":
        u_key, v_key = "silhouette", "ari"
    else:
        raise ValueError("selection_primary must be 'ari' or 'silhouette'; got %r"
                         % (selection_primary,))

    u_best = v_best = float("-inf")
    i_star = None
    for i, h in enumerate(history):
        u_e = _finite_or_neg_inf(h[u_key])
        v_e = _finite_or_neg_inf(h[v_key])
        if (u_e, v_e) > (u_best, v_best):
            i_star = i
        u_best = max(u_best, u_e)
        v_best = max(v_best, v_e)
    return i_star


def selected_epoch_scores(history, selection_primary="ari"):
    """[C2] (ARI, Sil) read at the SAME selected epoch e*, plus e* itself.

    This is the change C2 makes to what the search reads. The old rule took an
    independent max over epochs of the primary signal alone; two signals each
    maximized over its own epoch describe a model that never existed. Reading
    both at e* describes the model the run would actually hand on, because e* is
    the epoch whose weights train() restores.

    Returns
    -------
    (ari, sil, epoch) where
        ari   : float, -inf if no epoch was finite (caller maps to FAILED)
        sil   : float, NaN if no epoch was finite or the epoch's own Sil is
                non-finite (composite_objective treats a non-finite secondary
                as contributing 0 rather than poisoning a valid primary)
        epoch : int, the "epoch" field at e*, or 0 for "no epoch selected"
                (0 is train.py's own sentinel: its epochs are 1-based)
    """
    i_star = selected_epoch_index(history, selection_primary)
    if i_star is None:
        return float("-inf"), float("nan"), 0
    h = history[i_star]
    ari = _finite_or_neg_inf(h["ari"])
    sil = float(h["silhouette"])
    return ari, sil, int(h.get("epoch", i_star + 1))


def primary_secondary_scores(history, selection_primary="ari"):
    """[C3] (u, v, ari, sil, epoch) at e*, with (u, v) ordered by ROLE.

    selected_epoch_scores returns the pair ordered by NAME -- ARI first, always.
    That is the right contract for a reader that wants a named metric, and the
    WRONG one for the search, which needs whichever metric cfg.train.
    selection_primary designates as primary. Keeping both orderings available,
    from one function, is what stops the two from drifting apart.

    Returns
    -------
    (u, v, ari, sil, epoch) where
        u     : float, the PRIMARY at e*. Carries the -inf convention of
                _finite_or_neg_inf, so a degenerate embedding can never win a
                comparison and composite_objective maps it to the worst
                attainable objective.
        v     : float, the SECONDARY at e*. May be non-finite;
                composite_objective treats a non-finite secondary as
                contributing 0 rather than poisoning a valid primary.
        ari   : float, ARI at e*, ALWAYS, whatever the roles are.
        sil   : float, silhouette at e*, ALWAYS, whatever the roles are.
        epoch : int, the "epoch" field at e*, or 0 for "no epoch selected".

    ari and sil are returned alongside (u, v) so that a caller can record both
    metrics under their own names and never has to infer which one a role-keyed
    field holds.
    """
    ari, sil, epoch = selected_epoch_scores(history, selection_primary)
    if selection_primary == "ari":
        return ari, sil, ari, sil, epoch
    if selection_primary == "silhouette":
        # The primary must carry the -inf convention; selected_epoch_scores
        # applies it to ARI only, because ARI is the primary under the default.
        return _finite_or_neg_inf(sil), ari, ari, sil, epoch
    raise ValueError("selection_primary must be 'ari' or 'silhouette'; got %r"
                     % (selection_primary,))


# --------------------------------------------------------------------------- #
# (B) explicit budget split
# --------------------------------------------------------------------------- #
def resolve_n_initial_points(n_calls: int,
                             n_initial_points: Optional[int] = None) -> int:
    """Number of random initial-design trials, n_init, with 1 <= n_init <= n_calls.

    The current hard-coded rule in search.py is
        n_init = min(10, max(1, n_calls // 2)),
    which cannot be overridden. That matters: n_init trials are drawn WITHOUT
    the surrogate, so they set the floor on how much of the budget is pure
    random search. With n_calls = 50 it spends 10 trials randomly; with
    n_calls = 15 it spends 7 of 15.

    Parameters
    ----------
    n_calls          : total trials for this phase, n_calls >= 1.
    n_initial_points : requested n_init. None (or <= 0) reproduces the legacy
                       rule exactly, so existing configs are unaffected.

    Returns
    -------
    n_init : int in [1, n_calls].

    Raises
    ------
    ValueError if n_calls < 1, or if n_initial_points exceeds n_calls (which
    would leave the surrogate no trials at all and silently degrade the study
    to pure random search).
    """
    n_calls = int(n_calls)
    if n_calls < 1:
        raise ValueError("n_calls must be >= 1; got %d" % n_calls)
    if n_initial_points is None or int(n_initial_points) <= 0:
        return int(min(10, max(1, n_calls // 2)))
    n_init = int(n_initial_points)
    if n_init > n_calls or (n_init == n_calls and n_calls > 1):
        raise ValueError(
            "n_initial_points (%d) is not less than n_calls (%d): the surrogate "
            "would never be fitted and the study would be pure random search. "
            "The guard is >=, not >, because EQUALITY is the dangerous case: it "
            "arises exactly when a study of n_calls trials is split into lanes "
            "or segments of n_initial_points each, which looks like a search and "
            "is not one. A resumed segment must NOT reach this function -- it "
            "calls search_persistence.resolve_resume_budget instead, which is "
            "allowed to return 0 because the initial design was already paid "
            "for in an earlier segment." % (n_init, n_calls))
    return n_init
