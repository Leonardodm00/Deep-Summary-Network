"""[C4] Batch geometry for cross-culture positives: Eq. (2), Eq. (3), the caps.

WHAT THIS MODULE ESTABLISHES
----------------------------
Given the culture membership of the training windows and the requested batch
parameters, this module decides:

  * U_eff, the number of DISTINCT cultures of each class that a batch can
    actually draw, after the availability clamp of Eq. (3);
  * whether the requested (U_c, q, N_s) are admissible under the two caps;
  * the resulting group size n_g and batch row count M.

It raises on every inadmissible combination rather than silently repairing it,
because D1-section 9 established that inconsistent config values in this
codebase fail quietly.

WHY IT IS A SEPARATE MODULE
---------------------------
This is arithmetic over label and culture arrays. It touches no tensors, no
model and no data loading, so it is kept apart from data_pipeline (which owns
the torch Sampler) and from config (which owns the schema). The practical
consequence is that the whole of it is exercisable with numpy alone, which
matters because Change 4 is the change most able to go quietly wrong.

THE TWO CAPS, WHICH ARE NOT THE SAME CAP
-----------------------------------------
1. DEGENERACY CAP, on q. The defect Change 4 removes is a positive that is a
   near-copy of the anchor: an easy-positive miner selects it, the positive term
   saturates, and the network learns warp invariance rather than phenotype
   similarity. Cross-culture positives remove the original carrier (warps), but
   q > 1 puts several windows of the SAME culture in one group -- same
   preparation, same drift, and literally overlapping in time whenever
   train_stride_s < window_s. exclude_same_culture_positives forbids those PAIRS,
   but it is a toggle: a later config edit turning it off with a large q
   reinstates the defect silently. The cap makes that combination impossible.
   It is tied to the data as q <= max(1, floor(f * W_min)), W_min being the
   fewest windows in any training culture, because a 13-window culture can spare
   several and a 3-window culture cannot give 3 without handing one batch its
   entire timeline.

2. RESOURCE CAP, on M = C * U_eff * q * (1 + N_s), the rows the backbone embeds
   per batch. This is what memory and step time actually scale with. It is also
   where the cited source's mechanism really lived: Xuan, Stylianou and Pless
   fixed the batch at 128 and filled it by adding classes, so a larger group
   FORCED OUT classes, the negatives lost variation, and performance fell past a
   group size of 16. Here C is fixed and every batch is class-complete, so a
   larger group displaces nothing and simply enlarges the batch; stating the
   constraint on M restores the coupling their experiment had.

MINER GATING
------------
The group-size ceiling of 16 traces to an easy-positive result, and the same
experiments show hard-positive mining behaving the OPPOSITE way -- performance
there falls with group size well before 16. So:

  * "easy_positive" and "easy_pos_semihard_neg": n_g <= max_group_size (16),
    the degeneracy cap on q applies, and exclude_same_culture_positives MUST be
    True, since the easy-positive miner is precisely what would seize on a
    same-culture near-duplicate.
  * "hard": neither the group-size ceiling nor the q cap applies. Only the
    resource bound on M does, and it may permit a larger group. Memory does not
    care how you mine.

HPC note (hpc-python-compat): pure ASCII, LF endings, no non-ASCII anywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "EASY_POSITIVE_STRATEGIES",
    "BatchGeometry",
    "culture_census",
    "resolve_cultures_per_class",
    "resolve_group_size_cap",
    "resolve_q_cap",
    "resolve_batch_geometry",
]

# The strategies whose published behaviour justifies the group-size ceiling and
# the degeneracy cap. "hard" is deliberately absent; see MINER GATING above.
EASY_POSITIVE_STRATEGIES = ("easy_positive", "easy_pos_semihard_neg")

# Default fraction of a culture's windows that one batch may consume, used by
# the degeneracy cap on q. Chosen, not derived: at f = 0.5 a 13-window culture
# yields q <= 6, which leaves at least half its timeline out of any single batch.
DEFAULT_Q_CAP_FRACTION = 0.5


@dataclass(frozen=True)
class BatchGeometry:
    """The resolved geometry of one batch, and the record of how it was resolved.

    Attributes
    ----------
    active            : False when positives_mode != "cross_culture", in which
                        case every other field is a placeholder and nothing was
                        checked. The change is inert by default (D1-R12).
    n_classes         : C.
    cultures_requested: U_c as configured, BEFORE Eq. (3).
    cultures_effective: U_eff, AFTER Eq. (3).
    cultures_available: min over classes of the distinct training cultures.
    clamped           : True when Eq. (3) reduced U_c. Log this once.
    windows_per_culture_per_batch : q.
    n_surrogates      : N_s.
    group_size        : n_g = U_eff * q, rows sharing a class label per class
                        per batch. NOT M and NOT the mined-triplet count.
    batch_rows        : M = C * U_eff * q * (1 + N_s).
    group_size_cap    : the ceiling actually applied to n_g, or None if none was.
    group_size_cap_reason : which rule produced it.
    q_cap             : the ceiling applied to q, or None.
    min_windows_per_culture : W_min.
    notes             : lines intended for a single log at construction.
    """

    active: bool
    n_classes: int
    cultures_requested: int
    cultures_effective: int
    cultures_available: int
    clamped: bool
    windows_per_culture_per_batch: int
    n_surrogates: int
    group_size: int
    batch_rows: int
    group_size_cap: Optional[int]
    group_size_cap_reason: str
    q_cap: Optional[int]
    min_windows_per_culture: int
    notes: List[str] = field(default_factory=list)


def culture_census(trace_of_window, conditions):
    """Distinct cultures per class, and windows per culture, from the two arrays.

    Parameters
    ----------
    trace_of_window : int array g of length N, g[i] the GLOBAL culture index of
                      training window i (data_splits provides this).
    conditions      : int array y of length N, y[i] the class of window i.

    Returns
    -------
    dict with
        "cultures_per_class"  : {class c -> number of distinct cultures}
        "windows_per_culture" : {culture u -> number of windows}
        "class_of_culture"    : {culture u -> its class}

    Raises
    ------
    ValueError if the two arrays disagree in length, if either is empty, or if
    any culture carries windows of more than one class. The last is not a
    pedantic check: a culture spanning classes would make "cultures of class c"
    ill-defined and would quietly corrupt Eq. (3).
    """
    g = np.asarray(trace_of_window).ravel()
    y = np.asarray(conditions).ravel()
    if g.size == 0 or y.size == 0:
        raise ValueError("trace_of_window and conditions must be non-empty")
    if g.size != y.size:
        raise ValueError(
            "trace_of_window has %d entries but conditions has %d; they must be "
            "parallel arrays over the SAME windows" % (g.size, y.size))

    class_of_culture: Dict[int, int] = {}
    windows_per_culture: Dict[int, int] = {}
    cultures_by_class: Dict[int, set] = {}

    for u_raw, c_raw in zip(g.tolist(), y.tolist()):
        u = int(u_raw)
        c = int(c_raw)
        seen = class_of_culture.get(u)
        if seen is None:
            class_of_culture[u] = c
        elif seen != c:
            raise ValueError(
                "culture %d carries windows of class %d and of class %d: a "
                "culture must belong to exactly one class, or 'cultures of "
                "class c' in Eq. (3) is not well defined" % (u, seen, c))
        windows_per_culture[u] = windows_per_culture.get(u, 0) + 1
        cultures_by_class.setdefault(c, set()).add(u)

    return {
        "cultures_per_class": {c: len(us) for c, us in cultures_by_class.items()},
        "windows_per_culture": windows_per_culture,
        "class_of_culture": class_of_culture,
    }


def resolve_cultures_per_class(requested, cultures_per_class,
                               min_train_cultures_per_class=2):
    """Eq. (3): U_eff = min(U_c, min_c |{cultures of class c}|).

    Returns (U_eff, info). info["clamped"] is True when the request was reduced,
    which the caller is expected to log ONCE at construction.

    Raises
    ------
    ValueError when the availability bound is below min_train_cultures_per_class.
    At a bound of 1 a class has a single training culture and yields NO
    cross-culture positive at all, so the mode is unsatisfiable rather than
    merely degraded; failing is the only honest outcome.
    """
    if int(requested) < 1:
        raise ValueError("cultures_per_class_per_batch (U_c) must be >= 1")
    if not cultures_per_class:
        raise ValueError("cultures_per_class is empty: no classes were found")

    bound = min(int(v) for v in cultures_per_class.values())
    starved = sorted(c for c, v in cultures_per_class.items() if int(v) == bound)
    if bound < int(min_train_cultures_per_class):
        raise ValueError(
            "class(es) %s have only %d training culture(s), below "
            "min_train_cultures_per_class = %d. positives_mode='cross_culture' "
            "needs at least 2 cultures per class, since with one culture every "
            "same-class window shares the anchor's culture and no cross-culture "
            "positive exists."
            % (starved, bound, int(min_train_cultures_per_class)))

    u_eff = min(int(requested), bound)
    info = {
        "requested": int(requested),
        "available": int(bound),
        "effective": int(u_eff),
        "clamped": bool(u_eff < int(requested)),
        "binding_classes": starved,
    }
    return u_eff, info


def resolve_group_size_cap(mining_strategy, max_group_size, n_classes,
                           n_surrogates, max_batch_rows=None):
    """The ceiling on n_g, or None when nothing constrains it.

    Two independent sources, combined by taking the tighter:

      * the group-size ceiling, applied ONLY under the easy-positive strategies,
        because that is the only setting the cited threshold speaks to;
      * the resource bound floor(M_max / (C * (1 + N_s))), applied under EVERY
        strategy when max_batch_rows is given, because memory is indifferent to
        the mining rule.

    Returns (cap or None, reason).
    """
    caps = []
    reasons = []

    if mining_strategy in EASY_POSITIVE_STRATEGIES:
        caps.append(int(max_group_size))
        reasons.append("easy-positive ceiling %d" % int(max_group_size))

    if max_batch_rows is not None:
        denom = int(n_classes) * (1 + int(n_surrogates))
        if denom < 1:
            raise ValueError("n_classes and (1 + n_surrogates) must be >= 1")
        resource = int(math.floor(int(max_batch_rows) / float(denom)))
        if resource < 1:
            raise ValueError(
                "max_batch_rows = %d cannot fit even one window per class at "
                "C = %d and N_s = %d (needs at least %d rows)"
                % (int(max_batch_rows), int(n_classes), int(n_surrogates), denom))
        caps.append(resource)
        reasons.append("resource bound floor(%d / (%d * %d)) = %d"
                       % (int(max_batch_rows), int(n_classes),
                          1 + int(n_surrogates), resource))

    if not caps:
        return None, ("no cap: mining_strategy=%r is not an easy-positive "
                      "strategy and no max_batch_rows was given"
                      % (mining_strategy,))
    cap = min(caps)
    return cap, "min(%s) = %d" % (", ".join(reasons), cap)


def resolve_q_cap(mining_strategy, min_windows_per_culture,
                  fraction=DEFAULT_Q_CAP_FRACTION):
    """The degeneracy ceiling on q, or None under a non-easy-positive strategy.

    q <= max(1, floor(fraction * W_min)). Returns (cap or None, reason).
    """
    if mining_strategy not in EASY_POSITIVE_STRATEGIES:
        return None, ("no q cap: the degeneracy argument is an easy-positive "
                      "one and mining_strategy=%r is not" % (mining_strategy,))
    if float(fraction) <= 0.0 or float(fraction) > 1.0:
        raise ValueError("q_cap_fraction must lie in (0, 1]")
    w_min = int(min_windows_per_culture)
    if w_min < 1:
        raise ValueError("min_windows_per_culture must be >= 1")
    cap = max(1, int(math.floor(float(fraction) * w_min)))
    return cap, ("degeneracy cap max(1, floor(%.3g * W_min=%d)) = %d"
                 % (float(fraction), w_min, cap))


def resolve_batch_geometry(trace_of_window,
                           conditions,
                           positives_mode,
                           mining_strategy,
                           cultures_per_class_per_batch,
                           windows_per_culture_per_batch,
                           n_surrogates,
                           max_group_size,
                           exclude_same_culture_positives,
                           min_train_cultures_per_class=2,
                           max_batch_rows=None,
                           q_cap_fraction=DEFAULT_Q_CAP_FRACTION):
    """Resolve and CHECK the whole geometry. The single entry point.

    Returns a BatchGeometry. Under positives_mode == "augmentation" it returns an
    inactive one and checks nothing, so the change stays inert until switched on.

    Raises ValueError on any inadmissible combination, naming the quantities.
    """
    mode = str(positives_mode)
    if mode not in ("augmentation", "cross_culture"):
        raise ValueError(
            "positives_mode must be 'augmentation' or 'cross_culture'; got %r"
            % (positives_mode,))

    if mode == "augmentation":
        return BatchGeometry(
            active=False, n_classes=0, cultures_requested=0,
            cultures_effective=0, cultures_available=0, clamped=False,
            windows_per_culture_per_batch=0, n_surrogates=0, group_size=0,
            batch_rows=0, group_size_cap=None,
            group_size_cap_reason="inactive: positives_mode='augmentation'",
            q_cap=None, min_windows_per_culture=0,
            notes=["positives_mode='augmentation': cross-culture geometry "
                   "inactive, nothing checked"])

    q = int(windows_per_culture_per_batch)
    n_s = int(n_surrogates)
    if q < 1:
        raise ValueError("windows_per_culture_per_batch (q) must be >= 1")
    if n_s < 0:
        raise ValueError("n_surrogates (N_s) must be >= 0")

    # --- the easy-positive precondition ---------------------------------------
    # Checked before anything else: under an easy-positive miner a same-culture
    # near-duplicate is exactly what the miner will select, so allowing such
    # pairs reinstates the defect Change 4 exists to remove.
    if (mining_strategy in EASY_POSITIVE_STRATEGIES
            and not bool(exclude_same_culture_positives)):
        raise ValueError(
            "mining_strategy=%r requires exclude_same_culture_positives=True. "
            "An easy-positive miner selects the MOST similar same-class row; "
            "with same-culture pairs permitted that row is a window of the "
            "anchor's own culture, overlapping it in time whenever the train "
            "stride is below the window length, and the positive term saturates "
            "exactly as it does with warped copies." % (mining_strategy,))

    census = culture_census(trace_of_window, conditions)
    cultures_per_class = census["cultures_per_class"]
    windows_per_culture = census["windows_per_culture"]
    n_classes = len(cultures_per_class)
    w_min = min(int(v) for v in windows_per_culture.values())

    u_eff, clamp_info = resolve_cultures_per_class(
        cultures_per_class_per_batch, cultures_per_class,
        min_train_cultures_per_class=min_train_cultures_per_class)

    notes = []
    if clamp_info["clamped"]:
        notes.append(
            "Eq. (3) clamp: U_c = %d requested, %d available (binding "
            "class(es) %s), U_eff = %d"
            % (clamp_info["requested"], clamp_info["available"],
               clamp_info["binding_classes"], u_eff))

    # --- q must be drawable without replacement -------------------------------
    # Not in the handoff's assertion list. Without it, a culture with fewer than
    # q windows forces sampling WITH replacement, which puts identical rows back
    # into the group by a different route than the one Change 4 closes.
    if q > w_min:
        raise ValueError(
            "windows_per_culture_per_batch (q) = %d exceeds the %d window(s) "
            "held by the smallest training culture, so windows would have to be "
            "drawn WITH replacement and the group would contain identical rows."
            % (q, w_min))

    q_cap, q_reason = resolve_q_cap(mining_strategy, w_min, q_cap_fraction)
    if q_cap is not None and q > q_cap:
        raise ValueError(
            "windows_per_culture_per_batch (q) = %d exceeds the degeneracy cap "
            "%d (%s). Windows of one culture share preparation and drift and "
            "overlap in time; under mining_strategy=%r they are what the miner "
            "would select as positives."
            % (q, q_cap, q_reason, mining_strategy))

    n_g = u_eff * q
    m_rows = n_classes * u_eff * q * (1 + n_s)

    cap, cap_reason = resolve_group_size_cap(
        mining_strategy, max_group_size, n_classes, n_s, max_batch_rows)
    if cap is not None and n_g > cap:
        raise ValueError(
            "group size n_g = U_eff * q = %d * %d = %d exceeds the cap %d (%s). "
            "n_g counts rows sharing a class label, per class, per batch -- not "
            "M = %d and not the mined-triplet count."
            % (u_eff, q, n_g, cap, cap_reason, m_rows))

    notes.append("n_g = U_eff * q = %d * %d = %d (cap: %s)"
                 % (u_eff, q, n_g, cap_reason))
    notes.append("M = C * U_eff * q * (1 + N_s) = %d * %d * %d * %d = %d"
                 % (n_classes, u_eff, q, 1 + n_s, m_rows))
    notes.append("q cap: %s" % q_reason)

    return BatchGeometry(
        active=True,
        n_classes=n_classes,
        cultures_requested=int(cultures_per_class_per_batch),
        cultures_effective=int(u_eff),
        cultures_available=int(clamp_info["available"]),
        clamped=bool(clamp_info["clamped"]),
        windows_per_culture_per_batch=q,
        n_surrogates=n_s,
        group_size=int(n_g),
        batch_rows=int(m_rows),
        group_size_cap=cap,
        group_size_cap_reason=cap_reason,
        q_cap=q_cap,
        min_windows_per_culture=int(w_min),
        notes=notes,
    )
