"""
data_splits.py
==============

Two responsibilities, both decoupled from the model / trainer / plotting
(directive 2):

  1. C-class synthetic trace generation (MultiClassSyntheticProvider): a
     generalization of data_pipeline.SyntheticTraceProvider, which is hard-coded
     to two conditions (CONTROL / PATHO). This lets HPC dry-runs and smoke tests
     exercise the full C >= 2 phenotype path with labels 0..C-1.

  2. Boundary-safe TIME-SEGMENT train/val/test splitting (option A). Each full
     trace's time axis is cut into three CONTIGUOUS, DISJOINT segments by
     fraction (default 60/20/20). Windows are then formed WITHIN each segment by
     the already-tested data_pipeline.MEAWindowDataset (directive 1). Because
     windowing happens inside a segment, no window can straddle a split boundary
     and no sample can appear in two splits -- the leakage guarantee holds by
     construction. Train windows overlap (train_stride < window); val / test
     windows are disjoint (eval_stride >= window), which fixes the legacy
     low-diversity / duplicated-eval-rows problem at the source.

Notation
--------
    L                : full trace length in samples
    (f_tr, f_va, f_te): split fractions along time, f_tr + f_va + f_te = 1
    segment k of a trace : half-open sample interval [s_k, e_k), for
                           k in {train, val, test}, with
                           s_train = 0,
                           e_train = floor(f_tr * L)          = s_val,
                           e_val   = floor((f_tr + f_va) * L) = s_test,
                           e_test  = L.
    W                : window length in samples = round(window_s * fs)
    stride_split     : train_stride (train) or eval_stride (val / test), samples
    window starts within a segment of length Lk:
                       { j*stride : j = 0,1,...  and  j*stride + W <= Lk }
                       (this mirrors MEAWindowDataset's own tiling rule exactly)

HPC note (hpc-python-compat): pure ASCII. Import chain (data_pipeline.py,
augmentation.py) is pure ASCII as well.
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from data_pipeline import MEAWindowDataset
from config import DataConfig

__all__ = [
    "MultiClassSyntheticProvider",
    "make_synthetic_specs",
    "segment_bounds",
    "window_starts",
    "SplitBundle",
    "make_time_segment_splits",
    "apportion",
    "assign_cultures",
    "make_trace_splits",
]

_SPLIT_NAMES = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# C-class synthetic trace provider (generalizes SyntheticTraceProvider)
# --------------------------------------------------------------------------- #
class MultiClassSyntheticProvider:
    """Burst-like, non-negative synthetic traces for C phenotype classes (NOT
    biologically faithful; for pipeline validation / dry-runs only).

    Class 0 is the regular baseline (matches the original CONTROL: fixed burst
    width, unit amplitude). Higher class indices are progressively denser
    (higher burst rate) and more irregular (jittered width and amplitude), so the
    C classes are separable in debug clustering. For C == 2 this reduces to
    roughly the original CONTROL vs PATHO contrast.

    Call signature matches SyntheticTraceProvider: __call__(condition, trace_id).
    """

    def __init__(self, n_classes: int, duration_s: float = 600.0, fs: float = 50.0,
                 seed: int = 0, rate_min: float = 0.25, rate_max: float = 0.55,
                 width_min: float = 0.15, width_max: float = 0.70,
                 amp_jitter_min: float = 0.60, amp_jitter_max: float = 1.40,
                 per_class: Sequence = ()):
        """Parameters
        ----------
        n_classes       : number of phenotype classes C (labels 0..C-1).
        duration_s, fs  : trace duration (s) and sampling rate (Hz).
        seed            : base seed; per-trace rng seeds are derived as
                          seed + 1000*condition + trace_id (unchanged).
        rate_min/max    : global linear sweep of burst rate (bursts/s) across
                          classes; class c gets rate_min + (rate_max-rate_min)*
                          frac(c), frac(c)=c/(C-1).
        width_min/max   : global linear sweep of base burst width (s); class c
                          gets width_max - (width_max-width_min)*frac(c).
        amp_jitter_min/max : per-burst amplitude jitter bounds U(min,max) applied
                          to classes c != 0 (class 0 uses fixed a=1.0). Previously
                          hard-coded as U(0.6, 1.4); now configurable.
        per_class       : optional sequence of per-class override objects. Entry
                          c (if present) may carry .rate, .width, .amp_min,
                          .amp_max (each may be None = no override). per_class may
                          be shorter than C; missing entries fall back to the
                          global sweep. Supplying amp_min/amp_max for class 0
                          promotes it to jittered amplitude (breaks the a=1.0
                          special case for that class only).

        Backward compatibility: called with only (n_classes, duration_s, fs,
        seed) plus the original rate/width kwargs, the rng draw ORDER is
        identical to the previous implementation, so traces are byte-identical to
        prior cached runs. The amplitude-jitter draw for c != 0 uses the same
        rng.uniform call position as before (only its bounds are now
        parameterized, defaulting to the old 0.6/1.4).
        """
        if n_classes < 1:
            raise ValueError("n_classes must be >= 1")
        if duration_s <= 0 or fs <= 0:
            raise ValueError("duration_s and fs must be > 0")
        if not (0 < width_min <= width_max):
            raise ValueError("require 0 < width_min <= width_max")
        if not (0 < rate_min <= rate_max):
            raise ValueError("require 0 < rate_min <= rate_max")
        if not (0 < amp_jitter_min <= amp_jitter_max):
            raise ValueError("require 0 < amp_jitter_min <= amp_jitter_max")
        self.n_classes = int(n_classes)
        self.duration_s = float(duration_s)
        self.fs = float(fs)
        self.seed = int(seed)
        self.rate_min = float(rate_min)
        self.rate_max = float(rate_max)
        self.width_min = float(width_min)
        self.width_max = float(width_max)
        self.amp_jitter_min = float(amp_jitter_min)
        self.amp_jitter_max = float(amp_jitter_max)
        # normalize per_class into a plain list indexed by class; entries may be
        # None (no override for that class). Accept objects with attributes
        # (rate/width/amp_min/amp_max) or plain dicts.
        self.per_class = list(per_class) if per_class else []
        if len(self.per_class) > self.n_classes:
            raise ValueError(
                "per_class has %d entries but n_classes=%d"
                % (len(self.per_class), self.n_classes))

    def _override_for(self, condition: int):
        """Return (rate, width, amp_min, amp_max) overrides for a class as a
        4-tuple, each element None when not overridden. Reads either attribute-
        style objects (SyntheticClassOverride) or dicts."""
        if condition >= len(self.per_class):
            return (None, None, None, None)
        o = self.per_class[condition]
        if o is None:
            return (None, None, None, None)
        if isinstance(o, dict):
            return (o.get("rate"), o.get("width"), o.get("amp_min"), o.get("amp_max"))
        return (getattr(o, "rate", None), getattr(o, "width", None),
                getattr(o, "amp_min", None), getattr(o, "amp_max", None))

    def _class_fraction(self, condition: int) -> float:
        if self.n_classes == 1:
            return 0.0
        if not (0 <= condition < self.n_classes):
            raise ValueError(
                "condition %d out of range [0, %d)" % (condition, self.n_classes))
        return condition / (self.n_classes - 1)

    def __call__(self, condition: int, trace_id: int) -> Tuple[np.ndarray, float]:
        condition = int(condition)
        trace_id = int(trace_id)
        frac = self._class_fraction(condition)

        ov_rate, ov_width, ov_amp_min, ov_amp_max = self._override_for(condition)

        # swept defaults, then apply per-class overrides where present
        rate = self.rate_min + (self.rate_max - self.rate_min) * frac          # bursts / s
        base_width = self.width_max - (self.width_max - self.width_min) * frac  # seconds
        if ov_rate is not None:
            rate = float(ov_rate)
        if ov_width is not None:
            base_width = float(ov_width)

        amp_min = self.amp_jitter_min if ov_amp_min is None else float(ov_amp_min)
        amp_max = self.amp_jitter_max if ov_amp_max is None else float(ov_amp_max)
        # class 0 is fixed-amplitude (a=1.0) UNLESS an explicit amp override is
        # given for class 0, which promotes it to jittered like the other classes
        class0_amp_overridden = (condition == 0 and ov_amp_min is not None)

        rng = np.random.default_rng(self.seed + 1000 * condition + trace_id)
        T = int(self.duration_s * self.fs)
        t = np.arange(T) / self.fs
        x = np.zeros(T, dtype=np.float64)

        n_bursts = max(1, int(rng.poisson(rate * self.duration_s)))
        centers = rng.uniform(0.0, self.duration_s, n_bursts)
        for c in centers:
            if condition == 0 and not class0_amp_overridden:
                w = base_width                        # regular baseline
                a = 1.0
            else:
                w = float(rng.uniform(0.5 * base_width, 1.5 * base_width))  # irregular
                a = float(rng.uniform(amp_min, amp_max))
            x += a * np.exp(-0.5 * ((t - c) / w) ** 2)
        return x.astype(np.float32), self.fs


def make_synthetic_specs(n_per_class: Sequence[int]) -> List[dict]:
    """Build cache specs for a C-class synthetic dataset.

    n_per_class[c] = number of synthetic traces for class c (labels 0..C-1).
    Returns a list of dicts compatible with preprocessing_cache.cache_traces,
    where each provider call is provider(condition, trace_id).
    """
    if len(n_per_class) < 1:
        raise ValueError("n_per_class must list at least one class")
    specs = []
    for condition, n in enumerate(n_per_class):
        if int(n) < 1:
            raise ValueError("class %d has n=%d; need >= 1" % (condition, n))
        for trace_id in range(int(n)):
            specs.append({
                "name": "synthetic_c%d_t%d" % (condition, trace_id),
                "condition": int(condition),
                "args": (int(condition), int(trace_id)),
            })
    return specs


# --------------------------------------------------------------------------- #
# Time-segment splitting helpers (single source of truth for boundaries)
# --------------------------------------------------------------------------- #
def segment_bounds(length: int, fractions: Sequence[float]) -> List[Tuple[int, int]]:
    """Half-open [start, end) sample bounds for the (train, val, test) segments.

    Uses floor at the two interior cut points so the three segments are disjoint
    and exactly cover [0, length). Returns [(0, e_tr), (e_tr, e_va), (e_va, L)].
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    if len(fractions) != 3:
        raise ValueError("fractions must be (train, val, test)")
    f_tr, f_va, f_te = (float(f) for f in fractions)
    if abs(f_tr + f_va + f_te - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1.0")
    e_tr = int(np.floor(f_tr * length))
    e_va = int(np.floor((f_tr + f_va) * length))
    # clamp to keep a valid, ordered, covering partition
    e_tr = max(0, min(e_tr, length))
    e_va = max(e_tr, min(e_va, length))
    return [(0, e_tr), (e_tr, e_va), (e_va, length)]


def window_starts(seg_length: int, window_length: int, stride: int) -> List[int]:
    """Window start offsets within a segment of length seg_length.

    Mirrors MEAWindowDataset's tiling rule EXACTLY (s = 0; while s + W <= L:
    emit s; s += stride), so the provenance computed here matches the windows the
    Dataset actually produces.
    """
    if window_length < 1 or stride < 1:
        raise ValueError("window_length and stride must be >= 1")
    starts = []
    s = 0
    while s + window_length <= seg_length:
        starts.append(s)
        s += stride
    return starts


# --------------------------------------------------------------------------- #
# Split bundle (the 3 datasets + exact per-window provenance for leakage checks)
# --------------------------------------------------------------------------- #
@dataclass
class SplitBundle:
    """Container returned by make_time_segment_splits.

    train / val / test : MEAWindowDataset instances (train has overlapping
                         windows; val / test are disjoint).
    window_length, train_stride, eval_stride : resolved sample counts.
    coverage : split_name -> list of (orig_trace_idx, orig_start, orig_end,
               condition), one entry per window IN THE SAME ORDER the Dataset
               enumerates them. orig_start / orig_end are in ORIGINAL trace
               coordinates (segment offset already added), so downstream code and
               tests can verify no sample is shared across splits.
    seg_bounds : list (per original trace) of [(s,e)_train,(s,e)_val,(s,e)_test].
                 EMPTY for a trace split (no time cut is made).

    trace_of_window : split_name -> int array g, with g[i] the GLOBAL culture
                 (trace) index u of window i, in the SAME order the Dataset
                 enumerates windows. Load-bearing in three places: the
                 culture-first batch sampler, exclude_same_culture_positives
                 masking, and trace-level silhouette. Populated by BOTH
                 splitters (for the time-segment splitter every culture appears
                 in all three splits, which is exactly what this array makes
                 visible).
    cultures   : split_name -> sorted int array of the GLOBAL culture indices
                 assigned to that split. For a trace split the three arrays are
                 pairwise disjoint; for a time-segment split they are identical.
    split_kind : "time_segment" or "trace", so a consumer can tell which
                 leakage guarantee it is holding.
    fold       : the leave-one-out fold index, or None.
    """
    train: MEAWindowDataset
    val: MEAWindowDataset
    test: MEAWindowDataset
    window_length: int
    train_stride: int
    eval_stride: int
    coverage: Dict[str, List[Tuple[int, int, int, int]]]
    seg_bounds: List[List[Tuple[int, int]]]
    trace_of_window: Dict[str, np.ndarray] = field(default_factory=dict)
    cultures: Dict[str, np.ndarray] = field(default_factory=dict)
    split_kind: str = "time_segment"
    fold: Optional[int] = None


def make_time_segment_splits(traces: Sequence[np.ndarray],
                             conditions: Sequence[int],
                             fs: float,
                             data_cfg: DataConfig,
                             base_seed: int = 0) -> SplitBundle:
    """Cut each trace into (train, val, test) time segments and build one
    MEAWindowDataset per split.

    Parameters
    ----------
    traces     : list of 1-D float arrays (full-length traces, one per well)
    conditions : phenotype label per trace (0..C-1), aligned with traces
    fs         : common sampling rate [Hz] (all traces must share it)
    data_cfg   : DataConfig supplying window_s, train_stride_s, eval_stride_s,
                 split_fractions, and the augmentation params (fs is injected via
                 data_cfg.resolved_augmentation(fs)).
    base_seed  : seed for the datasets' per-worker augmentation RNG.

    Returns a SplitBundle (see its docstring). Raises a clear error if any split
    ends up with zero windows (window_s too large for that segment).
    """
    if len(traces) != len(conditions):
        raise ValueError("traces and conditions must have equal length")
    if fs <= 0:
        raise ValueError("fs must be > 0")

    W = int(round(data_cfg.window_s * fs))
    train_stride = int(round(data_cfg.train_stride_s * fs))
    eval_stride = int(round(data_cfg.eval_stride_s * fs))
    if W < 1:
        raise ValueError("window_s * fs rounds to < 1 sample")
    if train_stride < 1 or eval_stride < 1:
        raise ValueError("stride_s * fs rounds to < 1 sample")

    stride_by_split = {"train": train_stride, "val": eval_stride, "test": eval_stride}

    # segments + segment sub-traces per split, in ORIGINAL trace order
    seg_bounds_per_trace: List[List[Tuple[int, int]]] = []
    seg_traces = {name: [] for name in _SPLIT_NAMES}
    seg_conditions = {name: [] for name in _SPLIT_NAMES}
    coverage = {name: [] for name in _SPLIT_NAMES}

    for ti, (tr, cond) in enumerate(zip(traces, conditions)):
        tr = np.ascontiguousarray(tr, dtype=np.float32)
        L = tr.shape[0]
        bounds = segment_bounds(L, data_cfg.split_fractions)
        seg_bounds_per_trace.append(bounds)
        for name, (s, e) in zip(_SPLIT_NAMES, bounds):
            sub = tr[s:e]
            seg_traces[name].append(sub)                 # keep even if too short:
            seg_conditions[name].append(int(cond))       # preserves ti alignment
            for rel in window_starts(e - s, W, stride_by_split[name]):
                coverage[name].append((ti, s + rel, s + rel + W, int(cond)))

    # defensive: a phenotype absent from a split makes that split's per-cluster
    # metrics (e.g. silhouette) undefined even though the split is non-empty.
    all_conditions = set(int(c) for c in conditions)
    for name in _SPLIT_NAMES:
        present = set(c for (_, _, _, c) in coverage[name])
        missing = sorted(all_conditions - present)
        if missing:
            warnings.warn(
                "split '%s' has NO windows for condition(s) %s; downstream "
                "per-cluster metrics may be undefined. Use a longer recording, a "
                "smaller window_s, or more traces per class." % (name, missing),
                RuntimeWarning)

    aug_cfg = data_cfg.resolved_augmentation(fs)

    datasets = {}
    for name in _SPLIT_NAMES:
        n_windows = len(coverage[name])
        if n_windows == 0:
            raise ValueError(
                "split '%s' produced 0 windows: window_s=%.4gs (%d samples) "
                "exceeds every '%s' segment. Reduce window_s or adjust "
                "split_fractions." % (name, data_cfg.window_s, W, name))
        datasets[name] = MEAWindowDataset(
            traces=seg_traces[name],
            conditions=seg_conditions[name],
            window_length=W,
            stride=stride_by_split[name],
            aug_cfg=aug_cfg,
            base_seed=base_seed,
        )

    # per-window culture provenance (Section 8.3 of the v3 handoff). Derived from
    # `coverage`, which was built in the SAME nested order MEAWindowDataset
    # enumerates (outer: trace, inner: increasing start), so index i lines up.
    trace_of_window = {
        name: np.array([ti for (ti, _, _, _) in coverage[name]], dtype=int)
        for name in _SPLIT_NAMES
    }
    cultures = {
        name: np.array(sorted(set(int(ti) for (ti, _, _, _) in coverage[name])),
                       dtype=int)
        for name in _SPLIT_NAMES
    }

    return SplitBundle(
        train=datasets["train"],
        val=datasets["val"],
        test=datasets["test"],
        window_length=W,
        train_stride=train_stride,
        eval_stride=eval_stride,
        coverage=coverage,
        seg_bounds=seg_bounds_per_trace,
        trace_of_window=trace_of_window,
        cultures=cultures,
        split_kind="time_segment",
        fold=None,
    )


# --------------------------------------------------------------------------- #
# Whole-culture (per-trace) splitting -- Change 5 of the v3 handoff
# --------------------------------------------------------------------------- #
def apportion(n: int, fractions: Sequence[float],
              rule: str = "largest_remainder") -> List[int]:
    """Split n indivisible items into len(fractions) parts.

    rule = "largest_remainder" (DEFAULT, Hamilton apportionment)
        Give every part floor(f_k * n), then hand the leftovers to the parts with
        the largest fractional remainders (ties broken by ascending part index,
        so the result is deterministic). Every part then receives either
        floor(f_k * n) or ceil(f_k * n), hence

            |assigned_k - f_k * n| < 1   for every k,                        (A)

        and sum_k assigned_k == n exactly.

    rule = "floor"
        The literal rule in Section 8.2 of the handoff: floor for train and val,
        remainder to test. Kept so a pre-existing assignment can be reproduced
        bit-for-bit. WARNING: it does NOT satisfy (A). At n = 18 with
        (0.6, 0.2, 0.2) it returns (10, 3, 5) -- test overshoots its ideal 3.6 by
        1.4 cultures, i.e. 28 percent of the data instead of the requested 20.
        That also breaks assertion (c) of smoke_test_trace_splits.py, which
        requires every count to be within one of its request.

    Returns a list of length len(fractions).
    """
    n = int(n)
    if n < 0:
        raise ValueError("n must be >= 0; got %d" % (n,))
    fr = [float(f) for f in fractions]
    if len(fr) < 1:
        raise ValueError("fractions must be non-empty")
    if any(f < 0.0 for f in fr):
        raise ValueError("fractions must be non-negative; got %r" % (fr,))
    if abs(sum(fr) - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1.0; got %r (sum %.6f)"
                         % (fr, sum(fr)))

    if rule == "floor":
        if len(fr) != 3:
            raise ValueError("rule='floor' is defined for 3 parts only")
        n_tr = int(np.floor(fr[0] * n))
        n_va = int(np.floor(fr[1] * n))
        return [n_tr, n_va, n - n_tr - n_va]

    if rule != "largest_remainder":
        raise ValueError("rule must be 'largest_remainder' or 'floor'; got %r"
                         % (rule,))

    ideal = [f * n for f in fr]
    base = [int(np.floor(x)) for x in ideal]
    leftover = n - sum(base)                       # 0 <= leftover < len(fr)
    order = sorted(range(len(fr)),
                   key=lambda k: (-(ideal[k] - base[k]), k))
    for k in order[:leftover]:
        base[k] += 1
    return base


def _enforce_minima(counts: Sequence[int], minima: Sequence[int]) -> List[int]:
    """Move items between parts until counts[k] >= minima[k] for every k.

    Items are always taken from the part with the largest surplus over its own
    minimum (ties broken by ascending index), so the perturbation away from the
    requested apportionment is as small as possible. Returns None when
    sum(minima) > sum(counts), which is the genuinely infeasible case.
    """
    counts = [int(c) for c in counts]
    minima = [int(m) for m in minima]
    if sum(minima) > sum(counts):
        return None
    for k in range(len(counts)):
        while counts[k] < minima[k]:
            donors = sorted(
                ((counts[j] - minima[j], -j) for j in range(len(counts))
                 if j != k),
                reverse=True)
            surplus, neg_j = donors[0]
            if surplus <= 0:                       # unreachable given the guard
                return None
            counts[-neg_j] -= 1
            counts[k] += 1
    return counts


def assign_cultures(conditions: Sequence[int],
                    fractions: Sequence[float] = (0.6, 0.2, 0.2),
                    seed: int = 0,
                    mode: str = "fractional",
                    fold: Optional[int] = None,
                    min_train_cultures_per_class: int = 2,
                    alloc_rule: str = "largest_remainder"
                    ) -> Dict[str, List[int]]:
    """Assign each culture index u to EXACTLY ONE of train / val / test.

    This is the pure, torch-free, array-free core of make_trace_splits: it takes
    only the label vector and returns the three index lists, so it can be tested
    exhaustively without generating a single trace.

    Stratification. The fractions are applied WITHIN each class separately, so
    every class is present in every split (this is what makes per-class metrics
    defined on every split).

    mode = "fractional"
        For each class c, permute that class's culture indices with an RNG seeded
        from (seed, c), apportion by `alloc_rule`, then repair so that each split
        holds at least one culture of class c and train holds at least
        min_train_cultures_per_class of them.

    mode = "leave_one_out"
        n_folds = min_c n_c folds. In fold f, culture f of each class is test,
        culture (f + 1) mod n_c is validation, the rest are train. Cultures are
        taken in ASCENDING GLOBAL INDEX order, NOT permuted, so "fold 3" names
        the same held-out culture on every machine and in every log. Each culture
        is test in exactly one fold iff every class has the same n_c.

    Returns {"train": [...], "val": [...], "test": [...]}, each list sorted
    ascending. Raises ValueError with n_c and the fractions named when the
    minimum-occupancy constraints cannot be met.
    """
    if mode not in ("fractional", "leave_one_out"):
        raise ValueError("mode must be 'fractional' or 'leave_one_out'; got %r"
                         % (mode,))
    min_train = int(min_train_cultures_per_class)
    if min_train < 1:
        raise ValueError("min_train_cultures_per_class must be >= 1; got %d"
                         % (min_train,))

    cond = np.asarray(conditions, dtype=int).ravel()
    if cond.size == 0:
        raise ValueError("conditions is empty: there are no cultures to split")
    # sorted() rather than set iteration: assignment must not depend on hash order
    classes = sorted(set(int(c) for c in cond.tolist()))
    if classes[0] < 0:
        raise ValueError(
            "class labels must be non-negative (0..C-1); got %r. The per-class "
            "RNG is seeded with [seed, c], which requires c >= 0."
            % (classes[:8],))
    by_class = {c: [int(u) for u in np.flatnonzero(cond == c).tolist()]
                for c in classes}

    out = {name: [] for name in _SPLIT_NAMES}

    if mode == "fractional":
        for c in classes:
            idx = list(by_class[c])                # already ascending
            n_c = len(idx)
            if n_c < min_train + 2:
                raise ValueError(
                    "class %d has n_c = %d culture(s), but a whole-culture split "
                    "needs at least min_train_cultures_per_class + 2 = %d "
                    "(train >= %d, val >= 1, test >= 1) under fractions %r. Use "
                    "mode='leave_one_out', lower min_train_cultures_per_class, "
                    "or record more cultures."
                    % (c, n_c, min_train + 2, min_train, tuple(fractions)))
            # independent, reproducible stream per class: adding a class does not
            # perturb the assignment of the classes already there
            rng = np.random.default_rng([int(seed), int(c)])
            perm = rng.permutation(n_c)
            shuffled = [idx[int(p)] for p in perm]

            counts = apportion(n_c, fractions, rule=alloc_rule)
            repaired = _enforce_minima(counts, (min_train, 1, 1))
            if repaired is None:                   # unreachable given the guard
                raise ValueError(
                    "class %d: cannot satisfy (train >= %d, val >= 1, test >= 1) "
                    "with n_c = %d and fractions %r"
                    % (c, min_train, n_c, tuple(fractions)))
            n_tr, n_va, _n_te = repaired
            out["train"].extend(shuffled[:n_tr])
            out["val"].extend(shuffled[n_tr:n_tr + n_va])
            out["test"].extend(shuffled[n_tr + n_va:])

    else:                                          # leave_one_out
        n_folds = min(len(by_class[c]) for c in classes)
        sizes = sorted(set(len(by_class[c]) for c in classes))
        if len(sizes) > 1:
            warnings.warn(
                "leave_one_out with unequal class sizes %r: using n_folds = %d "
                "(the smallest). Cultures of the larger classes beyond that "
                "index are always training cultures, so 'each culture is test "
                "exactly once' does NOT hold here." % (sizes, n_folds),
                RuntimeWarning)
        if fold is None:
            raise ValueError(
                "mode='leave_one_out' requires an explicit fold in [0, %d); "
                "pass fold=0 for the first fold" % (n_folds,))
        f = int(fold)
        if not (0 <= f < n_folds):
            raise ValueError("fold must lie in [0, %d); got %d" % (n_folds, f))
        for c in classes:
            idx = list(by_class[c])                # ascending, NOT permuted
            n_c = len(idx)
            if n_c < min_train + 2:
                raise ValueError(
                    "class %d has n_c = %d culture(s); leave_one_out needs at "
                    "least min_train_cultures_per_class + 2 = %d (1 test, 1 val, "
                    ">= %d train)" % (c, n_c, min_train + 2, min_train))
            i_te = f % n_c
            i_va = (f + 1) % n_c
            out["test"].append(idx[i_te])
            out["val"].append(idx[i_va])
            out["train"].extend([idx[j] for j in range(n_c)
                                 if j not in (i_te, i_va)])

    for name in _SPLIT_NAMES:
        out[name] = sorted(out[name])
    return out


def make_trace_splits(traces: Sequence[np.ndarray],
                      conditions: Sequence[int],
                      fs: float,
                      data_cfg: DataConfig,
                      base_seed: int = 0,
                      mode: str = "fractional",
                      fold: Optional[int] = None,
                      split_seed: int = 0,
                      min_train_cultures_per_class: int = 2,
                      fractions: Optional[Sequence[float]] = None,
                      alloc_rule: str = "largest_remainder") -> SplitBundle:
    """Whole-culture train / val / test split: a culture belongs to ONE split.

    Replaces the time-segment split for the deployment question "classify a
    culture the network has never seen". make_time_segment_splits guarantees only
    that no WINDOW straddles a boundary; every culture still contributes windows
    to all three splits, so culture identity is exploitable. Here, assignment is
    at culture granularity and stratified by class.

    The first five parameters are positionally identical to
    make_time_segment_splits, so the two splitters are drop-in swappable at every
    existing call site.

    Parameters
    ----------
    traces      : list of 1-D float arrays, one FULL-LENGTH trace per culture
    conditions  : class label c per culture, aligned with traces
    fs          : common sampling rate [Hz]
    data_cfg    : DataConfig supplying window_s, train_stride_s, eval_stride_s,
                  split_fractions (unless `fractions` overrides it) and the
                  augmentation params
    base_seed   : seed for the datasets' per-worker augmentation RNG (NOT the
                  split assignment -- see split_seed)
    mode        : "fractional" or "leave_one_out"
    fold        : which leave-one-out fold to build; required for that mode
    split_seed  : seed for the culture PERMUTATION. Deliberately separate from
                  base_seed so that seed-averaging over training seeds does not
                  silently reshuffle the split underneath the average.
    min_train_cultures_per_class : floor on training cultures per class. The
                  default of 2 is the smallest value at which cross-culture
                  positives (Change 4) exist at all, since an anchor needs at
                  least one same-class partner from a DIFFERENT culture.
    fractions   : (train, val, test); defaults to data_cfg.split_fractions
    alloc_rule  : "largest_remainder" (default) or "floor"; see apportion()

    Windowing. Windows tile the WHOLE trace -- there is no time cut -- using
    exactly MEAWindowDataset's rule via window_starts(), with train_stride for
    training cultures and eval_stride for validation and test cultures.

    Determinism. Identical (split_seed, fractions, conditions, mode, fold) give
    an identical assignment on any machine: class order comes from sorted(), the
    per-class stream is np.random.default_rng([split_seed, c]), and no set or
    dict iteration order is consulted anywhere in the assignment.

    Returns a SplitBundle with split_kind == "trace" and a populated
    trace_of_window (GLOBAL culture indices, Section 8.3 of the handoff).
    """
    if len(traces) != len(conditions):
        raise ValueError("traces and conditions must have equal length")
    if fs <= 0:
        raise ValueError("fs must be > 0")

    frac = tuple(data_cfg.split_fractions if fractions is None else fractions)

    W = int(round(data_cfg.window_s * fs))
    train_stride = int(round(data_cfg.train_stride_s * fs))
    eval_stride = int(round(data_cfg.eval_stride_s * fs))
    if W < 1:
        raise ValueError("window_s * fs rounds to < 1 sample")
    if train_stride < 1 or eval_stride < 1:
        raise ValueError("stride_s * fs rounds to < 1 sample")
    stride_by_split = {"train": train_stride, "val": eval_stride,
                       "test": eval_stride}

    assignment = assign_cultures(
        conditions=conditions,
        fractions=frac,
        seed=split_seed,
        mode=mode,
        fold=fold,
        min_train_cultures_per_class=min_train_cultures_per_class,
        alloc_rule=alloc_rule,
    )

    # hard invariant, cheap to check, catastrophic to get wrong
    for a in range(len(_SPLIT_NAMES)):
        for b in range(a + 1, len(_SPLIT_NAMES)):
            na, nb = _SPLIT_NAMES[a], _SPLIT_NAMES[b]
            shared = sorted(set(assignment[na]) & set(assignment[nb]))
            if shared:
                raise AssertionError(
                    "internal error: culture(s) %s assigned to both '%s' and "
                    "'%s'" % (shared, na, nb))

    aug_cfg = data_cfg.resolved_augmentation(fs)
    arrays = [np.ascontiguousarray(t, dtype=np.float32) for t in traces]

    datasets = {}
    coverage = {name: [] for name in _SPLIT_NAMES}
    trace_of_window = {}
    too_short = []

    for name in _SPLIT_NAMES:
        globals_here = assignment[name]            # sorted GLOBAL culture indices
        stride = stride_by_split[name]
        sub_traces = [arrays[u] for u in globals_here]
        sub_conditions = [int(conditions[u]) for u in globals_here]

        for u in globals_here:
            if arrays[u].shape[0] < W:
                too_short.append((name, u, int(arrays[u].shape[0])))

        n_windows = sum(len(window_starts(arrays[u].shape[0], W, stride))
                        for u in globals_here)
        if n_windows == 0:
            raise ValueError(
                "split '%s' produced 0 windows: window_s = %.4g s (%d samples) "
                "exceeds every assigned culture's length. Reduce window_s, or "
                "check that the traces are full-length recordings."
                % (name, data_cfg.window_s, W))

        ds = MEAWindowDataset(
            traces=sub_traces,
            conditions=sub_conditions,
            window_length=W,
            stride=stride,
            aug_cfg=aug_cfg,
            base_seed=base_seed,
        )
        datasets[name] = ds

        # Provenance is read back OUT of the Dataset's own index rather than
        # re-derived, so it cannot drift from what the Dataset actually yields.
        # ds.index holds (local_trace_idx, start, condition); local -> global is
        # positional because sub_traces was built in globals_here order.
        g_of_local = {i: u for i, u in enumerate(globals_here)}
        trace_of_window[name] = np.array(
            [g_of_local[ti] for (ti, _s, _c) in ds.index], dtype=int)
        coverage[name] = [(g_of_local[ti], int(s), int(s) + W, int(c))
                          for (ti, s, c) in ds.index]

    if too_short:
        warnings.warn(
            "culture(s) shorter than one window (%d samples) contribute NO "
            "windows and are therefore silently absent from their split: %r. "
            "This is the one route by which a class can vanish from a split "
            "despite stratified assignment." % (W, too_short[:8]),
            RuntimeWarning)

    all_conditions = set(int(c) for c in conditions)
    for name in _SPLIT_NAMES:
        present = set(c for (_, _, _, c) in coverage[name])
        missing = sorted(all_conditions - present)
        if missing:
            warnings.warn(
                "split '%s' has NO windows for condition(s) %s; per-cluster "
                "metrics (ARI, silhouette) are undefined there." % (name, missing),
                RuntimeWarning)

    cultures = {name: np.array(assignment[name], dtype=int)
                for name in _SPLIT_NAMES}

    return SplitBundle(
        train=datasets["train"],
        val=datasets["val"],
        test=datasets["test"],
        window_length=W,
        train_stride=train_stride,
        eval_stride=eval_stride,
        coverage=coverage,
        seg_bounds=[],                             # no time cut is made
        trace_of_window=trace_of_window,
        cultures=cultures,
        split_kind="trace",
        fold=(None if mode != "leave_one_out" else int(fold)),
    )
