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
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

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
                 width_min: float = 0.15, width_max: float = 0.70):
        if n_classes < 1:
            raise ValueError("n_classes must be >= 1")
        if duration_s <= 0 or fs <= 0:
            raise ValueError("duration_s and fs must be > 0")
        if not (0 < width_min <= width_max):
            raise ValueError("require 0 < width_min <= width_max")
        self.n_classes = int(n_classes)
        self.duration_s = float(duration_s)
        self.fs = float(fs)
        self.seed = int(seed)
        self.rate_min = float(rate_min)
        self.rate_max = float(rate_max)
        self.width_min = float(width_min)
        self.width_max = float(width_max)

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
        rate = self.rate_min + (self.rate_max - self.rate_min) * frac          # bursts / s
        base_width = self.width_max - (self.width_max - self.width_min) * frac  # seconds

        rng = np.random.default_rng(self.seed + 1000 * condition + trace_id)
        T = int(self.duration_s * self.fs)
        t = np.arange(T) / self.fs
        x = np.zeros(T, dtype=np.float64)

        n_bursts = max(1, int(rng.poisson(rate * self.duration_s)))
        centers = rng.uniform(0.0, self.duration_s, n_bursts)
        for c in centers:
            if condition == 0:
                w = base_width                        # regular baseline
                a = 1.0
            else:
                w = float(rng.uniform(0.5 * base_width, 1.5 * base_width))  # irregular
                a = float(rng.uniform(0.6, 1.4))
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
    """
    train: MEAWindowDataset
    val: MEAWindowDataset
    test: MEAWindowDataset
    window_length: int
    train_stride: int
    eval_stride: int
    coverage: Dict[str, List[Tuple[int, int, int, int]]]
    seg_bounds: List[List[Tuple[int, int]]]


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

    return SplitBundle(
        train=datasets["train"],
        val=datasets["val"],
        test=datasets["test"],
        window_length=W,
        train_stride=train_stride,
        eval_stride=eval_stride,
        coverage=coverage,
        seg_bounds=seg_bounds_per_trace,
    )
