"""
data_pipeline.py
================

Data loading + windowing + Dataset + condition-balanced batching + collation
for the 1D-CNN contrastive pipeline. Decoupled from the model and from plotting
(directive 2): this module only produces labelled contrastive batches.

It implements the condition-level label scheme with OPTION (b): each anchor's
profile-destroying surrogates are kept as per-anchor hard negatives, realized by
giving every destroyed surrogate a UNIQUE label (disjoint from the condition
labels), so it only ever serves as a negative.

HPC notes
---------
    * Augmentation runs on CPU inside DataLoader worker processes
      (num_workers > 0), overlapping with model compute.
    * Reproducibility is guaranteed for a fixed (seed, num_workers): each worker
      gets a deterministic numpy Generator via `seed_worker`, and the batch
      sampler is seeded per epoch.
    * No interactive plotting, no hard-coded paths: everything is driven by the
      front-end config.

Label constants
---------------
    CONTROL = 0 , PATHO = 1            (condition-level labels for positives)
    destroyed surrogates -> unique labels >= unique_label_base (negatives-only)
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from augmentation import AugmentationConfig, build_triplet_instance

__all__ = [
    "CONTROL",
    "PATHO",
    "closest_power_of_2",
    "SyntheticTraceProvider",
    "NeuronalTracesProvider",
    "NumpyTraceProvider",
    "MEAWindowDataset",
    "ConditionBalancedBatchSampler",
    "TripletCollator",
    "seed_worker",
]

CONTROL = 0
PATHO = 1


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def closest_power_of_2(n: float) -> int:
    """Nearest power of two to n (in log space).

    NOTE: confirm this matches the engine's existing `closest_power_of_2`
    definition; if the engine FLOORS instead, replace round() with floor().
    """
    if n < 1:
        raise ValueError(f"closest_power_of_2 needs n >= 1, got {n}")
    return int(2 ** round(math.log2(n)))


# --------------------------------------------------------------------------- #
# trace providers (data loading is injectable -> testable without .mat files)
# --------------------------------------------------------------------------- #
class SyntheticTraceProvider:
    """Generate burst-like, non-negative synthetic traces for pipeline
    validation / HPC dry-runs (NOT biologically faithful). Patho traces are
    denser and more irregular so control vs patho look different in debug plots.
    """

    def __init__(self, duration_s: float = 600.0, fs: float = 50.0, seed: int = 0):
        self.duration_s = float(duration_s)
        self.fs = float(fs)
        self.seed = int(seed)

    def __call__(self, condition: int, trace_id: int) -> Tuple[np.ndarray, float]:
        rng = np.random.default_rng(self.seed + 1000 * int(condition) + int(trace_id))
        T = int(self.duration_s * self.fs)
        t = np.arange(T) / self.fs
        x = np.zeros(T, dtype=np.float64)
        rate = 0.25 if condition == CONTROL else 0.55          # bursts / s
        n_bursts = max(1, int(rng.poisson(rate * self.duration_s)))
        centers = rng.uniform(0.0, self.duration_s, n_bursts)
        for c in centers:
            w = 0.5 if condition == CONTROL else float(rng.uniform(0.15, 0.7))
            a = 1.0 if condition == CONTROL else float(rng.uniform(0.6, 1.4))
            x += a * np.exp(-0.5 * ((t - c) / w) ** 2)
        return x.astype(np.float32), self.fs


class NeuronalTracesProvider:
    """Thin wrapper around the project's existing `Neuronal_traces` loader
    (directive 1: reuse the tested loader). Imported lazily so this module
    stays importable without the engine present.

    `neuronal_traces_fn` must have the signature of the engine's function and
    return (smoothed_cumulative: np.ndarray, fs_downsampled: float).
    """

    def __init__(self, neuronal_traces_fn: Callable, w_size: float = 0.02,
                 gaussian_window: float = 0.04, t_rec: float = 600.0):
        self._fn = neuronal_traces_fn
        self.w_size = w_size
        self.gaussian_window = gaussian_window
        self.t_rec = t_rec

    def __call__(self, folder: str, base: str) -> Tuple[np.ndarray, float]:
        smoothed_cumulative, fs_downsampled = self._fn(
            Char_folder=folder, Char_base=base, w_size=self.w_size,
            Gaussian_window=self.gaussian_window, t_rec=self.t_rec, Visible=False,
        )
        return np.ascontiguousarray(smoothed_cumulative, dtype=np.float32), float(fs_downsampled)


# --------------------------------------------------------------------------- #
# Dataset: windows traces (with optional overlap) + augments per item
# --------------------------------------------------------------------------- #
class NumpyTraceProvider:
    """Load a pre-computed smoothed cumulative IFR trace from a .npz archive
    produced by generate_burst_data.py.

    The archive must contain at minimum:
        ifr_trace : (K,) float32 — smoothed cumulative IFR  R̃[k].
        fs_ifr    : scalar float — sampling rate f_s^{IFR}  [Hz].

    Parameters
    ----------
    (none at construction — the .npz path is passed at call time so a single
    provider instance can be reused across multiple files)

    Usage
    -----
        provider = NumpyTraceProvider()
        trace, fs = provider("/path/to/control_0.npz")
    """

    def __call__(self, npz_path: str) -> Tuple[np.ndarray, float]:
        data = np.load(npz_path, allow_pickle=True)
        ifr  = np.ascontiguousarray(data["ifr_trace"], dtype=np.float32)
        fs   = float(data["fs_ifr"])
        return ifr, fs


class MEAWindowDataset(Dataset):
    """Windows a set of (trace, condition) pairs and, on access, builds one
    contrastive instance (anchor + positives + negatives) for the window.

    Overlapping windows (stride < window_length) raise the number of distinct
    windows, addressing the low-diversity issue.
    """

    def __init__(
        self,
        traces: Sequence[np.ndarray],
        conditions: Sequence[int],
        window_length: int,
        stride: int,
        aug_cfg: AugmentationConfig,
        base_seed: int = 0,
    ):
        if len(traces) != len(conditions):
            raise ValueError("traces and conditions must have equal length.")
        self.traces = [np.ascontiguousarray(t, dtype=np.float32) for t in traces]
        self.window_length = int(window_length)
        self.stride = int(stride)
        self.aug_cfg = aug_cfg
        self.base_seed = int(base_seed)
        # per-worker RNG (replaced in seed_worker for num_workers > 0)
        self.rng = np.random.default_rng(self.base_seed)

        self.index: List[Tuple[int, int, int]] = []   # (trace_idx, start, condition)
        for ti, (tr, cond) in enumerate(zip(self.traces, conditions)):
            L = tr.shape[0]
            if L < self.window_length:
                continue
            s = 0
            while s + self.window_length <= L:
                self.index.append((ti, s, int(cond)))
                s += self.stride
        if not self.index:
            raise ValueError("No windows produced; check window_length vs trace lengths.")
        self.conditions_per_item = np.array([c for (_, _, c) in self.index], dtype=int)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict:
        ti, s, cond = self.index[i]
        window = self.traces[ti][s:s + self.window_length]
        window = torch.from_numpy(np.ascontiguousarray(window)).float()
        anchor, positives, negatives = build_triplet_instance(window, self.aug_cfg, self.rng)
        return {
            "anchor": anchor,            # (1, T)
            "positives": positives,      # (1+P, T)  -- condition label
            "negatives": negatives,      # (N, T)    -- unique negatives
            "condition": int(cond),
            "meta": (ti, s),
        }


def seed_worker(worker_id: int) -> None:
    """worker_init_fn: give each DataLoader worker a deterministic RNG."""
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    seed = (ds.base_seed + worker_id + 1) % (2 ** 31)
    ds.rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
# Condition-balanced batch sampler (every batch has BOTH conditions)
# --------------------------------------------------------------------------- #
class ConditionBalancedBatchSampler(Sampler):
    """Yield index batches with exactly `per_condition` windows from each
    condition, so every batch supports cross-condition triplets.

    Ecosystem alternative (directive 1): pytorch_metric_learning.samplers
    .MPerClassSampler(labels, m=per_condition, batch_size=...). A tiny custom
    sampler is used here to keep this module dependency-free and testable.
    """

    def __init__(self, conditions: Sequence[int], per_condition: int,
                 n_batches: int, seed: int = 0):
        self.by_cond: Dict[int, List[int]] = {}
        for idx, c in enumerate(conditions):
            self.by_cond.setdefault(int(c), []).append(idx)
        self.per_condition = int(per_condition)
        self.n_batches = int(n_batches)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.n_batches):
            batch: List[int] = []
            for c, idxs in self.by_cond.items():
                replace = len(idxs) < self.per_condition
                pick = rng.choice(idxs, size=self.per_condition, replace=replace)
                batch.extend(int(j) for j in pick)
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.n_batches


# --------------------------------------------------------------------------- #
# Collator: assemble (X, y) implementing OPTION (b)
# --------------------------------------------------------------------------- #
class TripletCollator:
    """Concatenate per-window instances into one embedding batch and assign
    labels per the option-(b) scheme.

    Returns
    -------
    X     : (M, T) float32 -- all positives + negatives, M = sum_b (1+P_b+N_b)
    y     : (M,)  long     -- condition for positives; unique >= base for negatives
    metas : list           -- per-source-window (trace_idx, start) for debugging
    """

    def __init__(self, destroyed_label_mode: str = "unique",
                 unique_label_base: int = 1_000_000, shared_destroyed_label: int = 2):
        if destroyed_label_mode not in ("unique", "shared"):
            raise ValueError("destroyed_label_mode must be 'unique' or 'shared'.")
        self.destroyed_label_mode = destroyed_label_mode
        self.unique_label_base = int(unique_label_base)
        self.shared_destroyed_label = int(shared_destroyed_label)

    def __call__(self, batch: List[Dict]):
        emb: List[torch.Tensor] = []
        lab: List[torch.Tensor] = []
        metas = []
        next_uniq = self.unique_label_base

        for item in batch:
            pos = item["positives"]            # (1+P, T)
            neg = item["negatives"]            # (N, T)
            cond = int(item["condition"])

            emb.append(pos)
            lab.append(torch.full((pos.shape[0],), cond, dtype=torch.long))

            emb.append(neg)
            n = neg.shape[0]
            if self.destroyed_label_mode == "unique":
                lab.append(torch.arange(next_uniq, next_uniq + n, dtype=torch.long))
                next_uniq += n
            else:
                lab.append(torch.full((n,), self.shared_destroyed_label, dtype=torch.long))

            metas.append(item["meta"])

        X = torch.cat(emb, dim=0).to(torch.float32)
        y = torch.cat(lab, dim=0).to(torch.long)
        return X, y, metas
