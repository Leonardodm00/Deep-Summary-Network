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

import logging
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from augmentation import AugmentationConfig, build_triplet_instance
from batch_geometry import DEFAULT_Q_CAP_FRACTION, resolve_batch_geometry

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
        ifr_trace : (K,) float32 -- smoothed cumulative IFR  R[k].
        fs_ifr    : scalar float -- sampling rate f_s^{IFR}  [Hz].

    Parameters
    ----------
    (none at construction -- the .npz path is passed at call time so a single
    provider instance can be reused across multiple files)

    Usage
    -----
        provider = NumpyTraceProvider()
        trace, fs = provider("/path/to/control_0.npz")
    """

    def __call__(self, npz_path: str) -> Tuple[np.ndarray, float]:
        data = np.load(npz_path, allow_pickle=True)
        if "ifr_trace" not in data.files:
            raise KeyError(
                "%s has no 'ifr_trace' key (found: %r). The channel-subset "
                "extractor writes its array under 'X'; re-run "
                "run_channel_subset_extraction.py, which now also writes the "
                "'ifr_trace' alias this provider requires."
                % (npz_path, sorted(data.files)))
        ifr = np.ascontiguousarray(data["ifr_trace"], dtype=np.float32)
        fs = float(data["fs_ifr"])
        if ifr.ndim not in (1, 2):
            raise ValueError(
                "%s: ifr_trace must be (K,) or (C, K); got shape %r"
                % (npz_path, ifr.shape))
        # [multichannel] C is read from the archive's own metadata when present,
        # NEVER inferred from ifr.shape[0]. The extractor emits (C, K) arrays in
        # BOTH mode='multichannel' (rows are channels, in_channels = C) and
        # mode='per_region_single' (rows are separate samples, in_channels = 1),
        # so shape alone is ambiguous and would silently mislabel the latter.
        if "in_channels" in data.files:
            C_file = int(data["in_channels"])
            C_arr = 1 if ifr.ndim == 1 else int(ifr.shape[0])
            if C_file != C_arr:
                raise ValueError(
                    "%s: in_channels=%d disagrees with ifr_trace shape %r. A "
                    "per_region_single archive holds independent SAMPLES, not "
                    "channels; split it into one .npz per subregion before "
                    "loading." % (npz_path, C_file, ifr.shape))
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
        cultures: Optional[Sequence[int]] = None,
    ):
        if len(traces) != len(conditions):
            raise ValueError("traces and conditions must have equal length.")
        # [K3] culture id per LOCAL trace. gamma(u) = u (identity) when not
        # given, i.e. one trace == one culture -- the pre-K3 behaviour, which is
        # what keeps every existing caller and every existing suite unchanged.
        # Non-identity values arrive when sibling subregion traces of one well
        # must be grouped; train.py maps the dataset's local trace index through
        # this vector to build the sampler's culture array g.
        if cultures is None:
            self.cultures = list(range(len(traces)))
        else:
            if len(cultures) != len(traces):
                raise ValueError(
                    "cultures has %d entries but traces has %d; they must be "
                    "parallel." % (len(cultures), len(traces)))
            self.cultures = [int(c) for c in cultures]
        self.traces = [np.ascontiguousarray(t, dtype=np.float32) for t in traces]
        self.window_length = int(window_length)
        self.stride = int(stride)
        self.aug_cfg = aug_cfg
        self.base_seed = int(base_seed)
        # per-worker RNG (replaced in seed_worker for num_workers > 0)
        self.rng = np.random.default_rng(self.base_seed)

        self.index: List[Tuple[int, int, int]] = []   # (trace_idx, start, condition)
        for ti, (tr, cond) in enumerate(zip(self.traces, conditions)):
            L = tr.shape[-1]                           # time axis (works for (T,) and (C, T))
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
        window = self.traces[ti][..., s:s + self.window_length]   # (W,) or (C, W)
        window = torch.from_numpy(np.ascontiguousarray(window)).float()
        anchor, positives, negatives, pos_pre, neg_pre = build_triplet_instance(
            window, self.aug_cfg, self.rng, return_pre_shift=True)
        return {
            "anchor":        anchor,     # (1, T)  or (1, C, T)   clean unshifted window
            "positives":     positives,  # (1+P,T) or (1+P,C,T)   warp + shift (network input)
            "negatives":     negatives,  # (N, T)  or (N, C, T)   warp + shift (network input)
            "pos_pre_shift": pos_pre,    # (1+P,T) or (1+P,C,T)   warp only    (viz only)
            "neg_pre_shift": neg_pre,    # (N, T)  or (N, C, T)   warp only    (viz only)
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
    """Yield index batches supporting the configured positive strategy.

    Two modes, selected by ``positives_mode``:

    * ``"augmentation"`` (default) -- the LEGACY behaviour, UNCHANGED. Each batch
      carries exactly ``per_condition`` windows drawn (with replacement only when
      a class is smaller than ``per_condition``) from every condition, so every
      batch supports cross-condition triplets. Each window is later expanded by
      the collator into (anchor + warp positives + surrogate negatives). This
      path is kept byte-identical so Change 4 stays inert until switched on
      (assertion [I]).

    * ``"cross_culture"`` -- CULTURES-FIRST-THEN-WINDOWS. The batch geometry is
      resolved ONCE at construction via ``resolve_batch_geometry`` (Eq. (2),
      Eq. (3), the two caps, the easy-positive precondition, and the q <= W_min
      guard), and ``geo.notes`` is logged ONCE. Every batch is then built by
      drawing, for each class, ``U_eff`` DISTINCT cultures WITHOUT replacement
      within the batch -- re-drawn independently across batches, never
      partitioned (H-section 5.2: ``U_eff`` rarely divides the available count and
      a partition would silently drop cultures) -- then ``q`` windows WITHOUT
      replacement from each sampled culture. The anchor's positive is then a real
      same-class window of a DIFFERENT culture, found by the miner, rather than a
      warp of the anchor's own window.

    Culture key
    -----------
    In ``"cross_culture"`` mode the caller passes ``trace_of_window``, an int
    array parallel to ``conditions`` giving each window's culture. The pipeline's
    natural value is the dataset-local trace index (``MEAWindowDataset.index[i][0]``:
    windows sharing a trace index are the same culture), which is sufficient for
    the census and the two-level draw -- both need only same-vs-different-culture
    grouping -- and avoids threading the global ``trace_of_window`` through
    ``train_model``.

    Ecosystem alternative (directive 1): pytorch_metric_learning.samplers
    .MPerClassSampler / .HierarchicalSampler. A small custom sampler is used to
    keep this module dependency-free and testable, and because neither ecosystem
    sampler implements the two-level cultures-then-windows draw with the Eq. (3)
    availability clamp.
    """

    def __init__(self, conditions: Sequence[int], per_condition: int,
                 n_batches: int, seed: int = 0,
                 positives_mode: str = "augmentation",
                 trace_of_window: Optional[Sequence[int]] = None,
                 mining_strategy: Optional[str] = None,
                 cultures_per_class_per_batch: Optional[int] = None,
                 windows_per_culture_per_batch: Optional[int] = None,
                 n_surrogates: Optional[int] = None,
                 max_group_size: Optional[int] = None,
                 exclude_same_culture_positives: Optional[bool] = None,
                 min_train_cultures_per_class: int = 2,
                 max_batch_rows: Optional[int] = None,
                 q_cap_fraction: float = DEFAULT_Q_CAP_FRACTION):
        # by_cond built EXACTLY as before, so the augmentation path is unchanged.
        self.by_cond: Dict[int, List[int]] = {}
        for idx, c in enumerate(conditions):
            self.by_cond.setdefault(int(c), []).append(idx)
        self.per_condition = int(per_condition)
        self.n_batches = int(n_batches)
        self.seed = int(seed)
        self.epoch = 0

        self.positives_mode = str(positives_mode)
        if self.positives_mode not in ("augmentation", "cross_culture"):
            raise ValueError(
                "positives_mode must be 'augmentation' or 'cross_culture'; "
                "got %r" % (positives_mode,))

        # cross-culture state (None/empty under augmentation, so the change is
        # inert by default)
        self.geometry = None
        self.geo_notes: List[str] = []
        self._cultures_by_class: Dict[int, List[int]] = {}
        self._windows_by_culture: Dict[int, List[int]] = {}

        if self.positives_mode == "cross_culture":
            self._init_cross_culture(
                conditions=conditions,
                trace_of_window=trace_of_window,
                mining_strategy=mining_strategy,
                cultures_per_class_per_batch=cultures_per_class_per_batch,
                windows_per_culture_per_batch=windows_per_culture_per_batch,
                n_surrogates=n_surrogates,
                max_group_size=max_group_size,
                exclude_same_culture_positives=exclude_same_culture_positives,
                min_train_cultures_per_class=min_train_cultures_per_class,
                max_batch_rows=max_batch_rows,
                q_cap_fraction=q_cap_fraction,
            )

    def _init_cross_culture(self, conditions, trace_of_window, mining_strategy,
                            cultures_per_class_per_batch,
                            windows_per_culture_per_batch, n_surrogates,
                            max_group_size, exclude_same_culture_positives,
                            min_train_cultures_per_class, max_batch_rows,
                            q_cap_fraction):
        if trace_of_window is None:
            raise ValueError(
                "positives_mode='cross_culture' requires trace_of_window, the "
                "per-window culture array g (parallel to conditions).")
        y = np.asarray(conditions, dtype=int).ravel()
        g = np.asarray(trace_of_window, dtype=int).ravel()
        if g.size != y.size:
            raise ValueError(
                "trace_of_window has %d entries but conditions has %d; they must "
                "be parallel arrays over the SAME windows." % (g.size, y.size))

        # Resolve and CHECK the geometry ONCE. This raises on every inadmissible
        # combination (Eq. (3) starvation, the easy-positive precondition, both
        # caps, and q > W_min) rather than silently repairing it.
        geo = resolve_batch_geometry(
            trace_of_window=g,
            conditions=y,
            positives_mode="cross_culture",
            mining_strategy=mining_strategy,
            cultures_per_class_per_batch=cultures_per_class_per_batch,
            windows_per_culture_per_batch=windows_per_culture_per_batch,
            n_surrogates=n_surrogates,
            max_group_size=max_group_size,
            exclude_same_culture_positives=exclude_same_culture_positives,
            min_train_cultures_per_class=min_train_cultures_per_class,
            max_batch_rows=max_batch_rows,
            q_cap_fraction=q_cap_fraction,
        )
        self.geometry = geo
        self.geo_notes = list(geo.notes)

        # windows of each culture, and cultures of each class, for the draw
        for idx, u in enumerate(g.tolist()):
            self._windows_by_culture.setdefault(int(u), []).append(idx)
        seen_pairs = set()
        for c_raw, u_raw in zip(y.tolist(), g.tolist()):
            key = (int(c_raw), int(u_raw))
            if key not in seen_pairs:
                seen_pairs.add(key)
                self._cultures_by_class.setdefault(int(c_raw), []).append(int(u_raw))
        for c in self._cultures_by_class:
            self._cultures_by_class[c] = sorted(self._cultures_by_class[c])

        # log the resolved geometry ONCE at construction (H-section 7.2)
        logger = logging.getLogger(__name__)
        for line in self.geo_notes:
            logger.info("[cross_culture geometry] %s", line)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        if self.positives_mode == "cross_culture":
            yield from self._iter_cross_culture()
        else:
            yield from self._iter_augmentation()

    def _iter_augmentation(self):
        # UNCHANGED legacy draw (assertion [I]): identical RNG call sequence.
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.n_batches):
            batch: List[int] = []
            for c, idxs in self.by_cond.items():
                replace = len(idxs) < self.per_condition
                pick = rng.choice(idxs, size=self.per_condition, replace=replace)
                batch.extend(int(j) for j in pick)
            rng.shuffle(batch)
            yield batch

    def _iter_cross_culture(self):
        geo = self.geometry
        u_eff = int(geo.cultures_effective)
        q = int(geo.windows_per_culture_per_batch)
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.n_batches):
            batch: List[int] = []
            for c in sorted(self._cultures_by_class.keys()):
                cultures = np.asarray(self._cultures_by_class[c], dtype=int)
                # U_eff DISTINCT cultures, WITHOUT replacement WITHIN this batch;
                # re-drawn independently next batch (never a fixed partition).
                chosen = rng.choice(cultures, size=u_eff, replace=False)
                for u in chosen.tolist():
                    wins = np.asarray(self._windows_by_culture[int(u)], dtype=int)
                    # q windows WITHOUT replacement (q <= W_min guaranteed by geo,
                    # so this never falls back to sampling with replacement).
                    pick = rng.choice(wins, size=q, replace=False)
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
    X     : (M, T) or (M, C, T) float32 -- all positives + negatives,
            M = sum_b (1+P_b+N_b); the channel axis (if present) rides through
            torch.cat unchanged.
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
            pos = item["positives"]            # (1+P, T), P may be 0
            neg = item["negatives"]            # (N, T),   N may be 0
            cond = int(item["condition"])

            # A block is appended ONLY when it has rows, so an empty pool (P_b = 0
            # under cross-culture positives, or N_s = 0) is dropped rather than
            # concatenated as a (0, T) tensor. Under "augmentation" both pools are
            # non-empty, so this is byte-identical to the pre-Change-4 collator.
            if pos.shape[0] > 0:
                emb.append(pos)
                lab.append(torch.full((pos.shape[0],), cond, dtype=torch.long))

            n = neg.shape[0]
            if n > 0:
                emb.append(neg)
                if self.destroyed_label_mode == "unique":
                    lab.append(torch.arange(next_uniq, next_uniq + n, dtype=torch.long))
                    next_uniq += n
                else:
                    lab.append(torch.full((n,), self.shared_destroyed_label, dtype=torch.long))

            metas.append(item["meta"])

        if not emb:
            raise ValueError(
                "TripletCollator produced no rows to embed: every window in the "
                "batch had empty positive AND negative pools. Check n_positives / "
                "n_negatives.")

        X = torch.cat(emb, dim=0).to(torch.float32)
        y = torch.cat(lab, dim=0).to(torch.long)
        return X, y, metas
