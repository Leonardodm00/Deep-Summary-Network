"""
config.py
=========

Single source of truth for the Topic-3 optimization / training / evaluation
pipeline. This module contains DATA ONLY: dataclasses plus their JSON
(de)serialization. It has no training logic, no model building, no plotting, and
no search construction (separation of concerns -- directive 2). Every downstream
stage (data splits, metrics, trainer, evaluator, search harness, driver) reads
its settings from an ExperimentConfig instance.

Design points
-------------
  * The already-tested Topic-1 AugmentationConfig and Topic-2 BackboneConfig are
    NESTED and imported unchanged (directive 1: reuse tested code; no field
    duplication, no drift). This module never redefines them.
  * AugmentationConfig requires a sampling rate fs at construction. In the config
    the stored fs is a PLACEHOLDER: the real fs is resolved at dataset-build time
    (synthetic: DataConfig.synthetic_fs; real / numpy: from the trace loader) via
    DataConfig.resolved_augmentation(fs), which returns a copy with fs replaced.
  * Serialization round-trips EXACTLY. JSON turns tuples into lists, so loading
    coerces list-valued fields back to tuples using each field's declared type.
    This is what makes  ExperimentConfig == ExperimentConfig.from_json(path)  hold
    (a tuple never equals a list), which the smoke test asserts directly.

Architecture search space vs the Topic-2 backbone
-------------------------------------------------
  The legacy driver searched (d, wm, blk, ws, es). The Topic-2 backbone REPLACED
  the width-shrink head with a multi-scale fusion head, so 'ws' (width_shrink) has
  NO analog and is intentionally dropped. group_width and the head options
  (head_fusion, head_pool_ops, head_prenorm) are held FIXED via BackboneConfig
  defaults but remain configurable. The searched architecture HPs are therefore:
  depth_exponent (Integer), width_multiplier (Real, aligned to the backbone's
  documented [1.5, 3.0]), block_family (Categorical over {0, 1}), embedding_size
  (Integer over {8..16}).

HPC note (hpc-python-compat): this file is pure ASCII. Its import chain
(backbone.py, augmentation.py) is pure ASCII as well.
"""

import json
import warnings
from dataclasses import dataclass, field, fields, is_dataclass, asdict, replace
from pathlib import Path
from typing import Tuple, get_type_hints, get_origin, get_args

from backbone import BackboneConfig
from augmentation import AugmentationConfig

__all__ = [
    "DataConfig",
    "TrainConfig",
    "SearchConfig",
    "RegularizationConfig",
    "EvalConfig",
    "RuntimeConfig",
    "ExperimentConfig",
    "config_from_dict",
    # re-exported for downstream convenience
    "BackboneConfig",
    "AugmentationConfig",
]

# Placeholder sampling rate stored inside the nested AugmentationConfig.
# It is ALWAYS overwritten at dataset-build time via resolved_augmentation(fs).
_PLACEHOLDER_FS = 50.0


# --------------------------------------------------------------------------- #
# JSON <-> dataclass reconstruction (handles nested dataclasses + tuple fields)
# --------------------------------------------------------------------------- #
def _coerce(ftype, value):
    """Coerce a JSON-loaded value to the declared field type.

    Handles: nested dataclasses (recurse), Tuple[...] (list -> tuple, both the
    fixed Tuple[a, b] and the variadic Tuple[a, ...] forms), List[...] (element
    coercion), and primitives (pass through).
    """
    if value is None:
        return None
    if isinstance(ftype, type) and is_dataclass(ftype):
        if isinstance(value, dict):
            return config_from_dict(ftype, value)
        return value
    origin = get_origin(ftype)
    if origin is tuple:
        args = get_args(ftype)
        seq = list(value)
        if len(args) == 2 and args[1] is Ellipsis:      # Tuple[a, ...]
            elem_t = args[0]
            return tuple(_coerce(elem_t, v) for v in seq)
        if args:                                         # Tuple[a, b, c]
            return tuple(_coerce(a, v) for a, v in zip(args, seq))
        return tuple(seq)
    if origin is list:
        args = get_args(ftype)
        elem_t = args[0] if args else None
        if elem_t is None:
            return list(value)
        return [_coerce(elem_t, v) for v in value]
    return value


def config_from_dict(cls, data):
    """Reconstruct a dataclass of type cls from a plain dict (as loaded from JSON).

    Missing keys fall back to the dataclass defaults; unknown keys are ignored
    with a warning (helps catch typos in hand-written config files).
    """
    if not (isinstance(cls, type) and is_dataclass(cls)):
        return data
    hints = get_type_hints(cls)
    field_names = {f.name for f in fields(cls)}
    unknown = [k for k in data.keys() if k not in field_names]
    if unknown:
        warnings.warn(
            "config_from_dict(%s): ignoring unknown keys %r"
            % (cls.__name__, unknown),
            RuntimeWarning,
        )
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        ftype = hints.get(f.name, f.type)
        kwargs[f.name] = _coerce(ftype, data[f.name])
    return cls(**kwargs)


# --------------------------------------------------------------------------- #
# Data: loading + windowing + splitting + augmentation params
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    """Data loading, windowing, time-segment splitting, and augmentation params."""

    # --- source ---
    data_mode: str = "synthetic"            # "synthetic" | "real" | "numpy"
    specs_json: str = ""                    # real mode: path to specs list
    npz_specs: str = ""                     # numpy mode: path to burst_specs.json

    # --- synthetic generation (multi-class capable) ---
    # one entry per phenotype class: number of synthetic traces for that class;
    # length of this tuple == number of classes C (labels 0..C-1).
    synthetic_n_per_class: Tuple[int, ...] = (2, 1)
    synthetic_duration_s: float = 600.0
    synthetic_fs: float = 50.0

    # --- windowing ---
    window_s: float = 200.0
    train_stride_s: float = 100.0           # < window_s -> overlapping train windows (diversity)
    eval_stride_s: float = 200.0            # >= window_s -> DISJOINT val / test windows (no dup)

    # --- time-segment split (option A): fractions along each trace's time axis ---
    split_fractions: Tuple[float, float, float] = (0.6, 0.2, 0.2)   # (train, val, test)
    drop_boundary_windows: bool = True      # drop windows straddling a split boundary

    # --- augmentation (fs is a PLACEHOLDER; resolved at build time) ---
    augmentation: AugmentationConfig = field(
        default_factory=lambda: AugmentationConfig(fs=_PLACEHOLDER_FS))

    def __post_init__(self):
        if self.data_mode not in ("synthetic", "real", "numpy"):
            raise ValueError("data_mode must be 'synthetic', 'real', or 'numpy'")
        if len(self.synthetic_n_per_class) < 1:
            raise ValueError("synthetic_n_per_class must have at least one class")
        if any(int(n) < 1 for n in self.synthetic_n_per_class):
            raise ValueError("each synthetic_n_per_class entry must be >= 1")
        if self.synthetic_duration_s <= 0 or self.synthetic_fs <= 0:
            raise ValueError("synthetic_duration_s and synthetic_fs must be > 0")
        if self.window_s <= 0 or self.train_stride_s <= 0 or self.eval_stride_s <= 0:
            raise ValueError("window_s and strides must be > 0")
        if len(self.split_fractions) != 3:
            raise ValueError("split_fractions must be (train, val, test)")
        if any((fr <= 0.0 or fr >= 1.0) for fr in self.split_fractions):
            raise ValueError("each split fraction must lie strictly in (0, 1)")
        if abs(sum(self.split_fractions) - 1.0) > 1e-6:
            raise ValueError("split_fractions must sum to 1.0")
        if self.eval_stride_s < self.window_s:
            warnings.warn(
                "eval_stride_s < window_s: evaluation windows will OVERLAP, which "
                "can inflate apparent sample size. Set eval_stride_s >= window_s "
                "for disjoint eval windows.",
                RuntimeWarning,
            )

    def resolved_augmentation(self, fs):
        """Copy of the augmentation config with fs set to the runtime value."""
        return replace(self.augmentation, fs=float(fs))


# --------------------------------------------------------------------------- #
# Trainer: settings shared identically by the HPO objective and the final run
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """Trainer settings used identically by the HPO objective and the final run."""

    # --- loss / miner ---
    margin: float = 0.3                     # loss margin m (searched in phase 2; default / fixed value)
    swap: bool = True                       # TripletMarginLoss swap
    mining_strategy: str = "hard"           # "hard" | "easy_positive"

    # --- batching (ConditionBalancedBatchSampler; added in Stage 5) ---
    # windows_per_condition = B_c: windows drawn from EACH phenotype class per
    # batch, so a batch holds exactly C * B_c source windows (C = #classes) and
    # every batch supports cross-condition triplets by construction.
    # batches_per_epoch = n_batches: 0 means DERIVE it at trainer build time as
    #     n_batches = ceil( N_train_windows / (C * B_c) ),
    # i.e. one nominal pass over the training windows per epoch. Any value >= 1
    # overrides that derivation with a fixed batch count.
    windows_per_condition: int = 8
    batches_per_epoch: int = 0

    # --- optimizer (AdamW). Also the deliberate FIXED optimizer for phase-1 arch search ---
    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 1e-4

    # --- budget / early stopping (see the derived rule in the design notes) ---
    max_epochs: int = 100                   # E_max: hard ceiling
    patience: int = 10                      # P: consecutive no-improvement epochs before stopping
    min_delta_ari: float = 0.0              # delta: min ARI improvement to reset patience
    min_delta_sil: float = 0.0              # epsilon: min silhouette improvement on an ARI plateau
    selection_primary: str = "ari"          # "ari" (default) | "silhouette"; the other breaks ties
    n_seeds: int = 3                        # trainings per config; objective returns mean val metric

    # --- optional accelerators (off by default) ---
    use_scheduler: bool = False
    scheduler_type: str = "cosine"          # "cosine" | "step" | "none"; used only if use_scheduler
    use_amp: bool = False                   # GPU-only; guarded at runtime

    # --- logging / checkpoint cadence ---
    log_every_epochs: int = 1
    checkpoint_every_epochs: int = 5

    def __post_init__(self):
        if self.mining_strategy not in ("hard", "easy_positive"):
            raise ValueError("mining_strategy must be 'hard' or 'easy_positive'")
        if self.margin <= 0.0:
            raise ValueError("margin must be > 0")
        if self.lr <= 0.0:
            raise ValueError("lr must be > 0")
        if not (0.0 < self.beta1 < 1.0) or not (0.0 < self.beta2 < 1.0):
            raise ValueError("beta1 and beta2 must lie in (0, 1)")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be >= 0")
        if self.max_epochs < 1 or self.patience < 1 or self.n_seeds < 1:
            raise ValueError("max_epochs, patience, n_seeds must be >= 1")
        if self.windows_per_condition < 1:
            raise ValueError("windows_per_condition must be >= 1")
        if self.batches_per_epoch < 0:
            raise ValueError(
                "batches_per_epoch must be >= 0 (0 -> derive from the training set size)")
        if self.selection_primary not in ("ari", "silhouette"):
            raise ValueError("selection_primary must be 'ari' or 'silhouette'")
        if self.scheduler_type not in ("cosine", "step", "none"):
            raise ValueError("scheduler_type must be 'cosine', 'step', or 'none'")
        if self.log_every_epochs < 1 or self.checkpoint_every_epochs < 1:
            raise ValueError("log / checkpoint cadences must be >= 1")
        if self.max_epochs <= self.patience:
            warnings.warn(
                "max_epochs <= patience: early stopping can never fire, so training "
                "is effectively fixed-length at max_epochs.",
                RuntimeWarning,
            )


# --------------------------------------------------------------------------- #
# Search: two-phase gp_minimize ranges + meta-settings (skopt, sequential)
# --------------------------------------------------------------------------- #
@dataclass
class SearchConfig:
    """Two-phase gp_minimize search ranges and meta-settings (skopt, sequential).

    Phase 1 (architecture) is searched under the FIXED optimizer in TrainConfig.
    Phase 2 betas are searched as (1 - beta) in log-space (ranges below). ws
    (width_shrink) from the legacy driver is intentionally absent: the Topic-2
    fusion head has no such knob. group_width and head options stay fixed via
    BackboneConfig defaults.
    """

    # --- phase 1: architecture ---
    depth_exponent_range: Tuple[int, int] = (3, 6)              # Integer d
    width_multiplier_range: Tuple[float, float] = (1.5, 3.0)    # Real wm (backbone: continuous)
    block_family_choices: Tuple[int, ...] = (0, 1)             # Categorical blk (0=ResNet, 1=ResNeXt)
    embedding_size_range: Tuple[int, int] = (8, 16)            # Integer es

    # --- phase 2: training HPs ---
    margin_range: Tuple[float, float] = (0.1, 1.0)             # Real m
    lr_range: Tuple[float, float] = (1e-4, 0.2)               # log-uniform
    one_minus_beta1_range: Tuple[float, float] = (1e-2, 1e-1)  # log-uniform -> b1 in [0.9, 0.99]
    one_minus_beta2_range: Tuple[float, float] = (1e-4, 1e-2)  # log-uniform -> b2 in [0.99, 0.9999]
    weight_decay_range: Tuple[float, float] = (1e-4, 1e-2)     # log-uniform

    # --- meta ---
    n_calls_arch: int = 100
    n_calls_train: int = 100
    gp_random_state: int = 0
    do_refine: bool = False                 # coarse-to-fine second pass (get_newspace)
    refine_top_fraction: float = 0.10       # fraction of best points used to narrow the box
    do_retune_arch: bool = False            # optional architecture re-tune after phase 2

    def __post_init__(self):
        def _check(name, r, positive=False, gt_one=False):
            if len(r) != 2:
                raise ValueError("%s must be a (low, high) pair" % name)
            lo, hi = r
            if lo > hi:
                raise ValueError("%s: low must be <= high" % name)
            if positive and lo <= 0:
                raise ValueError("%s: low must be > 0 for a log-uniform range" % name)
            if gt_one and lo <= 1.0:
                raise ValueError("%s: low must be > 1.0" % name)

        _check("depth_exponent_range", self.depth_exponent_range)
        if self.depth_exponent_range[0] < 1:
            raise ValueError("depth_exponent_range low must be >= 1")
        _check("width_multiplier_range", self.width_multiplier_range, gt_one=True)
        _check("embedding_size_range", self.embedding_size_range)
        if self.embedding_size_range[0] < 1:
            raise ValueError("embedding_size_range low must be >= 1")
        if len(self.block_family_choices) < 1 or any(
                b not in (0, 1) for b in self.block_family_choices):
            raise ValueError("block_family_choices must be a non-empty subset of {0, 1}")
        _check("margin_range", self.margin_range, positive=True)
        _check("lr_range", self.lr_range, positive=True)
        _check("one_minus_beta1_range", self.one_minus_beta1_range, positive=True)
        _check("one_minus_beta2_range", self.one_minus_beta2_range, positive=True)
        _check("weight_decay_range", self.weight_decay_range, positive=True)
        if self.one_minus_beta1_range[1] >= 1.0 or self.one_minus_beta2_range[1] >= 1.0:
            raise ValueError(
                "one_minus_beta*_range high must be < 1.0 (beta = 1 - x must stay > 0)")
        if self.n_calls_arch < 1 or self.n_calls_train < 1:
            raise ValueError("n_calls_arch and n_calls_train must be >= 1")
        if not (0.0 < self.refine_top_fraction <= 1.0):
            raise ValueError("refine_top_fraction must lie in (0, 1]")


# --------------------------------------------------------------------------- #
# Final regularization stage: search dropout + weight_decay on validation
# --------------------------------------------------------------------------- #
@dataclass
class RegularizationConfig:
    """Final regularization stage: search dropout and weight_decay on validation."""

    dropout_range: Tuple[float, float] = (0.0, 0.3)            # Real
    weight_decay_range: Tuple[float, float] = (1e-5, 1e-2)     # log-uniform
    n_calls: int = 50
    gp_random_state: int = 0

    def __post_init__(self):
        lo, hi = self.dropout_range
        if not (0.0 <= lo <= hi < 1.0):
            raise ValueError("dropout_range must satisfy 0 <= low <= high < 1")
        wlo, whi = self.weight_decay_range
        if not (0.0 < wlo <= whi):
            raise ValueError("weight_decay_range must satisfy 0 < low <= high")
        if self.n_calls < 1:
            raise ValueError("n_calls must be >= 1")


# --------------------------------------------------------------------------- #
# Evaluation / scoring (metrics + embedding plot)
# --------------------------------------------------------------------------- #
@dataclass
class EvalConfig:
    """Evaluation / scoring settings (clustering metrics + embedding plot)."""

    kmeans_seed: int = 0
    kmeans_n_init: int = 10
    silhouette_metric: str = "cosine"       # matches the cosine loss geometry
    pca_components: int = 2                  # for the saved embedding scatter (display only)

    def __post_init__(self):
        if self.kmeans_n_init < 1:
            raise ValueError("kmeans_n_init must be >= 1")
        if self.pca_components < 1:
            raise ValueError("pca_components must be >= 1")
        if not isinstance(self.silhouette_metric, str) or not self.silhouette_metric:
            raise ValueError("silhouette_metric must be a non-empty string")


# --------------------------------------------------------------------------- #
# Runtime: HPC, reproducibility, device, IO
# --------------------------------------------------------------------------- #
@dataclass
class RuntimeConfig:
    """HPC runtime, reproducibility, and IO settings."""

    seed: int = 0                           # base seed; per-trial seeds derive from this
    device: str = "cpu"                     # "cpu" (default) | "cuda" | "auto"
    deterministic: bool = True
    torch_threads: int = 1                  # intra-op threads (avoid oversubscription with workers)
    num_workers: int = 2
    pin_memory: bool = False
    out_dir: str = "./optim_out"
    cache_dir: str = "./preproc_cache"
    experiment_name: str = "run"

    def __post_init__(self):
        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError("device must be 'cpu', 'cuda', or 'auto'")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.torch_threads < 1:
            raise ValueError("torch_threads must be >= 1")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")


# --------------------------------------------------------------------------- #
# Top-level experiment config: the single object every stage reads from
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentConfig:
    """Top-level configuration aggregating every sub-config."""

    data: DataConfig = field(default_factory=DataConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # ----- serialization -----
    def to_dict(self):
        return asdict(self)

    def to_json(self, path):
        p = Path(path)
        if not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        # ensure_ascii=True (json default) keeps the artifact HPC-safe on disk.
        with open(p, "w", encoding="ascii") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return p

    @classmethod
    def from_dict(cls, data):
        return config_from_dict(cls, data)

    @classmethod
    def from_json(cls, path):
        # Read as UTF-8 so hand-written configs with accents still load; our own
        # writer always emits ASCII.
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    # ----- cross-field soft validation (warnings, not errors) -----
    def validate(self):
        msgs = []
        if self.train.max_epochs <= self.train.patience:
            msgs.append("train.max_epochs <= train.patience (early stopping cannot fire).")
        if self.data.eval_stride_s < self.data.window_s:
            msgs.append("data.eval_stride_s < data.window_s (eval windows overlap).")
        lo, hi = self.search.embedding_size_range
        if not (lo <= self.backbone.embedding_size <= hi):
            msgs.append(
                "backbone.embedding_size=%d is outside search.embedding_size_range=%s "
                "(fine if the final es is set by the search)."
                % (self.backbone.embedding_size, (lo, hi)))
        for m in msgs:
            warnings.warn(m, RuntimeWarning)
        return msgs
