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
from typing import (Dict, List, Optional, Tuple, get_type_hints,
                    get_origin, get_args)

from backbone import BackboneConfig
from augmentation import AugmentationConfig

__all__ = [
    "DataConfig",
    "SyntheticConfig",
    "SyntheticClassOverride",
    "LatentConfig",
    "LatentAxisOverride",
    "TrainConfig",
    "SearchConfig",
    "RegularizationConfig",
    "EvalConfig",
    "RuntimeConfig",
    "CohortConfig",
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
# Synthetic burst-generator parameters (MultiClassSyntheticProvider)
# --------------------------------------------------------------------------- #
@dataclass
class SyntheticClassOverride:
    """Per-class override of the swept synthetic burst-generator parameters.

    Every field defaults to None, meaning "do not override -- use the value the
    global linear sweep assigns to this class". Only the non-None fields take
    effect. A class index c that has no entry in SyntheticConfig.per_class (or an
    entry with all-None fields) is generated purely from the global sweep.

    Fields
    ------
    rate    : bursts/second for this class (overrides swept rate(c)).
    width   : base Gaussian burst width in seconds (overrides swept width(c)).
    amp_min : lower bound of per-burst amplitude jitter U(amp_min, amp_max).
    amp_max : upper bound of per-burst amplitude jitter U(amp_min, amp_max).

    Amplitude note: class 0 normally uses a FIXED amplitude a=1.0 (no jitter).
    Supplying amp_min/amp_max for class 0 PROMOTES it to jittered amplitude like
    any other class -- this is intentional and lets you break the class-0 special
    case if you want all classes on equal footing. amp_min and amp_max must be
    supplied together (both None, or both set); setting only one is an error.
    """

    rate: float = None
    width: float = None
    amp_min: float = None
    amp_max: float = None

    def __post_init__(self):
        if self.rate is not None and self.rate <= 0:
            raise ValueError("SyntheticClassOverride.rate must be > 0 if set")
        if self.width is not None and self.width <= 0:
            raise ValueError("SyntheticClassOverride.width must be > 0 if set")
        if (self.amp_min is None) != (self.amp_max is None):
            raise ValueError(
                "SyntheticClassOverride: amp_min and amp_max must be set together "
                "(both None or both provided)")
        if self.amp_min is not None:
            if self.amp_min <= 0 or self.amp_max <= 0:
                raise ValueError("amp_min and amp_max must be > 0 if set")
            if self.amp_min > self.amp_max:
                raise ValueError("require amp_min <= amp_max")


@dataclass
class SyntheticConfig:
    """Global burst-generator shape parameters for MultiClassSyntheticProvider,
    plus optional per-class overrides.

    The provider assigns each class c in {0..C-1} a sweep fraction
        frac(c) = c / (C - 1)      (and frac(0) = 0 when C == 1),
    then, for the GLOBAL sweep (any class without an override):
        rate(c)  = rate_min  + (rate_max  - rate_min ) * frac(c)   [bursts/s]
        width(c) = width_max - (width_max - width_min) * frac(c)   [seconds]
    i.e. higher class index -> denser (higher rate) and narrower base width.
    Per-burst amplitude jitter for classes c != 0 is U(amp_jitter_min,
    amp_jitter_max); class 0 uses fixed a=1.0 unless a per-class amp override is
    given for it.

    per_class[c] (if present and with non-None fields) overrides the swept
    rate/width and/or the amplitude jitter bounds for class c only. per_class may
    be shorter than C: missing trailing classes fall back to the global sweep.
    """

    rate_min: float = 0.25
    rate_max: float = 0.55
    width_min: float = 0.15
    width_max: float = 0.70
    amp_jitter_min: float = 0.60
    amp_jitter_max: float = 1.40
    per_class: Tuple[SyntheticClassOverride, ...] = ()

    def __post_init__(self):
        if not (0 < self.rate_min <= self.rate_max):
            raise ValueError("require 0 < rate_min <= rate_max")
        if not (0 < self.width_min <= self.width_max):
            raise ValueError("require 0 < width_min <= width_max")
        if not (0 < self.amp_jitter_min <= self.amp_jitter_max):
            raise ValueError("require 0 < amp_jitter_min <= amp_jitter_max")


# --------------------------------------------------------------------------- #
# Latent-factor generator parameters (LatentBurstProvider)   [C1]
# --------------------------------------------------------------------------- #
@dataclass
class LatentAxisOverride:
    """Per-axis override of a latent axis's PHYSICAL range [a_k, b_k].

    Every field except name defaults to None, meaning "keep the canonical value
    from latent_burst_generator.AXIS_REGISTRY". This is the calibration hook and
    it exists for a specific reason: the canonical ranges are NOT independently
    sourced. They were chosen to bracket the CONTROL_PARAMS / PATHO_PARAMS values
    already present in generate_burst_data.py, whose own provenance is
    undocumented. The literature grounds the IDENTITY and the DIRECTION of the
    axes, not the numbers. Refitting [a_k, b_k] to real recordings must therefore
    be possible from a config file, without editing code.

    Fields
    ------
    name        : which axis to override; must be one of LatentConfig.axis_names.
    lo, hi      : new range endpoints a_k, b_k in the axis's own physical units
                  (see LatentAxis.units). Require lo < hi when both are given.
    orientation : s_k in {+1, -1}; -1 makes phi_k = 1 map to the LOW end.
    """

    name: str = ""
    lo: float = None
    hi: float = None
    orientation: int = None

    def __post_init__(self):
        if not str(self.name):
            raise ValueError("LatentAxisOverride.name must name an axis")
        if self.orientation is not None and int(self.orientation) not in (1, -1):
            raise ValueError("LatentAxisOverride.orientation must be +1 or -1")
        if (self.lo is not None and self.hi is not None
                and not (float(self.lo) < float(self.hi))):
            raise ValueError(
                "LatentAxisOverride(%r): require lo < hi; got (%r, %r)"
                % (self.name, self.lo, self.hi))


@dataclass
class LatentConfig:
    """The n-latent-factor phenotype space (data_mode == "latent").   [C1]

    What this buys over data_mode == "synthetic": the synthetic provider drives
    BOTH of its degrees of freedom from the single scalar frac = c / (C - 1), so
    the true data manifold is one-dimensional and the C classes are three points
    on a line. Two consequences, both fatal for model selection: the task is
    solvable by one hand-crafted scalar (validation ARI saturates at 1.0 and the
    search objective becomes constant), and eff_rank ~= 1 is simultaneously the
    CORRECT answer and the signature of representation collapse, so the collapse
    tripwire cannot fire. Here the latent dimensionality, the class overlap, and
    the label-relevance of each factor are explicit and tunable.

    WHAT THIS BLOCK DOES NOT CONTAIN, and why. C, T_rec and f_s are NOT
    duplicated here: latent mode reads them from the fields that already exist
    and are already fingerprinted --
        C       = len(DataConfig.synthetic_n_per_class)
        n_c     =     DataConfig.synthetic_n_per_class[c]
        T_rec   =     DataConfig.synthetic_duration_s   [s]
        f_s     =     DataConfig.synthetic_fs           [Hz]  (w_size = 1 / f_s)
    Two sources of truth for a sampling rate is a bug waiting to happen, and
    those three fields already enter _data_fingerprint, so reusing them means
    only the genuinely new parameters below need adding to it.

    n is likewise NOT a field: n = len(axis_names) by construction, exposed as
    the read-only property n_latent. A separate n_latent field could disagree
    with the axis list, and there is no useful behaviour for it to have when it
    does.

    Attributes
    ----------
    axis_names    : ordered names of the latent factors, one per axis k. The
                    ORDER defines the axis indices, so label_axes refers to this
                    ordering. Names must exist in
                    latent_burst_generator.AXIS_REGISTRY; the six canonical ones
                    are irregularity, burst_rate, burst_duration,
                    intraburst_rate, participation, background.
    label_axes    : S, the 0-based indices into axis_names that CARRY the class
                    label. Axes not in S are label-IRRELEVANT but physically real
                    variation -- the analogue of biological variation the
                    phenotype label does not name, and the axes whose retention
                    the factor-retention metric measures.
                    DEFAULT (0, 1) = irregularity + burst_rate, and this is
                    CALIBRATED, not arbitrary: a single label axis on
                    irregularity alone reached only ARI ~= 0.17 even at tau = 0,
                    because at lambda_b in [0.10, 0.40] bursts/s a 30 s window
                    holds only ~3-12 bursts and a duration CV cannot be estimated
                    from so few. Do not revert to one label axis without
                    re-running that calibration.
    class_center_mode : where the class centres sit in the label subspace.
                    "simplex" (DEFAULT) gives each class a DIFFERENT centre on
                    EACH label axis, placed at the vertices of a regular
                    simplex. The other two modes return ONE SCALAR per class and
                    replicate it across every label axis, so all C centres lie
                    on the diagonal and rank Cov({mu_c}) = 1 whatever C and L
                    are -- the centres are collinear, and any objective asking
                    for an ARRANGEMENT of centres (the NC2 separation term) is
                    asking for a geometry the data does not contain. "simplex"
                    reaches the maximum achievable rank min(L, C-1) with
                    pairwise cosine -1/(C-1). NOTE the centre-to-centre spacing
                    differs from "interior", so class_overlap (tau) should be
                    re-measured rather than carried over.
                    "interior" places them at m_c = (c+1)/(C+1), so at
                    C = 3 they are 0.25, 0.50, 0.75 and none touches a boundary
                    of [0, 1]. "endpoints" reproduces the original
                    m_c = c/(C-1) = 0, 0.5, 1, whose OUTER centres sit exactly
                    on the boundaries, so the clip in the latent construction
                    pins ~50% of their draws at 0 or 1 -- MEASURED at 50.6% and
                    48.4% for C = 3, invariant in tau -- leaving the outer
                    classes ~42% tighter than the middle one. Use "endpoints"
                    only to reproduce a run made before this option existed.
                    CHANGING THIS CHANGES EVERY GENERATED TRACE (and so the
                    cache fingerprint, deliberately).
    class_overlap : tau >= 0, the spread of a trace's label coordinates about its
                    class centre m_c = c / (C - 1), in normalized latent units.
                    tau = 0 gives deterministic class centres; larger tau makes
                    the classes overlap. This is THE task-difficulty knob.
    n_neurons     : N, neurons per simulated culture.
    gaussian_window : IFR smoothing sd [s].
    axis_overrides  : optional per-axis range recalibration (see above).
    """

    axis_names: Tuple[str, ...] = (
        "irregularity", "burst_rate", "burst_duration",
        "intraburst_rate", "participation", "background")
    label_axes: Tuple[int, ...] = (0, 1)
    class_overlap: float = 0.10
    class_center_mode: str = "simplex"
    n_neurons: int = 100
    gaussian_window: float = 0.04
    axis_overrides: Tuple[LatentAxisOverride, ...] = ()

    def __post_init__(self):
        names = tuple(str(nm) for nm in self.axis_names)
        if len(names) < 1:
            raise ValueError("latent.axis_names must list at least one axis")
        if len(set(names)) != len(names):
            raise ValueError("latent.axis_names contains duplicates: %r" % (names,))
        n = len(names)
        if len(self.label_axes) < 1:
            raise ValueError(
                "latent.label_axes must be non-empty: with S empty no axis "
                "carries the class label and the task is unlearnable")
        for k in self.label_axes:
            if not (0 <= int(k) < n):
                raise ValueError(
                    "latent.label_axes index %r out of range [0, %d) for "
                    "axis_names %r" % (k, n, names))
        if len(set(int(k) for k in self.label_axes)) != len(self.label_axes):
            raise ValueError("latent.label_axes contains duplicates")
        if len(self.label_axes) == n:
            warnings.warn(
                "latent: EVERY axis carries the class label (S = all axes), so "
                "there are no label-irrelevant factors left. The factor-retention "
                "metric has nothing to measure and eff_rank recovers its "
                "ambiguity as a collapse tripwire.",
                RuntimeWarning)
        if self.class_overlap < 0.0:
            raise ValueError("latent.class_overlap (tau) must be >= 0")
        if self.class_center_mode not in ("simplex", "interior", "endpoints"):
            raise ValueError(
                "latent.class_center_mode must be 'simplex', 'interior' or "
                "'endpoints'; got %r" % (self.class_center_mode,))
        if int(self.n_neurons) < 1:
            raise ValueError("latent.n_neurons must be >= 1")
        if float(self.gaussian_window) <= 0.0:
            raise ValueError("latent.gaussian_window must be > 0")
        for i, ov in enumerate(self.axis_overrides):
            if str(ov.name) not in names:
                raise ValueError(
                    "latent.axis_overrides[%d] names axis %r, which is not among "
                    "axis_names %r" % (i, ov.name, names))

    @property
    def n_latent(self) -> int:
        """n, the number of latent factors. DERIVED: n = len(axis_names)."""
        return len(self.axis_names)

    @property
    def free_axes(self) -> Tuple[int, ...]:
        """Indices k not in S: the label-IRRELEVANT factors."""
        label = set(int(k) for k in self.label_axes)
        return tuple(k for k in range(self.n_latent) if k not in label)


# --------------------------------------------------------------------------- #
# Data: loading + windowing + splitting + augmentation params
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    """Data loading, windowing, time-segment splitting, and augmentation params."""

    # --- source ---
    data_mode: str = "synthetic"            # "synthetic" | "real" | "numpy" | "latent"
    specs_json: str = ""                    # real mode: path to specs list
    npz_specs: str = ""                     # numpy mode: path to burst_specs.json

    # --- channels (multichannel) ---
    # n_channels is the SINGLE source of truth for the trace channel axis
    # (the C in the shape contract  trace (C, T) -> window (C, W) -> (M, C, W)).
    # This is DISTINCT from the number of phenotype classes
    # (len(synthetic_n_per_class)); do NOT conflate the two.
    #   n_channels == 1 : single-channel population IFR (default, fully backward
    #                     compatible; windows are (W,) and the 1-D path is used).
    #   n_channels >  1 : per-region / per-channel IFRs; windows are (C, W).
    # ExperimentConfig.__post_init__ propagates this into
    # backbone.in_channels, so the backbone stem and the data always agree.
    n_channels: int = 1

    # --- synthetic generation (multi-class capable) ---
    # one entry per phenotype class: number of synthetic traces for that class;
    # length of this tuple == number of classes C (labels 0..C-1).
    # ALSO used by data_mode == "latent" (see LatentConfig): C, n_c, T_rec and
    # f_s are shared rather than duplicated.
    synthetic_n_per_class: Tuple[int, ...] = (2, 1)
    synthetic_duration_s: float = 600.0
    synthetic_fs: float = 50.0
    # burst-generator shape params (rate/width sweep, amplitude jitter, and
    # optional per-class overrides). Consumed by build_traces() when
    # data_mode == "synthetic". Ignored for real/numpy/latent modes.
    synthetic: "SyntheticConfig" = field(default_factory=lambda: SyntheticConfig())
    # [C1] n-latent-factor generator params. Consumed by build_traces() when
    # data_mode == "latent". Ignored for synthetic/real/numpy modes. It is a
    # SEPARATE mode rather than a mutation of "synthetic" on purpose: several
    # existing smoke tests and config_toy.json exercise "synthetic", so changing
    # its semantics would silently alter what those tests mean. A new mode is
    # additive and leaves every existing test meaningful.
    latent: "LatentConfig" = field(default_factory=lambda: LatentConfig())
    # data_mode == "synthetic". Ignored for real/numpy modes.
    synthetic: "SyntheticConfig" = field(default_factory=lambda: SyntheticConfig())

    # --- windowing ---
    window_s: float = 200.0
    train_stride_s: float = 100.0           # < window_s -> overlapping train windows (diversity)
    eval_stride_s: float = 200.0            # >= window_s -> DISJOINT val / test windows (no dup)

    # --- split: fractions along each trace's time axis, or whole cultures ---
    split_fractions: Tuple[float, float, float] = (0.6, 0.2, 0.2)   # (train, val, test)
    drop_boundary_windows: bool = True      # drop windows straddling a split boundary

    # split_mode selects WHICH splitter run_optimization builds:
    #   "time_segment" -- data_splits.make_time_segment_splits (the default, and
    #                     what every archived result used). Cuts each trace's
    #                     time axis; no WINDOW straddles a boundary, but every
    #                     culture contributes windows to all three splits, so
    #                     culture identity remains exploitable.
    #   "trace"        -- data_splits.make_trace_splits. A culture belongs to
    #                     exactly ONE split, stratified by class. This is the
    #                     split that matches the deployment question "classify a
    #                     culture the network has never seen".
    # Changing this INVALIDATES every quantity measured under the other mode:
    # N_train/N_val/N_eval, batches_per_epoch, the ARI floor, and Delta_min.
    split_mode: str = "time_segment"
    # only consulted when split_mode == "trace":
    trace_split_mode: str = "fractional"    # "fractional" | "leave_one_out"
    trace_split_fold: int = 0               # which LOO fold; ignored otherwise
    trace_split_seed: int = 0               # permutation seed, SEPARATE from
                                            # runtime.seed so that seed-averaging
                                            # does not reshuffle the split
    min_train_cultures_per_class: int = 2   # 2 is the minimum at which
                                            # cross-culture positives exist
    trace_alloc_rule: str = "largest_remainder"   # | "floor" (see apportion())

    # --- [C4] cross-culture positives ---------------------------------------
    # Default "augmentation" keeps this change INERT until switched on, matching
    # the posture of every other change on this branch (D1-R12).
    positives_mode: str = "augmentation"    # "augmentation" | "cross_culture"
    # U_c, the number of DISTINCT training cultures of each class drawn per
    # batch, BEFORE the availability clamp of Eq. (3). The clamp is applied at
    # sampler construction, where the actual culture counts are known; this
    # field is only the request.
    cultures_per_class_per_batch: int = 12
    windows_per_culture_per_batch: int = 1  # q
    # Forbid g_i == g_j for an anchor-positive pair. With U_c = 2 and q = 1 every
    # same-class pair is already cross-culture and this is vacuous; it bites for
    # q > 1, which is exactly when within-culture pairs reappear.
    exclude_same_culture_positives: bool = True
    # n_g^max, the cap on rows sharing a class label per class per batch. NOTE
    # the source cited for a cap of 14 supports 16, and its MECHANISM does not
    # transfer to this setting -- do not re-derive 14 from that paper. The cap's
    # value here is as a REGRESSION GUARD: it turns a later config edit that
    # reintroduces a large degenerate group into an immediate exception.
    max_group_size: int = 16

    # --- augmentation (fs is a PLACEHOLDER; resolved at build time) ---
    augmentation: AugmentationConfig = field(
        default_factory=lambda: AugmentationConfig(fs=_PLACEHOLDER_FS))

    def __post_init__(self):
        if self.data_mode not in ("synthetic", "real", "numpy", "latent"):
            raise ValueError(
                "data_mode must be 'synthetic', 'real', 'numpy', or 'latent'")
        if len(self.synthetic_n_per_class) < 1:
            raise ValueError("synthetic_n_per_class must have at least one class")
        if any(int(n) < 1 for n in self.synthetic_n_per_class):
            raise ValueError("each synthetic_n_per_class entry must be >= 1")
        if self.synthetic_duration_s <= 0 or self.synthetic_fs <= 0:
            raise ValueError("synthetic_duration_s and synthetic_fs must be > 0")
        # per-class overrides may not name a class index beyond the declared C
        n_classes = len(self.synthetic_n_per_class)
        if len(self.synthetic.per_class) > n_classes:
            raise ValueError(
                "synthetic.per_class has %d entries but there are only %d classes "
                "(synthetic_n_per_class); per_class[c] maps to class c"
                % (len(self.synthetic.per_class), n_classes))
        if int(self.n_channels) < 1:
            raise ValueError("n_channels must be >= 1")
        if self.window_s <= 0 or self.train_stride_s <= 0 or self.eval_stride_s <= 0:
            raise ValueError("window_s and strides must be > 0")
        if self.split_mode not in ("time_segment", "trace"):
            raise ValueError(
                "split_mode must be 'time_segment' or 'trace'; got %r"
                % (self.split_mode,))
        if self.trace_split_mode not in ("fractional", "leave_one_out"):
            raise ValueError(
                "trace_split_mode must be 'fractional' or 'leave_one_out'; "
                "got %r" % (self.trace_split_mode,))
        if self.trace_alloc_rule not in ("largest_remainder", "floor"):
            raise ValueError(
                "trace_alloc_rule must be 'largest_remainder' or 'floor'; "
                "got %r" % (self.trace_alloc_rule,))
        if self.trace_split_fold < 0:
            raise ValueError("trace_split_fold must be >= 0")
        if self.min_train_cultures_per_class < 1:
            raise ValueError("min_train_cultures_per_class must be >= 1")
        # --- [C4] cross-culture positives: validation ----------------------
        if self.positives_mode not in ("augmentation", "cross_culture"):
            raise ValueError(
                "positives_mode must be 'augmentation' or 'cross_culture'; "
                "got %r" % (self.positives_mode,))
        if self.cultures_per_class_per_batch < 1:
            raise ValueError("cultures_per_class_per_batch (U_c) must be >= 1")
        if self.windows_per_culture_per_batch < 1:
            raise ValueError("windows_per_culture_per_batch (q) must be >= 1")
        if self.max_group_size < 2:
            raise ValueError(
                "max_group_size must be >= 2: a group of one row per class "
                "contains no positive pair at all")
        if self.positives_mode == "cross_culture":
            # The group-size cap is NOT enforced here. It needs two things this
            # dataclass cannot see:
            #   - U_eff, the culture count after the Eq. (3) availability clamp,
            #     which depends on the split;
            #   - train.mining_strategy, which lives in TrainConfig.
            # The second is what gates it. The cap traces to an easy-positive
            # result: performance drops past a group size of 16 for easy-positive
            # mining, while the same experiments show hard-positive mining
            # behaving the OPPOSITE way. So the cap applies under
            # "easy_positive" and "easy_pos_semihard_neg", and must NOT be
            # applied under "hard", where it would import a constraint the source
            # gives no support for.
            # Enforcement therefore lives at sampler construction; see
            # data_pipeline. ExperimentConfig.validate() is not the place either:
            # it is documented as soft cross-field validation, warnings only.
            # Not silently ignored: D1-section 9 established that inconsistent
            # config values in this codebase fail quietly, which is why this
            # raises instead of overwriting n_positives to 0 on the user's behalf.
            if int(self.augmentation.n_positives) != 0:
                raise ValueError(
                    "positives_mode='cross_culture' requires "
                    "augmentation.n_positives == 0, because positives now come "
                    "from other cultures rather than from warps of the anchor's "
                    "own window; got %d. Set it to 0 explicitly."
                    % (self.augmentation.n_positives,))
        if self.split_mode == "trace" and self.min_train_cultures_per_class < 2:
            warnings.warn(
                "split_mode='trace' with min_train_cultures_per_class = %d: "
                "cross-culture positives need at least 2 training cultures per "
                "class, so a value below 2 will not support Change 4."
                % (self.min_train_cultures_per_class,), RuntimeWarning)
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
    # Which objective. ADDITIVE: "triplet" is the pre-existing behaviour and the
    # default, so every archived config reproduces byte-identically.
    #   "triplet"   : losses.TripletMarginLoss (margin searched, as before)
    #   "joint"     : dsn_joint_loss.JointTripletLoss -- margin hinge PLUS the
    #                 angular hinge, optionally on strict-semi-hard triplets.
    #                 margin is FIXED and angular_alpha_deg is searched instead:
    #                 both bind on the within/between ratio, so searching the
    #                 pair moves along a ridge (see the loss handoff, S3.4).
    #   "joint_sep" : the above PLUS the gated CentroidSeparationLoss.
    loss_type: str = "triplet"
    margin: float = 0.3                     # loss margin m, COSINE convention.
                                            # Searched under "triplet"; FIXED
                                            # under "joint" / "joint_sep".
    swap: bool = True                       # TripletMarginLoss swap
    # "hard"                  : TripletMarginMiner(type_of_triplets="hard") --
    #                           positives FARTHER than negatives (violating
    #                           triplets). The COLLAPSE-SEEKING choice: every
    #                           same-class pair is pulled together, so within-
    #                           class variance is driven down (NC1).
    # "easy_positive"         : BatchEasyHardMiner(pos=easy, neg=hard)
    # "easy_pos_semihard_neg" : BatchEasyHardMiner(pos=easy, neg=semihard)
    #
    # The two easy-positive strategies are ANTI-COLLAPSE by design: they require
    # only the CLOSEST same-class window to be near the anchor, which is exactly
    # what allows a class to spread over a manifold instead of contracting to a
    # point. That is the right choice when within-class structure must be
    # preserved and the WRONG one when collapse is the goal. preflight_config
    # warns when an easy-positive strategy is paired with the composite loss.
    mining_strategy: str = "hard"

    # --- composite objective (used only when loss_type != "triplet") ---
    # angular constraint alpha, in DEGREES. Equivalent to a silhouette floor
    # S >= 1 - 4 sin^2(alpha), which is why it is the parameter that is
    # searched. Chung and Lee's {30, 45, 60, 75} are VACUOUS on L2-normalised
    # embeddings: every one of them is already satisfied at this pipeline's
    # operating point, so the term would contribute no gradient.
    # SMALL alpha is the collapse-forcing direction: the floor rises
    # monotonically as alpha falls (5 deg -> S >= 0.970, 2 deg -> S >= 0.995),
    # and alpha -> 0 demands a/b -> 0, i.e. exact within-class collapse. Pair a
    # small alpha with mining_strategy="hard"; an easy-positive miner will fight
    # it.
    angular_alpha_deg: float = 18.0
    # strict semi-hard filter (Chung and Lee): keep a triplet only when the
    # negative sits inside the margin band from BOTH the anchor and the
    # positive. Acts on WHICH triplets are used; swap acts on how they are
    # SCORED, so the two are not substitutes.
    # DEFAULT FALSE, deliberately. The filter requires D_ap < D_an, which is
    # the exact complement of what mining_strategy="hard" (the default miner)
    # produces, so defaulting it True would make TrainConfig(loss_type="joint")
    # an error out of the box -- and, before the guard below existed, a silent
    # zero-gradient run. Enable it together with
    # mining_strategy="easy_pos_semihard_neg".
    strict_semihard: bool = False
    # weight on the centroid-separation term (loss_type = "joint_sep" only).
    # SCALE WARNING: L_sep has very different magnitudes either side of C = 3,
    # because at C = 2 it is computed on RAW class means and at C >= 3 on
    # CENTRED ones. On random unit embeddings the median is ~0.98 at C = 2 and
    # ~0.03 at C = 3. One fixed value cannot serve both, which is why
    # lambda_sep is searched (search.lambda_sep_range) rather than pinned.
    lambda_sep: float = 0.1
    # the separation term is GATED on a running estimate of the TRAINING cosine
    # silhouette, and switches on mid-epoch at the batch where the estimate
    # first reaches this threshold. None means "no gate": the term is active
    # from the first batch, which is the control arm.
    sep_gate_threshold: Optional[float] = None
    # EMA coefficient for that running estimate. 0.0 means a cumulative mean
    # over every batch seen instead of an EMA.
    sep_gate_momentum: float = 0.05
    # batches that must be seen before the gate may latch. Guards against
    # latching on one lucky early batch at 9 rows per class.
    sep_gate_min_batches: int = 20
    # DEPRECATED AND INERT. L_sep is now always built from the RAW normalised
    # class means. The centred form (mu_c - mu_G) was removed because centring
    # then normalising is invariant to translation AND scale, so it measures
    # only the SHAPE of the simplex of class means and never its SIZE: three
    # classes collapsed to a cap with raw pairwise cosine +0.9994 score
    # L_sep = 0.000035 centred against 2.248 raw. The field is kept only so
    # archived configs still parse; setting it to anything but None warns.
    sep_centre_means: Optional[bool] = None
    # tau: the WARM-UP fraction that replaces the latching gate. The weight on
    # the separation term ramps linearly from 0 to lambda_sep over the first
    # tau * T optimiser steps and is constant thereafter,
    #     lambda_sep(t) = lambda_sep * min(1, t / (tau * T)),
    #     T = max_epochs * batches_per_epoch,   for all t in {1, ..., T}.
    # tau = 0.0 is the DEFAULT and means "full weight from the first step",
    # which reproduces the pre-existing ungated behaviour
    # (sep_gate_threshold = None) exactly.
    #
    # tau IS NOW SEARCHED under the joint condition search (18th axis), and the
    # default 0.0 is therefore ALSO the CLAMP CONSTANT: it is the value every
    # trial whose sampled loss type is not "joint_sep" keeps, so that two
    # points differing only in tau build byte-identical configs. Leave it at
    # 0.0 in any base config for that search.
    #
    # SUPERSEDED ARGUMENT, kept because it was wrong in an instructive way: tau
    # was previously FIXED at 0.3 on the grounds that the loss integrates to
    # lambda_sep * T * (1 - tau/2), so tau and lambda_sep trade off almost
    # multiplicatively and would trace a ridge -- the same reasoning that fixes
    # margin whenever angular_alpha_deg is searched. That does not follow. Two
    # settings with equal DOSE have different TERMINAL weights
    # lambda_sep * g(T), and the epoch selector usually picks a late epoch, so
    # the terminal weight plausibly governs the converged geometry more than
    # the integral does. The dose is not a sufficient statistic; tau carries a
    # genuine second degree of freedom. Under the STAGED phase-2 search tau
    # remains fixed (see search._STAGED_EXCLUDED_LOSS_HPS).
    sep_warmup_frac: float = 0.0

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

    # how min_delta_sil is obtained:
    #   "absolute"       -- use min_delta_sil verbatim (current behaviour)
    #   "floor_scale"    -- kappa * sigma of the label-shuffled silhouette null
    #   "floor_location" -- kappa * mu of that null; RAISES when mu <= 0, which
    #                       is the normal case for a permutation null
    min_delta_sil_mode: str = "absolute"
    min_delta_sil_kappa: float = 2.0        # kappa
    sil_floor_permutations: int = 200       # R; 0 disables the floor measurement

    # --- optional accelerators (off by default) ---
    use_scheduler: bool = False
    scheduler_type: str = "cosine"          # "cosine" | "step" | "none"; used only if use_scheduler
    use_amp: bool = False                   # GPU-only; guarded at runtime

    # --- logging / checkpoint cadence ---
    log_every_epochs: int = 1
    checkpoint_every_epochs: int = 5

    def __post_init__(self):
        if self.mining_strategy not in ("hard", "easy_positive",
                                        "easy_pos_semihard_neg"):
            raise ValueError(
                "mining_strategy must be 'hard', 'easy_positive', or "
                "'easy_pos_semihard_neg'; got %r" % (self.mining_strategy,))
        if self.margin <= 0.0:
            raise ValueError("margin must be > 0")
        if self.loss_type not in ("triplet", "joint", "joint_sep"):
            raise ValueError(
                "loss_type must be 'triplet', 'joint' or 'joint_sep'; got %r"
                % (self.loss_type,))
        if not (0.0 < self.angular_alpha_deg < 90.0):
            raise ValueError(
                "angular_alpha_deg must lie in (0, 90); got %r"
                % (self.angular_alpha_deg,))
        if self.lambda_sep < 0.0:
            raise ValueError("lambda_sep must be >= 0")
        if not (0.0 <= self.sep_gate_momentum <= 1.0):
            raise ValueError(
                "sep_gate_momentum must lie in [0, 1] (0 = cumulative mean); "
                "got %r" % (self.sep_gate_momentum,))
        if self.sep_gate_min_batches < 1:
            raise ValueError("sep_gate_min_batches must be >= 1")
        if (self.sep_gate_threshold is not None
                and not (-1.0 <= float(self.sep_gate_threshold) <= 1.0)):
            raise ValueError(
                "sep_gate_threshold is a silhouette and must lie in [-1, 1] "
                "or be None; got %r" % (self.sep_gate_threshold,))
        if not (0.0 <= float(self.sep_warmup_frac) <= 1.0):
            raise ValueError(
                "sep_warmup_frac (tau) is a FRACTION OF TRAINING and must lie "
                "in [0, 1]: 0 means full lambda_sep from the first step, 1 "
                "means the ramp finishes exactly at the last planned step; "
                "got %r" % (self.sep_warmup_frac,))
        if self.sep_centre_means is not None:
            warnings.warn(
                "sep_centre_means=%r is INERT and ignored: L_sep is now always "
                "built from the RAW normalised class means. The centred form "
                "was removed because it is scale-invariant and therefore blind "
                "to collapse. Remove this field from the config."
                % (self.sep_centre_means,), RuntimeWarning)
        if (self.loss_type in ("joint", "joint_sep")
                and self.strict_semihard
                and self.mining_strategy == "hard"):
            # PROVABLY EMPTY, and silently so. TripletMarginMiner with
            # type_of_triplets="hard" returns exactly the triplets whose
            # negative is CLOSER than the positive (D_an < D_ap); the strict
            # semi-hard filter of Eq. (6) keeps only those with D_ap < D_an.
            # The intersection is empty by construction, so every batch yields
            # n_strict = 0, n_active = 0 and train_loss = 0.0: the network
            # never receives a gradient and the run looks stable rather than
            # broken. MEASURED: 16814 mined, 0 surviving.
            raise ValueError(
                "mining_strategy='hard' and strict_semihard=True are mutually "
                "exclusive: 'hard' mines D_an < D_ap while the strict "
                "semi-hard filter requires D_ap < D_an, so NO triplet ever "
                "survives and the loss is identically zero. Use "
                "mining_strategy='easy_pos_semihard_neg' (semi-hard negatives, "
                "compatible with the filter) or set strict_semihard=False to "
                "keep hard mining.")
        if self.loss_type != "joint_sep" and self.lambda_sep != 0.1:
            warnings.warn(
                "lambda_sep=%g is INERT under loss_type=%r: the separation term "
                "is only built for 'joint_sep'."
                % (self.lambda_sep, self.loss_type), RuntimeWarning)
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
        if self.min_delta_sil_mode not in ("absolute", "floor_scale",
                                           "floor_location"):
            raise ValueError(
                "min_delta_sil_mode must be 'absolute', 'floor_scale' or "
                "'floor_location'; got %r" % (self.min_delta_sil_mode,))
        if self.min_delta_sil_kappa <= 0.0:
            raise ValueError("min_delta_sil_kappa must be > 0")
        if self.sil_floor_permutations < 0:
            raise ValueError("sil_floor_permutations must be >= 0")
        if (self.min_delta_sil_mode != "absolute"
                and self.sil_floor_permutations < 2):
            raise ValueError(
                "min_delta_sil_mode = %r needs sil_floor_permutations >= 2 to "
                "estimate the null spread; got %d"
                % (self.min_delta_sil_mode, self.sil_floor_permutations))
        if self.scheduler_type not in ("cosine", "step", "none"):
            raise ValueError("scheduler_type must be 'cosine', 'step', or 'none'")
        if self.log_every_epochs < 1 or self.checkpoint_every_epochs < 1:
            raise ValueError("log / checkpoint cadences must be >= 1")
        if self.max_epochs <= self.patience:
            warnings.warn(
                "max_epochs (%d) <= patience (%d): early stopping can never fire, "
                "so training is effectively fixed-length at max_epochs."
                % (self.max_epochs, self.patience),
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
    # margin_range is used only when train.loss_type == "triplet". Under
    # "joint"/"joint_sep" the margin is FIXED and angular_alpha_deg_range is
    # searched in its place; the field is kept so archived configs still load
    # and so the two loss types remain runnable from the same file.
    margin_range: Tuple[float, float] = (0.1, 1.0)             # Real m
    # searched INSTEAD of margin_range under "joint"/"joint_sep". NOT Chung and
    # Lee's {30, 45, 60, 75}: those were chosen for UNNORMALISED embeddings and
    # are all vacuous on the unit hypersphere here (the floor goes non-positive
    # at 30 deg). The LOW end is the collapse-forcing end -- the implied
    # silhouette floor is 0.970 at 5 deg and 0.995 at 2 deg -- so the range
    # extends to 2 deg to leave the search room to ask for near-total
    # within-class collapse. Raise the low bound to 5.0 to reproduce the
    # pre-collapse setting.
    angular_alpha_deg_range: Tuple[float, float] = (2.0, 20.0)  # Real alpha
    # searched ADDITIONALLY under "joint_sep". Log-uniform, and deliberately
    # wide: the natural scale of L_sep differs by more than an order of
    # magnitude between C = 2 (raw class means) and C >= 3 (centred), so a
    # linear prior centred on the C >= 3 value would barely reach the C = 2 one.
    lambda_sep_range: Tuple[float, float] = (1e-3, 1.0)        # log-uniform
    # tau, the warm-up fraction. The UPPER bound is DERIVED at
    # space-construction time as min(1, patience / max_epochs) and this range
    # is clipped to it, so that the ramp always completes before the earliest
    # possible early stop. Stating a high value here is therefore a REQUEST,
    # not a guarantee; search._loss_dims warns when it clips.
    #
    # Uniform, NOT log-uniform: tau = 0 must be reachable, because it is the
    # no-warm-up control arm (sep_warmup_scale already handles it as constant
    # full weight), and a log prior cannot include zero.
    #
    # Searched ONLY by the joint condition search. The staged phase 2 never
    # searched tau, and search._STAGED_EXCLUDED_LOSS_HPS keeps it that way.
    sep_warmup_frac_range: Tuple[float, float] = (0.0, 0.5)    # Real tau
    lr_range: Tuple[float, float] = (1e-4, 0.2)               # log-uniform
    one_minus_beta1_range: Tuple[float, float] = (1e-2, 1e-1)  # log-uniform -> b1 in [0.9, 0.99]
    one_minus_beta2_range: Tuple[float, float] = (1e-4, 1e-2)  # log-uniform -> b2 in [0.99, 0.9999]
    weight_decay_range: Tuple[float, float] = (1e-4, 1e-2)     # log-uniform

    # --- meta ---
    n_calls_arch: int = 100
    n_calls_train: int = 100
    gp_random_state: int = 0
    # [C3] n_init: how many of the n_calls trials are the RANDOM INITIAL DESIGN,
    # drawn quasi-randomly BEFORE the GP surrogate is ever fitted. 0 means "use
    # the legacy rule n_init = min(10, max(1, n_calls // 2))", which is what
    # search.py hard-coded and which every pre-C3 config therefore reproduces
    # EXACTLY -- so this field is additive and changes no existing run. It is
    # exposed because n_init is the mechanism by which a study can return a
    # pre-surrogate random draw as its "optimum": with n_calls = 50 the legacy
    # rule spends 10 trials before the surrogate exists, and if the objective
    # saturates, argmin returns the FIRST of the tied trials, which is one of
    # those 10. Resolved by objective_utils.resolve_n_initial_points.
    n_initial_points: int = 0

    # Search STRATEGY.
    #   "staged" (default, and what every run before this option used):
    #       phase 1 optimizes the 4 ARCHITECTURE HPs with the optimizer frozen,
    #       phase 2 optimizes the 5 TRAINING HPs with the architecture frozen at
    #       the phase-1 winner, then the regularization stage optimizes 2 more
    #       with both frozen. Cheap per dimension, but it ASSUMES SEPARABILITY:
    #       that the best architecture is best regardless of the optimizer it
    #       ends up paired with. A depth that only pays off at a learning rate
    #       phase 1 never tried is invisible to this search.
    #   "joint":
    #       ONE GP over all 10 HPs at once. No separability assumption; pays for
    #       it in dimension, since 140 trials sample 10-D far more thinly than
    #       three GPs sample 4-D, 5-D and 2-D.
    # Which wins is empirical. Run both, same seed, same data.
    #   "joint_conditions":
    #       the joint space PLUS the four CATEGORICAL experimental factors
    #       (mining_strategy, loss_type, strict_semihard, head geometry), i.e.
    #       the 18-axis space that REPLACES the 52-cell factorial. The factors
    #       are searched rather than enumerated, because the screening found
    #       the head geometries differ chiefly in generalisation gap and the
    #       strict-filter x head interaction was the largest effect measured --
    #       a staged search would fix the head before loss_type is a variable.
    #       Illegal combinations are removed by the legality projection Pi
    #       (condition_space.project_condition) BEFORE any config is built.
    #
    # It is ONE knob rather than "joint" plus a separate boolean, so the
    # meaningless combination (staged + searched conditions) is not
    # representable at all.
    search_mode: str = "staged"
    # Joint budget. 0 = MATCH the staged total (n_calls_arch + n_calls_train +
    # regularization.n_calls), which is the only setting under which a
    # staged-vs-joint comparison is about strategy rather than about compute.
    n_calls_joint: int = 0
    # Size of the random initial design for the JOINT search specifically.
    # 0 = fall back to n_initial_points, and then to the legacy rule. It is a
    # separate field because the joint-with-conditions space is far wider than
    # any staged phase (22 surrogate columns against 4, 5 and 2), so the number
    # of pre-surrogate draws it wants is not the number those phases want.
    n_initial_points_joint: int = 0

    # --- the four categorical experimental factors -------------------------- #
    # These were the 52 config FILES of the screening factorial; the joint
    # search carries them as dimensions instead. Each list is the set of LEVELS
    # the search may sample. Defaults are the full level sets, i.e. the whole
    # factorial; restrict one to a single level to freeze that factor without
    # touching the code.
    #
    # They are INERT unless the joint condition search is running: nothing in
    # the staged pipeline reads them, so every archived config keeps its exact
    # semantics whether or not these fields are present in its JSON.
    mining_strategy_choices: Tuple[str, ...] = (
        "hard", "easy_positive", "easy_pos_semihard_neg")
    loss_type_choices: Tuple[str, ...] = ("triplet", "joint", "joint_sep")
    # The three binaries below are encoded as Integer(0, 1) rather than as a
    # two-level Categorical. A two-level one-hot is exactly redundant
    # (x2 = 1 - x1), so this saves one surrogate column each for free, and a
    # binary carries no false ordering. Same for block_family_choices above:
    # the BUG 2 warning in search.py is specifically that a REAL block_family
    # yields floats such as 0.37 and Block_array[0.37] raises TypeError;
    # Integer yields genuine Python ints and is safe. The Stage 4 smoke test
    # asserts that int-ness rather than trusting it.
    strict_semihard_choices: Tuple[int, ...] = (0, 1)
    head_fusion_choices: Tuple[int, ...] = (0, 1)
    # 0 -> head_pool_ops = ("mean",);  1 -> ("mean", "max", "std")
    head_pool_ops_choices: Tuple[int, ...] = (0, 1)
    # DEPRECATED AND INERT: sep_centre_means is no longer a searched axis, so
    # this list is read by nothing. Kept only so archived configs still parse.
    sep_centre_means_choices: Tuple[int, ...] = (0, 1)

    # [C2] Adaptive lexicographic tie-break. The search objective becomes
    #     J_eps(t) = -(1/S) sum_sigma [ ARI(t,sigma,e*) + eps * Sil(t,sigma,e*) ],
    # with BOTH metrics read at the SAME selected epoch e*, and
    #     eps = tie_break_gamma * Delta_min(y) / (tie_break_sil_hi - tie_break_sil_lo),
    # where Delta_min(y) is the ARI RESOLUTION of the validation labels y: the
    # smallest strictly-positive gap below 1 that ARI can take on that set
    # (search.resolve_tie_break_epsilon -> objective_utils.adaptive_epsilon).
    # Because eps * (s_hi - s_lo) = gamma * Delta_min(y) < Delta_min(y) for every
    # gamma in (0, 1), the secondary metric can only reorder configurations that
    # the primary metric CANNOT separate.
    #
    # tie_break_gamma = 0.0 DISABLES the tie-break and reproduces the pre-C2
    # objective exactly (primary metric only, still read at e*).
    #
    # The default is deliberately non-zero: the archived 3-class run had 47 of 50
    # phase-1 trials return validation ARI of exactly 1.0, so the objective had no
    # way to rank them and argmin returned the first, a pre-surrogate random draw.
    # gamma = 0.5 leaves a factor-2 margin on the guarantee.
    tie_break_gamma: float = 0.5
    # Assumed bounds [s_lo, s_hi] of the SECONDARY metric. Defaults are the
    # theoretical silhouette bounds, which is the SAFE choice: it makes the
    # guarantee hold universally rather than only for the silhouette values this
    # particular study happens to produce.
    tie_break_sil_lo: float = -1.0
    tie_break_sil_hi: float = 1.0
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
        _check("angular_alpha_deg_range", self.angular_alpha_deg_range,
               positive=True)
        if self.angular_alpha_deg_range[1] >= 90.0:
            raise ValueError(
                "angular_alpha_deg_range high must be < 90 degrees")
        _check("lambda_sep_range", self.lambda_sep_range, positive=True)
        # tau is a FRACTION of the planned step budget, so the range must sit
        # inside [0, 1], and it must be non-degenerate: lo == hi would ask
        # skopt for a zero-width Real. The upper bound is NOT checked against
        # patience / max_epochs here -- SearchConfig cannot see TrainConfig,
        # and the derived cap is applied where both are in scope
        # (search._loss_dims).
        _check("sep_warmup_frac_range", self.sep_warmup_frac_range)
        _lo_tau, _hi_tau = self.sep_warmup_frac_range
        if float(_lo_tau) < 0.0 or float(_hi_tau) > 1.0:
            raise ValueError(
                "sep_warmup_frac_range must lie within [0, 1] (tau is a "
                "fraction of the planned step budget); got %r"
                % (self.sep_warmup_frac_range,))
        if not (float(_lo_tau) < float(_hi_tau)):
            raise ValueError(
                "sep_warmup_frac_range must be non-degenerate (low < high); "
                "got %r" % (self.sep_warmup_frac_range,))
        _check("lr_range", self.lr_range, positive=True)
        _check("one_minus_beta1_range", self.one_minus_beta1_range, positive=True)
        _check("one_minus_beta2_range", self.one_minus_beta2_range, positive=True)
        _check("weight_decay_range", self.weight_decay_range, positive=True)
        if self.one_minus_beta1_range[1] >= 1.0 or self.one_minus_beta2_range[1] >= 1.0:
            raise ValueError(
                "one_minus_beta*_range high must be < 1.0 (beta = 1 - x must stay > 0)")
        if self.n_calls_arch < 1 or self.n_calls_train < 1:
            raise ValueError("n_calls_arch and n_calls_train must be >= 1")
        # [C3] n_initial_points > n_calls would leave the surrogate no trials at
        # all and silently degrade the phase to pure random search. Checked here
        # (config construction) as well as in resolve_n_initial_points (call
        # site), so the error fires BEFORE any trace is generated rather than
        # after the data cache is built.
        if self.search_mode not in ("staged", "joint", "joint_conditions"):
            raise ValueError(
                "search_mode must be 'staged', 'joint' or 'joint_conditions'; "
                "got %r" % (self.search_mode,))
        if self.n_calls_joint < 0:
            raise ValueError("n_calls_joint must be >= 0 (0 = match the staged total)")
        if self.n_initial_points_joint < 0:
            raise ValueError(
                "n_initial_points_joint must be >= 0 (0 = fall back to "
                "n_initial_points, then to the legacy rule)")
        # An EMPTY choice list is the failure this checks for: it would build a
        # zero-level dimension, which skopt rejects only later and obscurely.
        # An UNKNOWN level is the other: it would be sampled, written into the
        # config, and rejected by TrainConfig only once a trial started.
        def _check_choices(name, choices, allowed):
            seq = tuple(choices)
            if len(seq) < 1:
                raise ValueError(
                    "%s must list at least one level (it is the set the search "
                    "may sample); got an empty list" % name)
            bad = [c for c in seq if c not in allowed]
            if bad:
                raise ValueError("%s: unknown level(s) %r; allowed %r"
                                 % (name, bad, allowed))
            if len(set(seq)) != len(seq):
                raise ValueError("%s contains duplicate levels: %r" % (name, seq))

        _check_choices("mining_strategy_choices", self.mining_strategy_choices,
                       ("hard", "easy_positive", "easy_pos_semihard_neg"))
        _check_choices("loss_type_choices", self.loss_type_choices,
                       ("triplet", "joint", "joint_sep"))
        for _name, _val in (("strict_semihard_choices", self.strict_semihard_choices),
                            ("head_fusion_choices", self.head_fusion_choices),
                            ("head_pool_ops_choices", self.head_pool_ops_choices),
                            ("sep_centre_means_choices", self.sep_centre_means_choices)):
            _check_choices(_name, _val, (0, 1))
        if self.n_initial_points < 0:
            raise ValueError(
                "n_initial_points must be >= 0 (0 means the legacy rule "
                "min(10, max(1, n_calls // 2)))")
        if self.n_initial_points > 0:
            for _name, _n_calls in (("n_calls_arch", self.n_calls_arch),
                                    ("n_calls_train", self.n_calls_train)):
                if self.n_initial_points > _n_calls:
                    raise ValueError(
                        "n_initial_points (%d) exceeds %s (%d): the GP surrogate "
                        "would never be fitted in that phase and the study would "
                        "be pure random search."
                        % (self.n_initial_points, _name, _n_calls))
        if not (0.0 < self.refine_top_fraction <= 1.0):
            raise ValueError("refine_top_fraction must lie in (0, 1]")
        # [C2] gamma = 0 is the documented "disabled" value; gamma > 1 would let
        # the secondary metric overturn a genuine primary difference, which is
        # precisely the failure mode the derivation exists to exclude.
        if not (0.0 <= self.tie_break_gamma <= 1.0):
            raise ValueError(
                "tie_break_gamma must lie in [0, 1] (0 disables the tie-break; "
                "gamma > 1 would break the lexicographic guarantee "
                "eps * (s_hi - s_lo) < Delta_min(y)); got %r"
                % (self.tie_break_gamma,))
        if not (self.tie_break_sil_lo < self.tie_break_sil_hi):
            raise ValueError(
                "require tie_break_sil_lo < tie_break_sil_hi; got (%r, %r)"
                % (self.tie_break_sil_lo, self.tie_break_sil_hi))


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
class CohortConfig:
    """Declares WHERE the real MEA cohort lives on disk, per class.

    This block is DECLARATIVE. `run_optimization.py` never reads it: the search
    consumes `data.npz_specs`, a flat list of .npz records. What this block does
    is let ONE file describe both the cohort and the search, so the well list
    used to generate the specs cannot drift from the experiment that consumes
    them. `make_mea_specs.py` reads it and writes the specs file.

    Directory model (three levels, all real on disk):

        <root>/                      one ENTRY of class_roots[c]; a plate/batch
            ptrain_A1/               a WELL == a recording == a CULTURE
                ptrain_10.mat        one electrode's spike raster
                ...
            ptrain_A2/  ...  ptrain_B3/

    and, produced by run_channel_subset_extraction.py, the extraction outputs:

        <extract_root>/<extract_layout>/
            trace_subregion_00.npz   mode per_region_single: N_SUB traces,
            ...                      one CULTURE, K3 groups them
            trace_subregion_08.npz
        or
            traces.npz               mode multichannel: ONE (N_SUB, K) trace

    Both output shapes are recognised. Which one is present decides how many
    trace records a well contributes, so it is reported rather than assumed.

    Fields
    ------
    class_roots : {class index as a STRING : [root paths]}. JSON object keys are
        always strings, hence "0" / "1" rather than 0 / 1. Several roots per
        class is the expected case (one per plate/batch). Class indices must be
        contiguous from 0.
    class_names : optional human labels, parallel to the sorted class indices;
        used for directory naming and log lines, never for the label itself.
    well_glob   : pattern matched against the immediate children of each root.
    extract_root: where run_channel_subset_extraction.py wrote its outputs.
    extract_layout : path template BELOW extract_root, expanded per well with
        {class_index} {class_name} {root_name} {well}. Change this rather than
        editing code if the extraction outputs were laid out differently.
    culture_template : how a culture id is built. It MUST be unique per well
        across the whole cohort -- {root_name}__{well} is unique whenever two
        roots do not share a basename, which is why root_name is included.

    The extraction parameters are recorded so the specs file, the extraction
    array job, and this config cannot disagree about how the traces were made.
    They are metadata here; the extractor takes them as CLI flags.
    """

    class_roots: Dict[str, List[str]] = field(default_factory=dict)
    class_names: List[str] = field(default_factory=list)
    well_glob: str = "ptrain_*"

    extract_root: str = ""
    extract_layout: str = "{class_name}/{root_name}/{well}"
    culture_template: str = "{root_name}__{well}"

    # extraction parameters, mirroring run_channel_subset_extraction.py defaults
    n_subsets: int = 9
    electrodes_per_subset: int = 9
    fs_raw: float = 10110.09
    grid_width: int = 48
    index_base: int = 0
    mfr_threshold: float = 0.1
    w_size: float = 0.02
    gaussian_window: float = 0.04

    def __post_init__(self):
        if not self.class_roots:
            return                      # empty cohort: synthetic/latent configs

        keys = []
        for k in self.class_roots.keys():
            try:
                keys.append(int(k))
            except (TypeError, ValueError):
                raise ValueError(
                    "CohortConfig.class_roots: key %r is not an integer class "
                    "index. JSON object keys are strings, so write \"0\" / "
                    "\"1\"." % (k,))
        keys = sorted(keys)
        if keys != list(range(len(keys))):
            raise ValueError(
                "CohortConfig.class_roots: class indices must be contiguous "
                "from 0; got %r. The pipeline builds C = number of distinct "
                "conditions and indexes classes 0..C-1." % (keys,))

        for k, roots in self.class_roots.items():
            if isinstance(roots, str) or not isinstance(roots, (list, tuple)):
                raise ValueError(
                    "CohortConfig.class_roots[%r] must be a LIST of root paths, "
                    "got %r. A single root still goes in a list." % (k, roots))
            if len(roots) == 0:
                raise ValueError(
                    "CohortConfig.class_roots[%r] is empty; every class needs "
                    "at least one root folder." % (k,))
            seen = set()
            for r in roots:
                rs = str(r)
                if rs in seen:
                    raise ValueError(
                        "CohortConfig.class_roots[%r]: duplicate root %r. The "
                        "same folder listed twice would double-count its wells."
                        % (k, rs))
                seen.add(rs)

        # a root may not appear under two classes: that is a phenotype conflict
        owner = {}
        for k, roots in self.class_roots.items():
            for r in roots:
                rs = str(r)
                if rs in owner and owner[rs] != str(k):
                    raise ValueError(
                        "CohortConfig: root %r is listed under BOTH class %s "
                        "and class %s. A plate has one phenotype."
                        % (rs, owner[rs], k))
                owner[rs] = str(k)

        if self.class_names and len(self.class_names) != len(keys):
            raise ValueError(
                "CohortConfig.class_names has %d entry/entries but there are "
                "%d class(es); they must be parallel to the sorted class "
                "indices %r." % (len(self.class_names), len(keys), keys))

        if int(self.n_subsets) < 1:
            raise ValueError("CohortConfig.n_subsets must be >= 1")
        if int(self.electrodes_per_subset) < 1:
            raise ValueError("CohortConfig.electrodes_per_subset must be >= 1")
        if float(self.fs_raw) <= 0.0:
            raise ValueError("CohortConfig.fs_raw must be > 0")
        if int(self.grid_width) < 1:
            raise ValueError("CohortConfig.grid_width must be >= 1")
        if int(self.index_base) not in (0, 1):
            raise ValueError(
                "CohortConfig.index_base must be 0 or 1 (see REAL_DATA_FINDINGS "
                "O1: both fit the 48x48 grid, so this is an assumption)")
        if float(self.mfr_threshold) < 0.0:
            raise ValueError("CohortConfig.mfr_threshold must be >= 0")
        if float(self.w_size) <= 0.0:
            raise ValueError("CohortConfig.w_size must be > 0")
        if float(self.gaussian_window) < 0.0:
            raise ValueError("CohortConfig.gaussian_window must be >= 0 "
                             "(0 disables smoothing)")
        # sigma is in SECONDS but the filter runs on BINS, so the effective
        # smoothing is sigma/w_size bins. Below ~1 bin the Gaussian is
        # narrower than the sampling grid and does essentially nothing --
        # easy to cause accidentally by shrinking w_size and leaving sigma
        # alone, or vice versa, since the two are set independently.
        if float(self.gaussian_window) > 0.0:
            sigma_bins = float(self.gaussian_window) / float(self.w_size)
            if sigma_bins < 1.0:
                warnings.warn(
                    "CohortConfig: gaussian_window=%.4g s is only %.2f bin(s) "
                    "at w_size=%.4g s, so smoothing is close to a no-op. "
                    "sigma is specified in seconds and applied over bins; if "
                    "you changed one, check the other."
                    % (float(self.gaussian_window), sigma_bins,
                       float(self.w_size)), RuntimeWarning)

    # ----- convenience, used by make_mea_specs.py -----
    def n_classes(self):
        return len(self.class_roots)

    def name_of_class(self, c):
        """Human label for class index c; falls back to 'class<c>'."""
        if self.class_names and 0 <= int(c) < len(self.class_names):
            return str(self.class_names[int(c)])
        return "class%d" % int(c)

    def fs_ifr(self):
        """IFR rate implied by the smoothing window: f_s^IFR = 1 / w_size."""
        return 1.0 / float(self.w_size)

    def sigma_bins(self):
        """Effective Gaussian smoothing width in BINS: sigma / w_size.

        The two IFR knobs are easy to confuse, so stating them apart:

          w_size          the DOWNSAMPLING BIN, Delta_t [s]. Sets the output
                          rate, f_s^IFR = 1 / w_size, and hence K and the
                          window length in samples (T = window_s * f_s^IFR).
          gaussian_window sigma of the gaussian_filter1d applied AFTER
                          binning [s]. Does not change the rate, only how
                          much the binned counts are smoothed.

        They interact because sigma is given in seconds but the filter runs
        over bins: halving w_size doubles the smoothing measured in bins
        while leaving sigma nominally unchanged. This is the number to look
        at when deciding whether a change to either knob did what you meant.
        """
        return float(self.gaussian_window) / float(self.w_size)


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
    # [K3/cohort] Declarative: WHERE the real MEA cohort lives on disk.
    # run_optimization.py never reads this -- make_mea_specs.py does, to
    # generate data.npz_specs. It lives here so one file describes both
    # the cohort and the search, and the two cannot drift apart.
    cohort: CohortConfig = field(default_factory=CohortConfig)

    def __post_init__(self):
        # Single source of truth for the trace channel axis: data.n_channels
        # DRIVES backbone.in_channels, so the backbone stem input and the data
        # windows can never silently disagree. A config therefore only ever sets
        # data.n_channels; backbone.in_channels is derived here.
        want = int(self.data.n_channels)
        have = int(self.backbone.in_channels)
        if have != want:
            if have != 1:
                # backbone.in_channels was given a non-default value that
                # disagrees with data.n_channels -- flag it loudly (data wins).
                warnings.warn(
                    "ExperimentConfig: backbone.in_channels=%d conflicts with "
                    "data.n_channels=%d. data.n_channels is authoritative; "
                    "backbone.in_channels is being set to %d. Set the channel "
                    "count via data.n_channels only." % (have, want, want),
                    RuntimeWarning,
                )
            # replace() re-runs BackboneConfig.__post_init__ so the new value is
            # validated (in_channels >= 1) rather than blindly assigned.
            self.backbone = replace(self.backbone, in_channels=want)

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
        if int(self.backbone.in_channels) != int(self.data.n_channels):
            # Should never trip after __post_init__; guards against a caller
            # mutating one field without the other.
            msgs.append(
                "backbone.in_channels=%d != data.n_channels=%d (channel-axis "
                "mismatch; set data.n_channels only)."
                % (int(self.backbone.in_channels), int(self.data.n_channels)))
        if self.train.max_epochs <= self.train.patience:
            msgs.append(
                "train.max_epochs (%d) <= patience (%d): early stopping cannot "
                "fire, so training is fixed-length at max_epochs."
                % (self.train.max_epochs, self.train.patience))
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
