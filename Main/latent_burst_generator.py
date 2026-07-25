"""
latent_burst_generator.py
=========================

An n-latent-factor phenotype generator built ON TOP of the existing
network-burst model in generate_burst_data.py.

Why this module exists
----------------------
The current MultiClassSyntheticProvider (data_splits.py) drives BOTH of its
degrees of freedom (burst rate, burst width) from a SINGLE scalar
    frac = condition / (C - 1),
so the true data manifold is one-dimensional and the C classes are three points
on a line. Two consequences, both fatal for model selection:

  (i)  the task is solvable by one hand-crafted scalar, so validation ARI
       saturates at 1.0 and the Bayesian-optimization objective is constant;
  (ii) eff_rank ~= 1 is simultaneously the CORRECT answer (the data really is
       1-D) and the signature of representation collapse, so the collapse
       tripwire cannot fire and the hard-vs-easy-positive miner question is
       formally undecidable on this benchmark.

This module fixes both by making the latent dimensionality, the class overlap,
and the label-relevance of each factor explicit and tunable.

Separation of concerns (directive 2)
------------------------------------
  section 1 : latent-space definition and axis maps  (no sampling, no signals)
  section 2 : per-trace latent sampling              (no signal synthesis)
  section 3 : latent -> BurstParams                  (pure parameter mapping)
  section 4 : provider (signal synthesis)            (delegates to generate_burst_data)
  section 5 : ground-truth export for factor-retention analysis
This module never trains, never scores, never plots. It reuses
generate_burst_data.generate_spike_times and .compute_ifr_trace unchanged
(directive 1: do not reimplement what is already tested).

Notation (symbols introduced at first use; carried in full)
-----------------------------------------------------------
    n            : number of latent factors, n in N, n >= 1
    k            : latent axis index, k in {1, ..., n}
    phi          : latent coordinate vector, phi = (phi_1, ..., phi_n),
                   phi in [0, 1]^n. Normalized: 0 = low end of the axis range,
                   1 = high end. Dimensionless by construction.
    C            : number of phenotype classes, class label c in {0, ..., C-1}
    S            : label-carrying axis subset, S subset= {1, ..., n}, S nonempty.
                   Axes k in S determine the class; axes k not in S are
                   label-IRRELEVANT but physically real variation.
    m_c          : class mean position along every label axis,
                   m_c = c / (C - 1) in [0, 1] for C >= 2 (m_0 = 0 if C == 1)
    tau          : class overlap (spread of traces about their class mean),
                   tau >= 0, dimensionless in normalized latent units.
                   tau = 0 reproduces the current deterministic behaviour;
                   tau > 0 makes classes overlap and the task non-trivial.
    r            : trace index within a class, r in {0, ..., n_c - 1}
    theta        : the BurstParams instance produced by a latent vector phi
    x            : the synthesized IFR trace, x in R_{>=0}^{K}
    f_s          : IFR sampling rate [Hz], f_s = 1 / Delta_t

    Per-axis affine map. For axis k with range [a_k, b_k] and orientation
    s_k in {+1, -1}:
        value_k(phi_k) = a_k + (b_k - a_k) * u_k,
        u_k = phi_k        if s_k = +1,
        u_k = 1 - phi_k    if s_k = -1,
    for each fixed k in {1, ..., n} and every phi_k in [0, 1].

Biological grounding (see the accompanying document for full citations)
-----------------------------------------------------------------------
The CHOICE of axes and the DIRECTION of their effects follow the feature set
and the random-forest feature importances reported for developing rat cortical
cultures on multiwell MEAs (Cotterill et al., J Biomol Screen 2016; full text
read). In particular that study reports that the two most discriminative
features of network age were the coefficient of variation of the within-burst
inter-spike interval and the CV of the inter-burst interval, and that mean
burst duration did NOT show strong developmentally related changes. That is
why AX_IRREGULARITY is the default label axis and AX_BURST_DURATION is a
default label-IRRELEVANT axis.

The numeric RANGES below are NOT independently sourced. They are chosen to
bracket the CONTROL_PARAMS / PATHO_PARAMS values already present in
generate_burst_data.py, so that the existing two-condition setting is a point
inside the new latent space. They are exposed as configuration precisely
because they should be re-fitted to the user's own recordings before any
biological claim is made.

HPC note (hpc-python-compat): pure ASCII. The single local module imported
here, generate_burst_data.py, was byte-verified pure ASCII as well (Rule 6).
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from generate_burst_data import (
    BurstParams,
    generate_spike_times,
    compute_ifr_trace,
)

__all__ = [
    "LatentAxis",
    "CLASS_CENTER_MODES",
    "DEFAULT_AXES",
    "DEFAULT_AXIS_NAMES",
    "AXIS_REGISTRY",
    "resolve_axes",
    "build_latent_spec",
    "AX_BURST_RATE",
    "AX_IRREGULARITY",
    "AX_BURST_DURATION",
    "AX_INTRABURST_RATE",
    "AX_PARTICIPATION",
    "AX_BACKGROUND",
    "LatentSpec",
    "sample_latents",
    "latent_to_burst_params",
    "LatentBurstProvider",
    "latent_ground_truth_table",
]


# --------------------------------------------------------------------------- #
# section 1 -- latent-space definition (no sampling, no signal synthesis)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LatentAxis:
    """One latent factor and the BurstParams quantity it drives.

    Attributes
    ----------
    name        : identifier used in configs and in the ground-truth table.
    target      : which physical quantity this axis controls. One of
                  "lambda_b", "sigma_d", "median_duration_s", "lambda_burst",
                  "participation_mean", "lambda_bg".
    lo, hi      : range endpoints [a_k, b_k] in the target's PHYSICAL units.
    orientation : s_k in {+1, -1}; -1 means phi_k = 1 maps to the LOW end.
    units       : physical unit string, for the ground-truth table / docs.
    rationale   : short note on why this axis exists.
    """
    name: str
    target: str
    lo: float
    hi: float
    orientation: int = 1
    units: str = ""
    rationale: str = ""

    def __post_init__(self):
        if self.orientation not in (1, -1):
            raise ValueError("orientation must be +1 or -1; got %r" % (self.orientation,))
        if not (self.lo < self.hi):
            raise ValueError("require lo < hi for axis %r; got (%r, %r)"
                             % (self.name, self.lo, self.hi))

    def value(self, phi_k: float) -> float:
        """Affine map value_k(phi_k). Requires phi_k in [0, 1]."""
        if not (0.0 <= phi_k <= 1.0):
            raise ValueError("phi_%s must lie in [0, 1]; got %r" % (self.name, phi_k))
        u = phi_k if self.orientation == 1 else (1.0 - phi_k)
        return float(self.lo + (self.hi - self.lo) * u)


# Canonical axes. Ranges bracket CONTROL_PARAMS / PATHO_PARAMS of
# generate_burst_data.py (NOT independently sourced -- see module docstring).
AX_BURST_RATE = LatentAxis(
    name="burst_rate", target="lambda_b", lo=0.10, hi=0.40, orientation=1,
    units="bursts/s",
    rationale="network burst rate; 'Burst rate' feature, RF importance 0.49")
AX_IRREGULARITY = LatentAxis(
    name="irregularity", target="sigma_d", lo=0.30, hi=0.90, orientation=1,
    units="dimensionless (log-scale sd of burst duration)",
    rationale="drives CV of IBI / within-burst ISI, the two TOP-importance "
              "features (1.00 and 0.70) for network age")
AX_BURST_DURATION = LatentAxis(
    name="burst_duration", target="median_duration_s", lo=0.15, hi=0.35,
    orientation=1, units="s",
    rationale="median burst duration; reported as NOT strongly developmentally "
              "varying, hence a good label-IRRELEVANT axis")
AX_INTRABURST_RATE = LatentAxis(
    name="intraburst_rate", target="lambda_burst", lo=60.0, hi=140.0,
    orientation=1, units="spikes/s",
    rationale="within-burst firing rate; RF importance 0.31")
AX_PARTICIPATION = LatentAxis(
    name="participation", target="participation_mean", lo=0.45, hi=0.80,
    orientation=1, units="probability",
    rationale="mean per-neuron burst participation; analogue of the "
              "'fraction of bursting electrodes' feature")
AX_BACKGROUND = LatentAxis(
    name="background", target="lambda_bg", lo=0.01, hi=0.06, orientation=1,
    units="spikes/s per neuron",
    rationale="tonic inter-burst firing; sets the IFR floor")

DEFAULT_AXES: Tuple[LatentAxis, ...] = (
    AX_IRREGULARITY,       # k = 1  (default label axis)
    AX_BURST_RATE,         # k = 2
    AX_BURST_DURATION,     # k = 3  (default label-irrelevant)
    AX_INTRABURST_RATE,    # k = 4  (default label-irrelevant)
    AX_PARTICIPATION,      # k = 5  (default label-irrelevant)
    AX_BACKGROUND,         # k = 6  (default label-irrelevant)
)

# Name -> canonical axis. This is what lets a JSON config select axes BY NAME
# without the config module having to import this one (which would drag numpy
# and generate_burst_data into the config import chain).
AXIS_REGISTRY: Dict[str, LatentAxis] = {a.name: a for a in DEFAULT_AXES}
DEFAULT_AXIS_NAMES: Tuple[str, ...] = tuple(a.name for a in DEFAULT_AXES)

# Beta concentration kappa = alpha_p + beta_p, held FIXED so that
# participation_mean p_bar alone determines (alpha_p, beta_p) via
#     alpha_p = kappa * p_bar,  beta_p = kappa * (1 - p_bar).
# kappa = 4.0 exactly reproduces CONTROL Beta(3, 1) at p_bar = 0.75 and
# PATHO Beta(2, 2) at p_bar = 0.50.
PARTICIPATION_KAPPA = 4.0


@dataclass
class LatentSpec:
    """Full specification of the latent phenotype space.

    Attributes
    ----------
    axes            : ordered tuple of LatentAxis, length n.
    label_axes      : indices S (0-based into `axes`) that carry the class label.
                      DEFAULT (0, 1) = irregularity + burst_rate. Calibrated:
                      a SINGLE label axis on irregularity alone is under-powered
                      at a 30 s window, because such a window contains only
                      ~5-12 bursts and the duration CV cannot be estimated from
                      so few. Two axes give a task that is hard but solvable.
    n_classes       : C >= 1.
    class_overlap   : tau >= 0. Spread of a trace's label coordinate about its
                      class mean, in normalized latent units. tau = 0 gives
                      deterministic class centres; tau > 0 makes classes
                      overlap. DEFAULT 0.10, calibrated so that a 5-feature
                      hand-crafted baseline scores ARI ~= 0.30 (measured), i.e.
                      well above chance but with large headroom for a learned
                      model -- the regime in which architecture differences can
                      actually be resolved.
    n_per_class     : traces per class, length C.
    class_center_mode : where the class centres m_c sit along a label axis.
                      "interior" (DEFAULT) puts them at (c+1)/(C+1), strictly
                      inside [0, 1]; "endpoints" reproduces the original
                      c/(C-1), whose outer centres sit ON the boundaries and
                      therefore clip ~50% of their draws at every tau. See
                      _class_mean for the measurements behind the default.
                      Changing this changes EVERY generated trace.
    duration_s      : T_rec [s] per trace.
    n_neurons       : N neurons per trace.
    w_size          : IFR bin width Delta_t [s]; f_s = 1 / Delta_t.
    gaussian_window : IFR smoothing sd [s].
    seed            : base seed.
    """
    axes: Tuple[LatentAxis, ...] = DEFAULT_AXES
    label_axes: Tuple[int, ...] = (0, 1)
    n_classes: int = 3
    class_overlap: float = 0.10
    n_per_class: Tuple[int, ...] = (3, 3, 3)
    class_center_mode: str = "interior"
    duration_s: float = 600.0
    n_neurons: int = 100
    w_size: float = 0.02
    gaussian_window: float = 0.04
    seed: int = 0

    def __post_init__(self):
        if len(self.axes) < 1:
            raise ValueError("need at least one latent axis")
        if self.n_classes < 1:
            raise ValueError("n_classes must be >= 1")
        if len(self.n_per_class) != self.n_classes:
            raise ValueError("n_per_class has length %d but n_classes = %d"
                             % (len(self.n_per_class), self.n_classes))
        if any(int(v) < 1 for v in self.n_per_class):
            raise ValueError("every class needs at least one trace")
        if not self.label_axes:
            raise ValueError("label_axes must be non-empty")
        n = len(self.axes)
        for k in self.label_axes:
            if not (0 <= int(k) < n):
                raise ValueError("label axis index %r out of range [0, %d)" % (k, n))
        if len(set(self.label_axes)) != len(self.label_axes):
            raise ValueError("label_axes contains duplicates")
        if self.class_overlap < 0.0:
            raise ValueError("class_overlap (tau) must be >= 0")
        if self.class_center_mode not in CLASS_CENTER_MODES:
            raise ValueError("class_center_mode must be one of %r; got %r"
                             % (CLASS_CENTER_MODES, self.class_center_mode))
        if self.duration_s <= 0 or self.w_size <= 0:
            raise ValueError("duration_s and w_size must be > 0")

    @property
    def n_latent(self) -> int:
        """n, the number of latent factors."""
        return len(self.axes)

    @property
    def free_axes(self) -> Tuple[int, ...]:
        """Indices k not in S: the label-IRRELEVANT factors."""
        return tuple(k for k in range(self.n_latent) if k not in set(self.label_axes))

    @property
    def fs(self) -> float:
        """IFR sampling rate f_s [Hz]."""
        return 1.0 / float(self.w_size)


# --------------------------------------------------------------------------- #
# section 1b -- config -> LatentSpec (the ONE place the mapping lives)
# --------------------------------------------------------------------------- #
def resolve_axes(axis_names: Sequence[str],
                 axis_overrides: Sequence[Dict[str, object]] = ()) -> Tuple[LatentAxis, ...]:
    """Look up axes BY NAME in AXIS_REGISTRY and apply optional range overrides.

    Parameters
    ----------
    axis_names     : ordered names, one per latent factor k. Order IS the axis
                     index order, so label_axes indices refer to THIS ordering.
    axis_overrides : optional sequence of dicts, each with a "name" naming one of
                     axis_names and any of "lo", "hi", "orientation". A None (or
                     absent) entry leaves that endpoint at its canonical value.
                     This is the calibration hook: the canonical [a_k, b_k] are
                     NOT independently sourced (they bracket the repository's own
                     CONTROL_PARAMS / PATHO_PARAMS), so refitting them to real
                     recordings must be possible without editing code.

    Returns
    -------
    axes : tuple of LatentAxis, length n = len(axis_names).
    """
    names = tuple(str(nm) for nm in axis_names)
    if len(names) < 1:
        raise ValueError("need at least one latent axis name")
    if len(set(names)) != len(names):
        raise ValueError("axis_names contains duplicates: %r" % (names,))
    unknown = [nm for nm in names if nm not in AXIS_REGISTRY]
    if unknown:
        raise ValueError(
            "unknown latent axis name(s) %r; known axes are %r"
            % (unknown, sorted(AXIS_REGISTRY)))

    by_name = {nm: AXIS_REGISTRY[nm] for nm in names}
    for i, ov in enumerate(axis_overrides or ()):
        ov = dict(ov)
        nm = ov.get("name", None)
        if nm is None:
            raise ValueError("axis_overrides[%d] has no 'name'" % i)
        nm = str(nm)
        if nm not in by_name:
            raise ValueError(
                "axis_overrides[%d] names axis %r, which is not among the "
                "selected axes %r" % (i, nm, names))
        base = by_name[nm]
        lo = base.lo if ov.get("lo", None) is None else float(ov["lo"])
        hi = base.hi if ov.get("hi", None) is None else float(ov["hi"])
        orient = (base.orientation if ov.get("orientation", None) is None
                  else int(ov["orientation"]))
        by_name[nm] = LatentAxis(
            name=base.name, target=base.target, lo=lo, hi=hi,
            orientation=orient, units=base.units,
            rationale=base.rationale + " [range overridden by config]")
    return tuple(by_name[nm] for nm in names)


def build_latent_spec(axis_names: Sequence[str],
                      label_axes: Sequence[int],
                      n_per_class: Sequence[int],
                      duration_s: float,
                      fs: float,
                      class_overlap: float = 0.10,
                      class_center_mode: str = "interior",
                      n_neurons: int = 100,
                      gaussian_window: float = 0.04,
                      seed: int = 0,
                      axis_overrides: Sequence[Dict[str, object]] = ()) -> LatentSpec:
    """Assemble a LatentSpec from plain, JSON-representable arguments.

    Deliberately takes SCALARS and SEQUENCES rather than an ExperimentConfig, so
    the mapping can be unit-tested without importing config.py (whose import
    chain pulls in torch through backbone.py). run_optimization.py holds the
    ten-line adapter that unpacks cfg into this call; this function holds the
    logic.

    Note f_s vs w_size: the burst model is parameterized by the IFR bin width
    Delta_t = w_size [s], while the pipeline configures a sampling rate f_s [Hz].
    They are reciprocal, w_size = 1 / f_s, and this function performs that single
    conversion so the two never drift apart in a config file.

    C is taken as len(n_per_class), exactly as the synthetic branch does.
    """
    axes = resolve_axes(axis_names, axis_overrides)
    n_per_class = tuple(int(v) for v in n_per_class)
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be > 0; got %r" % (fs,))
    return LatentSpec(
        axes=axes,
        label_axes=tuple(int(k) for k in label_axes),
        n_classes=len(n_per_class),
        class_overlap=float(class_overlap),
        class_center_mode=str(class_center_mode),
        n_per_class=n_per_class,
        duration_s=float(duration_s),
        n_neurons=int(n_neurons),
        w_size=1.0 / fs,
        gaussian_window=float(gaussian_window),
        seed=int(seed),
    )


# --------------------------------------------------------------------------- #
# section 2 -- per-trace latent sampling (no signal synthesis)
# --------------------------------------------------------------------------- #
CLASS_CENTER_MODES = ("interior", "endpoints")


def _class_mean(c: int, n_classes: int, mode: str = "interior") -> float:
    """m_c, the class centre along every label axis, in [0, 1].

    Two placements, because where the centres sit interacts with the clip in
    Eq. (1) and that interaction is easy to miss:

      "endpoints" (the ORIGINAL):  m_c = c / (C - 1),      m_c in {0, ..., 1}
          The outer centres sit exactly ON the boundaries of [0, 1], so for
          c = 0 every negative draw of eps_k, and for c = C-1 every positive
          draw, is clipped. MEASURED at C = 3: 50.6% of class-0 and 48.4% of
          class-(C-1) label coordinates land exactly on a boundary, at EVERY
          tau. The realised within-class sd of the outer classes is therefore
          ~0.058 against a nominal tau = 0.10 -- about 42% tighter than the
          middle class, which is untouched. Raising tau does not fix this: the
          pinned FRACTION is invariant in tau (it is just P(eps < 0) = 1/2),
          so a larger tau only grows a spike of probability mass sitting
          exactly at 0 and at 1.

      "interior" (the DEFAULT):    m_c = (c + 1) / (C + 1), m_c in (0, 1)
          Every centre is strictly interior, at C = 3 giving 0.25, 0.50, 0.75.
          No centre touches a boundary, so clipping becomes rare rather than
          systematic and the realised within-class spread equals tau for every
          class alike. tau becomes a knob that means what it says.

    The trade: interior spacing HALVES the adjacent-centre gap (0.50 -> 0.25 at
    C = 3), so at fixed tau the task is substantially harder. MEASURED at
    C = 3, tau = 0.10, 180 windows: a mean-IFR-only baseline scores ARI 0.3821
    under "endpoints" but 0.1877 under "interior" -- i.e. interior spacing also
    closes about half of the single-scalar shortcut, which is the reason to
    prefer it beyond the clipping argument. Choose tau AFTER choosing the mode;
    the two are not independent.

    For C == 1 there is no meaningful spacing: "interior" returns 0.5 (the
    centre of the range) and "endpoints" returns 0.0 (its historical value).
    """
    if mode not in CLASS_CENTER_MODES:
        raise ValueError("class_center_mode must be one of %r; got %r"
                         % (CLASS_CENTER_MODES, mode))
    if n_classes <= 1:
        return 0.5 if mode == "interior" else 0.0
    if mode == "endpoints":
        return float(c) / float(n_classes - 1)
    return float(c + 1) / float(n_classes + 1)


def sample_latents(spec: LatentSpec, condition: int, trace_id: int) -> np.ndarray:
    """Draw the latent vector phi in [0, 1]^n for ONE trace.

    Determinism: the RNG is seeded from (spec.seed, condition, trace_id) only,
    so phi is reproducible and INDEPENDENT of the order in which traces are
    generated -- required because the pipeline may request traces out of order.

    For each label axis k in S:
        phi_k = clip( m_c + tau * eps_k , 0, 1 ),  eps_k ~ Normal(0, 1) i.i.d.
    For each free axis k not in S:
        phi_k ~ Uniform(0, 1) i.i.d.

    Returns
    -------
    phi : (n,) float64 array in [0, 1]^n.
    """
    if not (0 <= int(condition) < spec.n_classes):
        raise ValueError("condition %d out of range [0, %d)"
                         % (condition, spec.n_classes))
    rng = np.random.default_rng(
        (int(spec.seed), 1000003 * int(condition) + int(trace_id)))
    n = spec.n_latent
    phi = rng.uniform(0.0, 1.0, size=n)
    m_c = _class_mean(int(condition), spec.n_classes, spec.class_center_mode)
    for k in spec.label_axes:
        eps = rng.normal(0.0, 1.0)
        phi[int(k)] = float(np.clip(m_c + spec.class_overlap * eps, 0.0, 1.0))
    return phi


# --------------------------------------------------------------------------- #
# section 3 -- latent -> BurstParams (pure parameter mapping)
# --------------------------------------------------------------------------- #
def latent_to_burst_params(spec: LatentSpec, phi: np.ndarray,
                           condition: int = 0, tag: str = "") -> BurstParams:
    """Map phi in [0, 1]^n to a BurstParams theta.

    Any BurstParams field not driven by an axis keeps its dataclass default.
    The median-burst-duration axis is mapped through the log, because
    generate_burst_data parameterizes duration by mu_D = ln(median D).
    """
    phi = np.asarray(phi, dtype=np.float64).ravel()
    if phi.shape[0] != spec.n_latent:
        raise ValueError("phi has length %d, expected n = %d"
                         % (phi.shape[0], spec.n_latent))

    fields: Dict[str, float] = {}
    participation_mean = None
    for k, axis in enumerate(spec.axes):
        v = axis.value(float(phi[k]))
        if axis.target == "median_duration_s":
            if v <= 0.0:
                raise ValueError("median burst duration must be > 0; got %r" % (v,))
            fields["mu_d"] = float(np.log(v))
        elif axis.target == "participation_mean":
            participation_mean = float(v)
        else:
            fields[axis.target] = float(v)

    if participation_mean is not None:
        p_bar = float(np.clip(participation_mean, 1e-6, 1.0 - 1e-6))
        fields["alpha_p"] = PARTICIPATION_KAPPA * p_bar
        fields["beta_p"] = PARTICIPATION_KAPPA * (1.0 - p_bar)

    return BurstParams(
        n_neurons=int(spec.n_neurons),
        duration_s=float(spec.duration_s),
        w_size=float(spec.w_size),
        gaussian_window=float(spec.gaussian_window),
        condition=int(condition),
        tag=tag or ("class%d" % int(condition)),
        **fields,
    )


# --------------------------------------------------------------------------- #
# section 4 -- provider (signal synthesis; delegates to generate_burst_data)
# --------------------------------------------------------------------------- #
class LatentBurstProvider:
    """Drop-in replacement for MultiClassSyntheticProvider.

    Call signature is IDENTICAL -- provider(condition, trace_id) -> (x, f_s) --
    so make_synthetic_specs / preprocessing_cache.cache_traces consume it with
    no change. The returned trace is the smoothed population IFR at f_s Hz,
    i.e. the same physical quantity generate_burst_data.py produces.

    Every generated trace's latent vector phi is recorded in `self.latents`,
    keyed by (condition, trace_id), so the factor-retention analysis can
    regress each phi_k on the learned embedding afterwards.
    """

    def __init__(self, spec: LatentSpec):
        if not isinstance(spec, LatentSpec):
            raise TypeError("spec must be a LatentSpec")
        self.spec = spec
        self.latents: Dict[Tuple[int, int], np.ndarray] = {}

    def __call__(self, condition: int, trace_id: int) -> Tuple[np.ndarray, float]:
        condition, trace_id = int(condition), int(trace_id)
        phi = sample_latents(self.spec, condition, trace_id)
        self.latents[(condition, trace_id)] = phi
        theta = latent_to_burst_params(self.spec, phi, condition=condition)

        # Spike-level RNG is seeded separately from the latent RNG so that
        # changing tau (which perturbs phi) does not also reshuffle the spike
        # noise -- keeps the two sources of variability independent.
        rng = np.random.default_rng(
            (int(self.spec.seed) + 99991, 1000003 * condition + trace_id))
        spikes = generate_spike_times(theta, rng)
        x, fs = compute_ifr_trace(spikes, theta)
        return np.asarray(x, dtype=np.float32), float(fs)


# --------------------------------------------------------------------------- #
# section 5 -- ground truth export (for the factor-retention analysis)
# --------------------------------------------------------------------------- #
def latent_ground_truth_table(spec: LatentSpec) -> Dict[str, object]:
    """Enumerate every trace's latent vector WITHOUT synthesizing any signal.

    Returns a JSON-serializable dict with:
        axis_names   : list of n axis names, in order
        axis_units   : list of n unit strings
        label_axes   : list of label-carrying axis indices S
        free_axes    : list of label-irrelevant axis indices
        rows         : list of {condition, trace_id, phi: [n floats],
                                physical: {target: value}}
    Cheap (no spike generation), so it can be written next to results.json.
    """
    rows: List[Dict[str, object]] = []
    for condition, n_c in enumerate(spec.n_per_class):
        for trace_id in range(int(n_c)):
            phi = sample_latents(spec, condition, trace_id)
            theta = latent_to_burst_params(spec, phi, condition=condition)
            rows.append({
                "condition": int(condition),
                "trace_id": int(trace_id),
                "phi": [float(v) for v in phi],
                "physical": {
                    "lambda_b": float(theta.lambda_b),
                    "sigma_d": float(theta.sigma_d),
                    "median_duration_s": float(np.exp(theta.mu_d)),
                    "lambda_burst": float(theta.lambda_burst),
                    "participation_mean": float(
                        theta.alpha_p / (theta.alpha_p + theta.beta_p)),
                    "lambda_bg": float(theta.lambda_bg),
                },
            })
    return {
        "axis_names": [a.name for a in spec.axes],
        "axis_units": [a.units for a in spec.axes],
        "axis_targets": [a.target for a in spec.axes],
        "label_axes": [int(k) for k in spec.label_axes],
        "free_axes": [int(k) for k in spec.free_axes],
        "n_latent": int(spec.n_latent),
        "n_classes": int(spec.n_classes),
        "class_overlap": float(spec.class_overlap),
        "class_center_mode": str(spec.class_center_mode),
        "class_means": [_class_mean(c, spec.n_classes, spec.class_center_mode)
                        for c in range(spec.n_classes)],
        "fs": float(spec.fs),
        "rows": rows,
    }
