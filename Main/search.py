"""
search.py
=========

Two-phase sequential Bayesian hyper-parameter optimization (skopt gp_minimize),
plus a CORRECTED space-narrowing helper (get_newspace).

Separation of concerns (directive 2): this module SEARCHES ONLY. It does not
train (train.train), does not score (metrics / evaluate), does not load data
(data_splits), does not persist checkpoints (checkpoint). It builds candidate
ExperimentConfigs, hands each to the SAME train() the final run uses, and reads
back the validation objective. That identity is the whole point: a configuration
is scored by exactly the procedure that will later fit the final model, so the
search cannot optimize a proxy that differs from deployment.

Why two SEQUENTIAL phases instead of one joint space
----------------------------------------------------
Phase 1 searches the ARCHITECTURE (4 HPs) with the optimizer HELD FIXED.
Phase 2 fixes the winning architecture and searches the TRAINING HPs (5 HPs).
A joint 9-D space would need far more calls for the same surrogate quality, and
the two groups interact only weakly. Sequential is the locked decision; the cost
is that phase 2's optimum is conditional on phase 1's architecture, which is why
do_retune_arch optionally re-runs phase 1 under the tuned optimizer.

Notation (symbols introduced at first use; carried in full)
-----------------------------------------------------------
Architecture space (phase 1), 4 hyper-parameters:
    d    : depth_exponent, Integer over cfg.search.depth_exponent_range
    wm   : width_multiplier, Real over cfg.search.width_multiplier_range
           (Real, NOT log-uniform: the range (1.5, 3.0) spans well under one
           decade, so a log prior would buy nothing and risks log(0) after
           narrowing)
    blk  : block_family, Categorical over cfg.search.block_family_choices,
           blk in {0, 1}. MUST be Categorical, never Real -- see BUG 2 below.
    es   : embedding_size, Integer over cfg.search.embedding_size_range

Training space (phase 2), 5 hyper-parameters:
    m      : margin, Real over cfg.search.margin_range
    lr     : learning rate, Real LOG-uniform over cfg.search.lr_range
    u1     : u1 = 1 - beta1, Real LOG-uniform over one_minus_beta1_range;
             the config gets beta1 = 1 - u1
    u2     : u2 = 1 - beta2, Real LOG-uniform over one_minus_beta2_range;
             the config gets beta2 = 1 - u2
    wd     : weight_decay, Real LOG-uniform over cfg.search.weight_decay_range

    The betas are searched as (1 - beta) in LOG space because beta1, beta2 live
    at 0.9 / 0.999, i.e. they crowd against 1. A uniform prior on beta itself
    would spend almost all its samples in a region where the effective averaging
    horizon 1/(1 - beta) barely changes, while the interesting variation (horizon
    10 vs 100 vs 1000 steps) is compressed into the last few percent. Searching
    u = 1 - beta in log space makes the sampling uniform in that horizon.

Objective (locked decision 3)
-----------------------------
For a sampled point x, build the ExperimentConfig, then for n = 0, ..., N_s - 1
(N_s = cfg.train.n_seeds) call

    train(cfg, splits.train, splits.val, device,
          seed = cfg.runtime.seed + t * N_s + n)

where t is the trial number (0-based). Each run returns its per-epoch history; we
take that run's BEST validation ARI,

    A_n = max over epochs e of ARI_e   (NaN epochs treated as -inf; see below)

and return the NEGATIVE mean over seeds,

    f(x) = - (1 / N_s) * sum_{n=0..N_s-1} A_n

because gp_minimize MINIMIZES. The per-seed standard deviation

    s(x) = sqrt( (1 / N_s) * sum_n (A_n - mean_n A_n)^2 )     (population std)

is LOGGED, not optimized: it is the honest GP noise level, and reporting it is
what lets you see whether a "better" trial is actually better or just luckier.

The seed formula guarantees DISJOINT seed sets across trials (trial t uses seeds
[s0 + t*N_s, s0 + t*N_s + N_s), which never overlap for different t), so no two
trials are accidentally scored on the same random draws -- while remaining fully
reproducible for a fixed cfg.runtime.seed.

Degenerate trials
-----------------
A trial can fail to produce a finite ARI (e.g. a config that collapses, or a
train() that raises). Such a trial returns _FAILED_OBJECTIVE = +1.0, i.e. the
worst possible value of -ARI given ARI in [-0.5, 1] (so -ARI in [-1, 0.5]); +1.0
is strictly worse than any achievable score, so the GP learns to avoid that region
WITHOUT the search crashing. NaN is never returned: gp_minimize's surrogate cannot
fit NaN and would abort the whole study.

The four legacy get_newspace bugs this module fixes (read from the legacy source,
1D_CNN_functions.get_newspace, approx. lines 2592-2672 -- not from memory)
--------------------------------------------------------------------------------
  BUG 1 -- THE INDEX BUG. The legacy line reads
               es = Real(lower_bounds[3], upper_bounds[3], name='Embedding size')
           but [3] is the WIDTH-SHRINK column; embedding size is column [4]. So the
           refined embedding-size range was silently the width-shrink range, and es
           became perfectly correlated with ws. (This module's space has no ws at
           all -- the arch space is the 4 HPs above -- so the corrected code indexes
           each dimension BY NAME rather than by a hard-coded integer, which makes
           the class of bug unrepresentable.)

  BUG 2 -- EVERYTHING WAS Real. The legacy code built blk and es (both DISCRETE)
           as Real dimensions. A Real blk yields floats such as 0.37, and
           Block_array[0.37] raises TypeError. Here blk is Categorical and d / es
           are Integer, so the sampled values are directly usable as an index and
           as a size.

  BUG 3 -- log-uniform ON A POSSIBLY-ZERO LOWER BOUND. The legacy d and wm were
           Real(..., 'log-uniform'). skopt raises ValueError("search space should
           not contain 0 when using log-uniform prior") whenever narrowing pushes a
           lower bound to 0. Verified against skopt 0.10.2. No dimension in the
           refined ARCH space uses a log prior here.

  BUG 4 -- NO DEGENERATE-RANGE GUARD. If every top-scoring point shares one value
           for a hyper-parameter (which is exactly what happens when the search
           CONVERGES), then lower == upper and skopt raises ValueError("the lower
           bound X has to be less than the upper bound X"). Verified for both
           Integer(3,3) and Real(1.5,1.5). _guard_* below widens an Integer by +/-1
           WITHIN the original bounds, and pins a truly unwidenable dimension as a
           single-value Categorical (which skopt accepts).

HPC note (hpc-python-compat): pure ASCII. Figures (the optional PDP) use the Agg
backend forced by evaluate.py's import; search.py never calls plt.show().
"""

import warnings
from dataclasses import replace

import numpy as np
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
from skopt.utils import use_named_args

from condition_space import (
    LOSS_HP_SUPERSET,
    active_loss_hps,
    cell_name,
    decode_head_fusion,
    decode_head_pool_ops,
    project_condition,
    sep_warmup_frac_cap,
)
from config import ExperimentConfig
from objective_utils import (
    adaptive_epsilon,
    composite_objective,
    primary_secondary_scores,   # [C3] role-ordered (u, v); see evaluate_candidate
    resolve_n_initial_points,
    selected_epoch_index,       # [C2] re-exported below; lives in objective_utils
    selected_epoch_scores,      #      so it is testable without torch
    tie_break_applicable,       # [C3] the dispatch policy, testable without torch
)
from dsn_joint_loss import sep_warmup_scale
from train import train

__all__ = [
    "arch_space",
    "train_space",
    "get_newspace",
    "best_arch_dict",
    "best_train_dict",
    "loss_hp_names",
    "train_names",
    "joint_names",
    "config_from_arch_point",
    "config_from_train_point",
    "selected_epoch_index",
    "selected_epoch_scores",
    "resolve_tie_break_epsilon",
    "evaluate_candidate",
    "search_architecture",
    "search_training",
    "retune_architecture",
    "regularization_space",
    "joint_space",
    "config_from_joint_point",
    "search_joint",
    "best_joint_dict",
    "resolve_n_calls_joint",
    "joint_condition_names",
    "joint_condition_space",
    "project_joint_condition_point",
    "config_from_joint_condition_point",
    "best_joint_condition_dict",
    "annotate_joint_condition_point",
    "search_joint_conditions",
    "resolve_n_initial_points_joint",
    "config_from_reg_point",
    "search_regularization",
    "best_reg_dict",
    "plot_objective_pdp",
    "FAILED_OBJECTIVE",
]

# Worst achievable value of the objective f = -ARI. ARI is bounded above by 1, so
# f >= -1 always; +1.0 is strictly worse than any real trial and is finite, so the
# GP surrogate can fit it. NEVER return NaN -- gp_minimize cannot fit NaN.
FAILED_OBJECTIVE = 1.0

_ARCH_NAMES = ("depth_exponent", "width_multiplier", "block_family", "embedding_size")
_TRAIN_NAMES = ("margin", "lr", "one_minus_beta1", "one_minus_beta2", "weight_decay")
_REG_NAMES = ("dropout", "weight_decay")

# the four optimizer HPs every loss type searches, in their fixed order
_OPT_NAMES = ("lr", "one_minus_beta1", "one_minus_beta2", "weight_decay")

# Loss HPs that belong to A(l) -- the composite loss READS them -- but that the
# STAGED phase 2 never searched. Excluding them from the staged name list is
# what keeps loss_hp_names(train_cfg) byte-compatible with every archived
# staged run: phase 2 searched (angular_alpha_deg, lambda_sep) under
# "joint_sep", and nothing else.
#
# sep_warmup_frac (tau) is here because it joined A(joint_sep) when it became
# the 18th axis of the JOINT CONDITION search. Letting that propagate to the
# staged path would silently widen staged phase 2 from 2 dimensions to 3 under
# "joint_sep" and rewrite the meaning of every archived staged coordinate
# vector, which is a change nobody asked for. The joint condition space builds
# the superset (superset=True) and is therefore unaffected by this list.
_STAGED_EXCLUDED_LOSS_HPS = ("sep_warmup_frac",)


def loss_hp_names(train_cfg=None, superset=False):
    """The LOSS hyper-parameters of the space, for a given loss type.

    This is the single place that decides which loss HPs a space carries, so
    the space, the point->config writer and the winner reader can never
    disagree about the meaning of a coordinate.

    superset=False (DEFAULT, and every pre-existing caller) returns the names
    the STAGED phase 2 searches, i.e. A(l) minus the axes that phase predates:

        "triplet"   -> ("margin",)                 the pre-existing behaviour
        "joint"     -> ("angular_alpha_deg",)      margin is FIXED instead
        "joint_sep" -> ("angular_alpha_deg", "lambda_sep")

    margin and angular_alpha_deg are never BOTH active: both bind on the
    within/between distance ratio, so a space carrying the pair would spend
    trials moving along a ridge.

    superset=True returns the STATIC superset, independent of train_cfg. It
    exists because gp_minimize needs a fixed-length vector: once loss_type is
    itself a searched coordinate, every point must carry every loss HP, and the
    ones outside A(l) are INACTIVE for that trial (decision D1 -- see
    _write_loss_hps, which is what keeps them from reaching the config).

    NOTE ON THE DEFAULT. The handoff specifies "loss_hp_names becomes the
    static superset". Making that unconditional would silently widen the STAGED
    phase-2 space -- from 1 dimension to 4 under "triplet", from 2 to 4 under
    "joint_sep" -- and change every archived staged run, so the superset is a
    flag rather than the default. A(l) remains the single source of truth for
    both branches; the staged branch only removes the axes that phase never
    had.
    """
    if superset:
        return tuple(LOSS_HP_SUPERSET)
    loss_type = "triplet" if train_cfg is None else \
        str(getattr(train_cfg, "loss_type", "triplet"))
    return tuple(n for n in active_loss_hps(loss_type)
                 if n not in _STAGED_EXCLUDED_LOSS_HPS)


def train_names(train_cfg=None):
    """Names of the phase-2 dimensions, in the order train_space returns them."""
    return tuple(loss_hp_names(train_cfg)) + _OPT_NAMES


def _binary_dim(name, choices):
    """An Integer(0, 1) axis, or a pinned single-value Categorical.

    Integer rather than a two-level Categorical: the one-hot of a binary is
    exactly redundant (x2 = 1 - x1), so this is one surrogate column instead of
    two, at no cost in expressiveness and with no false ordering imposed.

    A one-element choice list FREEZES the factor. skopt rejects Integer(v, v)
    (the same degenerate-range failure _guard_integer exists for), so a frozen
    factor is pinned as a single-value Categorical, which skopt accepts.
    """
    vals = tuple(int(c) for c in choices)
    if len(vals) < 1:
        raise ValueError("%s: empty choice list" % name)
    if len(vals) == 1:
        return Categorical([vals[0]], name=name)
    return Integer(min(vals), max(vals), name=name)


def _sep_warmup_frac_cap(train_cfg):
    """tau_max for this trainer config, or 1.0 when it cannot be derived.

    A thin adapter over condition_space.sep_warmup_frac_cap, which owns the
    formula. train_cfg=None (the drop-in default of joint_condition_space) or a
    config missing either field means the cap is UNKNOWN, and an unknown cap
    must not silently narrow the space -- so 1.0 is returned and the requested
    range stands. Every real caller in this pipeline passes cfg.train, and the
    Stage-4 smoke test asserts the cap actually binds there.
    """
    if train_cfg is None:
        return 1.0
    return sep_warmup_frac_cap(int(getattr(train_cfg, "patience", 0)),
                               int(getattr(train_cfg, "max_epochs", 0)))


def _loss_dims(search_cfg, train_cfg, superset=False):
    """skopt dimensions for the loss HPs, in loss_hp_names order.

    superset=False builds only the dimensions the configured loss type
    searches (the staged phase-2 space, unchanged). superset=True builds ALL
    of them, which is what the joint condition search needs: loss_type is a
    searched coordinate there, so the vector length cannot depend on it.
    """
    dims = []
    for name in loss_hp_names(train_cfg, superset=superset):
        if name == "margin":
            lo, hi = search_cfg.margin_range
            dims.append(Real(float(lo), float(hi), name="margin"))
        elif name == "angular_alpha_deg":
            lo, hi = search_cfg.angular_alpha_deg_range
            dims.append(Real(float(lo), float(hi), name="angular_alpha_deg"))
        elif name == "lambda_sep":
            lo, hi = search_cfg.lambda_sep_range
            dims.append(Real(float(lo), float(hi), prior="log-uniform",
                             name="lambda_sep"))
        elif name == "sep_warmup_frac":
            lo, hi = search_cfg.sep_warmup_frac_range
            cap = _sep_warmup_frac_cap(train_cfg)
            hi_eff = min(float(hi), cap)
            if hi_eff < float(hi):
                warnings.warn(
                    "sep_warmup_frac_range upper bound %.3g exceeds patience/"
                    "max_epochs = %.3g; clipped. Above the cap the ramp cannot "
                    "finish before the earliest possible early stop, so a large "
                    "tau would be confounded with 'the separation term was "
                    "off'." % (float(hi), cap), RuntimeWarning)
            if not (float(lo) < hi_eff):
                raise ValueError(
                    "sep_warmup_frac_range lower bound %.3g is at or above the "
                    "derived cap %.3g (patience/max_epochs). Lower the range or "
                    "raise patience." % (float(lo), hi_eff))
            # uniform, NOT log-uniform: tau = 0 is the no-warm-up control arm
            # and a log prior cannot include zero
            dims.append(Real(float(lo), hi_eff, name="sep_warmup_frac"))
        else:
            raise ValueError("unhandled loss HP %r" % (name,))
    return dims


def _write_loss_hps(cfg, p, loss_type=None):
    """Write the ACTIVE loss HPs of the point into cfg.train, and only those.

    This is decision D1, and it is the whole reason the joint condition search
    is well-posed. A point carries every loss HP whatever loss type it sampled;
    the ones outside A(l) are INACTIVE and must not reach the config. Leaving
    them out means cfg.train keeps the BASE CONFIG's value for each inactive
    field, which is a fixed constant for the whole study, so two points
    differing only in inactive coordinates build BYTE-IDENTICAL configs and the
    GP sees one duplicated observation instead of two scattered ones. The
    surrogate then sees genuinely flat directions, which ARD length-scales
    absorb.

    "Inactive" is not "unused". Under "joint"/"joint_sep" the margin is still
    read -- JointTripletLoss takes margin = 2 * m_cos -- it is FIXED at the base
    config's value rather than searched. The base config therefore DEFINES the
    clamp constants; preflight should assert they are the intended ones (in
    particular lambda_sep = 0.1, the TrainConfig default, or every non-joint_sep
    trial fires the "INERT lambda_sep" RuntimeWarning).

    loss_type=None reads cfg.train.loss_type, which under the joint condition
    search MUST already have been written. Passing it explicitly removes that
    ordering dependency, and the caller in config_from_joint_condition_point
    does exactly that.
    """
    lt = str(cfg.train.loss_type) if loss_type is None else str(loss_type)
    active = active_loss_hps(lt)
    if "margin" in active and "margin" in p:
        cfg.train.margin = float(p["margin"])
    if "angular_alpha_deg" in active and "angular_alpha_deg" in p:
        cfg.train.angular_alpha_deg = float(p["angular_alpha_deg"])
    if "lambda_sep" in active and "lambda_sep" in p:
        cfg.train.lambda_sep = float(p["lambda_sep"])
    if "sep_warmup_frac" in active and "sep_warmup_frac" in p:
        cfg.train.sep_warmup_frac = float(p["sep_warmup_frac"])


# --------------------------------------------------------------------------- #
# spaces
# --------------------------------------------------------------------------- #
def arch_space(search_cfg):
    """The 4-HP architecture space (phase 1).

    Types are load-bearing (BUG 2): block_family MUST be Categorical so the sampled
    value can index the block family list; depth_exponent and embedding_size MUST be
    Integer so they are usable as counts.
    """
    lo_d, hi_d = search_cfg.depth_exponent_range
    lo_w, hi_w = search_cfg.width_multiplier_range
    lo_e, hi_e = search_cfg.embedding_size_range
    return [
        Integer(int(lo_d), int(hi_d), name="depth_exponent"),
        Real(float(lo_w), float(hi_w), name="width_multiplier"),
        Categorical(list(search_cfg.block_family_choices), name="block_family"),
        Integer(int(lo_e), int(hi_e), name="embedding_size"),
    ]


def train_space(search_cfg, train_cfg=None):
    """The training space (phase 2). The betas are searched as (1 - beta) in
    LOG space and converted back with beta = 1 - u when the config is built.

    Dimensionality depends on train_cfg.loss_type: 5 dims for "triplet" (the
    pre-existing space, unchanged), 5 for "joint" (alpha replaces margin), 6
    for "joint_sep" (alpha and lambda_sep). train_cfg=None reproduces the
    legacy 5-dim margin space exactly, so old callers are unaffected."""
    lo_lr, hi_lr = search_cfg.lr_range
    lo_b1, hi_b1 = search_cfg.one_minus_beta1_range
    lo_b2, hi_b2 = search_cfg.one_minus_beta2_range
    lo_wd, hi_wd = search_cfg.weight_decay_range
    return _loss_dims(search_cfg, train_cfg) + [
        Real(float(lo_lr), float(hi_lr), prior="log-uniform", name="lr"),
        Real(float(lo_b1), float(hi_b1), prior="log-uniform", name="one_minus_beta1"),
        Real(float(lo_b2), float(hi_b2), prior="log-uniform", name="one_minus_beta2"),
        Real(float(lo_wd), float(hi_wd), prior="log-uniform", name="weight_decay"),
    ]


# --------------------------------------------------------------------------- #
# get_newspace -- the CORRECTED narrowing helper
# --------------------------------------------------------------------------- #
def _guard_integer(name, lo, hi, orig_lo, orig_hi):
    """Return a VALID skopt dimension for an integer HP whose narrowed range may be
    degenerate (BUG 4).

    If lo < hi the range is already valid. If lo == hi we try to widen by +/-1
    while staying inside the ORIGINAL bounds. If even that is impossible (the
    original range is itself a single value), we pin the HP as a single-value
    Categorical, which skopt accepts where Integer(v, v) does not.
    """
    lo, hi = int(lo), int(hi)
    orig_lo, orig_hi = int(orig_lo), int(orig_hi)
    if lo < hi:
        return Integer(lo, hi, name=name)
    new_lo = max(orig_lo, lo - 1)
    new_hi = min(orig_hi, hi + 1)
    if new_lo < new_hi:
        return Integer(new_lo, new_hi, name=name)
    return Categorical([lo], name=name)          # unwidenable -> pin it


def _guard_real(name, lo, hi, orig_lo, orig_hi, rel_pad=0.05):
    """Same guard for a real HP: widen a collapsed range by a small relative pad,
    clipped to the original bounds; pin as a single-value Categorical if that is
    impossible. No log prior is used in the refined ARCH space (BUG 3)."""
    lo, hi = float(lo), float(hi)
    orig_lo, orig_hi = float(orig_lo), float(orig_hi)
    if lo < hi:
        return Real(lo, hi, name=name)
    span = max(abs(lo), 1e-12) * float(rel_pad)
    new_lo = max(orig_lo, lo - span)
    new_hi = min(orig_hi, hi + span)
    if new_lo < new_hi:
        return Real(new_lo, new_hi, name=name)
    return Categorical([lo], name=name)


def get_newspace(res, pers, search_cfg):
    """Narrow the ARCHITECTURE space around the best `pers` fraction of trials.

    Parameters
    ----------
    res        : the OptimizeResult returned by gp_minimize for phase 1
    pers       : fraction in (0, 1] of the best-scoring points to keep
    search_cfg : SearchConfig, supplying the ORIGINAL bounds so a widened guard can
                 never escape the space the user actually allowed

    Returns
    -------
    A list of 4 VALID skopt dimensions, in the SAME order as arch_space().

    How the legacy bugs are made unrepresentable
    --------------------------------------------
    * Columns are addressed BY NAME (via _ARCH_NAMES), never by a hard-coded index,
      so BUG 1 (es reading the ws column) cannot recur.
    * Each HP keeps its ORIGINAL TYPE: Integer stays Integer, Categorical stays
      Categorical (narrowed to the subset of families actually seen among the best
      points), Real stays Real. BUG 2 cannot recur.
    * No log prior is applied. BUG 3 cannot recur.
    * Every dimension goes through a _guard_*, so a converged (lower == upper)
      dimension is widened or pinned rather than raising. BUG 4 cannot recur.
    """
    if not (0.0 < float(pers) <= 1.0):
        raise ValueError("pers must be in (0, 1]; got %r" % (pers,))

    x_iters = list(res.x_iters)
    func_vals = np.asarray(res.func_vals, dtype=float)
    n_trials = len(x_iters)
    if n_trials < 1:
        raise ValueError("res has no trials")

    n_best = max(1, int(np.floor(n_trials * float(pers))))
    order = np.argsort(func_vals)                 # ascending: gp_minimize MINIMIZES
    best_rows = [x_iters[i] for i in order[:n_best]]

    # column-wise values, addressed by NAME (BUG 1 made unrepresentable)
    cols = {name: [row[j] for row in best_rows]
            for j, name in enumerate(_ARCH_NAMES)}

    d_lo, d_hi = min(cols["depth_exponent"]), max(cols["depth_exponent"])
    w_lo, w_hi = min(cols["width_multiplier"]), max(cols["width_multiplier"])
    e_lo, e_hi = min(cols["embedding_size"]), max(cols["embedding_size"])
    blk_seen = sorted(set(int(v) for v in cols["block_family"]))

    od_lo, od_hi = search_cfg.depth_exponent_range
    ow_lo, ow_hi = search_cfg.width_multiplier_range
    oe_lo, oe_hi = search_cfg.embedding_size_range

    return [
        _guard_integer("depth_exponent", d_lo, d_hi, od_lo, od_hi),
        _guard_real("width_multiplier", w_lo, w_hi, ow_lo, ow_hi),
        # Categorical STAYS Categorical, narrowed to the families actually seen
        Categorical(blk_seen, name="block_family"),
        _guard_integer("embedding_size", e_lo, e_hi, oe_lo, oe_hi),
    ]


# --------------------------------------------------------------------------- #
# building a candidate ExperimentConfig from a sampled point
# --------------------------------------------------------------------------- #
def _deep_copy_cfg(base_cfg):
    """An INDEPENDENT ExperimentConfig, safe to mutate for one trial.

    ExperimentConfig has no .copy(), and a shallow copy would be a silent disaster
    here: the nested dataclasses (backbone, train, ...) would be SHARED, so a trial
    that sets cfg.backbone.depth_exponent would corrupt the base config for every
    later trial and for the final run. We round-trip through the tested
    to_dict / from_dict pair (Stage-1 smoke test [2] asserts its fidelity), which
    reconstructs every nested config as a fresh object.
    """
    return ExperimentConfig.from_dict(base_cfg.to_dict())


def config_from_arch_point(base_cfg, point):
    """ExperimentConfig for a phase-1 point: the ARCHITECTURE varies, the optimizer
    is HELD FIXED at base_cfg.train, and dropout is pinned to 0 (regularization is
    tuned only in the Stage-8 stage, decision 11).

    NOTE: BackboneConfig is a FROZEN dataclass (the other sub-configs are not), so
    its fields cannot be assigned -- we rebuild it with dataclasses.replace, which
    also RE-RUNS its __post_init__ validation. That is a free correctness win: an
    architecture point outside the legal range raises here and the trial is scored
    as FAILED rather than silently building an invalid model.
    """
    p = dict(zip(_ARCH_NAMES, point))
    cfg = _deep_copy_cfg(base_cfg)
    cfg.backbone = replace(
        cfg.backbone,
        depth_exponent=int(p["depth_exponent"]),
        width_multiplier=float(p["width_multiplier"]),
        block_family=int(p["block_family"]),
        embedding_size=int(p["embedding_size"]),
        dropout=0.0,
    )
    cfg.validate()
    return cfg


def config_from_train_point(base_cfg, point, arch=None):
    """ExperimentConfig for a phase-2 point: the TRAINING HPs vary, the architecture
    is FIXED (to `arch`, a dict of the 4 arch HPs, when given).

    The beta conversion happens HERE and nowhere else: the search samples
    u = 1 - beta in log space, and the config stores beta = 1 - u."""
    p = dict(zip(train_names(base_cfg.train), point))
    cfg = _deep_copy_cfg(base_cfg)
    if arch is not None:
        cfg.backbone = replace(
            cfg.backbone,
            depth_exponent=int(arch["depth_exponent"]),
            width_multiplier=float(arch["width_multiplier"]),
            block_family=int(arch["block_family"]),
            embedding_size=int(arch["embedding_size"]),
        )
    _write_loss_hps(cfg, p)
    cfg.train.lr = float(p["lr"])
    cfg.train.beta1 = 1.0 - float(p["one_minus_beta1"])   # beta = 1 - u
    cfg.train.beta2 = 1.0 - float(p["one_minus_beta2"])
    cfg.train.weight_decay = float(p["weight_decay"])
    cfg.validate()
    return cfg


# --------------------------------------------------------------------------- #
# the objective
# --------------------------------------------------------------------------- #
def resolve_tie_break_epsilon(cfg, splits, verbose=False):
    """[C2] epsilon for THIS study, derived from the validation labels y.

    epsilon = gamma * Delta_min(y) / (s_hi - s_lo), which guarantees
        epsilon * (s_hi - s_lo) = gamma * Delta_min(y) < Delta_min(y)
    for every gamma in (0, 1): the total influence the secondary metric can
    exert is strictly smaller than the smallest genuine primary difference the
    evaluation set can express, so it reorders configurations ONLY inside an
    exact primary tie.

    cfg.search.tie_break_gamma == 0.0 disables the tie-break entirely and
    reproduces the pre-C2 objective (primary metric only). Returns
    (epsilon or None, info dict).

    [C3] It is ALSO disabled, whatever gamma is, when cfg.train.selection_primary
    is not "ari", because Eq. (4) presumes a discrete primary. See the inline
    note at the dispatch. info["reason"] distinguishes the two disabled cases.

    COST. min_ari_gap is O(N_eval * C) ARI evaluations, each O(N_eval), i.e.
    O(N_eval^2 * C) overall, computed ONCE per phase (not per trial). MEASURED
    by Main/Smoke_Tests/smoke_test_objective_wiring.py [J], which prints the
    timing on whatever machine it runs on: 43 ms at N_eval = 36 (the archived
    run), 222 ms at 180, 821 ms at 600. The growth is quadratic, so budget
    roughly 10 s at N_eval = 2000; if you ever evaluate on tens of thousands of
    windows, hoist this out of the per-phase path deliberately rather than
    discovering the cost in a cluster job.
    """
    gamma = float(getattr(cfg.search, "tie_break_gamma", 0.0))
    primary = str(getattr(cfg.train, "selection_primary", "ari"))
    # [C3] The policy lives in objective_utils so it is testable without torch;
    # this function keeps only the parts that need cfg and splits.
    applicable, reason = tie_break_applicable(primary, gamma)
    if not applicable:
        if reason == "continuous primary":
            # Not silently: a config value that is quietly ignored is the
            # failure mode this codebase has already been bitten by.
            warnings.warn(
                "search.tie_break_gamma = %g is INERT under "
                "train.selection_primary = %r: the tie-break of Eq. (4) "
                "requires a primary with a smallest expressible gap "
                "(Delta_min(y)), which a continuous metric does not have. The "
                "search will minimise the primary alone and the secondary "
                "metric will not influence trial ranking. Set tie_break_gamma "
                "= 0 to silence this." % (gamma, primary), RuntimeWarning)
        return None, {"enabled": False, "gamma": gamma, "reason": reason,
                      "selection_primary": primary}
    y = np.asarray(splits.val.conditions_per_item, dtype=int).ravel()
    info = adaptive_epsilon(
        y,
        sil_lo=float(cfg.search.tie_break_sil_lo),
        sil_hi=float(cfg.search.tie_break_sil_hi),
        gamma=gamma,
    )
    info["enabled"] = True
    if verbose:
        print("[search] tie-break: N_eval=%d C=%d Delta_min(y)=%.6f gamma=%.3g "
              "-> epsilon=%.6g (max secondary influence %.6g < %.6g)"
              % (info["n_eval"], info["n_classes"], info["delta_min"],
                 info["gamma"], info["epsilon"], info["max_secondary_influence"],
                 info["delta_min"]))
    return float(info["epsilon"]), info


def evaluate_candidate(cfg, splits, device, trial_number, log=None, epsilon=None,
                       train_verbose=False):
    """Score ONE candidate config: train n_seeds models, return the mean objective.

    This is THE objective. It calls the same train() the final run calls -- the
    smoke test asserts that identity by patching train and observing the call.

    [C2] With epsilon = None (the default) the objective is the pre-C2 one,
    -mean(primary metric at e*). With epsilon > 0 it is

        J_epsilon(t) = -(1/S) * sum_sigma [ u(t,sigma,e*) + epsilon * v(t,sigma,e*) ]

    with BOTH metrics read at the SAME selected epoch e*(t, sigma).

    [C3] (u, v) are the primary and secondary BY ROLE, i.e. (ARI, Sil) under
    cfg.train.selection_primary == "ari" and (Sil, ARI) under "silhouette".
    Before C3 this function read the pair ordered by NAME, so switching
    selection_primary changed which EPOCH was read but left the search still
    ranking trials by ARI. epsilon is None whenever the primary is continuous
    (see resolve_tie_break_epsilon), so under "silhouette" the objective is
    -mean(Sil at e*) and ARI does not enter the ranking at all.

    NOTE on what is reported vs what is optimized: record["mean"] / ["std"] /
    ["scores"] always carry the PRIMARY metric, so they remain comparable across
    runs with and without the tie-break, and so the per-seed std stays the honest
    GP noise level of the primary signal. record["objective"] is what
    gp_minimize actually minimizes. record["ari_*"] and record["sil_*"] carry
    each metric under its OWN name whatever the roles are, and
    record["selection_primary"] says which of them "mean" duplicates.

    Returns
    -------
    (objective, record)
    """
    n_seeds = int(cfg.train.n_seeds)
    base_seed = int(cfg.runtime.seed)
    scores = []
    sil_scores = []
    ari_scores = []          # [C3] ARI at e*, ALWAYS, whatever the roles are
    objectives = []
    epochs = []
    eff_ranks = []

    for n in range(n_seeds):
        # disjoint seed blocks across trials: trial t owns
        # [base + t*n_seeds, base + t*n_seeds + n_seeds)
        seed = base_seed + int(trial_number) * n_seeds + n
        try:
            _model, history = train(cfg, splits.train, splits.val, device, seed=seed,
                                    verbose=bool(train_verbose))
        except Exception as ex:                    # a bad config must not kill the study
            warnings.warn(
                "trial %d seed %d raised %s: %s -> scored as FAILED."
                % (trial_number, seed, type(ex).__name__, ex), RuntimeWarning)
            continue
        # [C2] both signals at the SAME selected epoch e*
        # [C3] ordered by ROLE: u is whatever cfg.train.selection_primary names
        # as primary. ari and sil come back under their own names as well, so
        # the record can carry both without any field meaning two things.
        u, v, ari_e, sil_e, e_star = primary_secondary_scores(
            history, cfg.train.selection_primary)
        if np.isfinite(u):
            scores.append(float(u))
            sil_scores.append(float(sil_e) if np.isfinite(sil_e) else float("nan"))
            ari_scores.append(float(ari_e) if np.isfinite(ari_e) else float("nan"))
            epochs.append(int(e_star))
            if epsilon is None:
                objectives.append(float(-u))
            else:
                objectives.append(float(composite_objective(u, v, float(epsilon))))
        # eff_rank is the collapse tripwire (mean_pairwise_cos is NOT a reliable
        # absolute signal on non-negative inputs -- it sits near 1 by construction)
        finite_er = [h["health"]["eff_rank"] for h in history
                     if np.isfinite(h["health"]["eff_rank"])]
        if finite_er:
            eff_ranks.append(float(np.mean(finite_er)))

    n_ok = len(scores)
    # A trial is VALID only if EVERY seed completed. (n_ok == 0 -- every seed raised
    # -- is just the extreme case of this same rule, so there is exactly ONE failure
    # path here rather than two that could drift apart.)
    #
    # Why this is strict rather than "use whatever seeds survived": the whole point
    # of averaging over n_seeds (decision 3) is to average out seed noise, so a
    # trial scored on 1 seed and a trial scored on n_seeds are NOT comparable. If we
    # kept the survivors, a config that crashed on 2 of 3 seeds but got lucky on the
    # third would report mean = 0.95 with std = 0.00 -- indistinguishable to the GP
    # from a config that genuinely worked on all three, and MORE attractive than an
    # honest 0.90 +/- 0.05. The surrogate would then actively steer the search TOWARD
    # the flaky region. Requiring all seeds makes "it did not reliably train" a
    # first-class failure instead of a confident, noise-free-looking success.
    if n_ok < n_seeds:
        warnings.warn(
            "trial %d: only %d of %d seeds completed -> scored as FAILED (a trial "
            "scored on fewer seeds is not comparable with a full one)."
            % (trial_number, n_ok, n_seeds), RuntimeWarning)
        record = {"trial": int(trial_number),
                  "scores": [float(v) for v in scores],
                  "mean": float("nan"), "std": float("nan"),
                  "objective": FAILED_OBJECTIVE, "eff_rank": float("nan"),
                  "sil_scores": [float(v) for v in sil_scores],
                  "sil_mean": float("nan"),
                  "ari_scores": [float(v) for v in ari_scores],
                  "ari_mean": float("nan"),
                  "selection_primary": str(cfg.train.selection_primary),
                  "selected_epochs": [int(e) for e in epochs],
                  "epsilon": (None if epsilon is None else float(epsilon)),
                  "n_seeds_ok": int(n_ok), "n_seeds": int(n_seeds), "failed": True}
        if log is not None:
            log.append(record)
        return FAILED_OBJECTIVE, record

    arr = np.asarray(scores, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std())                        # population std across seeds
    objective = float(np.mean(np.asarray(objectives, dtype=float)))
    sil_arr = np.asarray(sil_scores, dtype=float)
    ari_arr = np.asarray(ari_scores, dtype=float)
    record = {
        "trial": int(trial_number),
        "scores": [float(v) for v in arr],
        "mean": mean,
        "std": std,                               # the honest GP noise level
        "objective": float(objective),
        "eff_rank": float(np.mean(eff_ranks)) if eff_ranks else float("nan"),
        "sil_scores": [float(v) for v in sil_arr],
        "sil_mean": (float(np.nanmean(sil_arr))
                     if np.any(np.isfinite(sil_arr)) else float("nan")),
        "ari_scores": [float(v) for v in ari_arr],
        "ari_mean": (float(np.nanmean(ari_arr))
                     if np.any(np.isfinite(ari_arr)) else float("nan")),
        "selection_primary": str(cfg.train.selection_primary),
        "selected_epochs": [int(e) for e in epochs],
        "epsilon": (None if epsilon is None else float(epsilon)),
        "n_seeds_ok": int(n_ok),
        "n_seeds": int(n_seeds),
        "failed": False,
    }
    if log is not None:
        log.append(record)
    return float(objective), record


def _run_gp(space, base_cfg, splits, device, n_calls, random_state, build_cfg,
            verbose=False, tag="", n_initial_points=None, epsilon=None,
            train_verbose=False, annotate=None):
    """Shared gp_minimize driver: wires the objective, keeps a trial counter (so the
    seed blocks stay disjoint), and collects the per-trial log.

    [C3] n_initial_points : explicit size of the random initial design. None (or
        <= 0) reproduces the legacy hard-coded rule min(10, max(1, n_calls // 2))
        EXACTLY, so pre-C3 configs are unaffected.
    [C2] epsilon : tie-break weight, or None for the pre-C2 primary-only objective.
    annotate : optional point -> dict, merged into the trial record BEFORE the
        trial runs. This is how the legality projection Pi becomes visible: a
        projected trial is recorded as projected rather than silently building
        a different config than its coordinates suggest. It must be pure and
        cheap; if it raises, the trial still runs and the error is recorded
        instead, because a logging fault must never cost a training run.
    """
    trial_log = []
    counter = {"t": 0}

    def _note(point):
        if annotate is None:
            return {}
        try:
            return dict(annotate(point))
        except Exception as ex:
            warnings.warn("%s: annotate() raised %s: %s -> trial still runs."
                          % (tag, type(ex).__name__, ex), RuntimeWarning)
            return {"annotate_error": "%s: %s" % (type(ex).__name__, ex)}

    def objective(point):
        t = counter["t"]
        counter["t"] += 1
        note = _note(point)
        try:
            cfg = build_cfg(base_cfg, point)
        except Exception as ex:                   # an INVALID config (failed validate)
            warnings.warn(
                "%s trial %d: invalid config %r (%s) -> scored as FAILED."
                % (tag, t, point, ex), RuntimeWarning)
            rec = {"trial": t, "scores": [], "mean": float("nan"),
                   "std": float("nan"), "objective": FAILED_OBJECTIVE,
                   "eff_rank": float("nan"), "failed": True}
            rec.update(note)
            trial_log.append(rec)
            return FAILED_OBJECTIVE
        if train_verbose:
            print("[%s] trial %3d starting (%d seed(s)) ..." % (tag, t, int(base_cfg.train.n_seeds)),
                  flush=True)
        obj, rec = evaluate_candidate(cfg, splits, device, t, log=trial_log,
                                      epsilon=epsilon,
                                      train_verbose=train_verbose)
        rec.update(note)          # rec IS the object already in trial_log
        if verbose:
            print("[%s] trial %3d  obj %+.4f  (val %s = %.4f +/- %.4f, eff_rank %.2f)"
                  % (tag, t, obj, base_cfg.train.selection_primary,
                     rec["mean"], rec["std"], rec["eff_rank"]))
        return obj

    # [C3] n_initial_points must not exceed n_calls, or skopt never fits the
    # surrogate; resolve_n_initial_points enforces that and raises rather than
    # silently degrading the study to random search.
    n_initial = resolve_n_initial_points(n_calls, n_initial_points)
    if verbose:
        print("[%s] budget: n_calls=%d, n_initial_points=%d (%s)"
              % (tag, int(n_calls), n_initial,
                 "explicit" if (n_initial_points or 0) > 0 else "legacy rule"))
    res = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=int(n_calls),
        n_initial_points=n_initial,
        random_state=int(random_state),           # reproducible trial sequence
        acq_func="EI",
    )
    res.trial_log = trial_log
    res.n_initial_points_used = int(n_initial)
    return res


# --------------------------------------------------------------------------- #
# phase 1 / phase 2 / optional re-tune
# --------------------------------------------------------------------------- #
def search_architecture(cfg, splits, device, space=None, verbose=False,
                        train_verbose=False):
    """PHASE 1: search the 4-HP architecture with the OPTIMIZER HELD FIXED.

    Returns the skopt OptimizeResult, with .trial_log attached. res.x is the best
    point in arch_space() order; use best_arch_dict(res) to name it.
    """
    space = arch_space(cfg.search) if space is None else space
    epsilon, _info = resolve_tie_break_epsilon(cfg, splits, verbose=verbose)
    return _run_gp(
        space=space, base_cfg=cfg, splits=splits, device=device,
        n_calls=int(cfg.search.n_calls_arch),
        random_state=int(cfg.search.gp_random_state),
        build_cfg=config_from_arch_point, verbose=verbose, tag="arch",
        n_initial_points=int(cfg.search.n_initial_points),   # [C3]
        epsilon=epsilon,                                     # [C2]
        train_verbose=train_verbose)


def best_arch_dict(res):
    """Name the winning architecture point (arch_space order)."""
    return {name: v for name, v in zip(_ARCH_NAMES, res.x)}


def search_training(cfg, splits, device, best_arch, verbose=False,
                    train_verbose=False):
    """PHASE 2: fix the architecture to best_arch, search the 5 TRAINING HPs."""
    space = train_space(cfg.search, cfg.train)

    def build(base_cfg, point):
        return config_from_train_point(base_cfg, point, arch=best_arch)

    epsilon, _info = resolve_tie_break_epsilon(cfg, splits, verbose=verbose)
    return _run_gp(
        space=space, base_cfg=cfg, splits=splits, device=device,
        n_calls=int(cfg.search.n_calls_train),
        random_state=int(cfg.search.gp_random_state),
        build_cfg=build, verbose=verbose, tag="train",
        n_initial_points=int(cfg.search.n_initial_points),   # [C3]
        epsilon=epsilon,                                     # [C2]
        train_verbose=train_verbose)


def best_train_dict(res, train_cfg=None):
    """Name the winning training point, converting the betas back: beta = 1 - u.

    train_cfg selects which loss HPs the point carries; None reproduces the
    legacy margin-only reading."""
    p = dict(zip(train_names(train_cfg), res.x))
    out = {
        "lr": float(p["lr"]),
        "beta1": 1.0 - float(p["one_minus_beta1"]),
        "beta2": 1.0 - float(p["one_minus_beta2"]),
        "weight_decay": float(p["weight_decay"]),
    }
    for name in loss_hp_names(train_cfg):
        out[name] = float(p[name])
    return out


def retune_architecture(cfg, splits, device, res_arch, verbose=False,
                       train_verbose=False):
    """Optional: re-run phase 1 on the NARROWED space (get_newspace), now under the
    TUNED optimizer already written into cfg.train by the caller. This is what
    do_retune_arch / do_refine buy: phase 2's optimum was conditional on phase 1's
    architecture, so re-tuning the architecture under the tuned optimizer closes the
    loop once."""
    space = get_newspace(res_arch, cfg.search.refine_top_fraction, cfg.search)
    if verbose:
        print("[retune] narrowed space: %r" % ([ (d.name, type(d).__name__) for d in space ],))
    epsilon, _info = resolve_tie_break_epsilon(cfg, splits, verbose=verbose)
    return _run_gp(
        space=space, base_cfg=cfg, splits=splits, device=device,
        n_calls=int(cfg.search.n_calls_arch),
        random_state=int(cfg.search.gp_random_state),
        build_cfg=config_from_arch_point, verbose=verbose, tag="retune",
        n_initial_points=int(cfg.search.n_initial_points),   # [C3]
        epsilon=epsilon,                                     # [C2]
        train_verbose=train_verbose)


def regularization_space(reg_cfg):
    """The 2-HP final regularization space (Stage 8).

    dropout : Real over reg_cfg.dropout_range. NOT log-uniform -- the range starts
              at 0.0 and skopt raises on a log prior containing 0 (the legacy BUG 3
              again, in a new place). A uniform prior is also the right one here:
              dropout is a probability, and the interesting variation between 0.0 and
              0.3 is linear, not multiplicative.
    wd      : Real LOG-uniform over reg_cfg.weight_decay_range. Weight decay spans
              decades (1e-5 .. 1e-2), so the log prior is what makes the sampling
              uniform in order of magnitude.

    Both are searched on VALIDATION, with the architecture AND the training HPs held
    fixed at the phase-1 / phase-2 winners.
    """
    lo_d, hi_d = reg_cfg.dropout_range
    lo_w, hi_w = reg_cfg.weight_decay_range
    return [
        Real(float(lo_d), float(hi_d), name="dropout"),
        Real(float(lo_w), float(hi_w), prior="log-uniform", name="weight_decay"),
    ]


def config_from_reg_point(base_cfg, point):
    """ExperimentConfig for a regularization point.

    Only dropout and weight_decay move. Everything else -- architecture, margin, lr,
    betas -- is INHERITED from base_cfg, which the driver has already set to the
    phase-1 / phase-2 winners. dropout lives on the (frozen) BackboneConfig, weight
    decay on TrainConfig, so this is the one builder that touches both.
    """
    p = dict(zip(_REG_NAMES, point))
    cfg = _deep_copy_cfg(base_cfg)
    cfg.backbone = replace(cfg.backbone, dropout=float(p["dropout"]))
    cfg.train.weight_decay = float(p["weight_decay"])
    cfg.validate()
    return cfg


# --------------------------------------------------------------------------- #
# JOINT (single-stage) search -- the alternative to the staged pipeline
# --------------------------------------------------------------------------- #
_JOINT_NAMES = ("depth_exponent", "width_multiplier", "block_family",
                "embedding_size", "margin", "lr", "one_minus_beta1",
                "one_minus_beta2", "weight_decay", "dropout")


def joint_names(train_cfg=None):
    """Names of the joint-space dimensions, in the order joint_space builds them."""
    return (_ARCH_NAMES + tuple(loss_hp_names(train_cfg)) + _OPT_NAMES
            + ("dropout",))


def joint_space(search_cfg, reg_cfg, train_cfg=None):
    """ALL hyper-parameters in ONE space (10 dims for the legacy triplet loss).

    The staged pipeline searches 4 dims (architecture), then 5 (training HPs) with
    the architecture frozen at the phase-1 winner, then 2 (regularization) with both
    frozen. That decomposition assumes SEPARABILITY: that the best architecture is
    the best architecture whatever optimizer you eventually pair with it. The
    assumption is not obviously true -- a deeper network may only beat a shallower
    one at a learning rate the shallower one cannot tolerate -- and the staged
    search cannot discover such a pairing, because by the time it varies the
    learning rate the depth is already fixed.

    Searching jointly makes no separability assumption. It pays for that with
    dimension: a GP over 10 dims with the same number of trials is a far sparser
    sample than three GPs over 4, 5 and 2. Which wins is an EMPIRICAL question
    about this problem, which is why both are available and why the budget is
    matched (see n_calls_joint).

    weight_decay appears in BOTH train_space and regularization_space; jointly it
    is ONE dimension, taking the regularization range, which is the wider of the
    two and the one that wins in staged mode (the reg phase runs last and overwrites
    the phase-2 value).

    dropout is pinned to 0 throughout the staged arch/train phases and only freed in
    the final stage; jointly it is free from trial 0.
    """
    lo_d, hi_d = search_cfg.depth_exponent_range
    lo_w, hi_w = search_cfg.width_multiplier_range
    lo_e, hi_e = search_cfg.embedding_size_range
    lo_lr, hi_lr = search_cfg.lr_range
    lo_b1, hi_b1 = search_cfg.one_minus_beta1_range
    lo_b2, hi_b2 = search_cfg.one_minus_beta2_range
    lo_wd, hi_wd = reg_cfg.weight_decay_range          # the wider of the two
    lo_dr, hi_dr = reg_cfg.dropout_range
    return [
        Integer(int(lo_d), int(hi_d), name="depth_exponent"),
        Real(float(lo_w), float(hi_w), name="width_multiplier"),
        Categorical(list(search_cfg.block_family_choices), name="block_family"),
        Integer(int(lo_e), int(hi_e), name="embedding_size"),
    ] + _loss_dims(search_cfg, train_cfg) + [
        Real(float(lo_lr), float(hi_lr), prior="log-uniform", name="lr"),
        Real(float(lo_b1), float(hi_b1), prior="log-uniform", name="one_minus_beta1"),
        Real(float(lo_b2), float(hi_b2), prior="log-uniform", name="one_minus_beta2"),
        Real(float(lo_wd), float(hi_wd), prior="log-uniform", name="weight_decay"),
        Real(float(lo_dr), float(hi_dr), name="dropout"),
    ]


def config_from_joint_point(base_cfg, point):
    """ExperimentConfig for a joint point: EVERY searched HP moves at once.

    Mirrors config_from_arch_point + config_from_train_point + config_from_reg_point
    combined, including the beta = 1 - u conversion, so a joint trial and a staged
    trial that happen to land on the same values build the IDENTICAL config. That
    identity is what makes the two modes comparable at all, and the smoke test
    asserts it rather than trusting it.
    """
    p = dict(zip(joint_names(base_cfg.train), point))
    cfg = _deep_copy_cfg(base_cfg)
    cfg.backbone = replace(
        cfg.backbone,
        depth_exponent=int(p["depth_exponent"]),
        width_multiplier=float(p["width_multiplier"]),
        block_family=int(p["block_family"]),
        embedding_size=int(p["embedding_size"]),
        dropout=float(p["dropout"]),
    )
    _write_loss_hps(cfg, p)
    cfg.train.lr = float(p["lr"])
    cfg.train.beta1 = 1.0 - float(p["one_minus_beta1"])
    cfg.train.beta2 = 1.0 - float(p["one_minus_beta2"])
    cfg.train.weight_decay = float(p["weight_decay"])
    cfg.validate()
    return cfg


def resolve_n_calls_joint(search_cfg, reg_cfg):
    """The joint budget. 0 means MATCH the staged pipeline's total, which is the
    only setting under which the staged/joint comparison is about the STRATEGY
    rather than about who got more compute."""
    n = int(getattr(search_cfg, "n_calls_joint", 0))
    if n > 0:
        return n
    return (int(search_cfg.n_calls_arch) + int(search_cfg.n_calls_train)
            + int(reg_cfg.n_calls))


def best_joint_dict(res, train_cfg=None):
    """The winning joint point as a dict of HP names, ordered by joint_names."""
    return dict(zip(joint_names(train_cfg), res.x))


def search_joint(cfg, splits, device, verbose=False, train_verbose=False):
    """SINGLE-STAGE search: one GP over all 10 hyper-parameters.

    Uses the same objective, the same tie-break epsilon and the same seed-block
    discipline as the staged phases, so the only difference between the two modes
    is the SHAPE OF THE SEARCH, not how a candidate is scored.
    """
    n_calls = resolve_n_calls_joint(cfg.search, cfg.regularization)
    epsilon, _info = resolve_tie_break_epsilon(cfg, splits, verbose=verbose)
    if verbose:
        print("[joint] single-stage search over %d dimensions, %d trials "
              "(staged total would be %d + %d + %d = %d)"
              % (len(joint_names(cfg.train)), n_calls, cfg.search.n_calls_arch,
                 cfg.search.n_calls_train, cfg.regularization.n_calls,
                 int(cfg.search.n_calls_arch) + int(cfg.search.n_calls_train)
                 + int(cfg.regularization.n_calls)))
    return _run_gp(
        space=joint_space(cfg.search, cfg.regularization, cfg.train),
        base_cfg=cfg, splits=splits, device=device,
        n_calls=n_calls,
        random_state=int(cfg.search.gp_random_state),
        build_cfg=config_from_joint_point, verbose=verbose, tag="joint",
        n_initial_points=int(cfg.search.n_initial_points),   # [C3]
        epsilon=epsilon,                                     # [C2]
        train_verbose=train_verbose)


# --------------------------------------------------------------------------- #
# JOINT CONDITION search -- the joint space PLUS the four categorical factors
# --------------------------------------------------------------------------- #
# Axis order is the specification of the design document, and it is load-bearing
# twice over: gp_minimize's trial sequence is a function of it at fixed
# random_state, and every name <-> coordinate mapping in this section is built
# by zipping against it. Do not reorder without regenerating any study meant to
# be compared with an earlier one.
#
#   1-4   architecture      depth_exponent, width_multiplier, block_family,
#                           embedding_size
#   5-8   optimizer         lr, one_minus_beta1, one_minus_beta2, weight_decay
#   9     regularization    dropout        (free from trial 0, not a last stage)
#   10-12 loss HPs          margin, angular_alpha_deg, lambda_sep
#   13-15 objective factors mining_strategy, loss_type, strict_semihard
#   16-17 head geometry     head_fusion, head_pool_ops
#   18    loss HP           sep_warmup_frac (tau)
#
# sep_centre_means was axis 18 and has been REMOVED: the centred form of L_sep
# is scale-invariant and so cannot see collapse, leaving nothing to select.
#
# sep_warmup_frac was ADDED, and is APPENDED LAST rather than slotted in beside
# the other loss HPs at 10-12. Axis order is load-bearing at a fixed
# gp_random_state -- the initial design is a function of the ORDER of the
# dimensions -- so appending is the minimal-diff choice, and no study has yet
# been run against the 17-axis order that a reordering would have to preserve.
#
_JOINT_CONDITION_NAMES = (
    "depth_exponent", "width_multiplier", "block_family", "embedding_size",
    "lr", "one_minus_beta1", "one_minus_beta2", "weight_decay", "dropout",
    "margin", "angular_alpha_deg", "lambda_sep",
    "mining_strategy", "loss_type", "strict_semihard",
    "head_fusion", "head_pool_ops",
    "sep_warmup_frac",
)


def joint_condition_names():
    """Names of the joint CONDITION space, in the order the space builds them.

    Takes no train_cfg, unlike joint_names: the whole point of this space is
    that loss_type is a searched coordinate, so the vector length cannot depend
    on the configured loss type.
    """
    return _JOINT_CONDITION_NAMES


def joint_condition_space(search_cfg, reg_cfg, train_cfg=None):
    """The 18-axis space: every HP AND every categorical factor at once.

    This REPLACES the 52-cell factorial, in which each cell froze the four
    categorical factors and ran its own Bayesian search. Those factors are
    searched here instead, for a reason that is specific to this study rather
    than general: the screening found that head geometries differ chiefly in
    GENERALISATION GAP, and that the strict-filter x head interaction was the
    largest effect measured. A staged search would have fixed the head before
    the loss type existed as a variable at all.

    train_cfg is accepted and ignored for the loss HPs (the superset is always
    built); it is kept in the signature so this function is drop-in wherever
    joint_space is called.

    COLUMN ARITHMETIC. 18 declared axes, but the GP surrogate sees more,
    because skopt one-hots every Categorical:

        12 numeric  (1, 2, 4-12, 18 minus block_family)  -> 12 columns
         4 binaries as Integer(0, 1)                     ->  4 columns
         2 three-level Categorical (13, 14)              ->  6 columns
                                                            ---------
                                                             22 columns

    Encoding the five binaries as Integer rather than Categorical is what saves
    5 columns: a two-level one-hot is exactly redundant (x2 = 1 - x1). This
    includes block_family, which departs from the letter of the BUG 2 note
    above but not its substance -- that warning is specifically that a REAL
    block_family yields floats such as 0.37, so Block_array[0.37] raises
    TypeError. Integer yields genuine Python ints. The smoke test ASSERTS the
    int-ness rather than trusting this paragraph.

    Up to 3 of the 22 columns are INACTIVE for any given trial (the loss HPs
    outside A(l)). _write_loss_hps clamps them so they never reach the config,
    which turns them into genuinely flat directions for the surrogate rather
    than noise-injecting ones; ARD length-scales absorb flat directions. What
    that costs in sample efficiency at this budget is not estimated anywhere.

    Cells, Pi and the coverage arithmetic are UNAFFECTED by the tau axis: tau
    is not part of the cell definition, so the space is still 52 cells.
    """
    lo_d, hi_d = search_cfg.depth_exponent_range
    lo_w, hi_w = search_cfg.width_multiplier_range
    lo_e, hi_e = search_cfg.embedding_size_range
    lo_lr, hi_lr = search_cfg.lr_range
    lo_b1, hi_b1 = search_cfg.one_minus_beta1_range
    lo_b2, hi_b2 = search_cfg.one_minus_beta2_range
    lo_wd, hi_wd = reg_cfg.weight_decay_range          # the wider of the two
    lo_dr, hi_dr = reg_cfg.dropout_range
    dims = [
        Integer(int(lo_d), int(hi_d), name="depth_exponent"),
        Real(float(lo_w), float(hi_w), name="width_multiplier"),
        _binary_dim("block_family", search_cfg.block_family_choices),
        Integer(int(lo_e), int(hi_e), name="embedding_size"),
        Real(float(lo_lr), float(hi_lr), prior="log-uniform", name="lr"),
        Real(float(lo_b1), float(hi_b1), prior="log-uniform", name="one_minus_beta1"),
        Real(float(lo_b2), float(hi_b2), prior="log-uniform", name="one_minus_beta2"),
        Real(float(lo_wd), float(hi_wd), prior="log-uniform", name="weight_decay"),
        Real(float(lo_dr), float(hi_dr), name="dropout"),
    ]
    # the loss-HP superset. _loss_dims builds them in LOSS_HP_SUPERSET order,
    # so pick by name rather than rebuild, and let the assertion below catch
    # any drift between the two orderings instead of letting it become a silent
    # coordinate swap. Three of the four go here at 10-12; sep_warmup_frac is
    # deliberately placed LAST, at 18.
    loss_dims = _loss_dims(search_cfg, train_cfg, superset=True)
    by_name = dict((d.name, d) for d in loss_dims)
    dims += [by_name["margin"], by_name["angular_alpha_deg"],
             by_name["lambda_sep"]]
    dims += [
        Categorical(list(search_cfg.mining_strategy_choices),
                    name="mining_strategy"),
        Categorical(list(search_cfg.loss_type_choices), name="loss_type"),
        _binary_dim("strict_semihard", search_cfg.strict_semihard_choices),
        _binary_dim("head_fusion", search_cfg.head_fusion_choices),
        _binary_dim("head_pool_ops", search_cfg.head_pool_ops_choices),
        # axis 18, appended LAST: see the note above _JOINT_CONDITION_NAMES on
        # why the order is load-bearing. Its upper bound was DERIVED inside
        # _loss_dims, which is the only place where both the requested range
        # and (patience, max_epochs) are in scope.
        by_name["sep_warmup_frac"],
    ]
    built = tuple(d.name for d in dims)
    if built != _JOINT_CONDITION_NAMES:
        raise RuntimeError(
            "joint_condition_space built axes %r but joint_condition_names "
            "declares %r; the two MUST agree or every coordinate downstream "
            "means the wrong thing." % (built, _JOINT_CONDITION_NAMES))
    return dims


def _to_native(v):
    """A numpy scalar -> the equivalent Python scalar; anything else unchanged.

    NOT cosmetic. skopt's Integer.rvs returns numpy.int64 and Real.rvs returns
    numpy.float64, and those are NOT json-serialisable: json.dumps raises
    "Object of type int64 is not JSON serializable". The per-trial log and the
    winner report are both written as JSON, so the point is normalised ONCE,
    here, at the boundary where a sampled point first becomes our data.

    This also corrects a claim worth flagging: the design document justifies
    encoding block_family as Integer rather than Categorical by asserting that
    "Integer yields genuine Python ints". It does not -- it yields numpy ints.
    The SUBSTANCE of the BUG 2 warning is nonetheless respected, because that
    warning is about REAL dimensions yielding floats such as 0.37, and
    Block_array[0.37] raises TypeError while Block_array[numpy.int64(1)] does
    not. The smoke test asserts index-usability, which is the property that
    actually matters, rather than the exact type.
    """
    item = getattr(v, "item", None)
    return item() if callable(item) and getattr(v, "shape", None) == () else v


def project_joint_condition_point(point):
    """Pi at POINT level. Returns (projected_point, info).

    Wraps condition_space.project_condition so the caller never has to know
    which coordinates carry the condition. info is a small dict recording what
    happened, so a projected trial is VISIBLE in the per-trial log rather than
    silent:

        {"projected": bool, "raw": (m, l, s), "condition": (m, l, s),
         "cell": "<historical factorial cell name>"}

    The projection is idempotent, so calling this on an already-projected point
    returns it unchanged with projected=False.
    """
    out = [_to_native(v) for v in point]
    p = dict(zip(_JOINT_CONDITION_NAMES, out))
    raw = (str(p["mining_strategy"]), str(p["loss_type"]),
           bool(int(p["strict_semihard"])))
    cond = project_condition(*raw)
    out[_JOINT_CONDITION_NAMES.index("strict_semihard")] = int(cond[2])
    info = {
        "projected": bool(cond != raw),
        "raw": raw,
        "condition": cond,
        "cell": cell_name(cond[0], cond[1], cond[2],
                          decode_head_fusion(p["head_fusion"]),
                          decode_head_pool_ops(p["head_pool_ops"])),
    }
    return out, info


def config_from_joint_condition_point(base_cfg, point):
    """ExperimentConfig for a joint CONDITION point: HPs and factors move at once.

    Order of operations matters and is asserted by the smoke test:

      1. PROJECT the point (Pi), so the provably-empty
         (hard, joint*, strict=True) cell can never be built. Pi runs BEFORE
         any ExperimentConfig exists, which is the whole reason it is a pure
         function on the point rather than a config-level repair.
      2. Write the architecture and the head geometry.
      3. Write loss_type, THEN the loss HPs, so _write_loss_hps sees the
         sampled loss type (it is passed explicitly anyway).
      4. Rebuild cfg.train through dataclasses.replace so
         TrainConfig.__post_init__ RE-RUNS. Direct attribute assignment -- what
         config_from_train_point does -- skips validation entirely, and
         cfg.validate() only WARNS. Without the replace, a bug in Pi would
         produce a silently zero-loss run instead of a loud failure.

    A raised exception here is not a crash: _run_gp catches it, scores the
    trial FAILED_OBJECTIVE (finite, never NaN) and the study continues.
    """
    point, _info = project_joint_condition_point(point)
    p = dict(zip(_JOINT_CONDITION_NAMES, point))
    loss_type = str(p["loss_type"])

    cfg = _deep_copy_cfg(base_cfg)
    cfg.backbone = replace(
        cfg.backbone,
        depth_exponent=int(p["depth_exponent"]),
        width_multiplier=float(p["width_multiplier"]),
        block_family=int(p["block_family"]),
        embedding_size=int(p["embedding_size"]),
        dropout=float(p["dropout"]),
        head_fusion=decode_head_fusion(p["head_fusion"]),
        head_pool_ops=decode_head_pool_ops(p["head_pool_ops"]),
    )
    # the three fields that are conditions rather than hyper-parameters
    cfg.train.mining_strategy = str(p["mining_strategy"])
    cfg.train.loss_type = loss_type
    cfg.train.strict_semihard = bool(int(p["strict_semihard"]))
    # the optimizer, identically to config_from_joint_point (beta = 1 - u)
    cfg.train.lr = float(p["lr"])
    cfg.train.beta1 = 1.0 - float(p["one_minus_beta1"])
    cfg.train.beta2 = 1.0 - float(p["one_minus_beta2"])
    cfg.train.weight_decay = float(p["weight_decay"])
    # ACTIVE loss HPs only (decision D1); the rest keep the base config's value
    _write_loss_hps(cfg, p, loss_type=loss_type)
    # re-run TrainConfig.__post_init__ on the assembled result
    cfg.train = replace(cfg.train)
    cfg.validate()
    return cfg


def best_joint_condition_dict(res):
    """The winning joint-condition point as a dict, ordered by the axis list.

    The point is projected first, so the reported winner is the configuration
    that actually RAN, not the raw coordinates the surrogate proposed.
    """
    point, _info = project_joint_condition_point(res.x)
    return dict(zip(_JOINT_CONDITION_NAMES, point))


def resolve_n_initial_points_joint(search_cfg):
    """n_init for the joint searches, with its own field and a fallback chain.

    n_initial_points_joint > 0  ->  use it
    else n_initial_points > 0   ->  use that (one setting for every phase)
    else 0                      ->  resolve_n_initial_points applies the legacy
                                    rule min(10, max(1, n_calls // 2))

    The joint condition space is 22 surrogate columns against the staged
    phases' 4, 5 and 2, so the number of pre-surrogate draws it wants is not
    the number those phases want. The legacy rule caps at 10, which in 22
    columns is a very thin design; set n_initial_points_joint deliberately.
    """
    n = int(getattr(search_cfg, "n_initial_points_joint", 0))
    if n > 0:
        return n
    return int(getattr(search_cfg, "n_initial_points", 0))


def annotate_joint_condition_point(point, train_cfg=None):
    """The per-trial log entry for a joint-condition point: what Pi did.

    Everything returned is JSON-serialisable (project_joint_condition_point
    normalises numpy scalars), because the trial log is written to disk as
    JSON. Recording BOTH the raw and the projected condition is the point: a
    trial whose coordinates said (hard, joint_sep, strict=True) but which
    trained (hard, joint_sep, strict=False) must be readable as such, or the
    duplicated observations the GP receives look like noise.

    train_cfg is OPTIONAL and defaults to None, so every pre-existing
    one-argument caller keeps working. When it is supplied AND the sampled loss
    type is "joint_sep", two further quantities are recorded, and they are the
    whole reason tau became a searched axis rather than a fixed knob:

        sep_dose            lambda_sep * T * (1 - tau/2), the INTEGRAL of the
                            weight over the planned budget;
        sep_terminal_weight lambda_sep * g(T), the weight in force at the last
                            planned step.

    Equal-dose settings can differ in terminal weight, which is exactly why the
    dose is not a sufficient statistic for tau. Without BOTH in the log, the
    post-hoc question "did tau matter through the integral or through the
    shape?" cannot be answered at all.

    T IS NOT ALWAYS KNOWABLE HERE. batches_per_epoch = 0 means "derive n_b at
    trainer build time as ceil(N_train / (C * B_c))", so T = E_max * n_b does
    not exist yet. In that case sep_planned_steps and sep_dose are None and the
    T-FREE ratio sep_dose_per_step = lambda_sep * (1 - tau/2) = sep_dose / T is
    recorded instead, which is enough to rank trials against each other at a
    common T. Reporting None beats reporting a number computed from a guessed
    n_b.

    OFF BY ONE, STATED SO IT IS NOT REDISCOVERED. sep_terminal_weight is
    lambda_sep * g(T) as the design document defines it, but the last batch of
    a run passes t = T - 1, not t = T, because t counts steps COMPLETED. The
    difference is one step of the ramp and is immaterial at T >> 1; it is the
    same lag SepWarmup's own docstring records for the per-epoch history.
    """
    _pt, info = project_joint_condition_point(point)
    p = dict(zip(_JOINT_CONDITION_NAMES, _pt))
    m, l, s = info["condition"]
    note = {
        "cell": info["cell"],
        "projected": bool(info["projected"]),
        "condition": [m, l, bool(s)],
        "raw_condition": [info["raw"][0], info["raw"][1], bool(info["raw"][2])],
        "mining_strategy": m,
        "loss_type": l,
        "strict_semihard": bool(s),
        "head_fusion": bool(decode_head_fusion(p["head_fusion"])),
        "head_pool_ops": list(decode_head_pool_ops(p["head_pool_ops"])),
        "active_loss_hps": list(active_loss_hps(l)),
    }
    if l == "joint_sep" and "sep_warmup_frac" in p and "lambda_sep" in p:
        tau = float(p["sep_warmup_frac"])
        lam = float(p["lambda_sep"])
        note["sep_warmup_frac"] = tau
        note["lambda_sep"] = lam
        # g(T) = min(1, T / (tau * T)) = min(1, 1 / tau), independent of T, so
        # the terminal weight is computable even when T is not. Evaluated
        # through the pure schedule rather than restated, so that a future
        # change to the ramp shape propagates here automatically.
        note["sep_terminal_weight"] = lam * float(
            sep_warmup_scale(1, 1, tau))
        note["sep_dose_per_step"] = lam * (1.0 - 0.5 * tau)
        T = None
        if train_cfg is not None:
            n_b = int(getattr(train_cfg, "batches_per_epoch", 0))
            E = int(getattr(train_cfg, "max_epochs", 0))
            if n_b >= 1 and E >= 1:
                T = E * n_b
        note["sep_planned_steps"] = T
        note["sep_dose"] = None if T is None else lam * float(T) * (1.0 - 0.5 * tau)
        note["sep_full_weight_step"] = None if T is None else int(tau * T)
    return note


def search_joint_conditions(cfg, splits, device, verbose=False,
                            train_verbose=False):
    """THE search: one GP over the 18 axes, replacing the 52-cell factorial.

    Scored by the SAME evaluate_candidate, the same tie-break epsilon and the
    same disjoint seed blocks as every other phase, so the only thing that
    differs from the staged pipeline is the shape of the search.

    Failure policy is inherited unchanged and is load-bearing here: a point
    that builds an invalid config scores FAILED_OBJECTIVE = +1.0, which is
    finite and strictly worse than any achievable -silhouette, so the surrogate
    learns to avoid the region WITHOUT the study aborting. NaN is never
    returned; gp_minimize cannot fit it.

    WHAT THIS DOES NOT ANSWER. The GP allocates trials adaptively, so
    "triplet" may receive very few. This search returns a tuned configuration;
    it does NOT establish that the composite objective beats the triplet
    baseline at matched budget. That comparison needs its own matched-budget
    run. Recorded here so the limitation travels with the code.
    """
    n_calls = resolve_n_calls_joint(cfg.search, cfg.regularization)
    n_init = resolve_n_initial_points_joint(cfg.search)
    epsilon, _info = resolve_tie_break_epsilon(cfg, splits, verbose=verbose)
    space = joint_condition_space(cfg.search, cfg.regularization, cfg.train)
    if verbose:
        cols = sum(len(d.categories) if hasattr(d, "categories") else 1
                   for d in space)
        print("[joint_cond] %d axes / %d surrogate columns, %d trials x %d "
              "seed(s) = %d training runs"
              % (len(space), cols, n_calls, int(cfg.train.n_seeds),
                 n_calls * int(cfg.train.n_seeds)))
    return _run_gp(
        space=space,
        base_cfg=cfg, splits=splits, device=device,
        n_calls=n_calls,
        random_state=int(cfg.search.gp_random_state),
        build_cfg=config_from_joint_condition_point,
        verbose=verbose, tag="joint_cond",
        n_initial_points=n_init,
        epsilon=epsilon,
        # bound so the dose / terminal-weight decomposition can be
        # computed; the callback itself takes one argument
        annotate=lambda pt: annotate_joint_condition_point(pt, cfg.train),
        train_verbose=train_verbose)


def search_regularization(cfg, splits, device, verbose=False,
                          train_verbose=False):
    """STAGE 8, step 1: search {dropout, weight_decay} on VALIDATION with the
    architecture and the training HPs FIXED.

    This is deliberately LAST (decision 11): regularization is only meaningful once
    the model can actually fit, so dropout is pinned to 0 throughout phases 1 and 2
    and tuned only here, against the winning configuration.

    Note that weight_decay is searched in BOTH phase 2 and here. That is intentional,
    not a duplication bug: in phase 2 it is one of five interacting optimizer HPs
    tuned at dropout = 0, whereas here it is re-tuned jointly with dropout, because
    the two regularizers trade off against each other. The value found HERE wins.
    """
    # [C2] The tie-break applies here too: this phase is scored by the same
    # objective, on the same validation split, so epsilon is the same number.
    # [C3] n_initial_points is DELIBERATELY not taken from SearchConfig here.
    # This phase has its own budget (RegularizationConfig.n_calls, typically much
    # smaller than n_calls_arch), and silently applying a SearchConfig field to a
    # different phase's budget is exactly the kind of cross-wiring that makes a
    # config unreadable. It therefore keeps the legacy rule. If you want it
    # configurable, add n_initial_points to RegularizationConfig and pass it here
    # -- a two-line change, deliberately left to an explicit decision.
    epsilon, _info = resolve_tie_break_epsilon(cfg, splits, verbose=verbose)
    return _run_gp(
        space=regularization_space(cfg.regularization), base_cfg=cfg, splits=splits,
        device=device, n_calls=int(cfg.regularization.n_calls),
        random_state=int(cfg.regularization.gp_random_state),
        build_cfg=config_from_reg_point, verbose=verbose, tag="reg",
        n_initial_points=None,                               # [C3] legacy rule
        epsilon=epsilon,                                     # [C2]
        train_verbose=train_verbose)


def best_reg_dict(res):
    """Name the winning regularization point."""
    p = dict(zip(_REG_NAMES, res.x))
    return {"dropout": float(p["dropout"]),
            "weight_decay": float(p["weight_decay"])}


# --------------------------------------------------------------------------- #
# optional partial-dependence plot
# --------------------------------------------------------------------------- #
def plot_objective_pdp(res, out_path, dpi=130):
    """Save skopt's partial-dependence plot of the surrogate. Headless (savefig
    only, never plt.show). Returns out_path, or None if skopt cannot build the plot
    (it needs at least a couple of distinct points per dimension)."""
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skopt.plots import plot_objective

    try:
        axes = plot_objective(res)
    except Exception as ex:                       # too few / degenerate trials
        warnings.warn("plot_objective failed (%s: %s) -> no PDP written."
                      % (type(ex).__name__, ex), RuntimeWarning)
        return None
    fig = np.ravel(axes)[0].figure
    parent = os.path.dirname(str(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(str(out_path), dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return str(out_path)
