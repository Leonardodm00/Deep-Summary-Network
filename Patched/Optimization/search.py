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

from config import ExperimentConfig
from train import train

__all__ = [
    "arch_space",
    "train_space",
    "get_newspace",
    "best_arch_dict",
    "best_train_dict",
    "config_from_arch_point",
    "config_from_train_point",
    "evaluate_candidate",
    "search_architecture",
    "search_training",
    "retune_architecture",
    "regularization_space",
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


def train_space(search_cfg):
    """The 5-HP training space (phase 2). The betas are searched as (1 - beta) in
    LOG space and converted back with beta = 1 - u when the config is built."""
    lo_m, hi_m = search_cfg.margin_range
    lo_lr, hi_lr = search_cfg.lr_range
    lo_b1, hi_b1 = search_cfg.one_minus_beta1_range
    lo_b2, hi_b2 = search_cfg.one_minus_beta2_range
    lo_wd, hi_wd = search_cfg.weight_decay_range
    return [
        Real(float(lo_m), float(hi_m), name="margin"),
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
    p = dict(zip(_TRAIN_NAMES, point))
    cfg = _deep_copy_cfg(base_cfg)
    if arch is not None:
        cfg.backbone = replace(
            cfg.backbone,
            depth_exponent=int(arch["depth_exponent"]),
            width_multiplier=float(arch["width_multiplier"]),
            block_family=int(arch["block_family"]),
            embedding_size=int(arch["embedding_size"]),
        )
    cfg.train.margin = float(p["margin"])
    cfg.train.lr = float(p["lr"])
    cfg.train.beta1 = 1.0 - float(p["one_minus_beta1"])   # beta = 1 - u
    cfg.train.beta2 = 1.0 - float(p["one_minus_beta2"])
    cfg.train.weight_decay = float(p["weight_decay"])
    cfg.validate()
    return cfg


# --------------------------------------------------------------------------- #
# the objective
# --------------------------------------------------------------------------- #
def _best_val_ari(history, selection_primary="ari"):
    """The run's BEST validation score: max over epochs of the PRIMARY selection
    signal, NaN-safe (a NaN epoch is treated as -inf so it can never win).

    Returns -inf when no epoch produced a finite score, which the caller maps to
    FAILED_OBJECTIVE."""
    key = "ari" if selection_primary == "ari" else "silhouette"
    best = float("-inf")
    for h in history:
        v = float(h[key])
        if np.isfinite(v) and v > best:
            best = v
    return best


def evaluate_candidate(cfg, splits, device, trial_number, log=None):
    """Score ONE candidate config: train n_seeds models, return (-mean best-val ARI).

    This is THE objective. It calls the same train() the final run calls -- the
    smoke test asserts that identity by patching train and observing the call.

    Returns
    -------
    (objective, record) where
        objective : float, -mean(best val ARI) over seeds; FAILED_OBJECTIVE if no
                    seed produced a finite score
        record    : dict logged per trial (per-seed scores, mean, std, health)
    """
    n_seeds = int(cfg.train.n_seeds)
    base_seed = int(cfg.runtime.seed)
    scores = []
    eff_ranks = []

    for n in range(n_seeds):
        # disjoint seed blocks across trials: trial t owns
        # [base + t*n_seeds, base + t*n_seeds + n_seeds)
        seed = base_seed + int(trial_number) * n_seeds + n
        try:
            _model, history = train(cfg, splits.train, splits.val, device, seed=seed)
        except Exception as ex:                    # a bad config must not kill the study
            warnings.warn(
                "trial %d seed %d raised %s: %s -> scored as FAILED."
                % (trial_number, seed, type(ex).__name__, ex), RuntimeWarning)
            continue
        s = _best_val_ari(history, cfg.train.selection_primary)
        if np.isfinite(s):
            scores.append(float(s))
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
                  "n_seeds_ok": int(n_ok), "n_seeds": int(n_seeds), "failed": True}
        if log is not None:
            log.append(record)
        return FAILED_OBJECTIVE, record

    arr = np.asarray(scores, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std())                        # population std across seeds
    objective = -mean                             # gp_minimize MINIMIZES
    record = {
        "trial": int(trial_number),
        "scores": [float(v) for v in arr],
        "mean": mean,
        "std": std,                               # the honest GP noise level
        "objective": float(objective),
        "eff_rank": float(np.mean(eff_ranks)) if eff_ranks else float("nan"),
        "n_seeds_ok": int(n_ok),
        "n_seeds": int(n_seeds),
        "failed": False,
    }
    if log is not None:
        log.append(record)
    return float(objective), record


def _run_gp(space, base_cfg, splits, device, n_calls, random_state, build_cfg,
            verbose=False, tag=""):
    """Shared gp_minimize driver: wires the objective, keeps a trial counter (so the
    seed blocks stay disjoint), and collects the per-trial log."""
    trial_log = []
    counter = {"t": 0}

    def objective(point):
        t = counter["t"]
        counter["t"] += 1
        try:
            cfg = build_cfg(base_cfg, point)
        except Exception as ex:                   # an INVALID config (failed validate)
            warnings.warn(
                "%s trial %d: invalid config %r (%s) -> scored as FAILED."
                % (tag, t, point, ex), RuntimeWarning)
            rec = {"trial": t, "scores": [], "mean": float("nan"),
                   "std": float("nan"), "objective": FAILED_OBJECTIVE,
                   "eff_rank": float("nan"), "failed": True}
            trial_log.append(rec)
            return FAILED_OBJECTIVE
        obj, rec = evaluate_candidate(cfg, splits, device, t, log=trial_log)
        if verbose:
            print("[%s] trial %3d  obj %+.4f  (val %s = %.4f +/- %.4f, eff_rank %.2f)"
                  % (tag, t, obj, base_cfg.train.selection_primary,
                     rec["mean"], rec["std"], rec["eff_rank"]))
        return obj

    # n_initial_points must not exceed n_calls, or skopt never fits the surrogate
    n_initial = min(10, max(1, int(n_calls) // 2))
    res = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=int(n_calls),
        n_initial_points=n_initial,
        random_state=int(random_state),           # reproducible trial sequence
        acq_func="EI",
    )
    res.trial_log = trial_log
    return res


# --------------------------------------------------------------------------- #
# phase 1 / phase 2 / optional re-tune
# --------------------------------------------------------------------------- #
def search_architecture(cfg, splits, device, space=None, verbose=False):
    """PHASE 1: search the 4-HP architecture with the OPTIMIZER HELD FIXED.

    Returns the skopt OptimizeResult, with .trial_log attached. res.x is the best
    point in arch_space() order; use best_arch_dict(res) to name it.
    """
    space = arch_space(cfg.search) if space is None else space
    return _run_gp(
        space=space, base_cfg=cfg, splits=splits, device=device,
        n_calls=int(cfg.search.n_calls_arch),
        random_state=int(cfg.search.gp_random_state),
        build_cfg=config_from_arch_point, verbose=verbose, tag="arch")


def best_arch_dict(res):
    """Name the winning architecture point (arch_space order)."""
    return {name: v for name, v in zip(_ARCH_NAMES, res.x)}


def search_training(cfg, splits, device, best_arch, verbose=False):
    """PHASE 2: fix the architecture to best_arch, search the 5 TRAINING HPs."""
    space = train_space(cfg.search)

    def build(base_cfg, point):
        return config_from_train_point(base_cfg, point, arch=best_arch)

    return _run_gp(
        space=space, base_cfg=cfg, splits=splits, device=device,
        n_calls=int(cfg.search.n_calls_train),
        random_state=int(cfg.search.gp_random_state),
        build_cfg=build, verbose=verbose, tag="train")


def best_train_dict(res):
    """Name the winning training point, converting the betas back: beta = 1 - u."""
    p = dict(zip(_TRAIN_NAMES, res.x))
    return {
        "margin": float(p["margin"]),
        "lr": float(p["lr"]),
        "beta1": 1.0 - float(p["one_minus_beta1"]),
        "beta2": 1.0 - float(p["one_minus_beta2"]),
        "weight_decay": float(p["weight_decay"]),
    }


def retune_architecture(cfg, splits, device, res_arch, verbose=False):
    """Optional: re-run phase 1 on the NARROWED space (get_newspace), now under the
    TUNED optimizer already written into cfg.train by the caller. This is what
    do_retune_arch / do_refine buy: phase 2's optimum was conditional on phase 1's
    architecture, so re-tuning the architecture under the tuned optimizer closes the
    loop once."""
    space = get_newspace(res_arch, cfg.search.refine_top_fraction, cfg.search)
    if verbose:
        print("[retune] narrowed space: %r" % ([ (d.name, type(d).__name__) for d in space ],))
    return _run_gp(
        space=space, base_cfg=cfg, splits=splits, device=device,
        n_calls=int(cfg.search.n_calls_arch),
        random_state=int(cfg.search.gp_random_state),
        build_cfg=config_from_arch_point, verbose=verbose, tag="retune")


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


def search_regularization(cfg, splits, device, verbose=False):
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
    return _run_gp(
        space=regularization_space(cfg.regularization), base_cfg=cfg, splits=splits,
        device=device, n_calls=int(cfg.regularization.n_calls),
        random_state=int(cfg.regularization.gp_random_state),
        build_cfg=config_from_reg_point, verbose=verbose, tag="reg")


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
