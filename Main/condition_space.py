"""
condition_space.py
==================

The CONDITION of a Deep Summary Network run: the four categorical factors that
the 52-cell factorial enumerated one config file at a time and that the joint
GP search now carries as searched dimensions.

    mining_strategy  in {hard, easy_positive, easy_pos_semihard_neg}
    loss_type        in {triplet, joint, joint_sep}
    strict_semihard  in {False, True}                 (never read for triplet)
    head geometry    (head_fusion, head_pool_ops), a 2 x 2 factor

Separation of concerns (scientific-coding directive 2): this module is PURE.
It imports nothing but the standard library -- no torch, no skopt, not even
config -- so the search, the trainer-side preflight, the factorial generator
and the smoke tests can all import it without dragging in a deep-learning
stack. It decides only WHICH conditions are legal, how an illegal one is
projected onto a legal one, which loss hyper-parameters a loss type actually
reads, and how a condition is NAMED. It never builds an ExperimentConfig and
never touches a config file.

The legality projection Pi
--------------------------
The raw product 3 x 3 x 2 = 18 (mining, loss, strict) triples contains five
that must not be sampled as distinct experiments. Pi collapses those onto the
13 legal ones (13 x 4 head geometries = the historical 52 cells):

  (a) strict_semihard is MEANINGLESS under loss_type = "triplet".
      losses.TripletMarginLoss has no strict semi-hard filter at all, so the
      flag is written false and never read. Leaving it free would give the GP
      two coordinates that build the same config with different labels.
      Pi: strict_semihard <- False whenever loss_type == "triplet".
      This is decision D1 (clamp inactive coordinates) expressed as part of Pi,
      so exactly ONE function decides what a condition means.

  (b) mining_strategy = "hard" with strict_semihard = True is PROVABLY EMPTY.
      TripletMarginMiner(type_of_triplets="hard") returns only triplets with
      D_an < D_ap; the strict semi-hard filter keeps only those with
      D_ap < D_an. The intersection is empty by construction -- MEASURED on a
      real batch: 16814 mined, 0 surviving -- so train_loss stays exactly 0.0
      forever, no error is raised, and the run looks stable rather than broken.
      A GP sampling it would receive a perfectly reproducible objective value
      from a run that trained nothing.
      Pi: strict_semihard <- False whenever mining_strategy == "hard".

Pi is a deterministic, idempotent map from the 18 raw triples onto the 13 legal
ones (Pi(Pi(x)) == Pi(x) for every x). It is a PROJECTION, not a penalty: two
raw points can map to one config and therefore to one duplicated observation,
which a GP handles natively. A penalty would instead have taught the surrogate
that "hard" is bad for reasons that have nothing to do with "hard".

The activity mask A(l)
----------------------
gp_minimize requires a fixed-length vector, so every sampled point carries
every loss hyper-parameter whatever the sampled loss type. A(l) says which of
them the configured loss actually reads:

    A(triplet)   = {margin}
    A(joint)     = {angular_alpha_deg}
    A(joint_sep) = {angular_alpha_deg, lambda_sep, sep_warmup_frac}

    sep_centre_means was removed as an axis: the centred formulation of L_sep
    is invariant to scale and therefore blind to collapse (MEASURED: 0.000035
    against 2.248 raw on a collapsed batch), so the raw form is now the only
    one and there is nothing left to select.

    sep_warmup_frac (tau) was ADDED as an axis. It had been fixed at 0.3 on the
    argument that tau and lambda_sep trade off through the dose
    lambda_sep * T * (1 - tau/2) and would therefore trace a ridge. That
    argument does not hold: two settings with equal dose have different
    TERMINAL weights lambda_sep * g(T), and since the epoch selector usually
    picks a late epoch, the terminal weight plausibly governs the converged
    geometry more than the integral does. The dose is not a sufficient
    statistic, so tau carries a second degree of freedom and is searched.
    Its upper bound is DERIVED, not configured -- see search._loss_dims.

margin and angular_alpha_deg are never BOTH active, which is the point: both
bind on the within/between distance ratio, so a config that searched the pair
would move along a ridge. Under "joint"/"joint_sep" the margin is not unused --
JointTripletLoss still takes margin = 2 * m_cos for its hinge -- it is FIXED at
whatever the base config states, and is simply not searched.

HPC note (hpc-python-compat): pure ASCII, standard library only.
"""

__all__ = [
    "MINING_STRATEGIES",
    "LOSS_TYPES",
    "HEAD_POOL_OPS_LEVELS",
    "LOSS_HP_SUPERSET",
    "active_loss_hps",
    "sep_warmup_frac_cap",
    "reads_strict_semihard",
    "is_legal",
    "project_condition",
    "legal_conditions",
    "n_legal_conditions",
    "decode_head_pool_ops",
    "encode_head_pool_ops",
    "decode_head_fusion",
    "cell_name",
]

# ordered exactly as the factorial generator ordered them, so legal_conditions()
# and hpc/make_factorial_configs.grid() enumerate in the same sequence
MINING_STRATEGIES = ("hard", "easy_positive", "easy_pos_semihard_neg")
LOSS_TYPES = ("triplet", "joint", "joint_sep")

# the head_pool_ops factor, as the two levels the factorial used. Index 0 is the
# single-statistic head, index 1 the three-statistic one. The backbone reorders
# the ops canonically, so the tuple order here is documentation, not semantics.
HEAD_POOL_OPS_LEVELS = (("mean",), ("mean", "max", "std"))

# every loss hyper-parameter that ANY loss type reads, in one fixed order. The
# search space carries all of them in every point; A(l) selects.
LOSS_HP_SUPERSET = ("margin", "angular_alpha_deg", "lambda_sep",
                    "sep_warmup_frac")

_ACTIVE = {
    "triplet": ("margin",),
    "joint": ("angular_alpha_deg",),
    "joint_sep": ("angular_alpha_deg", "lambda_sep", "sep_warmup_frac"),
}

_MINING_TAG = {"hard": "h", "easy_positive": "ep",
               "easy_pos_semihard_neg": "epsh"}
_LOSS_TAG = {"triplet": "trip", "joint": "joint", "joint_sep": "jsep"}


def _check_mining(mining_strategy):
    m = str(mining_strategy)
    if m not in MINING_STRATEGIES:
        raise ValueError("unknown mining_strategy %r; expected one of %r"
                         % (mining_strategy, MINING_STRATEGIES))
    return m


def _check_loss(loss_type):
    l = str(loss_type)
    if l not in LOSS_TYPES:
        raise ValueError("unknown loss_type %r; expected one of %r"
                         % (loss_type, LOSS_TYPES))
    return l


def sep_warmup_frac_cap(patience, max_epochs):
    """tau_max = min(1, P / E_max): the DERIVED upper bound on the warm-up.

    THE SINGLE SOURCE OF TRUTH for the cap. search._loss_dims clips the
    requested range to it, hpc/preflight_config.py reports it, and
    hpc/make_joint_search_config.py prints it; all three call this, so the
    number cannot drift between the space that is built and the number that is
    printed next to it.

    WHY A CAP AT ALL. A run reaches full separation weight only if it completes
    at least tau * T of its T = E_max * n_b planned steps, i.e. only if
    e_tilde / E_max >= tau, where e_tilde is the number of epochs the run
    actually completes. Early stopping with patience P means the earliest
    possible stop is at roughly P + 1 epochs, so e_tilde >= P is the
    conservative worst case. Requiring the ramp to complete EVEN FOR THE
    SHORTEST RUN THE STOPPING RULE PERMITS gives tau_max = min(1, P / E_max).

    Without it the optimiser can sample tau large enough that the separation
    term never reaches full weight before patience fires; "large tau wins"
    would then mean "the term was effectively off", which is a finding about
    joint versus joint_sep -- ALREADY a searched axis -- arrived at by a
    confounded route. The cap makes that region unsamplable rather than merely
    discouraged.

    WHY IT IS DERIVED RATHER THAN CONFIGURED. tau_max depends on P and E_max, so
    a static config field would go stale the moment either changes -- which is
    exactly what happens if the wall-clock gate forces the 60/20 reduction.

    Arguments
    ---------
    patience    : P, consecutive no-improvement epochs before stopping, P >= 1.
    max_epochs  : E_max, the hard epoch ceiling, E_max >= 1.

    Returns 1.0 (i.e. no cap) when either argument is non-positive, which is
    the "cannot be derived" case rather than a claim that no cap is needed.

    Pure arithmetic: no torch, no config, no skopt.
    """
    P = int(patience)
    E = int(max_epochs)
    if P <= 0 or E <= 0:
        return 1.0
    return min(1.0, float(P) / float(E))


def active_loss_hps(loss_type):
    """A(l): the loss hyper-parameters the given loss type actually READS.

    Returned in LOSS_HP_SUPERSET order, so callers that iterate the mask and
    callers that iterate the superset agree on ordering without sorting.
    """
    return _ACTIVE[_check_loss(loss_type)]


def reads_strict_semihard(loss_type):
    """Whether the strict semi-hard filter is read at all under this loss.

    False for "triplet" (losses.TripletMarginLoss has no such filter), True for
    the composite objectives.
    """
    return _check_loss(loss_type) != "triplet"


def is_legal(mining_strategy, loss_type, strict_semihard):
    """True iff the (mining, loss, strict) triple is one of the 13 legal ones.

    A triple is legal iff it is its own projection, which is the definition
    that cannot drift away from project_condition().
    """
    m = _check_mining(mining_strategy)
    l = _check_loss(loss_type)
    s = bool(strict_semihard)
    return (m, l, s) == project_condition(m, l, s)


def project_condition(mining_strategy, loss_type, strict_semihard):
    """Pi: the legality projection. Returns (mining, loss_type, strict).

    Deterministic and idempotent. Only strict_semihard ever moves; the mining
    strategy and the loss type are always returned unchanged, so the projection
    can never silently swap the experiment for a different one.

    Raises ValueError on an unknown level, which is deliberate: an unrecognised
    mining strategy is a wiring bug in the caller, not a point to be projected.
    """
    m = _check_mining(mining_strategy)
    l = _check_loss(loss_type)
    s = bool(strict_semihard)
    if not reads_strict_semihard(l):        # clause (a): inert under triplet
        s = False
    elif m == "hard":                       # clause (b): the provably empty cell
        s = False
    return (m, l, s)


def legal_conditions():
    """The 13 legal (mining, loss_type, strict_semihard) triples, deduplicated.

    Enumeration order matches hpc/make_factorial_configs.grid(): mining outer,
    loss inner, filter innermost.
    """
    out = []
    for m in MINING_STRATEGIES:
        for l in LOSS_TYPES:
            for s in (False, True):
                cond = project_condition(m, l, s)
                if cond not in out:
                    out.append(cond)
    return out


def n_legal_conditions():
    """13. Stated as a function so the number is never hard-coded twice."""
    return len(legal_conditions())


def decode_head_pool_ops(code):
    """Search coordinate (0 or 1) -> the head_pool_ops tuple it denotes."""
    i = int(code)
    if i not in (0, 1):
        raise ValueError("head_pool_ops code must be 0 or 1; got %r" % (code,))
    return HEAD_POOL_OPS_LEVELS[i]


def encode_head_pool_ops(ops):
    """head_pool_ops (any order) -> the search coordinate 0 or 1.

    Compared as SETS, because the backbone reorders the ops canonically, so
    ["max", "mean", "std"] and ["mean", "max", "std"] are the same head.
    """
    want = frozenset(str(o) for o in ops)
    for i, level in enumerate(HEAD_POOL_OPS_LEVELS):
        if frozenset(level) == want:
            return i
    raise ValueError(
        "head_pool_ops %r is not one of the two searched levels %r"
        % (tuple(ops), HEAD_POOL_OPS_LEVELS))


def decode_head_fusion(code):
    """Search coordinate (0 or 1) -> the head_fusion bool it denotes."""
    i = int(code)
    if i not in (0, 1):
        raise ValueError("head_fusion code must be 0 or 1; got %r" % (code,))
    return bool(i)


def cell_name(mining_strategy, loss_type, strict_semihard, head_fusion,
              head_pool_ops):
    """The historical factorial cell name, e.g. "ep_jsep_filtON_multimean".

    Reproduces hpc/make_factorial_configs.grid() exactly, so a joint-search
    trial can be matched against the screening cell that shares its condition.
    The filter tag is ABSENT (not "_filtOFF") under "triplet", because the
    filter does not exist there; that is also what the generator emitted.

    The condition is projected first, so an illegal input is named by the legal
    cell it maps to rather than by a cell that never existed.
    """
    m, l, s = project_condition(mining_strategy, loss_type, strict_semihard)
    if not reads_strict_semihard(l):
        filt = ""
    else:
        filt = "_filtON" if s else "_filtOFF"
    head_h = "multi" if bool(head_fusion) else "single"
    head_p = "all" if encode_head_pool_ops(head_pool_ops) == 1 else "mean"
    return "%s_%s%s_%s%s" % (_MINING_TAG[m], _LOSS_TAG[l], filt, head_h, head_p)
