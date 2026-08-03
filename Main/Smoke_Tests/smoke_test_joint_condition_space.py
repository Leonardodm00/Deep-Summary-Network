"""
smoke_test_joint_condition_space.py
===================================

Acceptance tests for Stages 1-4 of the joint-GP-search implementation: the
config surface, the legality projection Pi, the loss-HP triple, and the 18-axis
joint condition space.

Nothing here trains. Every check is a pure-function or config-construction
check, so the whole file runs in seconds on a login node and needs no data, no
GPU and no cluster allocation.

WHAT EACH CHECK GUARDS
----------------------
  [1-A] Stage 1: JSON round-trip equality for the new TrainConfig fields.
  [1-B] Stage 1: JSON round-trip equality for the new SearchConfig fields.
  [1-C] Stage 1: an UNMODIFIED pre-existing config keeps identical semantics --
        the new fields default to current behaviour, so a config written before
        this change parses to the same object it always did.
  [1-D] Stage 1: validation rejects an empty choice list, an unknown level, a
        duplicate level, and tau outside [0, 1].

  [2-A] Stage 2: all 18 raw (mining, loss, strict) triples map into 13 legal
        ones.
  [2-B] Stage 2: Pi is idempotent, Pi(Pi(x)) == Pi(x), on all 18.
  [2-C] Stage 2: no projected combination is rejected by
        TrainConfig.__post_init__.
  [2-D] Stage 2: the 13 legal triples x 4 head geometries reproduce EXACTLY the
        52 historical cell names emitted by hpc/make_factorial_configs.grid().
  [2-E] Stage 2: the provably-empty cell is unreachable -- Pi kills it, and
        TrainConfig raises on it if it is ever built directly.

  [3-A] Stage 3: names <-> dimensions <-> writer agree, for each loss type, in
        both staged and superset mode.
  [3-B] Stage 3: the staged space is UNCHANGED by this work (1 dim for triplet,
        1 for joint, 2 for joint_sep -- sep_centre_means is not a staged axis).
  [3-C] Stage 3: inactive coordinates provably do not reach the config.
  [3-D] Stage 3: two points differing ONLY in inactive coordinates build
        BYTE-IDENTICAL configs (decision D1).

  [4-A] Stage 4: the space has 18 axes and 22 surrogate columns, and the axis
        names match joint_condition_names() in order.
  [4-B] Stage 4: point -> config -> point round-trip on the searched fields.
  [4-C] Stage 4: the five binaries sample as USABLE integers (the BUG 2
        property), and project to native, JSON-serialisable Python ints.
  [4-D] Stage 4: head_pool_ops decodes to the correct list.
  [4-E] Stage 4: sampled points cover all 13 conditions and all 4 heads given
        enough draws.
  [4-F] Stage 4: every sampled point builds a VALID config (Pi has removed the
        empty cell, so no draw can raise).

HOW TO RUN
----------
    cd Main
    python3 Smoke_Tests/smoke_test_joint_condition_space.py

    # or, from the repository root
    PYTHONPATH=Main python3 Main/Smoke_Tests/smoke_test_joint_condition_space.py

Exit code 0 and "ALL SMOKE TESTS PASSED" on success; any failure raises with a
diagnostic naming the offending case.

HPC note (hpc-python-compat): pure ASCII.
"""

import copy
import itertools
import json
import os
import sys
import warnings

# make Main/ importable whether this is run from Main/ or from Smoke_Tests/
_HERE = os.path.dirname(os.path.abspath(__file__))
_MAIN = os.path.dirname(_HERE)
for _p in (_MAIN, os.path.join(_MAIN, "hpc")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from skopt.space import Space

import condition_space as CS
import search as S
from config import ExperimentConfig, SearchConfig, TrainConfig


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _base_cfg(loss_type="triplet", n_classes_ok=True):
    """A small, valid ExperimentConfig to build trials from."""
    cfg = ExperimentConfig()
    cfg.train.loss_type = loss_type
    cfg.train = TrainConfig(**{f: getattr(cfg.train, f)
                               for f in cfg.train.__dataclass_fields__})
    return cfg


def _raw_triples():
    """All 18 raw (mining, loss, strict) combinations."""
    return list(itertools.product(CS.MINING_STRATEGIES, CS.LOSS_TYPES,
                                  (False, True)))


def _expect(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _draw(dims, n, seed):
    """Sample n points from a list of skopt dimensions.

    The space builders return a plain LIST, which is what gp_minimize's
    dimensions argument takes; skopt.space.Space is what knows how to sample.
    Wrapping here keeps the builders' return type unchanged.
    """
    return [list(pt) for pt in Space(dims).rvs(n_samples=n, random_state=seed)]


# --------------------------------------------------------------------------- #
# Stage 1 -- the config surface
# --------------------------------------------------------------------------- #
def test_1a_train_round_trip():
    for centre, tau in ((None, 0.0), (True, 0.3), (False, 1.0)):
        t = TrainConfig(loss_type="joint_sep", sep_centre_means=centre,
                        sep_warmup_frac=tau)
        cfg = ExperimentConfig()
        cfg.train = t
        back = ExperimentConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
        _expect(back.train.sep_centre_means is centre,
                "sep_centre_means round-trip: %r -> %r"
                % (centre, back.train.sep_centre_means))
        _expect(back.train.sep_warmup_frac == tau,
                "sep_warmup_frac round-trip: %r -> %r"
                % (tau, back.train.sep_warmup_frac))
    print("  [1-A] TrainConfig new fields round-trip through JSON OK")


def test_1b_search_round_trip():
    s = SearchConfig(mining_strategy_choices=("hard", "easy_positive"),
                     loss_type_choices=("joint", "joint_sep"),
                     strict_semihard_choices=(0, 1),
                     head_fusion_choices=(1,),
                     head_pool_ops_choices=(0,),
                     sep_centre_means_choices=(0, 1),
                     n_calls_joint=300, n_initial_points_joint=40)
    cfg = ExperimentConfig()
    cfg.search = s
    back = ExperimentConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    for f in ("mining_strategy_choices", "loss_type_choices",
              "strict_semihard_choices", "head_fusion_choices",
              "head_pool_ops_choices", "sep_centre_means_choices"):
        want, got = getattr(s, f), getattr(back.search, f)
        _expect(tuple(want) == tuple(got),
                "%s round-trip: %r -> %r" % (f, want, got))
        _expect(isinstance(got, tuple),
                "%s came back as %s, not tuple (JSON lists must be coerced)"
                % (f, type(got).__name__))
    _expect(back.search.n_calls_joint == 300, "n_calls_joint round-trip")
    _expect(back.search.n_initial_points_joint == 40,
            "n_initial_points_joint round-trip")
    print("  [1-B] SearchConfig new fields round-trip through JSON OK")


def test_1c_legacy_config_semantics():
    """A config written BEFORE this change must parse identically.

    Simulated by deleting every new key from a serialised config and checking
    the reconstructed object equals the default-constructed one on those keys.
    """
    new_train = ("sep_centre_means", "sep_warmup_frac")
    new_search = ("mining_strategy_choices", "loss_type_choices",
                  "strict_semihard_choices", "head_fusion_choices",
                  "head_pool_ops_choices", "sep_centre_means_choices",
                  "n_initial_points_joint")
    d = ExperimentConfig().to_dict()
    for k in new_train:
        d["train"].pop(k)
    for k in new_search:
        d["search"].pop(k)
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # an unknown-key warning = fail
        back = ExperimentConfig.from_dict(d)
    _expect(back.train.sep_centre_means is None,
            "sep_centre_means must default to None (the automatic rule), got %r"
            % (back.train.sep_centre_means,))
    _expect(back.train.sep_warmup_frac == 0.0,
            "sep_warmup_frac must default to 0.0 (full weight from step 1), "
            "got %r" % (back.train.sep_warmup_frac,))
    ref = ExperimentConfig()
    for k in new_search:
        _expect(getattr(back.search, k) == getattr(ref.search, k),
                "%s default changed" % k)
    # and the pre-existing fields are untouched
    _expect(back.to_dict()["train"]["loss_type"] == "triplet",
            "default loss_type changed")
    print("  [1-C] a pre-existing config parses with identical semantics OK")


def test_1d_validation():
    bad = [
        ("empty mining choices", dict(mining_strategy_choices=())),
        ("empty loss choices", dict(loss_type_choices=())),
        ("empty binary choices", dict(head_fusion_choices=())),
        ("unknown mining level", dict(mining_strategy_choices=("hard", "nope"))),
        ("unknown loss level", dict(loss_type_choices=("triplet", "nope"))),
        ("unknown binary level", dict(strict_semihard_choices=(0, 2))),
        ("duplicate level", dict(head_pool_ops_choices=(0, 0))),
        ("negative n_initial_points_joint", dict(n_initial_points_joint=-1)),
    ]
    for label, kwargs in bad:
        try:
            SearchConfig(**kwargs)
        except ValueError:
            continue
        raise AssertionError("SearchConfig accepted %s: %r" % (label, kwargs))
    for tau in (-0.1, 1.1):
        try:
            TrainConfig(sep_warmup_frac=tau)
        except ValueError:
            continue
        raise AssertionError("TrainConfig accepted sep_warmup_frac=%r" % (tau,))
    try:
        TrainConfig(sep_centre_means="yes")
    except ValueError:
        pass
    else:
        raise AssertionError("TrainConfig accepted a non-bool sep_centre_means")
    print("  [1-D] validation rejects empty / unknown / duplicate / out-of-range OK")


# --------------------------------------------------------------------------- #
# Stage 2 -- the legality layer
# --------------------------------------------------------------------------- #
def test_2a_eighteen_to_thirteen():
    raw = _raw_triples()
    _expect(len(raw) == 18, "expected 18 raw triples, got %d" % len(raw))
    projected = set(CS.project_condition(*t) for t in raw)
    legal = set(CS.legal_conditions())
    _expect(projected == legal,
            "Pi(18 raw) = %d distinct, legal_conditions() = %d; symmetric "
            "difference %r" % (len(projected), len(legal),
                               projected ^ legal))
    _expect(len(legal) == 13,
            "expected 13 legal conditions, got %d: %r" % (len(legal), legal))
    print("  [2-A] 18 raw triples project onto exactly 13 legal ones OK")


def test_2b_idempotent():
    for t in _raw_triples():
        once = CS.project_condition(*t)
        twice = CS.project_condition(*once)
        _expect(once == twice, "Pi not idempotent at %r: %r -> %r"
                % (t, once, twice))
        _expect(CS.is_legal(*once), "Pi(%r) = %r is not is_legal" % (t, once))
    print("  [2-B] Pi is idempotent on all 18 raw triples OK")


def test_2c_projection_accepted_by_trainconfig():
    for m, l, s in CS.legal_conditions():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # the INERT lambda_sep warning
            TrainConfig(mining_strategy=m, loss_type=l, strict_semihard=s)
    print("  [2-C] every legal condition is accepted by TrainConfig OK")


def test_2d_reproduces_52_cell_names():
    """The 13 legal triples x 4 heads must reproduce the historical 52 names."""
    try:
        import make_factorial_configs as MFC
    except ImportError:
        print("  [2-D] SKIPPED: hpc/make_factorial_configs.py not importable")
        return
    historical = sorted(cell["name"] for cell in MFC.grid())
    heads = ((False, ("mean",)), (False, ("mean", "max", "std")),
             (True, ("mean",)), (True, ("mean", "max", "std")))
    ours = sorted(CS.cell_name(m, l, s, fusion, ops)
                  for (m, l, s) in CS.legal_conditions()
                  for (fusion, ops) in heads)
    _expect(len(historical) == 52,
            "the generator emitted %d cells, expected 52" % len(historical))
    _expect(ours == historical,
            "cell names differ.\n  only ours: %r\n  only theirs: %r"
            % (sorted(set(ours) - set(historical)),
               sorted(set(historical) - set(ours))))
    print("  [2-D] 13 conditions x 4 heads reproduce the 52 historical cell "
          "names EXACTLY OK")


def test_2e_empty_cell_unreachable():
    for l in ("joint", "joint_sep"):
        m, ll, s = CS.project_condition("hard", l, True)
        _expect(s is False,
                "Pi left the provably-empty cell alive: (hard, %s, True)" % l)
        _expect(not CS.is_legal("hard", l, True),
                "(hard, %s, True) reported legal" % l)
        # and if it is ever built directly, TrainConfig must still raise
        try:
            TrainConfig(mining_strategy="hard", loss_type=l,
                        strict_semihard=True)
        except ValueError:
            continue
        raise AssertionError(
            "TrainConfig accepted the provably-empty (hard, %s, strict) cell" % l)
    print("  [2-E] the provably-empty cell is unreachable through Pi and still "
          "raises if built directly OK")


# --------------------------------------------------------------------------- #
# Stage 3 -- the loss-HP triple
# --------------------------------------------------------------------------- #
def test_3a_names_dims_writer_agree():
    cfg = ExperimentConfig()
    for l in CS.LOSS_TYPES:
        t = TrainConfig(loss_type=l)
        for superset in (False, True):
            names = S.loss_hp_names(t, superset=superset)
            dims = S._loss_dims(cfg.search, t, superset=superset)
            _expect(tuple(d.name for d in dims) == tuple(names),
                    "loss_type=%s superset=%s: names %r != dims %r"
                    % (l, superset, names, tuple(d.name for d in dims)))
        _expect(set(S.loss_hp_names(t, superset=True))
                == set(CS.LOSS_HP_SUPERSET),
                "superset mode must return the full superset")
    print("  [3-A] names <-> dimensions agree for every loss type, both modes OK")


def test_3b_staged_space_unchanged():
    expect = {"triplet": ("margin",),
              "joint": ("angular_alpha_deg",),
              "joint_sep": ("angular_alpha_deg", "lambda_sep")}
    for l, want in expect.items():
        got = S.loss_hp_names(TrainConfig(loss_type=l))
        _expect(got == want,
                "STAGED phase-2 loss HPs changed for %s: %r (want %r). Any "
                "change here silently rewrites archived staged runs."
                % (l, got, want))
    _expect(S.loss_hp_names(None) == ("margin",),
            "loss_hp_names(None) must reproduce the legacy margin space")
    cfg = ExperimentConfig()
    _expect(len(S.train_space(cfg.search, TrainConfig(loss_type="joint_sep")))
            == 2 + 4,
            "staged train_space width changed for joint_sep")
    print("  [3-B] the STAGED phase-2 space is unchanged by this work OK")


def test_3c_inactive_never_reaches_config():
    cfg = ExperimentConfig()
    base_margin = cfg.train.margin
    base_alpha = cfg.train.angular_alpha_deg
    base_lambda = cfg.train.lambda_sep
    base_centre = cfg.train.sep_centre_means
    # a point carrying EVERY loss HP at values far from the base config's
    p = {"margin": 0.77, "angular_alpha_deg": 3.5, "lambda_sep": 7.25,
         "sep_centre_means": 1}
    for l in CS.LOSS_TYPES:
        c = ExperimentConfig()
        c.train.loss_type = l
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            S._write_loss_hps(c, p, loss_type=l)
        active = CS.active_loss_hps(l)
        got = {"margin": c.train.margin,
               "angular_alpha_deg": c.train.angular_alpha_deg,
               "lambda_sep": c.train.lambda_sep,
               "sep_centre_means": c.train.sep_centre_means}
        base = {"margin": base_margin, "angular_alpha_deg": base_alpha,
                "lambda_sep": base_lambda, "sep_centre_means": base_centre}
        for name in CS.LOSS_HP_SUPERSET:
            if name in active:
                want = (bool(p[name]) if name == "sep_centre_means"
                        else float(p[name]))
                _expect(got[name] == want,
                        "%s is ACTIVE under %s but was not written (%r != %r)"
                        % (name, l, got[name], want))
            else:
                _expect(got[name] == base[name],
                        "%s is INACTIVE under %s but reached the config "
                        "(%r != base %r)" % (name, l, got[name], base[name]))
    print("  [3-C] inactive loss HPs provably do not reach the config OK")


def test_3d_byte_identical_configs():
    """D1: two points differing ONLY in inactive coordinates -> same config."""
    base = ExperimentConfig()
    names = S.joint_condition_names()
    space = S.joint_condition_space(base.search, base.regularization, base.train)
    pt = _draw(space, 1, 7)[0]
    i_loss = names.index("loss_type")
    i_strict = names.index("strict_semihard")
    checked = 0
    for l in CS.LOSS_TYPES:
        a = list(pt)
        a[i_loss] = l
        a[i_strict] = 0                     # keep the condition legal for all l
        inactive = [n for n in CS.LOSS_HP_SUPERSET
                    if n not in CS.active_loss_hps(l)]
        b = list(a)
        for n in inactive:
            j = names.index(n)
            if n == "sep_centre_means":
                b[j] = 1 - int(a[j])
            elif n == "margin":
                b[j] = 0.95 if float(a[j]) < 0.9 else 0.15
            elif n == "angular_alpha_deg":
                b[j] = 19.0 if float(a[j]) < 18.0 else 3.0
            elif n == "lambda_sep":
                b[j] = 0.9 if float(a[j]) < 0.5 else 0.02
        _expect(a != b, "test bug: no inactive coordinate was perturbed for %s" % l)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ca = S.config_from_joint_condition_point(base, a)
            cb = S.config_from_joint_condition_point(base, b)
        ja = json.dumps(ca.to_dict(), sort_keys=True)
        jb = json.dumps(cb.to_dict(), sort_keys=True)
        _expect(ja == jb,
                "loss_type=%s: perturbing INACTIVE coordinates %r changed the "
                "config -- decision D1 is broken, and the GP will see two "
                "scattered observations where it should see one duplicate."
                % (l, inactive))
        checked += 1
    _expect(checked == 3, "expected to check all three loss types")
    print("  [3-D] points differing only in inactive coordinates build "
          "BYTE-IDENTICAL configs OK")


# --------------------------------------------------------------------------- #
# Stage 4 -- the joint condition space
# --------------------------------------------------------------------------- #
def test_4a_axis_and_column_count():
    cfg = ExperimentConfig()
    space = S.joint_condition_space(cfg.search, cfg.regularization, cfg.train)
    names = S.joint_condition_names()
    _expect(len(space) == 18, "expected 18 axes, got %d" % len(space))
    _expect(tuple(d.name for d in space) == tuple(names),
            "space axis order != joint_condition_names()")
    # surrogate columns: skopt one-hots Categorical, everything else is 1 column
    cols = sum(len(d.categories) if hasattr(d, "categories") else 1
               for d in space)
    _expect(cols == 22,
            "expected 22 surrogate columns, got %d. Column arithmetic: 11 "
            "numeric + 5 Integer binaries + 2 three-level Categorical (6) = 22."
            % cols)
    print("  [4-A] 18 declared axes, 22 surrogate columns, names in order OK")


def test_4b_round_trip():
    base = ExperimentConfig()
    names = S.joint_condition_names()
    space = S.joint_condition_space(base.search, base.regularization, base.train)
    for pt in _draw(space, 40, 11):
        pt, info = S.project_joint_condition_point(list(pt))
        p = dict(zip(names, pt))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = S.config_from_joint_condition_point(base, pt)
        _expect(cfg.backbone.depth_exponent == int(p["depth_exponent"]),
                "depth_exponent round-trip")
        _expect(abs(cfg.backbone.width_multiplier - float(p["width_multiplier"]))
                < 1e-12, "width_multiplier round-trip")
        _expect(cfg.backbone.block_family == int(p["block_family"]),
                "block_family round-trip")
        _expect(cfg.backbone.embedding_size == int(p["embedding_size"]),
                "embedding_size round-trip")
        _expect(abs(cfg.backbone.dropout - float(p["dropout"])) < 1e-12,
                "dropout round-trip")
        _expect(cfg.train.mining_strategy == str(p["mining_strategy"]),
                "mining_strategy round-trip")
        _expect(cfg.train.loss_type == str(p["loss_type"]),
                "loss_type round-trip")
        _expect(cfg.train.strict_semihard == bool(int(p["strict_semihard"])),
                "strict_semihard round-trip")
        _expect(cfg.backbone.head_fusion == CS.decode_head_fusion(p["head_fusion"]),
                "head_fusion round-trip")
        _expect(abs(cfg.train.beta1 - (1.0 - float(p["one_minus_beta1"]))) < 1e-12,
                "beta1 = 1 - u round-trip")
        _expect(abs(cfg.train.beta2 - (1.0 - float(p["one_minus_beta2"]))) < 1e-12,
                "beta2 = 1 - u round-trip")
        # and the ACTIVE loss HPs specifically
        for n in CS.active_loss_hps(str(p["loss_type"])):
            if n == "sep_centre_means":
                _expect(cfg.train.sep_centre_means == bool(int(p[n])),
                        "sep_centre_means round-trip")
            else:
                _expect(abs(getattr(cfg.train, n) - float(p[n])) < 1e-12,
                        "%s round-trip" % n)
    print("  [4-B] point -> config -> point round-trips on 40 samples OK")


def test_4c_binaries_are_usable_integers():
    """The one place this design knowingly departs from an in-code warning.

    BUG 2 in search.py warns that a REAL block_family yields floats such as
    0.37, so Block_array[0.37] raises TypeError. Integer is safe from that --
    but NOT because it returns Python ints: skopt's Integer.rvs returns
    numpy.int64. Two properties are therefore asserted separately:

      (i)  the RAW sample is integral and usable as a list index (the property
           BUG 2 is actually about), and is not a float;
      (ii) after project_joint_condition_point the value is a NATIVE Python
           int, because the trial log and the winner report are JSON.
    """
    import numbers
    block_array = ["ResNet", "ResNeXt"]           # stands in for Block_array
    cfg = ExperimentConfig()
    space = S.joint_condition_space(cfg.search, cfg.regularization, cfg.train)
    names = S.joint_condition_names()
    binaries = ("block_family", "strict_semihard", "head_fusion",
                "head_pool_ops", "sep_centre_means")
    for pt in _draw(space, 60, 3):
        for name in binaries:
            k = names.index(name)
            v = pt[k]
            _expect(isinstance(v, numbers.Integral),
                    "%s sampled as %r (%s), which is not integral. BUG 2: a "
                    "float here makes Block_array[v] raise TypeError."
                    % (name, v, type(v).__name__))
            _expect(not isinstance(v, float),
                    "%s sampled as a float: %r" % (name, v))
            _expect(int(v) in (0, 1),
                    "%s sampled outside {0, 1}: %r" % (name, v))
        # the property BUG 2 is really about: it must INDEX
        try:
            _ = block_array[pt[names.index("block_family")]]
        except TypeError as ex:
            raise AssertionError(
                "block_family is not usable as a list index: %s" % ex)
        native, _info = S.project_joint_condition_point(pt)
        for name in binaries:
            v = native[names.index(name)]
            _expect(type(v) is int,
                    "%s is %s after projection, not a native int -- the trial "
                    "log is JSON and numpy scalars are not serialisable"
                    % (name, type(v).__name__))
        # the whole projected point must survive json.dumps
        try:
            json.dumps(dict(zip(names, native)))
        except TypeError as ex:
            raise AssertionError("projected point is not JSON-serialisable: %s"
                                 % ex)
    print("  [4-C] the five binaries sample as usable integers, project to "
          "native ints, and serialise to JSON OK")


def test_4d_head_pool_ops_decode():
    _expect(CS.decode_head_pool_ops(0) == ("mean",),
            "code 0 must decode to ('mean',)")
    _expect(CS.decode_head_pool_ops(1) == ("mean", "max", "std"),
            "code 1 must decode to ('mean', 'max', 'std')")
    _expect(CS.encode_head_pool_ops(["mean"]) == 0, "encode ['mean']")
    _expect(CS.encode_head_pool_ops(["std", "mean", "max"]) == 1,
            "encode must compare as SETS (the backbone reorders canonically)")
    base = ExperimentConfig()
    names = S.joint_condition_names()
    j = names.index("head_pool_ops")
    space = S.joint_condition_space(base.search, base.regularization, base.train)
    seen = set()
    for pt in _draw(space, 30, 13):
        pt = list(pt)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = S.config_from_joint_condition_point(base, pt)
        want = CS.decode_head_pool_ops(pt[j])
        _expect(tuple(cfg.backbone.head_pool_ops) == want,
                "head_pool_ops decoded to %r, expected %r"
                % (tuple(cfg.backbone.head_pool_ops), want))
        seen.add(want)
    _expect(len(seen) == 2, "only saw head_pool_ops levels %r in 30 draws" % seen)
    print("  [4-D] head_pool_ops decodes to the correct list OK")


def test_4e_coverage():
    base = ExperimentConfig()
    names = S.joint_condition_names()
    space = S.joint_condition_space(base.search, base.regularization, base.train)
    conds, heads = set(), set()
    for pt in _draw(space, 400, 17):
        _pt, info = S.project_joint_condition_point(list(pt))
        conds.add(info["condition"])
        p = dict(zip(names, pt))
        heads.add((int(p["head_fusion"]), int(p["head_pool_ops"])))
    _expect(len(conds) == 13,
            "400 draws covered %d of the 13 conditions; missing %r"
            % (len(conds), sorted(set(CS.legal_conditions()) - conds)))
    _expect(len(heads) == 4,
            "400 draws covered %d of the 4 head geometries" % len(heads))
    print("  [4-E] 400 sampled points cover all 13 conditions and all 4 heads OK")


def test_4f_every_draw_builds():
    base = ExperimentConfig()
    space = S.joint_condition_space(base.search, base.regularization, base.train)
    n_projected = 0
    for pt in _draw(space, 200, 23):
        pt = list(pt)
        _p, info = S.project_joint_condition_point(pt)
        n_projected += int(info["projected"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = S.config_from_joint_condition_point(base, pt)
        _expect(CS.is_legal(cfg.train.mining_strategy, cfg.train.loss_type,
                            cfg.train.strict_semihard),
                "built an ILLEGAL config: %r"
                % ((cfg.train.mining_strategy, cfg.train.loss_type,
                    cfg.train.strict_semihard),))
        _expect(isinstance(info["cell"], str) and len(info["cell"]) > 0,
                "the trial log needs a cell name")
    _expect(n_projected > 0,
            "no draw out of 200 was projected -- the test is not exercising Pi")
    print("  [4-F] all 200 draws build a valid, legal config (%d were projected "
          "by Pi) OK" % n_projected)


# --------------------------------------------------------------------------- #
def main():
    print("Stage 1 -- config surface")
    test_1a_train_round_trip()
    test_1b_search_round_trip()
    test_1c_legacy_config_semantics()
    test_1d_validation()
    print("Stage 2 -- legality layer (Pi)")
    test_2a_eighteen_to_thirteen()
    test_2b_idempotent()
    test_2c_projection_accepted_by_trainconfig()
    test_2d_reproduces_52_cell_names()
    test_2e_empty_cell_unreachable()
    print("Stage 3 -- the loss-HP triple")
    test_3a_names_dims_writer_agree()
    test_3b_staged_space_unchanged()
    test_3c_inactive_never_reaches_config()
    test_3d_byte_identical_configs()
    print("Stage 4 -- the joint condition space")
    test_4a_axis_and_column_count()
    test_4b_round_trip()
    test_4c_binaries_are_usable_integers()
    test_4d_head_pool_ops_decode()
    test_4e_coverage()
    test_4f_every_draw_builds()
    print("\nALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
