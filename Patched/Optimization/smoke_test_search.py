"""
smoke_test_search.py

Standalone correctness checks for search.py (Stage 7). CPU only, synthetic data,
tiny net / few epochs / few trials. Headless. No data files.

Run:
    python3 smoke_test_search.py

Checks:
  [A] Space types are LOAD-BEARING (legacy BUG 2): block_family is Categorical
      (never Real -> a float can never index the block family list); depth_exponent
      and embedding_size are Integer; sampled values are usable as indices/sizes.
  [B] get_newspace addresses columns BY NAME (legacy BUG 1: es read the ws column).
      Built on a synthetic OptimizeResult with KNOWN per-column values, the refined
      embedding_size range must bracket the embedding values of the best points --
      NOT some other column's.
  [C] Degenerate-range guard (legacy BUG 4): when every best point shares one value,
      skopt's Integer(v, v) / Real(v, v) RAISE. get_newspace must instead return a
      valid dimension (widened within the original bounds, or pinned Categorical),
      and every returned dimension must be a legal skopt type.
  [D] No log-uniform in the refined arch space (legacy BUG 3): skopt raises on
      log-uniform with a zero lower bound. Assert no refined dimension carries a log
      prior.
  [E] OBJECTIVE IDENTITY: the search provably calls the SAME train() the final run
      calls. Asserted by PATCHING train.train and recording every invocation --
      argument-for-argument -- not by reading the source.
  [F] The objective is -mean(best VALIDATION ARI) over n_seeds, and the per-seed std
      is logged. Verified against a stubbed train() with KNOWN per-seed histories.
  [G] Per-trial seeds are DISJOINT across trials and reproducible:
      trial t uses seeds [s0 + t*N_s, s0 + t*N_s + N_s).
  [H] A failing / invalid config does NOT kill the study: it scores FAILED_OBJECTIVE
      (finite, worse than any real trial) and the search continues. NaN is never
      returned (gp_minimize cannot fit NaN).
  [I] Reproducibility: a fixed gp_random_state reproduces the trial SEQUENCE exactly.
  [J] End-to-end tiny search: phase 1 -> phase 2 -> get_newspace -> retune all run
      on synthetic data, and the winning configs are valid ExperimentConfigs.
  [K] Betas: the search samples u = 1 - beta in LOG space; the built config carries
      beta = 1 - u. Asserted on the space priors AND on the built config.
"""

import sys
import tempfile
import warnings
from dataclasses import replace

import numpy as np
from skopt.space import Categorical, Integer, Real

from config import (
    ExperimentConfig, DataConfig, TrainConfig, SearchConfig, EvalConfig,
    RuntimeConfig, BackboneConfig, AugmentationConfig,
)
from preprocessing_cache import cache_traces, load_cached_traces
from data_splits import (
    MultiClassSyntheticProvider, make_synthetic_specs, make_time_segment_splits,
)
import search as S
from search import (
    arch_space, train_space, get_newspace, config_from_arch_point,
    config_from_train_point, evaluate_candidate, search_architecture,
    search_training, retune_architecture, best_arch_dict, best_train_dict,
    FAILED_OBJECTIVE, _ARCH_NAMES,
)

_DURATION_S = 120.0
_FS = 50.0
_WINDOW_S = 8.0


def _tiny_cfg(C=2, n_calls=5, n_seeds=2, max_epochs=2):
    aug = replace(AugmentationConfig(fs=_FS), n_positives=2, n_negatives=2,
                  shift_magnitude_s=2.0)
    data = DataConfig(
        data_mode="synthetic", synthetic_n_per_class=tuple([2] * C),
        synthetic_duration_s=_DURATION_S, synthetic_fs=_FS,
        window_s=_WINDOW_S, train_stride_s=4.0, eval_stride_s=8.0,
        split_fractions=(0.6, 0.2, 0.2), augmentation=aug,
    )
    bb = BackboneConfig(depth_exponent=3, width_multiplier=1.5, stem_width=8,
                        embedding_size=8)
    tr = TrainConfig(max_epochs=max_epochs, patience=max_epochs, n_seeds=n_seeds,
                     windows_per_condition=2, batches_per_epoch=2, lr=3e-3)
    se = SearchConfig(
        depth_exponent_range=(3, 4), width_multiplier_range=(1.5, 2.0),
        block_family_choices=(0, 1), embedding_size_range=(8, 10),
        n_calls_arch=n_calls, n_calls_train=n_calls, gp_random_state=0,
        refine_top_fraction=0.5,
    )
    rt = RuntimeConfig(seed=100, device="cpu", num_workers=0, torch_threads=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return ExperimentConfig(data=data, backbone=bb, train=tr, search=se,
                                eval=EvalConfig(), runtime=rt)


def _make_splits(C, cfg, cache_dir):
    provider = MultiClassSyntheticProvider(n_classes=C, duration_s=_DURATION_S,
                                           fs=_FS, seed=0)
    specs = make_synthetic_specs(cfg.data.synthetic_n_per_class)
    cache_traces(specs, provider, cache_dir)
    traces, conditions, fs = load_cached_traces(cache_dir)
    return make_time_segment_splits(traces, conditions, fs, cfg.data, base_seed=0)


class _FakeResult(object):
    """A minimal stand-in for skopt's OptimizeResult, with KNOWN columns, so
    get_newspace can be tested without running a real search."""

    def __init__(self, x_iters, func_vals):
        self.x_iters = list(x_iters)
        self.func_vals = np.asarray(func_vals, dtype=float)


# --------------------------------------------------------------------------- #
def check_space_types():
    cfg = _tiny_cfg()
    sp = arch_space(cfg.search)
    names = [d.name for d in sp]
    assert names == list(_ARCH_NAMES), names
    kinds = {d.name: type(d).__name__ for d in sp}
    assert kinds["block_family"] == "Categorical", kinds
    assert kinds["depth_exponent"] == "Integer", kinds
    assert kinds["embedding_size"] == "Integer", kinds
    assert kinds["width_multiplier"] == "Real", kinds

    # a sampled block_family must be usable as an INDEX (the legacy Real crashed here)
    blk_dim = [d for d in sp if d.name == "block_family"][0]
    families = ["basic", "bottleneck"]
    for v in blk_dim.rvs(5, random_state=0):
        assert isinstance(v, (int, np.integer)), (v, type(v))
        _ = families[int(v)]                         # would TypeError on a float
    # depth / embedding sampled values are ints
    for d in (sp[0], sp[3]):
        for v in d.rvs(5, random_state=0):
            assert isinstance(v, (int, np.integer)), (d.name, v, type(v))
    print("  [A] space types OK: block_family Categorical (indexable), depth/"
          "embedding Integer, width Real (legacy BUG 2 unrepresentable)")


def check_get_newspace_by_name():
    """[B] The legacy es = Real(lower_bounds[3], upper_bounds[3]) read the WS column.
    Build a result whose columns are deliberately DISJOINT so a wrong-column read is
    detectable, and assert the refined es range brackets the ES values."""
    cfg = _tiny_cfg()
    cfg.search = replace(cfg.search, depth_exponent_range=(3, 6),
                         width_multiplier_range=(1.5, 3.0),
                         embedding_size_range=(8, 16))
    # columns: depth (3..6), width (1.5..3.0), block (0/1), embedding (8..16)
    x_iters = [
        [3, 1.60, 0, 8],     # best
        [4, 1.90, 0, 9],     # 2nd best
        [6, 2.90, 1, 16],    # worst  (deliberately far away)
        [5, 2.50, 1, 14],    # 3rd
    ]
    func_vals = [-0.9, -0.8, -0.1, -0.5]          # lower = better
    res = _FakeResult(x_iters, func_vals)

    sp = get_newspace(res, pers=0.5, search_cfg=cfg.search)   # keeps the top 2
    by = {d.name: d for d in sp}
    # the top-2 points have embedding 8 and 9 -> refined es MUST bracket [8, 9]
    es = by["embedding_size"]
    assert isinstance(es, Integer), type(es)
    assert es.low == 8 and es.high == 9, (es.low, es.high)
    # if it had read the DEPTH column it would be [3,4]; the WIDTH column [1.6,1.9]
    assert not (es.low == 3 and es.high == 4), "es read the DEPTH column!"
    # depth of the top 2 is 3..4
    d = by["depth_exponent"]
    assert d.low == 3 and d.high == 4, (d.low, d.high)
    # width of the top 2 is 1.60..1.90
    w = by["width_multiplier"]
    assert abs(w.low - 1.60) < 1e-9 and abs(w.high - 1.90) < 1e-9, (w.low, w.high)
    # block: only family 0 among the top 2 -> Categorical([0]), still Categorical
    b = by["block_family"]
    assert isinstance(b, Categorical), type(b)
    assert list(b.categories) == [0], list(b.categories)
    print("  [B] get_newspace addresses columns BY NAME OK: refined es=[8,9] "
          "(the ES column), depth=[3,4], width=[1.60,1.90] (legacy BUG 1 fixed)")


def check_degenerate_guard():
    """[C] Converged search -> every best point shares a value -> skopt would RAISE."""
    cfg = _tiny_cfg()
    cfg.search = replace(cfg.search, depth_exponent_range=(3, 6),
                         width_multiplier_range=(1.5, 3.0),
                         embedding_size_range=(8, 16))
    # first prove skopt really does raise on the degenerate ranges
    for cls, args in ((Integer, (5, 5)), (Real, (2.0, 2.0))):
        raised = False
        try:
            cls(*args)
        except ValueError:
            raised = True
        assert raised, "%s%r did not raise (assumption broken)" % (cls.__name__, args)

    # all four best points IDENTICAL -> every column collapses
    x_iters = [[5, 2.0, 1, 12]] * 3 + [[3, 1.5, 0, 8]]
    func_vals = [-0.9, -0.9, -0.9, -0.1]
    sp = get_newspace(_FakeResult(x_iters, func_vals), pers=0.5, search_cfg=cfg.search)
    assert len(sp) == 4
    for d in sp:
        assert isinstance(d, (Integer, Real, Categorical)), type(d)
        if isinstance(d, (Integer, Real)):
            assert d.low < d.high, (d.name, d.low, d.high)   # valid, non-degenerate
        # every dimension must still be SAMPLEABLE
        vals = d.rvs(3, random_state=0)
        assert len(vals) == 3
    by = {d.name: d for d in sp}
    # depth collapsed to 5 -> widened to [4,6] within the original (3,6)
    assert isinstance(by["depth_exponent"], Integer)
    assert by["depth_exponent"].low == 4 and by["depth_exponent"].high == 6, \
        (by["depth_exponent"].low, by["depth_exponent"].high)

    # unwidenable: an original range that is itself a single value -> pinned Categorical
    cfg2 = _tiny_cfg()
    cfg2.search = replace(cfg2.search, depth_exponent_range=(4, 4),
                          embedding_size_range=(8, 16))
    sp2 = get_newspace(_FakeResult([[4, 1.7, 0, 9]], [-0.5]), pers=1.0,
                       search_cfg=cfg2.search)
    dd = {d.name: d for d in sp2}["depth_exponent"]
    assert isinstance(dd, Categorical) and list(dd.categories) == [4], dd
    print("  [C] degenerate-range guard OK: collapsed depth widened to [4,6]; an "
          "unwidenable HP pinned as Categorical([4]) (legacy BUG 4 fixed)")


def check_no_log_prior_in_refined_space():
    """[D] skopt raises on log-uniform with a 0 lower bound; the refined ARCH space
    must carry NO log prior anywhere."""
    raised = False
    try:
        Real(0.0, 5.0, prior="log-uniform")
    except ValueError:
        raised = True
    assert raised, "skopt no longer raises on log-uniform with 0 (assumption broken)"

    cfg = _tiny_cfg()
    cfg.search = replace(cfg.search, depth_exponent_range=(3, 6),
                         width_multiplier_range=(1.5, 3.0),
                         embedding_size_range=(8, 16))
    res = _FakeResult([[3, 1.6, 0, 8], [4, 1.9, 1, 12]], [-0.9, -0.5])
    for d in get_newspace(res, pers=1.0, search_cfg=cfg.search):
        prior = getattr(d, "prior", None)
        assert prior != "log-uniform", (d.name, prior)
    # and the ARCH space itself has no log prior either
    for d in arch_space(cfg.search):
        assert getattr(d, "prior", None) != "log-uniform", d.name
    print("  [D] no log-uniform in the arch / refined space OK (legacy BUG 3 fixed)")


def check_objective_calls_the_same_train(cache_dir):
    """[E] + [F] + [G]: patch train.train, observe every call, and verify the
    objective, the seed blocks, and the logged std -- all from the OBSERVED calls."""
    C = 2
    cfg = _tiny_cfg(C=C, n_seeds=3)
    splits = _make_splits(C, cfg, cache_dir)

    calls = []
    # KNOWN per-seed histories: best val ARI = 0.2, 0.4, 0.9 -> mean 0.5
    fake_aris = {0: 0.2, 1: 0.4, 2: 0.9}

    def fake_train(cfg_in, train_ds, val_ds, device, seed, ckpt_dir=None,
                   verbose=False):
        calls.append({"cfg": cfg_in, "train_ds": train_ds, "val_ds": val_ds,
                      "device": device, "seed": seed})
        n = len(calls) - 1
        peak = fake_aris[n % 3]
        history = [
            {"epoch": 1, "ari": peak * 0.5, "ami": 0.0, "silhouette": 0.0,
             "health": {"eff_rank": 2.0}},
            {"epoch": 2, "ari": peak, "ami": 0.0, "silhouette": 0.0,   # the BEST epoch
             "health": {"eff_rank": 2.5}},
            {"epoch": 3, "ari": peak * 0.3, "ami": 0.0, "silhouette": 0.0,
             "health": {"eff_rank": 2.2}},
        ]
        return None, history

    orig = S.train
    S.train = fake_train
    try:
        cand = config_from_arch_point(cfg, [3, 1.7, 1, 8])
        obj, rec = evaluate_candidate(cand, splits, "cpu", trial_number=0)
    finally:
        S.train = orig

    # [E] the objective called train() -- with the TRAIN and VAL splits, on the device
    assert len(calls) == 3, len(calls)
    for c in calls:
        assert c["train_ds"] is splits.train, "objective did not pass the TRAIN split"
        assert c["val_ds"] is splits.val, "objective did not pass the VAL split"
        assert c["device"] == "cpu"
        assert isinstance(c["cfg"], ExperimentConfig)
    print("  [E] objective provably calls train() with (train_ds, val_ds, device) "
          "-- the SAME train() the final run uses (asserted by patching, not by "
          "reading the source)")

    # [F] objective == -mean(best val ARI); std logged
    expected_mean = (0.2 + 0.4 + 0.9) / 3.0
    expected_std = float(np.std([0.2, 0.4, 0.9]))
    assert abs(obj - (-expected_mean)) < 1e-12, (obj, -expected_mean)
    assert abs(rec["mean"] - expected_mean) < 1e-12
    assert abs(rec["std"] - expected_std) < 1e-12, (rec["std"], expected_std)
    assert rec["scores"] == [0.2, 0.4, 0.9], rec["scores"]
    assert not rec["failed"]
    print("  [F] objective = -mean(BEST val ARI over epochs) = %+.4f; per-seed std "
          "%.4f logged (not optimized) OK" % (obj, rec["std"]))

    # [G] seeds: trial t owns [s0 + t*N_s, s0 + t*N_s + N_s)
    s0, N_s = cfg.runtime.seed, cfg.train.n_seeds
    got = [c["seed"] for c in calls]
    assert got == [s0 + 0 * N_s + n for n in range(N_s)], got

    calls.clear()
    S.train = fake_train
    try:
        evaluate_candidate(cand, splits, "cpu", trial_number=1)
        evaluate_candidate(cand, splits, "cpu", trial_number=2)
    finally:
        S.train = orig
    seeds_t1 = [c["seed"] for c in calls[:3]]
    seeds_t2 = [c["seed"] for c in calls[3:]]
    assert seeds_t1 == [s0 + 1 * N_s + n for n in range(N_s)], seeds_t1
    assert seeds_t2 == [s0 + 2 * N_s + n for n in range(N_s)], seeds_t2
    assert not (set(seeds_t1) & set(seeds_t2)), "seed blocks OVERLAP across trials"
    print("  [G] per-trial seed blocks DISJOINT + reproducible OK "
          "(t=1 -> %r, t=2 -> %r)" % (seeds_t1, seeds_t2))


def check_failed_trial_does_not_kill_study(cache_dir):
    """[H] A raising train() (or an invalid config) must score FAILED_OBJECTIVE --
    finite, worse than any real trial -- never NaN (gp_minimize cannot fit NaN)."""
    C = 2
    cfg = _tiny_cfg(C=C, n_seeds=2)
    splits = _make_splits(C, cfg, cache_dir)

    def exploding_train(*a, **kw):
        raise RuntimeError("simulated trial failure")

    orig = S.train
    S.train = exploding_train
    try:
        cand = config_from_arch_point(cfg, [3, 1.7, 0, 8])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            obj, rec = evaluate_candidate(cand, splits, "cpu", trial_number=0)
    finally:
        S.train = orig

    assert obj == FAILED_OBJECTIVE, obj
    assert np.isfinite(obj), "objective must be FINITE (gp_minimize cannot fit NaN)"
    assert rec["failed"] is True
    # and FAILED must be strictly worse than any achievable score: ARI <= 1 -> obj >= -1
    assert FAILED_OBJECTIVE > 1.0 - 1e-9 or FAILED_OBJECTIVE > -(-1.0), FAILED_OBJECTIVE
    assert obj > -1.0, "FAILED must be worse than the best achievable objective (-1)"

    # a train() that returns only NaN metrics is also a failure, not a NaN objective
    def nan_train(*a, **kw):
        return None, [{"epoch": 1, "ari": float("nan"), "ami": float("nan"),
                       "silhouette": float("nan"),
                       "health": {"eff_rank": float("nan")}}]

    S.train = nan_train
    try:
        obj2, rec2 = evaluate_candidate(cand, splits, "cpu", trial_number=1)
    finally:
        S.train = orig
    assert obj2 == FAILED_OBJECTIVE and np.isfinite(obj2), obj2
    print("  [H] a failing/NaN trial scores FAILED_OBJECTIVE=%.1f (finite, worse than "
          "any real trial) and never NaN -> the study survives OK" % FAILED_OBJECTIVE)


def check_betas_log_and_converted():
    """[K] u = 1 - beta searched in LOG space; the config carries beta = 1 - u."""
    cfg = _tiny_cfg()
    sp = train_space(cfg.search)
    by = {d.name: d for d in sp}
    for nm in ("lr", "one_minus_beta1", "one_minus_beta2", "weight_decay"):
        assert by[nm].prior == "log-uniform", (nm, by[nm].prior)
    assert by["margin"].prior != "log-uniform"          # margin is plain Real

    c = config_from_train_point(cfg, [0.5, 1e-3, 0.02, 0.001, 1e-4])
    assert abs(c.train.beta1 - (1.0 - 0.02)) < 1e-12, c.train.beta1
    assert abs(c.train.beta2 - (1.0 - 0.001)) < 1e-12, c.train.beta2
    assert abs(c.train.lr - 1e-3) < 1e-15
    print("  [K] betas OK: lr/1-b1/1-b2/wd are LOG-uniform; config gets "
          "b1=%.4f b2=%.5f from u1=0.02 u2=0.001" % (c.train.beta1, c.train.beta2))


def check_reproducible_sequence(cache_dir):
    """[I] A fixed gp_random_state reproduces the trial SEQUENCE exactly."""
    C = 2
    cfg = _tiny_cfg(C=C, n_calls=5, n_seeds=1, max_epochs=1)
    splits = _make_splits(C, cfg, cache_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r1 = search_architecture(cfg, splits, "cpu")
        r2 = search_architecture(cfg, splits, "cpu")
    assert [list(x) for x in r1.x_iters] == [list(x) for x in r2.x_iters], \
        "the trial sequence is not reproducible under a fixed gp_random_state"
    assert np.allclose(r1.func_vals, r2.func_vals), (r1.func_vals, r2.func_vals)
    print("  [I] reproducible trial sequence under a fixed gp_random_state OK "
          "(%d identical trials)" % len(r1.x_iters))
    return cfg, splits, r1


def check_end_to_end(cfg, splits, res_arch):
    """[J] phase 1 -> phase 2 -> get_newspace -> retune, all producing valid configs."""
    best_arch = best_arch_dict(res_arch)
    assert set(best_arch) == set(_ARCH_NAMES), best_arch
    c_arch = config_from_arch_point(cfg, res_arch.x)
    assert isinstance(c_arch, ExperimentConfig)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_tr = search_training(cfg, splits, "cpu", best_arch)
    bt = best_train_dict(res_tr)
    assert 0.0 < bt["beta1"] < 1.0 and 0.0 < bt["beta2"] < 1.0, bt
    assert bt["lr"] > 0 and bt["weight_decay"] > 0, bt
    # the phase-2 winner must carry the phase-1 ARCHITECTURE, unchanged
    c_tr = config_from_train_point(cfg, res_tr.x, arch=best_arch)
    assert c_tr.backbone.depth_exponent == int(best_arch["depth_exponent"])
    assert c_tr.backbone.embedding_size == int(best_arch["embedding_size"])
    assert c_tr.backbone.block_family == int(best_arch["block_family"])

    # re-tune on the narrowed space, under the TUNED optimizer
    cfg_tuned = ExperimentConfig.from_dict(cfg.to_dict())
    cfg_tuned.train.lr = bt["lr"]
    cfg_tuned.train.margin = bt["margin"]
    cfg_tuned.train.beta1 = bt["beta1"]
    cfg_tuned.train.beta2 = bt["beta2"]
    cfg_tuned.train.weight_decay = bt["weight_decay"]
    cfg_tuned.validate()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_re = retune_architecture(cfg_tuned, splits, "cpu", res_arch)
    assert len(res_re.x_iters) == cfg.search.n_calls_arch
    c_re = config_from_arch_point(cfg_tuned, res_re.x)
    assert isinstance(c_re, ExperimentConfig)

    # every trial's objective must be FINITE (never NaN) in all three phases
    for tag, r in (("arch", res_arch), ("train", res_tr), ("retune", res_re)):
        assert np.all(np.isfinite(r.func_vals)), (tag, r.func_vals)
        assert len(r.trial_log) == len(r.x_iters), (tag, len(r.trial_log))

    print("  [J] end-to-end OK: phase1 (%d trials, best obj %+.4f) -> phase2 "
          "(%d trials, best obj %+.4f) -> retune (%d trials, best obj %+.4f); all "
          "objectives finite; phase-2 config keeps the phase-1 architecture"
          % (len(res_arch.x_iters), res_arch.fun, len(res_tr.x_iters), res_tr.fun,
             len(res_re.x_iters), res_re.fun))


def check_partial_seed_failure_is_failed(cache_dir):
    """[L] A trial where only SOME seeds completed must be FAILED, not scored on the
    survivors.

    Why: seed-averaging (decision 3) is the noise control. A trial scored on 1 seed
    and one scored on n_seeds are not comparable. Under the naive "use what survived"
    policy, a config that crashed on 2 of 3 seeds but got lucky on the third reports
    mean 0.95 with std 0.00 -- MORE attractive to the GP than an honest 0.90 +/- 0.05
    -- so the surrogate would steer the search straight into the flaky region.
    """
    C = 2
    cfg = _tiny_cfg(C=C, n_seeds=3)
    splits = _make_splits(C, cfg, cache_dir)
    cand = config_from_arch_point(cfg, [3, 1.7, 0, 8])

    n = {"i": 0}

    def flaky_train(*a, **kw):
        n["i"] += 1
        if n["i"] in (1, 2):                      # 2 of 3 seeds crash
            raise RuntimeError("simulated seed crash")
        return None, [{"epoch": 1, "ari": 0.95, "ami": 0.0, "silhouette": 0.0,
                       "health": {"eff_rank": 3.0}}]   # the survivor scores WELL

    orig = S.train
    S.train = flaky_train
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj, rec = evaluate_candidate(cand, splits, "cpu", trial_number=0)
    finally:
        S.train = orig

    assert obj == FAILED_OBJECTIVE, \
        ("a 1-of-3-seed trial was scored as a real result (obj=%r) -- a flaky config "
         "can win the search on one lucky seed" % obj)
    assert rec["failed"] is True, rec
    assert rec["n_seeds_ok"] == 1 and rec["n_seeds"] == 3, rec
    assert np.isnan(rec["mean"]), rec["mean"]
    assert any("only 1 of 3 seeds" in str(x.message) for x in w), \
        "no warning about the partial seed failure"

    # a trial where ALL seeds complete is still scored normally
    def good_train(*a, **kw):
        return None, [{"epoch": 1, "ari": 0.8, "ami": 0.0, "silhouette": 0.0,
                       "health": {"eff_rank": 3.0}}]

    S.train = good_train
    try:
        obj2, rec2 = evaluate_candidate(cand, splits, "cpu", trial_number=1)
    finally:
        S.train = orig
    assert not rec2["failed"] and rec2["n_seeds_ok"] == 3, rec2
    assert abs(obj2 - (-0.8)) < 1e-12, obj2
    print("  [L] partial seed failure -> FAILED OK (1 of 3 seeds -> obj %.1f, not "
          "-0.95); a full 3-of-3 trial still scores normally (obj %+.2f)"
          % (obj, obj2))


def main():
    print("Running search smoke tests...")
    check_space_types()
    check_get_newspace_by_name()
    check_degenerate_guard()
    check_no_log_prior_in_refined_space()
    check_betas_log_and_converted()
    with tempfile.TemporaryDirectory() as d:
        check_objective_calls_the_same_train(d)
    with tempfile.TemporaryDirectory() as d:
        check_failed_trial_does_not_kill_study(d)
    with tempfile.TemporaryDirectory() as d:
        check_partial_seed_failure_is_failed(d)
    with tempfile.TemporaryDirectory() as d:
        cfg, splits, res_arch = check_reproducible_sequence(d)
        check_end_to_end(cfg, splits, res_arch)
    print("ALL SEARCH SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
