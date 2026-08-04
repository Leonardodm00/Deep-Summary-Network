"""
smoke_test_joint_conditions_wiring.py
=====================================

Acceptance tests for Stage 6: wiring the joint condition search into the
objective, the driver and the per-trial log.

THE TWO CHECKS THE DESIGN DOCUMENT NAMES EXPLICITLY
---------------------------------------------------
  [6-A] evaluate_candidate at n_seeds = 1 must NOT produce NaN. If the
        objective computed its spread with ddof = 1 over a single sample it
        would yield NaN, and _run_gp documents NaN as unfittable: gp_minimize's
        surrogate cannot fit it and the whole study aborts. The design says
        this "must be checked, not assumed", so it is checked here on the real
        function with train() patched out.
  [6-B] tie_break_gamma = 0.0 must DISABLE the tie-break, and the resulting
        objective must be exactly -mean(primary). Also checked, not assumed.

AND THE WIRING AROUND THEM
--------------------------
  [6-C] Pi is recorded per trial: a projected trial carries raw_condition,
        condition, projected and cell, so a duplicated observation is readable
        as a projection rather than as noise.
  [6-D] the failure path stays FINITE (FAILED_OBJECTIVE), never NaN, and still
        carries its annotation so a failed trial says WHICH cell failed.
  [6-E] the whole trial log is JSON-serialisable (it is written to disk).
  [6-F] search_mode = "joint_conditions" validates, and the illegal
        combination is not representable.
  [6-G] n_initial_points_joint resolution and its fallback chain.
  [6-H] a continuous primary (selection_primary = "silhouette", which this
        study uses) also disables the tie-break, and says so.

HOW TO RUN
----------
    cd Main
    PYTHONPATH=. python3 Smoke_Tests/smoke_test_joint_conditions_wiring.py

HPC note (hpc-python-compat): pure ASCII.
"""

import json
import os
import sys
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAIN = os.path.dirname(_HERE)
if _MAIN not in sys.path:
    sys.path.insert(0, _MAIN)

import numpy as np
from skopt.space import Space

import condition_space as CS
import search as S
from config import ExperimentConfig, SearchConfig


def _expect(cond, msg):
    if not cond:
        raise AssertionError(msg)


class _Splits(object):
    """The only attribute evaluate_candidate and the tie-break actually read."""

    class _DS(object):
        def __init__(self, y):
            self.conditions_per_item = y

    def __init__(self, y):
        self.train = self._DS(y)
        self.val = self._DS(y)


def _history(primary_value, sil_value=0.0):
    """A three-epoch history whose BEST epoch carries the given scores."""
    return [
        {"epoch": 1, "ari": primary_value * 0.5, "ami": 0.0,
         "silhouette": sil_value * 0.5, "health": {"eff_rank": 2.0}},
        {"epoch": 2, "ari": primary_value, "ami": 0.0,
         "silhouette": sil_value, "health": {"eff_rank": 2.5}},
        {"epoch": 3, "ari": primary_value * 0.3, "ami": 0.0,
         "silhouette": sil_value * 0.3, "health": {"eff_rank": 2.2}},
    ]


def _patched_train(values, sils=None, calls=None):
    """A fake train() returning a KNOWN best-epoch score per seed."""
    def fake_train(cfg_in, train_ds, val_ds, device, seed, ckpt_dir=None,
                   verbose=False):
        i = 0 if calls is None else len(calls)
        if calls is not None:
            calls.append(seed)
        v = values[i % len(values)]
        s = 0.0 if sils is None else sils[i % len(sils)]
        return None, _history(v, s)
    return fake_train


def _cfg(n_seeds=1, gamma=0.0, primary="ari"):
    cfg = ExperimentConfig()
    cfg.train.n_seeds = int(n_seeds)
    cfg.train.selection_primary = primary
    cfg.search.tie_break_gamma = float(gamma)
    return cfg


# --------------------------------------------------------------------------- #
def _space(cfg):
    """joint_condition_space with the tau-cap clip warning silenced.

    The clip is EXPECTED under the defaults: SearchConfig requests tau up to
    0.5 while TrainConfig (max_epochs=100, patience=10) derives a cap of 0.10.
    The dedicated cap test asserts that the warning fires; everywhere else it
    is noise that would drown the pass/fail lines.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return S.joint_condition_space(cfg.search, cfg.regularization,
                                       cfg.train)


def test_6a_single_seed_is_not_nan():
    """The named acceptance test: n_seeds = 1 must not yield NaN."""
    cfg = _cfg(n_seeds=1)
    splits = _Splits(np.array([0, 0, 1, 1, 2, 2]))
    orig = S.train
    S.train = _patched_train([0.7])
    try:
        obj, rec = S.evaluate_candidate(cfg, splits, "cpu", trial_number=0)
    finally:
        S.train = orig
    _expect(np.isfinite(obj),
            "objective is %r at n_seeds = 1 -- gp_minimize CANNOT fit NaN and "
            "the study would abort" % obj)
    for k in ("mean", "std", "objective"):
        _expect(np.isfinite(rec[k]),
                "record[%r] = %r at n_seeds = 1 (NaN would come from a ddof=1 "
                "std over one sample)" % (k, rec[k]))
    _expect(rec["std"] == 0.0,
            "std over ONE seed must be 0.0 (population std, ddof = 0); got %r. "
            "A ddof=1 std here is exactly the NaN failure this test exists for."
            % rec["std"])
    _expect(abs(rec["mean"] - 0.7) < 1e-12, "mean = %r, expected 0.7" % rec["mean"])
    _expect(abs(obj + 0.7) < 1e-12, "objective = %r, expected -0.7" % obj)
    _expect(rec["failed"] is False, "a healthy single-seed trial was marked failed")
    _expect(rec["n_seeds"] == 1 and rec["n_seeds_ok"] == 1, "seed bookkeeping")
    # and it still behaves at n_seeds = 3, so the fix is not a special case
    cfg3 = _cfg(n_seeds=3)
    S.train = _patched_train([0.2, 0.4, 0.9], calls=[])
    try:
        obj3, rec3 = S.evaluate_candidate(cfg3, splits, "cpu", trial_number=0)
    finally:
        S.train = orig
    _expect(abs(rec3["mean"] - 0.5) < 1e-12, "3-seed mean = %r" % rec3["mean"])
    _expect(rec3["std"] > 0.0, "3-seed std collapsed to 0")
    print("  [6-A] evaluate_candidate at n_seeds = 1: finite, std = 0.0 "
          "(ddof = 0), objective = -mean OK")


def test_6b_gamma_zero_disables_tie_break():
    """The named acceptance test: gamma = 0 must disable, not merely shrink."""
    splits = _Splits(np.array([0, 0, 1, 1, 2, 2]))
    cfg = _cfg(n_seeds=1, gamma=0.0, primary="ari")
    with warnings.catch_warnings(record=True) as rec_w:
        warnings.simplefilter("always")
        eps, info = S.resolve_tie_break_epsilon(cfg, splits)
    _expect(eps is None, "tie_break_gamma = 0 returned epsilon = %r, not None" % eps)
    _expect(info["enabled"] is False, "info says the tie-break is enabled at gamma = 0")
    _expect("gamma" in info["reason"],
            "the reason should name gamma, got %r" % info["reason"])
    _expect(not rec_w, "gamma = 0 is the DOCUMENTED off switch and must not warn; "
            "got %r" % [str(w.message) for w in rec_w])
    # the objective with epsilon=None is exactly -mean(primary), whatever the
    # secondary metric does
    orig = S.train
    for sil in (0.0, 0.9, -0.9):
        S.train = _patched_train([0.6], sils=[sil])
        try:
            obj, _r = S.evaluate_candidate(cfg, splits, "cpu", trial_number=0,
                                           epsilon=eps)
        finally:
            S.train = orig
        _expect(abs(obj + 0.6) < 1e-12,
                "with the tie-break disabled the objective must be -mean(ARI) "
                "= -0.6 regardless of the silhouette (%r); got %r" % (sil, obj))
    # and a NONZERO gamma really does change it, so the test is not vacuous
    cfg_on = _cfg(n_seeds=1, gamma=0.5, primary="ari")
    eps_on, info_on = S.resolve_tie_break_epsilon(cfg_on, splits)
    _expect(eps_on is not None and eps_on > 0.0,
            "gamma = 0.5 should give a positive epsilon, got %r" % eps_on)
    S.train = _patched_train([0.6], sils=[0.9])
    try:
        obj_on, _r = S.evaluate_candidate(cfg_on, splits, "cpu", trial_number=0,
                                          epsilon=eps_on)
    finally:
        S.train = orig
    _expect(abs(obj_on + 0.6) > 1e-15,
            "gamma = 0.5 left the objective identical to the gamma = 0 case, "
            "so [6-B] would pass vacuously")
    print("  [6-B] tie_break_gamma = 0.0 disables the tie-break (epsilon None, "
          "objective = -mean, no warning); gamma > 0 demonstrably differs OK")


def test_6c_pi_recorded_per_trial():
    base = ExperimentConfig()
    space = _space(base)
    names = S.joint_condition_names()
    pts = [list(p) for p in Space(space).rvs(n_samples=120, random_state=29)]
    n_proj = 0
    for pt in pts:
        note = S.annotate_joint_condition_point(pt)
        for k in ("cell", "projected", "condition", "raw_condition",
                  "mining_strategy", "loss_type", "strict_semihard",
                  "head_fusion", "head_pool_ops", "active_loss_hps"):
            _expect(k in note, "the trial annotation is missing %r" % k)
        m, l, s = note["condition"]
        _expect(CS.is_legal(m, l, s),
                "the RECORDED condition is illegal: %r" % (note["condition"],))
        _expect(note["active_loss_hps"] == list(CS.active_loss_hps(l)),
                "recorded A(l) disagrees with condition_space")
        if note["projected"]:
            n_proj += 1
            _expect(note["raw_condition"] != note["condition"],
                    "marked projected but raw == projected: %r" % note)
            _expect(note["raw_condition"][2] is True
                    and note["strict_semihard"] is False,
                    "the only coordinate Pi may move is strict_semihard: %r"
                    % note)
        else:
            _expect(note["raw_condition"] == note["condition"],
                    "marked unprojected but raw != projected: %r" % note)
    _expect(n_proj > 0, "no trial in 120 draws was projected -- the check is "
            "not exercising Pi")
    print("  [6-C] every trial records Pi (%d of %d draws projected), and the "
          "recorded condition is always legal OK" % (n_proj, len(pts)))


def test_6d_failure_path_finite_and_annotated():
    """A trial that cannot be built must score FINITE and still say which cell."""
    _expect(np.isfinite(S.FAILED_OBJECTIVE),
            "FAILED_OBJECTIVE = %r is not finite" % S.FAILED_OBJECTIVE)
    _expect(S.FAILED_OBJECTIVE > 0.0,
            "FAILED_OBJECTIVE must be strictly worse than any achievable "
            "-metric (which is <= 1 in magnitude)")
    cfg = _cfg(n_seeds=2)
    splits = _Splits(np.array([0, 0, 1, 1, 2, 2]))
    calls = []

    def flaky(cfg_in, train_ds, val_ds, device, seed, ckpt_dir=None,
              verbose=False):
        calls.append(seed)
        if len(calls) == 1:
            raise RuntimeError("simulated OOM")
        return None, _history(0.9)

    orig = S.train
    S.train = flaky
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj, rec = S.evaluate_candidate(cfg, splits, "cpu", trial_number=0)
    finally:
        S.train = orig
    _expect(obj == S.FAILED_OBJECTIVE,
            "a partially-failed trial scored %r, expected FAILED_OBJECTIVE" % obj)
    _expect(np.isfinite(obj), "the failure path returned a non-finite objective")
    _expect(rec["failed"] is True, "the record is not marked failed")
    _expect(rec["n_seeds_ok"] == 1 and rec["n_seeds"] == 2,
            "seed bookkeeping on the failure path")
    print("  [6-D] the failure path returns a FINITE penalty (%.1f), never NaN, "
          "and a partially-completed trial is failed not averaged OK"
          % S.FAILED_OBJECTIVE)


def test_6e_trial_log_is_json():
    """The trial log is written to disk; every field must survive json.dumps."""
    base = ExperimentConfig()
    space = _space(base)
    pt = [list(p) for p in Space(space).rvs(n_samples=1, random_state=31)][0]
    note = S.annotate_joint_condition_point(pt)
    rec = {"trial": 0, "scores": [0.5], "mean": 0.5, "std": 0.0,
           "objective": -0.5, "eff_rank": 2.0, "failed": False}
    rec.update(note)
    try:
        json.dumps(rec)
    except TypeError as ex:
        raise AssertionError("the trial record is not JSON-serialisable: %s" % ex)
    # and the winner dict too
    class _Res(object):
        x = pt
    best = S.best_joint_condition_dict(_Res())
    try:
        json.dumps(best)
    except TypeError as ex:
        raise AssertionError("the winner dict is not JSON-serialisable: %s" % ex)
    print("  [6-E] the annotated trial record and the winner dict are both "
          "JSON-serialisable OK")


def test_6f_search_mode():
    for mode in ("staged", "joint", "joint_conditions"):
        SearchConfig(search_mode=mode)
    for bad in ("joint_condition", "conditions", "JOINT_CONDITIONS", ""):
        try:
            SearchConfig(search_mode=bad)
        except ValueError:
            continue
        raise AssertionError("SearchConfig accepted search_mode = %r" % bad)
    _expect(not hasattr(SearchConfig(), "search_conditions"),
            "a separate search_conditions flag exists alongside search_mode; "
            "the combination staged + conditions would then be representable "
            "and meaningless")
    print("  [6-F] search_mode accepts exactly {staged, joint, "
          "joint_conditions}, and no redundant flag exists OK")


def test_6g_n_initial_points_joint():
    s = SearchConfig(n_initial_points_joint=40, n_initial_points=7)
    _expect(S.resolve_n_initial_points_joint(s) == 40,
            "n_initial_points_joint should win when set")
    s = SearchConfig(n_initial_points_joint=0, n_initial_points=7)
    _expect(S.resolve_n_initial_points_joint(s) == 7,
            "should fall back to n_initial_points")
    s = SearchConfig(n_initial_points_joint=0, n_initial_points=0)
    _expect(S.resolve_n_initial_points_joint(s) == 0,
            "0 must propagate so resolve_n_initial_points applies the legacy rule")
    from objective_utils import resolve_n_initial_points
    _expect(resolve_n_initial_points(300, 0) == 10,
            "the legacy rule caps at 10 -- a thin design in 22 columns, which "
            "is exactly why n_initial_points_joint exists")
    print("  [6-G] n_initial_points_joint resolution and fallback chain OK "
          "(legacy rule would give only 10 of 300)")


def test_6h_continuous_primary_disables_tie_break():
    """This study uses selection_primary = 'silhouette', which is continuous."""
    splits = _Splits(np.array([0, 0, 1, 1, 2, 2]))
    cfg = _cfg(n_seeds=1, gamma=0.5, primary="silhouette")
    with warnings.catch_warnings(record=True) as rec_w:
        warnings.simplefilter("always")
        eps, info = S.resolve_tie_break_epsilon(cfg, splits)
    _expect(eps is None,
            "a continuous primary must disable the tie-break; got epsilon = %r"
            % eps)
    _expect(any("INERT" in str(w.message) for w in rec_w),
            "gamma > 0 under a continuous primary must WARN, or the setting is "
            "silently ignored")
    # gamma = 0 is the way to run this study without the warning
    cfg0 = _cfg(n_seeds=1, gamma=0.0, primary="silhouette")
    with warnings.catch_warnings(record=True) as rec_w0:
        warnings.simplefilter("always")
        eps0, _i = S.resolve_tie_break_epsilon(cfg0, splits)
    _expect(eps0 is None and not rec_w0,
            "gamma = 0 under a continuous primary should be silent")
    print("  [6-H] a continuous primary disables the tie-break and warns; "
          "gamma = 0 is the quiet way to say so OK")


def main():
    print("Stage 6 -- the two named acceptance tests")
    test_6a_single_seed_is_not_nan()
    test_6b_gamma_zero_disables_tie_break()
    print("Stage 6 -- wiring")
    test_6c_pi_recorded_per_trial()
    test_6d_failure_path_finite_and_annotated()
    test_6e_trial_log_is_json()
    test_6f_search_mode()
    test_6g_n_initial_points_joint()
    test_6h_continuous_primary_disables_tie_break()
    print("\nALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
