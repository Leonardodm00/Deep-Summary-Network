"""
smoke_test_adaptive_patience.py

Correctness checks for adaptive_patience.py: the growing-patience early-stopping
counter and the label-shuffled silhouette floor that supplies its threshold.

The two things most worth catching here
---------------------------------------
1. NON-TERMINATION. A budget that grows while the wait counter grows is one
   inequality away from a run that never stops. [B] verifies the closed-form
   bound n* = ceil(P_0 / (1 - g)) against brute-force SIMULATION over a grid,
   rather than trusting the algebra, and [C] verifies that the configurations
   which cannot terminate are rejected at construction.

2. A THRESHOLD THAT SILENTLY DISABLES EARLY STOPPING. delta = 2 * mu_floor is
   the literal reading of the request, and mu_floor for a permutation null sits
   at or just below zero, so that threshold is approximately 0 or negative -- and
   a negative threshold makes a DECREASE count as an improvement. [E] measures
   mu_floor and sigma_floor on real embeddings to show where they actually sit,
   and [F] checks that mode='floor_location' REFUSES rather than returning such
   a threshold.

[A] is the compatibility check: with the defaults, the new class reproduces the
existing fixed-patience rule step for step, so wiring it into train.py cannot
change any archived result.

Run:
    cd Main && PYTHONPATH=. python3 Smoke_Tests/smoke_test_adaptive_patience.py

Checks:
  A. growth = 0.0 reproduces the fixed-patience rule exactly, on random
     improvement sequences.
  B. Closed-form effective_patience_bound == brute-force simulation, over a grid
     of (P_0, g, P_max, wait).
  C. Non-terminating and malformed configurations raise at construction.
  D. Budget accounting: reset_on_improvement True vs False, and the cap.
  E. silhouette_floor on structured and unstructured embeddings: mu near zero
     under the null, sigma strictly positive, and a real signal above the floor.
  F. resolve_min_delta_sil: floor_scale positive, floor_location refuses a
     non-positive mu, absolute passes through, bad modes rejected.
  G. State round-trips through state_dict / load_state_dict (checkpoint resume).
"""

import os
import sys

import numpy as np
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_patience import (                                  # noqa: E402
    AdaptivePatience,
    effective_patience_bound,
    resolve_min_delta_sil,
    silhouette_floor,
)


def _fixed_patience_reference(improvements, patience):
    """The rule currently in train.py, written out independently.

    Returns the 1-based epoch index at which it stops, or None if it never does.
    """
    counter = 0
    for e, improved in enumerate(improvements, start=1):
        if improved:
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            return e
    return None


def _simulate_plateau_run(P0, g, P_max, wait0):
    """Brute force: plateau updates survived after warming up to wait = wait0.

    The warm-up is done with REAL update(False) calls rather than by assigning
    to ap.wait, because wait and the budget counter must stay consistent -- a
    hand-set wait with an untouched counter is a state the object can never
    actually reach, and predicting it is meaningless. Returns (None, ap) when
    the counter stops during the warm-up, i.e. when wait0 is unreachable.
    """
    ap = AdaptivePatience(patience=P0, growth=g, max_patience=P_max)
    for _ in range(wait0):
        if ap.update(False):
            return None, ap
    assert ap.wait == wait0
    for n in range(1, 100000):
        if ap.update(False):
            return n, ap
    return None, ap


# --------------------------------------------------------------------------- #
def check_backward_compatible():
    rng = np.random.default_rng(0)
    n_seq = 0
    for patience in (1, 2, 3, 5, 10):
        for trial in range(40):
            seq = (rng.random(60) < 0.35).tolist()
            ref = _fixed_patience_reference(seq, patience)
            ap = AdaptivePatience(patience=patience, growth=0.0)
            got = None
            for e, improved in enumerate(seq, start=1):
                if ap.update(bool(improved)):
                    got = e
                    break
            assert got == ref, (
                "patience=%d trial=%d: AdaptivePatience stopped at %r but the "
                "fixed rule stops at %r" % (patience, trial, got, ref))
            n_seq += 1
    print("      %d random improvement sequences: growth=0.0 stops at exactly "
          "the same epoch as the current train.py rule" % n_seq)


def check_bound_matches_simulation():
    n_checked = 0
    for P0 in (1, 2, 3, 5, 10, 17):
        for g in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
            for P_max in (None, P0, P0 + 3, P0 + 50):
                for wait0 in (0, 1, max(0, min(2, P0 - 1))):
                    # a fresh counter, to read remaining() at the warm-up state
                    probe = AdaptivePatience(patience=P0, growth=g,
                                             max_patience=P_max)
                    stopped_early = False
                    for _ in range(wait0):
                        if probe.update(False):
                            stopped_early = True
                            break
                    if stopped_early:
                        continue           # wait0 is unreachable here; nothing to predict
                    predicted = effective_patience_bound(P0, g, P_max, wait0)
                    assert probe.remaining() == predicted, (
                        "P0=%d g=%r P_max=%r wait0=%d: remaining() says %r but "
                        "effective_patience_bound says %r"
                        % (P0, g, P_max, wait0, probe.remaining(), predicted))
                    simulated, _ap = _simulate_plateau_run(P0, g, P_max, wait0)
                    assert simulated is not None, (
                        "did not terminate at P0=%d g=%r P_max=%r wait0=%d"
                        % (P0, g, P_max, wait0))
                    assert float(simulated) == predicted, (
                        "P0=%d g=%r P_max=%r wait0=%d: closed form says %r, "
                        "simulation says %d"
                        % (P0, g, P_max, wait0, predicted, simulated))
                    n_checked += 1
    # The headline interpretation: g multiplies effective patience by 1/(1-g).
    # The reference is computed in EXACT rational arithmetic, not in floats --
    # 4 / (1 - 0.8) evaluates to 20.000000000000004 in binary and would make
    # this test demand 21 for an answer that is exactly 20.
    from fractions import Fraction
    import math as _math
    shown = []
    for P0 in (4, 10, 20):
        for g in (0.0, 0.5, 0.75, 0.8, 0.9):
            exact = _math.ceil(Fraction(P0) / (1 - Fraction(str(g))))
            got = effective_patience_bound(P0, g)
            assert got == float(exact), (
                "P_0 = %d, g = %r: effective_patience_bound gave %r, exact "
                "rational arithmetic gives %d" % (P0, g, got, exact))
            if P0 == 10:
                shown.append("g=%.2g -> %d" % (g, exact))
    print("      %d (P_0, g, P_max, wait) combinations: closed form == "
          "simulation == remaining(). At P_0 = 10: %s"
          % (n_checked, ", ".join(shown)))


def _expect(exc_type, fn, must_mention=()):
    try:
        fn()
    except exc_type as ex:
        for token in must_mention:
            assert token in str(ex), (
                "%s raised but message lacks %r: %s"
                % (exc_type.__name__, token, ex))
        return str(ex)
    raise AssertionError("expected %s, none was raised" % (exc_type.__name__,))


def check_rejects_nonterminating():
    # g >= 1 with no cap can never stop
    _expect(ValueError, lambda: AdaptivePatience(patience=5, growth=1.0),
            ["never terminates"])
    _expect(ValueError, lambda: AdaptivePatience(patience=5, growth=2.5),
            ["never terminates"])
    # ... but is allowed once capped, and then does stop
    ap = AdaptivePatience(patience=5, growth=1.0, max_patience=9)
    n = 0
    while not ap.update(False):
        n += 1
        assert n < 1000, "capped g = 1.0 failed to terminate"
    assert n + 1 == 9, "capped run stopped after %d plateaus, expected 9" % (n + 1)
    # malformed
    _expect(ValueError, lambda: AdaptivePatience(patience=0), ["patience"])
    _expect(ValueError, lambda: AdaptivePatience(patience=5, growth=-0.1),
            ["growth"])
    _expect(ValueError,
            lambda: AdaptivePatience(patience=10, max_patience=4),
            ["max_patience"])
    _expect(ValueError, lambda: AdaptivePatience(patience=5, delta=-1.0),
            ["delta"])
    # unarmed use is an error, not a silent default
    ap2 = AdaptivePatience(patience=5)
    _expect(RuntimeError, lambda: ap2.is_improvement(0.5, 0.1), ["not armed"])
    ap2.arm(0.01)
    assert ap2.is_improvement(0.5, 0.1) and not ap2.is_improvement(0.105, 0.1)
    assert not ap2.is_improvement(float("nan"), 0.1)
    assert ap2.is_improvement(0.1, float("-inf"))
    print("      g >= 1 uncapped, patience < 1, negative growth, cap below "
          "start, negative delta and unarmed use all rejected; capped g = 1.0 "
          "stops after exactly P_max plateaus")


def check_budget_accounting():
    # reset_on_improvement=True returns the budget to P_0
    ap = AdaptivePatience(patience=4, growth=0.5, reset_on_improvement=True)
    for _ in range(3):
        ap.update(False)
    assert abs(ap.budget - 5.5) < 1e-12, ap.budget
    ap.update(True)
    assert ap.wait == 0 and abs(ap.budget - 4.0) < 1e-12, ap

    # reset_on_improvement=False keeps the earned budget
    ap = AdaptivePatience(patience=4, growth=0.5, reset_on_improvement=False)
    for _ in range(3):
        ap.update(False)
    ap.update(True)
    assert ap.wait == 0 and abs(ap.budget - 5.5) < 1e-12, ap
    # ... and is therefore strictly more patient afterwards
    n_keep = 0
    while not ap.update(False):
        n_keep += 1
    assert n_keep + 1 == 11, n_keep + 1

    # the cap binds
    ap = AdaptivePatience(patience=3, growth=0.9, max_patience=5)
    for _ in range(20):
        ap.update(False)
        assert ap.budget <= 5.0 + 1e-12
    assert ap.n_plateau == 20 and ap.n_improvement == 0
    print("      budget resets to P_0 on improvement (or is kept when asked), "
          "never exceeds max_patience, and plateau/improvement counts track")


def check_floor_statistics():
    rng = np.random.default_rng(1)

    # (i) pure noise, C = 2 and C = 4: mu near 0, sigma > 0
    for C in (2, 4):
        N, E = 96, 12
        Z = rng.normal(size=(N, E))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)   # L2-normalised, as in the pipeline
        y = np.repeat(np.arange(C), N // C)
        floor = silhouette_floor(Z, y, n_permutations=120, metric="cosine",
                                 seed=0)
        assert floor["sigma"] > 0.0, floor
        assert abs(floor["mu"]) < 0.15, (
            "mu_floor = %.5f is not near zero at C = %d" % (floor["mu"], C))
        assert floor["sigma"] < 0.1, (
            "sigma_floor = %.5f is implausibly large for a permutation null"
            % floor["sigma"])
        assert floor["n_valid"] == 120
        print("      C = %d, N = %d noise: mu_floor = %+.5f, sigma_floor = "
              "%.5f, q95 = %+.5f" % (C, N, floor["mu"], floor["sigma"],
                                     floor["q95"]))

    # (ii) the sign asymmetry the module docstring claims: with C > 2, b(i) takes
    #      a minimum over C - 1 classes and is biased downward, so mu_floor tends
    #      NEGATIVE. Averaged over several draws to avoid asserting on one sample.
    mus = []
    for rep in range(6):
        Z = rng.normal(size=(80, 10))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        y = np.repeat(np.arange(4), 20)
        mus.append(silhouette_floor(Z, y, n_permutations=80, seed=rep)["mu"])
    mean_mu = float(np.mean(mus))
    assert mean_mu < 0.0, (
        "expected mu_floor < 0 at C = 4 (min-over-classes bias in b(i)); got "
        "mean %.6f over %d draws: %r" % (mean_mu, len(mus), mus))
    print("      C = 4, 6 independent draws: mean mu_floor = %+.5f (negative, "
          "as the min-over-classes bias in b(i) predicts) -- this is why "
          "2 * mu_floor is not a usable threshold" % mean_mu)

    # (iii) a REAL signal must sit above the floor, or the floor is useless
    centres = np.array([[3.0, 0.0], [-3.0, 0.0]])
    Z = np.vstack([centres[0] + 0.25 * rng.normal(size=(40, 2)),
                   centres[1] + 0.25 * rng.normal(size=(40, 2))])
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    y = np.repeat([0, 1], 40)
    real = float(silhouette_score(Z, y, metric="cosine"))
    floor = silhouette_floor(Z, y, n_permutations=200, seed=0)
    assert real > floor["q95"], (real, floor["q95"])
    z_score = (real - floor["mu"]) / floor["sigma"]
    assert z_score > 5.0, z_score
    print("      two well-separated clusters: s_bar = %.4f vs floor q95 = "
          "%+.4f (%.1f sigma above mu_floor)" % (real, floor["q95"], z_score))


def check_resolve_delta():
    floor_pos = {"mu": 0.02, "sigma": 0.01}
    floor_zero = {"mu": 0.0, "sigma": 0.01}
    floor_neg = {"mu": -0.008, "sigma": 0.01}

    assert abs(resolve_min_delta_sil(floor_pos, kappa=2.0,
                                     mode="floor_scale") - 0.02) < 1e-12
    assert abs(resolve_min_delta_sil(floor_neg, kappa=2.0,
                                     mode="floor_scale") - 0.02) < 1e-12
    assert abs(resolve_min_delta_sil(floor_pos, kappa=2.0,
                                     mode="floor_location") - 0.04) < 1e-12
    # the literal mode must REFUSE rather than hand back <= 0
    for bad in (floor_zero, floor_neg):
        msg = _expect(ValueError,
                      lambda f=bad: resolve_min_delta_sil(
                          f, kappa=2.0, mode="floor_location"),
                      ["floor_scale"])
        assert "disables early stopping" in msg
    # absolute passes through; a zero threshold is legal there (current default)
    assert resolve_min_delta_sil(None, mode="absolute", absolute=0.0) == 0.0
    assert resolve_min_delta_sil(None, mode="absolute", absolute=0.03) == 0.03
    _expect(ValueError, lambda: resolve_min_delta_sil(None, mode="nonsense"),
            ["mode"])
    _expect(ValueError, lambda: resolve_min_delta_sil(None,
                                                      mode="floor_scale"),
            ["floor"])
    _expect(ValueError, lambda: resolve_min_delta_sil(floor_pos, kappa=0.0,
                                                      mode="floor_scale"),
            ["kappa"])
    _expect(ValueError, lambda: resolve_min_delta_sil(
        {"mu": 0.1, "sigma": 0.0}, mode="floor_scale"), ["collapsed"])
    print("      floor_scale = kappa * sigma (positive whatever mu does); "
          "floor_location refuses mu <= 0 and names the alternative; absolute "
          "passes through unchanged")


def check_state_roundtrip():
    ap = AdaptivePatience(patience=6, growth=0.4, max_patience=12,
                          reset_on_improvement=False, delta=0.017)
    for improved in (False, False, True, False, False, False):
        ap.update(improved)
    state = ap.state_dict()

    fresh = AdaptivePatience(patience=6, growth=0.4, max_patience=12,
                             reset_on_improvement=False)
    fresh.load_state_dict(state)
    assert fresh.wait == ap.wait
    assert abs(fresh.budget - ap.budget) < 1e-12
    assert fresh.delta == ap.delta
    assert fresh.n_plateau == ap.n_plateau
    assert fresh.remaining() == ap.remaining()

    # resuming must not change WHEN it stops
    tail = [False] * 40
    a_stop = next((i for i, v in enumerate(tail, 1) if ap.update(v)), None)
    b_stop = next((i for i, v in enumerate(tail, 1) if fresh.update(v)), None)
    assert a_stop == b_stop and a_stop is not None, (a_stop, b_stop)
    print("      state_dict / load_state_dict round-trip: a resumed counter "
          "stops at the same plateau epoch (%d) as one that never stopped"
          % a_stop)


# --------------------------------------------------------------------------- #
def main():
    groups = [
        ("A", "growth=0 reproduces fixed patience", check_backward_compatible),
        ("B", "closed-form bound == simulation", check_bound_matches_simulation),
        ("C", "non-terminating configs rejected", check_rejects_nonterminating),
        ("D", "budget accounting and cap", check_budget_accounting),
        ("E", "silhouette floor statistics", check_floor_statistics),
        ("F", "threshold resolution", check_resolve_delta),
        ("G", "checkpoint state round-trip", check_state_roundtrip),
    ]
    print("smoke_test_adaptive_patience.py  [growing patience + silhouette floor]")
    failures = []
    for letter, title, fn in groups:
        try:
            fn()
        except Exception as ex:                    # noqa: BLE001
            failures.append((letter, title, ex))
            print("  [%s] %-40s FAIL" % (letter, title))
            print("      %s: %s" % (type(ex).__name__, ex))
        else:
            print("  [%s] %-40s PASS" % (letter, title))
    if failures:
        print("FAILED: %d of %d assertion group(s): %s"
              % (len(failures), len(groups),
                 ", ".join(f[0] for f in failures)))
        return 1
    print("ALL ADAPTIVE-PATIENCE CHECKS PASSED (%d groups)" % len(groups))
    return 0


if __name__ == "__main__":
    sys.exit(main())
