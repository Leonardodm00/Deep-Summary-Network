"""
smoke_test_objective_wiring.py

Standalone correctness checks for the C2 (adaptive tie-break) and C3 (explicit
n_initial_points) objective utilities. No torch, no data files, CPU only.

Run:
    cd Main/Smoke_Tests && python3 smoke_test_objective_wiring.py

These check the MATH and the PURE FUNCTIONS. Two further properties need torch
and are checked elsewhere:
  * that the recomputed selected epoch e* equals train.py's own best_epoch on a
    real run                       -> Smoke_Tests/smoke_test_selected_epoch.py
  * that search.evaluate_candidate wires them together correctly
                                   -> Smoke_Tests/smoke_test_search.py

Checks:
  A. selected_epoch_index mirrors train.py's rule on hand-built histories:
     lexicographic argmax over (u_e, v_e) against the COMPONENT-WISE running
     maxima, strict >, so the FIRST epoch attaining a tied pair wins.
  B. NaN handling: a non-finite metric can never win; an all-NaN history selects
     nothing and reports epoch 0 (train.py's own sentinel).
  C. selection_primary = "silhouette" swaps the roles of the two signals.
  D. Delta_min(y) reproduces the archived measurement: N_eval = 36, C = 3
     balanced -> max ARI below 1 = 0.9154, Delta_min = 0.0846. This is the
     combinatorial identity that independently reproduced the seed-1 test ARI.
  E. Delta_min(y) SHRINKS with N_eval, which is why a constant epsilon is unsafe.
  F. The lexicographic guarantee: eps * (s_hi - s_lo) < Delta_min(y) strictly,
     for every gamma in (0, 1); and gamma = 1 saturates it exactly.
  G. The guarantee OPERATIONALLY: the secondary metric flips the ranking of two
     configurations that TIE on the primary, and provably cannot flip two that
     differ on it by even the smallest amount the eval set can express.
  H. composite_objective conventions: non-finite primary -> +inf (worst);
     non-finite secondary -> contributes 0; and FAILED_OBJECTIVE = 1.0 stays
     strictly worse than any attainable composite value.
  I. resolve_n_initial_points reproduces the legacy hard-coded rule EXACTLY for
     None and 0, honours an explicit value, and raises when it exceeds n_calls.
  J. Cost of Delta_min(y): O(N_eval^2 * C), timed, so the budget is known rather
     than assumed.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from objective_utils import (                                    # noqa: E402
    adaptive_epsilon,
    composite_objective,
    min_ari_gap,
    resolve_n_initial_points,
    selected_epoch_index,
    selected_epoch_scores,
)

FAILED_OBJECTIVE = 1.0        # mirrors search.FAILED_OBJECTIVE without importing it


def _h(epoch, ari, sil):
    return {"epoch": epoch, "ari": ari, "silhouette": sil}


# --------------------------------------------------------------------------- #
# A / B / C -- the epoch-selection rule
# --------------------------------------------------------------------------- #
def check_epoch_rule():
    # primary strictly wins
    hist = [_h(1, 0.5, 0.9), _h(2, 0.9, 0.0), _h(3, 0.3, 0.99)]
    assert selected_epoch_index(hist) == 1
    ari, sil, e = selected_epoch_scores(hist)
    assert (ari, sil, e) == (0.9, 0.0, 2), (ari, sil, e)
    print("  [A] primary wins outright: e* = %d, and BOTH metrics are read there "
          "(ARI %.2f, Sil %.2f) -- not each at its own max OK" % (e, ari, sil))

    # this is the whole point of C2: the OLD rule would have reported
    # ARI = 0.9 (epoch 2) and Sil = 0.99 (epoch 3) -- a model that never existed
    old_style_ari = max(h["ari"] for h in hist)
    old_style_sil = max(h["silhouette"] for h in hist)
    assert (old_style_ari, old_style_sil) == (0.9, 0.99)
    assert (ari, sil) != (old_style_ari, old_style_sil)
    print("  [A] independent per-metric maxima would have been (%.2f, %.2f), "
          "which no single epoch attained OK" % (old_style_ari, old_style_sil))

    # ties: strict > means the FIRST epoch attaining the pair wins
    hist = [_h(1, 0.8, 0.5), _h(2, 0.8, 0.5), _h(3, 0.8, 0.4)]
    assert selected_epoch_index(hist) == 0
    print("  [A] exact tie -> the FIRST attaining epoch wins (strict >) OK")

    # secondary breaks a primary tie ONLY when it beats the RUNNING max
    hist = [_h(1, 0.8, 0.2), _h(2, 0.8, 0.7), _h(3, 0.8, 0.5)]
    assert selected_epoch_index(hist) == 1
    print("  [A] primary tied -> the epoch with the better secondary wins OK")

    # comparison is against COMPONENT-WISE running maxima, not the pair at e*
    hist = [_h(1, 0.9, 0.1), _h(2, 0.2, 0.9), _h(3, 0.9, 0.5)]
    # after e=1: (u*,v*) = (0.9, 0.1); after e=2: (0.9, 0.9)
    # e=3: (0.9, 0.5) > (0.9, 0.9)? no -> e* stays 1
    assert selected_epoch_index(hist) == 0, selected_epoch_index(hist)
    print("  [A] comparison is against the COMPONENT-WISE running maxima (u*, v*), "
          "a pair no epoch need have attained OK")

    # NaN can never win; all-NaN selects nothing
    hist = [_h(1, float("nan"), 0.9), _h(2, 0.4, 0.1)]
    assert selected_epoch_index(hist) == 1
    hist = [_h(1, float("nan"), float("nan")), _h(2, float("inf") * 0 + float("nan"), 0.5)]
    idx = selected_epoch_index(hist)
    ari, sil, e = selected_epoch_scores(hist)
    assert idx == 1 and e == 2, (idx, e)
    hist = [_h(1, float("nan"), float("nan")), _h(2, float("nan"), float("nan"))]
    assert selected_epoch_index(hist) is None
    ari, sil, e = selected_epoch_scores(hist)
    assert ari == float("-inf") and np.isnan(sil) and e == 0, (ari, sil, e)
    print("  [B] NaN never wins; an all-NaN history selects nothing and reports "
          "epoch 0 (train.py's sentinel) OK")

    # selection_primary = silhouette swaps the roles
    hist = [_h(1, 0.9, 0.1), _h(2, 0.1, 0.9)]
    assert selected_epoch_index(hist, "ari") == 0
    assert selected_epoch_index(hist, "silhouette") == 1
    ari, sil, e = selected_epoch_scores(hist, "silhouette")
    assert (ari, sil, e) == (0.1, 0.9, 2), (ari, sil, e)
    print("  [C] selection_primary='silhouette' swaps (u, v); the returned pair "
          "is still (ARI, Sil), still read at the same e* OK")

    try:
        selected_epoch_index(hist, "nonsense")
    except ValueError:
        print("  [C] an unknown selection_primary raises OK")
    else:
        raise AssertionError("unknown selection_primary did not raise")


# --------------------------------------------------------------------------- #
# D / E -- the resolution Delta_min(y)
# --------------------------------------------------------------------------- #
def check_resolution():
    y36 = np.repeat([0, 1, 2], 12)          # N_eval = 36, C = 3, balanced
    info = min_ari_gap(y36)
    assert abs(info["best_ari_below1"] - 0.9154) < 5e-4, info
    assert abs(info["delta_min"] - 0.0846) < 5e-4, info
    print("  [D] N_eval = 36, C = 3 balanced: max ARI below 1 = %.4f, "
          "Delta_min(y) = %.4f -- matches the archived measurement, and says "
          "the seed-1 test ARI of 0.9154 was exactly ONE misassigned window OK"
          % (info["best_ari_below1"], info["delta_min"]))

    prev = None
    row = []
    for n_per in (12, 30, 60):
        y = np.repeat([0, 1, 2], n_per)
        d = min_ari_gap(y)["delta_min"]
        row.append((3 * n_per, d))
        if prev is not None:
            assert d < prev, "Delta_min(y) must shrink as N_eval grows"
        prev = d
    print("  [E] Delta_min(y) shrinks with N_eval: %s -- so a CONSTANT epsilon "
          "tuned on one eval set silently violates the guarantee on a smaller "
          "one OK" % ", ".join("N=%d -> %.4f" % (n, d) for n, d in row))


# --------------------------------------------------------------------------- #
# F / G -- the lexicographic guarantee
# --------------------------------------------------------------------------- #
def check_guarantee():
    y = np.repeat([0, 1, 2], 12)
    for gamma in (0.1, 0.5, 0.9):
        info = adaptive_epsilon(y, gamma=gamma)
        assert info["max_secondary_influence"] < info["delta_min"], info
    info = adaptive_epsilon(y, sil_lo=-1.0, sil_hi=1.0, gamma=0.5)
    assert abs(info["epsilon"] - 0.02115) < 1e-4, info
    assert abs(info["max_secondary_influence"] - 0.04229) < 1e-4, info
    print("  [F] gamma = 0.5, s in [-1, 1]: epsilon = %.5f, so the secondary "
          "metric's TOTAL influence %.5f < Delta_min(y) = %.5f, strictly OK"
          % (info["epsilon"], info["max_secondary_influence"], info["delta_min"]))

    sat = adaptive_epsilon(y, gamma=1.0)
    assert abs(sat["max_secondary_influence"] - sat["delta_min"]) < 1e-12
    print("  [F] gamma = 1 saturates the condition exactly (influence == "
          "Delta_min), which is why the default leaves a factor-2 margin OK")

    eps = info["epsilon"]
    d = info["delta_min"]

    # (i) inside an exact primary tie, the secondary DOES decide
    a = composite_objective(0.90, 0.10, eps)
    b = composite_objective(0.90, 0.80, eps)
    assert b < a, (a, b)
    print("  [G] exact primary tie -> the better secondary wins (J %.5f < %.5f) OK"
          % (b, a))

    # (ii) across the SMALLEST primary difference the eval set can express, the
    #      secondary cannot flip the order even at its most extreme values
    better = composite_objective(1.0, -1.0, eps)        # best primary, worst secondary
    worse = composite_objective(1.0 - d, +1.0, eps)     # one step down, best secondary
    assert better < worse, (better, worse)
    print("  [G] one resolution step apart (ARI 1.0000 vs %.4f): the secondary "
          "CANNOT flip it even at its extremes (J %.5f < %.5f) OK"
          % (1.0 - d, better, worse))


# --------------------------------------------------------------------------- #
# H -- composite conventions and the failure sentinel
# --------------------------------------------------------------------------- #
def check_conventions():
    eps = 0.02115
    assert composite_objective(float("nan"), 0.5, eps) == float("inf")
    assert composite_objective(0.5, float("nan"), eps) == -0.5
    print("  [H] non-finite primary -> +inf (worst); non-finite secondary "
          "contributes 0 rather than poisoning a valid primary OK")

    # FAILED_OBJECTIVE must stay strictly worse than ANY attainable composite.
    # ARI >= -0.5 and Sil >= -1, so the worst attainable J is 0.5 + eps.
    worst_real = composite_objective(-0.5, -1.0, eps)
    assert worst_real < FAILED_OBJECTIVE, (worst_real, FAILED_OBJECTIVE)
    best_real = composite_objective(1.0, 1.0, eps)
    assert best_real > -1.0 - eps - 1e-12
    print("  [H] FAILED_OBJECTIVE = %.1f remains strictly worse than the worst "
          "attainable composite %.5f, and the best is %.5f -- so the GP can "
          "still fit the failure value OK"
          % (FAILED_OBJECTIVE, worst_real, best_real))

    try:
        composite_objective(0.5, 0.5, 0.0)
    except ValueError:
        print("  [H] epsilon <= 0 raises rather than silently disabling OK")
    else:
        raise AssertionError("epsilon = 0 did not raise")


# --------------------------------------------------------------------------- #
# I -- the budget split
# --------------------------------------------------------------------------- #
def check_budget():
    legacy = lambda n: min(10, max(1, n // 2))
    for n_calls in (1, 2, 3, 4, 15, 20, 50, 100, 753):
        for requested in (None, 0, -1):
            got = resolve_n_initial_points(n_calls, requested)
            assert got == legacy(n_calls), (n_calls, requested, got)
    print("  [I] None / 0 / negative reproduce the legacy rule "
          "min(10, max(1, n_calls // 2)) EXACTLY for n_calls in "
          "{1,2,3,4,15,20,50,100,753} -- so every pre-C3 config is unaffected OK")
    assert resolve_n_initial_points(50, 10) == 10
    assert resolve_n_initial_points(50, 25) == 25
    assert resolve_n_initial_points(50, 50) == 50
    print("  [I] an explicit value is honoured, including n_init == n_calls OK")

    for bad in ((50, 51), (1, 2)):
        try:
            resolve_n_initial_points(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError("n_initial_points > n_calls did not raise: %r" % (bad,))
    try:
        resolve_n_initial_points(0, None)
    except ValueError:
        pass
    else:
        raise AssertionError("n_calls < 1 did not raise")
    print("  [I] n_initial_points > n_calls raises (it would leave the surrogate "
          "no trials and silently degrade the study to random search) OK")

    # the archived run's numbers, for the record
    assert resolve_n_initial_points(50, None) == 10
    print("  [I] archived run: n_calls = 50 -> n_init = 10, i.e. trials 0-9 were "
          "drawn BEFORE the GP existed; trial 0 was the reported optimum OK")


# --------------------------------------------------------------------------- #
# J -- cost
# --------------------------------------------------------------------------- #
def check_cost():
    print("  [J] Delta_min(y) cost, O(N_eval^2 * C), computed once per phase:")
    for n_per in (12, 60, 200):
        y = np.repeat([0, 1, 2], n_per)
        t0 = time.time()
        min_ari_gap(y)
        dt = time.time() - t0
        print("        N_eval = %4d  ->  %7.1f ms" % (3 * n_per, 1e3 * dt))


def main():
    print("smoke_test_objective_wiring.py [C2 + C3]")
    check_epoch_rule()
    check_resolution()
    check_guarantee()
    check_conventions()
    check_budget()
    check_cost()
    print("ALL OBJECTIVE-WIRING CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
