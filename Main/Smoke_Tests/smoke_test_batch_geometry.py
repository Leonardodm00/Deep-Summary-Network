"""[C4] Smoke test for batch_geometry.py -- Eq. (2), Eq. (3) and the two caps.

Runs with numpy alone: no torch, no data, no model. That is deliberate, and it
is why this suite sits second in run_all_smoke_tests.py alongside the other
torch-free guards -- Change 4 is the change most able to go quietly wrong, so
its arithmetic should fail in milliseconds rather than inside a training run.

CHECKS
  [A] culture_census counts distinct cultures per class and windows per culture,
      and REFUSES a culture that spans two classes.
  [B] Eq. (3): U_eff = min(U_c, availability), the clamp is reported, and a
      class with one training culture RAISES.
  [C] Eq. (2): n_g = U_eff * q and M = C * U_eff * q * (1 + N_s), exactly, over
      a grid of (U_c, q, N_s).
  [D] Miner gating of the group-size ceiling: applied under the easy-positive
      strategies, NOT applied under "hard".
  [E] Miner gating of the degeneracy cap on q, and the cap's dependence on the
      smallest culture.
  [F] exclude_same_culture_positives=False RAISES under an easy-positive
      strategy and is permitted under "hard".
  [G] The resource cap floor(M_max / (C * (1 + N_s))) binds under every
      strategy, and the tighter of the two caps is the one applied.
  [H] q may not exceed the smallest culture's window count (no sampling with
      replacement).
  [I] Inertness: positives_mode="augmentation" checks nothing and reports
      active=False.

RUN
    cd Main
    python3 Smoke_Tests/smoke_test_batch_geometry.py
    echo $?        # 0 on success

HPC note (hpc-python-compat): pure ASCII.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from batch_geometry import (                                     # noqa: E402
    EASY_POSITIVE_STRATEGIES,
    culture_census,
    resolve_batch_geometry,
    resolve_cultures_per_class,
    resolve_group_size_cap,
    resolve_q_cap,
)

EP = "easy_positive"
EPSHN = "easy_pos_semihard_neg"
HARD = "hard"


def make_windows(cultures_per_class, windows_per_culture, n_classes):
    """Synthetic (trace_of_window, conditions) with a KNOWN geometry.

    cultures_per_class may be an int (same for every class) or a per-class list,
    which is how the availability bound is made to bind on one class only.
    """
    if isinstance(cultures_per_class, int):
        cultures_per_class = [cultures_per_class] * n_classes
    if isinstance(windows_per_culture, int):
        windows_per_culture = [windows_per_culture] * n_classes
    g, y, next_culture = [], [], 0
    for c in range(n_classes):
        for _ in range(cultures_per_class[c]):
            g.extend([next_culture] * windows_per_culture[c])
            y.extend([c] * windows_per_culture[c])
            next_culture += 1
    return np.asarray(g, dtype=int), np.asarray(y, dtype=int)


def geometry(g, y, **kw):
    """resolve_batch_geometry with the defaults this suite uses most."""
    params = dict(positives_mode="cross_culture", mining_strategy=EP,
                  cultures_per_class_per_batch=12,
                  windows_per_culture_per_batch=1, n_surrogates=2,
                  max_group_size=16, exclude_same_culture_positives=True)
    params.update(kw)
    return resolve_batch_geometry(g, y, **params)


def raises(fn, needle):
    """Assert fn() raises ValueError whose message contains `needle`."""
    try:
        fn()
    except ValueError as ex:
        assert needle in str(ex), (
            "raised, but the message does not mention %r:\n  %s" % (needle, ex))
        return str(ex)
    raise AssertionError("expected a ValueError mentioning %r, none raised"
                         % (needle,))


def check_census():
    """[A]"""
    g, y = make_windows(cultures_per_class=3, windows_per_culture=5, n_classes=4)
    cen = culture_census(g, y)
    assert cen["cultures_per_class"] == {0: 3, 1: 3, 2: 3, 3: 3}, cen
    assert set(cen["windows_per_culture"].values()) == {5}
    assert len(cen["class_of_culture"]) == 12
    print("      [A] 4 classes x 3 cultures x 5 windows counted exactly OK")

    g2, y2 = make_windows([2, 5, 2, 2], [4, 4, 4, 4], 4)
    assert culture_census(g2, y2)["cultures_per_class"] == {0: 2, 1: 5, 2: 2, 3: 2}
    print("      [A] unbalanced culture counts per class counted exactly OK")

    # a culture spanning two classes must be refused, not averaged over
    raises(lambda: culture_census([0, 0, 1], [0, 1, 1]),
           "must belong to exactly one class")
    print("      [A] a culture spanning two classes RAISES OK")

    raises(lambda: culture_census([0, 1], [0]), "parallel arrays")
    raises(lambda: culture_census([], []), "non-empty")
    print("      [A] mismatched and empty inputs RAISE OK")


def check_eq3_clamp():
    """[B]"""
    # the handoff's own worked case: 8 training cultures over 4 classes -> 2
    g, y = make_windows(2, 13, 4)
    geo = geometry(g, y)
    assert geo.cultures_available == 2, geo.cultures_available
    assert geo.cultures_effective == 2, geo.cultures_effective
    assert geo.clamped is True
    assert any("Eq. (3) clamp" in n for n in geo.notes), geo.notes
    print("      [B] U_c = 12 requested, 2 available -> U_eff = 2, clamp logged OK")

    # the user's real setting: ~9 training cultures per class, no starvation
    g, y = make_windows(9, 13, 4)
    geo = geometry(g, y)
    assert (geo.cultures_available, geo.cultures_effective) == (9, 9)
    assert geo.clamped is True and geo.group_size == 9
    print("      [B] 9 available -> U_eff = 9, n_g = 9 OK")

    # no clamp at all when the request is the smaller number
    geo = geometry(g, y, cultures_per_class_per_batch=4)
    assert geo.cultures_effective == 4 and geo.clamped is False
    print("      [B] U_c = 4 with 9 available -> no clamp OK")

    # the availability bound is a MINIMUM over classes, not a mean
    g, y = make_windows([2, 9, 9, 9], 13, 4)
    geo = geometry(g, y)
    assert geo.cultures_effective == 2, (
        "the bound must be the minimum over classes; a mean would give 7")
    print("      [B] bound is the MINIMUM over classes, not the mean OK")

    # one culture in a class: unsatisfiable, must raise
    g, y = make_windows([1, 9, 9, 9], 13, 4)
    msg = raises(lambda: geometry(g, y), "no cross-culture positive exists")
    assert "[0]" in msg, "the message must name the starved class; got: %s" % msg
    print("      [B] a class with ONE training culture RAISES, naming it OK")

    u, info = resolve_cultures_per_class(12, {0: 5, 1: 3})
    assert (u, info["available"], info["binding_classes"]) == (3, 3, [1])
    print("      [B] resolve_cultures_per_class reports the binding class OK")


def check_eq2_counts():
    """[C]"""
    for u_avail in (2, 3, 9):
        for q in (1, 2, 3):
            for n_s in (0, 1, 2):
                if q > 6:
                    continue
                g, y = make_windows(u_avail, 13, 4)
                geo = geometry(g, y, windows_per_culture_per_batch=q,
                               n_surrogates=n_s, mining_strategy=HARD)
                assert geo.group_size == u_avail * q, (u_avail, q, geo.group_size)
                assert geo.batch_rows == 4 * u_avail * q * (1 + n_s), geo.batch_rows
    print("      [C] n_g = U_eff*q and M = C*U_eff*q*(1+N_s) over 27 combinations OK")

    # the arithmetic the whole change turns on, stated once explicitly
    g, y = make_windows(9, 13, 4)
    geo = geometry(g, y, n_surrogates=2)
    assert (geo.group_size, geo.batch_rows) == (9, 108)
    print("      [C] the operating point: U_eff=9, q=1, N_s=2 -> n_g=9, M=108 OK")


def check_group_cap_gating():
    """[D]"""
    g, y = make_windows(9, 13, 4)          # U_eff = 9
    for strategy in EASY_POSITIVE_STRATEGIES:
        # q = 2 gives n_g = 18 > 16 and must raise under an easy-positive miner
        msg = raises(lambda s=strategy: geometry(
            g, y, mining_strategy=s, windows_per_culture_per_batch=2,
            q_cap_fraction=1.0), "exceeds the cap 16")
        assert "not M" in msg, "the message must distinguish n_g from M"
        print("      [D] %-22s n_g = 18 > 16 RAISES OK" % (strategy + ":",))

    # the SAME configuration is admissible under hard mining
    geo = geometry(g, y, mining_strategy=HARD, windows_per_culture_per_batch=2)
    assert geo.group_size == 18 and geo.group_size_cap is None, geo
    print("      [D] hard mining: the same n_g = 18 is permitted, cap is None OK")

    cap, why = resolve_group_size_cap(HARD, 16, 4, 2, None)
    assert cap is None and "not an easy-positive" in why
    cap, why = resolve_group_size_cap(EP, 16, 4, 2, None)
    assert cap == 16 and "easy-positive ceiling 16" in why
    print("      [D] resolve_group_size_cap gates on the strategy alone OK")


def check_q_cap():
    """[E]"""
    # W_min = 13 -> floor(0.5 * 13) = 6
    cap, why = resolve_q_cap(EP, 13, 0.5)
    assert cap == 6, (cap, why)
    # a small culture tightens it; never below 1
    assert resolve_q_cap(EP, 3, 0.5)[0] == 1
    assert resolve_q_cap(EP, 1, 0.5)[0] == 1
    assert resolve_q_cap(HARD, 13, 0.5)[0] is None
    print("      [E] q cap = max(1, floor(f*W_min)): 13->6, 3->1, hard->None OK")

    # q above the cap raises under an easy-positive miner ...
    g, y = make_windows(2, 13, 4)
    raises(lambda: geometry(g, y, windows_per_culture_per_batch=7),
           "exceeds the degeneracy cap 6")
    # ... and is permitted under hard mining, where the argument does not apply
    geo = geometry(g, y, mining_strategy=HARD, windows_per_culture_per_batch=7)
    assert geo.q_cap is None and geo.group_size == 14
    print("      [E] q = 7 > 6 RAISES under easy-positive, allowed under hard OK")

    for bad in (0.0, -0.5, 1.5):
        try:
            resolve_q_cap(EP, 13, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("q_cap_fraction %r must be rejected" % (bad,))
    print("      [E] an out-of-range q_cap_fraction RAISES OK")


def check_exclude_precondition():
    """[F]"""
    g, y = make_windows(9, 13, 4)
    for strategy in EASY_POSITIVE_STRATEGIES:
        raises(lambda s=strategy: geometry(g, y, mining_strategy=s,
                                           exclude_same_culture_positives=False),
               "requires exclude_same_culture_positives=True")
    print("      [F] both easy-positive strategies REQUIRE the exclusion OK")

    geo = geometry(g, y, mining_strategy=HARD,
                   exclude_same_culture_positives=False)
    assert geo.active is True
    print("      [F] hard mining permits it, as the argument does not apply OK")


def check_resource_cap():
    """[G]"""
    # M_max = 272, C = 4, N_s = 2 -> floor(272 / 12) = 22
    cap, why = resolve_group_size_cap(HARD, 16, 4, 2, 272)
    assert cap == 22 and "resource bound" in why, (cap, why)
    # ... and with no surrogates the same budget allows far more
    assert resolve_group_size_cap(HARD, 16, 4, 0, 272)[0] == 68
    print("      [G] resource bound floor(272/(4*3)) = 22, and 68 at N_s = 0 OK")

    # under an easy-positive strategy the TIGHTER of the two applies
    cap, why = resolve_group_size_cap(EP, 16, 4, 2, 272)
    assert cap == 16 and "min(" in why, (cap, why)
    # and when the resource bound is the tighter one, it wins
    cap, _ = resolve_group_size_cap(EP, 16, 4, 2, 120)
    assert cap == 10, cap
    print("      [G] the tighter of ceiling and resource bound is applied OK")

    # the resource cap binds under hard mining, where no ceiling exists
    g, y = make_windows(9, 13, 4)
    raises(lambda: geometry(g, y, mining_strategy=HARD,
                            windows_per_culture_per_batch=3,
                            max_batch_rows=272), "exceeds the cap 22")
    print("      [G] hard mining: n_g = 27 > 22 RAISES on the resource bound OK")

    try:
        resolve_group_size_cap(EP, 16, 4, 2, 4)
    except ValueError as ex:
        assert "cannot fit even one window per class" in str(ex)
    else:
        raise AssertionError("an impossibly small max_batch_rows must raise")
    print("      [G] a budget too small for one window per class RAISES OK")


def check_replacement_guard():
    """[H]"""
    g, y = make_windows(9, 4, 4)           # W_min = 4
    # hard mining throughout this group: with U_eff = 9 even q = 2 gives n_g = 18,
    # so under an easy-positive strategy the group CEILING would fire first and
    # this guard would never be the thing under test.
    geo = geometry(g, y, mining_strategy=HARD,
                   windows_per_culture_per_batch=2, q_cap_fraction=1.0)
    assert geo.min_windows_per_culture == 4 and geo.group_size == 18
    raises(lambda: geometry(g, y, windows_per_culture_per_batch=5,
                            q_cap_fraction=1.0, mining_strategy=HARD),
           "drawn WITH replacement")
    print("      [H] q = 5 from a 4-window culture RAISES OK")

    # the guard reads the SMALLEST culture, not the average
    g, y = make_windows([9, 9, 9, 9], [13, 13, 13, 2], 4)
    assert geometry(g, y).min_windows_per_culture == 2
    raises(lambda: geometry(g, y, windows_per_culture_per_batch=3,
                            mining_strategy=HARD), "drawn WITH replacement")
    print("      [H] the guard reads the SMALLEST culture, not the mean OK")


def check_inertness():
    """[I]"""
    g, y = make_windows([1, 1, 1, 1], 1, 4)   # would raise under cross_culture
    geo = resolve_batch_geometry(
        g, y, positives_mode="augmentation", mining_strategy=EP,
        cultures_per_class_per_batch=12, windows_per_culture_per_batch=99,
        n_surrogates=2, max_group_size=16,
        exclude_same_culture_positives=False)
    assert geo.active is False and geo.group_size == 0
    print("      [I] positives_mode='augmentation': inactive, nothing checked, "
          "even on inputs that would otherwise raise OK")

    try:
        resolve_batch_geometry(g, y, positives_mode="nonsense",
                               mining_strategy=EP,
                               cultures_per_class_per_batch=12,
                               windows_per_culture_per_batch=1,
                               n_surrogates=2, max_group_size=16,
                               exclude_same_culture_positives=True)
    except ValueError as ex:
        assert "positives_mode must be" in str(ex)
    else:
        raise AssertionError("an unknown positives_mode must raise")
    print("      [I] an unknown positives_mode RAISES OK")


def main():
    print("smoke_test_batch_geometry.py [C4 batch geometry]  numpy %s"
          % np.__version__)
    print("  [A] culture census:")
    check_census()
    print("  [B] Eq. (3), the availability clamp:")
    check_eq3_clamp()
    print("  [C] Eq. (2), group size and batch rows:")
    check_eq2_counts()
    print("  [D] group-size ceiling, gated on the miner:")
    check_group_cap_gating()
    print("  [E] degeneracy cap on q:")
    check_q_cap()
    print("  [F] exclude_same_culture_positives precondition:")
    check_exclude_precondition()
    print("  [G] resource cap on M:")
    check_resource_cap()
    print("  [H] no sampling with replacement:")
    check_replacement_guard()
    print("  [I] inertness under 'augmentation':")
    check_inertness()
    print("ALL BATCH-GEOMETRY CHECKS PASSED (9 groups)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
