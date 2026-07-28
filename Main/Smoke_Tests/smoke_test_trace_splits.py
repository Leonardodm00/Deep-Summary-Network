"""
smoke_test_trace_splits.py

Correctness checks for data_splits.make_trace_splits (Change 5 of the v3
handoff): the whole-culture train / val / test split that replaces the
time-segment split.

Why this test exists
--------------------
make_time_segment_splits guarantees only that no WINDOW straddles a split
boundary. Every culture still contributes windows to all three splits, so a
model can exploit culture identity and the reported test score answers a
question nobody asked. make_trace_splits confines a whole culture to one split.
The single most important assertion here is therefore [A]: no culture index
appears in two splits. [J] shows the contrast against the old splitter directly,
so that the test also documents WHY the change was made.

Requires numpy and torch (torch only because MEAWindowDataset is a
torch.utils.data.Dataset). No training is performed and no augmentation is
executed: only the window INDEX is inspected, so the whole file runs in about a
second.

Run:
    cd Main && PYTHONPATH=. python3 Smoke_Tests/smoke_test_trace_splits.py

Checks (letters match Section 9.2 of the handoff):
  A. No culture in two splits, for both modes and several seeds.
  B. Every class present in every split.
  C. Requested counts: exact when n_c is divisible, else within one and summing
     to n_c. Also demonstrates that alloc_rule='floor' FAILS this at n_c = 18.
  D. Window tiling equals MEAWindowDataset's own tiling, start for start.
  E. Leave-one-out coverage: each culture is test exactly once, no empty train.
  F. trace_of_window[i] is the culture window i was really cut from, verified by
     independent reconstruction rather than by trusting the splitter.
  G. Determinism across repeated calls.
  H. Guard rails raise ValueError naming the offending quantity.
  I. apportion() invariants over a sweep of n and fractions.
  J. Leakage contrast against make_time_segment_splits.
"""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DataConfig                                    # noqa: E402
from data_pipeline import MEAWindowDataset                       # noqa: E402
from data_splits import (                                        # noqa: E402
    apportion,
    assign_cultures,
    make_time_segment_splits,
    make_trace_splits,
    window_starts,
)

_SPLITS = ("train", "val", "test")

# toy geometry: fs = 10 Hz, 2 s window, 1 s train stride, 2 s eval stride
FS = 10.0
WINDOW_S = 2.0
TRAIN_STRIDE_S = 1.0
EVAL_STRIDE_S = 2.0
W = int(round(WINDOW_S * FS))                     # 20 samples
TRAIN_STRIDE = int(round(TRAIN_STRIDE_S * FS))    # 10
EVAL_STRIDE = int(round(EVAL_STRIDE_S * FS))      # 20


def _cfg(fractions=(0.6, 0.2, 0.2)):
    """A DataConfig with the toy geometry above and tiny augmentation pools."""
    cfg = DataConfig()
    cfg.window_s = WINDOW_S
    cfg.train_stride_s = TRAIN_STRIDE_S
    cfg.eval_stride_s = EVAL_STRIDE_S
    cfg.split_fractions = tuple(fractions)
    cfg.augmentation.n_positives = 1
    cfg.augmentation.n_negatives = 1
    return cfg


def _make_traces(n_per_class, length=400, seed=0):
    """Distinguishable traces: culture u is the constant array u + 1.

    Constant-valued traces make provenance checkable by VALUE: if window i came
    from culture u, every sample in it equals u + 1. That turns [F] from a
    bookkeeping comparison into a data comparison.
    """
    rng = np.random.default_rng(seed)
    traces, conditions = [], []
    u = 0
    for c, n_c in enumerate(n_per_class):
        for _ in range(n_c):
            traces.append(np.full(length, float(u + 1), dtype=np.float32))
            conditions.append(c)
            u += 1
    del rng
    return traces, np.asarray(conditions, dtype=int)


# --------------------------------------------------------------------------- #
# [A] no culture in two splits
# --------------------------------------------------------------------------- #
def check_disjoint():
    n_checked = 0
    for n_per_class in [(4, 4), (18, 18), (6, 6, 6, 6), (5, 7)]:
        _tr, cond = _make_traces(n_per_class, length=100)
        for seed in (0, 1, 2, 17):
            a = assign_cultures(cond, (0.6, 0.2, 0.2), seed=seed,
                                mode="fractional")
            sets = [set(a[s]) for s in _SPLITS]
            for i in range(3):
                for j in range(i + 1, 3):
                    assert not (sets[i] & sets[j]), (
                        "overlap between %s and %s at n_per_class=%r seed=%d: %r"
                        % (_SPLITS[i], _SPLITS[j], n_per_class, seed,
                           sorted(sets[i] & sets[j])))
            assert set().union(*sets) == set(range(len(cond))), (
                "assignment is not a partition: some culture went nowhere")
            assert sum(len(s) for s in sets) == len(cond)
            n_checked += 1
        # leave_one_out. The unequal-class-size warning is EXPECTED for (5, 7)
        # and is asserted explicitly in [H]; silence it here to keep the log
        # readable rather than to hide it.
        n_folds = min(n_per_class)
        for fold in range(min(3, n_folds)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                a = assign_cultures(cond, mode="leave_one_out", fold=fold)
            sets = [set(a[s]) for s in _SPLITS]
            for i in range(3):
                for j in range(i + 1, 3):
                    assert not (sets[i] & sets[j])
            assert set().union(*sets) == set(range(len(cond)))
            n_checked += 1
    print("      %d assignments checked, all three splits pairwise disjoint and "
          "covering" % n_checked)


# --------------------------------------------------------------------------- #
# [B] every class in every split
# --------------------------------------------------------------------------- #
def check_all_classes_present():
    for n_per_class in [(4, 4), (18, 18), (6, 6, 6, 6), (5, 7)]:
        _tr, cond = _make_traces(n_per_class, length=100)
        classes = sorted(set(int(c) for c in cond))
        for seed in (0, 1, 2):
            a = assign_cultures(cond, seed=seed, mode="fractional")
            for name in _SPLITS:
                present = sorted(set(int(cond[u]) for u in a[name]))
                assert present == classes, (
                    "split '%s' is missing class(es) %r (n_per_class=%r, seed=%d)"
                    % (name, sorted(set(classes) - set(present)), n_per_class,
                       seed))
        for fold in range(min(n_per_class)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                a = assign_cultures(cond, mode="leave_one_out", fold=fold)
            for name in _SPLITS:
                present = sorted(set(int(cond[u]) for u in a[name]))
                assert present == classes
    print("      every class occupies every split in both modes")


# --------------------------------------------------------------------------- #
# [C] requested counts
# --------------------------------------------------------------------------- #
def check_counts():
    frac = (0.6, 0.2, 0.2)

    # divisible case: 10 cultures per class -> exactly 6 / 2 / 2
    _tr, cond = _make_traces((10, 10))
    for seed in (0, 1, 2, 3):
        a = assign_cultures(cond, frac, seed=seed, mode="fractional")
        for c in (0, 1):
            counts = [sum(1 for u in a[name] if int(cond[u]) == c)
                      for name in _SPLITS]
            assert counts == [6, 2, 2], (
                "n_c = 10 under 60/20/20 must give 6/2/2 exactly; got %r "
                "(class %d, seed %d)" % (counts, c, seed))
    print("      n_c = 10, 60/20/20 -> exactly 6/2/2 for every class and seed")

    # non-divisible: within one of the ideal, and summing to n_c
    for n_c in (7, 11, 13, 17, 18, 19, 23):
        _tr, cond = _make_traces((n_c, n_c))
        a = assign_cultures(cond, frac, seed=0, mode="fractional")
        for c in (0, 1):
            counts = [sum(1 for u in a[name] if int(cond[u]) == c)
                      for name in _SPLITS]
            assert sum(counts) == n_c, "counts %r do not sum to n_c=%d" % (
                counts, n_c)
            for k, f in enumerate(frac):
                ideal = f * n_c
                assert abs(counts[k] - ideal) < 1.0, (
                    "n_c=%d split '%s': assigned %d but ideal is %.2f -- off by "
                    "%.2f, more than one culture"
                    % (n_c, _SPLITS[k], counts[k], ideal,
                       abs(counts[k] - ideal)))
    print("      n_c in {7,11,13,17,18,19,23}: every count within one of its "
          "ideal, and summing to n_c")

    # the handoff's literal floor rule does NOT satisfy that at n_c = 18
    floor_counts = apportion(18, frac, rule="floor")
    lr_counts = apportion(18, frac, rule="largest_remainder")
    assert floor_counts == [10, 3, 5], floor_counts
    assert lr_counts == [11, 4, 3], lr_counts
    worst_floor = max(abs(floor_counts[k] - frac[k] * 18) for k in range(3))
    worst_lr = max(abs(lr_counts[k] - frac[k] * 18) for k in range(3))
    assert worst_floor > 1.0, (
        "expected the floor rule to violate the within-one property at n_c=18")
    assert worst_lr < 1.0
    print("      n_c = 18: floor rule gives %r (worst error %.2f cultures, "
          "VIOLATES [C]); largest_remainder gives %r (worst error %.2f)"
          % (floor_counts, worst_floor, lr_counts, worst_lr))


# --------------------------------------------------------------------------- #
# [D] window tiling matches MEAWindowDataset exactly
# --------------------------------------------------------------------------- #
def check_tiling():
    cfg = _cfg()
    for length in (100, 137, 400, 401, 19):
        for stride in (TRAIN_STRIDE, EVAL_STRIDE):
            reference = [j * stride for j in range(10 ** 6)
                         if j * stride + W <= length]
            helper = window_starts(length, W, stride)
            assert helper == reference, (
                "window_starts disagrees with the reference rule at "
                "length=%d stride=%d" % (length, stride))
            if length >= W:
                trace = np.zeros(length, dtype=np.float32)
                ds = MEAWindowDataset([trace], [0], window_length=W,
                                      stride=stride,
                                      aug_cfg=cfg.resolved_augmentation(FS),
                                      base_seed=0)
                got = [s for (_ti, s, _c) in ds.index]
                assert got == reference, (
                    "MEAWindowDataset tiling %r != reference %r "
                    "(length=%d stride=%d)" % (got, reference, length, stride))

    # and end-to-end through the splitter, per culture
    traces, cond = _make_traces((4, 4), length=137)
    bundle = make_trace_splits(traces, cond, FS, cfg, mode="leave_one_out",
                              fold=0)
    stride_of = {"train": TRAIN_STRIDE, "val": EVAL_STRIDE, "test": EVAL_STRIDE}
    for name in _SPLITS:
        for u in bundle.cultures[name].tolist():
            starts = [s for (uu, s, _e, _c) in bundle.coverage[name] if uu == u]
            assert starts == window_starts(len(traces[u]), W, stride_of[name]), (
                "culture %d in split '%s' was tiled at %r" % (u, name, starts))
    print("      helper, MEAWindowDataset and the splitter agree on window "
          "starts for lengths {19,100,137,400,401} at both strides")


# --------------------------------------------------------------------------- #
# [E] leave-one-out coverage
# --------------------------------------------------------------------------- #
def check_loo():
    n_c = 4
    _tr, cond = _make_traces((n_c, n_c))
    test_count = {u: 0 for u in range(len(cond))}
    val_count = {u: 0 for u in range(len(cond))}
    for fold in range(n_c):
        a = assign_cultures(cond, mode="leave_one_out", fold=fold)
        assert len(a["train"]) > 0, "fold %d has an empty train split" % fold
        for c in (0, 1):
            n_train_c = sum(1 for u in a["train"] if int(cond[u]) == c)
            n_val_c = sum(1 for u in a["val"] if int(cond[u]) == c)
            n_test_c = sum(1 for u in a["test"] if int(cond[u]) == c)
            assert (n_test_c, n_val_c) == (1, 1), (
                "fold %d class %d: expected 1 test and 1 val culture, got "
                "(%d, %d)" % (fold, c, n_test_c, n_val_c))
            assert n_train_c == n_c - 2
            assert n_train_c >= 2, (
                "fold %d class %d has %d training culture(s); cross-culture "
                "positives need >= 2" % (fold, c, n_train_c))
        for u in a["test"]:
            test_count[u] += 1
        for u in a["val"]:
            val_count[u] += 1
    assert all(v == 1 for v in test_count.values()), (
        "not every culture was test exactly once: %r" % (test_count,))
    assert all(v == 1 for v in val_count.values())
    print("      n_c = 4: over 4 folds each culture is test exactly once and "
          "val exactly once; every fold leaves 2 training cultures per class")


# --------------------------------------------------------------------------- #
# [F] trace_of_window correctness, verified independently
# --------------------------------------------------------------------------- #
def check_trace_of_window():
    cfg = _cfg()
    traces, cond = _make_traces((6, 6), length=137)
    bundle = make_trace_splits(traces, cond, FS, cfg, split_seed=3)
    stride_of = {"train": TRAIN_STRIDE, "val": EVAL_STRIDE, "test": EVAL_STRIDE}

    for name in _SPLITS:
        ds = getattr(bundle, name)
        g = bundle.trace_of_window[name]
        assert g.shape[0] == len(ds), (
            "trace_of_window['%s'] has %d entries but the dataset has %d windows"
            % (name, g.shape[0], len(ds)))

        # independent reconstruction from the raw trace lengths
        expected = []
        for u in sorted(bundle.cultures[name].tolist()):
            expected.extend([u] * len(window_starts(len(traces[u]), W,
                                                    stride_of[name])))
        assert g.tolist() == expected, (
            "split '%s': trace_of_window %r != independently reconstructed %r"
            % (name, g.tolist()[:12], expected[:12]))

        # value-level confirmation: culture u's trace is the constant u + 1, so
        # the window's own samples name the culture it came from
        for i in (0, len(ds) // 2, len(ds) - 1):
            ti, s, _c = ds.index[i]
            sample = float(ds.traces[ti][s])
            assert abs(sample - (g[i] + 1.0)) < 1e-6, (
                "split '%s' window %d: samples say culture %d but "
                "trace_of_window says %d"
                % (name, i, int(round(sample)) - 1, g[i]))

        # labels must agree with the culture's own condition
        for i in range(len(ds)):
            assert int(ds.conditions_per_item[i]) == int(cond[g[i]]), (
                "split '%s' window %d: label %d but culture %d has condition %d"
                % (name, i, int(ds.conditions_per_item[i]), g[i],
                   int(cond[g[i]])))
    print("      trace_of_window matches independent reconstruction, the "
          "window CONTENTS, and the label vector, in all three splits")


# --------------------------------------------------------------------------- #
# [G] determinism
# --------------------------------------------------------------------------- #
def check_determinism():
    _tr, cond = _make_traces((9, 9, 9))
    for seed in (0, 1, 5, 99):
        a1 = assign_cultures(cond, seed=seed, mode="fractional")
        a2 = assign_cultures(cond, seed=seed, mode="fractional")
        assert a1 == a2, "assign_cultures is not deterministic at seed %d" % seed
    # different seeds must actually do something different, or "seeded" is a lie
    a0 = assign_cultures(cond, seed=0, mode="fractional")
    differs = any(assign_cultures(cond, seed=s, mode="fractional") != a0
                  for s in (1, 2, 3, 4, 5))
    assert differs, "every seed produced the same assignment; the RNG is inert"

    # end to end
    cfg = _cfg()
    traces, cond2 = _make_traces((6, 6), length=137)
    b1 = make_trace_splits(traces, cond2, FS, cfg, split_seed=7)
    b2 = make_trace_splits(traces, cond2, FS, cfg, split_seed=7)
    for name in _SPLITS:
        assert b1.cultures[name].tolist() == b2.cultures[name].tolist()
        assert b1.trace_of_window[name].tolist() == \
            b2.trace_of_window[name].tolist()

    # adding a class must not perturb the classes already present
    _t3, cond_2cls = _make_traces((6, 6))
    _t4, cond_3cls = _make_traces((6, 6, 6))
    a2c = assign_cultures(cond_2cls, seed=11, mode="fractional")
    a3c = assign_cultures(cond_3cls, seed=11, mode="fractional")
    for name in _SPLITS:
        old = [u for u in a2c[name]]
        new = [u for u in a3c[name] if u < 12]
        assert old == new, (
            "adding a third class reshuffled the first two: '%s' was %r, now %r"
            % (name, old, new))
    print("      repeated calls identical; distinct seeds differ; adding a "
          "class leaves the existing classes' assignment untouched")


# --------------------------------------------------------------------------- #
# [H] guard rails
# --------------------------------------------------------------------------- #
def _expect_value_error(fn, must_mention):
    try:
        fn()
    except ValueError as ex:
        msg = str(ex)
        for token in must_mention:
            assert token in msg, (
                "ValueError raised but its message does not mention %r: %s"
                % (token, msg))
        return msg
    raise AssertionError("expected a ValueError, none was raised")


def check_guard_rails():
    _tr, cond1 = _make_traces((1, 1))
    _expect_value_error(
        lambda: assign_cultures(cond1, mode="fractional"),
        ["n_c = 1"])
    _tr, cond2 = _make_traces((2, 2))
    _expect_value_error(
        lambda: assign_cultures(cond2, (0.6, 0.2, 0.2), mode="fractional",
                                min_train_cultures_per_class=2),
        ["n_c = 2", "0.6"])
    # 3 cultures with min_train = 1 is feasible; with min_train = 2 it is not.
    # 60/20/20 on n_c = 3 apportions to 2/1/0, which the minima repair to 1/1/1,
    # so each of the 2 classes puts exactly one culture in each split.
    _tr, cond3 = _make_traces((3, 3))
    ok = assign_cultures(cond3, mode="fractional",
                         min_train_cultures_per_class=1)
    assert all(len(ok[n]) == 2 for n in _SPLITS), (
        "expected 1 culture per class per split after the minima repair; got "
        "%r" % ({n: ok[n] for n in _SPLITS},))
    for c in (0, 1):
        per_split = [sum(1 for u in ok[n] if int(cond3[u]) == c)
                     for n in _SPLITS]
        assert per_split == [1, 1, 1], (c, per_split)
    _expect_value_error(
        lambda: assign_cultures(cond3, mode="fractional",
                                min_train_cultures_per_class=2),
        ["n_c = 3"])
    # leave_one_out needs an explicit, in-range fold
    _tr, cond4 = _make_traces((4, 4))
    _expect_value_error(lambda: assign_cultures(cond4, mode="leave_one_out"),
                        ["fold"])
    _expect_value_error(
        lambda: assign_cultures(cond4, mode="leave_one_out", fold=4), ["fold"])
    _expect_value_error(
        lambda: assign_cultures(cond4, mode="nonsense"), ["mode"])
    _expect_value_error(lambda: apportion(10, (0.5, 0.2, 0.2)), ["sum"])
    # a window longer than every trace
    cfg = _cfg()
    traces, cond5 = _make_traces((4, 4), length=5)     # 5 samples < W = 20
    _expect_value_error(
        lambda: make_trace_splits(traces, cond5, FS, cfg,
                                  mode="leave_one_out", fold=0),
        ["0 windows"])
    # unequal class sizes under leave_one_out must WARN, not pass silently:
    # "each culture is test exactly once" stops being true there
    _tr, cond6 = _make_traces((5, 7))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assign_cultures(cond6, mode="leave_one_out", fold=0)
    assert any(issubclass(w.category, RuntimeWarning)
               and "exactly once" in str(w.message) for w in caught), (
        "leave_one_out with class sizes (5, 7) must warn that exact-once test "
        "coverage does not hold; warnings caught: %r"
        % ([str(w.message) for w in caught],))
    print("      n_c=1, n_c=2 at min_train=2, n_c=3 at min_train=2, missing "
          "fold, out-of-range fold, bad mode, bad fractions and an oversized "
          "window all raise ValueError naming the offending quantity")


# --------------------------------------------------------------------------- #
# [I] apportion invariants
# --------------------------------------------------------------------------- #
def check_apportion():
    fracs = [(0.6, 0.2, 0.2), (0.5, 0.25, 0.25), (0.8, 0.1, 0.1),
             (1.0 / 3, 1.0 / 3, 1.0 / 3)]
    for frac in fracs:
        for n in range(0, 60):
            counts = apportion(n, frac)
            assert sum(counts) == n, (frac, n, counts)
            assert all(c >= 0 for c in counts)
            for k, f in enumerate(frac):
                ideal = f * n
                assert int(np.floor(ideal)) <= counts[k] <= int(np.ceil(ideal)), (
                    "apportion(%d, %r)[%d] = %d is outside "
                    "[floor(%.3f), ceil(%.3f)]" % (n, frac, k, counts[k],
                                                   ideal, ideal))
    print("      apportion: sums exact and every part in [floor(ideal), "
          "ceil(ideal)] over 4 fraction triples x n in [0, 60)")


# --------------------------------------------------------------------------- #
# [J] leakage contrast against the old splitter
# --------------------------------------------------------------------------- #
def check_leakage_contrast():
    cfg = _cfg()
    traces, cond = _make_traces((6, 6), length=400)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = make_time_segment_splits(traces, cond, FS, cfg)
    new = make_trace_splits(traces, cond, FS, cfg, split_seed=0)

    old_sets = [set(old.cultures[n].tolist()) for n in _SPLITS]
    new_sets = [set(new.cultures[n].tolist()) for n in _SPLITS]

    assert old_sets[0] == old_sets[1] == old_sets[2], (
        "expected the time-segment splitter to put every culture in every "
        "split; it did not, so this contrast no longer means what it says")
    assert len(old_sets[0]) == len(traces)
    for i in range(3):
        for j in range(i + 1, 3):
            assert not (new_sets[i] & new_sets[j])

    assert old.split_kind == "time_segment" and new.split_kind == "trace"
    assert old.seg_bounds and not new.seg_bounds
    print("      old splitter: all %d cultures in all 3 splits. new splitter: "
          "%d / %d / %d cultures, pairwise disjoint"
          % (len(old_sets[0]), len(new_sets[0]), len(new_sets[1]),
             len(new_sets[2])))


# --------------------------------------------------------------------------- #
def main():
    groups = [
        ("A", "no culture in two splits", check_disjoint),
        ("B", "every class in every split", check_all_classes_present),
        ("C", "requested counts", check_counts),
        ("D", "tiling matches MEAWindowDataset", check_tiling),
        ("E", "leave-one-out coverage", check_loo),
        ("F", "trace_of_window correctness", check_trace_of_window),
        ("G", "determinism", check_determinism),
        ("H", "guard rails fire", check_guard_rails),
        ("I", "apportion invariants", check_apportion),
        ("J", "leakage contrast vs time-segment split", check_leakage_contrast),
    ]
    print("smoke_test_trace_splits.py  [Change 5: whole-culture split]")
    failures = []
    for letter, title, fn in groups:
        try:
            fn()
        except Exception as ex:                    # noqa: BLE001
            failures.append((letter, title, ex))
            print("  [%s] %-42s FAIL" % (letter, title))
            print("      %s: %s" % (type(ex).__name__, ex))
        else:
            print("  [%s] %-42s PASS" % (letter, title))
    if failures:
        print("FAILED: %d of %d assertion group(s): %s"
              % (len(failures), len(groups),
                 ", ".join(f[0] for f in failures)))
        return 1
    print("ALL TRACE-SPLIT CHECKS PASSED (%d groups)" % len(groups))
    return 0


if __name__ == "__main__":
    sys.exit(main())
