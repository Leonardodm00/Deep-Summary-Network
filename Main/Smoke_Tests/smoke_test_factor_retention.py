"""
smoke_test_factor_retention.py

Standalone correctness checks for factor_retention.py [C5]. No torch, no data
files, CPU only, a few seconds. Every embedding is CONSTRUCTED, so the correct
answer is known in advance rather than judged by eye.

Run:
    cd Main/Smoke_Tests && python3 smoke_test_factor_retention.py
    # or, from Main/:  PYTHONPATH=. python3 Smoke_Tests/smoke_test_factor_retention.py

Checks:
  A. RECOVERY. Z is built to CONTAIN phi by fiat (phi in the first n columns,
     noise elsewhere) -> R^2_k ~= 1 for every axis k.
  B. NULL. Z is pure noise, independent of phi -> R^2_k is NOT high; the pooled
     out-of-fold R^2 sits at or below 0, which is what a useless predictor scores.
  C. GROUPING IS BY TRACE -- the decisive one. Z encodes ONLY trace identity
     (a one-hot of the trace index). Grouped by trace, a held-out trace's column
     is unseen in training, so the model cannot beat the constant and R^2 <= 0.
     Ungrouped (split by window), a sibling window of the same trace sits in the
     training fold carrying the identical target, so R^2 ~= 1. The test asserts
     BOTH, which is what proves the grouping is load-bearing and actually
     applied: a test that only checked the grouped number would pass even if
     GroupKFold were silently replaced by KFold on data where it makes no
     difference.
  D. The effective sample size warning case: n_groups < 2 must raise, and
     n_splits > n_groups must raise.
  E. A constant target (zero variance across the whole evaluation set) yields
     NaN, not a spurious 1.0, and warns.
  F. align_latents_to_windows expands per-TRACE latents to per-WINDOW rows
     correctly, both by position and by (condition, trace_id) identity.
  G. End-to-end through factor_retention_from_ground_truth with a stand-in
     dataset object, using a REAL latent ground-truth table.
  H. label vs free axes are reported separately and the free-axis mean ignores
     the label axes.
"""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor_retention import (                                   # noqa: E402
    align_latents_to_windows,
    factor_retention,
    factor_retention_from_ground_truth,
    grouped_cv_r2,
)
from latent_burst_generator import (                             # noqa: E402
    DEFAULT_AXIS_NAMES,
    build_latent_spec,
    latent_ground_truth_table,
)

from sklearn.linear_model import RidgeCV                         # noqa: E402
from sklearn.model_selection import KFold                        # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _make_case(n_traces=9, windows_per_trace=4, n_latent=4, seed=0):
    """A synthetic evaluation set: n_traces traces, each cut into
    windows_per_trace windows. phi is a property of the TRACE, so every window of
    a trace shares its latent vector exactly -- the structure that makes the
    grouping matter."""
    rng = np.random.default_rng(seed)
    phi_by_trace = rng.uniform(0.0, 1.0, size=(n_traces, n_latent))
    groups = np.repeat(np.arange(n_traces), windows_per_trace)
    Phi = phi_by_trace[groups, :]
    return Phi, groups, phi_by_trace


def _ungrouped_r2(Z, target, n_splits=5, seed=0):
    """The SAME computation as grouped_cv_r2 but split by WINDOW -- i.e. the bug
    this module exists to prevent. Used only to show the two disagree."""
    Z = np.asarray(Z, float)
    target = np.asarray(target, float).ravel()
    pred = np.full(target.shape, np.nan)
    for tr, te in KFold(n_splits=n_splits, shuffle=True,
                        random_state=seed).split(Z):
        model = RidgeCV(alphas=np.asarray([1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3]))
        model.fit(Z[tr], target[tr])
        pred[te] = model.predict(Z[te])
    bar = float(np.mean(target))
    return float(1.0 - np.sum((target - pred) ** 2) / np.sum((target - bar) ** 2))


# --------------------------------------------------------------------------- #
# A -- recovery
# --------------------------------------------------------------------------- #
def check_recovery():
    Phi, groups, _ = _make_case()
    rng = np.random.default_rng(1)
    n, k = Phi.shape
    # Z contains phi by fiat, plus 6 nuisance dimensions
    Z = np.hstack([Phi, rng.normal(0.0, 1.0, size=(n, 6))])
    out = factor_retention(Z, Phi, groups, label_axes=(0, 1),
                           axis_names=["a", "b", "c", "d"])
    r2 = [a["r2"] for a in out["per_axis"]]
    for i, v in enumerate(r2):
        assert v > 0.99, "axis %d: R^2 = %.4f, expected ~1" % (i, v)
    print("  [A] Z contains phi by fiat -> R^2_k = %s (all ~1) OK"
          % ["%.4f" % v for v in r2])


# --------------------------------------------------------------------------- #
# B -- null
# --------------------------------------------------------------------------- #
def check_null():
    Phi, groups, _ = _make_case()
    rng = np.random.default_rng(2)
    Z = rng.normal(0.0, 1.0, size=(Phi.shape[0], 10))
    out = factor_retention(Z, Phi, groups, label_axes=())
    r2 = [a["r2"] for a in out["per_axis"]]
    for i, v in enumerate(r2):
        assert v < 0.2, "axis %d: R^2 = %.4f on NOISE, expected <= ~0" % (i, v)
    print("  [B] Z is pure noise -> R^2_k = %s (none high) OK"
          % ["%+.4f" % v for v in r2])


# --------------------------------------------------------------------------- #
# C -- the decisive grouping test
# --------------------------------------------------------------------------- #
def check_grouping_is_by_trace():
    n_traces, wpt = 9, 4
    Phi, groups, _ = _make_case(n_traces=n_traces, windows_per_trace=wpt, seed=3)
    # Z encodes ONLY trace identity: a one-hot of the trace index.
    Z = np.eye(n_traces)[groups]
    target = Phi[:, 0]

    grouped = grouped_cv_r2(Z, target, groups, n_splits=3)["r2"]
    ungrouped = _ungrouped_r2(Z, target, n_splits=5)

    assert grouped < 0.1, (
        "GROUPED R^2 = %.4f: a held-out trace's one-hot column is unseen in "
        "training, so the model MUST fall back to the constant. A high value "
        "here means the split is not actually by trace." % grouped)
    assert ungrouped > 0.9, (
        "UNGROUPED R^2 = %.4f: expected the leak to be large and obvious on "
        "this construction; if it is not, the test has lost its teeth."
        % ungrouped)
    print("  [C] trace-identity embedding: grouped R^2 = %+.4f (correct: no "
          "generalization) vs ungrouped R^2 = %+.4f (leaked). Grouping is "
          "load-bearing AND applied OK" % (grouped, ungrouped))

    # and the grouping really is by trace: no group spans a fold boundary
    from sklearn.model_selection import GroupKFold
    for tr, te in GroupKFold(n_splits=3).split(Z, target, groups=groups):
        assert not (set(groups[tr]) & set(groups[te])), (
            "a trace appeared in BOTH the training and the held-out fold")
    print("  [C] no trace appears in both sides of any split OK")


# --------------------------------------------------------------------------- #
# D -- guards
# --------------------------------------------------------------------------- #
def check_guards():
    Phi, groups, _ = _make_case(n_traces=3, windows_per_trace=4)
    Z = np.hstack([Phi, np.zeros((Phi.shape[0], 2))])

    try:
        grouped_cv_r2(Z, Phi[:, 0], np.zeros_like(groups))   # one group only
    except ValueError as ex:
        assert "at least 2 distinct groups" in str(ex), str(ex)
    else:
        raise AssertionError("n_groups < 2 did not raise")

    try:
        grouped_cv_r2(Z, Phi[:, 0], groups, n_splits=9)      # > n_groups
    except ValueError as ex:
        assert "n_splits" in str(ex), str(ex)
    else:
        raise AssertionError("n_splits > n_groups did not raise")

    try:
        grouped_cv_r2(Z[:5], Phi[:, 0], groups)              # shape mismatch
    except ValueError as ex:
        assert "N_eval" in str(ex), str(ex)
    else:
        raise AssertionError("shape mismatch did not raise")
    print("  [D] guards fire: n_groups < 2, n_splits > n_groups, shape mismatch OK")


# --------------------------------------------------------------------------- #
# E -- constant target
# --------------------------------------------------------------------------- #
def check_constant_target():
    Phi, groups, _ = _make_case(n_traces=6, windows_per_trace=3)
    Z = np.hstack([Phi, np.zeros((Phi.shape[0], 1))])
    const = np.full(Phi.shape[0], 0.5)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = grouped_cv_r2(Z, const, groups, n_splits=3)
    assert np.isnan(out["r2"]), "constant target should give NaN, got %r" % out["r2"]
    assert any("constant" in str(x.message) for x in w), [str(x.message) for x in w]
    print("  [E] constant target -> R^2 = NaN (not a spurious 1.0) + warning OK")


# --------------------------------------------------------------------------- #
# F -- alignment
# --------------------------------------------------------------------------- #
def check_alignment():
    spec = build_latent_spec(DEFAULT_AXIS_NAMES, (0, 1), (3, 3, 3), 600.0, 50.0)
    gt = latent_ground_truth_table(spec)
    groups = np.repeat(np.arange(9), 4)

    Phi = align_latents_to_windows(gt, groups)
    assert Phi.shape == (36, 6), Phi.shape
    for t in range(9):
        rows = Phi[groups == t]
        assert np.allclose(rows, rows[0]), "windows of trace %d disagree" % t
        assert np.allclose(rows[0], gt["rows"][t]["phi"]), (
            "trace %d got the wrong latent vector" % t)
    print("  [F] positional alignment: 9 traces x 4 windows -> (36, 6), every "
          "window carries its own trace's phi OK")

    conds = np.asarray([r["condition"] for r in gt["rows"]])
    tids = np.asarray([r["trace_id"] for r in gt["rows"]])
    Phi_id = align_latents_to_windows(gt, groups, conditions=conds, trace_ids=tids)
    assert np.allclose(Phi, Phi_id)
    print("  [F] identity-based alignment agrees with positional alignment OK")

    try:
        align_latents_to_windows(gt, groups, conditions=conds,
                                 trace_ids=tids + 100)
    except KeyError:
        print("  [F] a (condition, trace_id) absent from the table raises OK")
    else:
        raise AssertionError("mismatched trace_ids did not raise")


# --------------------------------------------------------------------------- #
# G + H -- end to end on a real ground-truth table
# --------------------------------------------------------------------------- #
class _FakeDataset:
    """Stand-in for MEAWindowDataset: only .index is read."""

    def __init__(self, index):
        self.index = index


def check_end_to_end():
    spec = build_latent_spec(DEFAULT_AXIS_NAMES, (0, 1), (3, 3, 3), 600.0, 50.0)
    gt = latent_ground_truth_table(spec)
    n_traces, wpt = 9, 4
    index = [(t, 100 * w, int(t // 3)) for t in range(n_traces) for w in range(wpt)]
    ds = _FakeDataset(index)
    groups = np.asarray([t[0] for t in index])
    Phi = align_latents_to_windows(gt, groups)

    rng = np.random.default_rng(7)
    # An embedding that RETAINS the two label axes and DESTROYS the free ones --
    # the qualitative signature the hard-mining hypothesis predicts.
    Z = np.hstack([Phi[:, [0, 1]], rng.normal(0, 0.01, size=(len(index), 6))])
    out = factor_retention_from_ground_truth(Z, gt, ds, n_splits=3)

    by_idx = {a["index"]: a for a in out["per_axis"]}
    assert out["label_axes"] == [0, 1], out["label_axes"]
    assert out["free_axes"] == [2, 3, 4, 5], out["free_axes"]
    for k in (0, 1):
        assert by_idx[k]["is_label_axis"]
        assert by_idx[k]["r2"] > 0.95, (k, by_idx[k]["r2"])
    for k in (2, 3, 4, 5):
        assert not by_idx[k]["is_label_axis"]
        assert by_idx[k]["r2"] < 0.5, (k, by_idx[k]["r2"])
    assert out["mean_r2_label_axes"] > 0.95
    assert out["mean_r2_free_axes"] < 0.5
    assert by_idx[0]["name"] == "irregularity", by_idx[0]["name"]
    print("  [G] end-to-end via the dataset index: label axes R^2 = %.4f, free "
          "axes R^2 = %+.4f OK"
          % (out["mean_r2_label_axes"], out["mean_r2_free_axes"]))
    print("  [H] label / free axes reported separately, names carried through "
          "(axis 0 = %r) OK" % by_idx[0]["name"])

    try:
        factor_retention_from_ground_truth(Z[:10], gt, ds)
    except ValueError as ex:
        assert "dataset has" in str(ex), str(ex)
        print("  [H] Z / dataset length mismatch raises OK")
    else:
        raise AssertionError("length mismatch did not raise")


def main():
    print("smoke_test_factor_retention.py [C5]")
    check_recovery()
    check_null()
    check_grouping_is_by_trace()
    check_guards()
    check_constant_target()
    check_alignment()
    check_end_to_end()
    print("ALL FACTOR-RETENTION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
