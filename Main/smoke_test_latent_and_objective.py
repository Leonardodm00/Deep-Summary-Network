"""
smoke_test_latent_and_objective.py
==================================

Correctness harness for latent_burst_generator.py and objective_utils.py.

Run
---
    python3 smoke_test_latent_and_objective.py            # full suite, twice
    python3 smoke_test_latent_and_objective.py --fast     # skip signal synthesis
    python3 smoke_test_latent_and_objective.py --once     # single pass

Exit code 0 iff every check passes on BOTH passes (directive 4: control each
script twice). The two passes are independent interpreter-level repetitions of
the same assertions; because every generator path is explicitly seeded, a
discrepancy between pass 1 and pass 2 indicates hidden global RNG state, which
is itself a defect worth failing on.

What is asserted, and why each matters
--------------------------------------
  G1  phi lies in [0, 1]^n                      -- domain invariant of the map
  G2  determinism in (condition, trace_id)      -- reproducibility
  G3  order-independence                        -- pipeline may request traces
                                                   out of order
  G4  tau = 0 collapses label axis to m_c       -- reduces to legacy behaviour
  G5  tau > 0 spreads traces about m_c          -- the difficulty knob works
  G6  label axis tracks the class               -- the label is learnable
  G7  free axes do NOT track the class          -- they are genuinely
                                                   label-irrelevant, which is
                                                   the precondition for the
                                                   factor-retention experiment
  G8  axis maps are strictly monotone           -- invertibility of phi -> value
  G9  participation kappa reproduces CONTROL
      Beta(3,1) and PATHO Beta(2,2)             -- the new space CONTAINS the
                                                   old two-condition setting
  G10 traces are non-negative, correct length
      and sampling rate                         -- interface contract with
                                                   MEAWindowDataset
  G11 traces are genuinely bursty               -- not a flat or noise signal
  G12 the task is NOT trivially separable       -- THE decisive check: the
                                                   2-scalar baseline that
                                                   matched the network on the
                                                   old benchmark must now fail

  O1  Delta_min reproduces the hand-computed
      value for 36 windows / 3 balanced classes -- calibration against a known
                                                   number
  O2  Delta_min decreases as N_eval grows       -- resolution scales correctly
  O3  the lexicographic condition (*) holds     -- the secondary metric can
                                                   never overturn a genuine
                                                   primary difference
  O4  composite_objective NaN handling          -- degenerate embeddings lose
  O5  resolve_n_initial_points legacy + bounds  -- backward compatibility

HPC note (hpc-python-compat): pure ASCII. Every local module in the import
chain (latent_burst_generator, generate_burst_data, objective_utils) is pure
ASCII as well.
"""

import argparse
import sys

import numpy as np
from scipy import signal as sp_signal
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from latent_burst_generator import (
    LatentSpec, LatentBurstProvider, DEFAULT_AXES, PARTICIPATION_KAPPA,
    sample_latents, latent_to_burst_params, latent_ground_truth_table,
)
from objective_utils import (
    min_ari_gap, adaptive_epsilon, composite_objective, resolve_n_initial_points,
)

_RESULTS = []


def _check(name, ok, detail=""):
    _RESULTS.append(bool(ok))
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                           ("  -- " + detail) if detail else ""))
    return bool(ok)


# --------------------------------------------------------------------------- #
# generator checks
# --------------------------------------------------------------------------- #
def check_generator(fast=False):
    print("\n--- latent_burst_generator ---")
    spec = LatentSpec(n_classes=3, n_per_class=(3, 3, 3), class_overlap=0.12,
                      duration_s=120.0, n_neurons=60, seed=0)
    n = spec.n_latent

    # G1 / G2 / G3
    phis = {}
    ok_domain = True
    for c in range(3):
        for r in range(3):
            phi = sample_latents(spec, c, r)
            phis[(c, r)] = phi
            ok_domain &= bool(np.all(phi >= 0.0) and np.all(phi <= 1.0)
                              and phi.shape == (n,))
    _check("G1 phi in [0,1]^n", ok_domain)
    _check("G2 determinism", all(
        np.array_equal(phis[(c, r)], sample_latents(spec, c, r))
        for c in range(3) for r in range(3)))
    rev = {(c, r): sample_latents(spec, c, r)
           for c in reversed(range(3)) for r in reversed(range(3))}
    _check("G3 order-independence",
           all(np.array_equal(phis[k], rev[k]) for k in phis))

    # G4 / G5
    spec0 = LatentSpec(n_classes=3, n_per_class=(4, 4, 4), class_overlap=0.0, seed=0)
    lab0 = spec0.label_axes[0]
    vals_c1 = [sample_latents(spec0, 1, r)[lab0] for r in range(4)]
    _check("G4 tau=0 collapses label axis", float(np.std(vals_c1)) < 1e-12,
           "std=%.2e, all=%.4f" % (float(np.std(vals_c1)), vals_c1[0]))
    specT = LatentSpec(n_classes=3, n_per_class=(40, 40, 40), class_overlap=0.15, seed=0)
    labT = specT.label_axes[0]
    vals_c1T = np.array([sample_latents(specT, 1, r)[labT] for r in range(40)])
    _check("G5 tau>0 spreads label axis", float(vals_c1T.std()) > 0.05,
           "std=%.4f about m_1=0.5" % float(vals_c1T.std()))

    # G6 / G7  -- correlation of each axis with the class label
    specC = LatentSpec(n_classes=3, n_per_class=(40, 40, 40), class_overlap=0.15, seed=0)
    rows, ys = [], []
    for c in range(3):
        for r in range(40):
            rows.append(sample_latents(specC, c, r))
            ys.append(c)
    P = np.asarray(rows)
    yv = np.asarray(ys, dtype=float)
    corr = np.array([abs(np.corrcoef(P[:, k], yv)[0, 1]) for k in range(specC.n_latent)])
    lab_idx = list(specC.label_axes)
    free_idx = list(specC.free_axes)
    _check("G6 label axis tracks class",
           all(corr[k] > 0.7 for k in lab_idx),
           "|corr| = %s" % np.round(corr[lab_idx], 3).tolist())
    _check("G7 free axes do NOT track class",
           all(corr[k] < 0.30 for k in free_idx),
           "max |corr| over free axes = %.3f" % float(corr[free_idx].max()))

    # G8 monotone axis maps
    mono = True
    for ax in DEFAULT_AXES:
        grid = np.linspace(0.0, 1.0, 51)
        vals = np.array([ax.value(float(u)) for u in grid])
        d = np.diff(vals)
        mono &= bool(np.all(d > 0) or np.all(d < 0))
    _check("G8 axis maps strictly monotone", mono)

    # G9 participation kappa reproduces CONTROL / PATHO Beta parameters
    part_axis = [i for i, a in enumerate(DEFAULT_AXES)
                 if a.target == "participation_mean"][0]
    phi_ctrl = np.full(spec.n_latent, 0.5)
    ax = DEFAULT_AXES[part_axis]
    # solve phi such that participation_mean = 0.75 and = 0.50
    def phi_for(pbar):
        return (pbar - ax.lo) / (ax.hi - ax.lo)
    ok_beta = True
    for pbar, want in ((0.75, (3.0, 1.0)), (0.50, (2.0, 2.0))):
        u = phi_for(pbar)
        if not (0.0 <= u <= 1.0):
            ok_beta = False
            continue
        ph = phi_ctrl.copy()
        ph[part_axis] = u
        th = latent_to_burst_params(spec, ph)
        ok_beta &= (abs(th.alpha_p - want[0]) < 1e-9 and abs(th.beta_p - want[1]) < 1e-9)
    _check("G9 participation kappa=%.1f reproduces CONTROL/PATHO Beta"
           % PARTICIPATION_KAPPA, ok_beta)

    if fast:
        print("  (--fast: skipping signal-synthesis checks G10-G12)")
        return

    # G10 / G11 -- synthesize traces
    prov = LatentBurstProvider(spec)
    x, fs = prov(0, 0)
    K_expected = int(spec.duration_s / spec.w_size)
    _check("G10 trace contract (non-negative, length, fs)",
           bool(np.all(x >= 0.0)) and abs(fs - spec.fs) < 1e-9
           and abs(len(x) - K_expected) <= 1,
           "len=%d (expected ~%d), fs=%.1f, min=%.3f" % (len(x), K_expected, fs, float(x.min())))

    peaks, _ = sp_signal.find_peaks(x, prominence=float(np.std(x)))
    burst_rate_hz = len(peaks) / spec.duration_s
    _check("G11 traces are genuinely bursty",
           len(peaks) > 5 and float(x.max()) > 3.0 * float(x.mean()),
           "peaks=%d (%.3f /s), max/mean=%.1f"
           % (len(peaks), burst_rate_hz, float(x.max()) / max(float(x.mean()), 1e-9)))

    # G12 -- DECISIVE: the benchmark must be HARD BUT SOLVABLE.
    # Two hand-crafted baselines are scored on the same windows:
    #   naive : {spectral centroid, median peak width} -- the exact pair that
    #           reached ARI 0.9154 on the OLD benchmark, matching the network.
    #   rich  : adds CV of inter-burst interval, CV of peak height, CV of burst
    #           width, burst rate -- features aligned with the label axes.
    # Requirements: naive must FAIL (no longer a shortcut), and rich must land
    # strictly between chance and near-perfect, leaving headroom for a learned
    # model. A benchmark at chance for every baseline is over-corrected and is
    # just as useless for model selection as a saturated one.
    specB = LatentSpec(n_classes=3, n_per_class=(3, 3, 3),
                       duration_s=300.0, n_neurons=60, seed=0)
    provB = LatentBurstProvider(specB)
    W = 1500
    rich, naive, labs = [], [], []
    for c in range(3):
        for r in range(3):
            xt, fst = provB(c, r)
            for s0 in range(0, len(xt) - W + 1, W):
                w = xt[s0:s0 + W].astype(np.float64)
                f, Pxx = sp_signal.welch(w, fs=fst, nperseg=min(len(w), 512))
                f, Pxx = f[1:], Pxx[1:]
                centroid = float((f * Pxx).sum() / Pxx.sum()) if Pxx.sum() > 0 else 0.0
                idx, _ = sp_signal.find_peaks(w, prominence=max(float(w.std()), 1e-6))
                if idx.size >= 3:
                    widths = sp_signal.peak_widths(w, idx, rel_height=0.5)[0] / fst
                    ibi = np.diff(idx) / fst
                    hts = w[idx]
                    cv_ibi = float(np.std(ibi) / max(np.mean(ibi), 1e-9))
                    cv_ht = float(np.std(hts) / max(np.mean(hts), 1e-9))
                    cv_wd = float(np.std(widths) / max(np.mean(widths), 1e-9))
                    rate = float(idx.size / (len(w) / fst))
                    med_w = float(np.median(widths))
                else:
                    cv_ibi = cv_ht = cv_wd = rate = med_w = 0.0
                naive.append([centroid, med_w])
                rich.append([cv_ibi, cv_ht, cv_wd, rate, med_w])
                labs.append(c)
    yb = np.asarray(labs)

    def _ari(mat):
        Z = StandardScaler().fit_transform(np.asarray(mat))
        return float(adjusted_rand_score(
            yb, KMeans(3, random_state=0, n_init=10).fit_predict(Z)))

    ari_naive, ari_rich = _ari(naive), _ari(rich)
    _check("G12a naive 2-scalar shortcut is closed",
           ari_naive < 0.60,
           "naive ARI = %.4f on %d windows (was 0.9154 on the OLD benchmark)"
           % (ari_naive, len(yb)))
    _check("G12b task is hard BUT solvable (headroom exists)",
           0.10 < ari_rich < 0.90,
           "rich-feature ARI = %.4f -- above chance, far below ceiling"
           % ari_rich)
    _check("G12c informative features beat naive ones",
           ari_rich > ari_naive,
           "rich %.4f > naive %.4f" % (ari_rich, ari_naive))

    gt = latent_ground_truth_table(specB)
    _check("G13 ground-truth table complete",
           len(gt["rows"]) == 9 and len(gt["axis_names"]) == specB.n_latent
           and len(gt["free_axes"]) == specB.n_latent - len(specB.label_axes))


# --------------------------------------------------------------------------- #
# objective checks
# --------------------------------------------------------------------------- #
def check_objective():
    print("\n--- objective_utils ---")

    # O1 calibration: 36 windows, 3 balanced classes of 12
    y36 = np.repeat(np.arange(3), 12)
    info36 = min_ari_gap(y36)
    _check("O1 Delta_min calibration (36 windows, 3x12)",
           abs(info36["delta_min"] - 0.0846) < 0.002,
           "Delta_min = %.4f, best ARI below 1 = %.4f"
           % (info36["delta_min"], info36["best_ari_below1"]))

    # O2 resolution improves with N_eval
    gaps = []
    for per in (12, 30, 60):
        yv = np.repeat(np.arange(3), per)
        gaps.append(min_ari_gap(yv)["delta_min"])
    _check("O2 Delta_min shrinks as N_eval grows",
           gaps[0] > gaps[1] > gaps[2],
           "N=36: %.4f  N=90: %.4f  N=180: %.4f" % tuple(gaps))

    # O3 lexicographic condition
    eps_info = adaptive_epsilon(y36, sil_lo=-1.0, sil_hi=1.0, gamma=0.5)
    eps = eps_info["epsilon"]
    holds = eps_info["max_secondary_influence"] < eps_info["delta_min"]
    # empirical: a config that is better on ARI must win regardless of Sil
    j_good_ari_worst_sil = composite_objective(1.0, -1.0, eps)
    j_worse_ari_best_sil = composite_objective(
        1.0 - eps_info["delta_min"], 1.0, eps)
    _check("O3 lexicographic condition (*) holds",
           holds and j_good_ari_worst_sil < j_worse_ari_best_sil,
           "epsilon=%.5f, eps*W=%.5f < Delta_min=%.5f"
           % (eps, eps_info["max_secondary_influence"], eps_info["delta_min"]))

    # tie-break actually breaks ties
    _check("O3b secondary breaks an exact primary tie",
           composite_objective(1.0, 0.97, eps) < composite_objective(1.0, 0.89, eps))

    # O4 NaN handling
    _check("O4 NaN ARI loses, NaN Sil tolerated",
           composite_objective(float("nan"), 1.0, eps) == float("inf")
           and np.isfinite(composite_objective(1.0, float("nan"), eps)))

    # O5 n_initial_points
    legacy_ok = (resolve_n_initial_points(50, None) == 10
                 and resolve_n_initial_points(15, None) == 7
                 and resolve_n_initial_points(1, None) == 1)
    explicit_ok = (resolve_n_initial_points(50, 20) == 20
                   and resolve_n_initial_points(50, 5) == 5)
    raised = False
    try:
        resolve_n_initial_points(10, 20)
    except ValueError:
        raised = True
    _check("O5 n_initial_points legacy + explicit + guard",
           legacy_ok and explicit_ok and raised,
           "legacy(50)=10, legacy(15)=7, explicit honoured, over-budget raises")


# --------------------------------------------------------------------------- #
def run_all(fast=False):
    del _RESULTS[:]
    check_generator(fast=fast)
    check_objective()
    n_pass, n_tot = sum(_RESULTS), len(_RESULTS)
    print("\n  %d/%d checks passed" % (n_pass, n_tot))
    return 0 if n_pass == n_tot else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="skip signal-synthesis checks (G10-G13)")
    ap.add_argument("--once", action="store_true",
                    help="single pass instead of the mandated double run")
    args = ap.parse_args()

    print("=" * 62)
    print("PASS 1")
    print("=" * 62)
    rc1 = run_all(fast=args.fast)
    if args.once:
        return rc1
    print("\n" + "=" * 62)
    print("PASS 2 (directive 4: control each script twice)")
    print("=" * 62)
    rc2 = run_all(fast=args.fast)
    rc = max(rc1, rc2)
    print("\n%s" % ("ALL PASSES OK" if rc == 0 else "FAILURES PRESENT"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
