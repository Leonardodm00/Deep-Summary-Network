#!/usr/bin/env python3
"""
Smoke test for generate_multichannel_ifr (realistic multichannel synthetic).

Verifies:
  1. shape (C, K) and fs_ifr = 1/w_size;
  2. finite and non-negative;
  3. SYNCHRONY: within a recording, channels are positively correlated (they
     share one burst schedule);
  4. DISTINCTNESS: channels are not identical;
  5. SHARED-SCHEDULE demonstration: within-recording cross-channel correlation
     is much higher than across-recording (independent schedules) correlation;
  6. n_channels=1 consistency: the single row equals the whole-population IFR
     bit-for-bit under the same seed (gain_spread=0);
  7. neurons_per_channel scaling and per-channel gain heterogeneity.

Run:
    python3 smoke_test_generate_mc.py
Exit code 0 = all passed.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from generate_burst_data import (  # noqa: E402
    BurstParams, CONTROL_PARAMS, generate_multichannel_ifr,
    generate_spike_times, compute_ifr_trace,
)
from dataclasses import replace  # noqa: E402


def _check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print("[%s] %s%s" % (tag, name, ("  (%s)" % detail) if detail else ""))
    return bool(cond)


def mean_offdiag_corr(X):
    """Mean off-diagonal Pearson correlation of rows of X (C, K)."""
    C = X.shape[0]
    Xc = X - X.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(Xc, axis=1)
    denom[denom == 0] = 1e-12
    R = (Xc @ Xc.T) / np.outer(denom, denom)
    iu = np.triu_indices(C, k=1)
    return float(np.mean(R[iu]))


def main():
    ok = True
    C = 9
    # shorter recording for a fast test
    params = replace(CONTROL_PARAMS, duration_s=120.0)

    # ---- 1,2. shape / fs / finite / non-negative ------------------------- #
    ifr, fs = generate_multichannel_ifr(
        params, n_channels=C, neurons_per_channel=20,
        rng=np.random.default_rng(0))
    K_expected = int(params.duration_s / params.w_size)
    ok &= _check("shape (C, K)", ifr.shape == (C, K_expected),
                 "%s (K_expected=%d)" % (ifr.shape, K_expected))
    ok &= _check("fs_ifr == 1/w_size", abs(fs - 1.0 / params.w_size) < 1e-9,
                 "fs=%.3f" % fs)
    ok &= _check("finite", bool(np.all(np.isfinite(ifr))))
    ok &= _check("non-negative", float(ifr.min()) >= 0.0,
                 "min=%.4g" % ifr.min())

    # ---- 3. synchrony: channels positively correlated -------------------- #
    within = mean_offdiag_corr(ifr)
    ok &= _check("synchrony: mean cross-channel corr > 0.3", within > 0.3,
                 "corr=%.3f" % within)

    # ---- 4. distinctness ------------------------------------------------- #
    max_pair_diff = float(np.max(np.abs(ifr[0] - ifr[1])))
    ok &= _check("channels are distinct (not identical)",
                 within < 0.999 and max_pair_diff > 1e-3,
                 "corr=%.3f, max|c0-c1|=%.3g" % (within, max_pair_diff))

    # ---- 5. shared-schedule demonstration -------------------------------- #
    ifrA, _ = generate_multichannel_ifr(params, n_channels=C,
                                        neurons_per_channel=20,
                                        rng=np.random.default_rng(1))
    ifrB, _ = generate_multichannel_ifr(params, n_channels=C,
                                        neurons_per_channel=20,
                                        rng=np.random.default_rng(2))
    # correlate channel c of A with channel c of B (independent schedules)
    def corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb + 1e-12))
    across = float(np.mean([corr(ifrA[c], ifrB[c]) for c in range(C)]))
    withinA = mean_offdiag_corr(ifrA)
    ok &= _check("within-recording corr >> across-recording corr",
                 withinA > across + 0.2,
                 "within=%.3f across=%.3f" % (withinA, across))

    # ---- 6. n_channels=1 consistency vs whole-population IFR ------------- #
    seed = 123
    ifr1, _ = generate_multichannel_ifr(params, n_channels=1,
                                        channel_gain_spread=0.0,
                                        rng=np.random.default_rng(seed))
    spikes = generate_spike_times(params, np.random.default_rng(seed))
    ifr_pop, _ = compute_ifr_trace(spikes, params)
    ok &= _check("n_channels=1 row == whole-population IFR (bit-for-bit)",
                 ifr1.shape == (1, ifr_pop.shape[0])
                 and np.array_equal(ifr1[0], ifr_pop),
                 "maxdiff=%.3g" % float(np.max(np.abs(ifr1[0] - ifr_pop))))

    # ---- 7. neurons_per_channel scaling + gain heterogeneity ------------- #
    ifr_g, _ = generate_multichannel_ifr(params, n_channels=C,
                                         neurons_per_channel=15,
                                         channel_gain_spread=0.3,
                                         rng=np.random.default_rng(5))
    ok &= _check("neurons_per_channel scaling: shape (C, K)",
                 ifr_g.shape == (C, K_expected), "%s" % (ifr_g.shape,))
    per_ch_mean = ifr_g.mean(axis=1)
    ok &= _check("per-channel gain -> channel amplitudes differ",
                 float(np.std(per_ch_mean)) > 1e-3,
                 "std of channel means=%.4g" % float(np.std(per_ch_mean)))

    print("=" * 60)
    print("SMOKE RESULT: %s" % ("ALL PASSED" if ok else "FAILURES ABOVE"))
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
