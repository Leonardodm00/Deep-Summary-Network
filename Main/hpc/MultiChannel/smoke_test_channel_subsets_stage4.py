"""Smoke test -- Stage 4 (per-subregion IFR + whole-culture IFR).

Scope (one concern): subregion_ifr / subregion_ifrs / whole_culture_ifr.

Controls:
  - shape / dtype / K / fs_ifr and non-negativity;
  - correctness against an INDEPENDENT reference IFR (histogram + gaussian_filter1d
    + clip re-derived in this test), including the sample-index -> seconds
    conversion (t = sample/fs_raw) and the per-electrode normalization (/n_e);
  - reuse identity: subregion_ifr * E equals the raw compute_ifr_trace over the
    same pooled members (no extra processing sneaks in);
  - whole-culture pooling rule: pools FIRING electrodes only (silent excluded),
    applies NO theta filter (a sub-theta firing electrode is still pooled), and
    normalizes by the FIRING count (not the total electrode count);
  - all-silent culture raises; determinism.

Run:
    cd /home/claude/work/hpc_multichannel
    python3 smoke_test_channel_subsets_stage4.py

Run TWICE with exit codes:
    cd /home/claude/work/hpc_multichannel
    python3 smoke_test_channel_subsets_stage4.py; echo "exit1=$?"; \
    python3 smoke_test_channel_subsets_stage4.py; echo "exit2=$?"
"""

from __future__ import annotations

import sys
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from channel_subset_extraction import (
    DEFAULT_GAUSSIAN_WINDOW,
    DEFAULT_W_SIZE,
    PtrainInventory,
    Subregion,
    subregion_ifr,
    subregion_ifrs,
    whole_culture_ifr,
)
from generate_burst_data import compute_ifr_trace  # for the reuse-identity check

DT = DEFAULT_W_SIZE            # 0.02 s
GW = DEFAULT_GAUSSIAN_WINDOW   # 0.04 s


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def make_inv(spikes_samples: Dict[int, List[int]], n_samples: int,
             fs_raw: float) -> PtrainInventory:
    spikes = {int(k): np.asarray(sorted(v), dtype=np.int64)
              for k, v in spikes_samples.items()}
    return PtrainInventory(
        spikes=spikes,
        n_samples=int(n_samples),
        fs_raw=float(fs_raw),
        T_rec=n_samples / float(fs_raw),
        index_base=0,
    )


def _ref_ifr(spike_s_list: List[np.ndarray], T_rec: float,
             dt: float, gw: float) -> Tuple[np.ndarray, int]:
    """Independent reference: pooled histogram -> gaussian smooth -> clip."""
    K = int(T_rec / dt)
    edges = np.arange(K + 1, dtype=np.float64) * dt
    C = np.zeros(K, dtype=np.float64)
    for st in spike_s_list:
        st = np.asarray(st, dtype=np.float64)
        if st.size:
            c, _ = np.histogram(st, bins=edges)
            C += c
    R = gaussian_filter1d(C, sigma=gw / dt)
    R = np.clip(R, 0.0, None).astype(np.float32)
    return R, K


def _planted(rng: np.random.Generator, n: int, n_samples: int) -> List[int]:
    return sorted(int(x) for x in rng.integers(0, n_samples, size=n))


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #

def check_subregion_ifr_correct() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    fs_raw = 1000.0
    n_samples = 4000              # T_rec = 4.0 s -> K = 200
    rng = np.random.default_rng(4041)
    members = (100, 101, 102)     # E = 3
    spikes = {m: _planted(rng, 60 + 10 * i, n_samples) for i, m in enumerate(members)}
    inv = make_inv(spikes, n_samples, fs_raw)
    sub = Subregion(center=100, members=members, mean_mfr=0.0, center_mfr=0.0)

    ifr, fs_ifr = subregion_ifr(inv, sub)
    K_expect = int(inv.T_rec / DT)

    # shape / dtype / fs / K / non-negativity
    if ifr.shape != (K_expect,):
        ok = False; detail.append("shape %r != (%d,)" % (ifr.shape, K_expect))
    if ifr.dtype != np.float32:
        ok = False; detail.append("dtype %r != float32" % ifr.dtype)
    if not np.isclose(fs_ifr, 1.0 / DT):
        ok = False; detail.append("fs_ifr %r != %r" % (fs_ifr, 1.0 / DT))
    if np.any(ifr < 0):
        ok = False; detail.append("negative IFR values present")

    # independent reference (seconds conversion + normalization by E=3)
    secs = [np.asarray(spikes[m], dtype=np.float64) / fs_raw for m in members]
    ref_R, ref_K = _ref_ifr(secs, inv.T_rec, DT, GW)
    if ref_K != K_expect:
        ok = False; detail.append("reference K %d != %d" % (ref_K, K_expect))
    expected = (ref_R / float(len(members))).astype(np.float32)
    if not np.allclose(ifr, expected, rtol=1e-5, atol=1e-6):
        ok = False
        detail.append("IFR != reference/E (max abs diff %.3e)"
                      % float(np.max(np.abs(ifr - expected))))

    # reuse identity: subregion_ifr * E == raw compute_ifr_trace(pooled members)
    from channel_subset_extraction import _ifr_params
    raw, _ = compute_ifr_trace(secs, _ifr_params(inv.T_rec, DT, GW))
    if not np.allclose(ifr * len(members), raw, rtol=1e-5, atol=1e-6):
        ok = False
        detail.append("subregion_ifr*E != raw compute_ifr_trace (max abs diff %.3e)"
                      % float(np.max(np.abs(ifr * len(members) - raw))))
    if ok:
        detail.append("K=%d, fs_ifr=%.1f, matches reference/E and reuse identity"
                      % (K_expect, fs_ifr))
    return ("subregion_ifr correctness + normalization", ok, "; ".join(detail))


def check_subregion_ifrs_stack() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    fs_raw = 1000.0
    n_samples = 2000              # T_rec = 2.0 s -> K = 100
    rng = np.random.default_rng(909)
    subs = []
    all_spikes: Dict[int, List[int]] = {}
    for ch in range(3):
        members = tuple(200 + 10 * ch + j for j in range(4))  # E = 4
        for m in members:
            all_spikes[m] = _planted(rng, 40, n_samples)
        subs.append(Subregion(center=members[0], members=members,
                              mean_mfr=0.0, center_mfr=float(3 - ch)))
    inv = make_inv(all_spikes, n_samples, fs_raw)
    mc, fs_ifr = subregion_ifrs(inv, subs)
    K_expect = int(inv.T_rec / DT)
    if mc.shape != (3, K_expect):
        ok = False; detail.append("shape %r != (3, %d)" % (mc.shape, K_expect))
    if mc.dtype != np.float32:
        ok = False; detail.append("dtype %r != float32" % mc.dtype)
    # each row equals the single-subregion call
    for c, s in enumerate(subs):
        row, _ = subregion_ifr(inv, s)
        if not np.array_equal(mc[c], row):
            ok = False; detail.append("row %d != subregion_ifr" % c); break
    # empty list guard
    try:
        subregion_ifrs(inv, [])
        ok = False; detail.append("empty subregions did not raise")
    except ValueError:
        pass
    if ok:
        detail.append("(C,K)=(3,%d) float32 stack, rows match single calls" % K_expect)
    return ("subregion_ifrs stacking + shape", ok, "; ".join(detail))


def check_whole_culture_rule() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    fs_raw = 1000.0
    n_samples = 4000              # T_rec = 4.0 s
    rng = np.random.default_rng(77)
    A = 10   # firing, high rate
    B = 20   # firing but sub-theta (only a couple of spikes) -> MUST still be pooled
    Csil = 30  # silent -> MUST be excluded and must NOT count in the denominator
    spikes = {
        A: _planted(rng, 200, n_samples),
        B: [123, 2500],           # 2 spikes over 4 s -> 0.5 Hz? actually 0.5? = 2/4 =0.5
        Csil: [],                 # silent
    }
    inv = make_inv(spikes, n_samples, fs_raw)
    ifr, fs_ifr = whole_culture_ifr(inv)
    K_expect = int(inv.T_rec / DT)
    if ifr.shape != (1, K_expect):
        ok = False; detail.append("shape %r != (1, %d)" % (ifr.shape, K_expect))
    if ifr.dtype != np.float32:
        ok = False; detail.append("dtype %r != float32" % ifr.dtype)

    secA = np.asarray(spikes[A], dtype=np.float64) / fs_raw
    secB = np.asarray(spikes[B], dtype=np.float64) / fs_raw
    # correct: pool {A, B}, normalize by 2 (silent C excluded)
    ref_AB, _ = _ref_ifr([secA, secB], inv.T_rec, DT, GW)
    expected = (ref_AB / 2.0).astype(np.float32).reshape(1, -1)
    if not np.allclose(ifr, expected, rtol=1e-5, atol=1e-6):
        ok = False; detail.append("whole-culture != pooled{A,B}/2")
    # prove the sub-theta electrode B was included (NOT theta-filtered):
    ref_Aonly, _ = _ref_ifr([secA], inv.T_rec, DT, GW)
    if np.allclose(ifr, (ref_Aonly / 1.0).astype(np.float32).reshape(1, -1),
                   rtol=1e-5, atol=1e-6):
        ok = False; detail.append("B wrongly excluded (theta filter leaked in)")
    # prove silent C did NOT inflate the denominator (would give /3):
    if np.allclose(ifr, (ref_AB / 3.0).astype(np.float32).reshape(1, -1),
                   rtol=1e-5, atol=1e-6):
        ok = False; detail.append("silent electrode wrongly counted in denominator")
    if ok:
        detail.append("pooled firing {A,B}, normalized by 2; silent excluded, no theta")
    return ("whole_culture_ifr pooling rule", ok, "; ".join(detail))


def check_all_silent_raises() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    inv = make_inv({1: [], 2: [], 3: []}, 2000, 1000.0)
    try:
        whole_culture_ifr(inv)
        ok = False; detail.append("all-silent culture did not raise")
    except ValueError:
        pass
    if ok:
        detail.append("all-silent culture raises ValueError")
    return ("all-silent whole culture raises", ok, "; ".join(detail))


def check_determinism() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    fs_raw = 1000.0
    n_samples = 2000
    rng = np.random.default_rng(5)
    members = (300, 301, 302, 303)
    spikes = {m: _planted(rng, 50, n_samples) for m in members}
    inv = make_inv(spikes, n_samples, fs_raw)
    sub = Subregion(center=300, members=members, mean_mfr=0.0, center_mfr=1.0)
    a, _ = subregion_ifrs(inv, [sub])
    b, _ = subregion_ifrs(inv, [sub])
    if not np.array_equal(a, b):
        ok = False; detail.append("two runs differ")
    w1, _ = whole_culture_ifr(inv)
    w2, _ = whole_culture_ifr(inv)
    if not np.array_equal(w1, w2):
        ok = False; detail.append("whole-culture two runs differ")
    if ok:
        detail.append("subregion and whole-culture IFR reproducible")
    return ("determinism", ok, "; ".join(detail))


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def main() -> int:
    checks = [
        check_subregion_ifr_correct,
        check_subregion_ifrs_stack,
        check_whole_culture_rule,
        check_all_silent_raises,
        check_determinism,
    ]
    print("=" * 74)
    print("Stage 4 smoke test -- channel_subset_extraction (IFR + whole-culture)")
    print("=" * 74)
    n_fail = 0
    for fn in checks:
        name, passed, detail = fn()
        tag = "PASS" if passed else "FAIL"
        if not passed:
            n_fail += 1
        print("[%s] %s" % (tag, name))
        if detail:
            print("       %s" % detail)
    print("-" * 74)
    if n_fail == 0:
        print("ALL STAGE-4 CHECKS PASSED")
    else:
        print("STAGE-4 FAILURES: %d" % n_fail)
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
