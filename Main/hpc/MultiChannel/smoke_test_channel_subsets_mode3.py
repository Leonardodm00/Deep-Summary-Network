"""Smoke test -- Mode 3 (per-subregion SINGLE-CHANNEL traces).

Scope (one concern): subregion_single_channel_traces -- the same subregion
partition as the multichannel Mode 1, but each subregion returned as its own
independent single-channel (K,) trace (C samples per recording for the
in_channels=1 backbone), normalized per electrode.

Controls:
  - end-to-end: build a planted inventory -> mean_firing_rates -> partition ->
    extract, so the whole Stage-3 -> Stage-4 -> Mode-3 chain is exercised;
  - output contract: a list of C traces, each (K,) 1-D float32, non-negative,
    fs_ifr = 1/Dt, in subregion (hottest-first) order;
  - Mode 1 <-> Mode 3 agreement: np.stack(traces) equals subregion_ifrs(...)
    BIT-FOR-BIT, and traces[c] equals subregion_ifr(subs[c]) -- the two modes
    differ only in packaging;
  - empty guard; determinism.

Run:
    cd /home/claude/work/hpc_multichannel
    python3 smoke_test_channel_subsets_mode3.py

Run TWICE with exit codes:
    cd /home/claude/work/hpc_multichannel
    python3 smoke_test_channel_subsets_mode3.py; echo "exit1=$?"; \
    python3 smoke_test_channel_subsets_mode3.py; echo "exit2=$?"
"""

from __future__ import annotations

import sys
from typing import Dict, List, Tuple

import numpy as np

from channel_subset_extraction import (
    DEFAULT_W_SIZE,
    PtrainInventory,
    mean_firing_rates,
    partition_subregions,
    subregion_ifr,
    subregion_ifrs,
    subregion_single_channel_traces,
)

W = 48
DT = DEFAULT_W_SIZE


def idx_of(row: int, col: int) -> int:
    return row * W + col


def block(r0: int, c0: int, half: int) -> List[int]:
    return [idx_of(r0 + dr, c0 + dc)
            for dr in range(-half, half + 1)
            for dc in range(-half, half + 1)]


def build_planted_inv(fs_raw: float = 1000.0, n_samples: int = 20000,
                      seed: int = 1234) -> Tuple[PtrainInventory, Dict[int, Tuple[int, int]]]:
    """Two 3x3 clusters, every electrode firing well above theta."""
    rng = np.random.default_rng(seed)
    cluster1 = block(10, 10, 1)      # 9 electrodes
    cluster2 = block(30, 30, 1)      # 9 electrodes
    present = cluster1 + cluster2
    spikes: Dict[int, np.ndarray] = {}
    # cluster1 hotter than cluster2 so partition order is deterministic
    for i in cluster1:
        spikes[i] = np.sort(rng.integers(0, n_samples, size=60)).astype(np.int64)
    for i in cluster2:
        spikes[i] = np.sort(rng.integers(0, n_samples, size=30)).astype(np.int64)
    inv = PtrainInventory(spikes=spikes, n_samples=n_samples, fs_raw=fs_raw,
                          T_rec=n_samples / fs_raw, index_base=0)
    coords = {int(i): (int(i) // W, int(i) % W) for i in present}
    return inv, coords


def check_end_to_end_and_contract() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    inv, coords = build_planted_inv()
    mfrs = mean_firing_rates(inv, inv.T_rec)
    subs, _ = partition_subregions(coords, mfrs, n_subsets=2,
                                   electrodes_per_subset=9, mfr_threshold=0.1)
    traces, fs_ifr = subregion_single_channel_traces(inv, subs)
    K_expect = int(inv.T_rec / DT)

    if len(traces) != len(subs):
        ok = False; detail.append("got %d traces, want %d" % (len(traces), len(subs)))
    for c, tr in enumerate(traces):
        if tr.ndim != 1 or tr.shape != (K_expect,):
            ok = False; detail.append("trace %d shape %r != (%d,)" % (c, tr.shape, K_expect)); break
        if tr.dtype != np.float32:
            ok = False; detail.append("trace %d dtype %r != float32" % (c, tr.dtype)); break
        if np.any(tr < 0):
            ok = False; detail.append("trace %d has negative values" % c); break
    if not np.isclose(fs_ifr, 1.0 / DT):
        ok = False; detail.append("fs_ifr %r != %r" % (fs_ifr, 1.0 / DT))
    if ok:
        detail.append("C=%d single-channel (K=%d,) float32 traces, fs=%.1f, hottest-first"
                      % (len(traces), K_expect, fs_ifr))
    return ("Mode 3 end-to-end + single-channel contract", ok, "; ".join(detail))


def check_agrees_with_mode1() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    inv, coords = build_planted_inv(seed=99)
    mfrs = mean_firing_rates(inv, inv.T_rec)
    subs, _ = partition_subregions(coords, mfrs, n_subsets=2,
                                   electrodes_per_subset=9, mfr_threshold=0.1)
    traces, _ = subregion_single_channel_traces(inv, subs)
    mc, _ = subregion_ifrs(inv, subs)

    # stacked Mode 3 == Mode 1, bit-for-bit
    if not np.array_equal(np.stack(traces, axis=0).astype(np.float32), mc):
        ok = False; detail.append("stack(Mode3) != Mode1 (C,K)")
    # each trace == the single-subregion call
    for c, s in enumerate(subs):
        one, _ = subregion_ifr(inv, s)
        if not np.array_equal(traces[c], one):
            ok = False; detail.append("trace %d != subregion_ifr(subs[%d])" % (c, c)); break
        if not np.array_equal(traces[c], mc[c]):
            ok = False; detail.append("trace %d != Mode1 row %d" % (c, c)); break
    if ok:
        detail.append("stack(Mode3) == Mode1 and traces[c] == row c, bit-for-bit")
    return ("Mode 3 <-> Mode 1 agreement", ok, "; ".join(detail))


def check_guard_and_determinism() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    inv, coords = build_planted_inv(seed=7)
    mfrs = mean_firing_rates(inv, inv.T_rec)
    subs, _ = partition_subregions(coords, mfrs, n_subsets=2,
                                   electrodes_per_subset=9, mfr_threshold=0.1)
    # empty guard
    try:
        subregion_single_channel_traces(inv, [])
        ok = False; detail.append("empty subregions did not raise")
    except ValueError:
        pass
    # determinism
    a, _ = subregion_single_channel_traces(inv, subs)
    b, _ = subregion_single_channel_traces(inv, subs)
    if len(a) != len(b) or any(not np.array_equal(x, y) for x, y in zip(a, b)):
        ok = False; detail.append("two runs differ")
    if ok:
        detail.append("empty guard raises; repeated extraction reproducible")
    return ("Mode 3 guard + determinism", ok, "; ".join(detail))


def main() -> int:
    checks = [
        check_end_to_end_and_contract,
        check_agrees_with_mode1,
        check_guard_and_determinism,
    ]
    print("=" * 74)
    print("Mode 3 smoke test -- channel_subset_extraction (per-subregion single-channel)")
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
        print("ALL MODE-3 CHECKS PASSED")
    else:
        print("MODE-3 FAILURES: %d" % n_fail)
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
