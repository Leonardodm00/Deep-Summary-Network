"""Smoke test -- Stage 6 (channel_subset_viz).

Scope (one concern): the two headless PNG renderers do not crash and produce
non-empty files, across the relevant diagnostics/trace shapes, and refuse to map
a whole_culture diagnostics (no partition).

This is a render smoke test (visual correctness is confirmed by eye on the PNGs),
so it asserts: files created, size > 0, correct panel count is implied by no
exception, and the whole_culture guard fires.

Run:
    cd /home/claude/work/hpc_multichannel
    python3 smoke_test_channel_subset_viz.py

Run TWICE:
    cd /home/claude/work/hpc_multichannel
    python3 smoke_test_channel_subset_viz.py; echo "exit1=$?"; \
    python3 smoke_test_channel_subset_viz.py; echo "exit2=$?"
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import List, Tuple

import numpy as np
import scipy.io as sio

from channel_subset_extraction import extract_channel_subsets
from channel_subset_viz import plot_subregion_ifrs, plot_subregion_map

W = 48


def _write_ptrain(folder: str, idx: int, spikes, n_samples: int) -> None:
    raster = np.zeros((n_samples, 1), dtype=np.uint8)
    raster[np.asarray(spikes, dtype=np.int64), 0] = 1
    sio.savemat(os.path.join(folder, "ptrain_%d.mat" % idx),
                {"ptrain": raster}, do_compression=True)


def _make_folder(folder: str, n_samples: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for r0, c0, n in ((10, 10, 60), (30, 30, 30)):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                idx = (r0 + dr) * W + (c0 + dc)
                _write_ptrain(folder, idx, np.sort(rng.integers(0, n_samples, n)), n_samples)


def _nonempty(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def check_map_and_ifrs() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    with tempfile.TemporaryDirectory() as d:
        _make_folder(d, 20000, seed=1)
        traces, fs, diag = extract_channel_subsets(
            d, mode="multichannel", n_subsets=2, electrodes_per_subset=9,
            mfr_threshold=0.1, fs_raw=1000.0, index_base=0, return_diagnostics=True)
        map_png = os.path.join(d, "map.png")
        ifr_png = os.path.join(d, "ifr.png")
        rp = plot_subregion_map(diag, map_png)
        ri = plot_subregion_ifrs(traces[0], fs, ifr_png,
                                 centers=[s.center for s in diag.subregions])
        if rp != map_png or not _nonempty(map_png):
            ok = False; detail.append("subregion map PNG missing/empty")
        if ri != ifr_png or not _nonempty(ifr_png):
            ok = False; detail.append("IFR PNG missing/empty")
    if ok:
        detail.append("map + IFR PNGs rendered non-empty")
    return ("render map + per-subregion IFRs", ok, "; ".join(detail))


def check_ifr_shapes() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    # list-of-(K,), single (K,), and (1,K) all accepted by plot_subregion_ifrs
    with tempfile.TemporaryDirectory() as d:
        traces_list = [np.abs(np.random.default_rng(k).normal(size=200)).astype(np.float32)
                       for k in range(3)]
        p1 = os.path.join(d, "list.png")
        plot_subregion_ifrs(traces_list, 50.0, p1)
        p2 = os.path.join(d, "single.png")
        plot_subregion_ifrs(traces_list[0], 50.0, p2)
        p3 = os.path.join(d, "row.png")
        plot_subregion_ifrs(traces_list[0][None, :], 50.0, p3)
        if not all(_nonempty(p) for p in (p1, p2, p3)):
            ok = False; detail.append("one of list/single/(1,K) PNGs missing")
    if ok:
        detail.append("accepts list-of-(K,), (K,), and (1,K)")
    return ("IFR renderer shape flexibility", ok, "; ".join(detail))


def check_whole_culture_guard() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    with tempfile.TemporaryDirectory() as d:
        _make_folder(d, 20000, seed=2)
        _, _, diagw = extract_channel_subsets(
            d, mode="whole_culture", fs_raw=1000.0, return_diagnostics=True)
        try:
            plot_subregion_map(diagw, os.path.join(d, "nope.png"))
            ok = False; detail.append("whole_culture map did not raise")
        except ValueError:
            pass
    if ok:
        detail.append("plot_subregion_map refuses whole_culture diagnostics")
    return ("whole_culture map guard", ok, "; ".join(detail))


def main() -> int:
    checks = [check_map_and_ifrs, check_ifr_shapes, check_whole_culture_guard]
    print("=" * 74)
    print("Stage 6 smoke test -- channel_subset_viz (headless render)")
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
    print("ALL STAGE-6 CHECKS PASSED" if n_fail == 0 else "STAGE-6 FAILURES: %d" % n_fail)
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
