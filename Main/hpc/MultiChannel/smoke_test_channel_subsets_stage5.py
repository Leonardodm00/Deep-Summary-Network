"""Smoke test -- Stage 5 (public extract_channel_subsets entry point).

Scope (one concern): the public composed API over REAL ptrain_<idx>.mat files
written to a temp folder, exercising load -> geometry/MFR -> partition -> IFR.

Controls:
  - uniform sample-count contract: len(traces) == 1 (multichannel), C
    (per_region_single), 1 (whole_culture);
  - per-mode shapes: multichannel [(C,K)]; per_region_single C x (K,) 1-D;
    whole_culture [(K,)] 1-D; fs_ifr == 1/Dt;
  - cross-mode agreement: np.stack(per_region_single) == multichannel[0];
  - guards: bad mode -> ValueError; an out-of-grid index -> GeometryError
    (extract calls validate_grid);
  - diagnostics populated for partition modes, empty for whole_culture.

Run:
    cd /home/claude/work/hpc_multichannel
    python3 smoke_test_channel_subsets_stage5.py

Run TWICE:
    cd /home/claude/work/hpc_multichannel
    python3 smoke_test_channel_subsets_stage5.py; echo "exit1=$?"; \
    python3 smoke_test_channel_subsets_stage5.py; echo "exit2=$?"
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import scipy.io as sio

from channel_subset_extraction import (
    DEFAULT_W_SIZE,
    ExtractionDiagnostics,
    GeometryError,
    extract_channel_subsets,
)

W = 48
DT = DEFAULT_W_SIZE


def _write_ptrain(folder: str, idx: int, spike_samples, n_samples: int) -> None:
    raster = np.zeros((n_samples, 1), dtype=np.uint8)
    raster[np.asarray(spike_samples, dtype=np.int64), 0] = 1
    sio.savemat(os.path.join(folder, "ptrain_%d.mat" % idx),
                {"ptrain": raster}, do_compression=True)


def _block(r0: int, c0: int, half: int) -> List[int]:
    return [(r0 + dr) * W + (c0 + dc)
            for dr in range(-half, half + 1)
            for dc in range(-half, half + 1)]


def _make_folder(folder: str, n_samples: int, seed: int,
                 extra_idx=None) -> None:
    """Two 3x3 clusters (hot + cool), all electrodes firing above theta."""
    rng = np.random.default_rng(seed)
    for i in _block(10, 10, 1):
        _write_ptrain(folder, i, np.sort(rng.integers(0, n_samples, 60)), n_samples)
    for i in _block(30, 30, 1):
        _write_ptrain(folder, i, np.sort(rng.integers(0, n_samples, 30)), n_samples)
    if extra_idx is not None:
        _write_ptrain(folder, extra_idx, np.sort(rng.integers(0, n_samples, 10)), n_samples)


def check_modes_and_contract() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    fs_raw = 1000.0
    n_samples = 20000                         # T_rec = 20 s -> K = 1000
    K = int((n_samples / fs_raw) / DT)
    with tempfile.TemporaryDirectory() as d:
        _make_folder(d, n_samples, seed=11)
        common = dict(n_subsets=2, electrodes_per_subset=9, mfr_threshold=0.1,
                      fs_raw=fs_raw, index_base=0)

        mc, fs1 = extract_channel_subsets(d, mode="multichannel", **common)
        pr, fs2 = extract_channel_subsets(d, mode="per_region_single", **common)
        wc, fs3 = extract_channel_subsets(d, mode="whole_culture",
                                          fs_raw=fs_raw, index_base=0)

        # sample-count contract
        if len(mc) != 1:
            ok = False; detail.append("multichannel len %d != 1" % len(mc))
        if len(pr) != 2:
            ok = False; detail.append("per_region_single len %d != 2 (=C)" % len(pr))
        if len(wc) != 1:
            ok = False; detail.append("whole_culture len %d != 1" % len(wc))

        # shapes
        if mc and mc[0].shape != (2, K):
            ok = False; detail.append("multichannel shape %r != (2,%d)" % (mc[0].shape, K))
        if any(t.ndim != 1 or t.shape != (K,) for t in pr):
            ok = False; detail.append("per_region_single traces not (K,) 1-D")
        if wc and (wc[0].ndim != 1 or wc[0].shape != (K,)):
            ok = False; detail.append("whole_culture shape %r != (%d,)" % (wc[0].shape, K))
        for nm, fs in (("mc", fs1), ("pr", fs2), ("wc", fs3)):
            if not np.isclose(fs, 1.0 / DT):
                ok = False; detail.append("%s fs_ifr %r != 50" % (nm, fs))

        # cross-mode agreement: stacked per_region_single == multichannel (C,K)
        if ok and not np.array_equal(np.stack(pr, axis=0).astype(np.float32), mc[0]):
            ok = False; detail.append("stack(per_region_single) != multichannel")
    if ok:
        detail.append("3 modes; samples/recording 1 / C / 1; shapes + agreement OK")
    return ("all three modes + uniform contract", ok, "; ".join(detail))


def check_diagnostics() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    n_samples = 20000
    with tempfile.TemporaryDirectory() as d:
        _make_folder(d, n_samples, seed=22)
        traces, fs, diag = extract_channel_subsets(
            d, mode="multichannel", n_subsets=2, electrodes_per_subset=9,
            mfr_threshold=0.1, fs_raw=1000.0, index_base=0, return_diagnostics=True)
        if not isinstance(diag, ExtractionDiagnostics):
            ok = False; detail.append("no ExtractionDiagnostics returned")
        elif len(diag.subregions) != 2 or diag.n_present != 18 or len(diag.coords) != 18:
            ok = False; detail.append("diag fields wrong: subs=%d present=%d coords=%d"
                                      % (len(diag.subregions), diag.n_present, len(diag.coords)))
        # whole_culture diagnostics are empty on geometry fields
        _, _, diagw = extract_channel_subsets(
            d, mode="whole_culture", fs_raw=1000.0, return_diagnostics=True)
        if diagw.subregions != () or diagw.coords != {}:
            ok = False; detail.append("whole_culture diag not empty on geometry fields")
    if ok:
        detail.append("diagnostics populated for partition mode, empty for whole_culture")
    return ("ExtractionDiagnostics", ok, "; ".join(detail))


def check_guards() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    n_samples = 20000
    with tempfile.TemporaryDirectory() as d:
        _make_folder(d, n_samples, seed=33)
        # bad mode
        try:
            extract_channel_subsets(d, mode="bogus", fs_raw=1000.0)
            ok = False; detail.append("bad mode did not raise")
        except ValueError:
            pass
    # out-of-grid index -> GeometryError (extract calls validate_grid)
    with tempfile.TemporaryDirectory() as d2:
        _make_folder(d2, n_samples, seed=34, extra_idx=5000)  # 5000 > 2303 -> overflow
        try:
            extract_channel_subsets(d2, mode="multichannel", n_subsets=2,
                                    electrodes_per_subset=9, fs_raw=1000.0, index_base=0)
            ok = False; detail.append("out-of-grid index did not raise GeometryError")
        except GeometryError:
            pass
    if ok:
        detail.append("bad mode -> ValueError; out-of-grid index -> GeometryError")
    return ("guards (mode + wrong-base)", ok, "; ".join(detail))


def main() -> int:
    checks = [check_modes_and_contract, check_diagnostics, check_guards]
    print("=" * 74)
    print("Stage 5 smoke test -- channel_subset_extraction (public extract API)")
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
    print("ALL STAGE-5 CHECKS PASSED" if n_fail == 0 else "STAGE-5 FAILURES: %d" % n_fail)
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
