"""Smoke test -- Stage 1 (I/O + inventory) of channel_subset_extraction.

Scope (one concern): the ptrain loader stack
    parse_ptrain_index / load_ptrain_file / load_ptrain_folder -> PtrainInventory.

Strategy (handoff Section 7): only ONE real .mat exists (ptrain_100.mat), so the
LOADER is validated against a SYNTHETIC ptrain folder with PLANTED spikes whose
answers are known exactly, and additionally against the real ptrain_100.mat IF it
is present. The real-file check auto-SKIPS (does not fail) when the file is absent,
so the synthetic suite still gives a clean PASS/FAIL and the real check can be run
later by dropping ptrain_100.mat into one of REAL_FILE_CANDIDATES and rerunning.

Run:
    cd /home/claude/work/hpc_multichannel
    python3 smoke_test_channel_subsets_stage1.py

Copy-paste snippet to run it TWICE and see both exit codes:
    cd /home/claude/work/hpc_multichannel
    python3 smoke_test_channel_subsets_stage1.py; echo "exit1=$?"; \
    python3 smoke_test_channel_subsets_stage1.py; echo "exit2=$?"

Exit code 0 = all executed checks passed (skips are not failures); 1 = a failure.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import scipy.io as sio

from channel_subset_extraction import (
    DEFAULT_FS_RAW,
    PTRAIN_VARNAME,
    PtrainInventory,
    PtrainLoadError,
    load_ptrain_file,
    load_ptrain_folder,
    parse_ptrain_index,
)

# Candidate locations to look for the real sample file (first hit wins).
REAL_FILE_CANDIDATES: List[str] = [
    "/mnt/user-data/uploads/ptrain_100.mat",
    os.path.join(os.getcwd(), "ptrain_100.mat"),
    "/home/claude/work/hpc_multichannel/ptrain_100.mat",
]

# Empirically verified value for the real ptrain_100.mat (handoff Section 2).
REAL_N_SAMPLES_EXPECTED = 12_132_108


# --------------------------------------------------------------------------- #
# tiny synthetic-data helpers (separation of concerns: data build vs asserts)
# --------------------------------------------------------------------------- #

def _make_raster(n: int, spike_idx: np.ndarray) -> np.ndarray:
    """Build a (n, 1) uint8 binary raster with ones at spike_idx."""
    r = np.zeros((n, 1), dtype=np.uint8)
    r[np.asarray(spike_idx, dtype=np.int64), 0] = 1
    return r


def _write_ptrain(folder: str, k: int, n: int, spike_idx: np.ndarray) -> None:
    """Write ptrain_<k>.mat (MATLAB v5, compressed) into folder."""
    path = os.path.join(folder, "ptrain_%d.mat" % k)
    sio.savemat(path, {PTRAIN_VARNAME: _make_raster(n, spike_idx)},
                do_compression=True, format="5")


def _write_folder(folder: str, plants: Dict[int, Tuple[int, np.ndarray]]) -> None:
    """Write a whole folder from {k: (n, spike_idx)}."""
    os.makedirs(folder, exist_ok=True)
    for k, (n, idx) in plants.items():
        _write_ptrain(folder, k, n, idx)


# --------------------------------------------------------------------------- #
# individual checks -- each returns (name, passed, detail)
# --------------------------------------------------------------------------- #

def check_parse() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    # valid
    for name, want in [("ptrain_10.mat", 10),
                       ("ptrain_1016.mat", 1016),
                       ("ptrain_0.mat", 0),
                       ("/a/b/ptrain_2303.mat", 2303)]:
        got = parse_ptrain_index(name)
        if got != want:
            ok = False
            detail.append("parse(%r)=%d want %d" % (name, got, want))
    # invalid -> must raise
    for bad in ["ptrain.mat", "ptrain_x.mat", "foo_10.mat",
                "ptrain_10.txt", "ptrain_10.mat.bak", "ptrain_-1.mat"]:
        try:
            parse_ptrain_index(bad)
            ok = False
            detail.append("parse(%r) did not raise" % bad)
        except PtrainLoadError:
            pass
    return ("parse_ptrain_index (valid + reject)", ok, "; ".join(detail))


def check_single_file() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    n = 200_000
    planted = np.array([0, 5, 5000, 12345, n - 1], dtype=np.int64)
    with tempfile.TemporaryDirectory() as tmp:
        _write_ptrain(tmp, 7, n, planted)
        idx, n_out = load_ptrain_file(os.path.join(tmp, "ptrain_7.mat"))
        if n_out != n:
            ok = False
            detail.append("n_samples %d != %d" % (n_out, n))
        if idx.dtype != np.int64:
            ok = False
            detail.append("dtype %s != int64" % idx.dtype)
        if not np.array_equal(idx, np.sort(planted)):
            ok = False
            detail.append("indices %r != planted %r" % (idx.tolist(), planted.tolist()))
        if not np.all(np.diff(idx) > 0):
            ok = False
            detail.append("indices not strictly ascending")
    return ("load_ptrain_file (exact nonzero, dtype, order)", ok, "; ".join(detail))


def check_silent_electrode() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    n = 50_000
    with tempfile.TemporaryDirectory() as tmp:
        _write_ptrain(tmp, 3, n, np.array([], dtype=np.int64))
        idx, n_out = load_ptrain_file(os.path.join(tmp, "ptrain_3.mat"))
        if idx.size != 0:
            ok = False
            detail.append("silent electrode returned %d spikes" % idx.size)
        if n_out != n:
            ok = False
            detail.append("silent electrode n_samples %d != %d" % (n_out, n))
    return ("load_ptrain_file (silent electrode -> empty)", ok, "; ".join(detail))


def check_folder_inventory() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    n = 300_000
    fs = 1000.0
    plants = {
        10:  (n, np.array([1, 2, 3, 100, 250_000], dtype=np.int64)),   # 5 spikes
        100: (n, np.array([0, 50_000, 299_999], dtype=np.int64)),      # 3 spikes
        250: (n, np.arange(0, 10_000, dtype=np.int64)),                # 10000 spikes
    }
    with tempfile.TemporaryDirectory() as tmp:
        _write_folder(tmp, plants)
        # add a decoy non-ptrain file that must be ignored
        with open(os.path.join(tmp, "README.txt"), "w") as fh:
            fh.write("ignore me")
        inv = load_ptrain_folder(tmp, fs_raw=fs, index_base=0)

        if set(inv.spikes.keys()) != set(plants.keys()):
            ok = False
            detail.append("keys %r != %r" % (sorted(inv.spikes), sorted(plants)))
        if inv.indices != sorted(plants.keys()):
            ok = False
            detail.append("indices order %r not ascending" % inv.indices)
        if inv.n_samples != n:
            ok = False
            detail.append("n_samples %d != %d" % (inv.n_samples, n))
        if abs(inv.T_rec - n / fs) > 0:
            ok = False
            detail.append("T_rec %r != %r" % (inv.T_rec, n / fs))
        if inv.fs_raw != fs or inv.index_base != 0:
            ok = False
            detail.append("fs_raw/index_base not carried")
        for k, (_, planted) in plants.items():
            if not np.array_equal(inv.spikes[k], np.sort(planted)):
                ok = False
                detail.append("elec %d indices mismatch" % k)
    return ("load_ptrain_folder (keys/order/n/T_rec/spikes, decoy ignored)",
            ok, "; ".join(detail))


def check_mfr_sanity() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    n = 1_000_000
    fs = 500.0                       # -> T_rec = 2000 s
    plants = {
        1: (n, np.arange(0, 2000, dtype=np.int64)),   # 2000 spikes -> 1.0 Hz
        2: (n, np.arange(0, 500, dtype=np.int64)),    # 500 spikes  -> 0.25 Hz
    }
    with tempfile.TemporaryDirectory() as tmp:
        _write_folder(tmp, plants)
        inv = load_ptrain_folder(tmp, fs_raw=fs, index_base=0)
        for k, want_hz in [(1, 1.0), (2, 0.25)]:
            mfr = inv.n_spikes(k) / inv.T_rec
            if abs(mfr - want_hz) > 1e-12:
                ok = False
                detail.append("MFR[%d]=%r want %r" % (k, mfr, want_hz))
    return ("MFR sanity (n_spikes / T_rec)", ok, "; ".join(detail))


def check_length_mismatch() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    with tempfile.TemporaryDirectory() as tmp:
        _write_ptrain(tmp, 1, 100_000, np.array([1, 2], dtype=np.int64))
        _write_ptrain(tmp, 2, 100_001, np.array([3, 4], dtype=np.int64))  # different n
        try:
            load_ptrain_folder(tmp, fs_raw=1000.0)
            ok = False
            detail.append("did not raise on n_samples mismatch")
        except PtrainLoadError as exc:
            if "mismatch" not in str(exc).lower():
                ok = False
                detail.append("raised but message lacks 'mismatch': %s" % exc)
    return ("load_ptrain_folder (n_samples mismatch -> raise)", ok, "; ".join(detail))


def check_non_binary_guard() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    n = 10_000
    with tempfile.TemporaryDirectory() as tmp:
        raster = np.zeros((n, 1), dtype=np.uint8)
        raster[5, 0] = 2                      # illegal value
        sio.savemat(os.path.join(tmp, "ptrain_9.mat"),
                    {PTRAIN_VARNAME: raster}, do_compression=True, format="5")
        try:
            load_ptrain_file(os.path.join(tmp, "ptrain_9.mat"))
            ok = False
            detail.append("did not raise on non-binary raster")
        except PtrainLoadError as exc:
            if "binary" not in str(exc).lower():
                ok = False
                detail.append("raised but message lacks 'binary': %s" % exc)
    return ("load_ptrain_file (non-binary -> raise)", ok, "; ".join(detail))


def check_empty_folder_guard() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "notes.txt"), "w") as fh:
            fh.write("no ptrain here")
        try:
            load_ptrain_folder(tmp, fs_raw=1000.0)
            ok = False
            detail.append("did not raise on folder with no ptrain files")
        except PtrainLoadError:
            pass
    # missing directory
    try:
        load_ptrain_folder("/definitely/not/a/real/dir", fs_raw=1000.0)
        ok = False
        detail.append("did not raise on missing directory")
    except PtrainLoadError:
        pass
    # bad fs_raw / index_base
    with tempfile.TemporaryDirectory() as tmp:
        _write_ptrain(tmp, 1, 100, np.array([1], dtype=np.int64))
        for bad_kwargs in ({"fs_raw": 0.0}, {"fs_raw": -1.0}, {"index_base": 2}):
            try:
                load_ptrain_folder(tmp, **bad_kwargs)
                ok = False
                detail.append("did not raise on %r" % bad_kwargs)
            except ValueError:
                pass
    return ("guards (no-ptrain / missing dir / bad fs_raw / bad base)", ok, "; ".join(detail))


def check_determinism() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    n = 200_000
    plants = {
        5:  (n, np.array([10, 20, 30], dtype=np.int64)),
        50: (n, np.array([1, 2, 3, 4], dtype=np.int64)),
        500: (n, np.array([100, 200], dtype=np.int64)),
    }
    with tempfile.TemporaryDirectory() as tmp:
        _write_folder(tmp, plants)
        inv_a = load_ptrain_folder(tmp, fs_raw=1234.5)
        inv_b = load_ptrain_folder(tmp, fs_raw=1234.5)
        if inv_a.indices != inv_b.indices:
            ok = False
            detail.append("key order differs across loads")
        for k in inv_a.spikes:
            if not np.array_equal(inv_a.spikes[k], inv_b.spikes[k]):
                ok = False
                detail.append("elec %d differs across loads" % k)
    return ("determinism (two loads identical)", ok, "; ".join(detail))


def check_real_file() -> Tuple[str, bool, str]:
    """Validate the real ptrain_100.mat if present; SKIP (pass) if absent."""
    path = next((p for p in REAL_FILE_CANDIDATES if os.path.isfile(p)), None)
    if path is None:
        return ("real ptrain_100.mat", True,
                "SKIPPED -- file not found in %s. Drop ptrain_100.mat into "
                "/mnt/user-data/uploads/ and rerun to validate the real loader."
                % REAL_FILE_CANDIDATES)
    detail = []
    ok = True
    idx, n_out = load_ptrain_file(path)
    if n_out != REAL_N_SAMPLES_EXPECTED:
        ok = False
        detail.append("n_samples %d != expected %d" % (n_out, REAL_N_SAMPLES_EXPECTED))
    if idx.dtype != np.int64:
        ok = False
        detail.append("dtype %s != int64" % idx.dtype)
    if idx.size == 0:
        ok = False
        detail.append("zero spikes (unexpected for the sample)")
    if idx.size and (idx.min() < 0 or idx.max() >= n_out):
        ok = False
        detail.append("spike index out of [0, n)")
    t_rec = n_out / DEFAULT_FS_RAW
    mfr = idx.size / t_rec if t_rec > 0 else float("nan")
    detail.append("n=%d spikes=%d T_rec=%.3fs MFR=%.4fHz (fs_raw=%.2f)"
                  % (n_out, idx.size, t_rec, mfr, DEFAULT_FS_RAW))
    return ("real ptrain_100.mat", ok, "; ".join(detail))


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def main() -> int:
    checks = [
        check_parse,
        check_single_file,
        check_silent_electrode,
        check_folder_inventory,
        check_mfr_sanity,
        check_length_mismatch,
        check_non_binary_guard,
        check_empty_folder_guard,
        check_determinism,
        check_real_file,
    ]
    print("=" * 74)
    print("Stage 1 smoke test -- channel_subset_extraction (I/O + inventory)")
    print("=" * 74)
    n_fail = 0
    for fn in checks:
        name, passed, detail = fn()
        skipped = detail.startswith("SKIPPED")
        tag = "SKIP" if skipped else ("PASS" if passed else "FAIL")
        if (not passed) and (not skipped):
            n_fail += 1
        print("[%s] %s" % (tag, name))
        if detail:
            print("       %s" % detail)
    print("-" * 74)
    if n_fail == 0:
        print("ALL STAGE-1 CHECKS PASSED (skips are not failures)")
    else:
        print("STAGE-1 FAILURES: %d" % n_fail)
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
