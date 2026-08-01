"""Smoke test -- Stage 2 (geometry + MFR) of channel_subset_extraction.

Scope (one concern): the geometry/MFR helpers
    electrode_coords / validate_grid / mean_firing_rates / nearest_valid.

Controls used:
  - coord round-trip over the whole 48x48 grid for base in {0, 1};
  - nearest_valid checked against an INDEPENDENT brute-force KNN reference
    (pure-python sort by (distance^2, index)) on many random candidate subsets
    and centres, plus the explicit 8 Moore-neighbour case and the
    reaches-further case when neighbours are invalid;
  - tie determinism (equal-distance -> lower index first);
  - MFR exactness against a hand-built PtrainInventory;
  - the (row, col) range check catches a wrong index_base.

Run:
    cd /home/claude/work/hpc_multichannel
    python3 smoke_test_channel_subsets_stage2.py

Run TWICE with exit codes:
    cd /home/claude/work/hpc_multichannel
    python3 smoke_test_channel_subsets_stage2.py; echo "exit1=$?"; \
    python3 smoke_test_channel_subsets_stage2.py; echo "exit2=$?"
"""

from __future__ import annotations

import sys
from typing import List, Tuple

import numpy as np

from channel_subset_extraction import (
    GRID_WIDTH,
    N_ELECTRODES,
    PITCH_UM,
    GeometryError,
    PtrainInventory,
    electrode_coords,
    mean_firing_rates,
    nearest_valid,
    validate_grid,
)


# --------------------------------------------------------------------------- #
# independent references (deliberately NOT reusing module internals)
# --------------------------------------------------------------------------- #

def _rc_ref(idx: int, width: int, base: int) -> Tuple[int, int]:
    """Independent (row, col) mapping for cross-checking electrode_coords."""
    i0 = idx - base
    return (i0 // width, i0 % width)


def _bruteforce_knn(centre: int, candidates, k: int,
                    width: int, base: int) -> List[int]:
    """Reference KNN: sort by (distance^2, index), drop the centre, take k."""
    r0, c0 = _rc_ref(centre, width, base)
    scored = []
    for c in candidates:
        if c == centre:
            continue
        r, cc = _rc_ref(c, width, base)
        d2 = (r - r0) ** 2 + (cc - c0) ** 2
        scored.append((d2, c))
    scored.sort()  # tuples sort by d2 then by index -> matches the module rule
    return [c for _, c in scored[:k]]


def _moore(centre: int, width: int, base: int) -> set:
    """The <= 8 Moore-neighbour linear indices of centre inside the grid."""
    r0, c0 = _rc_ref(centre, width, base)
    out = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            r, c = r0 + dr, c0 + dc
            if 0 <= r < width and 0 <= c < width:
                out.add((r * width + c) + base)
    return out


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #

def check_coord_roundtrip() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    for base in (0, 1):
        for idx in range(base, base + N_ELECTRODES):
            ec = electrode_coords(idx, width=GRID_WIDTH, base=base)
            r_ref, c_ref = _rc_ref(idx, GRID_WIDTH, base)
            if (ec.row, ec.col) != (r_ref, c_ref):
                ok = False
                detail.append("idx %d base %d -> (%d,%d) ref (%d,%d)"
                              % (idx, base, ec.row, ec.col, r_ref, c_ref))
                break
            # physical coords
            if ec.x_um != ec.col * PITCH_UM or ec.y_um != ec.row * PITCH_UM:
                ok = False
                detail.append("idx %d phys mismatch" % idx)
                break
            # reconstruct the linear index
            recon = (ec.row * GRID_WIDTH + ec.col) + base
            if recon != idx:
                ok = False
                detail.append("idx %d recon %d" % (idx, recon))
                break
        if not ok:
            break
    if ok:
        detail.append("round-trip exact over full %dx%d grid for base 0 and 1"
                      % (GRID_WIDTH, GRID_WIDTH))
    return ("electrode_coords round-trip + physical coords", ok, "; ".join(detail))


def check_moore_all_valid() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    base = 0
    centre = 10 * GRID_WIDTH + 10        # interior electrode (10, 10)
    candidates = list(range(N_ELECTRODES))
    got = nearest_valid(centre, candidates, 8, width=GRID_WIDTH, base=base)
    want_set = _moore(centre, GRID_WIDTH, base)
    if set(got) != want_set:
        ok = False
        detail.append("neighbour set %r != Moore %r" % (sorted(got), sorted(want_set)))
    # first 4 must be the orthogonal ring (d2=1), next 4 the diagonal ring (d2=2)
    r0, c0 = _rc_ref(centre, GRID_WIDTH, base)
    d2s = [((_rc_ref(g, GRID_WIDTH, base)[0] - r0) ** 2
            + (_rc_ref(g, GRID_WIDTH, base)[1] - c0) ** 2) for g in got]
    if d2s != [1, 1, 1, 1, 2, 2, 2, 2]:
        ok = False
        detail.append("distance ordering %r != [1,1,1,1,2,2,2,2]" % d2s)
    # and each block sorted ascending by index
    if got[:4] != sorted(got[:4]) or got[4:] != sorted(got[4:]):
        ok = False
        detail.append("within-ring order not ascending by index")
    if ok:
        detail.append("8 Moore neighbours exact, ortho ring then diag ring")
    return ("nearest_valid == 8 Moore neighbours (all valid)", ok, "; ".join(detail))


def check_reaches_further() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    base = 0
    centre = 10 * GRID_WIDTH + 10
    r0, c0 = _rc_ref(centre, GRID_WIDTH, base)
    # invalidate the 4 orthogonal neighbours
    ortho = {((r0 + dr) * GRID_WIDTH + (c0 + dc)) + base
             for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]}
    candidates = [i for i in range(N_ELECTRODES) if i not in ortho]
    got = nearest_valid(centre, candidates, 8, width=GRID_WIDTH, base=base)
    ref = _bruteforce_knn(centre, candidates, 8, GRID_WIDTH, base)
    if got != ref:
        ok = False
        detail.append("got %r != brute-force %r" % (got, ref))
    if ortho & set(got):
        ok = False
        detail.append("returned an invalidated orthogonal neighbour")
    # must now reach a d2=4 axis-2 neighbour (e.g. (r0+2, c0))
    axis2 = ((r0 + 2) * GRID_WIDTH + c0) + base
    if axis2 not in set(got):
        ok = False
        detail.append("did not reach the expected d2=4 neighbour")
    if ok:
        detail.append("skipped invalid ring and reached further, matches brute force")
    return ("nearest_valid reaches further past invalid electrodes", ok, "; ".join(detail))


def check_knn_vs_bruteforce() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    rng = np.random.default_rng(20260717)
    all_idx = np.arange(N_ELECTRODES)
    n_trials = 400
    for _ in range(n_trials):
        centre = int(rng.integers(0, N_ELECTRODES))  # includes border electrodes
        m = int(rng.integers(5, 200))
        cand = rng.choice(all_idx, size=m, replace=False).tolist()
        k = int(rng.integers(1, 12))
        got = nearest_valid(centre, cand, k, width=GRID_WIDTH, base=0)
        ref = _bruteforce_knn(centre, cand, k, GRID_WIDTH, 0)
        if got != ref:
            ok = False
            detail.append("mismatch centre=%d k=%d: got %r ref %r"
                          % (centre, k, got, ref))
            break
    if ok:
        detail.append("%d random trials all match brute-force KNN" % n_trials)
    return ("nearest_valid == brute-force KNN (random trials)", ok, "; ".join(detail))


def check_tie_determinism() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    base = 0
    centre = 10 * GRID_WIDTH + 10
    left = 10 * GRID_WIDTH + 9        # (10,9) d2=1, lower index
    right = 10 * GRID_WIDTH + 11      # (10,11) d2=1, higher index
    got = nearest_valid(centre, [right, left], 1, width=GRID_WIDTH, base=base)
    if got != [left]:
        ok = False
        detail.append("tie k=1 -> %r, expected [%d] (lower index)" % (got, left))
    # a whole equidistant set (the 4 orthogonal) must come back ascending by index
    ortho = [((10 + dr) * GRID_WIDTH + (10 + dc)) + base
             for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
    got4 = nearest_valid(centre, list(reversed(ortho)), 4, width=GRID_WIDTH, base=base)
    if got4 != sorted(ortho):
        ok = False
        detail.append("equidistant set order %r != ascending %r" % (got4, sorted(ortho)))
    if ok:
        detail.append("equal-distance ties resolved by lower index, deterministic")
    return ("nearest_valid tie determinism", ok, "; ".join(detail))


def check_mfr_exact() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    T_rec = 1200.0
    inv = PtrainInventory(
        spikes={
            5:  np.arange(1200, dtype=np.int64),   # 1200 spikes -> 1.0 Hz
            10: np.arange(120, dtype=np.int64),    # 120 spikes  -> 0.1 Hz
            20: np.array([], dtype=np.int64),      # silent      -> 0.0 Hz
            3:  np.arange(60, dtype=np.int64),     # 60 spikes   -> 0.05 Hz
        },
        n_samples=12_000_000,
        fs_raw=10_000.0,
        T_rec=T_rec,
        index_base=0,
    )
    mfr = mean_firing_rates(inv, T_rec)
    if list(mfr.keys()) != sorted(inv.spikes.keys()):
        ok = False
        detail.append("keys not ascending: %r" % list(mfr.keys()))
    want = {5: 1.0, 10: 0.1, 20: 0.0, 3: 0.05}
    for k, w in want.items():
        if abs(mfr[k] - w) > 1e-12:
            ok = False
            detail.append("MFR[%d]=%r want %r" % (k, mfr[k], w))
    # bad T_rec must raise
    try:
        mean_firing_rates(inv, 0.0)
        ok = False
        detail.append("T_rec=0 did not raise")
    except ValueError:
        pass
    if ok:
        detail.append("MFR = N_spikes(e)/T_rec exact incl. silent electrode")
    return ("mean_firing_rates exactness", ok, "; ".join(detail))


def check_range_check_wrong_base() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    # A full 0-based grid is fine under base=0.
    try:
        validate_grid(range(N_ELECTRODES), width=GRID_WIDTH, base=0)
    except GeometryError:
        ok = False
        detail.append("valid 0-based grid wrongly rejected")
    # Index N_ELECTRODES (=2304) overflows under base=0 but is fine under base=1.
    try:
        validate_grid([N_ELECTRODES], width=GRID_WIDTH, base=0)
        ok = False
        detail.append("overflow index not caught with base=0")
    except GeometryError:
        pass
    try:
        validate_grid([N_ELECTRODES], width=GRID_WIDTH, base=1)
    except GeometryError:
        ok = False
        detail.append("valid base=1 index wrongly rejected")
    # Index 0 is below base under base=1 -> negative row -> must raise.
    try:
        validate_grid([0], width=GRID_WIDTH, base=1)
        ok = False
        detail.append("below-base index (0 with base=1) not caught")
    except GeometryError:
        pass
    # Simulate a recording whose TRUE base is 1 (indices 1..N): base=0 must flag it.
    true_base1 = list(range(1, N_ELECTRODES + 1))
    try:
        validate_grid(true_base1, width=GRID_WIDTH, base=0)
        ok = False
        detail.append("wrong-base (0 on a 1-based folder) not caught")
    except GeometryError:
        pass
    try:
        validate_grid(true_base1, width=GRID_WIDTH, base=1)  # correct base -> ok
    except GeometryError:
        ok = False
        detail.append("correct base=1 wrongly rejected")
    if ok:
        detail.append("range check catches wrong index_base in both directions")
    return ("validate_grid catches wrong index_base", ok, "; ".join(detail))


def check_determinism() -> Tuple[str, bool, str]:
    ok = True
    detail = []
    cand = list(range(0, N_ELECTRODES, 7))
    a = nearest_valid(1234, cand, 9, width=GRID_WIDTH, base=0)
    b = nearest_valid(1234, cand, 9, width=GRID_WIDTH, base=0)
    if a != b:
        ok = False
        detail.append("two identical calls differ: %r vs %r" % (a, b))
    return ("nearest_valid determinism", ok, "; ".join(detail))


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def main() -> int:
    checks = [
        check_coord_roundtrip,
        check_moore_all_valid,
        check_reaches_further,
        check_knn_vs_bruteforce,
        check_tie_determinism,
        check_mfr_exact,
        check_range_check_wrong_base,
        check_determinism,
    ]
    print("=" * 74)
    print("Stage 2 smoke test -- channel_subset_extraction (geometry + MFR)")
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
        print("ALL STAGE-2 CHECKS PASSED")
    else:
        print("STAGE-2 FAILURES: %d" % n_fail)
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
