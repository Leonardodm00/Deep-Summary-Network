"""Smoke test -- Stage 3 (greedy disjoint partition) of channel_subset_extraction.

Scope (one concern): partition_subregions and its Subregion / InsufficientElectrodesError.

KEY behaviours controlled (handoff Section 4 + Stage 3 test list):
  - adjacent top-MFR electrodes: the 2nd is ABSORBED into the hottest centre's
    subregion and is NOT itself a centre; the next-ranked non-absorbed electrode
    becomes the next centre;
  - subregions are fully DISJOINT (no electrode in two subregions), each size E;
  - a sub-theta electrode is excluded BOTH as a centre AND as a neighbour, and
    the search reaches further to a valid electrode instead;
  - ranking tie determinism: equal-MFR centres ordered by lower index;
  - channels ordered by centre MFR (channel 0 = hottest);
  - insufficient valid electrodes -> InsufficientElectrodesError; the C*E
    boundary succeeds;
  - determinism across repeated calls.

Run:
    cd /home/claude/work/hpc_multichannel
    python3 smoke_test_channel_subsets_stage3.py

Run TWICE with exit codes:
    cd /home/claude/work/hpc_multichannel
    python3 smoke_test_channel_subsets_stage3.py; echo "exit1=$?"; \
    python3 smoke_test_channel_subsets_stage3.py; echo "exit2=$?"
"""

from __future__ import annotations

import sys
from typing import Dict, List, Tuple

import numpy as np

from channel_subset_extraction import (
    GRID_WIDTH,
    InsufficientElectrodesError,
    Subregion,
    partition_subregions,
)

W = GRID_WIDTH  # 48


# --------------------------------------------------------------------------- #
# helpers (base=0 geometry throughout)
# --------------------------------------------------------------------------- #

def idx_of(row: int, col: int) -> int:
    return row * W + col


def rc_of(idx: int) -> Tuple[int, int]:
    return (idx // W, idx % W)


def block_indices(r0: int, c0: int, half: int) -> List[int]:
    """All indices in a (2*half+1) x (2*half+1) block centred at (r0, c0)."""
    out = []
    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            out.append(idx_of(r0 + dr, c0 + dc))
    return out


def coords_for(indices) -> Dict[int, Tuple[int, int]]:
    return {int(i): rc_of(int(i)) for i in indices}


def all_disjoint(subs: List[Subregion]) -> bool:
    seen = set()
    for s in subs:
        for m in s.members:
            if m in seen:
                return False
            seen.add(m)
    return True


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #

def check_absorbed_second_place() -> Tuple[str, bool, str]:
    """Hottest centre absorbs the adjacent 2nd-hottest; next-ranked is centre 1."""
    ok = True
    detail = []
    A = idx_of(10, 10)   # hottest
    B = idx_of(10, 11)   # 2nd hottest, Moore-adjacent to A -> must be absorbed
    Cc = idx_of(30, 30)  # hottest of the far cluster -> should become centre 1

    cluster1 = block_indices(10, 10, 1)  # 3x3 = 9 electrodes (exact E=9 fit)
    cluster2 = block_indices(30, 30, 1)  # 3x3 = 9 electrodes
    present = cluster1 + cluster2

    mfrs: Dict[int, float] = {}
    # cluster1 all above cluster2; A and B pinned as global top-2.
    v = 4.0
    for i in cluster1:
        if i == A:
            mfrs[i] = 5.0
        elif i == B:
            mfrs[i] = 4.5
        else:
            mfrs[i] = v
            v -= 0.2                 # 3.8, 3.6, ... all >= 1.0 and < 4.5
    # cluster2 all below 1.0; Cc highest of the cluster.
    v2 = 0.8
    for i in cluster2:
        if i == Cc:
            mfrs[i] = 0.9
        else:
            mfrs[i] = v2
            v2 -= 0.05               # 0.75, 0.70, ... all >= theta=0.1

    coords = coords_for(present)
    subs, discarded = partition_subregions(coords, mfrs, n_subsets=2,
                                           electrodes_per_subset=9, mfr_threshold=0.1)

    if len(subs) != 2:
        ok = False
        detail.append("got %d subregions, want 2" % len(subs))
    if subs and subs[0].center != A:
        ok = False
        detail.append("centre0 %d != A %d" % (subs[0].center, A))
    if subs and B not in subs[0].members:
        ok = False
        detail.append("B %d not absorbed into subregion0" % B)
    centers = {s.center for s in subs}
    if B in centers:
        ok = False
        detail.append("B %d wrongly became a centre" % B)
    if len(subs) > 1 and subs[1].center != Cc:
        ok = False
        detail.append("centre1 %d != Cc %d" % (subs[1].center, Cc))
    if not all_disjoint(subs):
        ok = False
        detail.append("subregions not disjoint")
    for s in subs:
        if len(s.members) != 9:
            ok = False
            detail.append("subregion size %d != 9" % len(s.members))
        if s.members[0] != s.center:
            ok = False
            detail.append("members[0] is not the centre")
    if discarded:
        ok = False
        detail.append("unexpected discarded: %r" % discarded)
    if ok:
        detail.append("A centre0, B absorbed (not a centre), Cc centre1, disjoint")
    return ("adjacent 2nd-MFR absorbed, next becomes centre", ok, "; ".join(detail))


def check_invalid_excluded() -> Tuple[str, bool, str]:
    """A sub-theta electrode is excluded as centre AND neighbour; search reaches on."""
    ok = True
    detail = []
    A = idx_of(10, 10)
    invalid = idx_of(9, 10)       # an orthogonal neighbour of A, set below theta
    block = block_indices(10, 10, 2)  # 5x5 = 25 electrodes around A

    mfrs: Dict[int, float] = {}
    hi = 3.0
    for i in block:
        if i == A:
            mfrs[i] = 9.0
        elif i == invalid:
            mfrs[i] = 0.01        # < theta -> discarded, must not appear anywhere
        else:
            mfrs[i] = hi
            hi -= 0.05            # all comfortably >= theta

    coords = coords_for(block)
    subs, discarded = partition_subregions(coords, mfrs, n_subsets=1,
                                           electrodes_per_subset=9, mfr_threshold=0.1)

    if len(subs) != 1:
        ok = False
        detail.append("got %d subregions, want 1" % len(subs))
    members = set(subs[0].members) if subs else set()
    if invalid in members:
        ok = False
        detail.append("sub-theta electrode %d wrongly included as member" % invalid)
    if invalid not in set(discarded):
        ok = False
        detail.append("sub-theta electrode %d missing from discarded" % invalid)
    if subs and subs[0].center == invalid:
        ok = False
        detail.append("sub-theta electrode wrongly became centre")
    # It must have reached beyond the invalidated orthogonal neighbour: the 8
    # members are A + 8 nearest valid; at least one member should be at d2 >= 4
    # (i.e. it reached past the broken Moore ring). Reference: nearest valid set.
    if subs:
        r0, c0 = rc_of(A)
        reached_far = any(((rc_of(m)[0] - r0) ** 2 + (rc_of(m)[1] - c0) ** 2) >= 4
                          for m in subs[0].members if m != A)
        if not reached_far:
            ok = False
            detail.append("did not reach past the broken Moore ring")
    if ok:
        detail.append("sub-theta electrode excluded as centre and neighbour; reached on")
    return ("sub-theta electrode excluded as centre AND neighbour", ok, "; ".join(detail))


def check_channel_ordering() -> Tuple[str, bool, str]:
    """Channels come out ordered by centre MFR descending (channel 0 hottest)."""
    ok = True
    detail = []
    # three well-separated single-Moore clusters with distinct centre MFRs
    centres = [(idx_of(5, 5), 2.0),
               (idx_of(5, 40), 3.0),
               (idx_of(40, 5), 1.0)]
    present = []
    mfrs: Dict[int, float] = {}
    for (cidx, cmfr), fill in zip(centres, (0.5, 0.6, 0.4)):
        blk = block_indices(*rc_of(cidx), half=1)
        present += blk
        for i in blk:
            mfrs[i] = cmfr if i == cidx else fill
    coords = coords_for(present)
    subs, _ = partition_subregions(coords, mfrs, n_subsets=3,
                                   electrodes_per_subset=9, mfr_threshold=0.1)
    got_order = [s.center_mfr for s in subs]
    if got_order != sorted(got_order, reverse=True):
        ok = False
        detail.append("centre MFRs not descending: %r" % got_order)
    if subs and subs[0].center != idx_of(5, 40):
        ok = False
        detail.append("channel 0 centre %d != hottest %d" % (subs[0].center, idx_of(5, 40)))
    if ok:
        detail.append("channels sorted by centre MFR desc, channel 0 = hottest")
    return ("channel ordering by centre MFR", ok, "; ".join(detail))


def check_tie_determinism() -> Tuple[str, bool, str]:
    """Equal-MFR centres: the lower-index cluster becomes channel 0."""
    ok = True
    detail = []
    cA = idx_of(5, 5)         # lower index
    cB = idx_of(40, 40)       # higher index
    present = []
    mfrs: Dict[int, float] = {}
    for cidx in (cA, cB):
        blk = block_indices(*rc_of(cidx), half=1)
        present += blk
        for i in blk:
            mfrs[i] = 5.0 if i == cidx else 1.0   # both centres EQUAL MFR
    coords = coords_for(present)
    subs, _ = partition_subregions(coords, mfrs, n_subsets=2,
                                   electrodes_per_subset=9, mfr_threshold=0.1)
    if subs and subs[0].center != cA:
        ok = False
        detail.append("equal-MFR tie: channel 0 centre %d != lower index %d"
                      % (subs[0].center, cA))
    # run twice -> identical
    subs2, _ = partition_subregions(coords, mfrs, n_subsets=2,
                                    electrodes_per_subset=9, mfr_threshold=0.1)
    if [s.center for s in subs] != [s.center for s in subs2] or \
       [s.members for s in subs] != [s.members for s in subs2]:
        ok = False
        detail.append("two runs differ")
    if ok:
        detail.append("equal-MFR tie resolved by lower index, deterministic")
    return ("ranking tie determinism", ok, "; ".join(detail))


def check_insufficient_raises() -> Tuple[str, bool, str]:
    """Too few valid electrodes -> InsufficientElectrodesError; C*E boundary OK."""
    ok = True
    detail = []
    # C=2, E=9 -> need 18. Provide 17 valid electrodes in one clump -> must raise.
    blk = block_indices(20, 20, 2)  # 25 electrodes
    present17 = blk[:17]
    coords = coords_for(present17)
    mfrs17 = {i: 1.0 for i in present17}
    raised = False
    try:
        partition_subregions(coords, mfrs17, n_subsets=2,
                             electrodes_per_subset=9, mfr_threshold=0.1)
    except InsufficientElectrodesError:
        raised = True
    if not raised:
        ok = False
        detail.append("17 valid for C*E=18 did not raise")

    # Exactly C*E = 18 valid -> must succeed with 2 full subregions.
    present18 = blk[:18]
    coords18 = coords_for(present18)
    mfrs18 = {i: 1.0 for i in present18}
    try:
        subs, _ = partition_subregions(coords18, mfrs18, n_subsets=2,
                                       electrodes_per_subset=9, mfr_threshold=0.1)
        if len(subs) != 2 or not all_disjoint(subs) or \
           any(len(s.members) != 9 for s in subs):
            ok = False
            detail.append("C*E boundary did not produce 2 disjoint size-9 subregions")
    except InsufficientElectrodesError:
        ok = False
        detail.append("C*E=18 boundary wrongly raised")

    # All electrodes below theta -> V empty -> raise.
    mfrs_low = {i: 0.01 for i in present18}
    raised2 = False
    try:
        partition_subregions(coords18, mfrs_low, n_subsets=1,
                             electrodes_per_subset=9, mfr_threshold=0.1)
    except InsufficientElectrodesError:
        raised2 = True
    if not raised2:
        ok = False
        detail.append("empty V did not raise")
    if ok:
        detail.append("insufficient raises; C*E boundary succeeds; empty V raises")
    return ("insufficient-electrodes policy", ok, "; ".join(detail))


def check_e_equals_one() -> Tuple[str, bool, str]:
    """E=1: each subregion is a bare centre; hottest C become the channels."""
    ok = True
    detail = []
    present = [idx_of(5, 5), idx_of(6, 6), idx_of(7, 7), idx_of(8, 8)]
    mfrs = {present[0]: 4.0, present[1]: 3.0, present[2]: 2.0, present[3]: 1.0}
    coords = coords_for(present)
    subs, _ = partition_subregions(coords, mfrs, n_subsets=3,
                                   electrodes_per_subset=1, mfr_threshold=0.1)
    if [s.center for s in subs] != present[:3]:
        ok = False
        detail.append("E=1 centres %r != top-3 %r"
                      % ([s.center for s in subs], present[:3]))
    if any(len(s.members) != 1 for s in subs):
        ok = False
        detail.append("E=1 subregion has != 1 member")
    if ok:
        detail.append("E=1 yields bare-centre channels, hottest C selected")
    return ("E=1 edge case", ok, "; ".join(detail))


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def main() -> int:
    checks = [
        check_absorbed_second_place,
        check_invalid_excluded,
        check_channel_ordering,
        check_tie_determinism,
        check_insufficient_raises,
        check_e_equals_one,
    ]
    print("=" * 74)
    print("Stage 3 smoke test -- channel_subset_extraction (greedy disjoint partition)")
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
        print("ALL STAGE-3 CHECKS PASSED")
    else:
        print("STAGE-3 FAILURES: %d" % n_fail)
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
