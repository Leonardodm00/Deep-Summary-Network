"""MEA per-region IFR extractor -- channel-subset extraction.

This module turns a folder of 3Brain per-electrode binary spike rasters
(ptrain_<k>.mat, one file per electrode) into a multichannel instantaneous
firing-rate (IFR) trace, and can also produce the traditional whole-culture
single-channel IFR.

Staged build (see HANDOFF_mea_extractor.md, Section 7):
  Stage 1 (THIS commit) -- I/O + inventory:
      load_ptrain_folder(folder, fs_raw, index_base) -> PtrainInventory
  Stages 2-5 -- geometry, MFR, greedy disjoint partition, IFR, public API
      (added in subsequent commits; not present yet).

Conventions used across the pipeline (fixed here for downstream stages):
  - An electrode's "linear index" k is the integer in its filename ptrain_<k>.mat.
  - A "spike sample index" is a raw-sample position (0-based into the raster) at
    which a spike occurred. Spike time in seconds is index / fs_raw.
  - n_samples (call it n) is the raw-sample length of the raster; it must be
    identical across all electrodes of one recording.
  - T_rec = n / fs_raw   [s]   (recording duration).
  - fs_raw is NOT stored in the .mat (it is the .brw SamplingRate); it is a
    tuning parameter with default DEFAULT_FS_RAW (see handoff Section 2).

Coding rules honoured here (handoff Section 6):
  - Pure ASCII source (HPC-safe; no greek/arrows/em-dashes/smart-quotes).
  - Leverage established libraries: scipy.io.loadmat for v5 .mat, numpy for arrays.
  - Strict separation of concerns: filename parsing, single-file load, and folder
    inventory are three distinct functions; geometry / partition / IFR live in
    later-stage functions (added in Stages 2+), not here.
"""

from __future__ import annotations

import os
import re
from collections import namedtuple
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

import numpy as np
import scipy.io as sio

__all__ = [
    # Stage 1 -- I/O + inventory
    "DEFAULT_FS_RAW",
    "PTRAIN_VARNAME",
    "PtrainLoadError",
    "PtrainInventory",
    "parse_ptrain_index",
    "load_ptrain_file",
    "load_ptrain_folder",
    # Stage 2 -- geometry + MFR
    "GRID_WIDTH",
    "PITCH_UM",
    "N_ELECTRODES",
    "GeometryError",
    "ElectrodeCoord",
    "electrode_coords",
    "validate_grid",
    "mean_firing_rates",
    "nearest_valid",
    # Stage 3 -- greedy disjoint partition
    "InsufficientElectrodesError",
    "Subregion",
    "partition_subregions",
    # Stage 4 -- per-subregion IFR + whole-culture IFR
    "DEFAULT_W_SIZE",
    "DEFAULT_GAUSSIAN_WINDOW",
    "subregion_ifr",
    "subregion_ifrs",
    "subregion_single_channel_traces",
    "whole_culture_ifr",
    # Stage 5 -- public entry point
    "MODES",
    "ExtractionDiagnostics",
    "extract_channel_subsets",
]

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #

# Default raw sampling rate [Hz]. With n_samples = 12,132,108 this gives
# T_rec = 12,132,108 / 10110.09 = 1200.0 s = 20 min. TUNABLE per recording.
DEFAULT_FS_RAW: float = 10110.09

# The single MATLAB variable expected inside each ptrain_<k>.mat file.
PTRAIN_VARNAME: str = "ptrain"

# Strict filename pattern: ptrain_<k>.mat where <k> is a non-negative integer.
_PTRAIN_RE = re.compile(r"^ptrain_(\d+)\.mat$")


# --------------------------------------------------------------------------- #
# exceptions
# --------------------------------------------------------------------------- #

class PtrainLoadError(ValueError):
    """Raised when a ptrain file / folder violates the expected format.

    Subclasses ValueError so existing callers that catch ValueError keep working,
    while a caller (e.g. the Stage-7 provider) can catch this specific type to
    discard an unreadable recording.
    """


# --------------------------------------------------------------------------- #
# inventory container
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PtrainInventory:
    """Immutable inventory of one recording folder (Stage 1 output).

    Attributes
    ----------
    spikes : dict[int, np.ndarray]
        Map linear electrode index k -> spike sample indices, a 1-D int64 array
        sorted ascending. An electrode present but silent yields an empty array.
        Iteration order of the dict is ascending k (deterministic).
    n_samples : int
        Common raw-sample length n of every raster in the recording.
    fs_raw : float
        Raw sampling rate [Hz] used to interpret sample indices as seconds.
    T_rec : float
        Recording duration [s], equal to n_samples / fs_raw.
    index_base : int
        Index base (0 or 1) of the linear indices. CARRIED for downstream
        geometry mapping (Stage 2) but NOT applied to the keys here: keys are the
        raw integers from the filenames exactly as found on disk.
    """

    spikes: Dict[int, np.ndarray]
    n_samples: int
    fs_raw: float
    T_rec: float
    index_base: int

    @property
    def indices(self) -> List[int]:
        """Sorted list of the electrode linear indices present in the folder."""
        return sorted(self.spikes.keys())

    def n_spikes(self, k: int) -> int:
        """Number of spikes recorded on electrode k."""
        return int(self.spikes[k].size)


# --------------------------------------------------------------------------- #
# Stage 1a -- filename -> linear index
# --------------------------------------------------------------------------- #

def parse_ptrain_index(name: str) -> int:
    """Parse a ptrain filename into its electrode linear index.

    Accepts either a bare basename ("ptrain_100.mat") or a path; only the
    basename is matched. The pattern is strict: ptrain_<k>.mat with <k> a run of
    ASCII digits. Anything else raises PtrainLoadError.

    Parameters
    ----------
    name : str
        Filename or path ending in a ptrain_<k>.mat basename.

    Returns
    -------
    int
        The parsed linear index k (>= 0).
    """
    base = os.path.basename(name)
    m = _PTRAIN_RE.match(base)
    if m is None:
        raise PtrainLoadError(
            "filename does not match ptrain_<k>.mat: %r" % (base,)
        )
    return int(m.group(1))


# --------------------------------------------------------------------------- #
# Stage 1b -- single-file load
# --------------------------------------------------------------------------- #

def load_ptrain_file(path: str) -> Tuple[np.ndarray, int]:
    """Load one ptrain_<k>.mat and return its spike sample indices.

    The file is a MATLAB v5 .mat (read with scipy.io.loadmat, NOT h5py) holding a
    single variable PTRAIN_VARNAME: a binary raster of shape (n_samples, 1),
    dtype uint8, values in {0, 1}. Spike sample indices are the positions of the
    ones: np.nonzero(raster.ravel())[0].

    Parameters
    ----------
    path : str
        Path to a single ptrain_<k>.mat file.

    Returns
    -------
    spike_sample_indices : np.ndarray
        1-D int64 array of raw-sample positions of spikes, sorted ascending
        (np.nonzero already yields ascending order). Empty if the electrode is
        silent.
    n_samples : int
        Raster length n (number of raw samples).

    Raises
    ------
    PtrainLoadError
        If the variable is missing, the raster is not effectively 1-D, or the
        raster is not binary (contains a value > 1).
    """
    try:
        md = sio.loadmat(path)
    except Exception as exc:  # noqa: BLE001 -- surface any scipy read failure loudly
        raise PtrainLoadError("failed to read %r with scipy.io.loadmat: %s"
                              % (path, exc)) from exc

    if PTRAIN_VARNAME not in md:
        present = [k for k in md.keys() if not k.startswith("__")]
        raise PtrainLoadError(
            "variable %r not found in %r (present variables: %s)"
            % (PTRAIN_VARNAME, path, present)
        )

    raster = md[PTRAIN_VARNAME]

    # Expect (n, 1) (or (1, n)); accept any 2-D shape with a singleton axis, and
    # a genuinely 1-D array. Reject anything with two non-singleton axes.
    if raster.ndim == 2:
        if 1 not in raster.shape:
            raise PtrainLoadError(
                "raster in %r is 2-D with shape %r; expected (n_samples, 1)"
                % (path, tuple(raster.shape))
            )
    elif raster.ndim != 1:
        raise PtrainLoadError(
            "raster in %r has ndim=%d; expected a 1-D or (n, 1) raster"
            % (path, raster.ndim)
        )

    flat = np.asarray(raster).ravel()
    n_samples = int(flat.size)

    # Binary guard: spike extraction via np.nonzero assumes values in {0, 1};
    # a value > 1 would silently lose a spike count, so fail loud instead.
    if n_samples > 0:
        mx = int(flat.max())
        if mx > 1:
            raise PtrainLoadError(
                "raster in %r is not binary (max value = %d); expected {0, 1}"
                % (path, mx)
            )

    spike_sample_indices = np.nonzero(flat)[0].astype(np.int64)
    return spike_sample_indices, n_samples


# --------------------------------------------------------------------------- #
# Stage 1c -- folder inventory
# --------------------------------------------------------------------------- #

def load_ptrain_folder(
    folder: str,
    fs_raw: float = DEFAULT_FS_RAW,
    index_base: int = 0,
) -> PtrainInventory:
    """Load every ptrain_<k>.mat in a folder into a PtrainInventory.

    Only files whose basename matches ptrain_<k>.mat are considered; other files
    in the folder (README, .DS_Store, screenshots, ...) are ignored. Every
    matched raster must share the same n_samples; a mismatch is a hard error.

    Parameters
    ----------
    folder : str
        Path to the recording folder.
    fs_raw : float, optional
        Raw sampling rate [Hz]; default DEFAULT_FS_RAW. Used to compute T_rec.
    index_base : int, optional
        Index base (0 or 1) recorded in the inventory for Stage-2 geometry; NOT
        applied to the keys here. Default 0.

    Returns
    -------
    PtrainInventory
        spikes keyed by ascending linear index, plus n_samples, fs_raw, T_rec,
        index_base.

    Raises
    ------
    PtrainLoadError
        If the folder does not exist, contains no ptrain files, has a duplicate
        linear index, or the rasters disagree on n_samples; also propagated from
        load_ptrain_file for per-file format problems.
    ValueError
        If fs_raw is not strictly positive or index_base is not in {0, 1}.
    """
    if fs_raw <= 0.0:
        raise ValueError("fs_raw must be > 0, got %r" % (fs_raw,))
    if index_base not in (0, 1):
        raise ValueError("index_base must be 0 or 1, got %r" % (index_base,))
    if not os.path.isdir(folder):
        raise PtrainLoadError("not a directory: %r" % (folder,))

    # Discover and parse ptrain files (sorted by linear index for determinism).
    matched: List[Tuple[int, str]] = []
    for name in os.listdir(folder):
        if _PTRAIN_RE.match(name) is None:
            continue
        k = parse_ptrain_index(name)
        matched.append((k, os.path.join(folder, name)))

    if not matched:
        raise PtrainLoadError(
            "no ptrain_<k>.mat files found in %r" % (folder,)
        )

    matched.sort(key=lambda kp: kp[0])

    spikes: Dict[int, np.ndarray] = {}
    n_ref: int = -1
    n_ref_path: str = ""
    for k, path in matched:
        if k in spikes:
            raise PtrainLoadError(
                "duplicate linear index %d in folder %r" % (k, folder)
            )
        idx, n_samples = load_ptrain_file(path)
        if n_ref < 0:
            n_ref = n_samples
            n_ref_path = path
        elif n_samples != n_ref:
            raise PtrainLoadError(
                "n_samples mismatch: %r has n=%d but %r has n=%d "
                "(all electrodes of one recording must share n_samples)"
                % (path, n_samples, n_ref_path, n_ref)
            )
        spikes[k] = idx

    t_rec = float(n_ref) / float(fs_raw)
    return PtrainInventory(
        spikes=spikes,
        n_samples=int(n_ref),
        fs_raw=float(fs_raw),
        T_rec=t_rec,
        index_base=int(index_base),
    )


# --------------------------------------------------------------------------- #
# Stage 2 -- geometry + MFR
# --------------------------------------------------------------------------- #
#
# Chip geometry (handoff Section 3):
#   - Square grid GRID_WIDTH x GRID_WIDTH electrodes (48 x 48 = 2304 per well).
#   - Linear index -> (row, col): ROW-MAJOR, row = i0 // width, col = i0 % width,
#     with i0 = idx - base (base in {0, 1}; base unconfirmed, default 0).
#   - Physical coords: PITCH_UM isotropic; (x_um, y_um) = (col*PITCH_UM, row*PITCH_UM).
#   - Nearest-neighbour ranking uses Euclidean distance on the INTEGER (row, col)
#     grid (handoff Section 4). Because the pitch is isotropic, this ordering is
#     identical to ranking on (x_um, y_um); (row, col) is used so the 8 Moore
#     neighbours come out exact.

GRID_WIDTH: int = 48                          # electrodes per side (square 48 x 48 well)
PITCH_UM: float = 60.0                        # electrode pitch [um], isotropic
N_ELECTRODES: int = GRID_WIDTH * GRID_WIDTH   # 2304

ElectrodeCoord = namedtuple("ElectrodeCoord", ["row", "col", "x_um", "y_um"])


class GeometryError(ValueError):
    """Raised when a linear index maps outside the electrode grid.

    Almost always means the index_base (0 vs 1) or orientation guess is wrong;
    flip the base (or confirm orientation via the Stage-6 electrode-map plot).
    """


def _rowcol(idx, width: int, base: int):
    """Vectorized (row, col) from linear index (single source of the mapping).

    Works for a Python int or a numpy integer array. Uses floor division so a
    below-base index yields a negative row that the grid range check rejects.
    """
    i0 = np.asarray(idx) - base
    return i0 // width, i0 % width


def electrode_coords(idx: int, width: int = GRID_WIDTH, base: int = 0) -> ElectrodeCoord:
    """Map one electrode linear index to grid and physical coordinates.

    Parameters
    ----------
    idx : int
        Electrode linear index as found in the filename ptrain_<idx>.mat.
    width : int, optional
        Grid width (electrodes per row). Default GRID_WIDTH (48).
    base : int, optional
        Index base (0 or 1). i0 = idx - base is the 0-based grid position.

    Returns
    -------
    ElectrodeCoord
        Named tuple (row, col, x_um, y_um) with row = i0 // width,
        col = i0 % width, x_um = col*PITCH_UM, y_um = row*PITCH_UM.

    Notes
    -----
    This is a PURE mapping: it does NOT range-check idx. Use validate_grid to
    verify that a whole set of indices falls inside [0, width) x [0, width).
    """
    r, c = _rowcol(int(idx), width, base)
    r = int(r)
    c = int(c)
    return ElectrodeCoord(row=r, col=c, x_um=c * PITCH_UM, y_um=r * PITCH_UM)


def validate_grid(indices, width: int = GRID_WIDTH, base: int = 0) -> None:
    """Range-check that every index maps inside the square grid.

    Every valid 0-based grid position i0 = idx - base must satisfy
    0 <= row < width and 0 <= col < width, i.e. 0 <= i0 < width*width.

    Parameters
    ----------
    indices : iterable of int
        Electrode linear indices to check.
    width : int, optional
        Grid width. Default GRID_WIDTH (48).
    base : int, optional
        Index base (0 or 1). Default 0.

    Raises
    ------
    GeometryError
        If any index maps outside the grid, listing up to 10 offenders. This
        signals a wrong base/orientation guess (flip base, or confirm via the
        Stage-6 plot).
    """
    arr = np.asarray(list(indices), dtype=np.int64)
    if arr.size == 0:
        return
    rows, cols = _rowcol(arr, width, base)
    inside = (rows >= 0) & (rows < width) & (cols >= 0) & (cols < width)
    if not np.all(inside):
        bad = arr[~inside]
        raise GeometryError(
            "%d electrode index/indices map outside the %dx%d grid with base=%d: "
            "%s ... (wrong index_base or orientation? try base=%d)"
            % (int(bad.size), width, width, base, bad[:10].tolist(), 1 - base)
        )


def mean_firing_rates(inv: PtrainInventory, T_rec: float) -> Dict[int, float]:
    """Mean firing rate MFR_e = N_spikes(e) / T_rec [Hz] for each electrode e.

    Parameters
    ----------
    inv : PtrainInventory
        Stage-1 inventory (uses inv.spikes for the per-electrode spike counts).
    T_rec : float
        Recording duration [s]. Normally inv.T_rec; passed explicitly so a caller
        can override (e.g. a trimmed analysis window). Must be > 0.

    Returns
    -------
    dict[int, float]
        Map electrode linear index -> MFR_e [Hz], keys in ascending order.
    """
    if T_rec <= 0.0:
        raise ValueError("T_rec must be > 0, got %r" % (T_rec,))
    return {k: inv.spikes[k].size / T_rec for k in sorted(inv.spikes.keys())}


def nearest_valid(centre: int, candidates, k: int,
                  width: int = GRID_WIDTH, base: int = 0) -> List[int]:
    """Return the k nearest candidate electrodes to centre.

    Distance is Euclidean on the integer (row, col) grid; ties are broken by
    lower linear index (deterministic). The centre itself is excluded from the
    result even if present in candidates. Fewer than k are returned if fewer
    candidates are available.

    Parameters
    ----------
    centre : int
        Linear index of the centre electrode.
    candidates : iterable of int
        Candidate electrode linear indices to search (e.g. the still-unassigned
        valid electrodes). The centre is removed if present.
    k : int
        Number of nearest neighbours to return (k >= 0).
    width : int, optional
        Grid width. Default GRID_WIDTH (48).
    base : int, optional
        Index base (0 or 1). Default 0.

    Returns
    -------
    list[int]
        Up to k candidate indices, ordered by increasing (distance^2, index).
    """
    cand = [int(c) for c in candidates if int(c) != int(centre)]
    if k <= 0 or not cand:
        return []
    # Build the (row, col) coords for centre + candidates via the single mapping,
    # then delegate to the shared ranking core (single source of the KNN rule).
    idxs = np.asarray([int(centre)] + cand, dtype=np.int64)
    rows, cols = _rowcol(idxs, width, base)
    coords = {int(i): (int(r), int(c))
              for i, r, c in zip(idxs.tolist(), rows.tolist(), cols.tolist())}
    return _nearest_indices(int(centre), cand, coords)[:k]


# --------------------------------------------------------------------------- #
# Stage 3 -- greedy DISJOINT partition (core)
# --------------------------------------------------------------------------- #
#
# Algorithm (handoff Section 4, locked):
#   Symbols: C = n_subsets (output channels); E = electrodes_per_subset
#   (1 centre + (E-1) neighbours); theta = MFR threshold.
#   1. Valid set V = { e : MFR_e >= theta }; the rest are discarded (kept for viz).
#   2. Rank V by MFR descending; ties -> lower linear index first (determinism).
#   3. Greedy disjoint partition:
#        assigned = {}. Iterate candidates e in rank order:
#          - if e already assigned: skip it AS A CENTRE (an already-absorbed
#            high-MFR electrode is not a new centre);
#          - else centre = e; members = {e} + the (E-1) nearest electrodes in V
#            that are NOT yet assigned (Euclidean on (row,col), ties -> lower
#            index); mark all E members assigned;
#          - stop when C centres formed.
#      NO radius cap. Subregions are fully DISJOINT (no electrode in two
#      subregions) -- required to avoid cross-channel spike double-counting.
#   4. INSUFFICIENT-ELECTRODES POLICY: if fewer than C disjoint subregions of
#      size E can be formed from V -> raise InsufficientElectrodesError so the
#      caller discards the recording. Hard error; do NOT degrade C or E.
#   5. Channel ordering is DATA-DRIVEN: channels ordered by centre MFR
#      (channel 0 = hottest). By construction the created-centre order already
#      equals (MFR desc, index asc); this is sorted explicitly for robustness.
#
# The (E-1)-nearest search reuses _nearest_indices, the same ranking core that
# backs Stage-2 nearest_valid, so the neighbour rule is defined in exactly one
# place. partition_subregions receives an explicit coords dict (its handoff
# signature), which keeps the partition geometry-agnostic (it does not need to
# know width/base).


class InsufficientElectrodesError(Exception):
    """Raised when V cannot yield C disjoint subregions of size E.

    Signals the caller (e.g. the Stage-7 provider) to DISCARD the recording.
    Deliberately not a ValueError: it is a data-sufficiency condition, not a
    malformed-argument error.
    """


@dataclass(frozen=True)
class Subregion:
    """One output channel's electrode subregion (Stage 3 result).

    Attributes
    ----------
    center : int
        Linear index of the centre (highest-MFR seed) electrode.
    members : tuple[int, ...]
        The E member linear indices: centre first, then the (E-1) neighbours in
        increasing (distance^2, index) order. Length is exactly E.
    mean_mfr : float
        Mean MFR over the E members [Hz] (diagnostic).
    center_mfr : float
        MFR of the centre electrode [Hz]; drives the data-driven channel order.
    """

    center: int
    members: Tuple[int, ...]
    mean_mfr: float
    center_mfr: float


def _nearest_indices(centre: int, candidates, coords: Dict[int, Tuple[int, int]]) -> List[int]:
    """Candidates (excluding centre) sorted by (distance^2, index).

    Single source of the nearest-neighbour rule shared by nearest_valid and
    partition_subregions.

    Parameters
    ----------
    centre : int
        Centre electrode linear index.
    candidates : iterable of int
        Candidate linear indices (centre removed if present).
    coords : dict[int, tuple[int, int]]
        Map linear index -> (row, col); must cover centre and all candidates.

    Returns
    -------
    list[int]
        All candidates (minus centre) ordered by increasing Euclidean distance
        on (row, col), ties broken by lower linear index.
    """
    cand = [int(c) for c in candidates if int(c) != int(centre)]
    if not cand:
        return []
    cand_arr = np.asarray(cand, dtype=np.int64)
    r0, c0 = coords[int(centre)]
    rows = np.fromiter((coords[c][0] for c in cand), dtype=np.int64, count=len(cand))
    cols = np.fromiter((coords[c][1] for c in cand), dtype=np.int64, count=len(cand))
    d2 = (rows - int(r0)) ** 2 + (cols - int(c0)) ** 2
    # np.lexsort: last key primary -> sort by d2 (nearest first), tie by index.
    order = np.lexsort((cand_arr, d2))
    return cand_arr[order].tolist()


def partition_subregions(
    coords: Dict[int, Tuple[int, int]],
    mfrs: Dict[int, float],
    n_subsets: int = 9,
    electrodes_per_subset: int = 9,
    mfr_threshold: float = 0.1,
) -> Tuple[List[Subregion], List[int]]:
    """Greedy disjoint partition of the valid electrodes into C subregions.

    Parameters
    ----------
    coords : dict[int, tuple[int, int]]
        Map electrode linear index -> (row, col). Must cover every electrode in
        mfrs (or at least every valid one).
    mfrs : dict[int, float]
        Map electrode linear index -> MFR_e [Hz] (from mean_firing_rates).
    n_subsets : int, optional
        C, the number of output channels/subregions to form. Default 9.
    electrodes_per_subset : int, optional
        E, electrodes per subregion (1 centre + (E-1) neighbours). Default 9.
    mfr_threshold : float, optional
        theta, minimum MFR [Hz] to be a valid electrode. Default 0.1.

    Returns
    -------
    subregions : list[Subregion]
        Exactly C subregions, ordered by centre MFR descending (channel 0 =
        hottest), ties by lower centre index.
    discarded : list[int]
        Sorted linear indices of electrodes with MFR_e < theta (kept for viz).

    Raises
    ------
    InsufficientElectrodesError
        If fewer than C disjoint subregions of size E can be formed from V.
    ValueError
        If n_subsets or electrodes_per_subset is not >= 1.
    KeyError
        If a valid electrode is missing from coords.
    """
    C = int(n_subsets)
    E = int(electrodes_per_subset)
    theta = float(mfr_threshold)
    if C < 1:
        raise ValueError("n_subsets must be >= 1, got %r" % (n_subsets,))
    if E < 1:
        raise ValueError("electrodes_per_subset must be >= 1, got %r" % (E,))

    # Step 1: valid set V and discarded set.
    valid = [e for e in mfrs if mfrs[e] >= theta]
    discarded = sorted(e for e in mfrs if mfrs[e] < theta)

    missing = [e for e in valid if e not in coords]
    if missing:
        raise KeyError(
            "coords is missing %d valid electrode(s), e.g. %s"
            % (len(missing), missing[:10])
        )

    # Step 2: rank V by (MFR desc, index asc).
    ranked = sorted(valid, key=lambda e: (-mfrs[e], e))

    # Step 3: greedy disjoint partition.
    assigned: set = set()
    subregions: List[Subregion] = []
    for e in ranked:
        if len(subregions) >= C:
            break
        if e in assigned:
            continue  # an already-absorbed electrode is NOT a new centre
        pool = [v for v in ranked if v not in assigned]  # unassigned valid (incl. e)
        neighbours = _nearest_indices(e, pool, coords)[: E - 1]
        if len(neighbours) < E - 1:
            raise InsufficientElectrodesError(
                "cannot complete subregion #%d (size E=%d) around centre %d: "
                "only %d unassigned valid neighbour(s) available, need %d. "
                "|V|=%d, required C*E=%d."
                % (len(subregions), E, e, len(neighbours), E - 1, len(valid), C * E)
            )
        members = (e,) + tuple(neighbours)
        assigned.update(members)
        subregions.append(
            Subregion(
                center=e,
                members=members,
                mean_mfr=float(np.mean([mfrs[m] for m in members])),
                center_mfr=float(mfrs[e]),
            )
        )

    if len(subregions) < C:
        raise InsufficientElectrodesError(
            "formed only %d of %d requested subregions (size E=%d) from |V|=%d "
            "valid electrodes; required C*E=%d."
            % (len(subregions), C, E, len(valid), C * E)
        )

    # Step 5: channel ordering by centre MFR (desc), ties by lower centre index.
    # Already in this order by construction; sort explicitly for robustness.
    subregions.sort(key=lambda s: (-s.center_mfr, s.center))
    return subregions, discarded


# --------------------------------------------------------------------------- #
# Stage 4 -- per-subregion IFR + whole-culture IFR
# --------------------------------------------------------------------------- #
#
# Reuses compute_ifr_trace from generate_burst_data (the SAME smoothed-cumulative
# IFR primitive the synthetic generator uses): population spike counts per bin
# C[k] = Sum_e |S_e intersect [k*Dt, (k+1)*Dt)|, then R_tilde = gaussian_filter1d(
# C, sigma = sigma_smooth/Dt), clipped to >= 0, at fs_ifr = 1/Dt Hz. Nothing about
# the IFR math is re-implemented here.
#
# Locked IFR parameters (handoff Section 4): Dt = w_size = 0.02 s (fs_ifr = 50 Hz),
# sigma_smooth = gaussian_window = 0.04 s, duration_s = T_rec. K = floor(T_rec/Dt).
#
# NORMALIZATION (handoff): each pooled IFR is divided by the number of electrodes
# pooled, R_norm[k] = R_tilde[k] / n_e. Because gaussian_filter1d is linear this
# equals smoothing the MEAN per-electrode count, R_tilde(C / n_e). For a subregion
# n_e = E (constant across channels); for the whole culture n_e = number of firing
# electrodes (variable).
#
# Spike times are stored by Stage 1 as raw integer SAMPLE indices; they are
# converted to seconds here via t_seconds = sample_index / fs_raw, the unit
# compute_ifr_trace expects.
#
# SCALE-PARITY FLAG (unresolved, surfaced deliberately): this per-electrode
# normalization puts REAL traces on a mean-per-electrode-rate scale, whereas the
# synthetic generate_multichannel_ifr emits UNNORMALIZED summed counts. The CNN is
# not scale-invariant, so the real and synthetic amplitude conventions MUST be
# reconciled before training on both. Left to the training/wiring stage on purpose;
# Stage 4 implements the locked (normalized) convention.
#
# compute_ifr_trace / CONTROL_PARAMS are imported lazily so that importing this
# extraction module does not trigger the matplotlib import side effect that
# generate_burst_data carries at import time (relevant on headless compute nodes).

DEFAULT_W_SIZE: float = 0.02            # Delta_t [s]  -> fs_ifr = 50 Hz
DEFAULT_GAUSSIAN_WINDOW: float = 0.04   # sigma_smooth [s]


def _ifr_params(T_rec: float, w_size: float, gaussian_window: float):
    """Build the BurstParams consumed by compute_ifr_trace.

    Only duration_s (= T_rec), w_size (= Delta_t) and gaussian_window
    (= sigma_smooth) are read by compute_ifr_trace; the generative fields carried
    by CONTROL_PARAMS are irrelevant to the IFR computation.
    """
    from generate_burst_data import CONTROL_PARAMS  # lazy (see module note above)
    return replace(CONTROL_PARAMS, duration_s=float(T_rec),
                   w_size=float(w_size), gaussian_window=float(gaussian_window))


def _spike_seconds(inv: PtrainInventory, electrodes) -> List[np.ndarray]:
    """Per-electrode spike-time arrays in SECONDS for the given electrodes.

    t_seconds = sample_index / fs_raw. Returns one float64 array per electrode;
    compute_ifr_trace pools them by summing per-bin histograms.
    """
    return [inv.spikes[int(e)].astype(np.float64) / inv.fs_raw for e in electrodes]


def subregion_ifr(inv: PtrainInventory, subregion: Subregion,
                  w_size: float = DEFAULT_W_SIZE,
                  gaussian_window: float = DEFAULT_GAUSSIAN_WINDOW) -> Tuple[np.ndarray, float]:
    """Normalized IFR (K,) for a single subregion.

    Pools the E member electrodes, computes the smoothed cumulative IFR, and
    divides by E (per-electrode normalization).

    Parameters
    ----------
    inv : PtrainInventory
        Stage-1 inventory (provides spikes [samples], fs_raw, T_rec).
    subregion : Subregion
        A Stage-3 subregion; its members are pooled.
    w_size : float, optional
        Delta_t [s]. Default DEFAULT_W_SIZE (0.02).
    gaussian_window : float, optional
        sigma_smooth [s]. Default DEFAULT_GAUSSIAN_WINDOW (0.04).

    Returns
    -------
    ifr : (K,) float32, K = floor(T_rec / w_size).
    fs_ifr : float = 1 / w_size [Hz].
    """
    from generate_burst_data import compute_ifr_trace  # lazy (see module note)
    params = _ifr_params(inv.T_rec, w_size, gaussian_window)
    ifr, fs_ifr = compute_ifr_trace(_spike_seconds(inv, subregion.members), params)
    n_e = len(subregion.members)
    return (ifr / float(n_e)).astype(np.float32), fs_ifr


def subregion_single_channel_traces(
    inv: PtrainInventory, subregions: List[Subregion],
    w_size: float = DEFAULT_W_SIZE,
    gaussian_window: float = DEFAULT_GAUSSIAN_WINDOW,
) -> Tuple[List[np.ndarray], float]:
    """Mode 3: each subregion as its OWN single-channel (K,) trace.

    Uses the SAME subregions and the SAME per-electrode-normalized IFRs as the
    multichannel Mode 1 (subregion_ifrs), but returns them UNSTACKED: one
    independent single-channel trace per subregion instead of one stacked (C, K)
    multichannel array. All C traces come from the same recording (they share its
    condition label, applied downstream by the provider), so a recording yields C
    single-channel samples for the in_channels=1 backbone rather than one
    C-channel sample.

    This is the primitive; subregion_ifrs is its stacked view, so
    stack(traces) == subregion_ifrs(...) bit-for-bit.

    Parameters
    ----------
    inv : PtrainInventory
        Stage-1 inventory.
    subregions : list[Subregion]
        Stage-3 subregions, ordered by centre MFR (hottest first).
    w_size, gaussian_window : float
        IFR parameters (defaults DEFAULT_W_SIZE, DEFAULT_GAUSSIAN_WINDOW).

    Returns
    -------
    traces : list[np.ndarray]
        C single-channel traces, each (K,) float32, normalized per electrode,
        in the same order as subregions (index 0 = hottest subregion).
    fs_ifr : float = 1 / w_size [Hz].

    Raises
    ------
    ValueError
        If subregions is empty.
    """
    if len(subregions) == 0:
        raise ValueError("subregions is empty; nothing to extract")
    traces: List[np.ndarray] = []
    fs_ifr = None
    for s in subregions:
        r, fs_ifr = subregion_ifr(inv, s, w_size, gaussian_window)
        traces.append(r)
    return traces, float(fs_ifr)


def subregion_ifrs(inv: PtrainInventory, subregions: List[Subregion],
                   w_size: float = DEFAULT_W_SIZE,
                   gaussian_window: float = DEFAULT_GAUSSIAN_WINDOW) -> Tuple[np.ndarray, float]:
    """Mode 1: stack per-subregion normalized IFRs into a (C, K) multichannel trace.

    This is the STACKED VIEW of subregion_single_channel_traces: channel c is
    subregions[c] (already ordered by centre MFR, so channel 0 = hottest), and
    subregion_ifrs(...)[0] == np.stack(subregion_single_channel_traces(...)[0]).

    Returns
    -------
    ifr_mc : (C, K) float32.
    fs_ifr : float = 1 / w_size [Hz].
    """
    traces, fs_ifr = subregion_single_channel_traces(
        inv, subregions, w_size, gaussian_window)
    return np.stack(traces, axis=0).astype(np.float32), float(fs_ifr)


def whole_culture_ifr(inv: PtrainInventory,
                      w_size: float = DEFAULT_W_SIZE,
                      gaussian_window: float = DEFAULT_GAUSSIAN_WINDOW) -> Tuple[np.ndarray, float]:
    """Whole-culture single-channel IFR (1, K), normalized per firing electrode.

    Pools ALL FIRING electrodes (spike count > 0). NO theta / MFR filter is applied
    here (a low-rate firing electrode still contributes); only strictly silent
    electrodes (0 spikes) are excluded. The pooled IFR is divided by the number of
    firing electrodes.

    Returns
    -------
    ifr : (1, K) float32.
    fs_ifr : float = 1 / w_size [Hz].

    Raises
    ------
    ValueError
        If no electrode fired at all.
    """
    from generate_burst_data import compute_ifr_trace  # lazy (see module note)
    firing = [k for k in sorted(inv.spikes) if inv.spikes[k].size > 0]
    if not firing:
        raise ValueError("no firing electrodes (all silent); cannot form whole-culture IFR")
    params = _ifr_params(inv.T_rec, w_size, gaussian_window)
    ifr, fs_ifr = compute_ifr_trace(_spike_seconds(inv, firing), params)
    return (ifr / float(len(firing))).astype(np.float32).reshape(1, -1), fs_ifr


# --------------------------------------------------------------------------- #
# Stage 5 -- public entry point
# --------------------------------------------------------------------------- #
#
# extract_channel_subsets composes Stages 1->4 (load -> geometry/MFR ->
# partition -> IFR) into a single call, with a UNIFORM output contract that lets
# all three delivery modes plug into build_traces identically:
#
#     traces, fs_ifr = extract_channel_subsets(folder, mode=...)
#
# where traces is a LIST whose length is the number of TRAINING SAMPLES this one
# recording contributes, and each element is that sample's trace:
#
#   mode="multichannel"       -> [ (C, K) ]           1 sample,  in_channels = C
#   mode="per_region_single"  -> [ (K,), (K,), ... ]  C samples, in_channels = 1
#   mode="whole_culture"      -> [ (K,) ]             1 sample,  in_channels = 1
#
# Element ndim encodes channel-ness: 2-D == multichannel sample (C, K); 1-D ==
# single-channel sample (K,). The caller does:
#     all_traces.extend(traces)
#     all_conditions.extend([condition] * len(traces))
# so the per-recording sample multiplication (1 or C) is handled uniformly.
#
# CAUTION (per_region_single): the C traces from one recording are NOT
# independent -- they share the recording and its condition. The train/val/test
# split MUST therefore be per-recording, not per-trace, or the C sibling traces
# leak across splits. This is enforced at the wiring stage, not here.

MODES: Tuple[str, ...] = ("multichannel", "per_region_single", "whole_culture")


@dataclass(frozen=True)
class ExtractionDiagnostics:
    """Side information for inspection / the Stage-6 electrode-map plots.

    subregions / discarded / coords / mfrs are empty for whole_culture mode
    (which needs neither geometry nor partition).
    """

    mode: str
    fs_ifr: float
    n_present: int                          # electrodes present in the folder
    subregions: Tuple[Subregion, ...]       # () for whole_culture
    discarded: Tuple[int, ...]              # sub-theta electrodes; () for whole_culture
    coords: Dict[int, Tuple[int, int]]      # present electrode -> (row, col); {} for whole_culture
    mfrs: Dict[int, float]                  # present electrode -> MFR [Hz]; {} for whole_culture
    index_base: int
    grid_width: int
    n_samples: int
    T_rec: float


def extract_channel_subsets(
    folder: str,
    mode: str = "multichannel",
    n_subsets: int = 9,
    electrodes_per_subset: int = 9,
    mfr_threshold: float = 0.1,
    fs_raw: float = DEFAULT_FS_RAW,
    index_base: int = 0,
    grid_width: int = GRID_WIDTH,
    w_size: float = DEFAULT_W_SIZE,
    gaussian_window: float = DEFAULT_GAUSSIAN_WINDOW,
    return_diagnostics: bool = False,
):
    """Extract IFR traces from a folder of ptrain_<idx>.mat spike rasters.

    Parameters
    ----------
    folder : str
        Directory containing ptrain_<idx>.mat files (one per electrode).
    mode : str
        One of MODES: "multichannel" (C, K); "per_region_single" (C separate
        (K,) single-channel traces); "whole_culture" (single (K,) pooled trace).
    n_subsets, electrodes_per_subset, mfr_threshold : partition parameters
        C, E, theta (ignored for whole_culture).
    fs_raw : float
        Raw sampling rate [Hz] for sample-index -> second conversion.
    index_base : int
        Electrode index base (0 or 1). Range-checked for the partition modes.
    grid_width : int
        Electrode grid width (square). Default GRID_WIDTH (48).
    w_size, gaussian_window : float
        IFR bin width Delta_t [s] and smoothing sigma_smooth [s].
    return_diagnostics : bool
        If True, also return an ExtractionDiagnostics.

    Returns
    -------
    traces : list[np.ndarray]
        One element per training sample (see module note for shapes).
    fs_ifr : float = 1 / w_size [Hz].
    diagnostics : ExtractionDiagnostics, only if return_diagnostics is True.

    Raises
    ------
    ValueError
        If mode is not in MODES (or on empty/invalid inputs from the stages).
    GeometryError
        If an electrode index maps outside the grid (wrong index_base).
    InsufficientElectrodesError
        If the partition modes cannot form C disjoint size-E subregions.
    """
    if mode not in MODES:
        raise ValueError("mode must be one of %r, got %r" % (MODES, mode))

    inv = load_ptrain_folder(folder, fs_raw=fs_raw, index_base=index_base)

    if mode == "whole_culture":
        ifr, fs_ifr = whole_culture_ifr(inv, w_size, gaussian_window)
        traces = [ifr.reshape(-1).astype(np.float32)]          # (K,) single sample
        if return_diagnostics:
            diag = ExtractionDiagnostics(
                mode=mode, fs_ifr=fs_ifr, n_present=len(inv.spikes),
                subregions=(), discarded=(), coords={}, mfrs={},
                index_base=index_base, grid_width=grid_width,
                n_samples=inv.n_samples, T_rec=inv.T_rec)
            return traces, fs_ifr, diag
        return traces, fs_ifr

    # partition modes: geometry range check -> coords -> MFR -> partition
    indices = inv.indices
    validate_grid(indices, width=grid_width, base=index_base)   # wrong-base guard
    idx_arr = np.asarray(indices, dtype=np.int64)
    rows, cols = _rowcol(idx_arr, grid_width, index_base)
    coords = {int(i): (int(r), int(c))
              for i, r, c in zip(indices, rows.tolist(), cols.tolist())}
    mfrs = mean_firing_rates(inv, inv.T_rec)
    subs, discarded = partition_subregions(
        coords, mfrs, n_subsets=n_subsets,
        electrodes_per_subset=electrodes_per_subset, mfr_threshold=mfr_threshold)

    if mode == "multichannel":
        mc, fs_ifr = subregion_ifrs(inv, subs, w_size, gaussian_window)
        traces = [mc]                                          # one (C, K) sample
    else:  # per_region_single
        chans, fs_ifr = subregion_single_channel_traces(inv, subs, w_size, gaussian_window)
        traces = list(chans)                                   # C (K,) samples

    if return_diagnostics:
        diag = ExtractionDiagnostics(
            mode=mode, fs_ifr=fs_ifr, n_present=len(inv.spikes),
            subregions=tuple(subs), discarded=tuple(discarded),
            coords=coords, mfrs=mfrs, index_base=index_base, grid_width=grid_width,
            n_samples=inv.n_samples, T_rec=inv.T_rec)
        return traces, fs_ifr, diag
    return traces, fs_ifr
