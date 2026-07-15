"""
preprocessing_cache.py
======================

Persist the expensive per-well trace computation ONCE, so the HPO loop (many
trials) and the final run all read the same pre-computed traces from disk
instead of recomputing them every time. Separation of concerns (directive 2):
this module does persistence only. It does not know how a trace is produced --
that is the job of an injectable "provider" callable (directive 1: reuse the
tested providers in data_pipeline.py: SyntheticTraceProvider / MultiClassSynthetic
Provider, NeuronalTracesProvider, NumpyTraceProvider).

A provider is any callable with signature

    provider(*args) -> (trace: np.ndarray of shape (K,), fs: float)

On-disk format (per trace)
--------------------------
    <cache_dir>/<name>.npz  with arrays / scalars:
        ifr_trace : (K,) float32   -- the trace samples
        fs_ifr    : float          -- sampling rate [Hz]
        condition : int            -- phenotype label (0..C-1)
        name      : str            -- unique id (well id / trace id)

The keys ifr_trace / fs_ifr match what NumpyTraceProvider already expects, so a
cached file is ALSO loadable by that provider.

    <cache_dir>/manifest.json : ordered list of entries
        [{"name","condition","fs","length","file"}, ...]

HPC note (hpc-python-compat): pure ASCII. The manifest is written atomically
(temp file + os.replace) so an interrupted run cannot leave a half-written index.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np

__all__ = [
    "TraceSpec",
    "cache_traces",
    "load_cached_traces",
    "manifest_path",
]

_MANIFEST_NAME = "manifest.json"


def manifest_path(cache_dir):
    return Path(cache_dir) / _MANIFEST_NAME


def TraceSpec(name, condition, args):
    """Build one cache spec (a plain dict) describing a single trace to compute.

    Parameters
    ----------
    name      : str   -- unique filesystem-safe id (becomes <name>.npz)
    condition : int   -- phenotype label (0..C-1), stored as metadata
    args      : tuple -- positional args passed to provider(*args)
    """
    return {"name": str(name), "condition": int(condition), "args": tuple(args)}


def _atomic_write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="ascii") as fh:
            json.dump(obj, fh, indent=2)
        os.replace(tmp, path)     # atomic on POSIX
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def cache_traces(specs, provider, cache_dir, overwrite=False):
    """Compute (via provider) and cache each trace in specs; write a manifest.

    Parameters
    ----------
    specs     : list of dicts from TraceSpec(name, condition, args)
    provider  : callable, provider(*args) -> (trace (K,), fs float)
    cache_dir : str / Path
    overwrite : if False (default), a spec whose <name>.npz already exists is
                skipped (the expensive computation is not repeated).

    Returns
    -------
    manifest : list of entry dicts (also written to <cache_dir>/manifest.json)
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    seen_names = set()
    manifest = []
    for spec in specs:
        name = str(spec["name"])
        if name in seen_names:
            raise ValueError("duplicate spec name %r; names must be unique" % name)
        seen_names.add(name)
        condition = int(spec["condition"])
        args = tuple(spec["args"])

        npz_file = cache_dir / (name + ".npz")
        if npz_file.exists() and not overwrite:
            with np.load(npz_file, allow_pickle=True) as data:
                trace = np.ascontiguousarray(data["ifr_trace"], dtype=np.float32)
                fs = float(data["fs_ifr"])
        else:
            trace, fs = provider(*args)
            trace = np.ascontiguousarray(trace, dtype=np.float32)
            fs = float(fs)
            if trace.ndim != 1:
                raise ValueError(
                    "provider for %r returned shape %r; expected 1-D (K,)"
                    % (name, trace.shape))
            np.savez(
                npz_file,
                ifr_trace=trace,
                fs_ifr=np.float64(fs),
                condition=np.int64(condition),
                name=name,
            )

        manifest.append({
            "name": name,
            "condition": condition,
            "fs": fs,
            "length": int(trace.shape[0]),
            "file": npz_file.name,
        })

    _atomic_write_json(manifest, manifest_path(cache_dir))
    return manifest


def load_cached_traces(cache_dir, fs_tol=1e-9):
    """Load all cached traces (in manifest order) and verify a single shared fs.

    Returns
    -------
    traces     : list of (K_i,) float32 np.ndarray  (in manifest order)
    conditions : list of int phenotype labels        (aligned with traces)
    fs         : float, the common sampling rate

    Raises if traces disagree on fs beyond fs_tol (mixing sampling rates would
    make windowing-by-seconds inconsistent).
    """
    cache_dir = Path(cache_dir)
    mpath = manifest_path(cache_dir)
    if not mpath.exists():
        raise FileNotFoundError(
            "no manifest at %s; run cache_traces first" % mpath)
    with open(mpath, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not manifest:
        raise ValueError("manifest is empty at %s" % mpath)

    traces = []
    conditions = []
    fs_values = []
    for entry in manifest:
        npz_file = cache_dir / entry["file"]
        with np.load(npz_file, allow_pickle=True) as data:
            traces.append(np.ascontiguousarray(data["ifr_trace"], dtype=np.float32))
            conditions.append(int(data["condition"]))
            fs_values.append(float(data["fs_ifr"]))

    fs0 = fs_values[0]
    for name, fsv in zip((e["name"] for e in manifest), fs_values):
        if abs(fsv - fs0) > fs_tol:
            raise ValueError(
                "inconsistent sampling rates in cache: %r has fs=%r but the first "
                "trace has fs=%r. All traces must share fs." % (name, fsv, fs0))
    return traces, conditions, float(fs0)
