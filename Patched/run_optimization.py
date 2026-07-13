#!/usr/bin/env python3
"""
run_optimization.py
===================

THE DRIVER for the contrastive MEA phenotype pipeline (Topic 1 augmentation,
Topic 2 backbone, Topic 3 optimization). One script that loads the data, trains
/ optimizes the architecture, and saves the trained architecture.

Pure orchestration (directive 2): this file contains NO science. It decides the
ORDER of the stages, runs the pre-flight checks, and wires artifacts to disk.
Every scientific decision lives in the already-tested modules it calls:

    preprocessing_cache.cache_traces / load_cached_traces  persist traces ONCE
    data_splits.make_time_segment_splits                   leakage-free splits
    search.search_architecture / search_training /
           retune_architecture / search_regularization      the HPO phases
    train.train                                            the ONE trainer
    evaluate.evaluate_and_plot                             held-out TEST scoring
    checkpoint.save_checkpoint                             self-describing .pt

Pipeline
--------
    resolve config -> resolve device -> seed -> PRE-FLIGHTS -> cache traces
      -> time-segment splits
      -> [PHASE 1: architecture] -> [PHASE 2: training HPs] -> [re-tune]
      -> [REGULARIZATION: dropout + weight decay]
      -> FINAL: train N_s models -> held-out TEST evaluation -> artifacts

Every bracketed stage is skipped by --skip-search, which reduces the driver to
"load the data, train the configured architecture, evaluate it, save it".

Notation (carried in full; symbols introduced at first use)
-----------------------------------------------------------
    C     : number of phenotype classes; labels in {0, ..., C-1}
    N_s   : cfg.train.n_seeds, models trained per configuration
    B_c   : cfg.train.windows_per_condition, source windows drawn from EACH
            class per batch
    P, N  : cfg.data.augmentation.n_positives / n_negatives, the profile-
            PRESERVING and profile-DESTROYING surrogates built per source window
    M     : rows in one embedding batch,  M = C * B_c * (1 + P + N)
    W     : window length in samples, W = round(cfg.data.window_s * fs)
    f_s   : sampling rate [Hz], resolved from the DATA (never from the config's
            AugmentationConfig.fs, which is a placeholder)
    E     : embedding dimension, cfg.backbone.embedding_size
    ARI_e : adjusted Rand index on the VALIDATION split at epoch e

Artifacts
---------
    <out_dir>/<experiment_name>/
      config_input.json       the config as resolved (file + CLI), before search
      config_best.json        the config after every search phase
      results.json            the deliverable
      figures/
        pdp_phase1_arch.png, pdp_phase2_train.png, pdp_retune_arch.png,
        pdp_regularization.png, embedding_test_seed_<n>.png
      checkpoints/
        seed_<n>/{last,best}.pt      resumable, written DURING the final train
        final_seed_<n>.pt            self-describing, best-epoch weights
        best_model.pt                [ADDED] the deployable model (see below)

Additions beyond the behaviour described in 02_TECHNICAL.md / 03_USAGE.md
--------------------------------------------------------------------------
Each is marked [ADDED] at its definition. They are additions, not changes: no
tested module is modified.

  [ADDED 1] DROPOUT IS PINNED TO 0 FOR THE WHOLE SEARCH. Decision 11 says
      dropout is tuned ONLY in the regularization stage. search.py enforces that
      in config_from_arch_point (which pins dropout=0.0) but NOT in
      config_from_train_point, which inherits dropout from the base config. So a
      user config with backbone.dropout > 0 would silently run phase 1 at
      dropout 0 and phase 2 at dropout > 0 -- two phases under different
      regularization. The driver pins it to 0 across phases 1, 2 and the re-tune,
      and lets the regularization stage set the final value.

  [ADDED 2] best_model.pt, SELECTED ON VALIDATION, NEVER ON TEST. Of the N_s
      final models, the one with the highest best-epoch VALIDATION ARI is copied
      to best_model.pt. Selecting on the test ARI would leak the held-out split
      into model selection and destroy the honesty of the reported test number.

  [ADDED 3] A DATA FINGERPRINT GUARDS THE TRACE CACHE. cache_traces skips any
      trace whose <name>.npz already exists. That is what makes an HPO run pay
      the trace cost once -- but it also means that changing the data source
      (e.g. synthetic_duration_s, or switching to numpy mode) while pointing at
      the SAME cache_dir would silently reuse the STALE traces. The driver writes
      a fingerprint of the data-source config into the cache and refuses to
      proceed on a mismatch, naming the fix (--overwrite-cache or a new
      --cache-dir).

  [ADDED 4] MODEL SIZES ARE COUNTED ON THE META DEVICE. The size pre-flight must
      not allocate the memory it exists to warn about. Building each corner under
      torch.device("meta") allocates zero bytes; verified to give parameter
      counts identical to a real construction.

  [ADDED 5] BOTH BLOCK FAMILIES ARE REPORTED AT EACH CORNER. The ResNet family
      (block_family=0) is ~3x heavier than ResNeXt (block_family=1) at the same
      (depth, width), and the search samples BOTH. Reporting only one family
      understates the worst corner the search will actually visit.

  [ADDED 6] FINAL SEEDS COME FROM A DISJOINT BLOCK. A search trial t uses seeds
      [s0 + t*N_s, s0 + t*N_s + N_s). The final models use
      s0 + FINAL_SEED_OFFSET + n, which cannot collide with any trial's block, so
      the final fit is not scored on the very draws that selected the config.

  [ADDED 7] STALE-RESUME GUARD. train() RESUMES automatically whenever a
      last.pt exists in the checkpoint directory it is given. Re-running with a
      different architecture and the same out_dir would therefore try to load old
      weights into a new model. Unless --resume is passed, the driver clears the
      seed's checkpoint directory first.

  [ADDED 8] data_mode="real" IS WIRED. 02_TECHNICAL.md sec. 15 lists it as a known
      gap ("Not wired into build_traces"). It is now wired through
      data_pipeline.NeuronalTracesProvider; the engine module that exports
      Neuronal_traces is named with --engine-module. Without that flag the branch
      raises a NotImplementedError that names the fix, exactly as before.

  [ADDED 9] EVALUABILITY PRE-FLIGHT. A split can be non-empty yet unscorable: a
      K-means with K = C needs at least C windows, and a silhouette needs at
      least 2 windows in each class. The driver warns per split, per class.

  [ADDED 10] --skip-regularization, and --dry-run reports the budget for the
      stages that will ACTUALLY run.

HPC note (hpc-python-compat): this file is pure ASCII, and every local module in
its import chain (config, backbone, augmentation, data_pipeline,
preprocessing_cache, data_splits, metrics, checkpoint, inference, train,
evaluate, search) is pure ASCII as well. Matplotlib's Agg backend is forced by
evaluate.py, which is imported here before any figure is produced, so the driver
cannot try to open a display.
"""

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import shutil
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from config import ExperimentConfig
from backbone import build_backbone
from preprocessing_cache import TraceSpec, cache_traces, load_cached_traces
from data_pipeline import NumpyTraceProvider, NeuronalTracesProvider
from data_splits import (
    MultiClassSyntheticProvider,
    make_synthetic_specs,
    make_time_segment_splits,
    segment_bounds,
    window_starts,
)
from train import train, set_global_seed, resolve_device, derive_batches_per_epoch
from checkpoint import save_checkpoint
from evaluate import evaluate_and_plot        # also forces the headless Agg backend

__all__ = [
    "build_traces",
    "check_window_feasibility",
    "estimate_model_sizes",
    "estimate_batch_rows",
    "estimate_budget",
    "run_search_phases",
    "run_final",
    "run",
    "load_config",
    "apply_cli_overrides",
    "build_parser",
    "main",
    "FINAL_SEED_OFFSET",
]

_SPLITS = ("train", "val", "test")

# [ADDED 6] Final-run seeds live in a block no search trial can reach. A trial t
# owns [s0 + t*N_s, s0 + t*N_s + N_s); with any realistic budget (t < 1e4,
# N_s < 1e2) the largest seed a search can touch is s0 + 1e6, well below this.
FINAL_SEED_OFFSET = 10_000_000

# AdamW keeps two moment buffers per parameter, each the same dtype as the
# parameter: weights (4 B) + exp_avg (4 B) + exp_avg_sq (4 B) = 12 B / param in
# float32. Gradients add a further 4 B/param transiently; activations are extra
# and depend on M and W, so this is a LOWER bound on peak RAM.
_BYTES_PER_PARAM_ADAMW = 12
_RAM_WARN_GB = 1.0

_CACHE_FINGERPRINT = "data_fingerprint.json"


# --------------------------------------------------------------------------- #
# JSON helpers (numpy-safe, non-finite-safe, ASCII on disk)
# --------------------------------------------------------------------------- #
def _to_jsonable(obj):
    """Recursively convert an object into something json.dump can write.

    Two hazards this closes:
      * skopt returns numpy scalars (np.int64 / np.float64) inside res.x and
        res.func_vals; json.dump raises TypeError on them.
      * A FAILED trial carries NaN (mean / std / eff_rank). json.dump would emit
        a bare NaN token, which is NOT valid JSON and which many parsers reject.
        Non-finite floats become null instead.
    """
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return _to_jsonable(float(obj))
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _write_json_ascii(obj, path):
    """Write JSON as pure ASCII (HPC-safe artifact), creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii") as fh:
        json.dump(_to_jsonable(obj), fh, indent=2, ensure_ascii=True,
                  allow_nan=False)
    return path


def _deep_copy_cfg(cfg):
    """An INDEPENDENT ExperimentConfig. ExperimentConfig has no .copy(), and a
    shallow copy would SHARE the nested dataclasses, so a later stage mutating
    cfg.train would corrupt the config the earlier stage was scored under. This
    is the same tested round-trip search._deep_copy_cfg uses."""
    return ExperimentConfig.from_dict(cfg.to_dict())


# --------------------------------------------------------------------------- #
# config resolution:  dataclass defaults -> JSON file -> CLI flags
# --------------------------------------------------------------------------- #
def load_config(path=None):
    """ExperimentConfig from a (partial) JSON file, or the dataclass defaults.

    NOTE: the DEFAULTS ARE INFEASIBLE by design of the geometry, not by accident:
    window_s = 200 s against a 600 s recording split 60/20/20 leaves val and test
    segments of 120 s, so both get ZERO windows. check_window_feasibility catches
    this before any work is done. Start from config_example.json.
    """
    if path is None:
        return ExperimentConfig()
    return ExperimentConfig.from_json(path)


def apply_cli_overrides(cfg, args):
    """Apply the CLI overrides on top of the file/defaults.

    Every override goes through dataclasses.replace rather than attribute
    assignment, because replace RE-RUNS __post_init__ and therefore re-validates.
    A plain assignment (cfg.train.max_epochs = 5) would skip validation entirely.
    """
    d, t, r = {}, {}, {}

    if getattr(args, "data_mode", None):
        d["data_mode"] = args.data_mode
    if getattr(args, "npz_specs", None):
        d["npz_specs"] = args.npz_specs
    if getattr(args, "specs_json", None):
        d["specs_json"] = args.specs_json
    if getattr(args, "window_s", None) is not None:
        d["window_s"] = float(args.window_s)

    if getattr(args, "n_seeds", None) is not None:
        t["n_seeds"] = int(args.n_seeds)
    if getattr(args, "max_epochs", None) is not None:
        t["max_epochs"] = int(args.max_epochs)

    if getattr(args, "device", None):
        r["device"] = args.device
    if getattr(args, "out_dir", None):
        r["out_dir"] = args.out_dir
    if getattr(args, "cache_dir", None):
        r["cache_dir"] = args.cache_dir
    if getattr(args, "experiment_name", None):
        r["experiment_name"] = args.experiment_name
    if getattr(args, "seed", None) is not None:
        r["seed"] = int(args.seed)
    if getattr(args, "num_workers", None) is not None:
        r["num_workers"] = int(args.num_workers)

    if d:
        cfg.data = replace(cfg.data, **d)
    if t:
        cfg.train = replace(cfg.train, **t)
    if r:
        cfg.runtime = replace(cfg.runtime, **r)
    cfg.validate()
    return cfg


# --------------------------------------------------------------------------- #
# data:  provider -> cache -> traces
# --------------------------------------------------------------------------- #
def _data_fingerprint(cfg, specs):
    """[ADDED 3] A stable hash of everything that determines the CACHED TRACES.

    cache_traces(overwrite=False) SKIPS any trace whose <name>.npz already
    exists -- which is exactly what makes a 753-trial study pay the trace cost
    once. The flip side is that it never notices when the trace that name refers
    to has CHANGED. Bump synthetic_duration_s, or repoint npz_specs at a
    different cohort, while keeping the same cache_dir, and the run would train
    on the OLD traces without a word. Fingerprinting the data-source config and
    the spec list turns that silent corruption into a loud refusal.
    """
    payload = {
        "data_mode": cfg.data.data_mode,
        "synthetic_n_per_class": list(cfg.data.synthetic_n_per_class),
        "synthetic_duration_s": float(cfg.data.synthetic_duration_s),
        "synthetic_fs": float(cfg.data.synthetic_fs),
        "npz_specs": str(cfg.data.npz_specs),
        "specs_json": str(cfg.data.specs_json),
        "seed": int(cfg.runtime.seed),
        "specs": [{"name": s["name"], "condition": int(s["condition"]),
                   "args": [str(a) for a in s["args"]]} for s in specs],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(blob.encode("ascii")).hexdigest()


def _check_cache_fingerprint(cache_dir, fingerprint, overwrite):
    cache_dir = Path(cache_dir)
    fp_path = cache_dir / _CACHE_FINGERPRINT
    if overwrite or not fp_path.exists():
        return
    try:
        with open(fp_path, "r", encoding="utf-8") as fh:
            old = json.load(fh).get("fingerprint")
    except Exception:
        return
    if old is not None and old != fingerprint:
        raise ValueError(
            "STALE TRACE CACHE at %s.\n"
            "The cached traces were produced by a DIFFERENT data configuration "
            "(fingerprint %s), but this run asks for %s. cache_traces() skips any "
            "trace whose .npz already exists, so continuing would silently train "
            "on the OLD traces.\n"
            "Fix: pass --overwrite-cache to recompute them, or point --cache-dir "
            "at a fresh directory."
            % (cache_dir, old[:12], fingerprint[:12]))


def _read_specs_json(path, required_keys, what):
    p = Path(path)
    if not str(path):
        raise ValueError("data_mode requires a specs file, but none was given.")
    if not p.exists():
        raise FileNotFoundError("%s specs file not found: %s" % (what, p))
    with open(p, "r", encoding="utf-8") as fh:
        recs = json.load(fh)
    if not isinstance(recs, list) or not recs:
        raise ValueError("%s must be a non-empty JSON list of records: %s"
                         % (what, p))
    for i, rec in enumerate(recs):
        missing = [k for k in required_keys if k not in rec]
        if missing:
            raise ValueError(
                "%s record %d is missing key(s) %r. Required schema: %r. Got: %r"
                % (what, i, missing, list(required_keys), sorted(rec.keys())))
    return recs, p.parent


def build_traces(cfg, overwrite_cache=False, engine_module=None,
                 engine_kwargs=None, verbose=False):
    """Resolve the data source, cache every trace ONCE, and load them back.

    A provider is any callable  provider(*args) -> (trace (K,), fs float).
    Providers are INJECTED, never hard-coded, which is what lets every downstream
    module be tested without .mat files.

    Returns
    -------
    traces     : list of (K_i,) float32 arrays, in manifest order
    conditions : list of int phenotype labels, aligned with traces
    fs         : float, the ONE sampling rate shared by every trace. THIS is the
                 fs that gets injected into the augmentation config at split time
                 (DataConfig.resolved_augmentation); the fs stored in the config's
                 nested AugmentationConfig is only a placeholder.
    """
    mode = cfg.data.data_mode
    cache_dir = Path(cfg.runtime.cache_dir)

    if mode == "synthetic":
        n_per_class = tuple(int(n) for n in cfg.data.synthetic_n_per_class)
        provider = MultiClassSyntheticProvider(
            n_classes=len(n_per_class),
            duration_s=float(cfg.data.synthetic_duration_s),
            fs=float(cfg.data.synthetic_fs),
            seed=int(cfg.runtime.seed),
        )
        specs = make_synthetic_specs(n_per_class)

    elif mode == "numpy":
        recs, base_dir = _read_specs_json(
            cfg.data.npz_specs, ("name", "condition", "path"), "npz_specs")
        provider = NumpyTraceProvider()
        specs = []
        for rec in recs:
            path = Path(rec["path"])
            if not path.is_absolute():
                path = base_dir / path          # relative to the specs file
            if not path.exists():
                raise FileNotFoundError(
                    "npz_specs record %r points at a missing file: %s"
                    % (rec["name"], path))
            specs.append(TraceSpec(rec["name"], rec["condition"], (str(path),)))

    elif mode == "real":
        # [ADDED 8] the branch 02_TECHNICAL.md 15 lists as a known gap.
        if not engine_module:
            raise NotImplementedError(
                "data_mode='real' loads traces through the group's MEA engine "
                "(the function Neuronal_traces), which is NOT part of this "
                "repository, so the driver cannot import it by guessing.\n"
                "Fix (either one):\n"
                "  (a) pass --engine-module <module_name>, where that module is "
                "importable on PYTHONPATH and exports Neuronal_traces(Char_folder, "
                "Char_base, w_size, Gaussian_window, t_rec, Visible); or\n"
                "  (b) pre-compute the traces once to .npz "
                "(keys: ifr_trace, fs_ifr) and use data_mode='numpy', which is "
                "simpler and gets the caching for free.")
        recs, _base = _read_specs_json(
            cfg.data.specs_json, ("folder", "base", "condition"), "specs_json")
        try:
            eng = importlib.import_module(str(engine_module))
        except Exception as ex:
            raise ImportError(
                "could not import --engine-module %r (%s: %s). It must be on "
                "PYTHONPATH." % (engine_module, type(ex).__name__, ex))
        if not hasattr(eng, "Neuronal_traces"):
            raise AttributeError(
                "module %r does not export Neuronal_traces." % (engine_module,))
        provider = NeuronalTracesProvider(eng.Neuronal_traces,
                                          **(engine_kwargs or {}))
        specs = [
            TraceSpec(rec.get("name", rec["base"].rstrip("_")),
                      rec["condition"], (rec["folder"], rec["base"]))
            for rec in recs
        ]

    else:                                       # unreachable: DataConfig validates
        raise ValueError("unknown data_mode %r" % (mode,))

    fingerprint = _data_fingerprint(cfg, specs)
    _check_cache_fingerprint(cache_dir, fingerprint, overwrite_cache)

    if verbose:
        print("[run] data_mode=%s: caching %d trace(s) -> %s"
              % (mode, len(specs), cache_dir))
    cache_traces(specs, provider, cache_dir, overwrite=bool(overwrite_cache))
    _write_json_ascii({"fingerprint": fingerprint, "data_mode": mode,
                       "n_traces": len(specs)},
                      cache_dir / _CACHE_FINGERPRINT)

    traces, conditions, fs = load_cached_traces(cache_dir)
    return traces, [int(c) for c in conditions], float(fs)


# --------------------------------------------------------------------------- #
# PRE-FLIGHTS (all three failure modes below actually occurred in development;
# two of them -- the OOM-killer SIGKILL and a zero-window split -- are either
# uncatchable or fail deep inside the first trial, hours in)
# --------------------------------------------------------------------------- #
def check_window_feasibility(cfg, trace_lengths, conditions, fs):
    """Raise unless EVERY split gets at least one window, naming the concrete fix.

    Windows are formed INSIDE a time segment (that is the leakage guarantee), so
    a window longer than the shortest segment yields ZERO windows for that split
    and the run dies. The shipped defaults are infeasible: window_s = 200 s
    against a 600 s recording split 60/20/20 leaves 120 s val and test segments.

    The rule, per split k with fraction f_k and trace of length L_c samples:

        W = round(window_s * fs)  <=  floor(f_k * L_c)      for every trace c

    Reuses data_splits.segment_bounds and data_splits.window_starts, which are
    the SAME functions the real split uses -- so this check cannot drift away
    from the thing it is checking (directive 1).

    Returns a dict: counts per split, and per (split, class) counts.
    """
    W = int(round(float(cfg.data.window_s) * fs))
    if W < 1:
        raise ValueError("data.window_s * fs rounds to < 1 sample.")
    strides = {
        "train": int(round(float(cfg.data.train_stride_s) * fs)),
        "val": int(round(float(cfg.data.eval_stride_s) * fs)),
        "test": int(round(float(cfg.data.eval_stride_s) * fs)),
    }
    for name, s in strides.items():
        if s < 1:
            raise ValueError("the '%s' stride rounds to < 1 sample." % name)

    counts = {k: 0 for k in _SPLITS}
    per_class = {k: {} for k in _SPLITS}
    shortest_s = {k: float("inf") for k in _SPLITS}

    for L, cond in zip(trace_lengths, conditions):
        bounds = segment_bounds(int(L), cfg.data.split_fractions)
        for name, (s, e) in zip(_SPLITS, bounds):
            seg_len = e - s
            shortest_s[name] = min(shortest_s[name], seg_len / float(fs))
            n = len(window_starts(seg_len, W, strides[name]))
            counts[name] += n
            per_class[name][int(cond)] = per_class[name].get(int(cond), 0) + n

    empty = [k for k in _SPLITS if counts[k] == 0]
    if empty:
        k = empty[0]
        raise ValueError(
            "INFEASIBLE WINDOW GEOMETRY: split '%s' would get 0 windows.\n"
            "  window_s   = %.4g s  (W = %d samples at fs = %.4g Hz)\n"
            "  shortest '%s' segment = %.4g s\n"
            "Windows are formed INSIDE a segment, so window_s must not exceed the "
            "shortest segment of every split.\n"
            "Fix: set data.window_s <= %.4g s, or widen split_fractions=%s, or use "
            "longer recordings."
            % (k, cfg.data.window_s, W, fs, k, shortest_s[k],
               min(shortest_s[j] for j in _SPLITS),
               tuple(cfg.data.split_fractions)))

    # [ADDED 9] non-empty is not the same as scorable.
    C = len(set(int(c) for c in conditions))
    for k in ("val", "test"):
        if counts[k] < C:
            warnings.warn(
                "split '%s' has only %d window(s) but K-means is fitted with "
                "K = C = %d: the clustering is degenerate and ARI / AMI will be "
                "meaningless. Reduce window_s or eval_stride_s." % (k, counts[k], C),
                RuntimeWarning)
        for c in range(C):
            n = per_class[k].get(c, 0)
            if n < 2:
                warnings.warn(
                    "split '%s' has %d window(s) for phenotype %d: the silhouette "
                    "is undefined for a class with fewer than 2 members, and ARI "
                    "degrades. Reduce window_s / eval_stride_s, or add traces."
                    % (k, n, c), RuntimeWarning)

    return {"counts": counts, "per_class": per_class, "window_length": W,
            "strides": strides}


def _count_params_meta(bcfg):
    """[ADDED 4] Parameter count WITHOUT allocating the parameters.

    The whole point of this pre-flight is to warn about a config that will not
    fit in RAM; building it for real to count it would trigger the very OOM we
    are trying to predict. Constructing under torch.device("meta") gives tensors
    with a shape but no storage. Verified to agree exactly with a real build.
    """
    with torch.device("meta"):
        model = build_backbone(bcfg)
    return int(sum(p.numel() for p in model.parameters()))


def estimate_model_sizes(cfg, skip_search=False):
    """Parameter counts at the corners of the architecture space (or at the single
    configured architecture when the search is skipped).

    Parameter count is roughly EXPONENTIAL in depth_exponent: the default range
    [3, 6] spans ~700x. The search WILL sample the deep corner during its random
    initialisation phase. A soft OOM raises and is correctly scored FAILED; a
    Linux OOM-KILLER SIGKILL is uncatchable and takes the whole study with it,
    silently -- which is why this is reported up front rather than discovered.

    [ADDED 5] Both block families are evaluated at every corner: ResNet is ~3x
    heavier than ResNeXt at the same (depth, width), and both are sampled.
    """
    corners = {}
    if skip_search:
        n = _count_params_meta(cfg.backbone)
        corners["configured_d%d_w%.2f_blk%d" % (
            cfg.backbone.depth_exponent, cfg.backbone.width_multiplier,
            cfg.backbone.block_family)] = n
    else:
        d_lo, d_hi = cfg.search.depth_exponent_range
        w_lo, w_hi = cfg.search.width_multiplier_range
        e_lo, e_hi = cfg.search.embedding_size_range
        for d in sorted({int(d_lo), int(d_hi)}):
            for w in sorted({float(w_lo), float(w_hi)}):
                for blk in sorted(set(int(b) for b in cfg.search.block_family_choices)):
                    bcfg = replace(cfg.backbone, depth_exponent=d,
                                   width_multiplier=w, block_family=blk,
                                   embedding_size=int(e_hi), dropout=0.0)
                    corners["depth%d_width%.1f_blk%d" % (d, w, blk)] = \
                        _count_params_meta(bcfg)

    max_params = max(corners.values()) if corners else 0
    ram_gb = max_params * _BYTES_PER_PARAM_ADAMW / 1e9
    out = {
        "corners": corners,
        "max_params": int(max_params),
        "max_ram_gb_weights_adamw": float(ram_gb),
        "bytes_per_param": _BYTES_PER_PARAM_ADAMW,
    }
    if ram_gb > _RAM_WARN_GB:
        warnings.warn(
            "the largest architecture in the search space has %.1f M parameters "
            "(~%.2f GB for weights + AdamW moments ALONE, before gradients and "
            "activations). A Linux OOM-killer SIGKILL cannot be caught and will "
            "take the whole study down with no traceback. Narrow "
            "search.depth_exponent_range if the node cannot hold this."
            % (max_params / 1e6, ram_gb), RuntimeWarning)
    return out


def estimate_batch_rows(cfg, n_classes):
    """M = C * B_c * (1 + P + N): the augmentation multiplier, made explicit.

    windows_per_condition is NOT the batch size. Every source window is expanded
    by the augmentation into 1 anchor + P positives + N negatives, so with the
    DEFAULTS (C = 2, B_c = 8, P = N = 30) a batch is 976 rows, not 16 -- a 61x
    multiplier that reading windows_per_condition alone gives no hint of. This is
    the first dial to turn on OOM.

    The formula is the same for both split methods: 'warp_bands' produces exactly
    P positives and N negatives, and 'percentile_mse' splits a pool of P + N
    surrogates by a quantile, so the row count 1 + P + N is identical either way.
    """
    P = int(cfg.data.augmentation.n_positives)
    N = int(cfg.data.augmentation.n_negatives)
    B_c = int(cfg.train.windows_per_condition)
    C = int(n_classes)
    M = C * B_c * (1 + P + N)
    return {"C": C, "windows_per_condition": B_c, "n_positives": P,
            "n_negatives": N, "rows_per_source_window": 1 + P + N,
            "rows_per_batch_M": int(M)}


def estimate_budget(cfg, skip_search=False, skip_regularization=False):
    """Total number of train() runs the pipeline will execute.

    Every search trial costs N_s trainings (the objective averages over seeds),
    and the final stage costs another N_s. The defaults cost 753 runs; that
    number, times the seconds per run, is the single most important quantity in
    the project -- and it is knowable BEFORE any training happens.
    """
    Ns = int(cfg.train.n_seeds)
    do_retune = bool(cfg.search.do_retune_arch)
    p1 = 0 if skip_search else int(cfg.search.n_calls_arch) * Ns
    p2 = 0 if skip_search else int(cfg.search.n_calls_train) * Ns
    rt = 0 if (skip_search or not do_retune) else int(cfg.search.n_calls_arch) * Ns
    rg = 0 if (skip_search or skip_regularization) else \
        int(cfg.regularization.n_calls) * Ns
    fin = Ns
    return {
        "phase1_arch": p1,
        "phase2_train": p2,
        "retune_arch": rt,
        "regularization": rg,
        "final": fin,
        "TOTAL_train_runs": int(p1 + p2 + rt + rg + fin),
        "n_seeds": Ns,
    }


# --------------------------------------------------------------------------- #
# small shared helpers
# --------------------------------------------------------------------------- #
def _best_finite(history, key):
    """max over epochs of history[key], NaN-safe; -inf when nothing is finite."""
    best = float("-inf")
    for h in history:
        v = float(h.get(key, float("nan")))
        if np.isfinite(v) and v > best:
            best = v
    return best


def _cast_arch(a):
    """skopt hands back numpy scalars; the configs (and JSON) want python types."""
    return {
        "depth_exponent": int(a["depth_exponent"]),
        "width_multiplier": float(a["width_multiplier"]),
        "block_family": int(a["block_family"]),
        "embedding_size": int(a["embedding_size"]),
    }


def _prepare_ckpt_dir(path, resume):
    """[ADDED 7] Guard against a SILENT STALE RESUME.

    train() resumes automatically whenever the ckpt_dir it is handed contains a
    last.pt. That is exactly right for an interrupted run -- and exactly wrong
    for a re-run whose architecture changed, where it would try to load the old
    weights into the new model. Unless --resume was asked for, we clear first.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    if not resume:
        for f in ("last.pt", "best.pt"):
            fp = p / f
            if fp.exists():
                fp.unlink()
        for fp in p.glob("epoch_*.pt"):
            fp.unlink()
    return p


def _agg(values):
    """mean / std / values over the final seeds. Population std, matching the
    convention search.evaluate_candidate uses for its per-trial seed spread."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "values": [float(v) for v in values]}
    return {"mean": float(arr.mean()), "std": float(arr.std()),
            "values": [float(v) for v in values]}


# --------------------------------------------------------------------------- #
# the search phases
# --------------------------------------------------------------------------- #
def run_search_phases(cfg, splits, device, fig_dir, skip_regularization=False,
                      verbose=False, on_stage_complete=None):
    """Phase 1 (architecture) -> Phase 2 (training HPs) -> [re-tune] ->
    [regularization], each handing its winner to the next.

    Returns (cfg_best, report). cfg is NOT mutated: a deep copy is carried
    through the phases and returned.

    skopt is imported HERE, not at module scope, so that a --skip-search run
    works on a node where scikit-optimize is not installed.

    on_stage_complete : optional callable(stage: str, cfg: ExperimentConfig).
        Called after EACH phase finishes (its artifacts, e.g. the PDP figure,
        are already on disk by the time it fires). Default None is a no-op, so
        this parameter changes NOTHING about the documented behaviour; it exists
        so a caller (e.g. a Colab wrapper syncing to Drive) can act between
        phases without the driver knowing anything about where its output goes.
        A callback that raises does NOT abort the run: the exception is caught
        and turned into a warning, because a convenience hook must never be able
        to lose real training progress.
    """
    def _fire(stage, c):
        if on_stage_complete is None:
            return
        try:
            on_stage_complete(stage, c)
        except Exception as ex:
            warnings.warn("on_stage_complete(%r) raised %s: %s -> ignored (the "
                          "run continues)." % (stage, type(ex).__name__, ex),
                          RuntimeWarning)
    import search as S                          # lazy: see docstring

    work = _deep_copy_cfg(cfg)
    report = {}

    # [ADDED 1] Decision 11: dropout is tuned ONLY in the regularization stage.
    # search.config_from_arch_point pins dropout = 0 for phase 1, but
    # config_from_train_point does NOT -- it inherits whatever the base config
    # carries. Pinning here keeps phases 1, 2 and the re-tune under one and the
    # same (zero) regularization, so their scores are comparable.
    if float(work.backbone.dropout) != 0.0:
        warnings.warn(
            "backbone.dropout = %.3g in the input config, but dropout is pinned to "
            "0 for the whole search (decision 11: regularization is tuned last, "
            "against the winning configuration). The regularization stage will set "
            "the final value." % work.backbone.dropout, RuntimeWarning)
    work.backbone = replace(work.backbone, dropout=0.0)

    # ---- PHASE 1: architecture, optimizer HELD FIXED ----------------------
    Ns = int(work.train.n_seeds)
    print("[run] PHASE 1: architecture (%d trials x %d seeds)"
          % (work.search.n_calls_arch, Ns))
    res1 = S.search_architecture(work, splits, device, verbose=verbose)
    best_arch = _cast_arch(S.best_arch_dict(res1))
    work.backbone = replace(work.backbone, dropout=0.0, **best_arch)
    work.validate()
    report["phase1_arch"] = {
        "best": best_arch,
        "best_objective": float(np.min(res1.func_vals)),
        "trial_log": list(getattr(res1, "trial_log", [])),
    }
    S.plot_objective_pdp(res1, Path(fig_dir) / "pdp_phase1_arch.png")
    print("[run]   best arch: %r  (objective %+.4f)"
          % (best_arch, report["phase1_arch"]["best_objective"]))
    _fire("phase1_arch", work)

    # ---- PHASE 2: training HPs, architecture FIXED -------------------------
    print("[run] PHASE 2: training HPs (%d trials x %d seeds)"
          % (work.search.n_calls_train, Ns))
    res2 = S.search_training(work, splits, device, best_arch, verbose=verbose)
    best_train = S.best_train_dict(res2)        # betas already converted: b = 1 - u
    work.train = replace(
        work.train,
        margin=float(best_train["margin"]),
        lr=float(best_train["lr"]),
        beta1=float(best_train["beta1"]),
        beta2=float(best_train["beta2"]),
        weight_decay=float(best_train["weight_decay"]),
    )
    work.validate()
    report["phase2_train"] = {
        "best": {k: float(v) for k, v in best_train.items()},
        "best_objective": float(np.min(res2.func_vals)),
        "trial_log": list(getattr(res2, "trial_log", [])),
    }
    S.plot_objective_pdp(res2, Path(fig_dir) / "pdp_phase2_train.png")
    print("[run]   best train HPs: %r  (objective %+.4f)"
          % (report["phase2_train"]["best"],
             report["phase2_train"]["best_objective"]))
    _fire("phase2_train", work)

    # ---- optional RE-TUNE of the architecture under the TUNED optimizer ----
    # search.py reaches get_newspace() ONLY through retune_architecture(), so
    # do_refine on its own has no effect -- say so rather than silently ignoring it.
    if bool(work.search.do_refine) and not bool(work.search.do_retune_arch):
        warnings.warn(
            "search.do_refine=True but search.do_retune_arch=False. In the shipped "
            "search.py the narrowed space (get_newspace) is reachable ONLY through "
            "retune_architecture(), so do_refine alone does nothing. Set "
            "do_retune_arch=true to run the narrowed second pass.", RuntimeWarning)
    if bool(work.search.do_retune_arch):
        print("[run] RE-TUNE: architecture on the narrowed space, under the "
              "tuned optimizer (%d trials x %d seeds)"
              % (work.search.n_calls_arch, Ns))
        res1b = S.retune_architecture(work, splits, device, res1, verbose=verbose)
        best_arch = _cast_arch(S.best_arch_dict(res1b))
        work.backbone = replace(work.backbone, dropout=0.0, **best_arch)
        work.validate()
        report["retune_arch"] = {
            "best": best_arch,
            "best_objective": float(np.min(res1b.func_vals)),
            "trial_log": list(getattr(res1b, "trial_log", [])),
        }
        S.plot_objective_pdp(res1b, Path(fig_dir) / "pdp_retune_arch.png")
        _fire("retune_arch", work)

    # ---- REGULARIZATION: dropout + weight decay, everything else FIXED -----
    # Deliberately LAST (decision 11): regularizing a model that cannot yet fit
    # tells you nothing. weight_decay is searched again here, jointly with
    # dropout, because the two regularizers trade off -- and the value found HERE
    # is the one that wins.
    if not skip_regularization:
        print("[run] REGULARIZATION: dropout + weight decay (%d trials x %d "
              "seeds)" % (work.regularization.n_calls, Ns))
        res3 = S.search_regularization(work, splits, device, verbose=verbose)
        best_reg = S.best_reg_dict(res3)
        work.backbone = replace(work.backbone, dropout=float(best_reg["dropout"]))
        work.train = replace(work.train,
                             weight_decay=float(best_reg["weight_decay"]))
        work.validate()
        report["regularization"] = {
            "best": {k: float(v) for k, v in best_reg.items()},
            "best_objective": float(np.min(res3.func_vals)),
            "trial_log": list(getattr(res3, "trial_log", [])),
        }
        S.plot_objective_pdp(res3, Path(fig_dir) / "pdp_regularization.png")
        print("[run]   best regularization: %r" % (report["regularization"]["best"],))
        _fire("regularization", work)

    return work, report


# --------------------------------------------------------------------------- #
# the final stage: train N_s models, score each on the HELD-OUT TEST split
# --------------------------------------------------------------------------- #
def run_final(cfg_best, splits, device, n_classes, out_dir, resume=False,
              verbose=False, on_stage_complete=None):
    """Train N_s models on the winning config and evaluate each on TEST.

    on_stage_complete : optional callable(stage, cfg). Fired as "final_seed_<n>"
        immediately after EACH seed's checkpoint and test figure are written (not
        only once at the end), so a caller can protect completed seeds one at a
        time. See run_search_phases for the exact contract (default no-op,
        exceptions from the callback are downgraded to a warning).

    The reported spread is over TRAINING SEEDS, not over K-means restarts: it
    answers "would I get this again if I retrained?", which is the question that
    matters. Jittering the clustering seed of a single model would report
    clustering instability instead -- a smaller, less honest number. Every model
    is therefore scored with the SAME eval.kmeans_seed.

    K = C is passed EXPLICITLY, taken from the full label set. If K were inferred
    from the test split alone, a split that happened to lose a rare phenotype
    would score a (C-1)-cluster partition and the number would not be comparable
    with the validation ARI the model was selected on.
    """
    fig_dir = Path(out_dir) / "figures"
    ckpt_root = Path(out_dir) / "checkpoints"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    Ns = int(cfg_best.train.n_seeds)
    per_seed = []

    def _fire(stage, c):
        if on_stage_complete is None:
            return
        try:
            on_stage_complete(stage, c)
        except Exception as ex:
            warnings.warn("on_stage_complete(%r) raised %s: %s -> ignored (the "
                          "run continues)." % (stage, type(ex).__name__, ex),
                          RuntimeWarning)

    print("[run] FINAL: training %d model(s) and evaluating on the HELD-OUT "
          "TEST split" % Ns)

    for n in range(Ns):
        seed = int(cfg_best.runtime.seed) + FINAL_SEED_OFFSET + n   # [ADDED 6]
        ckpt_dir = _prepare_ckpt_dir(ckpt_root / ("seed_%d" % n), resume)  # [ADDED 7]

        t0 = time.time()
        model, history = train(cfg_best, splits.train, splits.val, device,
                               seed=seed, ckpt_dir=str(ckpt_dir), verbose=verbose)
        secs = time.time() - t0
        epochs_run = len(history)
        best_val_ari = _best_finite(history, "ari")
        best_val_sil = _best_finite(history, "silhouette")

        print("[run]   seed %d (%d): %d epoch(s) in %.1f s (%.2f s/epoch) | "
              "best val ARI %.4f"
              % (n, seed, epochs_run, secs,
                 secs / max(1, epochs_run), best_val_ari))

        ev = evaluate_and_plot(
            model, splits.test, device,
            out_path=str(fig_dir / ("embedding_test_seed_%d.png" % n)),
            seed=int(cfg_best.eval.kmeans_seed),      # SAME K-means seed for all
            n_clusters=int(n_classes),                # K = C from the FULL label set
            eval_cfg=cfg_best.eval,
            title=("HELD-OUT TEST embeddings, training seed %d\n"
                   "colour = the SAME seeded full-D K-means labels used for the "
                   "metric; marker = true phenotype" % n),
        )

        final_ckpt = ckpt_root / ("final_seed_%d.pt" % n)
        save_checkpoint(
            final_ckpt,
            config=cfg_best, model=model,
            epoch=epochs_run,
            best_metric={"val_ari": best_val_ari, "val_silhouette": best_val_sil,
                         "test_ari": float(ev["ari"])},
            extra={"seed": seed, "history": history,
                   "test": {k: ev[k] for k in ("ari", "ami", "silhouette",
                                               "n_clusters", "n_windows")}},
        )

        per_seed.append({
            "seed": seed,
            "seed_index": n,
            "epochs_run": epochs_run,
            "seconds": float(secs),
            "seconds_per_epoch": float(secs / max(1, epochs_run)),
            "best_val_ari": float(best_val_ari),
            "best_val_silhouette": float(best_val_sil),
            "test_ari": float(ev["ari"]),
            "test_ami": float(ev["ami"]),
            "test_silhouette": float(ev["silhouette"]),
            "test_eff_rank": float(ev["health"]["eff_rank"]),
            "test_n_windows": int(ev["n_windows"]),
            "figure": str(ev["figure"]),
            "checkpoint": str(final_ckpt),
        })
        _fire("final_seed_%d" % n, cfg_best)

    # ---- [ADDED 2] the deployable model, SELECTED ON VALIDATION ------------
    # NOT on test_ari: choosing the model by its held-out score would fold the
    # test split into model selection, and the reported test number would stop
    # being an out-of-sample estimate of anything.
    finite = [s for s in per_seed if np.isfinite(s["best_val_ari"])]
    best_model_path = None
    if finite:
        winner = max(finite, key=lambda s: s["best_val_ari"])
        best_model_path = ckpt_root / "best_model.pt"
        shutil.copy2(winner["checkpoint"], best_model_path)
        print("[run]   best_model.pt <- seed_index %d (best val ARI %.4f; "
              "selected on VALIDATION, never on test)"
              % (winner["seed_index"], winner["best_val_ari"]))

    test = {
        "ari": _agg([s["test_ari"] for s in per_seed]),
        "ami": _agg([s["test_ami"] for s in per_seed]),
        "silhouette": _agg([s["test_silhouette"] for s in per_seed]),
        "eff_rank": _agg([s["test_eff_rank"] for s in per_seed]),
        "per_seed": per_seed,
        "n_seeds": Ns,
        "best_model": str(best_model_path) if best_model_path else None,
        "best_model_selected_on": "validation ARI (never on test)",
    }
    print("[run] TEST  ARI %.4f +/- %.4f | AMI %.4f +/- %.4f | silhouette "
          "%.4f +/- %.4f  (over %d training seeds)"
          % (test["ari"]["mean"], test["ari"]["std"],
             test["ami"]["mean"], test["ami"]["std"],
             test["silhouette"]["mean"], test["silhouette"]["std"], Ns))
    return test


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def run(cfg, args, on_stage_complete=None):
    """Execute the whole pipeline. Returns the results dict (also on disk).

    on_stage_complete : optional callable(stage: str, cfg: ExperimentConfig),
        threaded through to run_search_phases and run_final so it fires after
        EVERY phase and after every final seed, not just once at the end. This
        parameter is entirely additive: default None reproduces the documented
        behaviour exactly (nothing calls it, nothing changes). It exists so a
        caller can act between stages -- e.g. mirroring runtime.out_dir to
        Google Drive after each phase, which is the only real protection
        against losing a multi-hour search to a Colab disconnect. A raising
        callback is caught and downgraded to a warning; it can never abort a
        run that is otherwise succeeding.
    """
    t_start = time.time()

    def _fire(stage, c):
        if on_stage_complete is None:
            return
        try:
            on_stage_complete(stage, c)
        except Exception as ex:
            warnings.warn("on_stage_complete(%r) raised %s: %s -> ignored (the "
                          "run continues)." % (stage, type(ex).__name__, ex),
                          RuntimeWarning)

    out_dir = Path(cfg.runtime.out_dir) / cfg.runtime.experiment_name
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    verbose = bool(getattr(args, "verbose", False))
    skip_search = bool(getattr(args, "skip_search", False))
    skip_reg = bool(getattr(args, "skip_regularization", False))
    dry_run = bool(getattr(args, "dry_run", False))

    # record what was ASKED for, before anything can fail
    cfg.to_json(out_dir / "config_input.json")

    device = resolve_device(cfg.runtime.device)
    set_global_seed(cfg.runtime.seed,
                    deterministic=bool(cfg.runtime.deterministic),
                    torch_threads=int(cfg.runtime.torch_threads))
    if verbose:
        print("[run] experiment=%s device=%s seed=%d out=%s"
              % (cfg.runtime.experiment_name, device, cfg.runtime.seed, out_dir))

    # ---- data ------------------------------------------------------------
    engine_kwargs = None
    if getattr(args, "engine_kwargs", None):
        engine_kwargs = json.loads(args.engine_kwargs)
    traces, conditions, fs = build_traces(
        cfg,
        overwrite_cache=bool(getattr(args, "overwrite_cache", False)),
        engine_module=getattr(args, "engine_module", None),
        engine_kwargs=engine_kwargs,
        verbose=verbose,
    )
    n_classes = len(set(conditions))
    trace_lengths = [int(t.shape[0]) for t in traces]
    _fire("data", cfg)          # the trace cache is populated; worth protecting

    # ---- PRE-FLIGHTS (before any training) --------------------------------
    feas = check_window_feasibility(cfg, trace_lengths, conditions, fs)
    sizes = estimate_model_sizes(cfg, skip_search=skip_search)
    rows = estimate_batch_rows(cfg, n_classes)
    budget = estimate_budget(cfg, skip_search=skip_search,
                             skip_regularization=skip_reg)

    print("[run] %d trace(s), %d phenotype(s), fs = %.4g Hz, W = %d samples "
          "(%.4g s)" % (len(traces), n_classes, fs, feas["window_length"],
                        cfg.data.window_s))
    print("[run] windows: train=%d val=%d test=%d"
          % (feas["counts"]["train"], feas["counts"]["val"],
             feas["counts"]["test"]))
    print("[run] batch rows M = C*B_c*(1+P+N) = %d*%d*(1+%d+%d) = %d rows per batch"
          % (rows["C"], rows["windows_per_condition"], rows["n_positives"],
             rows["n_negatives"], rows["rows_per_batch_M"]))
    print("[run] arch-space model sizes: %s"
          % json.dumps({k: v for k, v in sizes["corners"].items()}))
    print("[run] max %.1f M params (~%.2f GB weights+AdamW)"
          % (sizes["max_params"] / 1e6, sizes["max_ram_gb_weights_adamw"]))
    print("[run] budget: %s" % json.dumps(budget))

    if dry_run:
        print("[dry-run] %d train() runs would be executed. No training performed."
              % budget["TOTAL_train_runs"])
        return {
            "experiment": cfg.runtime.experiment_name,
            "dry_run": True,
            "device": str(device),
            "budget": budget,
            "model_sizes": sizes,
            "batch_rows": rows,
            "n_traces": len(traces),
            "n_classes": n_classes,
            "fs": float(fs),
            "window_length": feas["window_length"],
            "n_windows": feas["counts"],
            "seconds": float(time.time() - t_start),
        }

    # ---- splits (fs is injected into the augmentation config HERE) ---------
    splits = make_time_segment_splits(traces, conditions, fs, cfg.data,
                                      base_seed=int(cfg.runtime.seed))

    # ---- search ------------------------------------------------------------
    report = {}
    if skip_search:
        cfg_best = _deep_copy_cfg(cfg)
        if verbose:
            print("[run] --skip-search: using the configured architecture and "
                  "training HPs as-is (no HPO).")
    else:
        cfg_best, report = run_search_phases(
            cfg, splits, device, fig_dir,
            skip_regularization=skip_reg, verbose=verbose,
            on_stage_complete=on_stage_complete)

    cfg_best.to_json(out_dir / "config_best.json")

    # ---- final train + held-out TEST evaluation ----------------------------
    test = run_final(cfg_best, splits, device, n_classes, out_dir,
                     resume=bool(getattr(args, "resume", False)), verbose=verbose,
                     on_stage_complete=on_stage_complete)

    results = {
        "experiment": cfg.runtime.experiment_name,
        "dry_run": False,
        "device": str(device),
        "seconds": float(time.time() - t_start),
        "skip_search": skip_search,
        "skip_regularization": skip_reg,
        "budget": budget,
        "model_sizes": sizes,
        "batch_rows": rows,
        "n_traces": len(traces),
        "n_classes": n_classes,
        "fs": float(fs),
        "window_length": feas["window_length"],
        "n_windows": feas["counts"],
        "config_best": cfg_best.to_dict(),
        "test": test,
    }
    results.update(report)          # phase1_arch / phase2_train / retune / reg

    res_path = _write_json_ascii(results, out_dir / "results.json")
    print("[run] results -> %s  (%.1f s)" % (res_path, results["seconds"]))
    _fire("results", cfg_best)
    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        prog="run_optimization.py",
        description="Load MEA traces, search / train the 1D-CNN summary network, "
                    "evaluate it on a held-out test split, and save it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None,
                   help="path to a (partial) ExperimentConfig JSON. Omit at your "
                        "peril: the dataclass defaults are geometrically "
                        "INFEASIBLE (window_s=200s vs 120s eval segments).")

    g = p.add_argument_group("data")
    g.add_argument("--data-mode", choices=("synthetic", "numpy", "real"),
                   default=None)
    g.add_argument("--npz-specs", default=None,
                   help="numpy mode: JSON list of {name, condition, path}")
    g.add_argument("--specs-json", default=None,
                   help="real mode: JSON list of {folder, base, condition}")
    g.add_argument("--window-s", type=float, default=None)
    g.add_argument("--overwrite-cache", action="store_true",
                   help="[ADDED] recompute every cached trace (use after changing "
                        "the data source)")
    g.add_argument("--engine-module", default=None,
                   help="[ADDED] real mode: importable module exporting "
                        "Neuronal_traces")
    g.add_argument("--engine-kwargs", default=None,
                   help="[ADDED] real mode: JSON dict of NeuronalTracesProvider "
                        "kwargs, e.g. '{\"w_size\": 0.02, \"t_rec\": 600.0}'")

    g = p.add_argument_group("stages")
    g.add_argument("--skip-search", action="store_true",
                   help="skip ALL HPO phases (arch, train HPs, re-tune, "
                        "regularization) and train the configured architecture")
    g.add_argument("--skip-regularization", action="store_true",
                   help="[ADDED] run phases 1-2 but not the regularization stage")
    g.add_argument("--dry-run", action="store_true",
                   help="resolve the config, run the pre-flights, print the "
                        "budget, and train NOTHING")
    g.add_argument("--resume", action="store_true",
                   help="resume the FINAL training from its last.pt (search "
                        "phases are not resumable)")

    g = p.add_argument_group("overrides")
    g.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--n-seeds", type=int, default=None)
    g.add_argument("--max-epochs", type=int, default=None)
    g.add_argument("--num-workers", type=int, default=None)
    g.add_argument("--out-dir", default=None)
    g.add_argument("--cache-dir", default=None)
    g.add_argument("--experiment-name", default=None)
    g.add_argument("--verbose", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)
    run(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
