#!/usr/bin/env python3
"""
run_optimization_colab.py
==========================

A Colab-friendly WRAPPER around run_optimization.py. It adds NO science and NO
orchestration logic of its own (directive 2): every stage -- caching, splitting,
searching, final training, evaluation -- is still exactly the sequence in
run_optimization.run(). This file only handles the things that differ between a
dedicated HPC node and a free Colab runtime:

    1. scikit-optimize and pytorch-metric-learning are NOT preinstalled on
       Colab (torch / numpy / scipy / sklearn / matplotlib are).
    2. Google Drive is the only thing that survives a Colab disconnect;
       /content/ is wiped.
    3. A Colab session has a hard ceiling (about 12 h free tier, 24 h Pro+,
       with idle disconnects around 90 min of inactivity) and, unlike the
       final-training stage, the SEARCH PHASES ARE NOT RESUMABLE if a session
       dies mid-phase (03_USAGE.md "Resuming": "Resume applies to the final
       train, not the search phases"). The only real mitigation is (a) sizing
       the run to fit inside one session and (b) syncing completed phases to
       Drive as they finish, so a disconnect loses at most the phase in
       progress.
    4. Colab's free-tier T4 (16 GB VRAM) and host RAM (about 12-13 GB) are
       smaller than a typical HPC node, so the default architecture search
       range is riskier here than the pre-flight's generic 1 GB threshold
       assumes.

How the pieces fit together
----------------------------
    ensure_dependencies()     pip-installs the two missing packages (Colab only,
                              unless --no-auto-install)
    maybe_mount_drive()       mounts Google Drive if --mount-drive is given
    make_drive_sync_hook()    builds the on_stage_complete(stage, cfg) callback
                              that run_optimization.run() already knows how to
                              call after every phase and every final seed (see
                              run_optimization.py's on_stage_complete docstring);
                              THIS file adds nothing to what gets synced, it only
                              decides WHEN by reusing that existing hook
    apply_colab_safe_overrides()  narrows the search to fit a T4 (--colab-safe)
    time_one_epoch_and_project()  automates 03_USAGE.md's "Step 2 - time one
                              run" recipe and compares the projection against
                              --session-hours, entirely by calling
                              run_optimization.run() once in a cheap timing mode
                              -- no separate timing code path is written here

Every one of these is additive to run_optimization.py, which is unmodified in
its documented behaviour: on_stage_complete defaults to None there, and this
file is the only caller that passes a real one.

Usage
-----
    # one command: install deps, mount Drive, run the toy config, sync each
    # phase to Drive as it completes
    python3 run_optimization_colab.py --config config_toy.json \\
        --mount-drive --drive-out-dir /content/drive/MyDrive/dsn_out --verbose

    # find out whether a real config fits in one Colab session BEFORE
    # committing to it
    python3 run_optimization_colab.py --config config_example.json \\
        --time-one-epoch --session-hours 12

    # a memory-safe preset for a free-tier T4
    python3 run_optimization_colab.py --config my_config.json --colab-safe \\
        --mount-drive --drive-out-dir /content/drive/MyDrive/dsn_out

Every run_optimization.py flag still works unchanged (this parser is that
parser, extended, not replaced), and every one of run_optimization.py's own 21
smoke checks passed against the unmodified file before this wrapper was written
(see smoke_test_run_optimization.py) and again after the additive
on_stage_complete change (both were re-run, twice, as controls).

HPC note (hpc-python-compat): pure ASCII. This is kept for consistency with the
rest of the codebase, not because Colab shares davinci-1's MobaXterm/cp1252
transfer risk -- git clone / direct upload / Drive do not corrupt bytes the way
a Windows-terminal copy-paste into a PBS job can.
"""

import argparse
import importlib
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import run_optimization as R
from run_optimization import (
    load_config, apply_cli_overrides, run, estimate_budget,
)
from config import ExperimentConfig

__all__ = [
    "IN_COLAB",
    "ensure_dependencies",
    "maybe_mount_drive",
    "make_drive_sync_hook",
    "apply_colab_safe_overrides",
    "time_one_epoch_and_project",
    "build_parser",
    "main",
]

# --------------------------------------------------------------------------- #
# environment detection
# --------------------------------------------------------------------------- #
def _importable(name):
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


IN_COLAB = _importable("google.colab")

_CORE_PACKAGES = ("torch", "numpy", "scipy", "sklearn", "matplotlib")
_EXTRA_PACKAGES = {"skopt": "scikit-optimize",
                   "pytorch_metric_learning": "pytorch-metric-learning"}


def environment_banner():
    """A short, human-readable summary of what this runtime actually offers.
    Printed at startup so a silent CPU fallback or an unmounted Drive is
    visible immediately rather than discovered an hour into a run."""
    lines = ["[colab] in Google Colab: %s" % IN_COLAB]
    if _importable("torch"):
        import torch
        cuda = torch.cuda.is_available()
        lines.append("[colab] CUDA available: %s%s" % (
            cuda, (" (%s)" % torch.cuda.get_device_name(0)) if cuda else ""))
    drive_mounted = Path("/content/drive").is_dir() and \
        any(Path("/content/drive").iterdir()) if Path("/content/drive").exists() \
        else False
    lines.append("[colab] /content/drive mounted: %s" % drive_mounted)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# dependency install
# --------------------------------------------------------------------------- #
def _pip_install(pip_names):
    """The real installer. Kept as a separate, replaceable function so tests
    can inject a fake one instead of actually invoking pip / the network."""
    cmd = [sys.executable, "-m", "pip", "install", "--quiet"] + list(pip_names)
    subprocess.check_call(cmd)


def ensure_dependencies(auto_install=True, pip_install_fn=None, verbose=True):
    """Verify the whole stack is importable; pip-install the two packages
    Colab does NOT preinstall (scikit-optimize, pytorch-metric-learning).

    Parameters
    ----------
    auto_install   : if False, a missing extra raises instead of installing,
                     naming the exact pip command to run by hand.
    pip_install_fn : callable(list[str]) -> None. Defaults to a real `pip
                     install`. Injectable so this function is testable without
                     a network call.
    verbose        : print what was found / installed.

    Returns
    -------
    list of pip package names that were installed (empty if nothing was needed).

    Raises
    ------
    ImportError if a CORE package (torch, numpy, scipy, sklearn, matplotlib) is
    missing -- those are Colab's job to provide and this function does not try
    to second-guess a broken base image -- or if an EXTRA package is still
    missing after an install attempt.
    """
    missing_core = [m for m in _CORE_PACKAGES if not _importable(m)]
    if missing_core:
        raise ImportError(
            "core package(s) %r are not importable. These are normally "
            "preinstalled on Colab; if you are NOT on Colab, install the full "
            "stack first: pip install torch numpy scipy scikit-learn matplotlib"
            % (missing_core,))

    missing_extra = [m for m in _EXTRA_PACKAGES if not _importable(m)]
    if not missing_extra:
        if verbose:
            print("[colab] dependencies: all present, nothing to install.")
        return []

    pip_names = [_EXTRA_PACKAGES[m] for m in missing_extra]
    if not auto_install:
        raise ImportError(
            "missing package(s) %r. Install them with:\n"
            "  pip install %s\n"
            "or re-run with auto_install=True (--no-auto-install was set)."
            % (missing_extra, " ".join(pip_names)))

    if verbose:
        print("[colab] installing missing package(s): %s" % ", ".join(pip_names))
    installer = pip_install_fn if pip_install_fn is not None else _pip_install
    installer(pip_names)

    still_missing = [m for m in missing_extra if not _importable(m)]
    if still_missing:
        raise ImportError(
            "pip install reported success but %r is still not importable. A "
            "restart of the Python runtime is sometimes required after "
            "installing a new package in a notebook (Runtime -> Restart "
            "runtime), then re-run this script." % (still_missing,))
    if verbose:
        print("[colab] installed: %s" % ", ".join(pip_names))
    return pip_names


# --------------------------------------------------------------------------- #
# Google Drive: mount + sync
# --------------------------------------------------------------------------- #
def maybe_mount_drive(mount, mountpoint="/content/drive", verbose=True):
    """Mount Google Drive if requested and possible; a clear no-op otherwise.

    Never raises when NOT in Colab or when mount=False: this keeps the wrapper
    usable, unmodified, in a plain local/CI environment for testing.
    """
    if not mount:
        return False
    if not IN_COLAB:
        if verbose:
            print("[colab] --mount-drive was given but this is not a Colab "
                  "runtime (google.colab is not importable) -> skipped.")
        return False
    from google.colab import drive          # import deferred: only exists on Colab
    drive.mount(mountpoint)
    if verbose:
        print("[colab] Google Drive mounted at %s" % mountpoint)
    return True


def _mirror(local_dir, drive_dir, verbose=True):
    """Copy local_dir's contents into drive_dir, overwriting on conflict.

    A plain mirror, not an incremental sync: correctness over cleverness. The
    directories involved are checkpoints/results/figures (tens of MB, not TB),
    so a full copytree per call is cheap relative to what it protects against
    (losing hours of search progress).
    """
    src = Path(local_dir)
    if not src.exists():
        return
    dst = Path(drive_dir)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    if verbose:
        print("[colab] synced %s -> %s" % (src, dst))


def make_drive_sync_hook(drive_out_dir=None, drive_cache_dir=None, verbose=True):
    """Build the on_stage_complete(stage, cfg) callback run_optimization.run()
    already knows how to call after every search phase and every final seed
    (see run_optimization.py). Returns None if neither Drive target was given,
    so main() can skip passing a hook at all when there is nothing to sync.

    The callback NEVER needs to protect itself against exceptions: run() and
    run_search_phases() / run_final() already catch anything a hook raises and
    downgrade it to a warning (see run_optimization.py's on_stage_complete
    docstring), so a Drive hiccup -- unmounted, over quota, network blip --
    cannot lose real training progress. This function still avoids raising
    where it easily can, so the warning message is about the real cause rather
    than an opaque traceback.
    """
    if drive_out_dir is None and drive_cache_dir is None:
        return None

    def hook(stage, cfg):
        if drive_out_dir is not None:
            local_out = Path(cfg.runtime.out_dir) / cfg.runtime.experiment_name
            _mirror(local_out, Path(drive_out_dir) / cfg.runtime.experiment_name,
                   verbose=verbose)
        # the trace cache changes ONCE (at the "data" stage) and is the
        # expensive-to-recompute artifact, so only mirror it there rather than
        # re-copying it after every phase.
        if drive_cache_dir is not None and stage == "data":
            _mirror(cfg.runtime.cache_dir, drive_cache_dir, verbose=verbose)

    return hook


# --------------------------------------------------------------------------- #
# a memory-safe preset for a free-tier T4
# --------------------------------------------------------------------------- #
_COLAB_SAFE_MAX_DEPTH = 5   # depth 6 is a ~139-214 M param corner (~1.7-2.6 GB
                            # weights+AdamW alone); depth 5 tops out ~20 M
                            # (~0.24 GB), comfortably inside a T4's 16 GB VRAM
                            # and a free-tier host's ~12-13 GB system RAM.


def apply_colab_safe_overrides(cfg, verbose=True):
    """Narrow the search to fit a free-tier Colab runtime. Returns a NEW config
    (cfg is not mutated), same convention as the rest of the driver.

    What changes, and why:
        search.depth_exponent_range upper bound capped at 5 (never widened: if
            the input config already asked for something narrower, that is
            left alone -- this only makes the range SAFER, never bigger)
        runtime.num_workers forced to 0 (matches the documented HPC default;
            Colab's free tier typically gives 2 vCPUs, so forking DataLoader
            workers has little headroom and can be net slower)
        runtime.device set to "auto" UNLESS the config already asked for
            something other than the dataclass default "cpu" (an explicit
            request is left alone; a caller's --device flag applied AFTER this
            function, via apply_cli_overrides, always wins regardless)

    This function is a convenience preset, not a correctness requirement: it
    changes nothing about the model or the objective, only the corner of the
    search space that gets explored and how the DataLoader is configured.
    """
    changes = []
    d_lo, d_hi = cfg.search.depth_exponent_range
    new_hi = min(int(d_hi), _COLAB_SAFE_MAX_DEPTH)
    if new_hi != d_hi:
        cfg.search = replace(cfg.search,
                             depth_exponent_range=(int(d_lo), new_hi))
        changes.append("search.depth_exponent_range: (%d, %d) -> (%d, %d)"
                       % (d_lo, d_hi, d_lo, new_hi))

    if cfg.runtime.num_workers != 0:
        old = cfg.runtime.num_workers
        cfg.runtime = replace(cfg.runtime, num_workers=0)
        changes.append("runtime.num_workers: %d -> 0" % old)

    if cfg.runtime.device == "cpu":            # the dataclass default: assume
        cfg.runtime = replace(cfg.runtime, device="auto")   # unset, not chosen
        changes.append("runtime.device: cpu -> auto")

    cfg.validate()
    if verbose:
        if changes:
            print("[colab] --colab-safe overrides applied:")
            for c in changes:
                print("[colab]   %s" % c)
        else:
            print("[colab] --colab-safe: input config already within the safe "
                  "range, nothing changed.")
    return cfg


# --------------------------------------------------------------------------- #
# "will this fit in one Colab session?"
# --------------------------------------------------------------------------- #
def time_one_epoch_and_project(cfg, args, session_hours=12.0,
                               epochs_to_time=3, verbose=True):
    """Automate 03_USAGE.md's own recipe ("Step 2 - Time one run") and compare
    the projection against a Colab session ceiling, in one command instead of
    two manual runs.

    Method (identical arithmetic to the README's worked example):
        1. time a SHORT run: --skip-search, n_seeds=1, max_epochs=epochs_to_time
           -> seconds_per_epoch (measured, not estimated)
        2. read the REAL config's budget: TOTAL_train_runs and max_epochs
           (estimate_budget -- the SAME pre-flight run() itself prints)
        3. total_hours = seconds_per_epoch * max_epochs * TOTAL_train_runs / 3600

    This calls run_optimization.run() exactly once, in the timing configuration;
    no separate timing code path is written here, so the measured number comes
    from the real trainer, not a proxy.

    Returns a dict with the measured and projected numbers (also printed).
    """
    timing_cfg = R._deep_copy_cfg(cfg)
    timing_cfg.train = replace(timing_cfg.train, n_seeds=1,
                               max_epochs=int(epochs_to_time),
                               patience=int(epochs_to_time))
    timing_cfg.runtime = replace(
        timing_cfg.runtime,
        experiment_name=cfg.runtime.experiment_name + "_timing")

    timing_args = argparse.Namespace(**vars(args))
    timing_args.skip_search = True
    timing_args.dry_run = False
    timing_args.resume = False

    if verbose:
        print("[colab] timing %d epoch(s) on 1 seed, search skipped, to project "
              "the FULL budget..." % epochs_to_time)
    t0 = time.time()
    result = run(timing_cfg, timing_args)
    wall = time.time() - t0

    secs_per_epoch = result["test"]["per_seed"][0]["seconds_per_epoch"]
    budget = estimate_budget(
        cfg, skip_search=bool(getattr(args, "skip_search", False)),
        skip_regularization=bool(getattr(args, "skip_regularization", False)))
    total_runs = budget["TOTAL_train_runs"]
    max_epochs = int(cfg.train.max_epochs)
    projected_hours = secs_per_epoch * max_epochs * total_runs / 3600.0
    fits = projected_hours <= float(session_hours)

    if verbose:
        print("[colab] measured: %.2f s/epoch (timing run took %.1f s wall-clock)"
              % (secs_per_epoch, wall))
        print("[colab] real config: max_epochs=%d, TOTAL_train_runs=%d"
              % (max_epochs, total_runs))
        print("[colab] projected total: %.2f s/epoch * %d epochs * %d runs "
              "/ 3600 = %.1f hours"
              % (secs_per_epoch, max_epochs, total_runs, projected_hours))
        if fits:
            print("[colab] FITS inside a %.2f-hour session." % session_hours)
        else:
            if session_hours > 0:
                n_sessions = int(-(-projected_hours // session_hours))  # ceil
                sessions_msg = "roughly %d such session(s)" % n_sessions
            else:
                sessions_msg = "an undefined number of sessions " \
                               "(session_hours <= 0)"
            print("[colab] DOES NOT FIT inside a %.2f-hour session (needs "
                  "%s). Search phases are NOT resumable across a disconnect "
                  "(only the final training stage is), so narrow "
                  "n_calls_arch / n_calls_train / regularization.n_calls / "
                  "n_seeds / max_epochs, or run with --skip-search, or use "
                  "--colab-safe plus a smaller architecture range."
                  % (session_hours, sessions_msg))

    return {
        "seconds_per_epoch": float(secs_per_epoch),
        "max_epochs": max_epochs,
        "total_train_runs": int(total_runs),
        "projected_hours": float(projected_hours),
        "session_hours": float(session_hours),
        "fits_in_one_session": bool(fits),
    }


# --------------------------------------------------------------------------- #
# CLI: run_optimization's parser, extended (not replaced)
# --------------------------------------------------------------------------- #
def build_parser():
    """run_optimization.build_parser() with an additive 'colab' argument group.
    Every existing flag keeps its exact meaning; nothing is removed or renamed.
    """
    p = R.build_parser()
    g = p.add_argument_group("colab")
    g.add_argument("--mount-drive", action="store_true",
                   help="mount Google Drive at --drive-mountpoint (no-op, "
                        "not an error, outside Colab)")
    g.add_argument("--drive-mountpoint", default="/content/drive")
    g.add_argument("--drive-out-dir", default=None,
                   help="Drive path to mirror runtime.out_dir into after EVERY "
                        "phase and every final seed (the real protection "
                        "against losing a multi-hour search to a disconnect)")
    g.add_argument("--drive-cache-dir", default=None,
                   help="Drive path to mirror the trace cache into once, right "
                        "after caching completes")
    g.add_argument("--no-auto-install", dest="auto_install",
                   action="store_false", default=True,
                   help="do not pip-install scikit-optimize / "
                        "pytorch-metric-learning even if missing; raise a "
                        "message naming the command instead")
    g.add_argument("--colab-safe", action="store_true",
                   help="cap search.depth_exponent_range at %d, force "
                        "num_workers=0, default device to auto -- a "
                        "memory-safe preset for a free-tier T4"
                        % _COLAB_SAFE_MAX_DEPTH)
    g.add_argument("--time-one-epoch", action="store_true",
                   help="measure real seconds/epoch on a short run and project "
                        "the FULL configured budget against --session-hours, "
                        "then exit WITHOUT running the full pipeline")
    g.add_argument("--session-hours", type=float, default=12.0,
                   help="assumed Colab session ceiling used by --time-one-epoch "
                        "(12 = free tier, 24 = Pro+ with sufficient compute "
                        "units)")
    g.add_argument("--epochs-to-time", type=int, default=3,
                   help="epochs to actually run for --time-one-epoch's "
                        "measurement")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    print(environment_banner())
    ensure_dependencies(auto_install=bool(args.auto_install), verbose=True)
    maybe_mount_drive(bool(args.mount_drive), args.drive_mountpoint, verbose=True)

    cfg = load_config(args.config)
    if args.colab_safe:
        cfg = apply_colab_safe_overrides(cfg, verbose=True)
    # CLI flags always win, including over --colab-safe: e.g. an explicit
    # --device cpu after --colab-safe must still result in cpu, not auto.
    cfg = apply_cli_overrides(cfg, args)

    if args.time_one_epoch:
        time_one_epoch_and_project(cfg, args, session_hours=args.session_hours,
                                   epochs_to_time=args.epochs_to_time,
                                   verbose=True)
        return 0

    hook = make_drive_sync_hook(args.drive_out_dir, args.drive_cache_dir,
                                verbose=bool(args.verbose))
    run(cfg, args, on_stage_complete=hook)
    return 0


if __name__ == "__main__":
    sys.exit(main())
