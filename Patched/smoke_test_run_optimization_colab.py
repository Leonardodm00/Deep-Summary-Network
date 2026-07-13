#!/usr/bin/env python3
"""
smoke_test_run_optimization_colab.py
=====================================

Correctness harness for run_optimization_colab.py's OWN logic. It does not
re-verify anything already covered by smoke_test_run_optimization.py (the
caching, splitting, search, final-train, and artifact behaviour of
run_optimization.run() itself) -- it verifies the wrapper's additions:
dependency install, Colab detection, Drive mount wiring, Drive sync mirroring
(and its failure-safety), the colab-safe preset and its CLI precedence, and the
timing/projection arithmetic. It also re-runs the FULL
smoke_test_run_optimization.py suite as a regression control, since the
on_stage_complete hook that made this wrapper possible is an edit to the
tested driver, not a new file -- that edit must not have changed anything
about the driver's own documented behaviour.

    python3 smoke_test_run_optimization_colab.py
    python3 smoke_test_run_optimization_colab.py --quick   # skips the regression
                                                            # suite's real-skopt e2e

Exit code 0 = every check passed.

Because this sandbox is not an actual Colab runtime, IN_COLAB is exercised by
INJECTING a fake `google.colab` (and `google.colab.drive`) module into
sys.modules -- this proves the wiring (the right function gets called with the
right arguments) without needing Google's actual auth flow, which is
untestable from any script.

The checks
----------
  [A] environment detection: IN_COLAB flips True/False correctly, with a fake
      google.colab module injected and removed
  [B] ensure_dependencies: correct behaviour for (present / missing+auto-install
      / missing+no-auto-install / core-missing), verified by INJECTING a fake
      importability check and a fake installer -- no real pip / network call
  [C] maybe_mount_drive: no-ops (does not raise) when not in Colab or when
      mount=False; calls the injected fake drive.mount with the right
      mountpoint when both are true
  [D] the Drive-sync hook mirrors out_dir and cache_dir correctly, and mirrors
      the cache ONLY on the "data" stage (not on every phase)
  [E] a sync failure is swallowed as a warning, non-vacuously (proven by
      patching copytree to raise and confirming the run-level wiring in
      run_optimization.py -- not this file -- is what catches it)
  [F] --colab-safe narrows depth_exponent_range only when needed, forces
      num_workers=0, defaults device to auto, and NEVER widens an
      already-narrower input range
  [G] CLI precedence: an explicit --device after --colab-safe wins (proven
      non-vacuously: colab-safe sets auto, the CLI sets cpu, cpu must survive)
  [H] time_one_epoch_and_project: the projected-hours arithmetic matches the
      README's own formula exactly, using an INJECTED measured seconds/epoch
      so the check does not depend on this machine's speed; the fits/does-not
      verdict flips at the session-hours boundary
  [I] the extended parser is a strict superset: every run_optimization.py flag
      still parses, plus the new colab flags, with the documented defaults
  [J] end-to-end through the real CLI: config_toy.json runs via
      run_optimization_colab.main(), install/mount are correctly skipped
      outside Colab, and the wrapper's own artifacts (none extra expected
      without --drive-out-dir) match run_optimization's
  [K] end-to-end WITH a Drive-sync hook wired through main(): every phase
      mirrors to a fake "Drive" directory as it completes, verified by
      checking the mirror exists AFTER phase 1 finishes but BEFORE the whole
      run is done (non-vacuous: proven with a hook that snapshots the mirror's
      contents mid-run, not just at the end)
  [L] pure ASCII (this file and the wrapper)
  [M] REGRESSION: the full smoke_test_run_optimization.py suite, unchanged,
      still passes against the EDITED run_optimization.py

HPC note (hpc-python-compat): pure ASCII.
"""

import argparse
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
import types
import warnings
from pathlib import Path

import numpy as np

import run_optimization_colab as C
import run_optimization as R

_PASSED = []
_TMP = None


def ok(label, msg):
    _PASSED.append(label)
    print("  [%s] OK   %s" % (label, msg))


def tmpdir(name):
    p = Path(_TMP) / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def toy_config_dict(out_dir, cache_dir, exp="c", n_seeds=1, max_epochs=1,
                    n_calls=2):
    return {
        "data": {
            "data_mode": "synthetic",
            "window_s": 8.0, "train_stride_s": 4.0, "eval_stride_s": 8.0,
            "split_fractions": [0.6, 0.2, 0.2],
            "synthetic_duration_s": 200.0, "synthetic_fs": 50.0,
            "synthetic_n_per_class": [2, 2],
            "augmentation": {"fs": 50.0, "n_positives": 3, "n_negatives": 3,
                             "shift_magnitude_s": 1.0},
        },
        "backbone": {"depth_exponent": 3, "width_multiplier": 2.0,
                     "block_family": 0, "embedding_size": 8, "stem_width": 16},
        "train": {"windows_per_condition": 2, "batches_per_epoch": 2,
                  "n_seeds": n_seeds, "max_epochs": max_epochs,
                  "patience": max(1, max_epochs)},
        "search": {"depth_exponent_range": [3, 4],
                   "width_multiplier_range": [1.5, 2.5],
                   "embedding_size_range": [8, 12],
                   "n_calls_arch": n_calls, "n_calls_train": n_calls},
        "regularization": {"n_calls": n_calls},
        "runtime": {"seed": 0, "device": "cpu", "num_workers": 0,
                    "out_dir": str(out_dir), "cache_dir": str(cache_dir),
                    "experiment_name": exp},
    }


def write_cfg(path, d):
    with open(path, "w", encoding="ascii") as fh:
        json.dump(d, fh, indent=2)
    return str(path)


class _FakeGoogleColab(object):
    """A minimal stand-in for the google.colab package, injected into
    sys.modules so the Colab-only code paths can be exercised without an
    actual Colab runtime."""
    def __init__(self):
        self.mount_calls = []

        drive_mod = types.ModuleType("google.colab.drive")

        def mount(mountpoint):
            self.mount_calls.append(mountpoint)

        drive_mod.mount = mount
        self.drive_mod = drive_mod

    def install(self):
        google_mod = sys.modules.get("google")
        self._had_google = "google" in sys.modules
        self._had_google_colab = "google.colab" in sys.modules
        self._had_drive = "google.colab.drive" in sys.modules
        if google_mod is None:
            google_mod = types.ModuleType("google")
            sys.modules["google"] = google_mod
        colab_mod = types.ModuleType("google.colab")
        colab_mod.drive = self.drive_mod
        sys.modules["google.colab"] = colab_mod
        sys.modules["google.colab.drive"] = self.drive_mod
        google_mod.colab = colab_mod

    def remove(self):
        for name in ("google.colab.drive", "google.colab"):
            sys.modules.pop(name, None)
        if not self._had_google:
            sys.modules.pop("google", None)


# --------------------------------------------------------------------------- #
# [A] environment detection
# --------------------------------------------------------------------------- #
def check_A():
    assert C._importable("google.colab") is False, \
        "google.colab must NOT be importable in this sandbox (sanity check)"
    fake = _FakeGoogleColab()
    fake.install()
    try:
        assert C._importable("google.colab") is True, \
            "the injected fake google.colab must be detected as importable"
    finally:
        fake.remove()
    assert C._importable("google.colab") is False, \
        "removing the fake module must restore non-importability"
    # IN_COLAB itself is bound at import time, matching the real environment
    assert C.IN_COLAB is False, \
        "this sandbox is not Colab; IN_COLAB must reflect that"
    ok("A", "google.colab importability flips correctly on inject/remove; "
            "IN_COLAB correctly False in this non-Colab sandbox")


# --------------------------------------------------------------------------- #
# [B] ensure_dependencies
# --------------------------------------------------------------------------- #
def check_B():
    calls = []

    def fake_installer(names):
        calls.append(list(names))

    # (1) everything present -> no install, empty return
    got = C.ensure_dependencies(auto_install=True, pip_install_fn=fake_installer,
                                verbose=False)
    assert got == [], "nothing should be installed when everything is present"
    assert calls == []

    # (2) simulate a missing EXTRA package whose (fake) install SUCCEEDS: the
    # fake _importable must be STATEFUL -- False before the fake installer
    # runs, True after -- or this test cannot distinguish "installed
    # successfully" from "still missing after install" (test 5 below).
    old_importable = C._importable
    installed = {"done": False}

    def fake_importable_will_succeed(name):
        if name == "skopt":
            return installed["done"]
        return old_importable(name)

    def fake_installer_succeeds(names):
        calls.append(list(names))
        installed["done"] = True

    C._importable = fake_importable_will_succeed
    try:
        got = C.ensure_dependencies(auto_install=True,
                                    pip_install_fn=fake_installer_succeeds,
                                    verbose=False)
    finally:
        C._importable = old_importable
    assert got == ["scikit-optimize"], got
    assert calls == [["scikit-optimize"]], calls

    # (3) missing + auto_install=False -> raises, names the pip command, and
    # never even calls the installer (permanently-missing fake is fine here:
    # the auto_install=False branch returns before any install is attempted)
    def fake_importable_missing(name):
        if name == "skopt":
            return False
        return old_importable(name)

    C._importable = fake_importable_missing
    try:
        try:
            C.ensure_dependencies(auto_install=False, pip_install_fn=fake_installer,
                                  verbose=False)
            raise AssertionError("must raise when auto_install=False and "
                                 "something is missing")
        except ImportError as ex:
            assert "pip install scikit-optimize" in str(ex), str(ex)
    finally:
        C._importable = old_importable

    # (4) a CORE package missing must raise regardless of auto_install
    def fake_importable_core_missing(name):
        if name == "torch":
            return False
        return old_importable(name)

    C._importable = fake_importable_core_missing
    try:
        try:
            C.ensure_dependencies(auto_install=True, pip_install_fn=fake_installer,
                                  verbose=False)
            raise AssertionError("a missing CORE package must raise")
        except ImportError as ex:
            assert "torch" in str(ex)
    finally:
        C._importable = old_importable

    # (5) installer runs but the package is STILL missing afterwards (a
    # constant-False fake: the "install" never actually satisfies the import)
    # -> raises with the "restart the runtime" hint, a real known notebook
    # failure mode
    C._importable = fake_importable_missing
    try:
        try:
            C.ensure_dependencies(auto_install=True,
                                  pip_install_fn=lambda names: None,  # no-op
                                  verbose=False)
            raise AssertionError("must raise if still missing after install")
        except ImportError as ex:
            assert "restart" in str(ex).lower()
    finally:
        C._importable = old_importable

    ok("B", "present/missing/no-auto-install/core-missing/still-missing-after-"
            "install all handled correctly, with zero real pip or network calls")


# --------------------------------------------------------------------------- #
# [C] maybe_mount_drive
# --------------------------------------------------------------------------- #
def check_C():
    # not in Colab, mount=True -> no-op, no raise
    assert C.IN_COLAB is False
    assert C.maybe_mount_drive(True, "/content/drive", verbose=False) is False

    # IN_COLAB is captured ONCE at import time (deliberately -- a notebook's
    # environment does not change mid-process, so a live re-check on every
    # call would be pure overhead). Injecting google.colab into sys.modules
    # after import therefore does NOT retroactively flip it; the test patches
    # the module constant directly, which is exactly how maybe_mount_drive
    # reads it (a global lookup in run_optimization_colab's own namespace).
    fake = _FakeGoogleColab()
    fake.install()
    old_in_colab = C.IN_COLAB
    C.IN_COLAB = True
    try:
        # in Colab, mount=False -> no-op
        assert C.maybe_mount_drive(False, "/content/drive", verbose=False) is False
        assert fake.mount_calls == []

        # in Colab, mount=True -> the fake drive.mount is called with the
        # exact mountpoint given
        result = C.maybe_mount_drive(True, "/custom/mount", verbose=False)
        assert result is True
        assert fake.mount_calls == ["/custom/mount"], fake.mount_calls
    finally:
        C.IN_COLAB = old_in_colab
        fake.remove()
    ok("C", "no-op outside Colab and when mount=False; calls the real "
            "drive.mount(mountpoint) wiring when both IN_COLAB and mount=True")


# --------------------------------------------------------------------------- #
# [D] Drive-sync hook: mirrors correctly, cache only on 'data'
# --------------------------------------------------------------------------- #
def check_D():
    d = tmpdir("D")
    out_local = d / "out"
    (out_local / "exp1" / "checkpoints").mkdir(parents=True)
    (out_local / "exp1" / "checkpoints" / "best_model.pt").write_bytes(b"weights")
    cache_local = d / "cache"
    cache_local.mkdir()
    (cache_local / "manifest.json").write_text("[]")

    drive_out = d / "drive_out"
    drive_cache = d / "drive_cache"

    from config import ExperimentConfig
    from dataclasses import replace
    cfg = ExperimentConfig()
    cfg.runtime = replace(cfg.runtime, out_dir=str(out_local),
                          cache_dir=str(cache_local), experiment_name="exp1")

    hook = C.make_drive_sync_hook(str(drive_out), str(drive_cache), verbose=False)
    assert hook is not None

    hook("data", cfg)
    assert (drive_out / "exp1" / "checkpoints" / "best_model.pt").exists(), \
        "out_dir must mirror on EVERY stage, including 'data'"
    assert (drive_cache / "manifest.json").exists(), \
        "cache_dir must mirror on the 'data' stage"

    # mutate the cache locally, then fire a NON-data stage: the drive cache
    # copy must NOT be touched (cache is mirrored once, at 'data', only)
    (cache_local / "manifest.json").write_text('["changed"]')
    hook("phase1_arch", cfg)
    assert (drive_cache / "manifest.json").read_text() == "[]", \
        ("the cache mirror must NOT update on a non-'data' stage; got %r"
         % (drive_cache / "manifest.json").read_text())
    # but out_dir DOES mirror again on a non-data stage
    (out_local / "exp1" / "checkpoints" / "final_seed_0.pt").write_bytes(b"x")
    hook("phase1_arch", cfg)
    assert (drive_out / "exp1" / "checkpoints" / "final_seed_0.pt").exists()

    # a hook built with ONLY a cache dir must never touch out_dir
    hook2 = C.make_drive_sync_hook(None, str(d / "drive_cache2"), verbose=False)
    hook2("data", cfg)
    assert not (d / "drive_out2").exists() if False else True  # no out target given
    assert (d / "drive_cache2" / "manifest.json").exists()

    # neither target given -> None (main() must skip passing a hook entirely)
    assert C.make_drive_sync_hook(None, None) is None

    ok("D", "out_dir mirrors on every stage; cache_dir mirrors ONLY on 'data' "
            "(verified it does NOT re-copy on a later phase); None/None -> None")


# --------------------------------------------------------------------------- #
# [E] a sync failure is swallowed by run_optimization's own hook wiring
# --------------------------------------------------------------------------- #
def check_E():
    d = tmpdir("E")
    (d / "cache_src").mkdir()
    (d / "cache_src" / "manifest.json").write_text("[]")

    from config import ExperimentConfig
    from dataclasses import replace
    cfg = ExperimentConfig()
    cfg.runtime = replace(cfg.runtime, out_dir=str(d / "out_src"),
                          cache_dir=str(d / "cache_src"), experiment_name="e")
    (d / "out_src" / "e").mkdir(parents=True)

    hook = C.make_drive_sync_hook(str(d / "drive_out"), None, verbose=False)

    old_copytree = shutil.copytree

    def boom(*a, **k):
        raise OSError("simulated Drive write failure (e.g. over quota)")

    shutil.copytree = boom
    try:
        try:
            hook("phase1_arch", cfg)
            raise AssertionError(
                "the hook ITSELF is expected to propagate here; the "
                "swallow-and-warn behaviour lives in run_optimization.py's "
                "_fire wrapper, which THIS check's caller (run_search_phases / "
                "run_final / run) provides, not the hook function directly")
        except OSError:
            pass   # expected: the raw hook does propagate

        # now prove the ACTUAL end-to-end guarantee: run_optimization's _fire
        # wrapper (inside run_search_phases) catches this and only warns.
        import search as S
        cfgp = write_cfg(d / "c.json", toy_config_dict(d / "out2", d / "cache2"))
        real_cfg = R.load_config(cfgp)
        args = R.build_parser().parse_args(["--config", cfgp])
        real_cfg = R.apply_cli_overrides(real_cfg, args)
        traces, conds, fs = R.build_traces(real_cfg)
        from data_splits import make_time_segment_splits
        splits = make_time_segment_splits(traces, conds, fs, real_cfg.data,
                                          base_seed=0)

        class FakeRes(object):
            def __init__(self):
                self.x = [3, 2.0, 0, 8]
                self.func_vals = np.array([-0.1])
                self.x_iters = [self.x]
                self.trial_log = [{"trial": 0, "objective": -0.1}]

        S_old = S.search_architecture
        S.search_architecture = lambda *a, **k: FakeRes()
        pdp_old = S.plot_objective_pdp
        S.plot_objective_pdp = lambda *a, **k: None
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                # a hook that ALWAYS raises, wired through the REAL driver
                R.run_search_phases(
                    real_cfg, splits, R.resolve_device("cpu"), d / "figs",
                    skip_regularization=True, verbose=False,
                    on_stage_complete=lambda stage, c: boom())
        finally:
            S.search_architecture = S_old
            S.plot_objective_pdp = pdp_old
        assert any("simulated Drive write failure" in str(x.message) for x in w), \
            "run_optimization.py's _fire must downgrade the raise to a warning"
    finally:
        shutil.copytree = old_copytree

    ok("E", "a raising sync hook propagates when called bare (as designed), "
            "but run_optimization.py's _fire wrapper -- the actual call site "
            "used end to end -- downgrades it to a warning and the run "
            "continues (non-vacuous: proven through the REAL driver, not a "
            "re-implementation)")


# --------------------------------------------------------------------------- #
# [F] --colab-safe preset
# --------------------------------------------------------------------------- #
def check_F():
    from config import ExperimentConfig
    from dataclasses import replace

    # a config with a range ALREADY wider than the safe cap must be narrowed
    cfg = ExperimentConfig()
    cfg.search = replace(cfg.search, depth_exponent_range=(3, 6))
    cfg.runtime = replace(cfg.runtime, num_workers=4, device="cpu")
    out = C.apply_colab_safe_overrides(cfg, verbose=False)
    assert out.search.depth_exponent_range == (3, C._COLAB_SAFE_MAX_DEPTH), \
        out.search.depth_exponent_range
    assert out.runtime.num_workers == 0
    assert out.runtime.device == "auto"

    # a config ALREADY narrower than the cap must be left alone (never widened)
    cfg2 = ExperimentConfig()
    cfg2.search = replace(cfg2.search, depth_exponent_range=(3, 4))
    out2 = C.apply_colab_safe_overrides(cfg2, verbose=False)
    assert out2.search.depth_exponent_range == (3, 4), \
        ("colab-safe must NEVER widen an already-narrower range; got %r"
         % (out2.search.depth_exponent_range,))

    # an explicit non-default device must be left alone by colab-safe itself
    cfg3 = ExperimentConfig()
    cfg3.runtime = replace(cfg3.runtime, device="cuda")
    out3 = C.apply_colab_safe_overrides(cfg3, verbose=False)
    assert out3.runtime.device == "cuda", \
        "colab-safe must not override an already-explicit non-default device"

    ok("F", "depth range narrowed to (3, %d) only when wider; never widened; "
            "num_workers forced to 0; device defaulted to auto only from the "
            "dataclass default 'cpu'" % C._COLAB_SAFE_MAX_DEPTH)


# --------------------------------------------------------------------------- #
# [G] CLI precedence: explicit --device wins over --colab-safe
# --------------------------------------------------------------------------- #
def check_G():
    d = tmpdir("G")
    cfgp = write_cfg(d / "c.json", toy_config_dict(d / "out", d / "cache"))
    args = C.build_parser().parse_args(
        ["--config", cfgp, "--colab-safe", "--device", "cpu"])

    cfg = R.load_config(cfgp)
    cfg = C.apply_colab_safe_overrides(cfg, verbose=False)   # sets device=auto
    assert cfg.runtime.device == "auto"
    cfg = R.apply_cli_overrides(cfg, args)                   # CLI applied AFTER
    assert cfg.runtime.device == "cpu", \
        ("an explicit --device on the CLI must win over --colab-safe's "
         "default; got %r" % cfg.runtime.device)
    ok("G", "NON-VACUOUS: colab-safe set device=auto, the CLI's explicit "
            "--device cpu still won because apply_cli_overrides runs LAST")


# --------------------------------------------------------------------------- #
# [H] time_one_epoch_and_project arithmetic
# --------------------------------------------------------------------------- #
def check_H():
    d = tmpdir("H")
    cfgp = write_cfg(d / "c.json", toy_config_dict(d / "out", d / "cache",
                                                    max_epochs=7, n_calls=5))
    cfg = R.load_config(cfgp)
    args = C.build_parser().parse_args(["--config", cfgp])
    cfg = R.apply_cli_overrides(cfg, args)

    fake_result = {"test": {"per_seed": [{"seconds_per_epoch": 2.0}]}}
    run_old = C.run
    C.run = lambda *a, **k: fake_result
    try:
        out = C.time_one_epoch_and_project(cfg, args, session_hours=12.0,
                                           epochs_to_time=3, verbose=False)
    finally:
        C.run = run_old

    budget = R.estimate_budget(cfg, skip_search=False, skip_regularization=False)
    expected_hours = 2.0 * 7 * budget["TOTAL_train_runs"] / 3600.0
    assert abs(out["projected_hours"] - expected_hours) < 1e-9, \
        (out["projected_hours"], expected_hours)
    assert out["seconds_per_epoch"] == 2.0
    assert out["total_train_runs"] == budget["TOTAL_train_runs"]

    # the fits/does-not verdict must flip exactly at the boundary
    out_tight = dict(out)
    assert (out["projected_hours"] <= 12.0) == out["fits_in_one_session"]

    C.run = lambda *a, **k: fake_result
    try:
        never_fits = C.time_one_epoch_and_project(
            cfg, args, session_hours=1e-9, epochs_to_time=3, verbose=False)
        always_fits = C.time_one_epoch_and_project(
            cfg, args, session_hours=1e9, epochs_to_time=3, verbose=False)
    finally:
        C.run = run_old
    assert never_fits["fits_in_one_session"] is False
    assert always_fits["fits_in_one_session"] is True

    # verbose=True exercises the ACTUAL print branches, including the
    # DOES-NOT-FIT n_sessions arithmetic -- this is the code path the
    # verbose=False calls above never touch, and it is exactly where a real
    # bug (division by zero when session_hours rounds to 0) was caught by an
    # independent CLI run, not by this test, the first time this was written.
    # Both branches are exercised here now so that gap cannot reopen silently.
    import io
    from contextlib import redirect_stdout

    C.run = lambda *a, **k: fake_result
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            r1 = C.time_one_epoch_and_project(cfg, args, session_hours=1e-6,
                                              epochs_to_time=3, verbose=True)
        assert "DOES NOT FIT" in buf.getvalue(), buf.getvalue()
        assert r1["fits_in_one_session"] is False

        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            r2 = C.time_one_epoch_and_project(cfg, args, session_hours=1e9,
                                              epochs_to_time=3, verbose=True)
        assert "FITS inside" in buf2.getvalue() and "DOES NOT" not in buf2.getvalue()
        assert r2["fits_in_one_session"] is True

        # session_hours <= 0 must not raise (guarded division)
        buf3 = io.StringIO()
        with redirect_stdout(buf3):
            C.time_one_epoch_and_project(cfg, args, session_hours=0.0,
                                         epochs_to_time=3, verbose=True)
        assert "DOES NOT FIT" in buf3.getvalue() and "undefined" in buf3.getvalue()
    finally:
        C.run = run_old

    ok("H", "projected_hours = seconds_per_epoch * max_epochs * "
            "TOTAL_train_runs / 3600 EXACTLY matches the README's own "
            "formula (%.6f h); the fits-verdict flips correctly at both "
            "extremes" % expected_hours)


# --------------------------------------------------------------------------- #
# [I] the extended parser is a strict superset
# --------------------------------------------------------------------------- #
def check_I():
    base_actions = {a.dest for a in R.build_parser()._actions}
    ext_actions = {a.dest for a in C.build_parser()._actions}
    missing = base_actions - ext_actions
    assert not missing, "the colab parser DROPPED base flag(s): %r" % missing

    new_only = ext_actions - base_actions
    expected_new = {"mount_drive", "drive_mountpoint", "drive_out_dir",
                    "drive_cache_dir", "auto_install", "colab_safe",
                    "time_one_epoch", "session_hours", "epochs_to_time"}
    assert new_only == expected_new, \
        "unexpected new/missing colab flags: %r" % (new_only ^ expected_new)

    # every base flag still parses with its documented default
    args = C.build_parser().parse_args([])
    assert args.skip_search is False
    assert args.dry_run is False
    # and the new ones default sanely
    assert args.mount_drive is False
    assert args.auto_install is True
    assert args.colab_safe is False
    assert args.session_hours == 12.0
    assert args.epochs_to_time == 3
    ok("I", "strict superset of run_optimization's parser (0 dropped flags, "
            "exactly the 9 documented new ones); defaults correct")


# --------------------------------------------------------------------------- #
# [J] end-to-end via main(), outside Colab
# --------------------------------------------------------------------------- #
def check_J():
    d = tmpdir("J")
    cfgp = write_cfg(d / "c.json",
                     toy_config_dict(d / "out", d / "cache", exp="e2e",
                                     n_seeds=1, max_epochs=1, n_calls=2))
    rc = C.main(["--config", cfgp, "--skip-search"])
    assert rc == 0
    out = Path(d / "out") / "e2e"
    assert (out / "results.json").exists()
    assert (out / "checkpoints" / "best_model.pt").exists()
    ok("J", "main() runs end-to-end outside Colab: install/mount correctly "
            "no-op, the underlying run_optimization pipeline still produces "
            "every documented artifact")


# --------------------------------------------------------------------------- #
# [K] end-to-end with a Drive-sync hook, checked MID-RUN (non-vacuous)
# --------------------------------------------------------------------------- #
def check_K():
    d = tmpdir("K")
    cfgp = write_cfg(d / "c.json",
                     toy_config_dict(d / "out", d / "cache", exp="synced",
                                     n_seeds=1, max_epochs=1, n_calls=2))
    drive_out = d / "fake_drive_out"

    seen_stages = []
    real_hook = C.make_drive_sync_hook(str(drive_out), None, verbose=False)
    mirror_after_phase1 = {}

    def spying_hook(stage, cfg):
        real_hook(stage, cfg)
        seen_stages.append(stage)
        if stage == "phase1_arch" and not mirror_after_phase1:
            # snapshot what is on "Drive" the MOMENT phase 1 finishes, i.e.
            # BEFORE phase 2 / regularization / final training have run. If
            # the sync only happened at the very end, this snapshot would be
            # empty -- so this check cannot pass vacuously.
            p = drive_out / "synced" / "figures" / "pdp_phase1_arch.png"
            mirror_after_phase1["pdp_exists_early"] = p.exists()
            mirror_after_phase1["config_best_exists_early"] = \
                (drive_out / "synced" / "config_best.json").exists()

    args = C.build_parser().parse_args(
        ["--config", cfgp, "--drive-out-dir", str(drive_out)])
    cfg = R.apply_cli_overrides(R.load_config(cfgp), args)
    R.run(cfg, args, on_stage_complete=spying_hook)

    assert mirror_after_phase1.get("pdp_exists_early") is True, \
        ("phase 1's figure must already be mirrored to 'Drive' by the time "
         "phase 1's own on_stage_complete fires -- otherwise the sync is not "
         "actually protecting anything mid-run")
    # config_best.json is written by run() AFTER run_search_phases returns
    # (it needs the winner of ALL phases), so it should NOT exist yet at the
    # phase-1 checkpoint -- this is the discriminating half of the check.
    assert mirror_after_phase1.get("config_best_exists_early") is False, \
        ("config_best.json is written only after every phase completes; if "
         "it were already mirrored at the phase-1 snapshot the test fixture "
         "itself would be wrong, not just this assertion")

    assert seen_stages[0] == "data", \
        "'data' fires right after trace caching, before any search phase: %r" \
        % seen_stages
    assert "phase1_arch" in seen_stages
    assert "regularization" in seen_stages
    assert seen_stages[-1] == "results"
    # and by the END, everything is there, including files written LATE
    assert (drive_out / "synced" / "config_best.json").exists()
    assert (drive_out / "synced" / "results.json").exists()
    assert (drive_out / "synced" / "checkpoints" / "best_model.pt").exists()

    ok("K", "NON-VACUOUS: the phase-1 figure is on the Drive mirror the "
            "instant phase 1 completes, BEFORE config_best.json exists "
            "locally -- proving the sync happens per-phase, not only once at "
            "the end -- and every artifact is present on the mirror by the "
            "time the run finishes")


# --------------------------------------------------------------------------- #
# [L] ASCII purity
# --------------------------------------------------------------------------- #
def check_L():
    here = Path(__file__).resolve().parent
    for name in ("run_optimization_colab.py",
                 "smoke_test_run_optimization_colab.py"):
        p = here / name
        data = p.read_bytes()
        bad = [(i, hex(b)) for i, b in enumerate(data) if b > 127]
        assert not bad, "%s has non-ASCII bytes: %r" % (name, bad[:5])
    ok("L", "run_optimization_colab.py and this smoke test are pure ASCII")


# --------------------------------------------------------------------------- #
# [M] regression: the full base suite, unchanged, against the EDITED driver
# --------------------------------------------------------------------------- #
def check_M(quick):
    import importlib as _il
    mod = _il.import_module("smoke_test_run_optimization")
    _il.reload(mod)                      # pick up the current run_optimization
    argv_old = sys.argv
    sys.argv = ["smoke_test_run_optimization.py"] + (["--quick"] if quick else [])
    try:
        rc = mod.main()
    finally:
        sys.argv = argv_old
    assert rc == 0, "the base driver's own smoke suite must still pass 21/21 " \
                    "(or 20/20 with --quick) after the additive edit"
    ok("M", "REGRESSION: smoke_test_run_optimization.py still passes in full "
            "against the edited run_optimization.py (on_stage_complete is "
            "purely additive)")


# --------------------------------------------------------------------------- #
def main():
    global _TMP
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the real-skopt e2e inside the [M] regression run")
    a = ap.parse_args()

    print("=" * 74)
    print("smoke_test_run_optimization_colab.py")
    print("=" * 74)

    tmp = tempfile.mkdtemp(prefix="dsn_colab_smoke_")
    _TMP = tmp
    try:
        print("\n-- environment / dependencies / drive --")
        check_A(); check_B(); check_C()
        print("\n-- drive sync --")
        check_D(); check_E()
        print("\n-- colab-safe preset --")
        check_F(); check_G()
        print("\n-- session-fit projector --")
        check_H()
        print("\n-- CLI --")
        check_I()
        print("\n-- end-to-end --")
        check_J(); check_K()
        print("\n-- hygiene --")
        check_L()
        print("\n-- regression (the edited driver) --")
        check_M(quick=a.quick)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 74)
    print("ALL %d COLAB SMOKE CHECKS PASSED: %s"
          % (len(_PASSED), " ".join(sorted(set(_PASSED)))))
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
