#!/usr/bin/env python3
"""
local_setup_and_run.py
======================

Local / VS Code analogue of colab_setup_and_run.py.

Paste each "# %%" block below into its own cell (VS Code's Python
Interactive window and the Jupyter extension both recognise "# %%" as a
cell delimiter), OR just run the file top-to-bottom as a plain script
(python3 local_setup_and_run.py) for the identical sequence in one shot.

Like the Colab version, this calls the functions from run_optimization.py
EXPLICITLY, one purpose per cell, instead of hiding the whole sequence
behind a single main() call -- so you can inspect, skip, or repeat any step
from the VS Code interactive window.

Difference from colab_setup_and_run.py (deliberate, per the target)
-------------------------------------------------------------------
    * NO Google Drive: there is no Cell 1 mount, no --drive-out-dir, no
      make_drive_sync_hook. On a laptop the local filesystem already
      persists across sessions, which is the only thing the Drive mirror
      was protecting against on Colab.
    * NO code-unzip step: the modules already sit next to this file in the
      repo you cloned/opened in VS Code. Cell 2 just puts THIS directory on
      sys.path and checks the 14 required modules are present.
    * NO auto-install: dependencies are ASSUMED already installed (you told
      me so). Cell 3 does a fail-fast importability CHECK and names the exact
      pip/conda command if something is missing -- it never installs for you.
    * NO --colab-safe preset: on a laptop YOU own the box, so device and
      search width are taken from your config as written. --colab-safe was a
      free-T4 memory guard; it does not belong here. (You can still cap the
      search yourself in the config if your laptop is small -- see Cell 4's
      note.)

Everything ELSE is intentionally identical to the Colab flow: same
build_parser(), same load_config -> apply_cli_overrides, same optional
time-one-epoch projection, the same single R.run(cfg, args) call that does
all the training, and the same zero-hyperparameter checkpoint reload.

Dependency manifest -- what must be in the SAME folder as this file
-------------------------------------------------------------------
The 14 import-time modules (verified against the actual files, same set the
Colab manifest lists):

    backbone.py            augmentation.py         data_pipeline.py
    config.py              preprocessing_cache.py  data_splits.py
    metrics.py             checkpoint.py           inference.py
    train.py               evaluate.py             search.py
    run_optimization.py    run_optimization_colab.py

run_optimization_colab.py is imported here ONLY to reuse its build_parser()
(which is run_optimization's parser plus a few flags that are harmless
no-ops off Colab) and its time_one_epoch_and_project() helper. Nothing
Colab- or Drive-specific is invoked.

Two more you want but are not import-time dependencies:

    config_example.json, config_toy.json     starting configs
    Smoke_Tests/smoke_test_run_optimization.py
                                              NOT imported by the pipeline;
                                              Cell 3 runs it once as a sanity
                                              check BEFORE any real training.

hpc-python-compat: pure ASCII, matching the rest of the codebase.
"""

# %% [Cell 1] -- put THIS directory on sys.path and confirm the 14 required
# modules are present. (No Drive mount, no unzip: in VS Code the code is
# already here, next to this file.)
import sys
from pathlib import Path

# When run as a script, __file__ is defined. When pasted cell-by-cell into
# the VS Code interactive window, __file__ may be absent -- fall back to CWD,
# which VS Code sets to the file's directory by default.
try:
    CODE_DIR = Path(__file__).resolve().parent
except NameError:
    CODE_DIR = Path.cwd()

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

present = sorted(p.name for p in CODE_DIR.glob("*.py"))
print("[setup] running from %s" % CODE_DIR)
print("[setup] %d .py files on sys.path" % len(present))

REQUIRED = {
    "augmentation.py", "backbone.py", "data_pipeline.py", "config.py",
    "preprocessing_cache.py", "data_splits.py", "metrics.py", "checkpoint.py",
    "inference.py", "train.py", "evaluate.py", "search.py",
    "run_optimization.py", "run_optimization_colab.py",
}
missing = REQUIRED - set(present)
if missing:
    raise FileNotFoundError(
        "local_setup_and_run.py must live in the SAME folder as the pipeline "
        "modules. Missing: %r. Open the repo's Main/ folder in VS Code and run "
        "this file from there." % sorted(missing))
print("[setup] all %d required modules present." % len(REQUIRED))


# %% [Cell 2] -- import the drivers and print what this machine actually
# offers (CPU vs CUDA), so a silent CPU fallback is visible immediately
# rather than discovered an hour into a run.
import run_optimization as R
import run_optimization_colab as RC   # reused only for build_parser + timing

# environment_banner() is Colab-worded ("in Google Colab: False", "/content/
# drive mounted: False") but its CUDA line is exactly what we want here, so
# print the CUDA fact directly rather than the Colab-framed banner.
def _local_banner():
    lines = []
    try:
        import torch
        cuda = torch.cuda.is_available()
        dev = (" (%s)" % torch.cuda.get_device_name(0)) if cuda else ""
        lines.append("[local] torch %s  CUDA available: %s%s"
                     % (torch.__version__, cuda, dev))
    except Exception as ex:   # noqa: BLE001  (report, do not crash the banner)
        lines.append("[local] torch import/probe failed: %s: %s"
                     % (type(ex).__name__, ex))
    lines.append("[local] python %s" % sys.version.split()[0])
    return "\n".join(lines)

print(_local_banner())


# %% [Cell 3] -- CHECK (do not install) that the whole stack imports, then run
# the smoke test ONCE on this machine as a sanity check before committing any
# real training time to it.
#
# ensure_dependencies(auto_install=False) verifies importability and, if
# anything is missing, raises with the exact command to run by hand -- it will
# NOT pip-install anything, because you said the deps are already installed.
RC.ensure_dependencies(auto_install=False, verbose=True)

import subprocess
smoke_path = CODE_DIR / "Smoke_Tests" / "smoke_test_run_optimization.py"
if not smoke_path.exists():
    # Some checkouts keep the smoke tests next to the modules instead of in a
    # Smoke_Tests/ subfolder; accept either layout.
    alt = CODE_DIR / "smoke_test_run_optimization.py"
    smoke_path = alt if alt.exists() else smoke_path

if smoke_path.exists():
    # The smoke test does a bare `import run_optimization`; when it lives in a
    # Smoke_Tests/ subfolder, the child's sys.path[0] is that subfolder, NOT
    # CODE_DIR, so the parent-dir modules would not import. Put CODE_DIR on the
    # child's PYTHONPATH so it imports the pipeline regardless of which layout
    # the smoke test sits in (subfolder or flat).
    import os
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = (
        str(CODE_DIR) + os.pathsep + child_env.get("PYTHONPATH", ""))
    smoke = subprocess.run(
        [sys.executable, str(smoke_path), "--quick"],
        cwd=str(CODE_DIR), env=child_env, capture_output=True, text=True)
    print(smoke.stdout[-1500:])
    if smoke.returncode != 0:
        print(smoke.stderr[-2000:])
        raise RuntimeError(
            "the smoke test failed on THIS machine -- fix this before running "
            "anything real. See the captured output above.")
    print("[setup] smoke test passed: this machine is good to train on.")
else:
    print("[setup] WARNING: smoke_test_run_optimization.py not found at %s -- "
          "skipping the pre-run sanity check. You can still proceed, but you "
          "are trusting the environment untested." % smoke_path)


# %% [Cell 4] -- build the CLI-equivalent args and load + adapt the config.
# This is exactly what `python3 run_optimization.py <these flags>` does;
# building args through the real parser (rather than by hand) keeps every
# flag's validation and precedence intact.
#
# Small-laptop note: there is no --colab-safe here on purpose. If your machine
# is memory-limited, cap the search yourself in the JSON -- lower
# search.depth_exponent_range's upper bound and set runtime.num_workers = 0 --
# rather than relying on a Colab preset. config_toy.json is already tiny.
args = RC.build_parser().parse_args([
    "--config", str(CODE_DIR / "config_toy.json"),   # swap for your own config
    "--out-dir", str(CODE_DIR / "local_out"),
    "--cache-dir", str(CODE_DIR / "local_cache"),
    "--experiment-name", "local_run1",
    "--device", "auto",                              # cuda if present, else cpu
    "--verbose",
])

cfg = R.load_config(args.config)
cfg = R.apply_cli_overrides(cfg, args)               # CLI flags win over JSON


# %% [Cell 5] -- OPTIONAL but recommended before a real (non-toy) config:
# measure real seconds/epoch and project the FULL configured budget, WITHOUT
# running the whole pipeline. On a laptop there is no session ceiling to hit,
# so this is a "how long will this actually take me" estimate rather than a
# hard gate. Pick session_hours to match how long you are willing to leave it.
projection = RC.time_one_epoch_and_project(
    cfg, args, session_hours=8.0, epochs_to_time=3, verbose=True)
# projection["fits_in_one_session"] is a plain bool if you want to gate on it,
# e.g.:  assert projection["fits_in_one_session"], "narrow the search first"


# %% [Cell 6] -- the real run. This is the SAME run() the plain driver and the
# Colab driver both call; the ONLY difference from Colab is that no Drive-sync
# hook is passed (on_stage_complete stays None -> documented default: nothing
# is mirrored, the local filesystem is the only sink, which is all a laptop
# needs). This is the one call that actually trains anything.
results = R.run(cfg, args)          # no on_stage_complete hook -> local only


# %% [Cell 7] -- inspect the results still live in memory, then reload the
# saved best model with ZERO hyper-parameters in this cell (it is rebuilt from
# the config embedded in the checkpoint). Identical to the Colab Cell 7.
print("TEST ARI  : %.4f +/- %.4f" % (results["test"]["ari"]["mean"],
                                     results["test"]["ari"]["std"]))
print("TEST AMI  : %.4f +/- %.4f" % (results["test"]["ami"]["mean"],
                                     results["test"]["ami"]["std"]))
print("best model: %s" % results["test"]["best_model"])

if results["test"]["best_model"]:
    from checkpoint import rebuild_model_from_checkpoint
    model, ckpt = rebuild_model_from_checkpoint(results["test"]["best_model"])
    model.eval()
    print("reloaded model:", type(model).__name__,
          "embedding_size =", ckpt["config"]["backbone"]["embedding_size"])
else:
    print("[run] no best_model.pt was produced (all final seeds non-finite "
          "val ARI) -- check the run log above.")
