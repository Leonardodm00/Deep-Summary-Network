#!/usr/bin/env python3
"""
colab_setup_and_run.py
=======================

Paste each "# %%" block below into its own Colab cell (or run this file as-is
in a plain Python session for a local dry run of the SAME sequence -- every
cell degrades gracefully outside Colab, which is how it was tested here).

This calls the functions from run_optimization_colab.py / run_optimization.py
EXPLICITLY, one purpose per cell, instead of hiding the whole sequence behind
a single main() call -- so you can inspect, skip, or repeat any step.

Dependency manifest -- what must be in the SAME folder on Colab
-----------------------------------------------------------------
run_optimization_colab.py imports run_optimization.py, which imports the
Topic-1/2/3 modules. The exact set (verified with Python's own ast module
against the actual files, not by hand-counting) is 14 .py files:

    backbone.py            augmentation.py         data_pipeline.py
    config.py               preprocessing_cache.py  data_splits.py
    metrics.py               checkpoint.py           inference.py
    train.py                 evaluate.py             search.py
    run_optimization.py      run_optimization_colab.py

None of these import anything outside this set except third-party packages
(torch, numpy, scipy, sklearn, matplotlib, skopt, pytorch_metric_learning).
Two more are not import-time dependencies but you almost certainly want them:

    config_example.json, config_toy.json     starting configs (or write your
                                              own JSON with the same schema)
    smoke_test_run_optimization.py,
    smoke_test_run_optimization_colab.py     NOT imported by the pipeline;
                                              run them once per fresh Colab
                                              environment as a sanity check
                                              BEFORE spending any GPU time
                                              (Cell 3 below does this)

All 18 files above are bundled in dsn_pipeline.zip. Upload that one file via
Colab's Files pane into /content/, or copy it once into
/content/drive/MyDrive/ so every future session finds it without a re-upload.

HPC note (hpc-python-compat): pure ASCII, for consistency with the rest of the
codebase (Colab's own transfer path -- upload / git clone / Drive -- does not
carry the cp1252 corruption risk that convention exists for on davinci-1).
"""

# %% [Cell 1] -- mount Google Drive (skip this cell if you don't need results
# or the trace cache to survive a disconnect; everything still runs without it,
# just entirely inside the ephemeral /content/ filesystem)
from google.colab import drive
drive.mount('/content/drive')


# %% [Cell 2] -- get the code onto this runtime and put it on sys.path
import shutil
import sys
import zipfile
from pathlib import Path

CODE_DIR = Path('/content/dsn')
ZIP_LOCAL = Path('/content/dsn_pipeline.zip')
ZIP_DRIVE = Path('/content/drive/MyDrive/dsn_pipeline.zip')   # optional cache

if not ZIP_LOCAL.exists() and ZIP_DRIVE.exists():
    shutil.copy2(ZIP_DRIVE, ZIP_LOCAL)
    print("[setup] copied dsn_pipeline.zip from Drive (persists across sessions)")

if not ZIP_LOCAL.exists():
    raise FileNotFoundError(
        "dsn_pipeline.zip is not in /content/ and not on Drive.\n"
        "Upload it once via the Files pane (left sidebar) into /content/, "
        "then re-run this cell. To skip re-uploading on every future session, "
        "also run:\n"
        "  import shutil; shutil.copy2('/content/dsn_pipeline.zip', "
        "'/content/drive/MyDrive/dsn_pipeline.zip')")

CODE_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ZIP_LOCAL) as zf:
    zf.extractall(CODE_DIR)

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

present = sorted(p.name for p in CODE_DIR.glob('*.py'))
print("[setup] %d .py files on sys.path from %s:" % (len(present), CODE_DIR))
for name in present:
    print("  ", name)

REQUIRED = {
    "augmentation.py", "backbone.py", "data_pipeline.py", "config.py",
    "preprocessing_cache.py", "data_splits.py", "metrics.py", "checkpoint.py",
    "inference.py", "train.py", "evaluate.py", "search.py",
    "run_optimization.py", "run_optimization_colab.py",
}
missing = REQUIRED - set(present)
if missing:
    raise FileNotFoundError(
        "dsn_pipeline.zip is missing required file(s): %r. The pipeline "
        "cannot import without them." % sorted(missing))
print("[setup] all %d required modules present." % len(REQUIRED))


# %% [Cell 3] -- import, install missing packages, print the environment,
# and run the smoke test ONCE per fresh runtime as a sanity check before
# committing any real GPU time to it
import run_optimization_colab as RC
import run_optimization as R

print(RC.environment_banner())
RC.ensure_dependencies(auto_install=True, verbose=True)

import subprocess
smoke = subprocess.run(
    [sys.executable, str(CODE_DIR / "smoke_test_run_optimization_colab.py"),
     "--quick"],
    cwd=str(CODE_DIR), capture_output=True, text=True)
print(smoke.stdout[-1500:])
if smoke.returncode != 0:
    print(smoke.stderr[-2000:])
    raise RuntimeError(
        "the smoke test failed in THIS environment -- fix this before running "
        "anything real. See the captured output above.")
print("[setup] smoke test passed: this environment is good to train on.")


# %% [Cell 4] -- build the CLI-equivalent args and load + adapt the config
# This is exactly what `python3 run_optimization_colab.py <these flags>` does;
# building args this way (rather than by hand) means every flag keeps its
# real validation and precedence (colab-safe first, then explicit CLI flags,
# which always win -- see run_optimization_colab.py's own smoke test [G]).
args = RC.build_parser().parse_args([
    '--config', str(CODE_DIR / 'config_toy.json'),        # swap for your own
    '--out-dir', '/content/drive/MyDrive/dsn_out',
    '--cache-dir', '/content/drive/MyDrive/dsn_cache',
    '--experiment-name', 'colab_run1',
    '--device', 'auto',
    '--colab-safe',                                        # safe for a T4
    '--drive-out-dir', '/content/drive/MyDrive/dsn_out_mirror',
    '--verbose',
])

cfg = R.load_config(args.config)
if args.colab_safe:
    cfg = RC.apply_colab_safe_overrides(cfg, verbose=True)
cfg = R.apply_cli_overrides(cfg, args)                     # CLI wins, always


# %% [Cell 5] -- OPTIONAL but strongly recommended before a real (non-toy)
# config: measure real seconds/epoch and project the FULL budget against a
# Colab session ceiling, WITHOUT running the full pipeline
projection = RC.time_one_epoch_and_project(
    cfg, args, session_hours=12.0, epochs_to_time=3, verbose=True)
# projection['fits_in_one_session'] is a plain bool if you want to gate on it
# programmatically, e.g.:
#   assert projection['fits_in_one_session'], "narrow the search before running"


# %% [Cell 6] -- the real run: build the Drive-sync hook, then call the SAME
# run() the plain (non-Colab) driver uses. This is the one call that actually
# trains anything; everything above it is setup, verification, or a dry
# projection.
hook = RC.make_drive_sync_hook(args.drive_out_dir, args.drive_cache_dir,
                               verbose=True)
results = R.run(cfg, args, on_stage_complete=hook)


# %% [Cell 7] -- inspect the results still live in memory, and reload the
# saved best model with ZERO hyper-parameters in this cell (it is rebuilt from
# the config embedded in the checkpoint)
print("TEST ARI  : %.4f +/- %.4f" % (results['test']['ari']['mean'],
                                     results['test']['ari']['std']))
print("TEST AMI  : %.4f +/- %.4f" % (results['test']['ami']['mean'],
                                     results['test']['ami']['std']))
print("best model: %s" % results['test']['best_model'])

from checkpoint import rebuild_model_from_checkpoint
model, ckpt = rebuild_model_from_checkpoint(results['test']['best_model'])
model.eval()
print("reloaded model:", type(model).__name__,
      "embedding_size =", ckpt['config']['backbone']['embedding_size'])
