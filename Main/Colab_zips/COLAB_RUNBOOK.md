# Running the Deep-Summary-Network pipeline on Google Colab

This is a step-by-step runbook for training / optimizing the 1D-CNN summary
network on Colab, **starting from the point where both zip files are already in
a Google Drive folder.**

Every code block below was executed end-to-end before this document was written:
the full two-phase search runs, the Drive sync fires after each phase, and the
results and checkpoints land on Drive. Paste each numbered block into its own
Colab cell and run them in order.

---

## Before you start

You need these two files sitting in **one folder on your Google Drive**:

```
dsn_pipeline.zip       the 23-file pipeline (code + configs + environment)
dsn_smoke_tests.zip    the 16-file test suite (15 suites + a runner)
```

This runbook assumes that folder is:

```
/content/drive/MyDrive/DSN/
```

If you put the zips somewhere else, change the single `DRIVE_DIR` line in
**Step 2** -- nothing else in the runbook needs editing.

**Why a GPU runtime.** Before running, switch Colab to a GPU: *Runtime -> Change
runtime type -> Hardware accelerator -> GPU (T4)*. The pipeline runs on CPU too,
but the search trains dozens of models, so a GPU matters. (The T4 is not
guaranteed on the free tier; `--device auto` in Step 5 falls back to CPU with a
warning if no GPU is actually attached.)

**The design in one sentence.** The **code** is extracted to Colab's fast local
disk (`/content/dsn`), while **all outputs** -- the trace cache, checkpoints,
results, figures -- are written to **Drive**, so a disconnect never loses them.
Drive is a network filesystem: fine to import from occasionally, but too slow to
run a training cache off directly.

---

## Step 1 -- Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Click through the authorization prompt. When it finishes, your Drive is visible
under `/content/drive/MyDrive/`.

---

## Step 2 -- Point the runbook at your folder

This is the **only** cell you may need to edit: set `DRIVE_DIR` to the folder
that holds the two zips.

```python
import os

# ---- EDIT THIS if your zips live elsewhere ----
DRIVE_DIR = '/content/drive/MyDrive/DSN'
# -----------------------------------------------

CODE_DIR = '/content/dsn'          # fast local extract target (do not change)

assert os.path.isdir(DRIVE_DIR), (
    "Folder not found: %s\n"
    "Set DRIVE_DIR to the Drive folder that contains dsn_pipeline.zip." % DRIVE_DIR)

have = [f for f in ('dsn_pipeline.zip', 'dsn_smoke_tests.zip')
        if os.path.exists(os.path.join(DRIVE_DIR, f))]
assert 'dsn_pipeline.zip' in have, \
    "dsn_pipeline.zip is not in %s (found: %s)" % (DRIVE_DIR, have)
print("Found in %s: %s" % (DRIVE_DIR, have))
```

---

## Step 3 -- Extract the code to local disk

Both zips are flat (no sub-folders) and share no filenames, so they extract into
the same directory cleanly.

```python
import zipfile, sys

os.makedirs(CODE_DIR, exist_ok=True)
for zname in have:
    with zipfile.ZipFile(os.path.join(DRIVE_DIR, zname)) as z:
        z.extractall(CODE_DIR)

if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)          # so `import backbone` etc. resolve

n_py = len([f for f in os.listdir(CODE_DIR) if f.endswith('.py')])
print("Extracted %d .py files to %s" % (n_py, CODE_DIR))
print("On sys.path:", CODE_DIR)
```

You should see **35 .py files** if you extracted both zips (19 pipeline + 16
tests).

---

## Step 4 -- Install the two missing packages, then smoke-test

Colab preinstalls torch, numpy, scipy, scikit-learn and matplotlib, but **not**
`scikit-optimize` or `pytorch-metric-learning`. `ensure_dependencies()` installs
exactly those two (and nothing else), then confirms the whole stack imports.

```python
import run_optimization_colab as RC
import run_optimization as R

print(RC.environment_banner())
RC.ensure_dependencies(auto_install=True, verbose=True)
```

Then run one quick suite as a **gate** -- if the environment itself is broken,
you want to know now, not 40 minutes into a search:

```python
import subprocess
r = subprocess.run(
    [sys.executable, os.path.join(CODE_DIR, 'smoke_test_metrics.py')],
    cwd=CODE_DIR, capture_output=True, text=True)
print(r.stdout[-800:])
assert r.returncode == 0, "smoke test FAILED -- see output above.\n" + r.stderr[-800:]
print("\nEnvironment is good to train on.")
```

> **Optional -- run the whole test battery (~7 min).** If you want full
> confidence before a long run, execute all 15 suites:
> ```python
> !cd {CODE_DIR} && python3 run_all_smoke_tests.py --quick
> ```
> You may see a message that `scikit-optimize` was just installed and a runtime
> restart is needed. If so: *Runtime -> Restart runtime*, then re-run Steps 2-4.
> (After a restart, `sys.path` and the mount are cleared, which is why those
> steps are written to be safely re-runnable.)

---

## Step 5 -- Build the configuration

This assembles exactly what `python3 run_optimization_colab.py <flags>` would,
but keeps the config object in hand so the next cell can project its cost.

The two paths that matter, `OUT_DIR` and `CACHE_DIR`, both point at **Drive** --
that is what makes results and the (expensive) trace cache survive a disconnect.

```python
CONFIG    = os.path.join(CODE_DIR, 'config_toy.json')   # <-- swap for your own config
OUT_DIR   = os.path.join(DRIVE_DIR, 'out')              # results  -> Drive
CACHE_DIR = os.path.join(DRIVE_DIR, 'cache')            # trace cache -> Drive

argv = [
    '--config', CONFIG,
    '--out-dir', OUT_DIR,
    '--cache-dir', CACHE_DIR,
    '--experiment-name', 'colab_run1',
    '--device', 'auto',              # GPU if attached, else CPU with a warning
    '--colab-safe',                  # memory-safe preset for a T4 (see note)
    '--drive-out-dir', os.path.join(DRIVE_DIR, 'out_mirror'),  # per-phase sync
]

args = RC.build_parser().parse_args(argv)
cfg = R.load_config(args.config)
cfg = RC.apply_colab_safe_overrides(cfg, verbose=True)   # narrows the search
cfg = R.apply_cli_overrides(cfg, args)                   # your flags win, always
print("Config ready. Outputs ->", OUT_DIR)
```

**`config_toy.json` is a 30-second sanity run, not a real experiment** -- its
scores are meaningless. Use it once to confirm the whole path works, then point
`CONFIG` at `config_example.json` (a realistic 243-run search) or your own JSON.

**What `--colab-safe` does:** caps the architecture search at `depth_exponent 5`
(the depth-6 corner is ~214 M parameters, ~2.6 GB just for weights -- too large a
fraction of a T4's memory), forces `num_workers=0`, and defaults the device to
`auto`. It never *widens* a config you already narrowed, and any explicit
`--device` you pass still wins over it.

---

## Step 6 -- Check it fits in one session (do this for real configs)

Colab free-tier sessions cap out around **12 hours**, and -- this is the key
caveat -- **the search phases do not resume across a disconnect** (only the final
training stage does). So before committing to a real search, measure one epoch
and project the whole budget:

```python
proj = RC.time_one_epoch_and_project(
    cfg, args, session_hours=12.0, epochs_to_time=3, verbose=True)

if not proj['fits_in_one_session']:
    print("\n>>> This will NOT finish in one 12-hour session. <<<")
    print(">>> Narrow the search (n_calls_arch / n_calls_train / n_calls_reg /")
    print(">>> n_seeds / max_epochs), or run with --skip-search, before Step 7.")
```

This runs one short training in the background and reports, e.g.,
`projected total: 1.8 s/epoch * 3 epochs * 26 runs / 3600 = 0.0 hours -> FITS`.
For `config_toy.json` it always fits; for a real config, believe the number.

---

## Step 7 -- Run it

The one cell that actually trains. `make_drive_sync_hook` builds a callback that
mirrors `OUT_DIR` to your Drive `out_mirror` folder **after every search phase
and every final seed** -- so even if the session dies mid-search, everything
completed so far is already on Drive.

```python
hook = RC.make_drive_sync_hook(args.drive_out_dir, args.drive_cache_dir,
                               verbose=True)
results = R.run(cfg, args, on_stage_complete=hook)

print("\n" + "=" * 60)
print("TEST ARI : %.4f +/- %.4f" % (results['test']['ari']['mean'],
                                    results['test']['ari']['std']))
print("TEST AMI : %.4f +/- %.4f" % (results['test']['ami']['mean'],
                                    results['test']['ami']['std']))
print("best model saved to:", results['test']['best_model'])
print("=" * 60)
```

You'll watch the phases stream by: `PHASE 1: architecture` -> `PHASE 2: training
HPs` -> `REGULARIZATION` -> `FINAL: training N model(s)`, with a
`[colab] synced ...` line after each. When it finishes, everything is on Drive.

---

## Step 8 -- Reload the trained model

The saved checkpoint is self-describing: the architecture rebuilds from the
config embedded inside it, so you need **zero** hyper-parameters in your own
code.

```python
from checkpoint import rebuild_model_from_checkpoint

model, ckpt = rebuild_model_from_checkpoint(results['test']['best_model'])
model.eval()
print("Reloaded model, embedding_size =",
      ckpt['config']['backbone']['embedding_size'])

# model(x)  maps  x: (M, W)  ->  z: (M, E),  rows L2-normalized onto the unit sphere
```

---

## Where your results are on Drive

After Step 7, under `/content/drive/MyDrive/DSN/out_mirror/colab_run1/`:

```
config_input.json     the config exactly as resolved (file + CLI + colab-safe)
config_best.json      the config after every search phase
results.json          the deliverable: TEST ARI/AMI/silhouette per seed, budget
figures/
    pdp_phase1_arch.png, pdp_phase2_train.png, pdp_regularization.png
    embedding_test_seed_0.png, ...          (PCA of the test embeddings)
checkpoints/
    seed_<n>/{last,best}.pt                 resumable per-seed
    final_seed_<n>.pt                       self-describing (best-epoch weights)
    best_model.pt                           the deployable model (Step 8 loads this)
```

The trace cache is under `/content/drive/MyDrive/DSN/cache/`. Because it lives on
Drive, the **next** session skips re-computing traces automatically -- the driver
notices they're already cached (and refuses to reuse them if you change the data
source, telling you to pass `--overwrite-cache`).

---

## If the session disconnects

- **Results and checkpoints are safe** -- they were synced to Drive after every
  phase. Look in `out_mirror/colab_run1/`.
- **Re-running from Step 1** re-mounts Drive, re-extracts the code (fast), and
  finds the cached traces still on Drive, so you don't re-pay that cost.
- **The final training stage can resume.** If the disconnect happened during
  `FINAL: training ...`, add `'--resume'` to the `argv` list in Step 5 and
  re-run from Step 5; it picks up from the last checkpoint.
- **The search phases cannot resume** -- a disconnect during Phase 1/2 means that
  phase restarts. This is why Step 6 exists: size the run to fit one session, or
  use `--skip-search` (which trains your configured architecture directly, and
  *is* resumable).

---

## Quick reference -- the flags you'll actually change

| Flag (in Step 5's `argv`) | What it does |
|---|---|
| `--config <path>` | which config JSON to run (`config_toy.json` -> `config_example.json` -> your own) |
| `--experiment-name <name>` | names the output sub-folder; change it per run so you don't overwrite |
| `--skip-search` | skip all HPO; just train + evaluate the configured architecture (resumable) |
| `--n-seeds <n>` | how many models to train (the reported spread is over these) |
| `--max-epochs <n>` | cap epochs per training run |
| `--n-calls-arch / --n-calls-train / --n-calls-reg <n>` | trials in each search phase |
| `--device cpu` | force CPU (overrides `--colab-safe`'s `auto`) |
| `--resume` | resume the final training stage from its last checkpoint |

Everything else has a sensible default; you rarely need to touch it.
