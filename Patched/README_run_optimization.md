# `run_optimization.py` — the missing driver

`02_TECHNICAL.md` (§10) and `03_USAGE.md` describe a driver called
`run_optimization.py` in full detail — the pipeline order, the pre-flights, the
artifact tree, the `results.json` schema, the CLI. **That file is not in the
repository.** Neither are `config_example.json` or `config_toy.json`, which the
usage guide tells you to start from. Every other module is present and
individually smoke-tested; nothing wires them together.

This is that file, reconstructed to the documented spec, plus ten flagged
additions. It is the single script that **loads the data, trains / optimizes the
architecture, and saves the trained architecture.**

---

## 1. File layout (this matters)

The Topic-3 modules import `backbone`, `augmentation` and `data_pipeline` **by
bare name**, but those live in sibling directories in the repo. As
`03_USAGE.md` §1.3 says: *put all modules in one directory.* Flatten like this:

```
project/
├── backbone.py              <- from Patched/CNN_Backbone/
├── augmentation.py          <- from Patched/Augmentation/
├── data_pipeline.py         <- from Patched/Optimization/   (see note)
├── config.py   preprocessing_cache.py   data_splits.py
├── metrics.py  checkpoint.py            inference.py
├── train.py    evaluate.py              search.py           <- Patched/Optimization/
├── run_optimization.py           <- NEW (the driver)
├── smoke_test_run_optimization.py <- NEW (its harness)
├── config_example.json           <- NEW (feasible starting point, 243 runs)
└── config_toy.json               <- NEW (30-second sanity run, 26 runs)
```

> **Note on `data_pipeline.py`.** It exists in **both** `Augmentation/` and
> `Optimization/`. They are functionally identical — the only difference is two
> docstring lines. Use the **`Optimization/`** copy: it is the pure-ASCII one.
> The `Augmentation/` copy contains an em-dash and a combining tilde, which a
> cp1252 transfer (MobaXterm, copy-paste, `scp` from Windows) can corrupt into a
> `SyntaxError` that only surfaces at job-submission time on the cluster.

Verified: the driver and all 12 modules in its import chain are **pure ASCII**
(byte-scanned, not eyeballed).

## 2. Install

```bash
pip install pytorch-metric-learning scikit-optimize      # add --break-system-packages on davinci
python3 -c "import torch, sklearn, skopt, pytorch_metric_learning, matplotlib; print('ok')"
```

## 3. Quick start

```bash
# (1) prove the driver works, on synthetic data, no files needed  (~3-5 min)
python3 smoke_test_run_optimization.py            # 21/21
python3 smoke_test_run_optimization.py --quick    # ~1 min, skips the skopt e2e

# (2) prove the WHOLE pipeline works end to end   (~50 s)
python3 run_optimization.py --config config_toy.json

# (3) NEVER skip this: how many trainings will this cost?
python3 run_optimization.py --config config_example.json --dry-run
#   -> [dry-run] 243 train() runs would be executed. No training performed.

# (4) how long is ONE run?  (the budget tells you how many, not how long)
python3 run_optimization.py --config config_example.json \
        --skip-search --n-seeds 1 --max-epochs 5 --experiment-name timing
#   -> read the "[run]   seed 0 (...): 5 epoch(s) in X s (Y s/epoch)" line, then
#      total hours ~= Y * max_epochs * TOTAL_train_runs / 3600

# (5) the real run
python3 run_optimization.py --config my_config.json --verbose
```

**Do not run without `--config`.** The dataclass defaults are geometrically
infeasible: `window_s = 200 s` against a 600 s recording split 60/20/20 leaves
120 s val/test segments, so both get **zero windows**. The driver catches this
before doing any work and names the fix — but `config_example.json` is the
place to start.

## 4. Your own data

```bash
# .npz per well:  keys ifr_trace (K,) float32  and  fs_ifr (float). All traces share fs.
# specs.json:     [{"name": "control_A1", "condition": 0, "path": "well_A1.npz"}, ...]
#                 (relative paths resolve against the specs file's own directory)
python3 run_optimization.py --config my_config.json --data-mode numpy \
        --npz-specs /data/specs.json --verbose

# .mat via the group's engine (previously an unwired gap - see ADDED 8):
python3 run_optimization.py --config my_config.json --data-mode real \
        --specs-json /data/specs_real.json --engine-module my_engine_module
```

## 5. What it does

```
resolve config -> resolve device -> seed -> PRE-FLIGHTS -> cache traces
  -> time-segment splits (leakage-free by construction)
  -> [PHASE 1: architecture]      4 HPs, optimizer held fixed
  -> [PHASE 2: training HPs]      5 HPs, architecture fixed, betas as log(1-beta)
  -> [optional re-tune]           narrowed arch space, under the tuned optimizer
  -> [REGULARIZATION]             dropout + weight decay, everything else fixed
  -> FINAL: train N_s models -> held-out TEST evaluation -> artifacts
```

`--skip-search` collapses every bracketed stage, reducing the driver to
"load the data, train the configured architecture, evaluate it, save it".

Four pre-flights run **before any training**, because all four failure modes
actually bite: an infeasible window geometry (zero-window split), an
architecture corner that gets the job SIGKILLed by the OOM-killer (uncatchable —
no traceback, no results), the 61x augmentation batch multiplier, and a budget
you did not realise you had signed up for.

**Artifacts** (`<out_dir>/<experiment_name>/`):

```
config_input.json    config_best.json    results.json
figures/    pdp_phase{1,2}_*.png, pdp_regularization.png, embedding_test_seed_<n>.png
checkpoints/seed_<n>/{last,best}.pt      resumable
            final_seed_<n>.pt            self-describing (best-epoch weights)
            best_model.pt                <- the deployable model
```

Reload the trained architecture with **zero hyper-parameters in your code** —
the architecture is rebuilt from the config embedded in the checkpoint:

```python
from checkpoint import rebuild_model_from_checkpoint
model, ckpt = rebuild_model_from_checkpoint("out/run1/checkpoints/best_model.pt")
model.eval()
# z = model(x)   # x: (M, W) -> z: (M, E), rows L2-normalized
```

## 6. Additions beyond the documented spec (each flagged `[ADDED n]` in the source)

| # | Addition | Why |
|---|---|---|
| 1 | **Dropout pinned to 0 for the whole search** | Decision 11 tunes dropout last. `search.config_from_arch_point` pins it for phase 1, but `config_from_train_point` does **not** — it inherits from the base config. A config with `dropout > 0` would silently run phase 1 at 0 and phase 2 at `>0`: two phases under different regularization, scores not comparable. |
| 2 | **`best_model.pt`, selected on VALIDATION** | Selecting the deployable model by its *test* ARI would fold the held-out split into model selection and the reported test number would stop being out-of-sample. |
| 3 | **Data fingerprint guards the trace cache** | `cache_traces(overwrite=False)` skips any trace whose `.npz` exists — that is what makes a 753-trial study pay the trace cost once. It also means changing `synthetic_duration_s`, or repointing `npz_specs`, while keeping the same `cache_dir` would silently train on the **old** traces. Now refused, with the fix named. |
| 4 | **Model sizes counted on the `meta` device** | The size pre-flight must not allocate the 2.6 GB it exists to warn about. Verified to give counts identical to a real build. |
| 5 | **Both block families at every corner** | ResNet is ~3x heavier than ResNeXt at the same (depth, width) and the search samples **both**; reporting one family understates the worst corner. |
| 6 | **Final seeds in a disjoint block** | Trial `t` owns `[s0+t*Ns, s0+t*Ns+Ns)`. Final models use `s0 + 10_000_000 + n`, so the final fit is not scored on the very draws that selected the config. |
| 7 | **Stale-resume guard** | `train()` resumes *automatically* from any `last.pt` in the ckpt dir it is handed. A re-run with a changed architecture would load the old weights into the new model. Cleared unless `--resume`. |
| 8 | **`data_mode="real"` wired** | Listed as a known gap in `02_TECHNICAL.md` §15. Now wired through `NeuronalTracesProvider`; name the engine with `--engine-module`. Without it, the same fix-naming `NotImplementedError` as before. |
| 9 | **Evaluability pre-flight** | A split can be non-empty yet unscorable: K-means with `K = C` needs `>= C` windows and a silhouette needs `>= 2` windows per class. Warned per split, per class. |
| 10 | **`--skip-regularization`; NaN-safe, numpy-safe `results.json`** | skopt returns `np.int64`/`np.float64` (which `json.dump` rejects) and failed trials carry `NaN` (which is not valid JSON). Both are sanitised. |

## 7. Smoke test

```bash
python3 smoke_test_run_optimization.py          # all 21 checks, ~3-5 min
python3 smoke_test_run_optimization.py --quick  # ~1 min (skips [U])
echo $?                                         # 0 = pass
```

Self-contained: no data files, no GPU, no display. It observes behaviour rather
than reading the source — `--dry-run` "trains nothing" is proven by patching
`train()` to **raise** in *both* namespaces that hold a reference to it, and the
validation-based model selection is proven with injected histories where the
validation winner and the test winner are **different seeds**, so the check
cannot pass vacuously.

| | Check |
|---|---|
| A | budget reproduces the documented **753** and **243** totals |
| B | `M = C*B_c*(1+P+N)` exact: 976 / 28 / 136 |
| C | the shipped default (200 s window) is **rejected** with a fix-naming message |
| D | meta counts == real counts; reproduces the docs' corner table (300192 / 2450144 / 17734112 / **214441248**); OOM warning fires |
| E–H | data path: synthetic, numpy (relative paths, missing file, bad schema), real (with and without an engine), stale-cache guard |
| I | `--dry-run` trains **nothing** (train patched to raise) and writes no `results.json` |
| J | every documented artifact is produced; `results.json` is strict, parseable, complete |
| K | `best_model.pt` **rebuilds from its embedded config alone and reproduces the reported test ARI exactly** |
| L | splits are **leakage-free** (every test window vs every train/val window) |
| M | `best_model.pt` selected on validation, **non-vacuously** |
| N | phase winners carried forward with the right precedence (`beta = 1-u`; regularization's `weight_decay` overrides phase 2's) |
| O | dropout pinned to 0 across the search even when the input config sets 0.25 |
| P | reproducible: same seed twice -> identical test ARI |
| Q | headless: `plt.show` patched to raise, never reached |
| R | driver + all 12 chain modules are **pure ASCII** |
| S | stale-resume guard | 
| T | final seeds cannot collide with a search trial block |
| U | end-to-end through the **real** skopt search |

Run it after any change to a module, and before submitting a long cluster job.

## 8. On the cluster (davinci / PBS)

```bash
#!/bin/bash
#PBS -N mea_hpo
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=24:00:00
cd $PBS_O_WORKDIR
source activate brian_env

python3 run_optimization.py \
    --config my_config.json --device cuda \
    --out-dir $PBS_O_WORKDIR/out --cache-dir $PBS_O_WORKDIR/cache \
    --experiment-name hpo_run1 --verbose
```

Before submitting: `--dry-run` for the count, time one run, run the smoke test on
the login node, and check the reported max-parameter corner against the node's RAM.

> **The trap the docs flag and the driver cannot fix for you** (`02_TECHNICAL.md`
> §15): if trials stop short of convergence, the search ranks configurations by
> *how fast they train*, not how well. Calibrate `max_epochs` on one timed trial
> before committing the budget.

## 9. Running on Google Colab

`run_optimization.py` needs zero logic changes to run on Colab — device,
`out_dir`, and `cache_dir` were already plain config fields. What differs is the
environment: two packages aren't preinstalled, `/content/` is wiped on
disconnect, and a free-tier session caps out around 12 hours with search phases
that **cannot resume** across a disconnect (only the final training stage can).
`run_optimization_colab.py` is a thin wrapper — no science, no orchestration
logic of its own — that handles exactly these differences and then calls the
same `run()` this whole document describes.

```bash
# one command: install deps, mount Drive, run, and mirror every phase to
# Drive AS IT COMPLETES (the real protection against a mid-search disconnect)
python3 run_optimization_colab.py --config config_toy.json \
    --mount-drive --drive-out-dir /content/drive/MyDrive/dsn_out --verbose

# find out whether a real config fits in one session BEFORE committing to it
# (automates the timing recipe from Step 2 above)
python3 run_optimization_colab.py --config config_example.json \
    --time-one-epoch --session-hours 12

# a memory-safe preset for a free-tier T4 (16 GB VRAM)
python3 run_optimization_colab.py --config my_config.json --colab-safe \
    --mount-drive --drive-out-dir /content/drive/MyDrive/dsn_out
```

| Flag | What it does |
|---|---|
| `--mount-drive` | mounts Google Drive; a clean no-op off Colab |
| `--drive-out-dir PATH` | mirrors `out_dir` to Drive after **every** phase and every final seed, not just at the end |
| `--drive-cache-dir PATH` | mirrors the trace cache to Drive once, right after caching |
| `--colab-safe` | caps `search.depth_exponent_range` at 5, forces `num_workers=0`, defaults `device` to `auto` — never widens an already-narrower config, and any explicit `--device` you also pass still wins |
| `--time-one-epoch` | measures real seconds/epoch on a short run and projects the full configured budget against `--session-hours`, then exits without running the full pipeline |
| `--no-auto-install` | skip the automatic `pip install scikit-optimize pytorch-metric-learning` |

Every `run_optimization.py` flag still works unchanged — this parser is that
parser, extended, not replaced.

**The one thing no flag can fix:** search phases aren't resumable. Use
`--time-one-epoch` *before* a long run to see whether it fits in one session;
if it doesn't, narrow the search or use `--skip-search` (final-stage training
*is* resumable via `--resume`).

Test with `python3 smoke_test_run_optimization_colab.py` (13 checks; embeds and
re-runs the base 21-check suite as a regression control, since the
`on_stage_complete` hook that makes Drive-syncing possible is an additive edit
to the tested driver).
