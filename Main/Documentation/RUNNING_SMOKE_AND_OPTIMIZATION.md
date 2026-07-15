# Running the Smoke Tests and the Optimization

**A hands-on operational guide for the `Main/` layout: how to verify the
environment with the smoke tests, then launch and control the hyper-parameter
optimization.**

---

## 0. Provenance and scope

Everything below is read directly from the repository source in `Main/`
(`run_optimization.py`, `run_optimization_colab.py`,
`local_setup_and_run.py`, `colab_setup_and_run.py`, and the `Smoke_Tests/`
suite), and every command, flag, timing, and printed line quoted here was
produced by **actually executing the code**, not recalled. Flag help text is
verbatim from `run_optimization.py --help`. No literature was consulted; this
is an operational document, so the project's PubMed/bioRxiv sourcing rules do
not apply to it.

This guide is the *operational* companion to the deeper design docs that live
in `Patched/Optimization/` (`02_TECHNICAL.md`, `03_USAGE.md`,
`README_run_optimization.md`). Where those explain *why* the pipeline is built
as it is, this explains *how to run it* and *what you will see*. It does not
duplicate the config-field reference (that is a separate document) nor the
statistics of the train/val/test protocol.

**One correction you should know about up front.** The header docstring of
`Smoke_Tests/run_all_smoke_tests.py` claims that three suites
(`smoke_test_data_splits.py`, `smoke_test_metrics.py`,
`smoke_test_end_to_end.py`) "DO NOT EXIST in the repository". That docstring is
**stale**: all three now exist in `Smoke_Tests/` and the runner executes them.
The runner's actual behaviour (it filters `ORDER` to files that exist on disk)
is what this document describes. `--list` prints the real, current set.

---

## 1. The two things you run, and the order

There are two distinct activities, and they are meant to be run in this order:

1. **The smoke tests** -- fast, synthetic-data, CPU-only correctness harnesses.
   They train nothing real. Their job is to prove *this machine / this
   environment* can import the whole stack and run the pipeline end to end
   before you commit real time to it. Run these **first, once per fresh
   environment**.

2. **The optimization** -- the real run: `run_optimization.py` loads data,
   runs the hyper-parameter search phases, trains the final models, evaluates
   on the held-out test split, and writes the deliverable artifacts.

The golden rule, enforced by both `local_setup_and_run.py` (Cell 3) and
`colab_setup_and_run.py` (Cell 3): **never spend real training time on an
environment whose smoke test has not passed.** A failing smoke test on a new
machine is almost always an environment problem (a missing package, a wrong
torch build, a transfer-corrupted file), and it is far cheaper to find that in
18 seconds than an hour into a search.

---

## 2. A note on `sys.path`: why the modules must be importable by bare name

Every module in this pipeline imports its siblings by **bare name**
(`import run_optimization`, `from config import ...`). There is no package
`__init__.py` and no `dsn.` prefix. The practical consequence:

* **When you run a file that sits in `Main/`** (e.g.
  `python3 run_optimization.py ...`), Python automatically puts `Main/` on
  `sys.path[0]`, so all the bare-name imports resolve. Nothing extra needed.

* **When you run a file that sits in `Main/Smoke_Tests/`** (e.g.
  `python3 Smoke_Tests/smoke_test_run_optimization.py`), Python puts
  `Smoke_Tests/` on `sys.path[0]`, **not** `Main/`. The bare-name import of
  `run_optimization` then fails with `ModuleNotFoundError`. You must put
  `Main/` on the path yourself:

  ```bash
  # from inside Main/
  PYTHONPATH=. python3 Smoke_Tests/smoke_test_run_optimization.py --quick
  ```

  or, equivalently, `cd Smoke_Tests` and run with `PYTHONPATH=..`.

`local_setup_and_run.py` and `run_all_smoke_tests.py` both handle this for you
(they inject the parent directory into the child process's `PYTHONPATH`), so if
you drive the smoke tests through either of those you never have to think about
it. The rule only bites when you invoke a suite file **directly** from the
`Main/` directory.

---

## 3. Running the smoke tests

### 3.1 The fastest possible check: the driver smoke test

If you want a single command that proves the whole pipeline works on this
machine, run the driver's own smoke test in quick mode:

```bash
# from inside Main/
PYTHONPATH=. python3 Smoke_Tests/smoke_test_run_optimization.py --quick
```

Measured wall time: **about 18 s on one CPU core**. On success the last lines
are:

```
==========================================================================
ALL 20 RUN_OPTIMIZATION SMOKE CHECKS PASSED: A B C D E F G H I J K L M N O P Q R S T
==========================================================================
```

Exit code `0` means every check passed. This one suite is deliberately the most
load-bearing: it exercises the full driver -- pre-flights, data path, the
staged search wiring, the artifact tree, resume, and the honesty guarantees
(model selected on validation not test; leakage-free splits). What `--quick`
skips is only check **[U]**, the one end-to-end run that invokes the *real*
skopt Gaussian-process search (the other checks use a fast stub). Dropping
`--quick` runs all 21 checks (A-U) and takes a few minutes instead of 18 s;
use the full run when you have changed anything in the search layer itself.

The full check list (from the suite's own docstring):

| Check | What it proves |
|---|---|
| A | budget formula reproduces the two documented totals |
| B | batch-row formula `M = C * B_c * (1 + P + N)` |
| C | the shipped default window config is rejected, and the message names the fix; a feasible config passes |
| D | model sizes: meta-device counts equal real counts; the >1 GB warning fires |
| E | `build_traces`, synthetic mode |
| F | `build_traces`, numpy mode (incl. specs-relative path resolution) |
| G | `build_traces`, real mode: raises without `--engine-module`, works with one |
| H | the stale-cache fingerprint guard fires, and `--overwrite-cache` clears it |
| I | `--dry-run` trains nothing and writes no `results.json` |
| J | end-to-end with `--skip-search`: every documented artifact is produced |
| K | `best_model.pt` rebuilds and reproduces the reported test ARI exactly |
| L | the splits are leakage-free (test windows disjoint from train and val) |
| M | `best_model.pt` is selected on validation, never on test (non-vacuous) |
| N | search phases wired correctly: arch -> train HPs -> regularization |
| O | dropout pinned to 0 for the whole search, even if the input config sets it >0 |
| P | reproducibility: the same seed twice gives the identical test ARI |
| Q | headless: `plt.show` is patched to raise and is never reached |
| R | the driver and its whole import chain are pure ASCII (HPC transfer safety) |
| S | the stale-resume guard clears a stale `last.pt` unless `--resume` |
| T | the final seeds cannot collide with any search trial's seed block |
| U | end-to-end with the real skopt search (**skipped by `--quick`**) |

These are all *behavioural* assertions, not source inspections: e.g. "M" is
proven by injecting histories where the validation winner and the test winner
are different seeds, so a test-based selection would pick the wrong file and the
check would fail. It cannot pass vacuously.

### 3.2 The full suite: `run_all_smoke_tests.py`

To verify **every** module (not just the driver), use the orchestrator. It runs
the suites in dependency order -- foundational modules first -- so a failure
lands as close as possible to its cause (a broken `config.py` fails
`smoke_test_config` first, rather than surfacing as a confusing failure deep
inside `smoke_test_search` minutes later).

```bash
# from inside Main/Smoke_Tests/
PYTHONPATH=.. python3 run_all_smoke_tests.py            # all suites
PYTHONPATH=.. python3 run_all_smoke_tests.py --quick    # faster; passes --quick where accepted
PYTHONPATH=.. python3 run_all_smoke_tests.py --list      # print the order, run nothing
PYTHONPATH=.. python3 run_all_smoke_tests.py --only train search   # substring match
```

The current run order (from `--list`) is **15 suites**:

```
smoke_test_config.py
smoke_test_backbone.py                   (accepts --quick)
smoke_test_augmentation.py
smoke_test_data_pipeline.py
smoke_test_data_splits.py
smoke_test_metrics.py
smoke_test_inference.py
smoke_test_checkpoint.py
smoke_test_evaluate.py
smoke_test_burst_pipeline.py
smoke_test_train.py
smoke_test_search.py
smoke_test_end_to_end.py
smoke_test_run_optimization.py           (accepts --quick)
smoke_test_run_optimization_colab.py     (accepts --quick)
```

Only three suites define a `--quick` flag (backbone, run_optimization,
run_optimization_colab); the runner detects this per suite and passes `--quick`
only to those, so a global `--quick` never crashes a suite that would reject the
flag. Exit code `0` means every suite passed.

Two suites in this list guard the project's most load-bearing invariants and are
worth naming: `smoke_test_data_splits.py` (the leakage guarantee -- test
windows disjoint from train/val by construction) and `smoke_test_metrics.py`
(the objective the entire search maximises). They are now present and run; the
runner's stale docstring says otherwise, and should be disregarded.

### 3.3 Driving the smoke test from `local_setup_and_run.py`

If you are working in VS Code, `local_setup_and_run.py` Cell 3 runs the driver
smoke test for you (with the `PYTHONPATH` handling already built in) and
**refuses to proceed to training if it fails**. That is the recommended entry
point on a laptop: run Cells 1-3 once per environment, confirm
`smoke test passed: this machine is good to train on`, and only then move on.

---

## 4. Running the optimization

### 4.1 The one indispensable flag: `--config`

`run_optimization.py` will run with no config, but **do not**: the bare
dataclass defaults are geometrically **infeasible** (they ask for a 200 s window
against 120 s evaluation segments), and the run will be rejected by the window
pre-flight. Always pass a config. The help text says as much:

> `--config` ... Omit at your peril: the dataclass defaults are geometrically
> INFEASIBLE (window_s=200s vs 120s eval segments).

Two configs ship with the repo:

* **`config_toy.json`** -- a deliberately tiny end-to-end sanity run
  (8 s windows, 2 synthetic classes, 2 seeds, 3 epochs, 4-trial searches). The
  ARI it produces is meaningless; it exists so the *smallest possible real run*
  is genuinely small. Use it to confirm the *machine* trains, not to get a
  result.
* **`config_example.json`** -- a realistic starting point. Copy it, edit it, and
  point `--config` at your copy for real work. The field-by-field meaning of
  every key is covered in the separate config reference document.

### 4.2 The always-do-this-first pre-flight: `--dry-run`

Before any real run, do a dry run. It resolves the config, runs all three
pre-flights (window feasibility, model size, evaluability), prints the exact
compute budget, and trains **nothing**:

```bash
# from inside Main/
PYTHONPATH=. python3 run_optimization.py \
    --config config_toy.json \
    --out-dir ./dry_out --cache-dir ./dry_cache \
    --experiment-name dry --dry-run
```

Real output for the toy config:

```
[run] 4 trace(s), 2 phenotype(s), fs = 50 Hz, W = 400 samples (8 s)
[run] windows: train=116 val=20 test=20
[run] batch rows M = C*B_c*(1+P+N) = 2*2*(1+3+3) = 28 rows per batch
[run] arch-space model sizes: {"depth3_width1.5": 299676, ... "depth4_width2.5": 2997468}
[run] budget: {"phase1_arch": 8, "phase2_train": 8, "retune_arch": 0, "regularization": 8, "final": 2, "TOTAL_train_runs": 26}
[dry-run] 26 train() runs would be executed. No training performed.
```

The `TOTAL_train_runs` number is the single most useful pre-flight fact: **it is
how many times `train()` will be called**, and therefore (multiplied by your
measured seconds-per-run) the total wall-clock cost. Each search trial costs
`n_seeds` train runs; the budget line breaks that down per phase. If the total
is larger than you are willing to wait for, narrow the search **before** you
start it (fewer `--n-calls-*`, fewer `--n-seeds`, or a tighter search range in
the config) rather than killing a run midway.

### 4.3 The real run

The minimal real invocation, on any machine (the pipeline auto-detects CPU vs
GPU via `--device auto`):

```bash
# from inside Main/
PYTHONPATH=. python3 run_optimization.py \
    --config config_example.json \
    --out-dir ./out --cache-dir ./cache \
    --experiment-name run1 \
    --device auto --verbose
```

This executes the full pipeline:

```
resolve config -> resolve device -> seed -> PRE-FLIGHTS -> cache traces
  -> time-segment splits
  -> [PHASE 1: architecture] -> [PHASE 2: training HPs] -> [re-tune]
  -> [REGULARIZATION: dropout + weight decay]
  -> FINAL: train N_s models -> held-out TEST evaluation -> artifacts
```

On a laptop, the equivalent through `local_setup_and_run.py` is Cell 6
(`results = R.run(cfg, args)`), which calls the identical `run()`.

### 4.4 Controlling the stages

You do not have to run the whole thing every time. The `stages` flag group lets
you cut the pipeline down:

| Flag | Effect |
|---|---|
| `--skip-search` | Skip **all** HPO phases (arch, train HPs, re-tune, regularization) and just train + evaluate + save the architecture exactly as configured. Reduces the driver to "load data, train the configured net, score it, save it." |
| `--skip-regularization` | Run phases 1-2 but not the regularization (dropout + weight-decay) stage. |
| `--dry-run` | Resolve config, run pre-flights, print the budget, train nothing. |
| `--resume` | Resume the **final** training from its `last.pt`. Search phases are **not** resumable. |

`--skip-search` is the fastest way to get a real (if unoptimized) trained model
and a full artifact tree -- useful for validating your **data** end to end
before spending time on a search.

### 4.5 Overriding config values from the CLI

Every flag in the `overrides` group takes precedence over the JSON config (CLI
wins, always). The commonly useful ones:

| Flag | Overrides |
|---|---|
| `--device {cpu,cuda,auto}` | compute device |
| `--seed`, `--n-seeds` | RNG seed / number of final models trained |
| `--max-epochs` | training-loop length cap |
| `--num-workers` | DataLoader workers (set `0` on a memory-limited box) |
| `--train-stride-s`, `--eval-stride-s` | window strides for the two split roles |
| `--synthetic-duration-s` | length of each synthetic trace (synthetic mode) |
| `--n-calls-arch`, `--n-calls-train`, `--n-calls-reg` | trial counts per phase (each trial costs `n_seeds` train runs) |
| `--out-dir`, `--cache-dir`, `--experiment-name` | where artifacts and the trace cache go |

Example -- a quick real run capped hard for a small machine:

```bash
PYTHONPATH=. python3 run_optimization.py \
    --config config_example.json \
    --out-dir ./out --cache-dir ./cache --experiment-name quicklook \
    --device cpu --num-workers 0 \
    --n-seeds 2 --n-calls-arch 4 --n-calls-train 4 --n-calls-reg 4 \
    --verbose
```

### 4.6 Data modes

`--data-mode` selects where traces come from:

* **`synthetic`** (default in the toy/example configs) -- generated on the fly;
  no data files needed. This is what the smoke tests and `config_toy.json` use.
* **`numpy`** -- your own `.npz` traces, described by a specs JSON:
  `--npz-specs specs.json`, a list of `{name, condition, path}`. Relative paths
  in the specs are resolved relative to the specs file.
* **`real`** -- live MEA recordings, wired through
  `data_pipeline.NeuronalTracesProvider`. You must name the engine module that
  exports `Neuronal_traces` with `--engine-module`, and may pass provider kwargs
  as a JSON dict via `--engine-kwargs` (e.g.
  `'{"w_size": 0.02, "t_rec": 600.0}'`). Without `--engine-module` the real
  branch raises a `NotImplementedError` that names the fix.

### 4.7 The trace cache and its fingerprint guard

Traces are cached **once** into `--cache-dir` so an HPO run does not recompute
them per trial. This is a real speedup but has one sharp edge, guarded for you:
if you change the **data source** (e.g. `synthetic_duration_s`, or switch from
synthetic to numpy) while pointing at the **same** `--cache-dir`, the old traces
would be silently reused. The driver writes a fingerprint of the data-source
config into the cache and **refuses to proceed on a mismatch**, naming the fix.
The fix is either `--overwrite-cache` (recompute every trace) or a fresh
`--cache-dir`. If you see the fingerprint-mismatch error, that guard is doing
its job -- do not work around it by deleting the check.

---

## 5. What a successful run produces

All artifacts land under `<out_dir>/<experiment_name>/`:

```
<out_dir>/<experiment_name>/
  config_input.json       the config as resolved (file + CLI), before search
  config_best.json        the config after every search phase
  results.json            the deliverable (all metrics, per-seed detail)
  figures/
    pdp_phase1_arch.png, pdp_phase2_train.png, pdp_retune_arch.png,
    pdp_regularization.png       partial-dependence plots per search phase
    embedding_test_seed_<n>.png  test-split embedding scatter, per final seed
  checkpoints/
    seed_<n>/{last,best}.pt      resumable, written DURING the final train
    final_seed_<n>.pt            self-describing, best-epoch weights
    best_model.pt                the deployable model (see below)
```

Two artifacts deserve emphasis:

* **`results.json`** is the deliverable. Its `test` block holds the headline
  numbers -- e.g. `results["test"]["ari"]["mean"]` / `["std"]` aggregated over
  the final seeds, and `results["test"]["best_model"]` (the path to
  `best_model.pt`). The per-seed breakdown is under
  `results["test"]["per_seed"]`.

* **`best_model.pt`** is the one deployable model, and it is **selected on the
  validation ARI, never on the test ARI**. Of the `N_s` final models, the one
  with the highest best-epoch validation ARI is copied here. Selecting on the
  test score would fold the held-out split into model selection and destroy the
  honesty of the reported test number. The file is self-describing: it embeds
  its own config, so it rebuilds with **zero** hyper-parameters supplied by the
  caller:

  ```python
  from checkpoint import rebuild_model_from_checkpoint
  model, ckpt = rebuild_model_from_checkpoint("out/run1/checkpoints/best_model.pt")
  model.eval()
  # ckpt["config"]["backbone"]["embedding_size"], etc. all present
  ```

  This is exactly what smoke-test check **[K]** proves end to end: it rebuilds
  from `best_model.pt`, re-embeds the test split, re-clusters, and confirms the
  ARI matches `results.json`.

---

## 6. Ten behaviours the driver adds beyond the base design docs

The driver is pure orchestration -- it contains no science -- but it does add
ten operational guarantees on top of the tested modules it calls. Each is
marked `[ADDED]` at its definition in `run_optimization.py`. Knowing they exist
explains several messages you may see:

1. **Dropout pinned to 0 for the whole search.** Tuned only in the
   regularization stage, so phases 1/2/re-tune all run at dropout 0 regardless
   of the input config. (Check O.)
2. **`best_model.pt` selected on validation, never test.** (Checks M, ADDED 2.)
3. **Data fingerprint guards the trace cache.** (Section 4.7; check H.)
4. **Model sizes counted on the meta device** -- the size pre-flight allocates
   zero bytes. (Check D.)
5. **Both block families reported at each corner** -- ResNet (family 0) is ~3x
   heavier than ResNeXt (family 1) at the same depth/width, and the search
   samples both.
6. **Final seeds come from a disjoint block** -- the final fit is never scored
   on the very seed draws that selected the config. (Check T.)
7. **Stale-resume guard** -- clears a seed's checkpoint dir before a fresh
   final train unless `--resume` is passed. (Check S.)
8. **`data_mode="real"` is wired** through `NeuronalTracesProvider`. (Section
   4.6; check G.)
9. **Evaluability pre-flight** -- warns per split, per class, when a split is
   non-empty but unscorable (K-means with `K = C` needs at least `C` windows;
   silhouette needs at least 2 per class). (Related: ADDED 9.)
10. **`--skip-regularization`, and `--dry-run` reports the budget for the
    stages that will actually run.** (Sections 4.2, 4.4.)

---

## 7. A recommended first session, start to finish

On a fresh machine, in order:

```bash
cd Main/

# 1. verify the environment can run the pipeline at all (~18 s)
PYTHONPATH=. python3 Smoke_Tests/smoke_test_run_optimization.py --quick

# 2. (optional, thorough) verify every module (~a few minutes)
cd Smoke_Tests && PYTHONPATH=.. python3 run_all_smoke_tests.py --quick && cd ..

# 3. dry-run your real config to see the budget before committing time
PYTHONPATH=. python3 run_optimization.py --config config_example.json \
    --out-dir ./out --cache-dir ./cache --experiment-name run1 \
    --dry-run --verbose

# 4. the real run
PYTHONPATH=. python3 run_optimization.py --config config_example.json \
    --out-dir ./out --cache-dir ./cache --experiment-name run1 \
    --device auto --verbose

# 5. read the headline number
python3 -c "import json; r=json.load(open('out/run1/results.json')); \
print('TEST ARI %.4f +/- %.4f' % (r['test']['ari']['mean'], r['test']['ari']['std']))"
```

In VS Code, steps 1, 3, 4 and 5 are Cells 3, 5, 6 and 7 of
`local_setup_and_run.py` respectively -- run that file cell by cell for the same
sequence with the `sys.path` and pre-flight handling done for you.
