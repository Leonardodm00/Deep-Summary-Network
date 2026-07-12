# Usage Guide

## How to run the pipeline, the smoke tests, and the config

**Companion to** `01_THEORY.md` (why) and `02_TECHNICAL.md` (what).
This document covers **how to actually use it**.

---

## 1. Installation

```bash
pip install pytorch-metric-learning scikit-optimize
```

On the cluster (davinci, inside your conda env), you may need:

```bash
pip install pytorch-metric-learning scikit-optimize --break-system-packages
```

Everything else — torch, numpy, scipy, scikit-learn, matplotlib — is assumed present.

**Verify the install before anything else:**

```bash
python3 -c "import torch, sklearn, skopt, pytorch_metric_learning, matplotlib; print('ok')"
```

### File layout

Put all modules in one directory. They import each other by bare name (`from train import train`), so they must be on the same `PYTHONPATH`.

```
project/
├── config.py              backbone.py          augmentation.py
├── data_pipeline.py       preprocessing_cache.py   data_splits.py
├── metrics.py             checkpoint.py        inference.py
├── train.py               evaluate.py          search.py
├── run_optimization.py
├── config_example.json         ← your starting point (a real, feasible run)
├── config_toy.json             ← the 30-second end-to-end sanity run
└── smoke_test_*.py             ← nine test files
```

---

## 2. The smoke tests

Each module has one. They are **standalone**: no data files, no GPU, no display, no cluster. Every one is CPU-only and uses synthetic data.

### 2.1 Run them all

```bash
for t in config data_splits metrics checkpoint inference train evaluate search end_to_end; do
    echo "--- $t ---"
    python3 smoke_test_$t.py || echo "FAILED: $t"
done
```

Each prints a labelled check line and ends with `ALL … SMOKE TESTS PASSED`. Exit code 0 = pass.

### 2.2 What each one covers, and roughly how long

| Test | Time | What it proves |
|---|---|---|
| `smoke_test_config.py` | ~1 s | JSON round-trip is exact; validation fires; on-disk JSON is ASCII |
| `smoke_test_data_splits.py` | ~10 s | **Splits are leakage-free**; window provenance hasn't drifted; every phenotype in every split |
| `smoke_test_metrics.py` | ~5 s | ARI/AMI/silhouette correct; `eff_rank` matches a known participation ratio |
| `smoke_test_checkpoint.py` | ~5 s | **Resume ≡ uninterrupted**; a crashed write leaves the old checkpoint intact |
| `smoke_test_inference.py` | ~20 s | Embedding rows are **distinct**; no cross-sample leakage; model mode restored |
| `smoke_test_train.py` | ~5 min | It **learns** (ARI → 1.0); early stopping fires; **best-epoch** weights restored; determinism |
| `smoke_test_evaluate.py` | ~2 min | **The plot's colours ARE the metric's labels**; headless; no re-clustering |
| `smoke_test_search.py` | ~3 min | All 4 `get_newspace` bugs fixed; the objective **provably calls `train()`** |
| `smoke_test_end_to_end.py` | ~5 min | **The whole pipeline**, plus the three pre-flight checks |

Total: roughly **15–20 minutes** on a CPU.

### 2.3 When to run which

- **After installing, on a new machine or a fresh env:** run all nine. This is your "does anything work here" check.
- **After changing one module:** run that module's test *and* `smoke_test_end_to_end.py`. The latter catches integration breakage the unit test cannot.
- **Before submitting a long cluster job:** run `smoke_test_end_to_end.py`. It exercises every code path the real run will hit, in about five minutes, on synthetic data.
- **After a torch / sklearn / skopt upgrade:** run all nine. Several tests assert *library* behaviour (e.g. that skopt raises on a degenerate range) and will tell you if an upgrade changed it.

### 2.4 Reading a failure

The tests are deliberately loud. A failure prints the assertion with the actual values:

```
AssertionError: batch_size=1 moved the embedding by 0.0034 (>= tol 1e-05):
cross-sample dependence, not float noise
```

Two failures worth knowing how to interpret:

**`smoke_test_train.py [C] learns` fails with a low ARI.** This is *usually* an under-powered test rather than a broken trainer — the check needs enough gradient steps to converge. If you've reduced `max_epochs` or `batches_per_epoch` in the test, that's why. (See §7.2: the same trap applies to your real runs.)

**Anything mentioning `eff_rank`.** That is the collapse tripwire. A `mean_pairwise_cos` near 1.0 is *normal* and expected on this data (see `01_THEORY.md` §8.1) — do not chase it. A falling `eff_rank` is the real signal.

---

## 3. Your first real run

**Do not use the defaults.** They are infeasible — `window_s = 200 s` against a 600 s recording split 60/20/20 leaves val/test segments of only 120 s, so both get **zero windows** and the run dies. Start from `config_example.json`.

### Step 1 — Check the budget before doing anything

```bash
python3 run_optimization.py --config config_example.json --dry-run
```

This prints the resolved config and, at the bottom:

```
[dry-run] 243 train() runs would be executed. No training performed.
```

**This number is the single most important one in the project.** It trains nothing.

### Step 2 — Time one run

The budget tells you *how many* training runs. Only a timed trial tells you *how long each takes*.

```bash
python3 run_optimization.py --config config_example.json \
    --skip-search --n-seeds 1 --max-epochs 5 --experiment-name timing
```

Read the `[run]` timing line, divide by 5 → seconds per epoch. Then:

$$
\text{total hours} \;\approx\; \frac{(\text{s/epoch}) \times \texttt{max\_epochs} \times \texttt{TOTAL\_train\_runs}}{3600}
$$

**A worked example, measured on a CPU:** `config_example.json` runs at **17 s/epoch**. With `max_epochs = 60` and 243 runs:

$$
\frac{17 \times 60 \times 243}{3600} \;\approx\; \mathbf{69\ \text{hours}}
$$

That is why you check first. On a GPU it will be far less — but *measure it*, don't assume.

### Step 3 — Run the toy pipeline end to end

Before committing the real thing, prove the whole path works on synthetic data in about 30 seconds:

```bash
python3 run_optimization.py --config config_toy.json
```

`config_toy.json` ships alongside `config_example.json`. It is deliberately tiny: 2 phenotypes × 2 wells, 8 s windows, 3 epochs, 4 trials per phase, and — critically — **3 positives and 3 negatives instead of the default 30**.

> **Why the toy run needs its own config.** If you pass only CLI flags and no `--config`, you inherit the **default** augmentation (`n_positives = n_negatives = 30`), which makes a 2-class batch $2 \times 8 \times 61 = 976$ rows and will OOM a small box. This is the §4.4 trap, and it is easy to walk into. The toy config exists precisely so the smallest possible run is genuinely small.

You should see the pipeline walk through every phase and finish with:

```
[run] batch rows M = C*B_c*(1+P+N) = 2*2*(1+3+3) = 28 rows per batch
[run] PHASE 1: architecture (4 trials x 2 seeds)
[run] PHASE 2: training HPs (4 trials x 2 seeds)
[run] REGULARIZATION: dropout + weight decay (4 trials x 2 seeds)
[run] FINAL: training 2 model(s) and evaluating on the HELD-OUT TEST split
[run] TEST  ARI 0.1224 +/- 0.0881 | AMI ... | silhouette ...
[run] results -> out/toy/results.json  (29.7 s)
```

(The ARI is meaningless at this scale — you are checking that it *runs*, not that it *works*.)

### Step 4 — The real run

```bash
python3 run_optimization.py --config my_config.json --verbose
```

---

## 4. The config file

### 4.1 How resolution works

```
dataclass defaults  →  overridden by  →  JSON file  →  overridden by  →  CLI flags
```

The JSON is **partial**: omit any section and its defaults apply. You only write what you're changing.

### 4.2 A minimal working config

```json
{
  "data": {
    "data_mode": "synthetic",
    "window_s": 30.0,
    "train_stride_s": 15.0,
    "eval_stride_s": 30.0,
    "split_fractions": [0.6, 0.2, 0.2],
    "synthetic_duration_s": 600.0,
    "synthetic_fs": 50.0,
    "synthetic_n_per_class": [3, 3],
    "augmentation": {"fs": 50.0, "n_positives": 8, "n_negatives": 8,
                     "shift_magnitude_s": 5.0}
  },
  "backbone": {"stem_width": 16},
  "train": {"windows_per_condition": 4, "batches_per_epoch": 0,
            "n_seeds": 3, "max_epochs": 60, "patience": 10},
  "search": {"depth_exponent_range": [3, 5],
             "width_multiplier_range": [1.5, 2.5],
             "embedding_size_range": [8, 16],
             "n_calls_arch": 30, "n_calls_train": 30},
  "regularization": {"n_calls": 20},
  "runtime": {"device": "cpu", "num_workers": 0, "out_dir": "out",
              "cache_dir": "cache", "experiment_name": "run1"}
}
```

This is `config_example.json`, and it is **validated feasible**: segments 360/120/120 s, batch = 136 rows, max 24 M params, 243 runs.

### 4.3 The fields you will actually touch

#### `data` — the window geometry

| Field | Meaning | How to choose |
|---|---|---|
| `window_s` | Window length in seconds | **Must fit in the shortest segment.** Long enough to contain several network bursts. |
| `train_stride_s` | Step between train windows | `< window_s` → overlap → more training windows. Half the window is a reasonable start. |
| `eval_stride_s` | Step between val/test windows | **Set `>= window_s`** so eval windows are disjoint. Overlapping eval windows bias every metric. |
| `split_fractions` | `[train, val, test]` along time | Must sum to 1. |
| `synthetic_n_per_class` | Traces per phenotype | `[3, 3]` = 2 phenotypes, 3 wells each. `[3, 3, 3]` = 3 phenotypes. |

**The one arithmetic rule you must satisfy:**

$$
\texttt{window\_s} \;\le\; \min_k \bigl( f_k \times \text{recording duration} \bigr)
$$

With a 600 s recording and `[0.6, 0.2, 0.2]`, the shortest segment is $0.2 \times 600 = 120$ s, so `window_s` must be $\le 120$. The driver checks this **before doing any work** and names the fix.

#### `train` — the training loop

| Field | Meaning | Notes |
|---|---|---|
| `windows_per_condition` | $B_c$: source windows per phenotype per batch | **This is not the batch size.** See §4.4. |
| `batches_per_epoch` | `0` → derive one pass over the training windows | Leave at 0 unless you know why not. |
| `n_seeds` | $N_s$: models trained per configuration | **Multiplies your entire budget.** 3 is a reasonable minimum for an honest std. |
| `max_epochs` | Ceiling on epochs | See §7.2 — the most common way to get a meaningless search. |
| `patience` | Early-stopping patience | Must be `<= max_epochs`, or it can never fire. |
| `mining_strategy` | `"hard"` or `"easy_positive"` | |
| `margin` | Cosine-similarity gap | **Not an angle.** |

#### `search` — the budget

| Field | Cost |
|---|---|
| `n_calls_arch` | $\times\, N_s$ training runs |
| `n_calls_train` | $\times\, N_s$ training runs |
| `depth_exponent_range` | **See §4.5 — this is the memory dial** |

#### `runtime`

| Field | Notes |
|---|---|
| `device` | `"cpu"` (default), `"cuda"`, or `"auto"` (cuda iff available) |
| `num_workers` | **Use 0** unless you've profiled otherwise. Each worker forks the process. |
| `out_dir`, `cache_dir` | No hard-coded paths anywhere; these drive everything. |
| `seed` | Fix it. Everything downstream is reproducible from it. |

### 4.4 The batch-size trap

`windows_per_condition = 8` does **not** mean 8 rows per batch. The augmentation expands every source window into `1 + n_positives + n_negatives` rows:

$$
M \;=\; C \times \texttt{windows\_per\_condition} \times \bigl(1 + \texttt{n\_positives} + \texttt{n\_negatives}\bigr)
$$

With the **defaults** ($C = 2$, $B_c = 8$, $P = N = 30$):

$$
M \;=\; 2 \times 8 \times 61 \;=\; \mathbf{976\ \text{rows}}
$$

Sixty-one times what you'd guess from `windows_per_condition` alone. The driver prints this:

```
[run] batch rows M = C*B_c*(1+P+N) = 2*4*(1+8+8) = 136 rows per batch
```

**If you hit OOM, this is the first dial to turn** — reduce `n_positives` / `n_negatives` or `windows_per_condition`.

### 4.5 The memory trap

Parameter count is roughly **exponential** in `depth_exponent`:

| `depth_exponent` | parameters | RAM (weights + AdamW) |
|---|---|---|
| 3 | 0.3 M | negligible |
| 4 | 2.5 M | ~0.03 GB |
| 5 | 20 M | ~0.25 GB |
| 6 | **214 M** | **~2.6 GB** |

The default `depth_exponent_range = [3, 6]` spans a **~700× parameter range**. The search *will* sample depth 6 during its random-initialisation phase.

A soft out-of-memory error raises, and the search correctly scores that trial as FAILED and learns to avoid the region. But a **Linux OOM-killer SIGKILL cannot be caught** — it takes the whole study with it, with no traceback and no results.

The driver reports the corners up front:

```
[run] arch-space model sizes: {"depth3_width1.5": 72532, ..., "depth5_width2.5": 24340416}
```

and **warns** if the space exceeds ~1 GB. If your box can't hold the big corner, **narrow `depth_exponent_range`.**

---

## 5. Using your own data

Synthetic mode needs no files. For real recordings, use `data_mode: "numpy"`.

### Step 1 — Pre-compute your traces to `.npz`

Each file needs exactly two keys:

```python
np.savez("well_A1.npz",
         ifr_trace=trace,     # (K,) float32 — the population rate signal
         fs_ifr=50.0)         # float — sampling rate in Hz
```

**All traces must share the same `fs`.** The loader raises if they disagree, because windowing by seconds would otherwise be inconsistent.

### Step 2 — Write a specs JSON

```json
[
  {"name": "control_A1", "condition": 0, "path": "/data/well_A1.npz"},
  {"name": "control_A2", "condition": 0, "path": "/data/well_A2.npz"},
  {"name": "patho_B1",   "condition": 1, "path": "/data/well_B1.npz"},
  {"name": "patho_B2",   "condition": 1, "path": "/data/well_B2.npz"}
]
```

`condition` is the phenotype label, `0 .. C-1`. `name` must be filesystem-safe and unique.

### Step 3 — Point the config at it

```json
{"data": {"data_mode": "numpy", "npz_specs": "/data/specs.json",
          "window_s": 30.0, "train_stride_s": 15.0, "eval_stride_s": 30.0}}
```

### On `.mat` files

`data_mode: "mat"` is **not wired in**. The provider exists (`data_pipeline.NeuronalTracesProvider`, which wraps the engine's `Neuronal_traces`); it needs a branch in `run_optimization.build_traces`. The code raises a `NotImplementedError` that says exactly this. Converting to `.npz` once (Step 1) is usually simpler and has the side benefit of caching.

---

## 6. Running on the cluster (davinci / PBS)

The pipeline is designed to run unattended: no display, no interactive prompts, no hard-coded paths.

### A PBS script

```bash
#!/bin/bash
#PBS -N mea_hpo
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=24:00:00
#PBS -o mea_hpo.out
#PBS -e mea_hpo.err

cd $PBS_O_WORKDIR
source activate brian_env

python3 run_optimization.py \
    --config my_config.json \
    --device cuda \
    --out-dir $PBS_O_WORKDIR/out \
    --cache-dir $PBS_O_WORKDIR/cache \
    --experiment-name hpo_run1 \
    --verbose
```

### Before you submit

1. **`--dry-run` first.** Get the run count.
2. **Time one run** (§3, Step 2). Multiply. Compare against your walltime.
3. **Run `smoke_test_end_to_end.py` on the login node.** Five minutes, and it exercises every path the job will hit.
4. **Check the memory pre-flight output** against the node's RAM.

### If the job dies

| Symptom | Likely cause |
|---|---|
| `Killed` with no traceback | **OOM-killer.** Reduce `depth_exponent_range` (§4.5) or the batch multiplier (§4.4). |
| `SyntaxError` on a non-ASCII byte | A file got re-encoded in transfer. Every source file here is pure ASCII by design — re-transfer in binary mode. |
| Hangs with no output | Should be impossible (no interactive calls), but check that `matplotlib` isn't being imported before `evaluate.py` forces the Agg backend. |
| `ValueError: infeasible window / split geometry` | Good — that's the pre-flight working. It names the fix. |

### Resuming

Every seed's final training is checkpointed. To continue an interrupted run:

```bash
python3 run_optimization.py --config my_config.json --resume ...
```

Resume applies to the **final train**, not to the search phases. A search that dies mid-way must restart (the trial cache is the trace cache, which *is* preserved — so you don't recompute the traces).

---

## 7. Reading the results

### 7.1 `results.json`

The deliverable. The block you want:

```json
"test": {
  "ari":        {"mean": 0.72, "std": 0.08, "values": [0.79, 0.68, 0.69]},
  "ami":        {"mean": 0.65, "std": 0.06, "values": [...]},
  "silhouette": {"mean": 0.41, "std": 0.05, "values": [...]},
  "eff_rank":   {"mean": 4.2,  "std": 0.3,  "values": [...]},
  "per_seed":   [{"seed": 0, "epochs_run": 47, "best_val_ari": 0.75,
                  "test_ari": 0.79, ...}],
  "n_seeds": 3
}
```

**Report `mean ± std`.** The std is over **training seeds** — it answers *"would I get this again if I reran it?"*

**The std is your noise floor.** A difference smaller than it is not a difference. If `ari.std = 0.08`, then 0.72 and 0.68 are the same result.

### 7.2 The trap that will actually bite you

**Check whether your trials converged.**

Look at `per_seed[*].epochs_run`. If it equals `max_epochs` for most seeds, your runs hit the **ceiling**, not early stopping — meaning they were still improving when you cut them off.

This is not a cosmetic problem. If trials stop short of convergence, **the search ranks configurations by how fast they train, not by how well they can train.** Those are different questions, and you asked the wrong one.

We observed models still improving at **960 gradient steps**. If `epochs_run == max_epochs` everywhere, raise `max_epochs` (and re-check your budget) or accept that the search is measuring training speed.

### 7.3 The figures

| File | What it shows |
|---|---|
| `figures/embedding_test_seed_<n>.png` | The held-out test embedding. **Colour = the $k$-means cluster that produced the reported ARI. Marker shape = the true phenotype.** A region uniform in colour but mixed in shape is a visible mis-clustering. |
| `figures/pdp_phase1_arch.png` | Partial dependence of the objective on each architecture hyper-parameter — where the search thinks the good region is. |
| `figures/pdp_phase2_train.png` | Same, for the optimiser hyper-parameters. |
| `figures/pdp_regularization.png` | Same, for dropout + weight decay. |

The embedding plot's PCA is **display only**. No metric is computed in that 2-D space.

### 7.4 The health diagnostics

In `results.json` → `test.eff_rank`, and per-epoch in each checkpoint's `history`.

- **`eff_rank`** is the collapse tripwire. It ranges $[1, E]$. Near $E$ = the embedding uses its whole space. Near 1 = collapsed to a line. **Watch this one.**
- **`mean_pairwise_cos`** will sit near **1.0**, and **that is normal.** Your input is non-negative, every layer is post-ReLU, so all embeddings live in the positive orthant where cosines are structurally near 1. It is present at initialisation, it does not prevent learning (models reach ARI 1.0 with it pinned at 1.0000), and it cannot be normalised away. **Do not chase it.** See `01_THEORY.md` §8.1 for the full derivation.

---

## 8. Common tasks

### Re-fit a known-good config without searching

```bash
python3 run_optimization.py --config config_best.json --skip-search --n-seeds 5
```

`config_best.json` is written by every run. This trains 5 final models and re-evaluates — useful for getting a tighter seed-std on a configuration you've already chosen.

### Just look at the budget

```bash
python3 run_optimization.py --config my_config.json --dry-run
```

### Change one thing without editing the JSON

Every CLI flag overrides the file:

```bash
python3 run_optimization.py --config my_config.json --n-seeds 5 --max-epochs 100
```

### Use the pipeline as a library

The driver is *only* orchestration. Each piece is independently usable:

```python
from config import ExperimentConfig
from run_optimization import build_traces
from data_splits import make_time_segment_splits
from train import train
from evaluate import evaluate_and_plot

cfg = ExperimentConfig.from_json("my_config.json")
traces, conditions, fs = build_traces(cfg, "cache")
splits = make_time_segment_splits(traces, conditions, fs, cfg.data, base_seed=0)

model, history = train(cfg, splits.train, splits.val, "cpu", seed=0)

results = evaluate_and_plot(model, splits.test, "cpu", "fig.png",
                            seed=cfg.eval.kmeans_seed,
                            n_clusters=len(set(conditions)),
                            eval_cfg=cfg.eval)
print(results["ari"], results["silhouette"])
```

### Rebuild a model from a checkpoint

Checkpoints are **self-describing** — the architecture comes from the embedded config, so you need no hyper-parameters in code:

```python
from checkpoint import rebuild_model_from_checkpoint
model, ckpt = rebuild_model_from_checkpoint("out/run1/checkpoints/final_seed_0.pt")
model.eval()
# ckpt["config"], ckpt["extra"]["history"], ckpt["extra"]["test"] all available
```

---

## 9. Troubleshooting

| Error / symptom | Cause | Fix |
|---|---|---|
| `ValueError: infeasible window / split geometry` | `window_s` exceeds a segment | The message names the fix. Usually: reduce `window_s`. |
| `Killed`, no traceback | OOM-killer (uncatchable) | Narrow `depth_exponent_range`; reduce `n_positives`/`n_negatives`/`windows_per_condition`; `num_workers 0` |
| `split 'val' produced 0 windows` | Same as the first row, caught deeper | Reduce `window_s` |
| `FrozenInstanceError` | You assigned to `cfg.backbone.<field>` | `cfg.backbone = replace(cfg.backbone, field=value)` |
| `AttributeError: 'ExperimentConfig' object has no attribute 'copy'` | There is no `.copy()` | `ExperimentConfig.from_dict(cfg.to_dict())` |
| Trials all score `FAILED_OBJECTIVE = 1.0` | Every seed of every trial crashed | Run one config with `--skip-search --verbose` to see the real exception |
| Search results look random | Trials aren't converging | §7.2 — check `epochs_run` vs `max_epochs` |
| `mean_pairwise_cos ≈ 1.0` | **Normal.** Not a bug. | Ignore it; watch `eff_rank` (§7.4) |
| Two runs with the same seed differ | Should be impossible | Check `runtime.deterministic`; note that bit-exactness across *different batch sizes* is not guaranteed (float addition isn't associative) |
| `ModuleNotFoundError: skopt` | Missing dep | `pip install scikit-optimize` |

---

## 10. Checklist before a real run

- [ ] All nine smoke tests pass on this machine
- [ ] `--dry-run` gives a run count you've looked at
- [ ] You have **timed one training run** and multiplied it out
- [ ] The estimated wall-clock fits your walltime, with margin
- [ ] The memory pre-flight's largest corner fits the node's RAM
- [ ] `window_s` fits the shortest segment (the pre-flight will tell you)
- [ ] `eval_stride_s >= window_s` (disjoint eval windows)
- [ ] `patience <= max_epochs`
- [ ] `n_seeds >= 3` (you need an honest std)
- [ ] `runtime.seed` is fixed and recorded
- [ ] `out_dir` and `cache_dir` are on a filesystem with room, and are **not** scratch that gets wiped

After the run:

- [ ] `per_seed[*].epochs_run < max_epochs` — trials converged, not truncated
- [ ] `test.ari.std` is small enough that your result means something
- [ ] `test.eff_rank` is comfortably above 1 (no collapse)
- [ ] The embedding figure's colours and shapes broadly agree
