# The `config.json` Reference

**A complete field-by-field reference for the `ExperimentConfig` schema: every
key you can put in a config file, its type, its default, what validates it, and
what actually reads it at runtime.**

---

## 0. Provenance

Every field name, default value, type, and validation rule below is read
directly from `Main/config.py` (and the two nested dataclasses it imports
unchanged: `AugmentationConfig` from `augmentation.py`, `BackboneConfig` from
`backbone.py`). The partial-merge behaviour, the `beta = 1 - u` sampling, and
the list of search-overwritten fields were each **verified by executing the
code**, not recalled. Nothing here comes from literature; this is a schema
reference, so the project's PubMed/bioRxiv sourcing rules do not apply.

This document is the companion to the two operational documents (the local
runner and the running guide). It answers one question exhaustively: *what can
go in the JSON, and what does each key do?*

---

## 1. How the config file works (read this first)

### 1.1 It is a partial overlay on dataclass defaults

`config.py` is the single source of truth. It defines seven dataclasses --
`DataConfig`, `BackboneConfig`, `TrainConfig`, `SearchConfig`,
`RegularizationConfig`, `EvalConfig`, `RuntimeConfig` -- aggregated into one
top-level `ExperimentConfig`. Every field has a default **in the code**.

Your JSON file does not need to be complete. `ExperimentConfig.from_json` merges
it onto the defaults: **any key you omit keeps its dataclass default; any key
you provide overrides it.** Verified:

```python
c = ExperimentConfig.from_dict({"train": {"n_seeds": 5}})
# c.train.n_seeds            -> 5      (your value)
# c.train.margin             -> 0.3    (default kept)
# c.data.window_s            -> 200.0  (default kept)
# c.backbone.embedding_size  -> 16     (default kept)
```

This is why `config_toy.json` and `config_example.json` are short: they set only
what differs from the defaults. Write your own the same way -- specify the
handful of things you care about, let the rest default.

### 1.2 Unknown keys warn, they do not crash

If you misspell a key, `config_from_dict` keeps the value out of the object and
emits a `RuntimeWarning` naming the ignored key. So a typo like `"window_sec"`
instead of `"window_s"` will **silently use the default** and warn -- it will
not error. Watch the warnings on startup; an "ignoring unknown keys" message
means a key of yours did nothing.

### 1.3 Two kinds of validation

* **Hard validation (`__post_init__`, raises `ValueError`).** Runs the moment
  the config is built. These are the invariants a config *cannot* violate --
  e.g. `split_fractions` must sum to 1, `margin` must be > 0, `device` must be
  one of `cpu|cuda|auto`. A violation stops you immediately. Every such rule is
  listed per field below.
* **Soft validation (`validate()`, warns).** Cross-field sanity checks that are
  *suspicious but legal*: `max_epochs <= patience` (early stopping can never
  fire), `eval_stride_s < window_s` (eval windows overlap), and
  `backbone.embedding_size` outside `search.embedding_size_range` (fine if the
  search sets the final value). These warn and continue.

### 1.4 The critical interaction: the search overwrites some of your fields

This is the single most important thing to understand about the config. Ten
fields you can set in the JSON are **starting points that the hyper-parameter
search overwrites** during a normal run. If you run with the search enabled
(the default), setting these has limited or no effect on the final model,
because a search phase replaces them. They matter only when you pass
`--skip-search`.

The search-overwritten fields (verified against `search.py`):

| Sub-config | Field | Overwritten by |
|---|---|---|
| `backbone` | `depth_exponent` | phase 1 (architecture) |
| `backbone` | `width_multiplier` | phase 1 |
| `backbone` | `block_family` | phase 1 |
| `backbone` | `embedding_size` | phase 1 |
| `backbone` | `dropout` | pinned to 0 through phases 1-2, then set by the regularization stage |
| `train` | `margin` | phase 2 (training HPs) |
| `train` | `lr` | phase 2 |
| `train` | `beta1` | phase 2 |
| `train` | `beta2` | phase 2 |
| `train` | `weight_decay` | phase 2, then re-set by the regularization stage |

Everything **not** in this table (window sizes, split fractions, augmentation
counts, `n_seeds`, `max_epochs`, `patience`, batching, eval settings, runtime
settings, and all the search *ranges* themselves) is honoured as written
throughout the run. The tables below mark each overwritten field explicitly with
**[search sets this]**.

### 1.5 The `fs` placeholder

`data.augmentation.fs` is stored in the config but is a **placeholder**. The
real sampling rate is resolved at dataset-build time -- from `synthetic_fs` in
synthetic mode, or from the trace loader in numpy/real mode -- and injected via
`DataConfig.resolved_augmentation(fs)`. Set it to your true rate for tidiness,
but know that the value in the file does not drive anything; the data source
does.

---

## 2. `data` -- `DataConfig`

Data loading, windowing, time-segment splitting, and the (nested) augmentation
parameters.

### 2.1 Source selection

| Key | Type | Default | Meaning / validation |
|---|---|---|---|
| `data_mode` | str | `"synthetic"` | One of `"synthetic"`, `"numpy"`, `"real"`. **Hard:** must be one of those three. |
| `specs_json` | str | `""` | `real` mode only: path to the specs list (`{folder, base, condition}`). Ignored in other modes. |
| `npz_specs` | str | `""` | `numpy` mode only: path to a `burst_specs.json` (`{name, condition, path}`). Ignored otherwise. |

Note the CLI flags `--npz-specs` / `--specs-json` / `--engine-module` on
`run_optimization.py` are the usual way to point at real/numpy data; the config
keys are the file-based equivalents.

### 2.2 Synthetic generation (used only when `data_mode="synthetic"`)

| Key | Type | Default | Meaning / validation |
|---|---|---|---|
| `synthetic_n_per_class` | tuple[int, ...] | `(2, 1)` | One entry per phenotype class; the value is how many synthetic traces that class gets. **The length is the number of classes C** (labels `0..C-1`). **Hard:** at least one class; every entry >= 1. |
| `synthetic_duration_s` | float | `600.0` | Seconds per synthetic trace. **Hard:** > 0. |
| `synthetic_fs` | float | `50.0` | Sampling rate [Hz] for synthetic data; this is the real `fs` the augmentation resolves to in synthetic mode. **Hard:** > 0. |

`synthetic_n_per_class` doubles as your class-count control in synthetic mode:
`[3, 3]` is a balanced 2-class problem, `[4, 4, 4]` a balanced 3-class one.

### 2.3 Windowing

| Key | Type | Default | Meaning / validation |
|---|---|---|---|
| `window_s` | float | `200.0` | Window length in seconds. `W = round(window_s * fs)` samples. **Hard:** > 0. **Note:** the default 200 s is geometrically infeasible against the default 120 s eval segments -- always set this. |
| `train_stride_s` | float | `100.0` | Stride between **training** windows. Set `< window_s` for overlapping training windows (more diversity). **Hard:** > 0. |
| `eval_stride_s` | float | `200.0` | Stride between **val/test** windows. Set `>= window_s` for **disjoint** eval windows. **Hard:** > 0. **Soft:** if `< window_s`, warns that eval windows overlap and can inflate apparent sample size. |

The stride asymmetry is deliberate: overlap on train (diversity is good), no
overlap on eval (duplicated eval windows would fake a larger, correlated test
set).

### 2.4 Time-segment split

| Key | Type | Default | Meaning / validation |
|---|---|---|---|
| `split_fractions` | tuple[float, float, float] | `(0.6, 0.2, 0.2)` | Fractions along each trace's **time axis** for `(train, val, test)`. **Hard:** exactly 3 values; each strictly in `(0, 1)`; must sum to 1.0 (within 1e-6). |
| `drop_boundary_windows` | bool | `True` | Drop windows that straddle a split boundary, so no window spans two splits. Keep `True` -- this is part of the leakage guarantee. |

The split is along time *within* each trace, not across traces. This is what
`smoke_test_data_splits.py` checks is leakage-free.

### 2.5 `data.augmentation` -- nested `AugmentationConfig`

Per-anchor surrogate generation. The `sigma_*` bands are the positive/negative
warp strengths and are the main thing to tune against your burst time scale.

| Key | Type | Default | Meaning / validation |
|---|---|---|---|
| `fs` | float | *(required, placeholder)* | Sampling rate [Hz]. **Placeholder** -- overwritten at build time (section 1.5). |
| `intra_knot_dist` | float | `0.2` | Seconds between spline knots for the warps. |
| `sigma_mag_pos` | tuple[float, float] | `(0.01, 0.10)` | **Positive** band, magnitude warp: log-amplitude std (dimensionless). `[TUNE]` |
| `sigma_mag_neg` | tuple[float, float] | `(0.20, 0.50)` | **Negative** band, magnitude warp. `[TUNE]` |
| `sigma_time_pos_s` | tuple[float, float] | `(0.005, 0.050)` | **Positive** band, time warp: temporal std in **seconds**. `[TUNE]` |
| `sigma_time_neg_s` | tuple[float, float] | `(0.100, 0.400)` | **Negative** band, time warp. `[TUNE]` |
| `shift_magnitude_s` | float | `30.0` | Max absolute circular shift in seconds (applied to both classes as a label-preserving augmentation). |
| `n_positives` | int | `30` | Positives per anchor. Exact count for `"warp_bands"`; part of the pool for `"percentile_mse"`. |
| `n_negatives` | int | `30` | Negatives per anchor. |
| `split_method` | str | `"warp_bands"` | `"warp_bands"` (label by which strength band was sampled) or `"percentile_mse"` (label by a per-anchor MSE quantile). |
| `percentile_q` | float | `0.30` | Fraction labelled positive; used **only** when `split_method="percentile_mse"`. |
| `k_min` | int | `4` | Minimum spline knots. |
| `max_retries` | int | `5` | Empty-class re-draws before giving up. |
| `enforce_nonneg` | bool | `True` | Clamp surrogates to >= 0 (physical firing rate cannot be negative). Keep `True` for firing-rate data. |

**Sizing warning.** `n_positives` and `n_negatives` drive batch size directly:
one batch has `M = C * B_c * (1 + P + N)` embedding rows, where `P = n_positives`
and `N = n_negatives`. The default `30/30` gives a large batch and can OOM a
small machine -- `config_toy.json` uses `3/3` and `config_example.json` uses
`8/8` for exactly this reason. If a run OOMs, lower these first.

---

## 3. `backbone` -- `BackboneConfig`

The Topic-2 1D-CNN encoder. **Frozen dataclass** (immutable after
construction). Four of its fields are set by the architecture search; the rest
are fixed knobs you configure directly.

### 3.1 Body / architecture (searched)

| Key | Type | Default | Meaning / validation |
|---|---|---|---|
| `depth_exponent` | int | `4` | `d`; total blocks = `2 ** d`. **[search sets this]** (phase 1). **Hard:** >= 1. |
| `width_multiplier` | float | `2.0` | `wm`, continuous. **[search sets this]**. **Hard:** > 1.0. |
| `block_family` | int | `0` | `0` = ResNet, `1` = ResNeXt. **[search sets this]**. **Hard:** in `{0, 1}`. |
| `embedding_size` | int | `16` | `E`, output embedding dimension. **[search sets this]**. **Hard:** >= 1. **Soft:** warns if outside `search.embedding_size_range`. |

### 3.2 Body / architecture (fixed knobs)

| Key | Type | Default | Meaning / validation |
|---|---|---|---|
| `stem_width` | int | `16` | `w0 = wa = stem_width` (simplified RegNet variant). **Hard:** >= 1. |
| `group_width` | int | `16` | ResNeXt channels per group (only relevant when `block_family=1`). **Hard:** >= 1. |
| `stem_kernel` | int | `5` | Stem conv kernel. **Hard:** >= 1. |
| `stem_stride` | int | `4` | Stem stride. **Hard:** >= 1. |
| `stage_kernel` | int | `3` | Per-stage conv kernel. **Hard:** >= 1. |
| `downsampling_rate` | int | `2` | Per-stage downsampling. **Hard:** >= 1. |

### 3.3 Head / embedding

| Key | Type | Default | Meaning / validation |
|---|---|---|---|
| `l2_normalize` | bool | `True` | L2-normalize the embedding (matches the cosine loss geometry). |
| `head_fusion` | bool | `False` | `False`: use last stage only. `True`: fuse all stages. |
| `head_pool_ops` | tuple[str, ...] | `("mean",)` | Pooling ops in the head. **Hard:** non-empty; each must be one of `"mean"`, `"max"`, `"std"`. |
| `head_prenorm` | bool | `True` | Per-stage LayerNorm before concat (only meaningful when `head_fusion=True`). |

### 3.4 Normalization (GroupNorm)

| Key | Type | Default | Meaning / validation |
|---|---|---|---|
| `norm_target_cpg` | int | `16` | Target channels per group; intended set `{4, 8, 16, 24, 32}`. **Hard:** >= 1. |
| `norm_g_max` | int | `32` | Cap on number of groups. **Hard:** >= 1. |

### 3.5 Regularization

| Key | Type | Default | Meaning / validation |
|---|---|---|---|
| `dropout` | float | `0.0` | Dropout rate. **[search sets this]** -- pinned to 0 through phases 1-2, then chosen by the regularization stage. **Hard:** in `[0, 1)`. |

Setting `dropout > 0` in the config has **no effect during a normal search
run**: the driver pins it to 0 for phases 1-2 and the regularization stage
assigns the final value. It only takes effect with `--skip-search`.

---

## 4. `train` -- `TrainConfig`

Trainer settings, used **identically** by the HPO objective and the final run.

### 4.1 Loss / miner

| Key | Type | Default | Meaning / validation |
|---|---|---|---|
| `margin` | float | `0.3` | Triplet loss margin `m`. **[search sets this]** (phase 2). **Hard:** > 0. |
| `swap` | bool | `True` | `TripletMarginLoss` swap. Honoured as written. |
| `mining_strategy` | str | `"hard"` | `"hard"` or `"easy_positive"`. **Hard:** must be one of those. |

### 4.2 Batching (`ConditionBalancedBatchSampler`)

| Key | Type | Default | Meaning / validation |
|---|---|---|---|
| `windows_per_condition` | int | `8` | `B_c`: source windows drawn from **each** phenotype class per batch, so a batch holds `C * B_c` source windows and every batch supports cross-condition triplets. **Hard:** >= 1. |
| `batches_per_epoch` | int | `0` | `0` -> **derive** at build time as `ceil(N_train_windows / (C * B_c))` (one nominal pass per epoch). Any value >= 1 fixes the batch count. **Hard:** >= 0. |

### 4.3 Optimizer (AdamW)

The default values here are also the **fixed** optimizer used during phase-1
architecture search (so architectures are compared under one optimizer); phase 2
then searches them.

| Key | Type | Default | Meaning / validation |
|---|---|---|---|
| `lr` | float | `3e-4` | Learning rate. **[search sets this]** (phase 2). **Hard:** > 0. |
| `beta1` | float | `0.9` | AdamW beta1. **[search sets this]** -- phase 2 samples `1 - beta1` log-uniformly and stores `beta1 = 1 - u`. **Hard:** in `(0, 1)`. |
| `beta2` | float | `0.999` | AdamW beta2. **[search sets this]** (same `1 - u` scheme). **Hard:** in `(0, 1)`. |
| `weight_decay` | float | `1e-4` | AdamW weight decay. **[search sets this]** -- phase 2, then re-set by the regularization stage. **Hard:** >= 0. |

### 4.4 Budget / early stopping

| Key | Type | Default | Meaning / validation |
|---|---|---|---|
| `max_epochs` | int | `100` | `E_max`, hard ceiling on epochs. **Hard:** >= 1. |
| `patience` | int | `10` | `P`: consecutive no-improvement epochs before stopping. **Hard:** >= 1. **Soft:** warns if `max_epochs <= patience` (early stopping can never fire; training is fixed-length). |
| `min_delta_ari` | float | `0.0` | Min ARI improvement to reset patience. |
| `min_delta_sil` | float | `0.0` | Min silhouette improvement on an ARI plateau. |
| `selection_primary` | str | `"ari"` | Primary early-stopping metric; the other breaks ties. **Hard:** `"ari"` or `"silhouette"`. |
| `n_seeds` | int | `3` | Trainings per config; the HPO objective returns the mean validation metric over these. Also the number of final models trained. **Hard:** >= 1. |

`n_seeds` is a major cost multiplier: **every search trial runs `n_seeds`
trainings**, so the total `train()` calls scale with it. Lower it (e.g. to 2)
for quick looks; raise it for a more stable final estimate.

### 4.5 Optional accelerators (off by default)

| Key | Type | Default | Meaning / validation |
|---|---|---|---|
| `use_scheduler` | bool | `False` | Enable an LR scheduler. |
| `scheduler_type` | str | `"cosine"` | `"cosine"`, `"step"`, or `"none"`; used only if `use_scheduler=True`. **Hard:** must be one of those. |
| `use_amp` | bool | `False` | Automatic mixed precision. **GPU-only**, guarded at runtime (a no-op on CPU). |

### 4.6 Logging / checkpoint cadence

| Key | Type | Default | Meaning / validation |
|---|---|---|---|
| `log_every_epochs` | int | `1` | Epochs between log lines. **Hard:** >= 1. |
| `checkpoint_every_epochs` | int | `5` | Epochs between checkpoint writes. **Hard:** >= 1. |

---

## 5. `search` -- `SearchConfig`

The two-phase `gp_minimize` (skopt, sequential) ranges and meta-settings. These
are the **ranges the search explores** -- they are honoured as written (the
search does not overwrite its own ranges). Note `width_shrink` from the legacy
driver is intentionally absent (the Topic-2 fusion head has no such knob).

### 5.1 Phase 1: architecture ranges

| Key | Type | Default | Meaning / validation |
|---|---|---|---|
| `depth_exponent_range` | tuple[int, int] | `(3, 6)` | Integer `d` search range. **Hard:** `(low, high)` with `low <= high`; `low >= 1`. |
| `width_multiplier_range` | tuple[float, float] | `(1.5, 3.0)` | Real `wm` range (backbone-documented interval). **Hard:** `low <= high`; `low > 1.0`. |
| `block_family_choices` | tuple[int, ...] | `(0, 1)` | Categorical over block families. **Hard:** non-empty subset of `{0, 1}`. |
| `embedding_size_range` | tuple[int, int] | `(8, 16)` | Integer `E` range. **Hard:** `low <= high`; `low >= 1`. |

### 5.2 Phase 2: training-HP ranges

| Key | Type | Default | Meaning / validation |
|---|---|---|---|
| `margin_range` | tuple[float, float] | `(0.1, 1.0)` | Real margin `m`. **Hard:** `low <= high`; `low > 0`. |
| `lr_range` | tuple[float, float] | `(1e-4, 0.2)` | Learning rate, log-uniform. **Hard:** `low <= high`; `low > 0`. |
| `one_minus_beta1_range` | tuple[float, float] | `(1e-2, 1e-1)` | Log-uniform `u = 1 - beta1`, so `beta1 in [0.9, 0.99]`. **Hard:** `low > 0`; `high < 1.0`. |
| `one_minus_beta2_range` | tuple[float, float] | `(1e-4, 1e-2)` | Log-uniform `u = 1 - beta2`, so `beta2 in [0.99, 0.9999]`. **Hard:** `low > 0`; `high < 1.0`. |
| `weight_decay_range` | tuple[float, float] | `(1e-4, 1e-2)` | Weight decay, log-uniform. **Hard:** `low <= high`; `low > 0`. |

### 5.3 Meta-settings

| Key | Type | Default | Meaning / validation |
|---|---|---|---|
| `n_calls_arch` | int | `100` | Phase-1 trial count. Each trial costs `n_seeds` train runs. **Hard:** >= 1. |
| `n_calls_train` | int | `100` | Phase-2 trial count. Each trial costs `n_seeds` train runs. **Hard:** >= 1. |
| `gp_random_state` | int | `0` | RNG state for the GP search. |
| `do_refine` | bool | `False` | Coarse-to-fine second pass that narrows the box. |
| `refine_top_fraction` | float | `0.10` | Fraction of best points used to narrow the box on refine. **Hard:** in `(0, 1]`. |
| `do_retune_arch` | bool | `False` | Optional architecture re-tune after phase 2. |

`n_calls_arch` and `n_calls_train` are the two biggest levers on total runtime.
Total `train()` calls from the search phases is roughly
`(n_calls_arch + n_calls_train + n_calls_reg) * n_seeds`, plus the final
`n_seeds`. The CLI flags `--n-calls-arch/-train/-reg` override these per run.

---

## 6. `regularization` -- `RegularizationConfig`

The final stage that searches dropout and weight decay on the validation split.

| Key | Type | Default | Meaning / validation |
|---|---|---|---|
| `dropout_range` | tuple[float, float] | `(0.0, 0.3)` | Real dropout range. **Hard:** `0 <= low <= high < 1`. |
| `weight_decay_range` | tuple[float, float] | `(1e-5, 1e-2)` | Weight decay, log-uniform. **Hard:** `0 < low <= high`. |
| `n_calls` | int | `50` | Regularization trial count; each costs `n_seeds` train runs. **Hard:** >= 1. Overridden by `--n-calls-reg`. |
| `gp_random_state` | int | `0` | RNG state for this stage's GP search. |

This stage is what finally sets `backbone.dropout` and re-sets
`train.weight_decay` (section 1.4). Skipped entirely by `--skip-regularization`.

---

## 7. `eval` -- `EvalConfig`

Clustering-metric and embedding-plot settings for held-out scoring. Honoured as
written.

| Key | Type | Default | Meaning / validation |
|---|---|---|---|
| `kmeans_seed` | int | `0` | KMeans RNG seed (for reproducible clustering). |
| `kmeans_n_init` | int | `10` | KMeans restarts. **Hard:** >= 1. |
| `silhouette_metric` | str | `"cosine"` | Distance metric for silhouette; `"cosine"` matches the cosine loss geometry. **Hard:** non-empty string. |
| `pca_components` | int | `2` | PCA dimensions for the **saved embedding scatter plot only** (display, not scoring). **Hard:** >= 1. |

---

## 8. `runtime` -- `RuntimeConfig`

HPC / reproducibility / device / IO. Honoured as written (though several have
CLI equivalents that win when passed).

| Key | Type | Default | Meaning / validation |
|---|---|---|---|
| `seed` | int | `0` | Base seed; per-trial seeds derive from this. **Hard:** >= 0. |
| `device` | str | `"cpu"` | `"cpu"`, `"cuda"`, or `"auto"` (auto -> cuda if available, else cpu). **Hard:** must be one of those. Overridden by `--device`. |
| `deterministic` | bool | `True` | Enable deterministic algorithms for reproducibility. |
| `torch_threads` | int | `1` | Intra-op threads. Keep low to avoid oversubscription when `num_workers > 0`. **Hard:** >= 1. |
| `num_workers` | int | `2` | DataLoader workers. Set `0` on a memory-limited box or to sidestep worker issues. **Hard:** >= 0. Overridden by `--num-workers`. |
| `pin_memory` | bool | `False` | Pin host memory (GPU transfer speedup). |
| `out_dir` | str | `"./optim_out"` | Where artifacts go. Overridden by `--out-dir`. |
| `cache_dir` | str | `"./preproc_cache"` | Trace cache directory. Overridden by `--cache-dir`. |
| `experiment_name` | str | `"run"` | Names the artifact subfolder (`out_dir/experiment_name/`). Overridden by `--experiment-name`. |

---

## 9. Serialization guarantees

Two properties of the config that matter in practice, both enforced in
`config.py`:

* **Round-trip exactness.** `ExperimentConfig == ExperimentConfig.from_json(path)`
  holds. JSON turns tuples into lists, so loading coerces every list-valued
  field back to a tuple using its declared type (a tuple never equals a list).
  This is asserted directly by the config smoke test.
* **ASCII on disk.** The config writer emits pure ASCII (`ensure_ascii=True`),
  keeping the artifact HPC-transfer-safe. The reader accepts UTF-8, so a
  hand-written config with accents still loads.

The driver writes two configs into every run's output folder:
`config_input.json` (as resolved from file + CLI, **before** the search) and
`config_best.json` (**after** every search phase). Comparing them shows exactly
which fields the search changed -- a concrete, per-run view of the section 1.4
table.

---

## 10. A minimal but complete example, annotated

This is `config_example.json` with the reasoning behind each non-default choice.
It sets only what differs from the code defaults; everything else defaults.

```jsonc
{
  "data": {
    "data_mode": "synthetic",
    "window_s": 30.0,          // feasible (the 200 default is not)
    "train_stride_s": 15.0,    // < window_s: overlapping train windows
    "eval_stride_s": 30.0,     // == window_s: disjoint eval windows
    "split_fractions": [0.6, 0.2, 0.2],
    "synthetic_duration_s": 600.0,
    "synthetic_fs": 50.0,
    "synthetic_n_per_class": [3, 3],   // balanced 2-class
    "augmentation": {
      "fs": 50.0,              // placeholder; real fs = synthetic_fs
      "n_positives": 8,        // small batch (30/30 default can OOM)
      "n_negatives": 8,
      "shift_magnitude_s": 5.0
    }
  },
  "backbone": { "stem_width": 16 },   // arch HPs left to the search
  "train": {
    "windows_per_condition": 4,
    "batches_per_epoch": 0,    // derive one pass/epoch from the data size
    "n_seeds": 3,
    "max_epochs": 60,
    "patience": 10             // < max_epochs, so early stopping can fire
  },
  "search": {
    "depth_exponent_range": [3, 5],
    "width_multiplier_range": [1.5, 2.5],
    "embedding_size_range": [8, 16],
    "n_calls_arch": 30,        // 30 * n_seeds train runs for phase 1
    "n_calls_train": 30
  },
  "regularization": { "n_calls": 20 },
  "runtime": {
    "device": "cpu",
    "num_workers": 0,
    "out_dir": "out",
    "cache_dir": "cache",
    "experiment_name": "run1"
  }
}
```

To adapt it for real data: set `data.data_mode` to `"numpy"` or `"real"`, point
at your traces (via `--npz-specs` / `--specs-json` + `--engine-module`, or the
config keys), set `data.augmentation.fs` for tidiness (the loader resolves the
real value anyway), and tune the four augmentation `sigma_*` bands against your
measured burst duration. Everything in the search and regularization blocks can
stay as ranges; the search fills in the final architecture and training HPs.
