# Technical Reference

## The contrastive MEA phenotype pipeline: architecture, APIs, and invariants

**Companion to** `01_THEORY.md` (why) and `03_USAGE.md` (how to run).
This document covers **what the code is and how it is put together**.

---

## 1. Module map

Thirteen modules, ~5,100 lines of source plus ~4,000 lines of tests. The dependency graph is a DAG — no module imports a module that (directly or transitively) imports it.

```
                        config.py
                            |
        +-------------------+-------------------+
        |                   |                   |
   backbone.py       augmentation.py            |
        |                   |                   |
        |            data_pipeline.py           |
        |                   |                   |
        |         preprocessing_cache.py        |
        |                   |                   |
        |             data_splits.py            |
        |                   |                   |
        +---------+---------+                   |
                  |                             |
            inference.py <----------------- metrics.py
                  |                             |
                  +--------- train.py ----------+
                  |              |              |
            evaluate.py     checkpoint.py       |
                  |              |              |
                  +----- search.py -------------+
                                 |
                      run_optimization.py
```

| Module | Lines | Responsibility | Depends on |
|---|---|---|---|
| `config.py` | 468 | Single source of truth. Nested dataclasses, JSON round-trip, validation. | — |
| `backbone.py` | 458 | The 1-D CNN design space. `build_backbone(BackboneConfig) -> nn.Module`. | — |
| `augmentation.py` | 351 | Warps, the positive/negative split, the per-anchor triplet builder. | — |
| `data_pipeline.py` | 312 | Trace providers, `MEAWindowDataset`, balanced sampler, collator. | `augmentation` |
| `preprocessing_cache.py` | 180 | Persist per-trace computation once; manifest + `.npz` per trace. | — |
| `data_splits.py` | 316 | Time-segment splitting; C-class synthetic provider. | `data_pipeline`, `config` |
| `metrics.py` | 244 | ARI/AMI/silhouette; collapse diagnostics. **No torch.** | — |
| `checkpoint.py` | 220 | Atomic, self-describing checkpoints; RNG capture/restore. | `backbone`, `config` |
| `inference.py` | 163 | The **shared** clean-window embedder. | — |
| `train.py` | 638 | The single-seed training loop. | `backbone`, `checkpoint`, `data_pipeline`, `inference`, `metrics` |
| `evaluate.py` | 359 | Scoring + the headless embedding figure. | `inference`, `metrics` |
| `search.py` | 674 | Two-phase Bayesian search + regularization stage. | `config`, `train` |
| `run_optimization.py` | 654 | Pure orchestration + CLI. | everything |

**Separation of concerns is enforced, not aspirational.** `metrics.py` has no torch import (embeddings arrive as arrays; tensors are duck-typed). `train.py` never scores or plots. `evaluate.py` never trains. `search.py` never trains directly — it calls `train.train`. `run_optimization.py` contains no science at all; it decides order and wires artifacts to disk.

---

## 2. Configuration system (`config.py`)

### 2.1 Structure

`ExperimentConfig` is a nested dataclass tree:

```python
ExperimentConfig
├── data:           DataConfig
│   └── augmentation: AugmentationConfig   # reused from augmentation.py
├── backbone:       BackboneConfig         # reused from backbone.py — FROZEN
├── train:          TrainConfig
├── search:         SearchConfig
├── regularization: RegularizationConfig
├── eval:           EvalConfig
└── runtime:        RuntimeConfig
```

`BackboneConfig` and `AugmentationConfig` are **imported from their owning modules**, not re-declared. There is exactly one definition of each.

### 2.2 Two properties that bite

**`BackboneConfig` is a frozen dataclass.** Every other sub-config is mutable. Attribute assignment raises `FrozenInstanceError`:

```python
cfg.backbone.depth_exponent = 6              # FrozenInstanceError
cfg.backbone = replace(cfg.backbone,          # correct
                       depth_exponent=6)
```

`dataclasses.replace` re-runs `__post_init__`, so validation fires on the rebuilt object. This is a free correctness win: an illegal architecture point raises at construction rather than silently building a broken model.

**`ExperimentConfig` has no `.copy()`.** A shallow copy would share the nested dataclasses, so a search trial mutating `cfg.train.lr` would corrupt the base config for every later trial. The deep-copy idiom used throughout is the tested JSON round-trip:

```python
def _deep_copy_cfg(base_cfg):
    return ExperimentConfig.from_dict(base_cfg.to_dict())
```

### 2.3 Serialization contract

`to_dict()` / `from_dict()` / `to_json()` / `from_json()`. The round-trip is **exact**, including tuple coercion: JSON has no tuple type, so lists come back as tuples (`split_fractions`, `head_pool_ops`, all `*_range` fields). Smoke check `[2]` asserts `cfg == from_json(to_json(cfg))` on a fully non-default config.

JSON on disk is **pure ASCII** (HPC-safe artifact). Partial JSON is legal — omitted sections fall back to defaults.

### 2.4 Key fields

| Config | Field | Default | Notes |
|---|---|---|---|
| `DataConfig` | `window_s` | `200.0` | **See §9.1 — the default is infeasible** |
| | `train_stride_s` | `100.0` | `< window_s` → overlapping train windows |
| | `eval_stride_s` | `200.0` | `>= window_s` → disjoint eval windows |
| | `split_fractions` | `(0.6, 0.2, 0.2)` | must sum to 1 |
| `BackboneConfig` | `depth_exponent` | `4` | params ~exponential in this |
| | `width_multiplier` | `2.0` | |
| | `embedding_size` | `16` | = $E$ |
| | `l2_normalize` | `True` | embeddings on the unit sphere |
| `TrainConfig` | `margin` | `0.3` | cosine-similarity gap, **not** an angle |
| | `mining_strategy` | `"hard"` | or `"easy_positive"` |
| | `windows_per_condition` | `8` | = $B_c$ (added in Stage 5) |
| | `batches_per_epoch` | `0` | `0` → derive; see §5.2 |
| | `n_seeds` | `3` | = $N_s$ |
| | `selection_primary` | `"ari"` | or `"silhouette"` |
| `AugmentationConfig` | `n_positives` | `30` | = $P$ — **see §9.3** |
| | `n_negatives` | `30` | = $N$ |
| `EvalConfig` | `kmeans_seed` | `0` | |
| | `silhouette_metric` | `"cosine"` | |
| `RuntimeConfig` | `device` | `"cpu"` | `"cuda"` / `"auto"` |
| | `num_workers` | `2` | |

---

## 3. Data path

### 3.1 Providers

A **provider** is any callable `provider(*args) -> (trace: np.ndarray[(K,)], fs: float)`.

| Provider | Call signature | Source |
|---|---|---|
| `SyntheticTraceProvider` | `(condition, trace_id)` | 2-class synthetic |
| `MultiClassSyntheticProvider` | `(condition, trace_id)` | C-class synthetic (generalises the above) |
| `NumpyTraceProvider` | `(npz_path)` | pre-computed `.npz` |
| `NeuronalTracesProvider` | `(folder, base)` | wraps the engine's `Neuronal_traces` |

Providers are **injected**, never hard-coded — this is what makes every downstream module testable without `.mat` files.

### 3.2 The cache (`preprocessing_cache.py`)

The per-trace computation happens **once**. Each trace is stored as `<cache_dir>/<name>.npz` with keys `ifr_trace`, `fs_ifr`, `condition`, `name` — deliberately the same keys `NumpyTraceProvider` expects, so a cached file is itself loadable by that provider.

A `manifest.json` indexes them, written **atomically** (temp file + `os.replace`), so an interrupted run cannot leave a half-written index.

```python
specs = [TraceSpec(name, condition, args), ...]
manifest = cache_traces(specs, provider, cache_dir, overwrite=False)
traces, conditions, fs = load_cached_traces(cache_dir)   # verifies a single shared fs
```

`overwrite=False` (default) means re-running **skips** already-cached traces. This is why an HPO run of 753 trials pays the trace cost exactly once.

### 3.3 Time-segment splitting (`data_splits.py`)

```python
bundle = make_time_segment_splits(traces, conditions, fs, data_cfg, base_seed=0)
```

Returns a `SplitBundle`:

| Field | Type | Meaning |
|---|---|---|
| `train`, `val`, `test` | `MEAWindowDataset` | the three datasets |
| `window_length` | `int` | $W$ = `round(window_s * fs)` |
| `train_stride`, `eval_stride` | `int` | in samples |
| `coverage` | `Dict[str, List[(trace_idx, start, end, condition)]]` | per-window provenance in **original trace coordinates**, in the exact order the Dataset enumerates them |
| `seg_bounds` | `List[List[(s, e)]]` | per-trace segment bounds |

`coverage` is what makes leakage **testable**: the smoke test compares every test-window interval against every train/val interval of the same trace and asserts zero overlap (1,176 comparisons in the toy case).

**The leakage guarantee is structural.** Windows are formed inside a segment (`window_starts(seg_len, W, stride)` mirrors `MEAWindowDataset`'s own tiling rule exactly), so no window can straddle a boundary. A drift between the two would be caught by smoke check `[B]`, which reconstructs the provenance from the Dataset's own `.index` and compares.

### 3.4 Batch assembly

```
ConditionBalancedBatchSampler  →  B_c indices from EACH phenotype per batch
              ↓
MEAWindowDataset.__getitem__   →  per window: anchor + P positives + N negatives
              ↓
TripletCollator                →  X: (M, W) float32,  y: (M,) int64
```

with

$$M \;=\; C \times B_c \times (1 + P + N)$$

**The label scheme (option b).** Positive rows from a source window of phenotype $c$ get label $c$. Destroyed surrogates get **unique** labels starting at `unique_label_base = 1_000_000`. A unique label can never be matched by a second row, so a destroyed surrogate is *structurally* only ever a negative — a per-anchor hard negative.

---

## 4. Backbone (`backbone.py`)

```python
model = build_backbone(BackboneConfig(...))
z = model(x)          # x: (M, W) → z: (M, E), rows L2-normalized
```

Forward accepts `(M, W)` directly (no channel dim needed). Output rows are L2-normalised in the head when `l2_normalize=True` (default).

**GroupNorm everywhere, never BatchNorm.** Consequences:

- `model.train()` and `model.eval()` compute the **same function** — no batch statistics to pollute.
- The embedding of a window is **independent of its batch companions**. Verified empirically: placing a window in a batch with companions of magnitude $10^3$ moves its embedding by $\sim 1.2 \times 10^{-7}$, i.e. float32 noise. Genuine cross-sample leakage would produce a large, companion-dependent shift.

Note this means bit-exactness across different batch sizes is **not** a property any correct implementation has — float addition is not associative, and conv kernels pick different blocking per batch size. The smoke test therefore asserts float32-tolerance invariance plus a direct extreme-companion no-leakage probe, not bit-equality.

**Parameter count explodes with depth:**

| depth | width 1.5 | width 3.0 |
|---|---|---|
| 3 | 300 K | 397 K |
| 4 | 2.45 M | 2.09 M |
| 5 | 17.7 M | 20.0 M |
| 6 | 139 M | **214 M** |

~700× across the default search range. See §9.2.

---

## 5. Training (`train.py`)

### 5.1 The contract

```python
model, history = train(cfg, train_ds, val_ds, device, seed,
                       ckpt_dir=None, verbose=False)
```

**Single seed.** The $N_s$-seed averaging is done by the *caller* (`search.evaluate_candidate`, `run_optimization.run`). Keeping the seed loop out of `train()` is what lets the search report an honest seed-to-seed std.

`history` is a list of per-epoch dicts:

```python
{"epoch", "train_loss", "ari", "ami", "silhouette", "n_triplets",
 "lr", "seconds",
 "health": {"min_std", "mean_std", "eff_rank", "mean_pairwise_cos"}}
```

### 5.2 Batching

```python
n_batches = derive_batches_per_epoch(n_windows, n_classes,
                                     windows_per_condition, batches_per_epoch)
```

`batches_per_epoch = 0` → derive $\lceil N_{\text{train}} / (C \cdot B_c) \rceil$ (one nominal pass). Any value $\ge 1$ overrides.

### 5.3 Loss and miner — the geometry trap

```python
loss_fn, miner = build_loss_and_miner(train_cfg)
```

| | object | distance |
|---|---|---|
| loss | `TripletMarginLoss(margin, swap, reducer=AvgNonZeroReducer())` | `CosineSimilarity()` |
| miner (`"hard"`) | `TripletMarginMiner(margin, type_of_triplets="hard")` | `CosineSimilarity()` |
| miner (`"easy_positive"`) | `BatchEasyHardMiner(pos_strategy="easy", neg_strategy="hard")` | `CosineSimilarity()` |

**The trap:** pytorch-metric-learning's miners **default to `LpDistance` (Euclidean)**. Passing cosine only to the loss leaves the miner selecting triplets under Euclidean geometry while the loss scores them under cosine — a silent objective/miner mismatch. Both receive `CosineSimilarity()` **explicitly**. Smoke check `[A]` asserts `is_inverted is True` on both.

**Sign convention (verified from library source, not assumed):** `CosineSimilarity.is_inverted == True`, and both loss and miner form the margin as `(d_ap - d_an)` for inverted metrics instead of `(d_an - d_ap)`. So $m$ keeps its usual meaning and **no manual sign flip is needed**.

### 5.4 The epoch loop

```
for epoch in 1..E_max:
    reseed_dataset_rng(train_ds, seed, epoch)   # see §5.6
    sampler.set_epoch(epoch)
    for X, y, _ in loader:
        Z = model(X); pairs = miner(Z, y); loss = loss_fn(Z, y, pairs)
        loss.backward(); optimizer.step()
        loss_accum += loss.detach()             # ON-DEVICE accumulation
    train_loss = loss_accum.item() / n_batches  # ← the ONE .item() per epoch
    Z_val, y_val = embed_clean_windows(model, val_ds, device)
    m = clustering_metrics(Z_val, y_val, seed=eval.kmeans_seed, n_clusters=C)
    health = embedding_health(Z_val)            # monitor-only
    ... early-stopping rule ...
```

**One `.item()` per epoch** (one GPU sync, not one per batch). The smoke test asserts this by **attributing every `.item()` call to its stack frame** — a naive process-wide count is meaningless, because PyTorch's own AdamW calls `.item()` inside `optimizer.step()` (195 times per epoch in the toy config) and the DataLoader calls it once at construction.

### 5.5 Early stopping and best-epoch restore

With $(u_e, v_e)$ = (primary, secondary) and running bests $(u^*, v^*)$:

```python
improved = (u_e > u* + delta) or ((u_e <= u* + delta) and (v_e > v* + eps))
```

Patience advances only when **both** signals flatten. The guard on clause 2 is logically redundant (`A ∨ (¬A ∧ B) ≡ A ∨ B`, verified by truth table) but is written explicitly so the code can be audited line-by-line against the spec.

Best epoch = **lexicographic argmax** of $(u_e, v_e)$. **Best-epoch weights are restored** at the end, never the last epoch's. NaN → $-\infty$, so it can neither win selection nor reset patience.

The smoke test for this is **non-vacuous**: it searches seeds until it finds a run where best ≠ last, then asserts the returned model reproduces the *best* epoch's ARI **and** that the last epoch scored differently — so returning last-epoch weights would have failed the check.

### 5.6 The RNG carry-over fix

`MEAWindowDataset` owns a **persistent** `np.random.Generator` (`self.rng`) that every `__getitem__` advances. Two consequences:

1. The augmentation a window receives depends on **how many items were drawn before it in that dataset object's lifetime**.
2. `search.evaluate_candidate` calls `train()` $N_s$ times **on the same `SplitBundle`**. Without intervention, run $k$ inherits run $k-1$'s RNG state — so identical seeds produce **different** augmentations and HPO trials stop being reproducible.

`data_pipeline.seed_worker` fixes this only for `num_workers > 0`. With `num_workers = 0` (the CPU/HPC default) nothing re-seeds it.

**Fix:** `train()` calls `reseed_dataset_rng(train_ds, seed, epoch)` at the top of every epoch, making the augmentation stream a **pure function of `(seed, epoch)`**. This also means a resumed epoch $e$ sees the same stream as an uninterrupted epoch $e$. `data_pipeline.py` itself is **not modified** — only its public `.rng` attribute is re-seeded.

Smoke check `[H]` is the regression guard: same seed twice on the *same dataset objects* → identical history and bit-identical weights; a different seed still diverges (so seed variance is preserved).

### 5.7 $K = C$ from the union

```python
C = |labels(train) ∪ labels(val)|      # NOT labels(val) alone
```

`data_splits` **warns but permits** a phenotype having no windows in a split. If $K$ were inferred from the validation split alone, a split that lost a rare class would silently fit $K = C-1$ clusters and the ARI would change meaning **between HPO trials**. Smoke check `[L]` guards it. `evaluate.py` does the same for the test split (via the caller's explicit `n_clusters`).

---

## 6. Inference (`inference.py`) — the shared helper

```python
Z, y = embed_clean_windows(model, dataset, device, batch_size=256)
X    = clean_windows(dataset)      # (N, W) raw slices
```

Imported by **both** `train.py` (per-epoch validation) and `evaluate.py` (Stage-6 scoring), so the embed step exists in exactly one place and the two cannot drift.

Guarantees:

- **Distinct rows by construction.** Enumerates `dataset.index` once → $N = $ `len(dataset.index)`. This is the fix for the legacy resample-with-replacement bug, which duplicated rows and thereby inflated the apparent $N$ that every clustering statistic assumes.
- **Clean.** `X[i]` is the raw slice `traces[t_i][s_i : s_i+W]` — bit-identical to `MEAWindowDataset.__getitem__(i)["anchor"]`, without paying for surrogate generation.
- **`.eval()` + `no_grad()`**, with the caller's train/eval mode **restored on exit** — so calling this mid-training-loop cannot silently leave dropout disabled.
- `batch_size` is a **pure throughput knob**; the returned `Z` is invariant to it (GroupNorm).

---

## 7. Metrics (`metrics.py`)

```python
m = clustering_metrics(Z, y, seed=0, n_clusters=None,
                       n_init=10, silhouette_metric="cosine")
# → {"ari", "ami", "silhouette", "labels_pred", "n_clusters"}

h = embedding_health(Z)
# → {"min_std", "mean_std", "eff_rank", "mean_pairwise_cos", "n", "dim"}
```

**Exactly one $k$-means is fit**, on the full-dimensional $Z$, and its `labels_pred` are **returned** so the evaluator can colour its plot with the same labels that produced the metric.

Silhouette is computed against the **true** labels `y`, not against `labels_pred` — making it a $k$-means-*independent* companion.

`effective_rank` = participation ratio $\left(\sum \lambda_k\right)^2 / \sum \lambda_k^2 \in [1, E]$, with `0.0` reserved for exact total collapse (zero covariance).

`mean_pairwise_cosine` uses the $O(NE)$ identity $\sum_{i,j} \hat z_i \cdot \hat z_j = \|\sum_i \hat z_i\|^2$ rather than an $O(N^2)$ matrix.

**Degrades gracefully:** fewer than 2 label classes → `nan` ARI/AMI/silhouette with a warning, not an exception. $N < 2$ → `nan` health.

**No torch import.** Tensors are duck-typed via `.detach()` / `.cpu()` / `.numpy()`.

---

## 8. Evaluation (`evaluate.py`)

```python
results = evaluate_and_plot(model, dataset, device, out_path,
                            seed, n_clusters, eval_cfg, title)
```

Returns `{ari, ami, silhouette, labels_pred, n_clusters, Z, y, health, n_windows, figure}`.

Use `evaluate_and_plot` rather than `evaluate` + `plot_embedding` separately: it makes the metric/plot consistency **structural** — the figure cannot be coloured by anything other than the labels that produced the scores.

**The four legacy bugs removed** (read from `1D_CNN_functions.Embedding_Scores`, ~lines 3156–3239, not from a description):

| # | Legacy | Fix |
|---|---|---|
| 1 | Metric fit a **seeded** $k$-means on full-D $Z$; the plot fit a **second, unseeded** $k$-means on **PCA(2)** data. The clusters *scored* and the clusters *drawn* were different fits in different spaces. | Exactly one clustering. `labels_pred` is passed *in* to `plot_embedding`, which **never clusters**. PCA is display-only. |
| 2 | `reduced_data` assigned only inside `if Visible == True:` but returned unconditionally → **`NameError` on every headless call**. | `Visible` flag removed entirely. Always `savefig`, never `plt.show()`. |
| 3 | The scatter of the actual points was **commented out** — the figure drew label-free black dots (`"k."`) over a mesh. | Every point drawn, **doubly encoded**: colour = `labels_pred`, marker = true `y`. Mis-clustering is visible. |
| 4 | Decision-boundary mesh computed from the **PCA-space** $k$-means, i.e. the wrong fit. | Mesh dropped. |

Smoke check `[B]` proves the identity **from the rendered artefact**: it intercepts the Axes, reads the facecolours back out of the scatter collections, maps each plotted point to its embedding row via the PCA coordinates, and asserts the recovered cluster assignment equals `labels_pred` exactly (for $C \in \{2,3,4\}$).

Smoke check `[F]` proves the plot fits no clustering: it patches `sklearn.cluster.KMeans` to **raise**, and drawing still succeeds.

Headlessness is asserted by **AST**, not grep: zero `.show()` call sites and zero `Visible` identifiers in *executable code*. (A grep would false-positive on the docstrings, which legitimately *mention* both while documenting their removal.)

**Colours are keyed on the cluster ID**, not on its position among the present IDs. If $k$-means leaves a cluster empty, a positional map would give cluster 2 a different colour in one figure than another — silently breaking visual comparability between the validation and test figures, and across HPO trials. Smoke check `[L]` guards it (and was verified to fail on the positional version).

---

## 9. Search (`search.py`)

### 9.1 Spaces

| Phase | Dimension | Type | Prior |
|---|---|---|---|
| **1: arch** | `depth_exponent` | `Integer` | uniform |
| | `width_multiplier` | `Real` | uniform |
| | `block_family` | `Categorical` | — |
| | `embedding_size` | `Integer` | uniform |
| **2: train** | `margin` | `Real` | uniform |
| | `lr` | `Real` | **log-uniform** |
| | `one_minus_beta1` | `Real` | **log-uniform** |
| | `one_minus_beta2` | `Real` | **log-uniform** |
| | `weight_decay` | `Real` | **log-uniform** |
| **3: reg** | `dropout` | `Real` | uniform (range starts at 0 → log prior raises) |
| | `weight_decay` | `Real` | **log-uniform** |

Betas are searched as $u = 1 - \beta$ and converted back with $\beta = 1 - u$ **in exactly one place** (`config_from_train_point`).

### 9.2 `get_newspace` — the corrected narrowing helper

```python
space = get_newspace(res, pers, search_cfg)
```

Narrows the arch space around the best `pers` fraction of trials. **Four legacy bugs, each verified empirically against skopt 0.10.2:**

| # | Legacy | Verified failure | Fix |
|---|---|---|---|
| 1 | `es = Real(lower_bounds[3], upper_bounds[3])` — copy-paste from `ws`; embedding size is column **`[4]`** | refined `es` range was silently the **width-shrink** range; `es` and `ws` became perfectly correlated | columns addressed **by name** → the class of bug is unrepresentable |
| 2 | every dimension built as `Real`, incl. `blk` and `es` (both discrete) | `Real(0,1).rvs()` → `0.549`; `Block_array[0.549]` → `TypeError` | `Categorical` / `Integer` |
| 3 | `Real(..., 'log-uniform')` on `d`, `wm` | `Real(0, 5, 'log-uniform')` → `ValueError: search space should not contain 0` | no log prior in the arch space |
| 4 | no degenerate-range guard | `Integer(3,3)` → `ValueError: lower bound 3 has to be less than upper bound 3` — **fires exactly when the search converges** | widen ±1 within original bounds; else pin as single-value `Categorical` |

Bug 4 deserves emphasis: the legacy code crashed **hardest on success**. A converged search produces best points that share a value, which is precisely the degenerate case.

### 9.3 The objective

```python
objective, record = evaluate_candidate(cfg, splits, device, trial_number, log=None)
```

For trial $t$, seeds are $s_0 + t \cdot N_s + n$ for $n = 0 \dots N_s-1$ — **disjoint blocks across trials**, fully reproducible.

$$f(x) = -\frac{1}{N_s}\sum_n A_n, \qquad A_n = \max_e \text{ARI}_e^{(n)}$$

`record` carries `{trial, scores, mean, std, objective, eff_rank, n_seeds_ok, n_seeds, failed}`.

**`std` is logged, never optimised** — it is the honest GP noise level.

**`eff_rank` is the collapse tripwire**, not `mean_pairwise_cos` (which sits near 1 by construction on non-negative inputs — see `01_THEORY.md` §8.1).

### 9.4 Failure policy

`FAILED_OBJECTIVE = 1.0` — finite, and strictly worse than any achievable score ($\text{ARI} \le 1 \Rightarrow f \ge -1$). **NaN is never returned**: `gp_minimize`'s surrogate cannot fit NaN and would abort the study.

**A trial is valid only if EVERY seed completed.** This is strict by design. Under a "use the survivors" policy, a config that crashed on 2 of 3 seeds but got lucky on the third reports **mean 0.95, std 0.00** — indistinguishable to the GP from a robustly excellent config, and *more* attractive than an honest 0.90 ± 0.05. The surrogate would actively steer the search **into** the flaky region. Smoke check `[L]` guards it.

---

## 10. Driver (`run_optimization.py`)

Pure orchestration. Pipeline:

```
resolve config → resolve device → seed → PRE-FLIGHTS → cache traces → splits
   → phase 1 → phase 2 → [re-tune] → regularization
   → final train (N_s models) → held-out TEST eval → artifacts
```

### 10.1 Three pre-flight checks

These exist because **all three failure modes actually occurred** during development, and two of them are uncatchable in Python (SIGKILL).

**`check_window_feasibility(cfg, trace_length, fs)`** — raises with the concrete fix named.

> **The shipped defaults are infeasible.** `window_s = 200 s`, but a 600 s recording split 60/20/20 leaves val and test segments of **120 s** each. Windows are formed *inside* a segment, so both splits get **zero windows**. A default run dies immediately.

**`estimate_model_sizes(cfg)`** — reports parameter counts at the arch-space corners and warns above ~1 GB.

> Depth 3 → 300 K params. Depth 6 → **214 M** (~2.6 GB for weights + AdamW state alone). The search **will** sample that corner during random initialisation. A soft OOM raises and is correctly scored `FAILED`; a Linux OOM-**killer** SIGKILL is **uncatchable** and takes the whole study with it, silently.

**`estimate_batch_rows(cfg, C)`** — makes the augmentation multiplier explicit.

> $M = C \times B_c \times (1 + P + N)$. With the defaults ($B_c=8$, $P=N=30$), a 2-class batch is **976 rows, not 16** — a 61× multiplier that reading `windows_per_condition = 8` gives no hint of.

**`estimate_budget(cfg, skip_search)`** — total `train()` runs. **The defaults cost 753.**

### 10.2 Artifacts

```
<out_dir>/<experiment_name>/
├── config_input.json          # as resolved (file + CLI)
├── config_best.json           # after all search phases
├── results.json               # ← the deliverable
├── figures/
│   ├── pdp_phase1_arch.png
│   ├── pdp_phase2_train.png
│   ├── pdp_regularization.png
│   └── embedding_test_seed_<n>.png
└── checkpoints/
    ├── seed_<n>/{last,best}.pt   # resumable
    └── final_seed_<n>.pt         # self-describing
```

### 10.3 `results.json`

```jsonc
{
  "experiment", "device", "seconds",
  "budget":       {"phase1_arch", "phase2_train", "regularization",
                   "final", "TOTAL_train_runs"},
  "model_sizes":  {"corners", "max_params", "max_ram_gb_..."},
  "batch_rows":   {"rows_per_batch_M", ...},
  "n_traces", "n_classes", "fs", "window_length", "n_windows",
  "phase1_arch":  {"best", "best_objective", "trial_log"},
  "phase2_train": {"best", "best_objective", "trial_log"},
  "regularization": {...},
  "config_best":  { ...full ExperimentConfig... },
  "test": {
    "ari":        {"mean", "std", "values": [per-seed]},
    "ami":        {...}, "silhouette": {...}, "eff_rank": {...},
    "per_seed":   [{"seed", "epochs_run", "best_val_ari",
                    "test_ari", "test_ami", "test_silhouette", "figure"}],
    "n_seeds": N
  }
}
```

**The `test` spread is over TRAINING SEEDS**, not $k$-means restarts. It answers *"would I get this again?"* Jittering the clustering seed of one model would report clustering instability — a smaller, less honest number.

---

## 11. Checkpoints (`checkpoint.py`)

```python
save_checkpoint(path, config, model, optimizer=None, scheduler=None,
                epoch=0, best_metric=None, rng_state=None, extra=None)
model, ckpt = rebuild_model_from_checkpoint(path)   # ← zero HPs in code
```

**Self-describing.** The model is rebuilt from the **embedded config alone**: `build_backbone(BackboneConfig(**ckpt["config"]["backbone"]))`. No hyper-parameter is remembered in code, which is what makes HPO trials and the final run interchangeable and safe to resume.

**Atomic.** Every write goes to a temp file in the destination directory and is promoted with `os.replace` (atomic on POSIX). A failing write leaves the previous checkpoint **intact** and no temp file behind. Smoke check `[C]` verifies by patching `torch.save` to raise mid-write.

**RNG capture/restore** (`torch`, `cuda`, `numpy`, `python`), so a resumed step reproduces an uninterrupted step bit-for-bit. Smoke check `[B]` verifies: 3 steps → save → 2 more steps, versus 3 steps → save → rebuild → restore → 2 steps, must give **identical parameters**.

---

## 12. Reproducibility contract

Fix `runtime.seed`, `search.gp_random_state`, `eval.kmeans_seed`, and `runtime.deterministic=True`, and:

| Level | Guarantee | Mechanism |
|---|---|---|
| Trial sequence | identical | `gp_minimize(random_state=...)` |
| Per-trial seeds | disjoint, reproducible | $s_0 + t N_s + n$ |
| Augmentation stream | pure function of `(seed, epoch)` | `reseed_dataset_rng` |
| Weight init / batch order | identical | `set_global_seed` |
| $k$-means | identical | `eval.kmeans_seed` |
| Resume | bit-identical to uninterrupted | RNG capture/restore |

**Not guaranteed:** bit-exactness across different `batch_size` values, or across CPU/GPU. Float addition is not associative. This is a property of floating-point arithmetic, not a defect.

---

## 13. HPC compliance

Every source file in the import chain is **pure ASCII** (byte-scanned; no code point > 127). Rationale: transfer through Windows tooling (MobaXterm, copy-paste, `scp` from a Windows box) can re-encode non-ASCII bytes, producing a `SyntaxError` that surfaces only at job-submission time on the cluster.

**No interactive calls.** `evaluate.py` forces `matplotlib.use("Agg")` **before** `pyplot` is imported, so importing anything downstream cannot try to open a display. Verified by AST (zero `.show()` call sites) **and** at runtime (`plt.show` patched to raise across the whole pipeline, never hit).

**No hard-coded paths.** Everything derives from `runtime.out_dir` / `runtime.cache_dir`.

**Dependencies to install on the cluster:**

```bash
pip install pytorch-metric-learning scikit-optimize --break-system-packages
```

(torch, numpy, scipy, scikit-learn, matplotlib assumed present.)

---

## 14. Test coverage

Nine suites, all green. Every module has one.

| Suite | Checks | Highlights |
|---|---|---|
| `smoke_test_config` | 9 | JSON round-trip equality; tuple coercion; ASCII on disk |
| `smoke_test_data_splits` | 9 | **leakage-free by construction**; provenance no-drift; stride semantics |
| `smoke_test_metrics` | 11 | eff_rank vs known participation ratio; closed-form cosine vs brute force |
| `smoke_test_checkpoint` | 4 | **resume ≡ uninterrupted**; atomic write under simulated crash |
| `smoke_test_inference` | 9 | distinct rows; **cross-sample independence** (extreme-companion probe) |
| `smoke_test_train` | 12 | miner/loss geometry; **one `.item()`/epoch by stack attribution**; **non-vacuous** best-epoch restore; RNG carry-over guard |
| `smoke_test_evaluate` | 12 | **metric/plot identity from the rendered artefact**; plot re-clusters → KMeans patched to raise |
| `smoke_test_search` | 12 | all 4 `get_newspace` bugs; **objective provably calls `train()`** (patched, not read); partial-seed-failure guard |
| `smoke_test_end_to_end` | 12 | full pipeline; all 3 pre-flights; artifacts; resume; **no interactive call reachable** |

**Testing philosophy.** Where a property could be asserted either by reading the source or by observing behaviour, the tests observe behaviour:

- The metric/plot identity is recovered **from the rendered figure's facecolours**, not by trusting that the right variable was passed.
- "The objective calls the same `train()`" is proven by **patching `train` and recording every invocation**, not by inspection.
- "The plot doesn't re-cluster" is proven by **patching `KMeans` to raise** and drawing successfully.
- Headlessness is proven by **AST** (code, not prose) plus a runtime exploding `plt.show`.
- The best-epoch restore test **searches seeds until best ≠ last**, so it cannot pass vacuously.

---

## 15. Known gaps

| Gap | Status |
|---|---|
| `data_mode="mat"` | Not wired into `build_traces`. `NeuronalTracesProvider` exists; add a branch. There is an explicit `NotImplementedError` pointing at it. |
| Trial budget vs convergence | Models were still improving at 960 gradient steps. A search whose trials stop short of convergence ranks configs by **how fast they train**, not how well. Calibrate `max_epochs` on one trial first. |
| Default config | Infeasible (§10.1). Use `config_example.json`. |
| Wall-clock estimate | The driver reports the **count** of training runs; only a timed trial gives the seconds. |
