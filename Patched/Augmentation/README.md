Augmentation pipeline


# Topic 1 — Data Augmentation & Contrastive Pipeline: Module Reference

| File | Role |
|---|---|
| `augmentation.py` | Pure transforms (log-space `magnitude_warp`, endpoint-pinned `time_warp`, vectorized `random_circular_shift`), the two split methods, and `build_triplet_instance`. CPU / `float32`, explicit `Generator`. |
| `data_pipeline.py` | Trace providers (synthetic + a wrapper for your `Neuronal_traces`), `MEAWindowDataset` (overlapping windows), `ConditionBalancedBatchSampler`, and `TripletCollator` implementing the option-(b) label scheme. |
| `run_data_pipeline.py` | HPC front-end: config + argparse, deterministic seeding, the DataLoader, a sanity report, debug plots, config dump, and the marked extension point for Topic 2/3. |
| `augmentation_viz.py` | Headless visual-debug plotter (separate module, Agg backend). |
| `smoke_test_augmentation.py` | 32 checks on the transforms and split logic. |
| `smoke_test_data_pipeline.py` | 18 checks on windowing, labels, condition balancing, determinism, and both split methods. |

## Validation

Both smoke tests were run and controlled twice.

| Test | Result | Second control |
|---|---|---|
| `smoke_test_augmentation.py` | 32 / 32 | 40-trial sweep over random signals / seeds: positivity 40/40, endpoint preservation 40/40, warp-band MSE ordering 40/40 |
| `smoke_test_data_pipeline.py` | 18 / 18 | Front-end integration run (`run_data_pipeline.py --data-mode synthetic`): balanced batches, 120 unique-labelled negatives per batch, config dump written |

## Quick start

```bash
pip install numpy scipy torch matplotlib

# unit tests
python smoke_test_augmentation.py        # 32/32
python smoke_test_data_pipeline.py       # 18/18

# HPC dry-run (no data files needed)
python -u run_data_pipeline.py \
    --data-mode synthetic \
    --num-workers 4 \
    --n-debug-plots 6

# real data
python -u run_data_pipeline.py \
    --data-mode real \
    --specs-json specs.json \
    --num-workers 8
```

`specs.json` schema (one record per well):

```json
[
  {"folder": "/path/ptrain_Control00_Well11",
   "base":   "ptrain_Control00_Well11_",
   "condition": 0},
  {"folder": "/path/pgroup02_Well14",
   "base":   "pgroup02_Well14_",
   "condition": 1}
]
```

## Items to confirm before a real run

| # | Item | Where to change |
|---|---|---|
| 1 | `closest_power_of_2` convention: **nearest** (current) vs **floor** — must match the encoder engine | `data_pipeline.py`, `closest_power_of_2()` |
| 2 | Real-mode import: set `from <your_engine_module> import Neuronal_traces` to the actual filename | `run_data_pipeline.py`, `load_traces()` |
| 3 | σ bands in `AugmentationConfig` are placeholders (`[TUNE]`); tune against the measured burst time scale $\tau_{\text{burst}}$ (from `calculate_mean_burst_duration`) | `augmentation.py`, `AugmentationConfig` |
| 4 | `--num-workers` should equal the CPUs-per-task granted by the SLURM allocation | CLI flag |
