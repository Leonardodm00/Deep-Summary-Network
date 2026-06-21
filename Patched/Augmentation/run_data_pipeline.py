"""
run_data_pipeline.py  --  front-end / entry point for the contrastive data pipeline
===================================================================================

Loads data (synthetic for dry-runs, or the project's real MEA traces), builds the
condition-balanced contrastive DataLoader (option-(b) labels), pulls a few batches,
prints a sanity report, and writes visual-debug plots. Designed to run as a plain
SLURM batch job:

    python -u run_data_pipeline.py --data-mode synthetic --num-workers 4 \
           --out-dir ./pipeline_out --n-debug-plots 6

then, once your paths are set, switch to real data:

    python -u run_data_pipeline.py --data-mode real --specs-json specs.json \
           --num-workers 8

`specs.json` (real mode) is a list of records:
    [{"folder": "/path/ptrain_Control00_Well11", "base": "ptrain_Control00_Well11_", "condition": 0},
     {"folder": "/path/pgroup02_Well14",         "base": "pgroup02_Well14_",         "condition": 1}]

This script is the data/augmentation front-end (Topic 1). The model + loss +
training loop (Topics 2-3) plug in at the marked EXTENSION POINT.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from augmentation import AugmentationConfig
from augmentation_viz import plot_triplet_instance
from data_pipeline import (
    CONTROL, PATHO,
    closest_power_of_2,
    SyntheticTraceProvider, NeuronalTracesProvider, NumpyTraceProvider,
    MEAWindowDataset, ConditionBalancedBatchSampler, TripletCollator, seed_worker,
)


# --------------------------------------------------------------------------- #
# configuration (everything explicit; CLI flags override a subset)
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    # --- data ---
    data_mode: str = "synthetic"          # "synthetic" | "real" | "numpy"
    specs_json: str = ""                  # real mode: path to the specs list
    npz_specs: str  = ""                  # numpy mode: path to burst_specs.json
    n_control: int = 2                    # synthetic mode
    n_patho: int = 1
    duration_s: float = 600.0
    fs: float = 50.0                      # synthetic fs (real fs comes from the loader)

    # --- windowing ---
    window_s: float = 200.0
    stride_s: float = 100.0               # stride < window_s -> overlapping windows

    # --- batching ---
    windows_per_condition: int = 2        # per batch, per condition
    n_batches: int = 20
    num_workers: int = 2
    pin_memory: bool = False

    # --- augmentation (subset; rest defaults in AugmentationConfig) ---
    split_method: str = "warp_bands"      # "warp_bands" | "percentile_mse"
    percentile_q: float = 0.30
    n_positives: int = 30
    n_negatives: int = 30
    shift_magnitude_s: float = 30.0

    # --- labels (option b) ---
    destroyed_label_mode: str = "unique"  # "unique" | "shared"

    # --- runtime / HPC ---
    seed: int = 0
    deterministic: bool = True
    torch_threads: int = 1                # intra-op threads (avoid oversubscription with workers)
    device: str = "auto"                  # "auto" | "cpu" | "cuda"  (model device; aug is always CPU)

    # --- debug ---
    out_dir: str = "./pipeline_out"
    n_debug_plots: int = 4


def parse_args() -> PipelineConfig:
    cfg = PipelineConfig()
    p = argparse.ArgumentParser(description="Contrastive data-pipeline front-end")
    p.add_argument("--data-mode", choices=["synthetic", "real", "numpy"], default=cfg.data_mode)
    p.add_argument("--specs-json", default=cfg.specs_json)
    p.add_argument("--npz-specs",  default=cfg.npz_specs,
                   help="numpy mode: path to burst_specs.json from generate_burst_data.py")
    p.add_argument("--n-control", type=int, default=cfg.n_control)
    p.add_argument("--n-patho", type=int, default=cfg.n_patho)
    p.add_argument("--duration-s", type=float, default=cfg.duration_s)
    p.add_argument("--fs", type=float, default=cfg.fs)
    p.add_argument("--window-s", type=float, default=cfg.window_s)
    p.add_argument("--stride-s", type=float, default=cfg.stride_s)
    p.add_argument("--windows-per-condition", type=int, default=cfg.windows_per_condition)
    p.add_argument("--n-batches", type=int, default=cfg.n_batches)
    p.add_argument("--num-workers", type=int, default=cfg.num_workers)
    p.add_argument("--pin-memory", action="store_true", default=cfg.pin_memory)
    p.add_argument("--split-method", choices=["warp_bands", "percentile_mse"], default=cfg.split_method)
    p.add_argument("--percentile-q", type=float, default=cfg.percentile_q)
    p.add_argument("--n-positives", type=int, default=cfg.n_positives)
    p.add_argument("--n-negatives", type=int, default=cfg.n_negatives)
    p.add_argument("--shift-magnitude-s", type=float, default=cfg.shift_magnitude_s)
    p.add_argument("--destroyed-label-mode", choices=["unique", "shared"], default=cfg.destroyed_label_mode)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--no-deterministic", dest="deterministic", action="store_false", default=cfg.deterministic)
    p.add_argument("--torch-threads", type=int, default=cfg.torch_threads)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=cfg.device)
    p.add_argument("--out-dir", default=cfg.out_dir)
    p.add_argument("--n-debug-plots", type=int, default=cfg.n_debug_plots)
    a = p.parse_args()
    return PipelineConfig(**vars(a))


# --------------------------------------------------------------------------- #
# setup helpers
# --------------------------------------------------------------------------- #
def set_reproducibility(seed: int, deterministic: bool, torch_threads: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, torch_threads))
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def load_traces(cfg: PipelineConfig):
    """Return (traces, conditions, fs). Synthetic or real (project loader)."""
    if cfg.data_mode == "synthetic":
        provider = SyntheticTraceProvider(duration_s=cfg.duration_s, fs=cfg.fs, seed=cfg.seed)
        traces, conditions = [], []
        for tid in range(cfg.n_control):
            tr, fs = provider(CONTROL, tid); traces.append(tr); conditions.append(CONTROL)
        for tid in range(cfg.n_patho):
            tr, fs = provider(PATHO, tid); traces.append(tr); conditions.append(PATHO)
        return traces, conditions, cfg.fs

    # --- numpy mode (generated .npz files from generate_burst_data.py) ---
    if cfg.data_mode == "numpy":
        if not cfg.npz_specs or not os.path.exists(cfg.npz_specs):
            raise FileNotFoundError(f"--npz-specs not found: {cfg.npz_specs!r}")
        with open(cfg.npz_specs) as fh:
            specs = json.load(fh)
        provider = NumpyTraceProvider()
        traces, conditions, fs_common = [], [], None
        for rec in specs:
            tr, fs = provider(rec["npz_path"])
            if fs_common is None:
                fs_common = fs
            elif abs(fs - fs_common) > 1e-9:
                raise ValueError("All .npz traces must share the same f_s^{IFR}.")
            traces.append(tr); conditions.append(int(rec["condition"]))
        return traces, conditions, fs_common

    # --- real mode ---
    if not cfg.specs_json or not os.path.exists(cfg.specs_json):
        raise FileNotFoundError(f"--specs-json not found: {cfg.specs_json!r}")
    with open(cfg.specs_json) as fh:
        specs = json.load(fh)
    # import the engine's loader lazily (directive 1: reuse the tested function)
    try:
        from oneD_cnn_functions import Neuronal_traces  # adjust module name if different
    except Exception as exc:                              # pragma: no cover
        raise ImportError(
            "Could not import Neuronal_traces from the engine module. Edit the "
            "import in load_traces() to point at your engine file."
        ) from exc
    provider = NeuronalTracesProvider(Neuronal_traces)
    traces, conditions, fs_common = [], [], None
    for rec in specs:
        tr, fs = provider(rec["folder"], rec["base"])
        if fs_common is None:
            fs_common = fs
        elif abs(fs - fs_common) > 1e-9:
            raise ValueError("All traces must share the same sampling rate.")
        traces.append(tr); conditions.append(int(rec["condition"]))
    return traces, conditions, fs_common


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    cfg = parse_args()
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_reproducibility(cfg.seed, cfg.deterministic, cfg.torch_threads)
    device = resolve_device(cfg.device)

    print("=" * 70, flush=True)
    print("Contrastive data-pipeline front-end", flush=True)
    print(f"  data_mode={cfg.data_mode}  device(model)={device}  aug device=cpu(workers)", flush=True)
    print("=" * 70, flush=True)

    # ---- data ----
    traces, conditions, fs = load_traces(cfg)
    n_ctrl = sum(c == CONTROL for c in conditions)
    n_path = sum(c == PATHO for c in conditions)
    print(f"Loaded {len(traces)} traces  (control={n_ctrl}, patho={n_path}), fs={fs} Hz", flush=True)

    window_length = closest_power_of_2(cfg.window_s * fs)
    stride = max(1, int(cfg.stride_s * fs))
    print(f"window_length={window_length} samples ({window_length / fs:.2f} s), "
          f"stride={stride} samples ({stride / fs:.2f} s)", flush=True)

    aug_cfg = AugmentationConfig(
        fs=fs,
        split_method=cfg.split_method,
        percentile_q=cfg.percentile_q,
        n_positives=cfg.n_positives,
        n_negatives=cfg.n_negatives,
        shift_magnitude_s=cfg.shift_magnitude_s,
    )

    dataset = MEAWindowDataset(traces, conditions, window_length, stride, aug_cfg, base_seed=cfg.seed)
    print(f"dataset windows: {len(dataset)}  "
          f"(control={int((dataset.conditions_per_item == CONTROL).sum())}, "
          f"patho={int((dataset.conditions_per_item == PATHO).sum())})", flush=True)

    batch_sampler = ConditionBalancedBatchSampler(
        dataset.conditions_per_item, cfg.windows_per_condition, cfg.n_batches, seed=cfg.seed)
    collator = TripletCollator(destroyed_label_mode=cfg.destroyed_label_mode)

    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        worker_init_fn=seed_worker,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.num_workers > 0),
    )

    # ---- sanity report over a few batches ----
    print("\n-- batch sanity report --", flush=True)
    for bi, (X, y, metas) in enumerate(loader):
        n_ctrl_lab = int((y == CONTROL).sum())
        n_path_lab = int((y == PATHO).sum())
        neg_mask = y >= collator.unique_label_base if cfg.destroyed_label_mode == "unique" else (y == collator.shared_destroyed_label)
        n_neg = int(neg_mask.sum())
        n_uniq_neg = int(torch.unique(y[neg_mask]).numel())
        print(f"  batch {bi:02d}: X={tuple(X.shape)} y={tuple(y.shape)} "
              f"| pos(control={n_ctrl_lab}, patho={n_path_lab}) "
              f"| neg={n_neg} (unique labels={n_uniq_neg}) "
              f"| source windows={len(metas)}", flush=True)
        if bi >= 2:
            break

    # ---- visual debug plots (from dataset items; separation of concerns) ----
    if cfg.n_debug_plots > 0:
        dbg_dir = os.path.join(cfg.out_dir, "aug_debug")
        rng = np.random.default_rng(cfg.seed)
        picks = rng.choice(len(dataset), size=min(cfg.n_debug_plots, len(dataset)), replace=False)
        for j, idx in enumerate(picks):
            item = dataset[int(idx)]
            cond_name = "control" if item["condition"] == CONTROL else "patho"
            plot_triplet_instance(
                item["anchor"], item["positives"], item["negatives"], fs=fs,
                out_dir=dbg_dir, instance_id=j,
                title=f"{cond_name} | window meta={item['meta']} | split={cfg.split_method}")
        print(f"\nWrote {len(picks)} debug plots to {dbg_dir}", flush=True)

    # ---- persist config + a tiny report ----
    with open(os.path.join(cfg.out_dir, "pipeline_config.json"), "w") as fh:
        json.dump(asdict(cfg), fh, indent=2)

    # =====================================================================
    # EXTENSION POINT (Topics 2-3): plug the refactored backbone + loss here.
    #   model = OneDCNNBackbone(...).to(device)
    #   for X, y, _ in loader:
    #       emb = model(X.unsqueeze(1).to(device))          # (M, 1, T) -> (M, E)
    #       hard = miner(emb, y.to(device))
    #       loss = loss_fn(emb, y.to(device), hard)
    #       ...
    # =====================================================================
    print("\nDone. (Model/loss extension point is marked in run_data_pipeline.py.)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
