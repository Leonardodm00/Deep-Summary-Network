"""
checkpoint.py
=============

Atomic, self-describing checkpointing for the contrastive pipeline. Separation of
concerns (directive 2): this module does persistence + reconstruction ONLY -- it
never trains, never evaluates, never plots.

What a checkpoint stores
------------------------
    format_version : int
    config         : ExperimentConfig.to_dict() -- a PLAIN dict, so the file can
                     be unpickled and the model rebuilt without depending on the
                     dataclass definitions being importable.
    model_state    : model.state_dict()
    optimizer_state: optimizer.state_dict() or None
    scheduler_state: scheduler.state_dict() or None
    epoch          : int (epochs completed)
    best_metric    : whatever the trainer tracks (float or dict) or None
    norm_stats     : input-normalization statistics used at train time, or None
    rng_state      : {torch, cuda, numpy, python} so a resumed step reproduces an
                     uninterrupted step bit-for-bit
    extra          : free-form dict (e.g. best_epoch, history)

Reconstruction (zero remembered hyper-parameters)
-------------------------------------------------
    rebuild_model_from_checkpoint reads ONLY the embedded config to build the
    model (build_backbone(BackboneConfig(**config["backbone"]))) and then loads
    the weights. No hyper-parameter is carried in code; the checkpoint is the sole
    source of truth. This is what makes HPO trials and the final run interchange-
    able and safe to resume.

Atomicity
---------
    Every write goes to a temp file in the destination directory and is then
    promoted with os.replace (atomic on POSIX). An interrupted or failing write
    leaves any previous checkpoint at the destination untouched, and no partial
    temp file is left behind.

HPC note (hpc-python-compat): pure ASCII. torch.load is called with
weights_only=False because the checkpoint intentionally contains non-tensor
objects (config dict, numpy / python RNG state); these are our own trusted files.
"""

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from backbone import build_backbone, BackboneConfig
from config import config_from_dict, ExperimentConfig

__all__ = [
    "FORMAT_VERSION",
    "capture_rng_state",
    "restore_rng_state",
    "save_checkpoint",
    "load_checkpoint",
    "rebuild_model_from_checkpoint",
    "experiment_config_from_checkpoint",
    "CheckpointManager",
]

FORMAT_VERSION = 1


# --------------------------------------------------------------------------- #
# RNG state capture / restore (torch + cuda + numpy + python)
# --------------------------------------------------------------------------- #
def capture_rng_state() -> dict:
    """Snapshot every RNG stream so training can resume deterministically."""
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state) -> None:
    """Restore RNG streams previously captured by capture_rng_state."""
    if state is None:
        return
    if state.get("torch") is not None:
        # torch expects a CPU uint8 ByteTensor
        torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8))
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# --------------------------------------------------------------------------- #
# atomic write
# --------------------------------------------------------------------------- #
def _atomic_torch_save(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".pt.tmp")
    os.close(fd)                       # torch.save reopens by name
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)          # atomic promotion on POSIX
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# --------------------------------------------------------------------------- #
# save / load
# --------------------------------------------------------------------------- #
def save_checkpoint(path, config, model, optimizer=None, scheduler=None,
                    epoch=0, best_metric=None, norm_stats=None,
                    rng_state=None, extra=None, capture_rng=True):
    """Atomically write a checkpoint. `config` may be an ExperimentConfig or a
    plain dict; it is stored as a dict. If rng_state is None and capture_rng is
    True, the current RNG streams are snapshotted."""
    if rng_state is None and capture_rng:
        rng_state = capture_rng_state()
    cfg_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    ckpt = {
        "format_version": FORMAT_VERSION,
        "config": cfg_dict,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "best_metric": best_metric,
        "norm_stats": norm_stats,
        "rng_state": rng_state,
        "extra": dict(extra) if extra is not None else {},
    }
    _atomic_torch_save(ckpt, Path(path))
    return Path(path)


def load_checkpoint(path, map_location="cpu") -> dict:
    """Load a checkpoint dict. weights_only=False is required because the file
    contains our own non-tensor objects (config dict, RNG state)."""
    return torch.load(path, map_location=map_location, weights_only=False)


def _as_ckpt(path_or_ckpt, map_location="cpu"):
    if isinstance(path_or_ckpt, dict):
        return path_or_ckpt
    return load_checkpoint(path_or_ckpt, map_location=map_location)


def rebuild_model_from_checkpoint(path_or_ckpt, map_location="cpu",
                                  build_fn=build_backbone,
                                  config_cls=BackboneConfig,
                                  config_key="backbone"):
    """Rebuild the model from the EMBEDDED config alone and load its weights.

    Returns (model, ckpt). The model architecture comes solely from
    ckpt["config"][config_key]; no hyper-parameter is remembered in code.
    """
    ckpt = _as_ckpt(path_or_ckpt, map_location=map_location)
    cfg_dict = ckpt["config"]
    sub = cfg_dict[config_key] if config_key is not None else cfg_dict
    model_cfg = config_from_dict(config_cls, sub)
    model = build_fn(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


def experiment_config_from_checkpoint(path_or_ckpt, map_location="cpu"):
    """Reconstruct the full ExperimentConfig embedded in a checkpoint.
    Returns (experiment_config, ckpt)."""
    ckpt = _as_ckpt(path_or_ckpt, map_location=map_location)
    return ExperimentConfig.from_dict(ckpt["config"]), ckpt


# --------------------------------------------------------------------------- #
# cadence manager (best / last / periodic) -- used by the trainer
# --------------------------------------------------------------------------- #
class CheckpointManager:
    """Thin helper that owns a checkpoint directory and the best / last / periodic
    filename convention. Each save_* forwards its kwargs to save_checkpoint."""

    def __init__(self, ckpt_dir, periodic_every: int = 0):
        self.dir = Path(ckpt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.periodic_every = int(periodic_every)

    def last_path(self):
        return self.dir / "last.pt"

    def best_path(self):
        return self.dir / "best.pt"

    def periodic_path(self, epoch):
        return self.dir / ("epoch_%04d.pt" % int(epoch))

    def save_last(self, **kw):
        return save_checkpoint(self.last_path(), **kw)

    def save_best(self, **kw):
        return save_checkpoint(self.best_path(), **kw)

    def maybe_save_periodic(self, **kw):
        epoch = int(kw.get("epoch", 0))
        if self.periodic_every > 0 and epoch % self.periodic_every == 0:
            return save_checkpoint(self.periodic_path(epoch), **kw)
        return None

    def has_last(self):
        return self.last_path().exists()

    def load_last(self, map_location="cpu"):
        p = self.last_path()
        return load_checkpoint(p, map_location=map_location) if p.exists() else None
