"""
dsn_embed.py -- EMBEDDING EXTRACTION LAYER ONLY.

Separation of concerns (directive 2): this module turns a checkpoint into
embeddings and their metrics. It does NOT train, does NOT plot, and does NOT
write figures. dsn_embed_figures draws; plot_embeddings.py orchestrates.

Directive 1 (leverage the ecosystem): every scientific step here is delegated to
the repository's own tested implementation rather than reimplemented --
  checkpoint.rebuild_model_from_checkpoint   model + weights
  checkpoint.experiment_config_from_checkpoint  the config the model was fit under
  run_optimization.build_traces / build_cultures / build_splits  the ONE split path
  inference.embed_clean_windows              the embedding
  metrics.clustering_metrics                 ARI / AMI / silhouette + labels_pred
  metrics.embedding_health                   collapse diagnostics
No scoring mathematics is written in this file. In particular labels_pred is
taken from clustering_metrics and passed through, so the clusters SCORED and the
clusters DRAWN are the same fit -- the invariant evaluate.py exists to protect.

WHY THE CONFIG COMES FROM THE CHECKPOINT
The checkpoint embeds the full ExperimentConfig, so the splits are rebuilt from
the same runtime.seed, split_mode, trace_split_seed and window geometry the
model was trained under. Passing a config file separately would let a plot be
made against a different split than the model ever saw, silently. A --config
override exists only for relocating runtime.cache_dir / out_dir, and it warns.

Pure ASCII (hpc-python-compat).
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np

__all__ = ["EmbeddingSet", "load_model_and_config", "build_split_bundle",
           "embed_split", "embed_all_splits", "SPLIT_NAMES"]

SPLIT_NAMES = ("train", "val", "test")


def _ensure_repo_on_path(main_dir):
    """Put the repository's Main/ on sys.path so its modules import as top-level
    names, exactly as run_optimization.py does when run from Main/."""
    main_dir = os.path.abspath(main_dir)
    if not os.path.isfile(os.path.join(main_dir, "config.py")):
        raise SystemExit(
            "not a repository Main/ directory (no config.py): %s\n"
            "  pass --main-dir /path/to/Deep_multich/Main" % main_dir)
    if main_dir not in sys.path:
        sys.path.insert(0, main_dir)
    return main_dir


class EmbeddingSet(object):
    """Everything one (checkpoint, split) pair produces. Plain data, no methods
    that compute: the numbers are all made by the repository's own functions."""

    def __init__(self, split, Z, y, labels_pred, metrics, health, cultures,
                 culture_ids, tag):
        self.split = str(split)
        self.Z = Z                       # (N, E) float32, L2-normalised rows
        self.y = y                       # (N,) int64 TRUE phenotype
        self.labels_pred = labels_pred   # (N,) int64 K-means, K = C
        self.metrics = dict(metrics)     # ari, ami, silhouette, n_clusters
        self.health = dict(health)       # min_std, mean_std, eff_rank, ...
        self.cultures = cultures         # (N,) culture id per window, or None
        self.culture_ids = culture_ids   # the split's culture list, or None
        self.tag = str(tag)              # e.g. "refit_top1 / seed_0"

    @property
    def n(self):
        return int(self.Z.shape[0])

    @property
    def dim(self):
        return int(self.Z.shape[1])

    def __repr__(self):
        return ("EmbeddingSet(%s, N=%d, E=%d, ari=%.4f, sil=%.4f)"
                % (self.split, self.n, self.dim,
                   float(self.metrics.get("ari", float("nan"))),
                   float(self.metrics.get("silhouette", float("nan")))))


def load_model_and_config(ckpt_path, main_dir, map_location="cpu",
                          config_override=None):
    """(model, experiment_config, ckpt) rebuilt from a checkpoint.

    config_override, if given, is a path to a config JSON whose runtime block
    replaces the checkpoint's. Use it ONLY to relocate cache_dir / out_dir when
    the run has moved; anything else silently changes the splits.
    """
    _ensure_repo_on_path(main_dir)
    import checkpoint as C
    from config import ExperimentConfig

    model, ckpt = C.rebuild_model_from_checkpoint(ckpt_path,
                                                  map_location=map_location)
    cfg, _ = C.experiment_config_from_checkpoint(ckpt_path,
                                                 map_location=map_location)
    if config_override:
        import json
        with open(config_override, "r") as fh:
            over = json.load(fh)
        rt = over.get("runtime", {})
        for key in ("cache_dir", "out_dir"):
            if key in rt:
                warnings.warn(
                    "runtime.%s overridden from %r to %r. This is safe only for "
                    "relocating files; it must not change which data is loaded."
                    % (key, getattr(cfg.runtime, key), rt[key]), RuntimeWarning)
                setattr(cfg.runtime, key, rt[key])
    model.eval()
    return model, cfg, ckpt


def build_split_bundle(cfg, main_dir, verbose=False):
    """The SplitBundle, via the repository's single split path.

    Reuses run_optimization.build_traces / build_cultures / build_splits so the
    windows here are byte-identical to the ones the model was scored on.
    """
    _ensure_repo_on_path(main_dir)
    import run_optimization as R

    traces, conditions, fs = R.build_traces(cfg, verbose=verbose)
    cultures = R.build_cultures(cfg)
    splits = R.build_splits(cfg, traces, conditions, fs, verbose=verbose,
                            cultures=cultures)
    return splits, fs


def _dataset_cultures(dataset):
    """(per-window culture id array, sorted unique culture list) or (None, None).

    MEAWindowDataset.cultures is indexed by TRACE, not by window: it has one
    entry per trace, while .index has one (trace_idx, start, condition) tuple
    per window. The per-window map is therefore cultures[trace_idx] gathered
    over .index -- reading .cultures directly would silently mis-length, and
    that is exactly the bug this function exists to avoid.

    Returns (None, None) when the dataset exposes no culture information; the
    caller must then skip the per-culture panel rather than fabricate one.
    """
    idx = getattr(dataset, "index", None)
    cultures = getattr(dataset, "cultures", None)
    if idx is None or cultures is None:
        return None, None
    cultures = list(cultures)
    try:
        trace_ids = np.asarray([int(t[0]) for t in idx], dtype=np.int64)
    except (TypeError, IndexError, ValueError):
        return None, None
    if trace_ids.size == 0 or trace_ids.max() >= len(cultures):
        warnings.warn(
            "dataset.cultures has %d entries but .index references trace %d; "
            "the per-culture panel is SKIPPED rather than guessed."
            % (len(cultures), int(trace_ids.max())), RuntimeWarning)
        return None, None
    per_window = np.asarray([cultures[t] for t in trace_ids])
    return per_window, np.unique(per_window)


def embed_split(model, splits, split, cfg, device="cpu", tag="", seed=None,
                main_dir=None, batch_size=256):
    """Embed ONE split and score it. Returns an EmbeddingSet.

    K for K-means is C, the number of phenotype classes in the FULL label set,
    not the number present in this split -- the locked invariant stated in
    evaluate.py. It is passed explicitly rather than inferred here.
    """
    _ensure_repo_on_path(main_dir)
    import inference as I
    import metrics as M

    if split not in SPLIT_NAMES:
        raise ValueError("split must be one of %s; got %r" % (SPLIT_NAMES, split))
    dataset = getattr(splits, split)
    Z, y = I.embed_clean_windows(model, dataset, device, batch_size=batch_size)

    n_classes = int(np.unique(
        np.concatenate([np.asarray(getattr(splits, s).conditions_per_item,
                                   dtype=np.int64).ravel()
                        for s in SPLIT_NAMES])).size)
    ev = getattr(cfg, "eval", None)
    km_seed = int(getattr(ev, "kmeans_seed", 0)) if ev is not None else 0
    n_init = int(getattr(ev, "kmeans_n_init", 10)) if ev is not None else 10
    sil_metric = (getattr(ev, "silhouette_metric", "cosine")
                  if ev is not None else "cosine")

    res = M.clustering_metrics(Z, y, seed=km_seed, n_clusters=n_classes,
                               n_init=n_init, silhouette_metric=sil_metric)
    labels_pred = np.asarray(res.pop("labels_pred"), dtype=np.int64)
    health = M.embedding_health(Z)
    cult, cult_ids = _dataset_cultures(dataset)
    return EmbeddingSet(split, np.asarray(Z, dtype=np.float32),
                        np.asarray(y, dtype=np.int64), labels_pred, res,
                        health, cult, cult_ids, tag)


def embed_all_splits(model, splits, cfg, device="cpu", tag="", main_dir=None,
                     which=SPLIT_NAMES, batch_size=256):
    """{split_name: EmbeddingSet} for each requested split."""
    return dict(
        (s, embed_split(model, splits, s, cfg, device=device, tag=tag,
                        main_dir=main_dir, batch_size=batch_size))
        for s in which)
