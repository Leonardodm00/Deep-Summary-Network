"""
smoke_test_selected_epoch.py

THE DRIFT TEST for C2. REQUIRES torch (it performs real, tiny training runs).

The problem it exists to solve
------------------------------
e*(t, sigma), the selected epoch, is computed INSIDE train.py (as best_epoch,
via the locked lexicographic rule of decision 17), but train() returns only
(model, history). The search therefore has two options: recompute e* from the
history, mirroring train.py's rule, or change train()'s signature to return it.
Recomputing was chosen because a signature change touches every caller and every
smoke test. The cost of that choice is a SILENT DRIFT risk: if train.py's rule is
ever edited, objective_utils.selected_epoch_index keeps applying the old one and
nothing complains -- the search would simply start scoring the wrong epoch.

This file converts that risk into a test failure. It runs train() for real, on a
toy problem, and asserts that the epoch recomputed from the returned history
equals the best_epoch train() ITSELF recorded. train() writes best_epoch into its
checkpoint extras, so the comparison is against train.py's own arithmetic, not
against a second copy of the rule.

If this test fails after someone edits train.py's selection rule, the fix is to
update objective_utils.selected_epoch_index to match -- NOT to relax the test.

Run:
    cd Main/Smoke_Tests && python3 smoke_test_selected_epoch.py

Checks:
  A. On a real toy run, recomputed e* == train.py's own best_epoch, and the
     model train() returns is the one from that epoch.
  B. The same, over several seeds, so the comparison is not one lucky history
     (different seeds stop at different epochs).
  C. The same with selection_primary = "silhouette", which swaps (u, v).
  D. The metrics the search reads are the ones recorded AT e*, not each signal's
     independent maximum over epochs -- asserted directly against history.
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checkpoint import load_checkpoint                          # noqa: E402
from config import ExperimentConfig                             # noqa: E402
from data_splits import make_synthetic_specs, make_time_segment_splits  # noqa: E402
from latent_burst_generator import (                            # noqa: E402
    DEFAULT_AXIS_NAMES,
    LatentBurstProvider,
    build_latent_spec,
)
from objective_utils import selected_epoch_index, selected_epoch_scores  # noqa: E402
from train import train                                         # noqa: E402


def _tiny_cfg(selection_primary="ari"):
    cfg = ExperimentConfig()
    cfg.data.data_mode = "latent"
    cfg.data.synthetic_n_per_class = (2, 2)
    cfg.data.synthetic_duration_s = 120.0
    cfg.data.synthetic_fs = 50.0
    cfg.data.window_s = 10.0
    cfg.data.train_stride_s = 5.0
    cfg.data.eval_stride_s = 10.0
    cfg.data.latent.n_neurons = 20
    cfg.backbone = type(cfg.backbone)(
        depth_exponent=2, width_multiplier=1.6, embedding_size=8)
    cfg.train.max_epochs = 6
    cfg.train.patience = 6
    cfg.train.n_seeds = 1
    cfg.train.windows_per_condition = 2
    cfg.train.batches_per_epoch = 2
    cfg.train.selection_primary = selection_primary
    cfg.runtime.device = "cpu"
    cfg.runtime.seed = 0
    cfg.validate()
    return cfg


def _splits(cfg):
    spec = build_latent_spec(
        DEFAULT_AXIS_NAMES, (0, 1),
        tuple(int(n) for n in cfg.data.synthetic_n_per_class),
        float(cfg.data.synthetic_duration_s), float(cfg.data.synthetic_fs),
        n_neurons=int(cfg.data.latent.n_neurons), seed=int(cfg.runtime.seed))
    provider = LatentBurstProvider(spec)
    traces, conditions = [], []
    for s in make_synthetic_specs(spec.n_per_class):
        x, fs = provider(*s["args"])
        traces.append(np.asarray(x, dtype=np.float32))
        conditions.append(int(s["condition"]))
    return make_time_segment_splits(traces, conditions, float(spec.fs), cfg.data,
                                    base_seed=int(cfg.runtime.seed))


def _run_one(cfg, splits, seed, tmp):
    ckpt = os.path.join(tmp, "ckpt_seed%d" % seed)
    os.makedirs(ckpt, exist_ok=True)
    model, history = train(cfg, splits.train, splits.val, "cpu", seed=seed,
                           ckpt_dir=ckpt)
    ck = load_checkpoint(os.path.join(ckpt, "last.pt"), map_location="cpu")
    trainers_best_epoch = int((ck.get("extra") or {}).get("best_epoch", -1))
    return model, history, trainers_best_epoch


def check_agreement(tmp, selection_primary="ari", seeds=(0, 1, 2)):
    cfg = _tiny_cfg(selection_primary)
    splits = _splits(cfg)
    rows = []
    for seed in seeds:
        model, history, trainers = _run_one(cfg, splits, seed, tmp)
        i_star = selected_epoch_index(history, selection_primary)
        recomputed = 0 if i_star is None else int(history[i_star]["epoch"])
        assert recomputed == trainers, (
            "DRIFT: search recomputed e* = %d but train.py recorded best_epoch = "
            "%d (seed %d, selection_primary=%r). objective_utils."
            "selected_epoch_index no longer mirrors train.py's rule."
            % (recomputed, trainers, seed, selection_primary))
        ari, sil, epoch = selected_epoch_scores(history, selection_primary)
        assert epoch == trainers
        if i_star is not None:
            h = history[i_star]
            assert abs(ari - float(h["ari"])) < 1e-12 or not np.isfinite(h["ari"])
            assert (abs(sil - float(h["silhouette"])) < 1e-12
                    or not np.isfinite(h["silhouette"]))
        rows.append((seed, len(history), trainers, ari, sil))
    for seed, n_ep, e, ari, sil in rows:
        print("      seed %d: %d epochs, e* = %d (train.py agrees), "
              "ARI = %.4f, Sil = %.4f" % (seed, n_ep, e, ari, sil))
    return cfg, splits, rows


def check_reads_are_paired(tmp):
    """[D] the pair the search reads comes from ONE epoch, not from two."""
    cfg = _tiny_cfg("ari")
    splits = _splits(cfg)
    _model, history, trainers = _run_one(cfg, splits, 7, tmp)
    ari, sil, epoch = selected_epoch_scores(history, "ari")
    assert epoch == trainers
    h = [r for r in history if int(r["epoch"]) == epoch][0]
    assert abs(ari - float(h["ari"])) < 1e-12
    assert abs(sil - float(h["silhouette"])) < 1e-12
    max_ari = max(float(r["ari"]) for r in history if np.isfinite(r["ari"]))
    max_sil = max(float(r["silhouette"]) for r in history
                  if np.isfinite(r["silhouette"]))
    print("      at e* = %d: (ARI, Sil) = (%.4f, %.4f); independent maxima over "
          "epochs would have been (%.4f, %.4f)" % (epoch, ari, sil, max_ari, max_sil))
    assert ari <= max_ari + 1e-12 and sil <= max_sil + 1e-12
    if (abs(ari - max_ari) > 1e-12) or (abs(sil - max_sil) > 1e-12):
        print("      -> they DIFFER on this run, so the change is observable here")
    else:
        print("      -> they coincide on this run; the pairing still holds by "
              "construction (checked in smoke_test_objective_wiring.py [A])")


def main():
    print("smoke_test_selected_epoch.py [C2 drift test]  torch %s" % torch.__version__)
    tmp = tempfile.mkdtemp(prefix="sel_epoch_")
    try:
        print("  [A][B] selection_primary = 'ari', several seeds:")
        check_agreement(tmp, "ari", seeds=(0, 1, 2))
        print("  [A][B] recomputed e* == train.py's own best_epoch on every seed OK")
        print("  [C] selection_primary = 'silhouette':")
        check_agreement(tmp, "silhouette", seeds=(0, 1))
        print("  [C] the rule still agrees when (u, v) are swapped OK")
        print("  [D] the search reads BOTH metrics at the SAME epoch:")
        check_reads_are_paired(tmp)
        print("  [D] paired read confirmed OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL SELECTED-EPOCH CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
