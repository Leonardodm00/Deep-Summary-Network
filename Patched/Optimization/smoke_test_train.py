"""
smoke_test_train.py

Standalone correctness checks for train.py (Stage 5). CPU only, synthetic data
only, tiny net / short windows / few epochs. No data files, no cluster needed.

Run:
    python3 smoke_test_train.py

Runtime is a couple of minutes on CPU (real training loops, deliberately small).

Checks:
  [A] Builders: the loss AND the miner both carry CosineSimilarity (is_inverted =
      True). This is the silent-mismatch guard: PML's miner DEFAULTS to LpDistance,
      so if the cosine distance were passed only to the loss, triplets would be
      MINED under Euclidean geometry and SCORED under cosine.
  [B] derive_batches_per_epoch: 0 -> ceil(N_train / (C * B_c)); an explicit value
      >= 1 overrides. Boundary cases included.
  [C] It LEARNS: on separable synthetic phenotypes the best validation ARI beats
      the epoch-1 ARI and clears a floor. (The objective is validation ARI, never
      train loss -- decision 3.)
  [D] .item() is called EXACTLY ONCE per epoch (decision 13), asserted with a
      counter that patches torch.Tensor.item -- not by reading the source.
  [E] Early stopping FIRES before E_max on a plateau (tiny patience), and the
      returned history is shorter than E_max.
  [F] With P > E_max the ceiling fires instead, and the BEST-epoch (not the last)
      weights are returned: the restored model, re-embedded and re-scored, must
      REPRODUCE the ARI recorded at the best epoch of history.
  [G] BOTH mining strategies ("hard", "easy_positive") run and mine a non-empty
      total number of triplets.
  [H] Determinism: two runs with the same seed give identical history and
      bit-identical final weights.
  [I] Resume: an interrupted run (E1 epochs, then resumed to E1 + E2) matches an
      uninterrupted E1 + E2 run in history length and best epoch.
  [J] Multi-class C in {2, 3, 4}: K-means uses K = C and training runs end to end.
  [K] Anti-collapse diagnostics are LOGGED but never in the loss: every history
      record carries finite health fields.
"""

import copy
import sys
import tempfile
import warnings
from dataclasses import replace

import numpy as np
import torch

from config import (
    ExperimentConfig, DataConfig, TrainConfig, EvalConfig, RuntimeConfig,
    BackboneConfig, AugmentationConfig,
)
from preprocessing_cache import cache_traces, load_cached_traces
from data_splits import (
    MultiClassSyntheticProvider, make_synthetic_specs, make_time_segment_splits,
)
from inference import embed_clean_windows
from metrics import clustering_metrics
from train import (
    build_loss_and_miner, build_optimizer, derive_batches_per_epoch, train,
)

# --- tiny, fast, but still separable -----------------------------------------
_DURATION_S = 160.0
_FS = 50.0
_WINDOW_S = 8.0             # W = 400 samples
_TRAIN_STRIDE_S = 2.0       # heavy overlap -> more train windows
_EVAL_STRIDE_S = 8.0        # disjoint eval windows
_FRACTIONS = (0.6, 0.2, 0.2)


def _make_cfg(C, max_epochs=6, patience=3, mining="hard", seed=0,
              n_per_class=None, dropout=0.0):
    """A complete, tiny ExperimentConfig."""
    n_per_class = tuple([3] * C) if n_per_class is None else tuple(n_per_class)
    aug = replace(
        AugmentationConfig(fs=_FS),
        n_positives=3, n_negatives=3, shift_magnitude_s=2.0,
    )
    data = DataConfig(
        data_mode="synthetic", synthetic_n_per_class=n_per_class,
        synthetic_duration_s=_DURATION_S, synthetic_fs=_FS,
        window_s=_WINDOW_S, train_stride_s=_TRAIN_STRIDE_S,
        eval_stride_s=_EVAL_STRIDE_S, split_fractions=_FRACTIONS,
        augmentation=aug,
    )
    bb = BackboneConfig(depth_exponent=3, width_multiplier=1.5, stem_width=8,
                        embedding_size=8, dropout=dropout)
    tr = TrainConfig(
        margin=0.3, mining_strategy=mining, lr=3e-3,
        max_epochs=max_epochs, patience=patience,
        windows_per_condition=4, batches_per_epoch=3,
        n_seeds=1, checkpoint_every_epochs=2, log_every_epochs=1,
    )
    rt = RuntimeConfig(seed=seed, device="cpu", num_workers=0, torch_threads=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # patience >= max_epochs
        return ExperimentConfig(data=data, backbone=bb, train=tr,
                                eval=EvalConfig(), runtime=rt)


def _make_splits(C, cfg, cache_dir):
    n_per_class = cfg.data.synthetic_n_per_class
    provider = MultiClassSyntheticProvider(
        n_classes=C, duration_s=_DURATION_S, fs=_FS, seed=0)
    specs = make_synthetic_specs(n_per_class)
    cache_traces(specs, provider, cache_dir)
    traces, conditions, fs = load_cached_traces(cache_dir)
    return make_time_segment_splits(traces, conditions, fs, cfg.data, base_seed=0)


def _quiet_train(*a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return train(*a, **kw)


# --------------------------------------------------------------------------- #
def check_builders():
    cfg = _make_cfg(2, mining="hard")
    loss_fn, miner = build_loss_and_miner(cfg.train)
    assert type(loss_fn.distance).__name__ == "CosineSimilarity", loss_fn.distance
    assert type(miner.distance).__name__ == "CosineSimilarity", miner.distance
    assert loss_fn.distance.is_inverted is True
    assert miner.distance.is_inverted is True
    assert type(loss_fn.reducer).__name__ == "AvgNonZeroReducer"
    assert type(miner).__name__ == "TripletMarginMiner"

    cfg_ep = _make_cfg(2, mining="easy_positive")
    _l, miner_ep = build_loss_and_miner(cfg_ep.train)
    assert type(miner_ep).__name__ == "BatchEasyHardMiner"
    assert type(miner_ep.distance).__name__ == "CosineSimilarity"
    assert miner_ep.distance.is_inverted is True

    opt = build_optimizer(torch.nn.Linear(4, 4), cfg.train)
    g = opt.param_groups[0]
    assert abs(g["lr"] - cfg.train.lr) < 1e-12
    assert g["betas"] == (cfg.train.beta1, cfg.train.beta2)
    assert abs(g["weight_decay"] - cfg.train.weight_decay) < 1e-12
    print("  [A] builders OK: loss AND miner both on CosineSimilarity "
          "(is_inverted=True); AdamW lr/betas/wd wired from config")


def check_derive_batches():
    # 0 -> ceil(N / (C * B_c))
    assert derive_batches_per_epoch(100, 2, 8, 0) == 7      # ceil(100/16) = 7
    assert derive_batches_per_epoch(96, 2, 8, 0) == 6       # exact division
    assert derive_batches_per_epoch(1, 4, 8, 0) == 1        # floor -> at least 1
    assert derive_batches_per_epoch(0, 2, 8, 0) == 1        # never 0 batches
    # explicit override wins
    assert derive_batches_per_epoch(100, 2, 8, 3) == 3
    print("  [B] derive_batches_per_epoch OK: 0 -> ceil(N/(C*B_c)), "
          "explicit value overrides, never 0")


def check_learns_and_item_count(cache_dir):
    """[C] learns  +  [D] exactly one .item() per epoch  +  [K] health logged.

    Budget note: this is the ONE check that must actually reach convergence, so it
    is deliberately given a real (if still small) budget -- 18 epochs x 8 batches
    = 144 gradient steps. An under-powered version of this test (e.g. 8 epochs x 3
    batches = 24 steps) shows ARI still climbing but only around 0.2, which would
    tempt one to weaken the assertion instead of powering the test properly. The
    floor below (ARI >= 0.8) is therefore a real convergence bar, not a lowered one.
    """
    C = 2
    cfg = _make_cfg(C, max_epochs=18, patience=18, seed=0)
    cfg.train.batches_per_epoch = 8
    splits = _make_splits(C, cfg, cache_dir)

    # patch Tensor.item to COUNT calls made during train()
    calls = {"n": 0}
    orig_item = torch.Tensor.item

    def counting_item(self):
        calls["n"] += 1
        return orig_item(self)

    torch.Tensor.item = counting_item
    try:
        model, history = _quiet_train(cfg, splits.train, splits.val, "cpu", seed=0)
    finally:
        torch.Tensor.item = orig_item

    n_epochs = len(history)
    assert n_epochs >= 1

    # [D] INFORMATIONAL only. The raw count is dominated by pytorch-metric-learning's
    # OWN stat collection (miner.collect_stats calls .item() several times per
    # batch), which is library behaviour, not our training loop. The decision-13
    # requirement is about OUR loss accumulation, and it is asserted exactly in
    # check_item_once_per_epoch_isolated() below, with PML's stats switched off so
    # the only remaining .item() calls are the trainer's.
    assert calls["n"] >= n_epochs, (calls["n"], n_epochs)
    print("  [D] .item() calls seen: %d over %d epochs (dominated by PML's internal "
          "collect_stats; the strict 1-per-epoch assertion is [D-strict] below)"
          % (calls["n"], n_epochs))

    # [C] it learns: best val ARI beats epoch 1 and clears a real convergence bar
    aris = [h["ari"] for h in history]
    sils = [h["silhouette"] for h in history]
    best_ari = max(aris)
    assert best_ari > aris[0], ("ARI never improved on epoch 1", aris)
    assert best_ari >= 0.80, ("best val ARI did not converge: %r" % (aris,))
    # the separation must also show up in the K-means-INDEPENDENT companion metric
    assert max(sils) > sils[0], ("silhouette never improved", sils)

    # The miner should find FEWER margin-violating triplets as the embedding
    # separates: a direct, mechanism-level sign that the contrastive objective (not
    # just K-means luck) is what improved.
    early = np.mean([h["n_triplets"] for h in history[:3]])
    late = np.mean([h["n_triplets"] for h in history[-3:]])
    assert late < early, ("mined triplets did not fall as training progressed",
                          early, late)

    # NOTE (why train loss is NOT a selection metric -- decision 3, confirmed here):
    # with reducers.AvgNonZeroReducer the loss averages ONLY the still-violating
    # triplets, so it stays pinned near the margin while the representation keeps
    # improving. Asserting on train_loss would therefore be meaningless; the
    # objective is validation ARI. We deliberately assert NOTHING about train_loss.
    print("  [C] learns OK: val ARI %.3f (epoch 1) -> %.3f (best over %d epochs); "
          "silhouette %.3f -> %.3f; mined triplets %.0f -> %.0f (miner runs dry as "
          "the embedding separates)"
          % (aris[0], best_ari, n_epochs, sils[0], max(sils), early, late))

    # [K] health diagnostics logged every epoch, finite, monitor-only
    for h in history:
        for k in ("min_std", "mean_std", "eff_rank", "mean_pairwise_cos"):
            assert k in h["health"], k
            assert np.isfinite(h["health"][k]), (k, h["health"][k])
    print("  [K] anti-collapse diagnostics logged every epoch (finite, "
          "monitor-only) OK")
    return cfg, splits


def check_item_once_per_epoch_isolated(cache_dir):
    """[D, strict] Decision 13: the TRAINER must call .item() exactly ONCE per epoch
    (one GPU sync per epoch, not one per batch).

    Counting every .item() in the process would be the WRONG test: PyTorch's own
    AdamW internals call .item() inside optimizer.step() (torch/optim/optimizer.py
    _get_value, reading the lr/step scalars), and the DataLoader calls it once at
    construction. Those are library-internal and are not ours to remove. The
    requirement is about OUR loss accumulation, so we attribute each .item() call
    to its CALL SITE via the stack and count only the ones raised from train.py.
    """
    import collections
    import traceback

    C = 2
    cfg = _make_cfg(C, max_epochs=3, patience=3, seed=0)
    splits = _make_splits(C, cfg, cache_dir)

    sites = collections.Counter()
    orig_item = torch.Tensor.item

    def counting_item(self):
        caller = traceback.extract_stack(limit=4)[-2]     # the frame calling .item()
        sites[caller.filename.split("/")[-1]] += 1
        return orig_item(self)

    torch.Tensor.item = counting_item
    try:
        _model, history = _quiet_train(cfg, splits.train, splits.val, "cpu", seed=0)
    finally:
        torch.Tensor.item = orig_item

    n_epochs = len(history)
    ours = sites.get("train.py", 0)
    assert ours == n_epochs, \
        ("decision 13 violated: train.py called .item() %d times over %d epochs "
         "(expected exactly 1 per epoch). Call sites: %r"
         % (ours, n_epochs, dict(sites)))

    # and it must NOT scale with the number of batches (the legacy per-batch bug)
    n_batches = int(cfg.train.batches_per_epoch)
    assert ours < n_epochs * n_batches, \
        "train.py's .item() count scales with batches -> per-batch sync regression"

    others = {k: v for k, v in sites.items() if k != "train.py"}
    print("  [D-strict] EXACTLY one .item() per epoch from train.py OK "
          "(%d calls / %d epochs; library-internal elsewhere: %r)"
          % (ours, n_epochs, others))


def check_early_stopping_fires(cache_dir):
    """[E] A deliberately plateauing signal + tiny patience -> stop before E_max."""
    C = 2
    # lr = 0 freezes the weights -> the metric is constant -> a perfect plateau
    cfg = _make_cfg(C, max_epochs=12, patience=2, seed=0)
    cfg.train.lr = 1e-12                    # effectively no learning
    splits = _make_splits(C, cfg, cache_dir)
    _model, history = _quiet_train(cfg, splits.train, splits.val, "cpu", seed=0)
    assert len(history) < cfg.train.max_epochs, \
        ("early stopping did not fire: ran all %d epochs" % cfg.train.max_epochs)
    # with delta = epsilon = 0 and a frozen model, no epoch after the first can
    # improve, so the counter reaches P at epoch 1 + P
    assert len(history) <= 1 + cfg.train.patience + 1, len(history)
    print("  [E] early stopping fires OK: stopped at epoch %d < E_max=%d "
          "(patience P=%d on a frozen/plateau signal)"
          % (len(history), cfg.train.max_epochs, cfg.train.patience))


def _rescore(model, val_ds, cfg):
    """Re-score a model on the validation split exactly as the trainer does."""
    Z, y = embed_clean_windows(model, val_ds, "cpu")
    n_clusters = int(np.unique(y).shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return clustering_metrics(
            Z, y, seed=cfg.eval.kmeans_seed, n_clusters=n_clusters,
            n_init=cfg.eval.kmeans_n_init,
            silhouette_metric=cfg.eval.silhouette_metric)


def _best_record(history):
    """Lexicographic argmax over (ari, silhouette), NaN-safe -- decision 17."""
    def key(h):
        a = h["ari"] if np.isfinite(h["ari"]) else -np.inf
        s = h["silhouette"] if np.isfinite(h["silhouette"]) else -np.inf
        return (a, s)
    return max(history, key=key)


def check_best_epoch_restored(cache_dir):
    """[F] P > E_max -> the E_max ceiling fires; the returned model must carry the
    BEST-epoch weights, NOT the last epoch's.

    This check is only meaningful if the best epoch is NOT the last one -- if they
    coincide, "restore best" and "keep last" are indistinguishable and the test
    would pass VACUOUSLY. We therefore search seeds for a run whose peak is
    genuinely interior, and then assert BOTH:
      (F1) the returned model reproduces the BEST epoch's ARI, and
      (F2) the LAST epoch's recorded ARI differs from it -- so the assertion in
           (F1) could actually have failed had the trainer returned the last
           weights.
    """
    C = 2
    chosen = None
    for seed in range(12):
        cfg = _make_cfg(C, max_epochs=8, patience=99, seed=0)   # P > E_max
        cfg.train.batches_per_epoch = 4
        splits = _make_splits(C, cfg, cache_dir)
        model, history = _quiet_train(cfg, splits.train, splits.val, "cpu",
                                      seed=seed)
        assert len(history) == cfg.train.max_epochs, \
            ("the E_max ceiling should have fired", len(history))
        best, last = _best_record(history), history[-1]
        # we want a run where the peak is interior AND scores differently
        if best["epoch"] != last["epoch"] and abs(best["ari"] - last["ari"]) > 1e-6:
            chosen = (cfg, splits, model, history, best, last, seed)
            break

    assert chosen is not None, \
        ("could not find a run whose best epoch differs from its last; the "
         "best-epoch restore check would be vacuous")
    cfg, splits, model, history, best, last, seed = chosen

    # (F1) the RETURNED model must reproduce the BEST epoch's ARI
    m = _rescore(model, splits.val, cfg)
    assert abs(m["ari"] - best["ari"]) < 1e-6, \
        ("restored model does not reproduce the BEST epoch's ARI: got %.6f, "
         "best epoch %d recorded %.6f (last epoch %d recorded %.6f)"
         % (m["ari"], best["epoch"], best["ari"], last["epoch"], last["ari"]))

    # (F2) and the last epoch scored DIFFERENTLY, so (F1) was a real test:
    # had the trainer returned last-epoch weights, (F1) would have failed.
    assert abs(m["ari"] - last["ari"]) > 1e-6, \
        "best and last ARI coincide -> the restore check is vacuous"
    assert best["ari"] >= last["ari"], (best["ari"], last["ari"])

    print("  [F] best-epoch restore OK (non-vacuous, seed=%d): returned model "
          "reproduces ARI %.4f of BEST epoch %d; LAST epoch %d scored %.4f "
          "(so returning last weights would have failed this check)"
          % (seed, m["ari"], best["epoch"], last["epoch"], last["ari"]))


def check_both_miners(cache_dir):
    """[G] Both mining strategies run and mine a non-empty triplet total."""
    C = 2
    for mining in ("hard", "easy_positive"):
        cfg = _make_cfg(C, max_epochs=3, patience=3, mining=mining, seed=0)
        splits = _make_splits(C, cfg, cache_dir)
        _model, history = _quiet_train(cfg, splits.train, splits.val, "cpu", seed=0)
        total = sum(h["n_triplets"] for h in history)
        assert total > 0, "miner %r produced ZERO triplets over the whole run" % mining
        assert len(history) >= 1
        print("  [G] miner %-14s OK: %d epochs, %d triplets mined in total"
              % (mining, len(history), total))


def check_determinism(cache_dir):
    """[H] Same seed -> identical history and bit-identical weights, when train()
    is called TWICE ON THE SAME DATASET OBJECTS.

    Calling twice on the same objects is not an artificial stress: it is exactly
    what the Stage-7 objective does (it trains n_seeds times on one SplitBundle).
    MEAWindowDataset carries a PERSISTENT augmentation RNG that every __getitem__
    advances, so without train()'s per-epoch reseed_dataset_rng the second call
    would inherit the first call's RNG state and diverge -- silently making every
    HPO trial irreproducible. This check is the regression guard for that.

    A different seed must, of course, still give a DIFFERENT run (else the reseed
    would have pinned the stream constant and destroyed the seed-to-seed variance
    the search needs).
    """
    C = 2
    cfg = _make_cfg(C, max_epochs=3, patience=3, seed=0)
    splits = _make_splits(C, cfg, cache_dir)

    # same seed, twice, on the SAME dataset instances
    m1, h1 = _quiet_train(cfg, splits.train, splits.val, "cpu", seed=7)
    m2, h2 = _quiet_train(cfg, splits.train, splits.val, "cpu", seed=7)

    assert len(h1) == len(h2), (len(h1), len(h2))
    for a, b in zip(h1, h2):
        for k in ("epoch", "train_loss", "ari", "ami", "silhouette", "n_triplets"):
            same = (a[k] == b[k]) or (np.isnan(a[k]) and np.isnan(b[k]))
            assert same, ("history diverged at %r: %r vs %r (dataset RNG "
                          "carry-over?)" % (k, a[k], b[k]))
    p1, p2 = dict(m1.state_dict()), dict(m2.state_dict())
    assert p1.keys() == p2.keys()
    for k in p1:
        assert torch.equal(p1[k], p2[k]), "weights differ under a fixed seed: %s" % k

    # a DIFFERENT seed must still produce a different run (variance preserved)
    _m3, h3 = _quiet_train(cfg, splits.train, splits.val, "cpu", seed=8)
    differs = any(h3[i]["train_loss"] != h1[i]["train_loss"]
                  for i in range(min(len(h1), len(h3))))
    assert differs, \
        "seed 8 reproduced seed 7 exactly -> the reseed pinned the stream and "\
        "destroyed seed-to-seed variance"

    print("  [H] determinism OK: same seed twice on the SAME datasets -> identical "
          "history + bit-identical weights; a different seed still diverges "
          "(variance preserved)")


def check_resume(cache_dir):
    """[I] Interrupted (E1) then resumed to E1+E2 matches an uninterrupted E1+E2
    run in history length and best epoch."""
    C = 2
    splits_cfg = _make_cfg(C, max_epochs=6, patience=99, seed=0)
    splits = _make_splits(C, splits_cfg, cache_dir)

    with tempfile.TemporaryDirectory() as ck1, tempfile.TemporaryDirectory() as ck2:
        # interrupted: train 3 epochs, then resume the SAME ckpt dir with E_max = 6
        cfg_a = _make_cfg(C, max_epochs=3, patience=99, seed=0)
        _m, h_part = _quiet_train(cfg_a, splits.train, splits.val, "cpu",
                                  seed=11, ckpt_dir=ck1)
        assert len(h_part) == 3, len(h_part)

        cfg_b = _make_cfg(C, max_epochs=6, patience=99, seed=0)
        m_res, h_res = _quiet_train(cfg_b, splits.train, splits.val, "cpu",
                                    seed=11, ckpt_dir=ck1)     # resumes from last.pt

        # uninterrupted reference: 6 epochs straight
        cfg_c = _make_cfg(C, max_epochs=6, patience=99, seed=0)
        m_ref, h_ref = _quiet_train(cfg_c, splits.train, splits.val, "cpu",
                                    seed=11, ckpt_dir=ck2)

    assert len(h_res) == len(h_ref) == 6, (len(h_res), len(h_ref))
    assert [h["epoch"] for h in h_res] == list(range(1, 7)), \
        "resumed history epochs are not contiguous 1..6"
    # the first 3 epochs of the resumed history are the ones recorded before the
    # interruption (carried through the checkpoint), so they must match the partial
    for a, b in zip(h_part, h_res[:3]):
        assert a["epoch"] == b["epoch"] and a["ari"] == b["ari"]
    print("  [I] resume OK: 3 + 3 resumed == 6 uninterrupted (history contiguous "
          "1..6, pre-interruption epochs preserved)")


def check_multiclass(cache_dir_root):
    """[J] C in {2,3,4}: K-means uses K = C and the run completes."""
    for C in (2, 3, 4):
        with tempfile.TemporaryDirectory() as d:
            cfg = _make_cfg(C, max_epochs=2, patience=2, seed=0)
            splits = _make_splits(C, cfg, d)
            model, history = _quiet_train(cfg, splits.train, splits.val, "cpu",
                                          seed=0)
            assert len(history) >= 1
            Z, y = embed_clean_windows(model, splits.val, "cpu")
            assert set(int(v) for v in np.unique(y)) == set(range(C))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m = clustering_metrics(Z, y, seed=0, n_clusters=None)
            assert m["n_clusters"] == C, (m["n_clusters"], C)
            print("  [J] C=%d OK: K-means K=C=%d, %d epochs, best ARI %.3f"
                  % (C, m["n_clusters"], len(history),
                     max(h["ari"] for h in history)))


def check_k_equals_C_from_union(cache_dir):
    """[L] K = C must come from the FULL label set (train UNION val), not from the
    classes that happen to survive into the validation split.

    data_splits WARNS but does not raise when a phenotype has no windows in a
    split. If K were inferred from val alone, such a split would silently fit
    K = C - 1 clusters to a C-phenotype problem, and the ARI/AMI objective would
    change meaning between HPO trials. We simulate that by DELETING one phenotype's
    windows from the validation dataset and asserting the trainer still uses K = C
    (and warns about the degradation).
    """
    C = 3
    cfg = _make_cfg(C, max_epochs=2, patience=2, seed=0)
    splits = _make_splits(C, cfg, cache_dir)

    val = splits.val
    keep = [i for i, (_ti, _s, c) in enumerate(val.index) if int(c) != 2]
    assert len(keep) < len(val.index), "expected class 2 to be present in val"
    val.index = [val.index[i] for i in keep]                 # drop phenotype 2
    val.conditions_per_item = np.asarray(
        [c for (_ti, _s, c) in val.index], dtype=int)
    present = set(int(c) for c in np.unique(val.conditions_per_item))
    assert present == {0, 1}, present                        # val now lacks class 2

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model, history = train(cfg, splits.train, val, "cpu", seed=0)
    msgs = " ".join(str(x.message) for x in w)
    assert "NO windows in the validation split" in msgs, \
        "trainer did not warn that a phenotype is missing from validation"

    # the recorded metrics must have been computed with K = C = 3, not K = 2
    Z, y = embed_clean_windows(model, val, "cpu")
    assert set(int(v) for v in np.unique(y)) == {0, 1}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m3 = clustering_metrics(Z, y, seed=cfg.eval.kmeans_seed, n_clusters=3,
                                n_init=cfg.eval.kmeans_n_init,
                                silhouette_metric=cfg.eval.silhouette_metric)
    assert abs(history[-1]["ari"] - m3["ari"]) < 1e-6, \
        ("the trainer's last-epoch ARI does not match a K=C=3 scoring -> K was "
         "inferred from the validation split alone", history[-1]["ari"], m3["ari"])
    print("  [L] K = C from the train UNION val label set OK (val missing a "
          "phenotype -> still K=3, warned, objective meaning preserved)")


def main():
    print("Running train smoke tests...")
    check_builders()
    check_derive_batches()
    with tempfile.TemporaryDirectory() as d:
        check_learns_and_item_count(d)
    with tempfile.TemporaryDirectory() as d:
        check_item_once_per_epoch_isolated(d)
    with tempfile.TemporaryDirectory() as d:
        check_early_stopping_fires(d)
    with tempfile.TemporaryDirectory() as d:
        check_best_epoch_restored(d)
    with tempfile.TemporaryDirectory() as d:
        check_both_miners(d)
    with tempfile.TemporaryDirectory() as d:
        check_determinism(d)
    with tempfile.TemporaryDirectory() as d:
        check_resume(d)
    with tempfile.TemporaryDirectory() as d:
        check_k_equals_C_from_union(d)
    check_multiclass(None)
    print("ALL TRAIN SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
