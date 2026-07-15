"""
train.py
========

The core trainer. ONE function, train(), used IDENTICALLY by the Stage-7 HPO
objective and by the Stage-8 final run -- that identity is the whole point: a
hyper-parameter configuration is scored by exactly the procedure that will later
be used to fit the final model, so the search cannot optimize a proxy that
differs from deployment.

Separation of concerns (directive 2): this module OPTIMIZES ONLY. It does not
load data (data_pipeline / data_splits), does not score (metrics), does not embed
(inference), does not persist (checkpoint), does not plot (Stage 6). It calls
those tested modules and owns nothing but the optimization loop.

Scope: train() runs a SINGLE seed. The N_seeds averaging demanded by the
objective (decision 3) is done by the Stage-7 caller, which invokes train()
n_seeds times and averages the returned best validation metrics. Keeping the
seed loop OUT of here is what lets the search report an honest seed-to-seed std.

Notation (carried in full; symbols introduced at first use)
-----------------------------------------------------------
Data / batching
    C          : number of distinct phenotype classes, labels in {0, ..., C-1}
    B_c        : cfg.train.windows_per_condition, source windows drawn from EACH
                 class per batch
    n_batches  : cfg.train.batches_per_epoch, batches per epoch. When it is 0 it
                 is DERIVED, for the given training set, as
                     n_batches = ceil( N_train / (C * B_c) ),
                 with N_train = len(train_ds.index) the number of training
                 windows: one nominal pass over the training windows per epoch.
    P_b, N_b   : for source window b of a batch, the number of profile-PRESERVING
                 surrogates (positives, including the clean anchor) and
                 profile-DESTROYING surrogates (negatives) produced by the
                 augmentation module for that window.
    M          : rows in one embedding batch, M = sum_b (1 + P_b + N_b), summed
                 over the C * B_c source windows b of the batch.
    X          : (M, T) float32, the batch of windows fed to the network; T is the
                 window length in samples.
    y          : (M,) int64 labels under the locked option-(b) scheme:
                 y_r = c_b in {0, ..., C-1} for every positive row r coming from a
                 source window b of phenotype c_b, and y_r = a UNIQUE integer
                 (>= TripletCollator.unique_label_base) for every destroyed
                 surrogate row r. A unique label can never be matched by a second
                 row, so a destroyed surrogate is, by construction, only ever
                 usable as a NEGATIVE -- it is a per-anchor hard negative and
                 never a positive for anything.

Model / objective
    theta      : the network parameters
    f_theta    : the backbone, f_theta(X) = Z
    Z          : (M, E) float32 embeddings, E = cfg.backbone.embedding_size. Rows
                 are L2-normalized by the backbone head (cfg.backbone
                 .l2_normalize, default True), so every z_r lies on the unit
                 hypersphere S^{E-1}.
    d_cos(u,v) : cosine distance, d_cos(u, v) = 1 - (u . v) / (||u||_2 ||v||_2).
    m          : cfg.train.margin, the loss margin. On L2-normalized embeddings
                 with the cosine distance, m is a COSINE-SIMILARITY GAP, not a raw
                 angular margin: the two are related nonlinearly through
                 cos(angle). (Decision 7. Flagged explicitly so the searched range
                 in Stage 7 is not misread as an angle in radians.)
    T_mined    : the set of triplets (a, p, n) returned by the miner for a batch,
                 where a, p, n are ROW indices into Z with y_a = y_p and
                 y_a != y_n. |T_mined| may be 0 for a batch (no triplet violates
                 the margin); such a batch contributes 0 to the loss.
    L_batch    : the reduced triplet loss over T_mined, as computed by
                 pytorch_metric_learning's TripletMarginLoss with
                 reducers.AvgNonZeroReducer (the mean over the NON-ZERO triplet
                 losses only, so already-satisfied triplets do not dilute the
                 signal toward 0).
    L_epoch    : the epoch training loss reported in history, the mean of L_batch
                 over the n_batches batches of the epoch,
                     L_epoch = (1 / n_batches) * sum_{batches} L_batch.
                 It is the BACKPROP TARGET ONLY and is NEVER a selection metric
                 (decision 3).

Sign convention (verified against the library, not assumed)
-----------------------------------------------------------
    distances.CosineSimilarity is an INVERTED metric (is_inverted = True: larger
    means CLOSER). Both the loss and the miner read that flag and internally form
    the triplet margin as (d_ap - d_an) for inverted metrics instead of
    (d_an - d_ap). Therefore the margin m keeps its usual "how much closer the
    positive must be than the negative" meaning and NO manual sign flip is done
    here. CRITICAL: the miner's DEFAULT distance is LpDistance (Euclidean), so the
    SAME CosineSimilarity object family must be passed EXPLICITLY to the miner as
    well as to the loss; otherwise the miner would select triplets under Euclidean
    geometry while the loss scored them under cosine -- a silent
    miner/objective mismatch.

Model selection (decision 3 + the locked early-stopping rule, decision 17)
--------------------------------------------------------------------------
After each epoch e (e = 1, ..., E_max) the model is evaluated on the VALIDATION
split by embedding its CLEAN, DISTINCT windows (inference.embed_clean_windows)
and scoring them with metrics.clustering_metrics:

    ARI_e : adjusted Rand index at epoch e, of a single seeded K-means (K = C) on
            the full-dimensional validation embeddings, against the true labels.
    Sil_e : mean silhouette at epoch e, against the TRUE labels, with cosine
            distance.

Let ARI*_{e-1} and Sil*_{e-1} be the running best-so-far over epochs 1..e-1
(with ARI*_0 = Sil*_0 = -infinity), delta = cfg.train.min_delta_ari, and
epsilon = cfg.train.min_delta_sil. Write (u_e, v_e) for the (primary, secondary)
signals: (u_e, v_e) = (ARI_e, Sil_e) when cfg.train.selection_primary == "ari"
(the default), and (u_e, v_e) = (Sil_e, ARI_e) when it is "silhouette"; delta and
epsilon follow their signal. Then epoch e counts as an IMPROVEMENT iff

    u_e > u*_{e-1} + delta                                     (primary improves)
      OR ( u_e <= u*_{e-1} + delta   AND   v_e > v*_{e-1} + epsilon )
                                        (primary flat AND secondary improves)

The patience counter increments ONLY when BOTH clauses fail (i.e. both signals
have flattened), and RESETS to 0 on any improvement. Training stops at

    E_stop = min(e_patience, E_max),

where e_patience is the first epoch at which the counter reaches
P = cfg.train.patience. The BEST epoch is the lexicographic argmax over e of the
pair (u_e, v_e) -- primary first, secondary as the tie-break -- and the
BEST-EPOCH weights are RESTORED into the returned model (NOT the last epoch's).
A NaN metric (possible on a degenerate embedding) is treated as -infinity for
both comparison and argmax, so it can never win selection or reset patience.

Anti-collapse (decision 5): metrics.embedding_health is computed on the same
validation embedding and logged into history. It is MONITOR-ONLY and is NEVER
added to the loss.

HPC note (hpc-python-compat): pure ASCII. Every local module in the import chain
(config, backbone, augmentation, data_pipeline, metrics, checkpoint, inference) is
pure ASCII as well.
"""

import math
import random
import time
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from pytorch_metric_learning import distances, losses, miners, reducers

from backbone import build_backbone
from checkpoint import (
    CheckpointManager, capture_rng_state, restore_rng_state,
)
from data_pipeline import (
    ConditionBalancedBatchSampler, TripletCollator, seed_worker,
)
from inference import embed_clean_windows
from metrics import clustering_metrics, embedding_health

__all__ = [
    "build_loss_and_miner",
    "build_optimizer",
    "build_scheduler",
    "derive_batches_per_epoch",
    "reseed_dataset_rng",
    "set_global_seed",
    "resolve_device",
    "train",
]

_NEG_INF = float("-inf")


# --------------------------------------------------------------------------- #
# small, individually testable builders (kept public so the smoke test can probe
# them without running a full training loop)
# --------------------------------------------------------------------------- #
def set_global_seed(seed: int, deterministic: bool = True,
                    torch_threads: int = 1) -> None:
    """Seed torch / numpy / python and set the HPC thread + determinism policy."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, int(torch_threads)))
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(spec):
    """Resolve a device spec ("cpu" | "cuda" | "auto" | torch.device) to a device.
    CPU is the default; "auto" picks cuda only when it is actually available."""
    if isinstance(spec, torch.device):
        return spec
    s = str(spec)
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if s == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            "device='cuda' requested but CUDA is not available -> falling back to "
            "CPU.", RuntimeWarning)
        return torch.device("cpu")
    return torch.device(s)


def build_loss_and_miner(train_cfg):
    """Assemble the cosine triplet loss and the miner.

    The SAME distance family (CosineSimilarity, an inverted metric) is passed
    EXPLICITLY to both. Passing it only to the loss would leave the miner on its
    LpDistance default, so the mined triplets and the scored triplets would live
    in different geometries.

    Returns (loss_fn, miner).
    """
    loss_fn = losses.TripletMarginLoss(
        margin=float(train_cfg.margin),
        swap=bool(train_cfg.swap),
        distance=distances.CosineSimilarity(),
        reducer=reducers.AvgNonZeroReducer(),
    )
    if train_cfg.mining_strategy == "hard":
        miner = miners.TripletMarginMiner(
            margin=float(train_cfg.margin),
            type_of_triplets="hard",
            distance=distances.CosineSimilarity(),
        )
    elif train_cfg.mining_strategy == "easy_positive":
        miner = miners.BatchEasyHardMiner(
            pos_strategy="easy",
            neg_strategy="hard",
            distance=distances.CosineSimilarity(),
        )
    else:
        raise ValueError(
            "unknown mining_strategy %r (expected 'hard' or 'easy_positive')"
            % (train_cfg.mining_strategy,))
    return loss_fn, miner


def build_optimizer(model, train_cfg):
    """AdamW with the config's lr, betas and weight decay (decision 9 / 11:
    weight decay is present as AdamW wd throughout, tuned only in Stage 8)."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.lr),
        betas=(float(train_cfg.beta1), float(train_cfg.beta2)),
        weight_decay=float(train_cfg.weight_decay),
    )


def build_scheduler(optimizer, train_cfg):
    """Optional LR scheduler (decision 19: present but OFF by default).
    Returns None when disabled, so the caller's step is a no-op guard."""
    if not train_cfg.use_scheduler or train_cfg.scheduler_type == "none":
        return None
    if train_cfg.scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(train_cfg.max_epochs)))
    if train_cfg.scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, int(train_cfg.max_epochs) // 3), gamma=0.1)
    raise ValueError("unknown scheduler_type %r" % (train_cfg.scheduler_type,))


def reseed_dataset_rng(dataset, seed: int, epoch: int) -> None:
    """Re-seed the dataset's AUGMENTATION RNG deterministically for a given epoch.

    Why this is necessary (a real reproducibility hole, not a cosmetic touch):
    MEAWindowDataset owns a PERSISTENT numpy Generator (self.rng), created once in
    its __init__ from base_seed and ADVANCED by every __getitem__ call. The
    surrogate draw for a window therefore depends on how many items were drawn
    before it IN THE LIFETIME OF THAT DATASET OBJECT, not on the seed alone.

    That matters here because the Stage-7 objective calls train() n_seeds times on
    the SAME dataset instances: without this re-seed, run k inherits whatever RNG
    state run k-1 left behind, so two runs with identical seeds see DIFFERENT
    augmentations, HPO trials stop being reproducible, and the seed-to-seed std the
    search reports gets contaminated by dataset call-history effects.

    data_pipeline.seed_worker already handles this for num_workers > 0 (each worker
    process re-seeds its own copy of the dataset). It does NOT run when
    num_workers == 0 -- which is the CPU / HPC-safe default -- so the epoch loop
    re-seeds explicitly. We derive the stream from (seed, epoch) so that:
      * two train() calls with the same seed replay the same augmentations, and
      * a RESUMED run's epoch e sees the same stream as the uninterrupted run's
        epoch e (the state is a pure function of (seed, epoch), not of history).

    The tested data_pipeline.py is NOT modified (convention: do not silently
    rewrite the user's tested modules); we only re-seed the public .rng attribute
    it already exposes.
    """
    stream = (int(seed) * 1_000_003 + int(epoch)) % (2 ** 31)
    dataset.rng = np.random.default_rng(stream)
    dataset.base_seed = int(stream)   # so seed_worker (num_workers > 0) forks from it


def derive_batches_per_epoch(n_windows: int, n_classes: int,
                             windows_per_condition: int,
                             batches_per_epoch: int) -> int:
    """n_batches for the epoch's ConditionBalancedBatchSampler.

    A configured value >= 1 wins. A configured 0 means DERIVE one nominal pass
    over the training windows:

        n_batches = ceil( N_train / (C * B_c) ),   at least 1.

    N_train = n_windows, C = n_classes, B_c = windows_per_condition.
    """
    if int(batches_per_epoch) >= 1:
        return int(batches_per_epoch)
    per_batch = max(1, int(n_classes) * int(windows_per_condition))
    return max(1, int(math.ceil(float(n_windows) / float(per_batch))))


# --------------------------------------------------------------------------- #
# early-stopping bookkeeping (the exact locked rule, decision 17)
# --------------------------------------------------------------------------- #
def _finite_or_neg_inf(x) -> float:
    """NaN-safe read of a metric: a NaN (degenerate embedding) becomes -inf, so it
    can neither win the lexicographic argmax nor reset the patience counter."""
    v = float(x)
    return v if math.isfinite(v) else _NEG_INF


def _primary_secondary(metrics_dict, selection_primary, min_delta_ari,
                       min_delta_sil):
    """Map the epoch's metrics onto (u_e, v_e, delta, epsilon) per
    selection_primary. Returns (u, v, delta_primary, epsilon_secondary)."""
    ari = _finite_or_neg_inf(metrics_dict["ari"])
    sil = _finite_or_neg_inf(metrics_dict["silhouette"])
    if selection_primary == "ari":
        return ari, sil, float(min_delta_ari), float(min_delta_sil)
    if selection_primary == "silhouette":
        return sil, ari, float(min_delta_sil), float(min_delta_ari)
    raise ValueError("selection_primary must be 'ari' or 'silhouette'")


def _is_improvement(u_e, v_e, u_best, v_best, delta, epsilon) -> bool:
    """The locked composite rule (decision 17), written in its two-clause form so
    it can be audited line-by-line against the spec:

        improvement iff  u_e > u* + delta                        (clause 1)
                     OR ( u_e <= u* + delta                                    )
                        ( AND v_e > v* + epsilon )               (clause 2)

    i.e. the patience counter advances ONLY when BOTH signals have flattened.
    (u*, v*) = (u_best, v_best) are the running bests over the epochs strictly
    before e; each is the running best of its OWN signal.

    NOTE (equivalence, verified by truth table): clause 2's guard "u_e <= u* +
    delta" is exactly "NOT clause 1", and  A OR (NOT A AND B)  ==  A OR B, so the
    guard is logically redundant. It is kept EXPLICIT here for auditability rather
    than collapsed into a bare OR.
    """
    primary_improves = u_e > u_best + delta
    if primary_improves:
        return True                                   # clause 1
    primary_flat = not primary_improves               # == (u_e <= u* + delta)
    secondary_improves = v_e > v_best + epsilon
    return primary_flat and secondary_improves        # clause 2


# --------------------------------------------------------------------------- #
# the trainer
# --------------------------------------------------------------------------- #
def train(cfg, train_ds, val_ds, device, seed, ckpt_dir=None, verbose=False):
    """Train ONE seed and return the BEST-EPOCH model plus the epoch history.

    Parameters
    ----------
    cfg      : ExperimentConfig (the single source of truth; nothing is remembered
               in code -- the architecture comes from cfg.backbone, the optimizer
               from cfg.train, the scoring from cfg.eval)
    train_ds : MEAWindowDataset for the TRAIN time-segments (overlapping windows)
    val_ds   : MEAWindowDataset for the VAL time-segments (disjoint windows)
    device   : "cpu" | "cuda" | "auto" | torch.device
    seed     : the single seed for THIS run (Stage 7 varies it across n_seeds)
    ckpt_dir : if given, last / best / periodic checkpoints are written there and
               an existing last.pt is RESUMED from
    verbose  : print a per-epoch line (honours cfg.train.log_every_epochs)

    Returns
    -------
    (model, history)
      model   : the network with the BEST-EPOCH weights restored (not the last)
      history : list of per-epoch dicts, each with
                {epoch, train_loss, ari, ami, silhouette, n_triplets,
                 lr, seconds, health: {...}}
    """
    device = resolve_device(device)
    set_global_seed(seed, deterministic=cfg.runtime.deterministic,
                    torch_threads=cfg.runtime.torch_threads)

    tcfg = cfg.train

    # ---- model / loss / miner / optimizer / scheduler ----------------------
    model = build_backbone(cfg.backbone).to(device)
    loss_fn, miner = build_loss_and_miner(tcfg)
    optimizer = build_optimizer(model, tcfg)
    scheduler = build_scheduler(optimizer, tcfg)

    # AMP: GPU-only (decision 19). Guard on the resolved device, so a CPU run with
    # use_amp=True silently and safely runs in full precision.
    amp_enabled = bool(tcfg.use_amp) and device.type == "cuda"
    if bool(tcfg.use_amp) and not amp_enabled:
        warnings.warn(
            "use_amp=True but device is %r -> AMP is GPU-only and stays OFF."
            % device.type, RuntimeWarning)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # ---- batching ---------------------------------------------------------
    conditions = np.asarray(train_ds.conditions_per_item, dtype=int).ravel()
    n_classes = int(np.unique(conditions).shape[0])
    n_windows = int(len(train_ds.index))
    n_batches = derive_batches_per_epoch(
        n_windows=n_windows, n_classes=n_classes,
        windows_per_condition=tcfg.windows_per_condition,
        batches_per_epoch=tcfg.batches_per_epoch)

    sampler = ConditionBalancedBatchSampler(
        conditions=conditions,
        per_condition=int(tcfg.windows_per_condition),
        n_batches=n_batches,
        seed=int(seed),
    )
    collate = TripletCollator()

    loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=int(cfg.runtime.num_workers),
        worker_init_fn=seed_worker if int(cfg.runtime.num_workers) > 0 else None,
        pin_memory=bool(cfg.runtime.pin_memory) and device.type == "cuda",
    )

    # ---- K for K-means: K = C, the number of phenotype classes ---------------
    # C is the size of the FULL label set (train UNION val), NOT the number of
    # classes that happen to survive into the validation split.
    #
    # Why this distinction is load-bearing: data_splits.make_time_segment_splits
    # WARNS but does not raise when a phenotype ends up with no windows in a split.
    # If K were inferred from val alone, a validation split that lost a rare class
    # would silently fit K = C - 1 clusters to a C-phenotype problem, and the ARI /
    # AMI objective would change meaning BETWEEN HPO TRIALS -- the search would then
    # be comparing scores computed under different definitions. Taking the union
    # pins K to the experiment's true C for every trial.
    val_conditions = np.asarray(val_ds.conditions_per_item, dtype=int).ravel()
    classes_train = set(int(c) for c in np.unique(conditions))
    classes_val = set(int(c) for c in np.unique(val_conditions))
    all_classes = classes_train | classes_val
    n_clusters = int(len(all_classes))                       # K = C

    missing_in_val = sorted(all_classes - classes_val)
    if missing_in_val:
        warnings.warn(
            "phenotype(s) %s have NO windows in the validation split, but K-means "
            "is still fitted with K = C = %d (the full label set). ARI / AMI will "
            "compare a %d-cluster partition against only %d true classes present in "
            "validation, so the objective is degraded. Use a longer recording, a "
            "smaller window_s, or more traces per class."
            % (missing_in_val, n_clusters, n_clusters, len(classes_val)),
            RuntimeWarning)

    # ---- checkpointing / resume ------------------------------------------
    manager = None
    start_epoch = 0
    u_best, v_best = _NEG_INF, _NEG_INF
    best_epoch = 0
    best_state = None
    patience_counter = 0
    history = []

    if ckpt_dir is not None:
        manager = CheckpointManager(
            ckpt_dir, periodic_every=int(tcfg.checkpoint_every_epochs))
        ck = manager.load_last(map_location=device)
        if ck is not None:
            # resume: weights, optimizer, scheduler, RNG, counters. restore_rng_state
            # runs AFTER the model is built, because building consumed torch RNG.
            model.load_state_dict(ck["model_state"])
            if ck.get("optimizer_state") is not None:
                optimizer.load_state_dict(ck["optimizer_state"])
            if scheduler is not None and ck.get("scheduler_state") is not None:
                scheduler.load_state_dict(ck["scheduler_state"])
            restore_rng_state(ck.get("rng_state"))
            start_epoch = int(ck.get("epoch", 0))
            extra = ck.get("extra") or {}
            history = list(extra.get("history", []))
            best_epoch = int(extra.get("best_epoch", 0))
            patience_counter = int(extra.get("patience_counter", 0))
            u_best = float(extra.get("u_best", _NEG_INF))
            v_best = float(extra.get("v_best", _NEG_INF))
            best_state = extra.get("best_state", None)
            if best_state is not None:
                best_state = {k: v.clone() for k, v in best_state.items()}
            if verbose:
                print("[train] resumed from %s at epoch %d"
                      % (manager.last_path(), start_epoch))

    # ---- epoch loop -------------------------------------------------------
    E_max = int(tcfg.max_epochs)
    P = int(tcfg.patience)

    for epoch in range(start_epoch + 1, E_max + 1):
        t0 = time.time()
        model.train()
        sampler.set_epoch(epoch)               # per-epoch batch composition
        # make the augmentation stream a pure function of (seed, epoch): see
        # reseed_dataset_rng. Without this, a second train() call on the SAME
        # dataset object inherits the first call's RNG state.
        reseed_dataset_rng(train_ds, seed, epoch)

        # loss is accumulated ON-DEVICE; .item() is called ONCE per epoch, after
        # the loop (decision 13: one GPU sync per epoch, not per batch).
        loss_accum = torch.zeros((), device=device, dtype=torch.float32)
        triplets_epoch = 0

        for X, y, _metas in loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
                with torch.amp.autocast("cuda"):
                    Z = model(X)                       # (M, E), L2-normalized
                    pairs = miner(Z, y)
                    loss_val = loss_fn(Z, y, pairs)
                scaler.scale(loss_val).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                Z = model(X)
                pairs = miner(Z, y)                    # T_mined
                loss_val = loss_fn(Z, y, pairs)        # L_batch
                loss_val.backward()
                optimizer.step()

            loss_accum += loss_val.detach().to(torch.float32)
            triplets_epoch += int(pairs[0].numel())    # |T_mined| this batch

        if scheduler is not None:
            scheduler.step()

        # the ONE .item() of the epoch
        train_loss = float(loss_accum.item()) / max(1, n_batches)   # L_epoch

        # ---- validation: clean, DISTINCT windows -> metrics ---------------
        Z_val, y_val = embed_clean_windows(model, val_ds, device)
        m = clustering_metrics(
            Z_val, y_val,
            seed=int(cfg.eval.kmeans_seed),
            n_clusters=n_clusters,                     # K = C
            n_init=int(cfg.eval.kmeans_n_init),
            silhouette_metric=cfg.eval.silhouette_metric,
        )
        health = embedding_health(Z_val)               # monitor-only (decision 5)

        u_e, v_e, delta, epsilon = _primary_secondary(
            m, tcfg.selection_primary, tcfg.min_delta_ari, tcfg.min_delta_sil)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "ari": float(m["ari"]),
            "ami": float(m["ami"]),
            "silhouette": float(m["silhouette"]),
            "n_triplets": int(triplets_epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": float(time.time() - t0),
            "health": {k: float(health[k]) for k in
                       ("min_std", "mean_std", "eff_rank", "mean_pairwise_cos")},
        }
        history.append(record)

        # ---- the locked early-stopping / best-epoch rule ------------------
        improved = _is_improvement(u_e, v_e, u_best, v_best, delta, epsilon)

        # best epoch = lexicographic argmax over (u_e, v_e), NaN-safe
        is_new_best = (u_e, v_e) > (u_best, v_best)
        if is_new_best:
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        # running bests are updated component-wise: u* and v* are each the best
        # SO FAR of their own signal (that is what the rule compares against).
        u_best = max(u_best, u_e)
        v_best = max(v_best, v_e)

        if improved:
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch % int(tcfg.log_every_epochs) == 0):
            print("[train] epoch %3d | loss %8.5f | ARI %6.3f | AMI %6.3f | "
                  "sil %6.3f | triplets %6d | eff_rank %5.2f | patience %d/%d"
                  % (epoch, train_loss, record["ari"], record["ami"],
                     record["silhouette"], triplets_epoch,
                     record["health"]["eff_rank"], patience_counter, P))

        # ---- checkpoints --------------------------------------------------
        if manager is not None:
            extra = {
                "best_epoch": best_epoch,
                "patience_counter": patience_counter,
                "u_best": u_best,
                "v_best": v_best,
                "history": history,
                "best_state": best_state,
            }
            save_kw = dict(
                config=cfg, model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, best_metric={"ari": record["ari"],
                                          "silhouette": record["silhouette"]},
                rng_state=capture_rng_state(), extra=extra,
            )
            manager.save_last(**save_kw)
            if is_new_best:
                manager.save_best(**save_kw)
            manager.maybe_save_periodic(**save_kw)

        # E_stop = min(e_patience, E_max): stop as soon as the counter reaches P
        if patience_counter >= P:
            if verbose:
                print("[train] early stop at epoch %d (patience %d reached); "
                      "best epoch %d" % (epoch, P, best_epoch))
            break

    # ---- restore the BEST-EPOCH weights (never the last) -------------------
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    else:
        warnings.warn(
            "train(): no epoch produced a finite selection metric; returning the "
            "last-epoch weights. Check that the validation split has >= 2 "
            "phenotype classes with >= 2 windows each.", RuntimeWarning)

    return model, history
