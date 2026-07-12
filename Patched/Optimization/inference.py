"""
inference.py
============

The single, shared "embed clean windows" routine. Separation of concerns
(directive 2): this module does INFERENCE ONLY -- it never trains, never scores,
never plots, never augments. It is imported by BOTH the trainer (train.py, for
its per-epoch validation embedding) and the evaluator (evaluate.py, Stage 6), so
the embed step exists in exactly ONE place and the two can never drift apart.

Why this module exists (the legacy bug it removes)
--------------------------------------------------
The legacy evaluation embedded a RESAMPLED-WITH-REPLACEMENT set of windows, so
the embedding matrix contained DUPLICATED rows. Duplicated rows inflate the
apparent sample size N and bias every clustering statistic computed from it
(ARI / AMI / silhouette all assume N distinct observations). Here we instead
enumerate the dataset's OWN window index exactly once, so the returned rows are
distinct BY CONSTRUCTION.

What "clean" means
------------------
We slice the raw window straight out of the trace and DO NOT call the
augmentation pipeline:

    x_i = trace_{t_i}[ s_i : s_i + W ],    i = 1, ..., N

for the i-th entry (t_i, s_i, c_i) of dataset.index, where
    t_i : index of the source trace for window i,
    s_i : window start offset in samples, within that trace,
    W   : dataset.window_length, the window length in samples,
    c_i : the phenotype label of window i, c_i in {0, ..., C-1}.

This is precisely MEAWindowDataset.__getitem__'s "anchor" (the clean, UNSHIFTED
window) but WITHOUT paying for the surrogate generation that __getitem__ would
also do. It is the distribution the network sees at inference time, so the
embedding is calibrated on it.

Notation
--------
    N : number of windows in the dataset, N = len(dataset.index)
    W : window length in samples (dataset.window_length)
    E : embedding dimension (cfg.backbone.embedding_size)
    M : forward-pass mini-batch size (batch_size argument; a pure throughput
        knob -- it does NOT change the returned Z, only how many rows are pushed
        through the network at a time)
    X : (M, W) float32 batch of clean windows
    Z : (N, E) float32 embedding matrix, rows z_i = f_theta(x_i); each row is
        L2-normalized (||z_i||_2 = 1) whenever cfg.l2_normalize is True in the
        BackboneConfig, which is the default -- the normalization happens INSIDE
        the backbone head, not here.
    y : (N,) int64 phenotype labels, y_i = c_i

Determinism / model state
-------------------------
The model is switched to .eval() and the forward runs under torch.no_grad(). The
backbone uses GroupNorm (never BatchNorm), so eval() and train() compute the SAME
function -- there are no batch statistics to pollute, and the embedding does not
depend on how the rows are grouped into forward batches. Dropout, when enabled,
IS disabled by .eval(), which is what we want for a deterministic evaluation
embedding. The caller's train/eval mode is RESTORED on exit, so calling this in
the middle of a training loop cannot silently leave the model in eval mode.

HPC note (hpc-python-compat): pure ASCII. Import chain (torch, numpy) is safe.
"""

import numpy as np
import torch

__all__ = [
    "clean_windows",
    "embed_clean_windows",
]


def clean_windows(dataset) -> np.ndarray:
    """Stack the dataset's CLEAN (un-augmented, unshifted) windows.

    Enumerates dataset.index exactly once, in order, so the rows are distinct by
    construction (one row per (trace_idx, start) pair).

    Parameters
    ----------
    dataset : MEAWindowDataset -- needs .index (list of (trace_idx, start,
              condition)), .traces (list of 1-D float arrays) and .window_length.

    Returns
    -------
    X : (N, W) float32 numpy array, X[i] = traces[t_i][s_i : s_i + W]
    """
    index = dataset.index
    W = int(dataset.window_length)
    N = len(index)
    if N == 0:
        raise ValueError("dataset has no windows (empty .index)")

    X = np.empty((N, W), dtype=np.float32)
    for i, (ti, s, _cond) in enumerate(index):
        win = dataset.traces[ti][s:s + W]
        if win.shape[0] != W:
            raise ValueError(
                "window %d of trace %d has length %d, expected %d; the dataset "
                "index and the traces disagree." % (i, ti, win.shape[0], W))
        X[i] = win
    return X


def embed_clean_windows(model, dataset, device, batch_size: int = 256):
    """Embed every CLEAN window of `dataset` exactly once.

    Parameters
    ----------
    model      : the backbone (forward accepts (M, W) and returns (M, E))
    dataset    : MEAWindowDataset (see clean_windows)
    device     : torch.device (or a string accepted by torch.device)
    batch_size : M, forward-pass chunk size. Throughput knob only: the returned
                 Z is identical for any M, because the backbone uses GroupNorm
                 (no cross-sample batch statistics).

    Returns
    -------
    Z : (N, E) float32 numpy array -- one embedding row per DISTINCT window, in
        dataset.index order. Rows are L2-normalized when the backbone config has
        l2_normalize=True (the default); the normalization is done by the model.
    y : (N,)  int64  numpy array -- the phenotype label of each window, taken
        from dataset.conditions_per_item (aligned with dataset.index).

    Notes
    -----
    N == len(dataset.index) always, so no window is duplicated and none is
    dropped. The model's original training/eval mode is restored before return.
    """
    if int(batch_size) < 1:
        raise ValueError("batch_size must be >= 1")

    device = torch.device(device)
    X = clean_windows(dataset)                    # (N, W) float32, distinct rows
    N = X.shape[0]

    y = np.asarray(dataset.conditions_per_item, dtype=np.int64).ravel()
    if y.shape[0] != N:
        raise ValueError(
            "dataset.conditions_per_item has %d entries but .index has %d windows"
            % (y.shape[0], N))

    was_training = model.training
    model.eval()
    chunks = []
    try:
        with torch.no_grad():
            for start in range(0, N, int(batch_size)):
                xb = torch.from_numpy(X[start:start + int(batch_size)]).to(device)
                zb = model(xb)                    # (M, E), L2-normalized by the head
                chunks.append(zb.detach().to("cpu", dtype=torch.float32).numpy())
    finally:
        if was_training:
            model.train()                         # restore the caller's mode

    Z = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
    Z = np.ascontiguousarray(Z, dtype=np.float32)
    if Z.shape[0] != N:
        raise RuntimeError(
            "embed_clean_windows produced %d rows for %d windows" % (Z.shape[0], N))
    return Z, y
