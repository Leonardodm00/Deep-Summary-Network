"""
factor_retention.py
===================

[C5] How much of the LABEL-IRRELEVANT latent structure survives in the learned
embedding, measured as a cross-validated ridge R^2 from embedding to each known
latent coordinate.

Why this exists (and why eff_rank is not enough)
------------------------------------------------
eff_rank(Z) -- the participation ratio of Cov(Z) -- is a proxy for HOW MANY
dimensions an embedding uses. It cannot say WHICH structure survives, and on a
benchmark whose latent manifold is one-dimensional by construction it cannot even
do its proxy job: eff_rank ~= 1 is simultaneously the CORRECT answer and the
signature of representation collapse, so the two hypotheses make identical
predictions and the question is formally undecidable. Once the generator has n
independent latent factors of which only S carry the label, a sharper question
becomes askable: for each label-IRRELEVANT axis k not in S, is phi_k still
linearly decodable from the embedding?

That question is what makes the hard-mining vs easy-positive comparison
decidable. Both miners can reach the same ARI on the labels while differing in
what they DESTROY: pulling every positive of a class to one point is, under the
latent construction, an instruction to erase phi_k for every k not in S.

Separation of concerns (directive 2)
------------------------------------
  section 1 : ground-truth alignment (windows -> latent coordinates, no fitting)
  section 2 : the metric itself       (fitting only, no I/O, no plotting)
  section 3 : a thin convenience wrapper for the pipeline's own objects
This module never trains, never plots, never writes files.

Notation (symbols introduced at first use; carried in full)
-----------------------------------------------------------
    n           : number of latent factors, n in N, n >= 1
    k           : latent axis index, k in {1, ..., n} mathematically, 0-based in
                  code. Both are used below and every occurrence says which.
    S           : label-carrying axis subset, S subset= {1, ..., n}, S nonempty
    S^c         : the label-IRRELEVANT ("free") axes, S^c = {1, ..., n} \\ S
    N_eval      : number of evaluation windows, N_eval >= 2
    i           : evaluation-window index, i in {0, ..., N_eval - 1}
    E           : embedding dimension, E in N
    Z           : held-out embedding matrix, Z in R^{N_eval x E}, rows L2-normalized
    z_i         : row i of Z, z_i in R^E
    phi_k^(i)   : the TRUE k-th latent coordinate of the trace window i was cut
                  from, phi_k^(i) in [0, 1]
    g_i         : the GROUP of window i: the index of the trace it was cut from,
                  g_i in {0, ..., n_traces - 1}
    hatphi_k^(i)(Z) : the cross-validated ridge prediction of phi_k^(i) from z_i,
                  hatphi_k^(i)(Z) in R
    barphi_k    : the mean of phi_k over the evaluation windows,
                  barphi_k = (1 / N_eval) * sum_i phi_k^(i), barphi_k in [0, 1]
    R^2_k       : the factor-retention score of axis k, R^2_k in (-inf, 1]

    The metric, for each fixed axis k:

        R^2_k = 1 - sum_i ( phi_k^(i) - hatphi_k^(i)(Z) )^2
                    / sum_i ( phi_k^(i) - barphi_k )^2 ,                     (6)

    where BOTH sums run over all i in {0, ..., N_eval - 1} and hatphi_k^(i)(Z)
    is the prediction made by a model fitted on folds NOT containing window i.

    Note precisely which R^2 this is: predictions from every fold are POOLED and
    a single R^2 is computed against the GLOBAL mean barphi_k. That is Eq. (6)
    as written. It is NOT the average of per-fold R^2 values, which would use a
    different (per-fold) mean in each denominator and is a different quantity.

THE GROUPING CONSTRAINT (the part that is easy to get wrong)
------------------------------------------------------------
Every window cut from the same trace shares that trace's latent vector EXACTLY:
phi is a property of the TRACE, not of the window. With n_c traces per class and
C classes there are only C * n_c distinct values of phi_k in the entire dataset,
so the effective sample size of (6) is closer to C * n_c than to N_eval --
9 rather than 36 at the archived run's settings.

If the cross-validation splits by WINDOW, then for almost every held-out window
some sibling window of the SAME trace, carrying the IDENTICAL target value, sits
in the training fold. The ridge does not have to generalize across traces at all;
it can recognize the trace. R^2_k is then optimistically biased, and the bias is
largest exactly when the embedding has memorized trace identity -- the case the
metric is supposed to detect.

The cross-validation therefore MUST be grouped by trace
(sklearn.model_selection.GroupKFold with the trace index as the group). This is
the record-wise / subject-wise distinction: splitting by record rather than by
subject inflates apparent performance because records from one subject are not
independent. Raising n_c is the cheapest structural fix for the small effective
sample size, and is worth doing before reading much into any single R^2_k.

Interpretation
--------------
    R^2_k ~= 1   : axis k is (linearly) fully retained in the embedding
    R^2_k ~= 0   : axis k is no better predicted than by the constant barphi_k,
                   i.e. NOT linearly retained. Note R^2 = 0 is what a useless
                   predictor scores, so this is the reference level, not a floor.
    R^2_k <  0   : the fitted model does WORSE out-of-fold than that constant.
                   Common and unalarming at small n_groups; read it as "no
                   retained signal", not as anti-signal.
Only LINEAR decodability is measured. A negative R^2_k does not prove the
information is absent -- it proves it is not linearly available, which is the
operative sense for an embedding meant to be consumed by distance-based methods.

HPC note (hpc-python-compat): pure ASCII. Imports only numpy and scikit-learn.
"""

import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold

__all__ = [
    "DEFAULT_ALPHAS",
    "align_latents_to_windows",
    "grouped_cv_r2",
    "factor_retention",
    "factor_retention_from_ground_truth",
]

# Ridge penalty grid, selected INSIDE each training fold (see grouped_cv_r2).
DEFAULT_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)


# --------------------------------------------------------------------------- #
# section 1 -- ground-truth alignment (no fitting)
# --------------------------------------------------------------------------- #
def align_latents_to_windows(ground_truth: Dict[str, object],
                             groups: Sequence[int],
                             conditions: Optional[Sequence[int]] = None,
                             trace_ids: Optional[Sequence[int]] = None) -> np.ndarray:
    """Expand the per-TRACE latent table into a per-WINDOW matrix Phi.

    Parameters
    ----------
    ground_truth : the dict written by
                   latent_burst_generator.latent_ground_truth_table (or the
                   latent_ground_truth.json the driver saves), whose "rows" carry
                   {condition, trace_id, phi}.
    groups       : (N_eval,) g_i, the index of the trace window i was cut from,
                   in the SAME ordering the ground-truth rows were generated in
                   (that ordering is make_synthetic_specs order: class-major,
                   trace_id-minor). MEAWindowDataset.index supplies g_i as its
                   first element.
    conditions, trace_ids : optional (n_traces,) arrays giving, for each group
                   index, the (condition, trace_id) it corresponds to. When given,
                   rows are matched by (condition, trace_id) instead of by
                   position, which is the safe path if the manifest order is ever
                   not the generation order. When omitted, position is used and
                   the row count must match the group count.

    Returns
    -------
    Phi : (N_eval, n) float64, Phi[i, k] = phi_k^(i).
    """
    rows = list(ground_truth["rows"])
    n_latent = int(ground_truth["n_latent"])
    groups = np.asarray(groups, dtype=int).ravel()
    if groups.size < 1:
        raise ValueError("groups is empty")

    if conditions is None or trace_ids is None:
        n_traces = int(groups.max()) + 1
        if len(rows) < n_traces:
            raise ValueError(
                "ground truth has %d trace rows but the windows reference %d "
                "traces. Pass conditions/trace_ids to match by identity instead "
                "of by position, or regenerate the ground-truth table."
                % (len(rows), n_traces))
        phi_by_trace = np.asarray([r["phi"] for r in rows[:n_traces]], dtype=float)
    else:
        conditions = np.asarray(conditions, dtype=int).ravel()
        trace_ids = np.asarray(trace_ids, dtype=int).ravel()
        if conditions.shape != trace_ids.shape:
            raise ValueError("conditions and trace_ids must have equal length")
        lookup = {(int(r["condition"]), int(r["trace_id"])): np.asarray(r["phi"],
                                                                       dtype=float)
                  for r in rows}
        phi_by_trace = np.empty((conditions.shape[0], n_latent), dtype=float)
        for t, (c, tid) in enumerate(zip(conditions, trace_ids)):
            key = (int(c), int(tid))
            if key not in lookup:
                raise KeyError(
                    "no ground-truth row for (condition=%d, trace_id=%d); the "
                    "table was generated from a different LatentSpec than the "
                    "one that produced these traces." % key)
            phi_by_trace[t] = lookup[key]

    if phi_by_trace.shape[1] != n_latent:
        raise ValueError("ground truth declares n_latent=%d but rows carry %d"
                         % (n_latent, phi_by_trace.shape[1]))
    return phi_by_trace[groups, :]


# --------------------------------------------------------------------------- #
# section 2 -- the metric (fitting only)
# --------------------------------------------------------------------------- #
def grouped_cv_r2(Z: np.ndarray,
                  target: np.ndarray,
                  groups: Sequence[int],
                  n_splits: Optional[int] = None,
                  alphas: Sequence[float] = DEFAULT_ALPHAS) -> Dict[str, object]:
    """R^2 of Eq. (6) for ONE target, cross-validated with GroupKFold.

    The ridge penalty alpha is selected by RidgeCV WITHIN each training fold, so
    the held-out fold takes no part in choosing it. (RidgeCV's internal
    leave-one-out is by sample rather than by group; that is a hyper-parameter
    choice made on training data only and does not leak the held-out fold. It can
    make alpha slightly better-suited to within-trace structure, which if
    anything makes this metric more conservative about declaring an axis lost.)

    Parameters
    ----------
    Z        : (N_eval, E) embedding matrix.
    target   : (N_eval,) the true coordinate to predict, phi_k^(i) for fixed k.
    groups   : (N_eval,) g_i, the trace index of each window. THE GROUPING IS THE
               POINT -- see the module docstring.
    n_splits : number of folds; defaults to min(5, n_groups). Must satisfy
               2 <= n_splits <= n_groups, since GroupKFold cannot place more
               folds than there are groups.
    alphas   : ridge penalty grid.

    Returns
    -------
    dict with keys r2, ss_res, ss_tot, n_splits, n_groups, n_samples,
    predictions (the pooled out-of-fold hatphi^(i)).
    """
    Z = np.asarray(Z, dtype=float)
    target = np.asarray(target, dtype=float).ravel()
    groups = np.asarray(groups, dtype=int).ravel()
    if Z.ndim != 2:
        raise ValueError("Z must be 2-D (N_eval, E); got shape %r" % (Z.shape,))
    if not (Z.shape[0] == target.shape[0] == groups.shape[0]):
        raise ValueError(
            "Z, target and groups must agree on N_eval; got %d, %d, %d"
            % (Z.shape[0], target.shape[0], groups.shape[0]))

    uniq = np.unique(groups)
    n_groups = int(uniq.size)
    if n_groups < 2:
        raise ValueError(
            "need at least 2 distinct groups (traces) to cross-validate across "
            "them; got %d. With one trace there is nothing to generalize TO."
            % n_groups)
    if n_splits is None:
        n_splits = min(5, n_groups)
    n_splits = int(n_splits)
    if not (2 <= n_splits <= n_groups):
        raise ValueError(
            "n_splits must satisfy 2 <= n_splits <= n_groups (%d); got %d"
            % (n_groups, n_splits))

    pred = np.full(target.shape, np.nan, dtype=float)
    splitter = GroupKFold(n_splits=n_splits)
    for tr_idx, te_idx in splitter.split(Z, target, groups=groups):
        y_tr = target[tr_idx]
        if float(np.std(y_tr)) <= 0.0:
            # Every training trace shares one target value: the best available
            # model IS the constant, and RidgeCV cannot select alpha by LOO on a
            # constant target. Fall back to a fixed mild penalty.
            model = Ridge(alpha=1.0, fit_intercept=True)
        else:
            model = RidgeCV(alphas=np.asarray(alphas, dtype=float),
                            fit_intercept=True)
        model.fit(Z[tr_idx], y_tr)
        pred[te_idx] = model.predict(Z[te_idx])

    if np.any(~np.isfinite(pred)):
        raise RuntimeError("some windows received no out-of-fold prediction")

    bar = float(np.mean(target))                       # barphi_k
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - bar) ** 2))
    if ss_tot <= 0.0:
        # phi_k is constant across the WHOLE evaluation set, so Eq. (6) is 0/0.
        # Undefined rather than perfect: there is no variance to explain.
        warnings.warn(
            "grouped_cv_r2: the target is constant across all evaluation "
            "windows (sum_i (phi_k^(i) - barphi_k)^2 = 0), so R^2 is undefined "
            "(0/0) and is reported as NaN. This usually means the axis was not "
            "actually varied, or every window came from one trace.",
            RuntimeWarning)
        r2 = float("nan")
    else:
        r2 = float(1.0 - ss_res / ss_tot)

    return {"r2": r2, "ss_res": ss_res, "ss_tot": ss_tot,
            "n_splits": int(n_splits), "n_groups": n_groups,
            "n_samples": int(target.shape[0]),
            "predictions": pred}


def factor_retention(Z: np.ndarray,
                     Phi: np.ndarray,
                     groups: Sequence[int],
                     label_axes: Sequence[int] = (),
                     axis_names: Optional[Sequence[str]] = None,
                     n_splits: Optional[int] = None,
                     alphas: Sequence[float] = DEFAULT_ALPHAS) -> Dict[str, object]:
    """R^2_k for EVERY latent axis k, with the free axes S^c flagged as such.

    Both label and free axes are scored. The free axes are the measurement of
    interest -- they are the structure a miner can destroy without hurting ARI --
    but the label axes are reported alongside as a positive control: an embedding
    that scores well on the label but has lost the label axes themselves is
    telling you something odd about what it encoded.

    Parameters
    ----------
    Z          : (N_eval, E) embedding matrix, rows aligned with Phi and groups.
    Phi        : (N_eval, n) true latent coordinates, Phi[i, k] = phi_k^(i).
    groups     : (N_eval,) trace index g_i of each window.
    label_axes : S, 0-based indices into the axis ordering. Axes NOT in S are the
                 free axes S^c.
    axis_names : optional n names, for readable output.
    n_splits   : folds for GroupKFold; default min(5, n_groups).
    alphas     : ridge penalty grid.

    Returns
    -------
    dict with:
        per_axis   : list of n dicts {index, name, is_label_axis, r2, ss_res,
                     ss_tot, n_groups, n_splits, n_samples}
        free_axes  : the indices in S^c
        label_axes : the indices in S
        mean_r2_free_axes  : mean of R^2_k over k in S^c, NaN-skipping
        mean_r2_label_axes : mean of R^2_k over k in S
        n_groups, n_splits, n_samples, embedding_dim
    """
    Z = np.asarray(Z, dtype=float)
    Phi = np.asarray(Phi, dtype=float)
    if Phi.ndim != 2:
        raise ValueError("Phi must be 2-D (N_eval, n); got %r" % (Phi.shape,))
    if Z.shape[0] != Phi.shape[0]:
        raise ValueError("Z and Phi must agree on N_eval; got %d vs %d"
                         % (Z.shape[0], Phi.shape[0]))
    n_latent = int(Phi.shape[1])
    label_set = set(int(k) for k in label_axes)
    for k in label_set:
        if not (0 <= k < n_latent):
            raise ValueError("label axis index %d out of range [0, %d)"
                             % (k, n_latent))
    if axis_names is not None and len(axis_names) != n_latent:
        raise ValueError("axis_names has %d entries but Phi has %d columns"
                         % (len(axis_names), n_latent))

    per_axis: List[Dict[str, object]] = []
    for k in range(n_latent):
        out = grouped_cv_r2(Z, Phi[:, k], groups, n_splits=n_splits, alphas=alphas)
        per_axis.append({
            "index": int(k),
            "name": (str(axis_names[k]) if axis_names is not None
                     else "axis_%d" % k),
            "is_label_axis": bool(k in label_set),
            "r2": float(out["r2"]),
            "ss_res": float(out["ss_res"]),
            "ss_tot": float(out["ss_tot"]),
            "n_groups": int(out["n_groups"]),
            "n_splits": int(out["n_splits"]),
            "n_samples": int(out["n_samples"]),
        })

    free_idx = [k for k in range(n_latent) if k not in label_set]

    def _mean(idxs):
        vals = [per_axis[k]["r2"] for k in idxs]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "per_axis": per_axis,
        "free_axes": [int(k) for k in free_idx],
        "label_axes": sorted(int(k) for k in label_set),
        "mean_r2_free_axes": _mean(free_idx),
        "mean_r2_label_axes": _mean(sorted(label_set)),
        "n_groups": int(per_axis[0]["n_groups"]) if per_axis else 0,
        "n_splits": int(per_axis[0]["n_splits"]) if per_axis else 0,
        "n_samples": int(Z.shape[0]),
        "embedding_dim": int(Z.shape[1]),
    }


# --------------------------------------------------------------------------- #
# section 3 -- convenience wrapper for the pipeline's own objects
# --------------------------------------------------------------------------- #
def factor_retention_from_ground_truth(Z: np.ndarray,
                                       ground_truth: Dict[str, object],
                                       dataset,
                                       n_splits: Optional[int] = None,
                                       alphas: Sequence[float] = DEFAULT_ALPHAS
                                       ) -> Dict[str, object]:
    """factor_retention() driven straight from a MEAWindowDataset + the table.

    dataset must expose .index as a sequence of (trace_index, start, condition)
    triples, one per window, in the SAME row order as Z -- which is what
    MEAWindowDataset provides and what embed_clean_windows preserves. The trace
    index is used as the group g_i, which is exactly the grouping the metric
    requires.

    Z must be the embedding of the CLEAN (unaugmented) windows of that dataset,
    in dataset order.
    """
    index = list(getattr(dataset, "index"))
    if len(index) != int(np.asarray(Z).shape[0]):
        raise ValueError(
            "Z has %d rows but the dataset has %d windows; Z must be the "
            "embedding of THIS dataset's windows, in dataset order."
            % (np.asarray(Z).shape[0], len(index)))
    groups = np.asarray([int(t[0]) for t in index], dtype=int)
    Phi = align_latents_to_windows(ground_truth, groups)
    return factor_retention(
        Z, Phi, groups,
        label_axes=tuple(int(k) for k in ground_truth.get("label_axes", ())),
        axis_names=list(ground_truth.get("axis_names", [])) or None,
        n_splits=n_splits, alphas=alphas)
