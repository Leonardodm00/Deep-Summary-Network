"""
metrics.py
==========

Pure scoring for the contrastive summary-network pipeline. Separation of concerns
(directive 2): this module computes numbers ONLY -- no training, no model, no
plotting, no data loading, no torch dependency (embeddings arrive as arrays;
torch tensors are accepted by duck typing and detached/moved/converted here).
Built on scikit-learn (directive 1) for the clustering metrics.

Two entry points
----------------
  clustering_metrics(Z, y, ...) : cross-trial cluster-recovery quality. This is
      the model-selection objective (locked decision: validation ARI / AMI, mean
      over seeds, NOT training loss). Exactly ONE seeded K-means is fit on the
      full-dimensional embeddings; its labels are RETURNED so the evaluator can
      colour its plot with the SAME labels used for the metric (metric / plot
      consistency).

  embedding_health(Z) : VICReg-style collapse diagnostics (locked decision:
      monitor-ONLY, never in the loss). Logged passively during training.

Notation
--------
    N        : number of embedding vectors (samples)
    E        : embedding dimension
    Z        : the embedding matrix, Z in R^{N x E}, rows z_i (i = 1..N)
    y        : true phenotype label per row, y_i in {0, ..., C-1}
    C        : number of distinct phenotype labels present in y
    z_bar    : mean embedding, z_bar = (1/N) sum_{i=1..N} z_i
    hat_z_i  : L2-normalized row, hat_z_i = z_i / ||z_i||_2   (||z_i||_2 > 0)

Clustering metrics
------------------
    labels_pred = KMeans(n_clusters = K, seed).fit_predict(Z)   (K = C by default)
    ari = adjusted_rand_score(y, labels_pred)                   in [-0.5, 1]
    ami = adjusted_mutual_info_score(y, labels_pred)            in [ ~0 , 1]
    silhouette : mean over i of s(i), computed against the TRUE labels y with
                 COSINE distance d_cos(u, v) = 1 - (u . v) / (||u|| ||v||):
        a(i) = mean_{j != i, y_j = y_i} d_cos(z_i, z_j)          (intra-cluster)
        b(i) = min_{ell != y_i} mean_{j : y_j = ell} d_cos(z_i, z_j)  (nearest other)
        s(i) = (b(i) - a(i)) / max(a(i), b(i)),  s(i) = 0 if y_i's cluster is a singleton
    (silhouette scores the TRUE phenotype separation in embedding space, so it is
    a K-means-independent companion / tie-breaker for model selection; cosine
    matches the unit-sphere geometry of the L2-normalized embeddings.)

Embedding-health diagnostics
----------------------------
    per-dimension sample std (ddof = 1): for each dim k = 1..E,
        sigma_k = sqrt( (1/(N-1)) sum_{i=1..N} (z_{i,k} - z_bar_k)^2 )
        min_std  = min_k sigma_k ,   mean_std = (1/E) sum_{k=1..E} sigma_k
    (a small min_std flags a collapsed dimension; VICReg's variance term.)

    effective rank (participation ratio of the covariance). With the sample
    covariance Sigma = (1/(N-1)) sum_{i=1..N} (z_i - z_bar)(z_i - z_bar)^T in
    R^{E x E} and its eigenvalues lambda_1 >= ... >= lambda_E >= 0,
        eff_rank = ( sum_{k=1..E} lambda_k )^2 / ( sum_{k=1..E} lambda_k^2 )
    which lies in [1, E]: eff_rank -> 1 when one eigenvalue dominates (collapse to
    a line / point), eff_rank = E when the covariance is isotropic. (Degenerate
    total collapse, all lambda_k = 0, is reported as eff_rank = 0.)

    mean pairwise cosine similarity:
        mean_pairwise_cos = (1 / (N(N-1))) sum_{i=1..N} sum_{j != i} hat_z_i . hat_z_j
    computed in O(N E) via the identity
        sum_{i,j} hat_z_i . hat_z_j = || sum_{i=1..N} hat_z_i ||_2^2 ,
    so mean_pairwise_cos = ( || sum_i hat_z_i ||^2 - N ) / ( N (N-1) )
    (subtracting the N unit self-terms). It approaches 1 under representation
    collapse (all directions aligned).

HPC note (hpc-python-compat): pure ASCII.
"""

import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    silhouette_score,
)

__all__ = [
    "clustering_metrics",
    "embedding_health",
    "effective_rank",
    "mean_pairwise_cosine",
]


def _to_numpy(a) -> np.ndarray:
    """Array-like / CPU-or-GPU torch tensor -> contiguous float64 numpy array.
    Duck-typed so this module stays torch-free."""
    if hasattr(a, "detach"):
        a = a.detach()
    if hasattr(a, "cpu"):
        a = a.cpu()
    if hasattr(a, "numpy"):
        a = a.numpy()
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64))


def _as_2d(Z) -> np.ndarray:
    Z = _to_numpy(Z)
    if Z.ndim != 2:
        raise ValueError("Z must be 2-D (N, E); got shape %r" % (Z.shape,))
    if Z.shape[0] < 1:
        raise ValueError("Z has no rows")
    return Z


# --------------------------------------------------------------------------- #
# health diagnostics (each also exposed standalone for unit testing)
# --------------------------------------------------------------------------- #
def effective_rank(Z) -> float:
    """Participation ratio of the sample-covariance eigenvalues (see module doc).
    Returns a float in [1, E]; 0.0 for total collapse (zero covariance)."""
    Z = _as_2d(Z)
    N = Z.shape[0]
    if N < 2:
        return float("nan")
    Sigma = np.cov(Z, rowvar=False, ddof=1)      # (E, E), rows are variables
    Sigma = np.atleast_2d(Sigma)
    # symmetric PSD -> real, non-negative eigenvalues (clamp numerical negatives)
    lam = np.linalg.eigvalsh(Sigma)
    lam = np.clip(lam, 0.0, None)
    s1 = float(lam.sum())
    s2 = float((lam * lam).sum())
    if s2 <= 0.0:
        return 0.0
    return (s1 * s1) / s2


def mean_pairwise_cosine(Z) -> float:
    """Mean over ordered pairs (i != j) of cosine similarity between rows
    (see module doc). Zero-norm rows are dropped (cosine undefined)."""
    Z = _as_2d(Z)
    norms = np.linalg.norm(Z, axis=1)
    keep = norms > 0.0
    if not np.all(keep):
        Z = Z[keep]
        norms = norms[keep]
    N = Z.shape[0]
    if N < 2:
        return float("nan")
    hat = Z / norms[:, None]
    s = hat.sum(axis=0)                           # (E,)
    total = float(s @ s)                          # sum_{i,j} hat_i . hat_j
    return (total - N) / (N * (N - 1))


def embedding_health(Z) -> dict:
    """VICReg-style collapse diagnostics for an embedding matrix Z (N, E).
    Returns min_std, mean_std, eff_rank, mean_pairwise_cos, plus n and dim.
    Monitor-only; never fed into the loss."""
    Z = _as_2d(Z)
    N, E = Z.shape
    if N < 2:
        warnings.warn(
            "embedding_health needs N >= 2 rows for variance; got N=%d." % N,
            RuntimeWarning)
        nan = float("nan")
        return {"min_std": nan, "mean_std": nan, "eff_rank": nan,
                "mean_pairwise_cos": nan, "n": int(N), "dim": int(E)}
    sigma = Z.std(axis=0, ddof=1)                 # (E,)
    return {
        "min_std": float(sigma.min()),
        "mean_std": float(sigma.mean()),
        "eff_rank": float(effective_rank(Z)),
        "mean_pairwise_cos": float(mean_pairwise_cosine(Z)),
        "n": int(N),
        "dim": int(E),
    }


# --------------------------------------------------------------------------- #
# clustering metrics (the model-selection objective)
# --------------------------------------------------------------------------- #
def clustering_metrics(Z, y, seed: int = 0, n_clusters=None, n_init: int = 10,
                       silhouette_metric: str = "cosine") -> dict:
    """Cluster-recovery quality of embeddings Z against true phenotype labels y.

    Parameters
    ----------
    Z                 : (N, E) embeddings (array-like or torch tensor)
    y                 : (N,) true phenotype labels in {0..C-1}
    seed              : random_state for the single K-means fit (reproducible)
    n_clusters        : K for K-means; default = C = number of distinct labels
    n_init            : K-means restarts (passed explicitly to avoid version drift)
    silhouette_metric : distance for silhouette (default "cosine")

    Returns
    -------
    dict with:
        ari         : adjusted Rand index    (nan if fewer than 2 label classes)
        ami         : adjusted mutual info    (nan if fewer than 2 label classes)
        silhouette  : mean silhouette vs TRUE y (nan if undefined / degenerate)
        labels_pred : (N,) int K-means labels on full-D Z (for consistent plots)
        n_clusters  : the K actually used

    The single seeded K-means here is the ONLY clustering used; the evaluator
    reuses labels_pred so the reported metric and the coloured plot never diverge.
    """
    Z = _as_2d(Z)
    y = _to_numpy(y).astype(np.int64).ravel()
    N = Z.shape[0]
    if y.shape[0] != N:
        raise ValueError("Z has %d rows but y has %d labels" % (N, y.shape[0]))

    classes = np.unique(y)
    C = int(classes.shape[0])
    K = int(n_clusters) if n_clusters is not None else C
    if K < 1:
        raise ValueError("n_clusters must be >= 1")
    if K > N:
        raise ValueError("n_clusters=%d exceeds N=%d samples" % (K, N))

    # single seeded K-means on the full-dimensional embeddings
    km = KMeans(n_clusters=K, random_state=int(seed), n_init=int(n_init))
    labels_pred = km.fit_predict(Z).astype(np.int64)

    nan = float("nan")
    if C < 2:
        warnings.warn(
            "clustering_metrics: y has %d distinct label(s); ARI/AMI/silhouette "
            "are undefined with fewer than 2 phenotype classes -> returning nan."
            % C, RuntimeWarning)
        return {"ari": nan, "ami": nan, "silhouette": nan,
                "labels_pred": labels_pred, "n_clusters": K}

    ari = float(adjusted_rand_score(y, labels_pred))
    ami = float(adjusted_mutual_info_score(y, labels_pred))

    # silhouette vs TRUE labels; degrade gracefully on degenerate inputs
    try:
        sil = float(silhouette_score(Z, y, metric=silhouette_metric))
    except ValueError as ex:
        warnings.warn(
            "silhouette_score undefined for this input (%s) -> nan." % ex,
            RuntimeWarning)
        sil = nan

    return {"ari": ari, "ami": ami, "silhouette": sil,
            "labels_pred": labels_pred, "n_clusters": K}
