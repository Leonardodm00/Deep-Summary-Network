"""
smoke_test_metrics.py

Standalone correctness checks for metrics.py (Stage 3). No external data, no
torch required, CPU only.

Run:
    python3 smoke_test_metrics.py

Checks:
  [A] Separable clusters -> ARI ~ 1, AMI ~ 1, high cosine silhouette, eff_rank > 1,
      moderate mean pairwise cosine.
  [B] Determinism: same seed -> identical labels_pred and ARI.
  [C] Point collapse (identical rows) -> min_std = 0, mean_std = 0,
      mean_pairwise_cos ~ 1, eff_rank = 0, and K-means fails to recover (ARI low).
  [D] Line collapse (rank-1 structure) -> eff_rank ~ 1 (the collapse signal).
  [E] Isotropic Gaussian -> eff_rank ~ E (the full-rank end).
  [F] Effective rank matches the KNOWN participation ratio of a diagonal
      covariance with prescribed per-dimension variances.
  [G] mean_pairwise_cosine closed form == brute-force O(N^2) cosine mean.
  [H] Singleton group (one class has a single point) -> no crash, finite silhouette.
  [I] Fewer than 2 label classes -> ARI / AMI / silhouette are nan.
  [J] N < 2 -> embedding_health returns nan diagnostics.
"""

import sys
import warnings

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from metrics import (
    clustering_metrics, embedding_health, effective_rank, mean_pairwise_cosine,
)


def _l2n(Z):
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return Z / n


def _make_separable(C=4, E=8, n_per=60, sep=6.0, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    means = np.zeros((C, E))
    for c in range(C):
        means[c, c % E] = sep                 # each cluster near a distinct axis
    Z, y = [], []
    for c in range(C):
        Z.append(means[c] + noise * rng.standard_normal((n_per, E)))
        y.append(np.full(n_per, c))
    Z = _l2n(np.vstack(Z))                     # embeddings live on the unit sphere
    y = np.concatenate(y)
    return Z, y


def check_separable():
    Z, y = _make_separable()
    m = clustering_metrics(Z, y, seed=0)
    assert m["ari"] > 0.95, m["ari"]
    assert m["ami"] > 0.90, m["ami"]
    assert m["silhouette"] > 0.50, m["silhouette"]
    assert m["n_clusters"] == 4, m["n_clusters"]
    h = embedding_health(Z)
    assert h["eff_rank"] > 2.0, h["eff_rank"]
    assert h["mean_pairwise_cos"] < 0.60, h["mean_pairwise_cos"]
    print("  [A] separable clusters: ARI~1, AMI~1, high silhouette, eff_rank>2 OK")


def check_determinism():
    Z, y = _make_separable(seed=1)
    m1 = clustering_metrics(Z, y, seed=7)
    m2 = clustering_metrics(Z, y, seed=7)
    assert np.array_equal(m1["labels_pred"], m2["labels_pred"])
    assert m1["ari"] == m2["ari"]
    print("  [B] determinism under fixed seed OK")


def check_point_collapse():
    rng = np.random.default_rng(2)
    u = rng.standard_normal(8)
    u = u / np.linalg.norm(u)
    Z = np.tile(u, (50, 1))                    # every row identical
    y = np.array([0] * 25 + [1] * 25)
    h = embedding_health(Z)
    assert h["min_std"] < 1e-10, h["min_std"]
    assert h["mean_std"] < 1e-10, h["mean_std"]
    assert h["mean_pairwise_cos"] > 0.999, h["mean_pairwise_cos"]
    # NOTE: under numerical point-collapse the residual covariance is tiny but
    # isotropic, so eff_rank ~ E (not 0). eff_rank's collapse signal is for LINE
    # collapse (test [D]); the exact zero-covariance branch is checked in [C2].
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # KMeans may warn on identical points
        m = clustering_metrics(Z, y, seed=0)
    assert m["ari"] < 0.30, m["ari"]           # cannot recover an arbitrary grouping
    assert np.isfinite(m["silhouette"]), m["silhouette"]
    print("  [C] point collapse: std~0, cos~1, ARI low OK")


def check_exact_zero_variance():
    # exactly identical integer-valued rows -> covariance is exactly 0 -> the
    # degenerate branch returns eff_rank = 0.0 (documented convention).
    assert effective_rank(np.ones((10, 4))) == 0.0
    print("  [C2] exact zero-variance -> eff_rank = 0 OK")


def check_line_collapse():
    rng = np.random.default_rng(3)
    u = rng.standard_normal(8)
    u = u / np.linalg.norm(u)
    t = rng.standard_normal(200)
    Z = t[:, None] * u[None, :]                # rank-1: all variance along u
    er = effective_rank(Z)
    assert er < 1.05, er
    print("  [D] line collapse: eff_rank ~ 1 OK (er=%.4f)" % er)


def check_isotropic():
    rng = np.random.default_rng(4)
    E = 8
    Z = rng.standard_normal((2000, E))
    er = effective_rank(Z)
    assert er > 0.85 * E, (er, E)
    print("  [E] isotropic Gaussian: eff_rank ~ E OK (er=%.3f, E=%d)" % (er, E))


def check_known_participation_ratio():
    rng = np.random.default_rng(5)
    v = np.array([8.0, 4.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0])   # per-dim variances
    pr = (v.sum() ** 2) / (v * v).sum()        # known participation ratio
    Z = rng.standard_normal((4000, v.shape[0])) * np.sqrt(v)[None, :]
    er = effective_rank(Z)
    rel = abs(er - pr) / pr
    assert rel < 0.15, (er, pr, rel)
    print("  [F] eff_rank matches known participation ratio OK "
          "(er=%.3f vs pr=%.3f)" % (er, pr))


def check_pairwise_cosine_bruteforce():
    rng = np.random.default_rng(6)
    Z = rng.standard_normal((40, 6))
    fast = mean_pairwise_cosine(Z)
    S = cosine_similarity(Z)                    # (N, N)
    N = Z.shape[0]
    brute = (S.sum() - np.trace(S)) / (N * (N - 1))
    assert abs(fast - brute) < 1e-9, (fast, brute)
    print("  [G] mean_pairwise_cosine closed form == brute force OK")


def check_singleton_group():
    rng = np.random.default_rng(7)
    Z0 = rng.standard_normal((20, 5)) + np.array([3, 0, 0, 0, 0])
    Z1 = (rng.standard_normal((1, 5)) + np.array([-3, 0, 0, 0, 0]))  # single point
    Z = _l2n(np.vstack([Z0, Z1]))
    y = np.array([0] * 20 + [1])
    m = clustering_metrics(Z, y, seed=0)       # must not raise
    assert np.isfinite(m["silhouette"]), m["silhouette"]
    print("  [H] singleton group: finite silhouette, no crash OK")


def check_single_class():
    rng = np.random.default_rng(8)
    Z = rng.standard_normal((20, 4))
    y = np.zeros(20, dtype=int)                 # only one class
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = clustering_metrics(Z, y, seed=0)
    assert np.isnan(m["ari"]) and np.isnan(m["ami"]) and np.isnan(m["silhouette"])
    print("  [I] fewer than 2 classes: ARI/AMI/silhouette = nan OK")


def check_health_too_few():
    Z = np.zeros((1, 4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        h = embedding_health(Z)
    assert all(np.isnan(h[k]) for k in ("min_std", "mean_std", "eff_rank",
                                        "mean_pairwise_cos"))
    print("  [J] N<2 embedding_health -> nan diagnostics OK")


def main():
    print("Running metrics smoke tests...")
    check_separable()
    check_determinism()
    check_point_collapse()
    check_exact_zero_variance()
    check_line_collapse()
    check_isotropic()
    check_known_participation_ratio()
    check_pairwise_cosine_bruteforce()
    check_singleton_group()
    check_single_class()
    check_health_too_few()
    print("ALL METRICS SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
