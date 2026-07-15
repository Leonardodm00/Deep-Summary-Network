"""
evaluate.py
===========

Evaluation + the embedding figure. Separation of concerns (directive 2): this
module SCORES and PLOTS ONLY. It never trains, never optimizes, never loads raw
data, never builds a model. It calls the already-tested modules:

    inference.embed_clean_windows  -- the DISTINCT-window embedding (shared with
                                      the trainer, so eval and validation embed
                                      identically; it is not re-implemented here)
    metrics.clustering_metrics     -- the single seeded full-D K-means + ARI / AMI
                                      / silhouette
    metrics.embedding_health       -- the monitor-only collapse diagnostics

Headless by construction: matplotlib's Agg backend is selected BEFORE pyplot is
imported, so this module is importable and usable on a compute node with no
display. There is no plt.show() anywhere, and there is no Visible flag.

The four legacy bugs this module removes (read from the legacy source, not from
memory -- 1D_CNN_functions.Embedding_Scores, approx. lines 3156-3239)
--------------------------------------------------------------------------------
  (1) METRIC / PLOT MISMATCH. The legacy metric fitted a SEEDED K-means
      (random_state=0) on the FULL-DIMENSIONAL embeddings, but the figure then
      fitted a SECOND, UNSEEDED K-means (n_init=4, no random_state) on the
      PCA(2)-REDUCED data. The clusters that were SCORED and the clusters that
      were DRAWN were therefore two different fits, in two different spaces, and
      could disagree arbitrarily. Here there is EXACTLY ONE clustering: the
      labels_pred returned by metrics.clustering_metrics are the SAME array that
      colours the scatter. PCA is used for DISPLAY ONLY -- it never re-clusters.

  (2) Visible=False -> NameError. The legacy function assigned reduced_data only
      inside `if Visible == True:` but returned it unconditionally, so every
      headless call crashed. The Visible flag is REMOVED entirely; the figure is
      always written with savefig, never shown.

  (3) The legacy figure's scatter of the actual points was COMMENTED OUT: it drew
      undifferentiated black dots ("k.") over a decision-boundary mesh, carrying
      NO label information at all. Here every point is drawn and doubly encoded
      (see plot_embedding).

  (4) The decision-boundary MESH is DROPPED. It was computed from the PCA-space
      K-means, i.e. from the wrong fit; and a Voronoi mesh of a 2-D projection
      says nothing about the decision structure of the full-dimensional embedding
      it is meant to illustrate.

Notation (symbols introduced at first use; carried in full)
-----------------------------------------------------------
    N            : number of DISTINCT windows in the evaluated dataset,
                   N = len(dataset.index)
    E            : embedding dimension, E = cfg.backbone.embedding_size
    C            : number of phenotype classes, labels in {0, ..., C-1}
    Z            : (N, E) float32 embedding matrix, rows z_i = f_theta(x_i) for the
                   i-th CLEAN window x_i; each row is L2-normalized by the backbone
                   head (||z_i||_2 = 1 when cfg.backbone.l2_normalize, the default),
                   so the z_i live on the unit hypersphere S^{E-1}
    y            : (N,) int64 TRUE phenotype labels, y_i in {0, ..., C-1}
    K            : number of K-means clusters. K = C ALWAYS (locked invariant), and
                   C is taken from the FULL label set supplied by the caller, NOT
                   from the labels that happen to appear in the evaluated split
                   (see n_clusters below)
    labels_pred  : (N,) int64 K-means assignments, labels_pred_i in {0, ..., K-1},
                   from the ONE seeded K-means fitted on the FULL-dimensional Z.
                   These are the labels that BOTH define the reported ARI / AMI AND
                   colour the figure.
    P            : PCA display dimension, P = eval_cfg.pca_components (default 2).
                   The PCA map is a linear projection R^E -> R^P used for DRAWING
                   ONLY; no metric is ever computed in the projected space.
    Z_pca        : (N, P) float64, Z_pca = PCA(P).fit_transform(Z)

Scoring
-------
    ari         = adjusted_rand_score(y, labels_pred)             in [-0.5, 1]
    ami         = adjusted_mutual_info_score(y, labels_pred)      in [~0, 1]
    silhouette  = mean silhouette of Z against the TRUE labels y under the cosine
                  distance d_cos(u, v) = 1 - (u . v) / (||u||_2 ||v||_2). It is
                  computed against y (not against labels_pred), so it is a
                  K-means-INDEPENDENT companion to ARI / AMI, and cosine matches
                  the unit-sphere geometry the network was trained in.
    All three come from metrics.clustering_metrics; this module adds no scoring
    mathematics of its own (directive 1: reuse the tested implementation).

HPC note (hpc-python-compat): pure ASCII. The Agg backend is forced before
pyplot is imported, so importing this module cannot try to open a display.
"""

import os

import matplotlib
matplotlib.use("Agg")                 # MUST precede the pyplot import (headless)
import matplotlib.pyplot as plt       # noqa: E402  (import order is deliberate)

import numpy as np
from sklearn.decomposition import PCA

from inference import embed_clean_windows
from metrics import clustering_metrics, embedding_health

__all__ = [
    "evaluate",
    "plot_embedding",
    "evaluate_and_plot",
]

# Marker glyphs used to encode the TRUE phenotype label. Colour encodes the
# PREDICTED cluster (labels_pred, the metric's own labels -- the consistency fix),
# so a point whose colour and shape disagree with its neighbours is a visible
# mis-clustering. Cycled if C exceeds the list length.
_TRUE_LABEL_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")


def _resolve_n_clusters(y, n_clusters):
    """K for the K-means. K = C, the number of phenotype classes (locked invariant).

    IMPORTANT (same hazard the trainer guards against): C must be the size of the
    EXPERIMENT's label set, not of the labels that happen to survive into the split
    being evaluated. data_splits WARNS but permits a phenotype having no windows in
    a split; if K were inferred from that split alone, a TEST split that lost a rare
    class would silently score a K = C - 1 partition and the reported ARI / AMI
    would not be comparable to the validation numbers the model was selected on.
    The caller therefore passes n_clusters explicitly (Stage 8 passes C from the
    full dataset). When it is None we fall back to the labels present, which is
    correct only when every class IS present -- so we return the fallback and let
    the caller decide.
    """
    if n_clusters is not None:
        K = int(n_clusters)
        if K < 1:
            raise ValueError("n_clusters must be >= 1")
        return K
    return int(np.unique(np.asarray(y).ravel()).shape[0])


def evaluate(model, dataset, device, seed=0, n_clusters=None, eval_cfg=None):
    """Score ONE model on ONE dataset split (single model, single K-means seed).

    Scope, deliberately narrow (mirrors train()'s single-seed scope): this function
    evaluates ONE trained model. The "+/- seed std" reported downstream comes from
    TRAINING variability -- Stage 8 trains n_seeds models and calls evaluate() on
    each, then aggregates -- NOT from jittering the K-means seed of a single model.
    Averaging here would hide exactly the variability the search is meant to expose.

    Parameters
    ----------
    model      : trained backbone (forward: (M, T) -> (M, E), rows L2-normalized)
    dataset    : MEAWindowDataset for the split being evaluated. For the final
                 report this MUST be the held-out TEST split (SplitBundle.test),
                 whose windows are DISJOINT from train and val by construction
                 (time-segment splitting), so there is no leakage.
    device     : torch device or "cpu" / "cuda" / "auto"
    seed       : random_state of the ONE K-means fit (reproducible)
    n_clusters : K. Pass C from the FULL label set (see _resolve_n_clusters).
                 None -> inferred from the labels present in `dataset`.
    eval_cfg   : EvalConfig (kmeans_seed / kmeans_n_init / silhouette_metric /
                 pca_components). When given, its kmeans_n_init and
                 silhouette_metric are used; `seed` still wins over kmeans_seed so
                 a caller can sweep the K-means seed explicitly if it wants to.

    Returns
    -------
    results : dict with
        ari, ami, silhouette : the scores (float; nan when undefined)
        labels_pred          : (N,) int64 -- THE labels behind those scores. The
                               plot MUST be coloured with this exact array; that
                               identity is what makes metric and figure consistent.
        n_clusters           : the K actually used
        Z                    : (N, E) float32 embeddings (distinct windows)
        y                    : (N,) int64 true labels
        health               : the embedding_health diagnostics (monitor-only)
        n_windows            : N
    """
    n_init = int(eval_cfg.kmeans_n_init) if eval_cfg is not None else 10
    sil_metric = (eval_cfg.silhouette_metric
                  if eval_cfg is not None else "cosine")

    # DISTINCT clean windows -- the shared helper, not a re-implementation. This is
    # the leakage / duplication fix: no resample-with-replacement, one row per window.
    Z, y = embed_clean_windows(model, dataset, device)

    K = _resolve_n_clusters(y, n_clusters)

    m = clustering_metrics(Z, y, seed=int(seed), n_clusters=K, n_init=n_init,
                           silhouette_metric=sil_metric)
    health = embedding_health(Z)

    return {
        "ari": float(m["ari"]),
        "ami": float(m["ami"]),
        "silhouette": float(m["silhouette"]),
        "labels_pred": np.asarray(m["labels_pred"], dtype=np.int64),
        "n_clusters": int(m["n_clusters"]),
        "Z": Z,
        "y": np.asarray(y, dtype=np.int64),
        "health": {k: float(v) for k, v in health.items()},
        "n_windows": int(Z.shape[0]),
    }


def plot_embedding(Z, y, labels_pred, out_path, pca_components=2, title=None,
                   dpi=150):
    """Save a PCA scatter of the embeddings, COLOURED BY labels_pred.

    The consistency fix (legacy bug 1): labels_pred is passed IN -- it is the same
    array metrics.clustering_metrics produced for the reported ARI / AMI. This
    function does NOT cluster. PCA here is a DISPLAY projection only; no metric is
    computed in the projected space, and no second K-means is ever fitted.

    Double encoding (fixes legacy bug 3, which drew label-free black dots):
        colour  = labels_pred_i  (the PREDICTED cluster -- the metric's own labels)
        marker  = y_i            (the TRUE phenotype)
    A correctly clustered region is therefore uniform in BOTH colour and shape; a
    point whose shape disagrees with the shapes around its colour is a visible
    mis-clustering. This keeps the locked "colour by labels_pred" requirement
    exactly, while still letting the reader see WHERE the model is wrong -- which a
    colour-only plot cannot show.

    No decision-boundary mesh (legacy bug 4). No plt.show(); the figure is always
    written to out_path and closed (headless Agg).

    Parameters
    ----------
    Z              : (N, E) embeddings (the SAME matrix that was scored)
    y              : (N,) true labels
    labels_pred    : (N,) K-means labels from clustering_metrics -- NOT recomputed
    out_path       : file path for the figure (parent dirs are created)
    pca_components : P, the display dimension. Only the first 2 components are
                     drawn; P > 2 is accepted (and reported in the axis labels'
                     explained variance) but the scatter remains 2-D.
    title          : optional figure title
    dpi            : savefig resolution

    Returns
    -------
    out_path (str) -- the file actually written.
    """
    Z = np.ascontiguousarray(np.asarray(Z, dtype=np.float64))
    y = np.asarray(y).ravel().astype(np.int64)
    labels_pred = np.asarray(labels_pred).ravel().astype(np.int64)

    if Z.ndim != 2:
        raise ValueError("Z must be 2-D (N, E); got %r" % (Z.shape,))
    N, E = Z.shape
    if y.shape[0] != N or labels_pred.shape[0] != N:
        raise ValueError(
            "Z has %d rows but y has %d and labels_pred has %d"
            % (N, y.shape[0], labels_pred.shape[0]))

    P = int(pca_components)
    if P < 1:
        raise ValueError("pca_components must be >= 1")
    # PCA cannot extract more components than min(N, E)
    P_eff = int(min(P, N, E))
    if P_eff < 1:
        raise ValueError("cannot run PCA on Z with shape %r" % (Z.shape,))

    pca = PCA(n_components=P_eff)
    Z_pca = pca.fit_transform(Z)                       # (N, P_eff), DISPLAY ONLY
    evr = pca.explained_variance_ratio_

    # pad to 2 columns so a degenerate 1-component case still plots
    if Z_pca.shape[1] == 1:
        Z_pca = np.hstack([Z_pca, np.zeros((N, 1), dtype=Z_pca.dtype)])
        evr = np.concatenate([evr, [0.0]])

    pred_ids = np.unique(labels_pred)
    true_ids = np.unique(y)

    # Colour is keyed on the CLUSTER ID ITSELF (cmap(p % N)), NOT on the id's
    # position in the list of ids that happen to be present. This matters: K-means
    # can leave a cluster empty, so enumerating the PRESENT ids would give cluster 2
    # colour index 1 in a run where cluster 1 is empty, but colour index 2 in a run
    # where it is not -- the same cluster would render in DIFFERENT colours across
    # figures, silently destroying visual comparability between the validation and
    # test figures and across HPO trials. Keying on the id makes the colour of
    # cluster k a fixed function of k alone.
    cmap = plt.get_cmap("tab10" if int(pred_ids.max(initial=0)) < 10 else "tab20")
    colour_of = {int(p): cmap(int(p) % cmap.N) for p in pred_ids}

    # Same reasoning for the marker: keyed on the TRUE LABEL ID, not its position
    # among the labels present, so phenotype k always draws with the same glyph even
    # in a split where some other phenotype is absent.
    marker_of = {int(t): _TRUE_LABEL_MARKERS[int(t) % len(_TRUE_LABEL_MARKERS)]
                 for t in true_ids}

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    # one scatter call per (predicted cluster, true label) cell: colour = predicted,
    # marker = true. Empty cells are skipped.
    for t in true_ids:
        marker = marker_of[int(t)]
        for p in pred_ids:
            sel = (y == t) & (labels_pred == p)
            if not np.any(sel):
                continue
            ax.scatter(Z_pca[sel, 0], Z_pca[sel, 1],
                       c=[colour_of[int(p)]], marker=marker, s=42,
                       edgecolors="black", linewidths=0.4, alpha=0.85,
                       label=None, zorder=3)

    # legend 1: colour -> predicted cluster (the metric's labels)
    colour_handles = [
        plt.Line2D([], [], linestyle="none", marker="o", markersize=8,
                   markerfacecolor=colour_of[int(p)], markeredgecolor="black",
                   markeredgewidth=0.4, label="pred cluster %d" % int(p))
        for p in pred_ids
    ]
    # legend 2: marker -> true phenotype
    marker_handles = [
        plt.Line2D([], [], linestyle="none", marker=marker_of[int(t)],
                   markersize=8, markerfacecolor="white", markeredgecolor="black",
                   label="true class %d" % int(t))
        for t in true_ids
    ]
    leg1 = ax.legend(handles=colour_handles, loc="upper left", fontsize=8,
                     title="colour = K-means label\n(the metric's own labels)",
                     title_fontsize=8, framealpha=0.9)
    leg1.get_title().set_fontsize(8)
    ax.add_artist(leg1)                      # keep the first legend when adding the second
    ax.legend(handles=marker_handles, loc="lower right", fontsize=8,
              title="marker = true phenotype", title_fontsize=8, framealpha=0.9)

    ax.set_xlabel("PC 1 (%.1f%% var)" % (100.0 * float(evr[0])))
    ax.set_ylabel("PC 2 (%.1f%% var)" % (100.0 * float(evr[1])))
    if title is None:
        title = ("Embeddings, PCA display (%d distinct windows)\n"
                 "colour = the SAME seeded full-D K-means labels used for the metric"
                 % N)
    ax.set_title(title, fontsize=10)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()

    out_path = str(out_path)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi))       # ALWAYS savefig; never plt.show()
    plt.close(fig)                            # release the figure (long HPO runs)
    return out_path


def evaluate_and_plot(model, dataset, device, out_path, seed=0, n_clusters=None,
                      eval_cfg=None, title=None):
    """evaluate() then plot_embedding() with the EXACT labels_pred it returned.

    This is the convenience path Stage 8 should call, because it makes the
    metric / figure consistency structural: the figure cannot be coloured by
    anything other than the labels that produced the reported scores.

    Returns the results dict from evaluate(), with "figure" added (the path
    written, or None when eval_cfg.pca_components makes a plot impossible).
    """
    results = evaluate(model, dataset, device, seed=seed, n_clusters=n_clusters,
                       eval_cfg=eval_cfg)
    P = int(eval_cfg.pca_components) if eval_cfg is not None else 2
    fig_path = plot_embedding(
        results["Z"], results["y"], results["labels_pred"],   # <- the same array
        out_path=out_path, pca_components=P, title=title,
    )
    results["figure"] = fig_path
    return results
