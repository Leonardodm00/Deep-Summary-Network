"""
dsn_embed_figures.py -- PLOTTING LAYER ONLY for trace embeddings.

Draws. Does not load checkpoints, does not embed, does not cluster, does not
save. Every function takes EmbeddingSet objects from dsn_embed and returns a
matplotlib Figure.

THE INVARIANT THIS MODULE MUST NOT BREAK (evaluate.py states it, and the
repository has a smoke check that proves it from the rendered artefact):
    the clusters SCORED and the clusters DRAWN are the same fit.
So labels_pred is passed IN and never recomputed here. No function in this file
calls KMeans, and E1 double-encodes every point -- colour = predicted cluster,
marker = true phenotype -- so a mis-clustered window is visible as a marker that
disagrees with its neighbours. PCA is a DISPLAY projection only; no metric is
ever computed in the projected space.

Figures
    E1 fig_embedding_pca ......... the classic view, one panel per split
    E2 fig_split_comparison ...... train/val/test side by side + metric bars,
                                   i.e. the generalisation gap
    E3 fig_per_class_silhouette .. per-sample silhouette by TRUE class, showing
                                   WHICH phenotype is poorly separated
    E4 fig_similarity_heatmap .... cosine similarity, rows ordered by true label;
                                   block structure is separation seen directly
    E5 fig_culture_structure ..... coloured by culture, to expose whether windows
                                   cluster by culture rather than by phenotype
    E6 fig_embedding_health ...... per-dimension std and effective rank

Pure ASCII, headless (hpc-python-compat).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")                      # MUST precede pyplot

import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402
from matplotlib.lines import Line2D        # noqa: E402

__all__ = ["fig_embedding_pca", "fig_split_comparison",
           "fig_per_class_silhouette", "fig_similarity_heatmap",
           "fig_culture_structure", "fig_embedding_health",
           "CLUSTER_COLORS", "CLASS_MARKERS"]

# Colour is keyed on the cluster ID itself, never on its position among the IDs
# present. If K-means leaves a cluster empty, a positional map would give the
# same cluster different colours in two figures and silently break visual
# comparability between splits. evaluate.py's smoke check [L] guards this.
CLUSTER_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
                  "#8c564b", "#e377c2", "#17becf"]
CLASS_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _style(ax):
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def _pca(Z, n_components=2):
    """Display projection only. Returns (scores, explained_variance_ratio)."""
    from sklearn.decomposition import PCA
    p = PCA(n_components=int(n_components))
    return p.fit_transform(np.asarray(Z, dtype=np.float64)), \
        p.explained_variance_ratio_


def _double_encoded_scatter(ax, P, y, labels_pred, s=34, alpha=0.85):
    """colour = predicted cluster, marker = true phenotype."""
    for c in np.unique(labels_pred):
        for k in np.unique(y):
            m = (labels_pred == c) & (y == k)
            if not m.any():
                continue
            ax.scatter(P[m, 0], P[m, 1],
                       c=CLUSTER_COLORS[int(c) % len(CLUSTER_COLORS)],
                       marker=CLASS_MARKERS[int(k) % len(CLASS_MARKERS)],
                       s=s, alpha=alpha, linewidths=0.4, edgecolors="white")
    return ax


def _encoding_legend(fig, y, labels_pred, ncol=2):
    h = [Line2D([], [], color=CLUSTER_COLORS[int(c) % len(CLUSTER_COLORS)],
                marker="o", linestyle="", markersize=7,
                label="cluster %d (colour)" % int(c))
         for c in np.unique(labels_pred)]
    h += [Line2D([], [], color="0.35", marker=CLASS_MARKERS[int(k) % len(CLASS_MARKERS)],
                 linestyle="", markersize=7, label="phenotype %d (marker)" % int(k))
          for k in np.unique(y)]
    fig.legend(handles=h, loc="lower center", ncol=ncol, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))


# ------------------------------------------------------------------ E1 ------ #
def fig_embedding_pca(embsets, figsize=None):
    """E1. PCA scatter per split, double-encoded.

    embsets : dict {split_name: EmbeddingSet}, drawn in the given order.
    """
    names = list(embsets.keys())
    n = len(names)
    figsize = figsize or (5.0 * n, 5.4)
    fig, axarr = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axarr = axarr.ravel()
    for ax, name in zip(axarr, names):
        e = embsets[name]
        P, evr = _pca(e.Z, 2)
        _double_encoded_scatter(ax, P, e.y, e.labels_pred)
        ax.set_xlabel("PC1 (%.1f%% var)" % (100 * evr[0]), fontsize=9)
        ax.set_ylabel("PC2 (%.1f%% var)" % (100 * evr[1]), fontsize=9)
        ax.set_title("%s   N=%d\nARI %.3f | AMI %.3f | sil %.3f"
                     % (name, e.n, e.metrics.get("ari", float("nan")),
                        e.metrics.get("ami", float("nan")),
                        e.metrics.get("silhouette", float("nan"))), fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        _style(ax)
    e0 = embsets[names[0]]
    _encoding_legend(fig, e0.y, e0.labels_pred, ncol=min(6, len(names) * 3))
    fig.suptitle("E1  Embedding (PCA display projection only; metrics computed "
                 "in full %d-D space)   %s" % (e0.dim, e0.tag), fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    return fig


# ------------------------------------------------------------------ E2 ------ #
def fig_split_comparison(embsets, figsize=(11, 4.6)):
    """E2. The generalisation gap: the three metrics across splits.

    train >> val ~ test means the embedding memorised the training cultures.
    Since the split is WHOLE-CULTURE, a gap here is a gap across cultures, which
    is the quantity that actually matters for a phenotype claim.
    """
    names = list(embsets.keys())
    keys = ("ari", "ami", "silhouette")
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize,
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.arange(len(keys))
    w = 0.8 / max(1, len(names))
    for i, nm in enumerate(names):
        vals = [float(embsets[nm].metrics.get(k, np.nan)) for k in keys]
        ax0.bar(x + i * w - 0.4 + w / 2, vals, width=w, label="%s (N=%d)"
                % (nm, embsets[nm].n), edgecolor="k", linewidth=0.5)
        for xx, v in zip(x + i * w - 0.4 + w / 2, vals):
            if np.isfinite(v):
                ax0.text(xx, v, "%.3f" % v, ha="center",
                         va="bottom" if v >= 0 else "top", fontsize=7)
    ax0.set_xticks(x)
    ax0.set_xticklabels(["ARI", "AMI", "silhouette"])
    ax0.axhline(0.0, color="k", linewidth=0.8)
    ax0.set_ylabel("score")
    ax0.set_title("E2a  metrics by split (whole-culture split:\na gap here is a "
                  "gap ACROSS cultures)", fontsize=9)
    ax0.legend(fontsize=8, frameon=False)
    _style(ax0)

    for nm in names:
        e = embsets[nm]
        er = float(e.health.get("eff_rank", np.nan))
        ax1.bar(nm, er, edgecolor="k", linewidth=0.5, color="#9ecae1")
        ax1.text(nm, er, "%.2f" % er, ha="center", va="bottom", fontsize=8)
    C = int(np.unique(embsets[names[0]].y).size)
    ax1.axhline(C - 1, color="#d62728", linestyle="--", linewidth=1.4,
                label="C - 1 = %d (C well-separated\nclusters span a %d-D affine\n"
                      "subspace)" % (C - 1, C - 1))
    ax1.set_ylabel("effective rank of Z")
    ax1.set_title("E2b  collapse tripwire\n(eff_rank near 1 = collapsed)",
                  fontsize=9)
    ax1.legend(fontsize=7, frameon=False)
    _style(ax1)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ E3 ------ #
def fig_per_class_silhouette(embsets, metric="cosine", figsize=None):
    """E3. Per-sample silhouette against the TRUE labels, grouped by class.

    The reported silhouette is a mean over all windows, so one badly separated
    phenotype can hide behind two good ones. This panel is where that shows.
    Uses sklearn.metrics.silhouette_samples (directive 1: the tested one).
    """
    from sklearn.metrics import silhouette_samples
    names = list(embsets.keys())
    n = len(names)
    figsize = figsize or (4.6 * n, 4.4)
    fig, axarr = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axarr = axarr.ravel()
    for ax, nm in zip(axarr, names):
        e = embsets[nm]
        classes = np.unique(e.y)
        if classes.size < 2:
            ax.set_visible(False)
            continue
        s = silhouette_samples(np.asarray(e.Z, dtype=np.float64), e.y,
                               metric=metric)
        data = [s[e.y == k] for k in classes]
        pos = np.arange(classes.size)
        bp = ax.boxplot(data, positions=pos, widths=0.6, patch_artist=True,
                        showfliers=False,
                        medianprops=dict(color="k", linewidth=1.4))
        for patch, k in zip(bp["boxes"], classes):
            patch.set_facecolor(CLUSTER_COLORS[int(k) % len(CLUSTER_COLORS)])
            patch.set_alpha(0.55)
        for p, d in zip(pos, data):
            ax.scatter(np.full(d.shape, p) + np.random.uniform(-0.12, 0.12, d.size),
                       d, s=6, color="k", alpha=0.35, linewidths=0)
        ax.axhline(0.0, color="#d62728", linestyle="--", linewidth=1.2)
        ax.axhline(float(e.metrics.get("silhouette", np.nan)), color="k",
                   linestyle=":", linewidth=1.2,
                   label="reported mean = %.3f"
                         % e.metrics.get("silhouette", np.nan))
        ax.set_xticks(pos)
        ax.set_xticklabels(["phenotype %d\n(n=%d)" % (k, int((e.y == k).sum()))
                            for k in classes], fontsize=8)
        ax.set_ylabel("per-window silhouette (%s)" % metric, fontsize=8)
        ax.set_title("%s" % nm, fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        _style(ax)
    fig.suptitle("E3  Per-window silhouette by TRUE phenotype: which class is "
                 "actually poorly separated", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


# ------------------------------------------------------------------ E4 ------ #
def fig_similarity_heatmap(embsets, figsize=None):
    """E4. Cosine similarity matrix, rows/cols ordered by TRUE label.

    A separated embedding shows C bright diagonal blocks on a dark background.
    This is the separation the silhouette summarises, shown without any
    projection or clustering in between.
    """
    names = list(embsets.keys())
    n = len(names)
    figsize = figsize or (4.8 * n, 4.6)
    fig, axarr = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axarr = axarr.ravel()
    for ax, nm in zip(axarr, names):
        e = embsets[nm]
        order = np.argsort(e.y, kind="stable")
        Zn = np.asarray(e.Z, dtype=np.float64)[order]
        norm = np.linalg.norm(Zn, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        S = (Zn / norm) @ (Zn / norm).T
        im = ax.imshow(S, cmap="magma", vmin=-1, vmax=1, interpolation="nearest")
        ys = e.y[order]
        bounds = np.flatnonzero(np.diff(ys)) + 1
        for b in bounds:
            ax.axhline(b - 0.5, color="#00e5ff", linewidth=1.0)
            ax.axvline(b - 0.5, color="#00e5ff", linewidth=1.0)
        centres, start = [], 0
        for b in list(bounds) + [len(ys)]:
            centres.append((start + b) / 2.0)
            start = b
        ax.set_xticks(centres)
        ax.set_xticklabels(["ph %d" % k for k in np.unique(ys)], fontsize=8)
        ax.set_yticks(centres)
        ax.set_yticklabels(["ph %d" % k for k in np.unique(ys)], fontsize=8)
        ax.set_title("%s  (N=%d, sil %.3f)"
                     % (nm, e.n, e.metrics.get("silhouette", np.nan)), fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8, label="cosine similarity")
    fig.suptitle("E4  Pairwise cosine similarity, ordered by TRUE phenotype "
                 "(cyan lines = class boundaries)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


# ------------------------------------------------------------------ E5 ------ #
def fig_culture_structure(embsets, figsize=None):
    """E5. The same PCA, coloured by CULTURE instead of by cluster.

    This is the panel that matters most for a phenotype claim on MEA data. The
    split is whole-culture and positives are drawn cross-culture, precisely so
    the embedding cannot solve the task by identifying the culture. If the
    points form one tight blob per culture rather than one per phenotype, the
    model learned culture identity and the ARI is not measuring phenotype.

    Returns None when no per-window culture map is available -- it is never
    fabricated.
    """
    usable = [(nm, e) for nm, e in embsets.items() if e.cultures is not None]
    if not usable:
        return None
    n = len(usable)
    figsize = figsize or (5.0 * n, 5.0)
    fig, axarr = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axarr = axarr.ravel()
    cmap = plt.get_cmap("tab20")
    for ax, (nm, e) in zip(axarr, usable):
        P, evr = _pca(e.Z, 2)
        cult = np.asarray(e.cultures).ravel()
        uniq = list(dict.fromkeys(cult.tolist()))
        for i, c in enumerate(uniq):
            m = cult == c
            k_here = np.unique(e.y[m])
            mk = CLASS_MARKERS[int(k_here[0]) % len(CLASS_MARKERS)] \
                if k_here.size == 1 else "o"
            ax.scatter(P[m, 0], P[m, 1], color=cmap(i % 20), marker=mk, s=34,
                       alpha=0.85, linewidths=0.4, edgecolors="white",
                       label=str(c))
        ax.set_xlabel("PC1 (%.1f%% var)" % (100 * evr[0]), fontsize=9)
        ax.set_ylabel("PC2 (%.1f%% var)" % (100 * evr[1]), fontsize=9)
        ax.set_title("%s   %d culture(s)\ncolour = culture, marker = phenotype"
                     % (nm, len(uniq)), fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        if len(uniq) <= 12:
            ax.legend(fontsize=6, frameon=False, ncol=2, title="culture",
                      title_fontsize=7)
        _style(ax)
    fig.suptitle("E5  Culture structure: tight per-culture blobs would mean the "
                 "model learned CULTURE, not phenotype", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


# ------------------------------------------------------------------ E6 ------ #
def fig_embedding_health(embsets, figsize=(11, 4.4)):
    """E6. Per-dimension standard deviation and the health scalars."""
    names = list(embsets.keys())
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)
    for nm in names:
        e = embsets[nm]
        sd = np.sort(np.asarray(e.Z, dtype=np.float64).std(axis=0, ddof=1))[::-1]
        ax0.plot(np.arange(1, sd.size + 1), sd, marker="o", markersize=3,
                 linewidth=1.3, label="%s (min %.4f)" % (nm, sd.min()))
    ax0.set_xlabel("embedding dimension, sorted by std")
    ax0.set_ylabel("std across windows")
    ax0.set_yscale("log")
    ax0.set_title("E6a  per-dimension spread\n(a floor near zero = dead "
                  "dimensions)", fontsize=9)
    ax0.legend(fontsize=8, frameon=False)
    _style(ax0)

    keys = ("eff_rank", "mean_std", "min_std")
    x = np.arange(len(keys))
    w = 0.8 / max(1, len(names))
    for i, nm in enumerate(names):
        v = [float(embsets[nm].health.get(k, np.nan)) for k in keys]
        ax1.bar(x + i * w - 0.4 + w / 2, v, width=w, label=nm,
                edgecolor="k", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(keys)
    ax1.set_yscale("log")
    ax1.set_title("E6b  health scalars (monitor-only:\nnever added to the loss)",
                  fontsize=9)
    ax1.legend(fontsize=8, frameon=False)
    _style(ax1)
    fig.tight_layout()
    return fig
