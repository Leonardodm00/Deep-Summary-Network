"""
dsn_trace_figures.py -- PLOTTING LAYER ONLY for the synthetic latent traces.

Draws. Does not generate signals, does not sample latents, does not save.
Generation is delegated entirely to the repository's own
latent_burst_generator.LatentBurstProvider (directive 1: do not reimplement
what is already tested); this module only receives arrays and renders them.

Figures
    T1 fig_trace_gallery ....... one IFR trace per class, stacked, shared axes
    T2 fig_class_overlay ....... several traces per class overlaid, to show
                                 WITHIN-class variability against BETWEEN-class
    T3 fig_latent_space ........ the phi coordinates themselves: class centres,
                                 realised draws, label vs free axes
    T4 fig_window_view ......... how a trace is cut into the windows the model
                                 actually embeds
    T5 fig_summary_stats ....... simple descriptive statistics per class, to
                                 show the task is NOT solvable by one scalar

CONVENTION. A "trace" here is the smoothed population IFR (instantaneous
firing rate), R~ in R_{>=0}^K at f_s Hz -- NOT a raw voltage and NOT a spike
raster. That is the physical quantity the pipeline consumes.

Pure ASCII, headless (hpc-python-compat).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")                      # MUST precede pyplot

import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402

__all__ = ["fig_trace_gallery", "fig_class_overlay", "fig_latent_space",
           "fig_window_view", "fig_summary_stats", "CLASS_COLORS"]

CLASS_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
                "#8c564b", "#e377c2", "#17becf"]


def _style(ax):
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def _c(k):
    return CLASS_COLORS[int(k) % len(CLASS_COLORS)]


# ------------------------------------------------------------------ T1 ------ #
def fig_trace_gallery(traces, labels, fs, n_per_class=1, figsize=None):
    """T1. One (or a few) full IFR traces per class, stacked with shared axes.

    Shared x AND y is deliberate: the classes differ in amplitude as well as in
    timing, and per-panel autoscaling would hide exactly that.
    """
    classes = sorted(set(int(l) for l in labels))
    rows = len(classes) * int(n_per_class)
    figsize = figsize or (13, 1.9 * rows + 1.0)
    fig, axarr = plt.subplots(rows, 1, figsize=figsize, sharex=True, sharey=True,
                              squeeze=False)
    axarr = axarr.ravel()
    r = 0
    for k in classes:
        idx = [i for i, l in enumerate(labels) if int(l) == k][:n_per_class]
        for j, i in enumerate(idx):
            x = np.asarray(traces[i], dtype=float)
            t = np.arange(x.size) / float(fs)
            ax = axarr[r]
            ax.plot(t, x, color=_c(k), linewidth=0.7)
            ax.set_ylabel("IFR\n[spikes/s]", fontsize=8)
            ax.text(0.005, 0.92, "phenotype %d, culture %d" % (k, i),
                    transform=ax.transAxes, fontsize=8, va="top",
                    color=_c(k), fontweight="bold")
            _style(ax)
            r += 1
    axarr[-1].set_xlabel("time [s]")
    fig.suptitle("T1  Synthetic population IFR traces, one per phenotype "
                 "(shared y: amplitude differences are real)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ------------------------------------------------------------------ T2 ------ #
def fig_class_overlay(traces, labels, fs, n_per_class=5, t_max_s=None,
                      figsize=None):
    """T2. Several traces per class overlaid in one panel per class.

    This is the figure that shows the task is NON-TRIVIAL: within-class spread
    (several lines in one panel) is visibly comparable to between-class
    difference (panel to panel). If the classes were trivially separable the
    panels would look like three clean, distinct templates.
    """
    classes = sorted(set(int(l) for l in labels))
    figsize = figsize or (13, 2.3 * len(classes) + 1.0)
    fig, axarr = plt.subplots(len(classes), 1, figsize=figsize, sharex=True,
                              sharey=True, squeeze=False)
    axarr = axarr.ravel()
    for ax, k in zip(axarr, classes):
        idx = [i for i, l in enumerate(labels) if int(l) == k][:n_per_class]
        for i in idx:
            x = np.asarray(traces[i], dtype=float)
            t = np.arange(x.size) / float(fs)
            if t_max_s is not None:
                m = t <= float(t_max_s)
                t, x = t[m], x[m]
            ax.plot(t, x, color=_c(k), linewidth=0.6, alpha=0.65)
        ax.set_ylabel("IFR\n[spikes/s]", fontsize=8)
        ax.text(0.005, 0.92, "phenotype %d  (%d cultures overlaid)"
                % (k, len(idx)), transform=ax.transAxes, fontsize=8,
                va="top", color=_c(k), fontweight="bold")
        _style(ax)
    axarr[-1].set_xlabel("time [s]")
    fig.suptitle("T2  Within-phenotype variability vs between-phenotype "
                 "difference (the task is deliberately non-trivial)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# ------------------------------------------------------------------ T3 ------ #
def fig_latent_space(phis, labels, axis_names, label_axes, centers=None,
                     figsize=None):
    """T3. The latent coordinates phi themselves.

    phis        : (n_traces, n_axes) array of latent coordinates in [0,1]^n
    label_axes  : the index set S of LABEL-CARRYING axes; the rest are
                  label-irrelevant but physically real variation
    centers     : optional (C, n_axes) class-centre matrix, drawn as crosses

    Two panels: a scatter over the first two LABEL axes (where the classes
    live), and a per-axis strip plot showing that free axes carry no class
    information by construction.
    """
    phis = np.asarray(phis, dtype=float)
    labels = np.asarray(labels, dtype=int)
    S = list(label_axes)
    free = [k for k in range(phis.shape[1]) if k not in S]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize or (13, 5.2),
                                   gridspec_kw={"width_ratios": [1, 1.5]})

    a, b = (S + [0, 1])[0], (S + [0, 1])[1]
    for k in sorted(set(labels.tolist())):
        m = labels == k
        ax0.scatter(phis[m, a], phis[m, b], s=38, color=_c(k), alpha=0.8,
                    linewidths=0.4, edgecolors="white",
                    label="phenotype %d" % k)
    if centers is not None:
        cen = np.asarray(centers, dtype=float)
        for k in range(cen.shape[0]):
            ax0.scatter(cen[k, a], cen[k, b], marker="X", s=190, color=_c(k),
                        edgecolors="k", linewidths=1.2, zorder=5)
    ax0.set_xlabel("phi[%d] = %s  (LABEL axis)" % (a, axis_names[a]), fontsize=9)
    ax0.set_ylabel("phi[%d] = %s  (LABEL axis)" % (b, axis_names[b]), fontsize=9)
    ax0.set_xlim(-0.03, 1.03)
    ax0.set_ylim(-0.03, 1.03)
    ax0.set_title("T3a  class structure lives in the LABEL axes\n"
                  "X = class centre, dots = realised cultures", fontsize=9)
    ax0.legend(fontsize=8, frameon=False)
    _style(ax0)

    rng = np.random.default_rng(0)
    for k in sorted(set(labels.tolist())):
        m = labels == k
        for pos, axis_k in enumerate(range(phis.shape[1])):
            y = phis[m, axis_k]
            ax1.scatter(np.full(y.shape, pos) + rng.uniform(-0.16, 0.16, y.size),
                        y, s=12, color=_c(k), alpha=0.6, linewidths=0)
    for pos in free:
        ax1.axvspan(pos - 0.42, pos + 0.42, color="0.88", zorder=0)
    ax1.set_xticks(range(phis.shape[1]))
    ax1.set_xticklabels(["%s\n%s" % (axis_names[k], "LABEL" if k in S else "free")
                         for k in range(phis.shape[1])], fontsize=7)
    ax1.set_ylabel("phi (normalised latent coordinate)", fontsize=9)
    ax1.set_ylim(-0.03, 1.03)
    ax1.set_title("T3b  per-axis draws; grey = label-IRRELEVANT axes\n"
                  "(colours separate on LABEL axes only, by construction)",
                  fontsize=9)
    _style(ax1)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ T4 ------ #
def fig_window_view(trace, label, fs, window_s, stride_s, culture_id=0,
                    figsize=(13, 4.4)):
    """T4. How ONE trace is cut into the windows the model actually embeds.

    The distinction matters for the whole-culture split: the model never sees a
    trace, it sees windows; but the SPLIT is done at the culture level, so all
    windows drawn here belong to the same split.
    """
    x = np.asarray(trace, dtype=float)
    t = np.arange(x.size) / float(fs)
    w = int(round(float(window_s) * float(fs)))
    s = int(round(float(stride_s) * float(fs)))
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t, x, color=_c(label), linewidth=0.7, zorder=3)
    starts = list(range(0, max(1, x.size - w + 1), max(1, s)))
    for n, st in enumerate(starts):
        ax.axvspan(st / fs, (st + w) / fs, color="0.85" if n % 2 else "0.93",
                   zorder=0)
        ax.text((st + w / 2.0) / fs, ax.get_ylim()[1], "w%d" % n, ha="center",
                va="top", fontsize=7, color="0.35")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("IFR [spikes/s]")
    ax.set_title("T4  culture %d (phenotype %d) cut into %d window(s) of %.0f s "
                 "(stride %.0f s)\nthe model embeds WINDOWS; the split is by "
                 "CULTURE" % (culture_id, label, len(starts), window_s, stride_s),
                 fontsize=9)
    _style(ax)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ T5 ------ #
def fig_summary_stats(traces, labels, fs, figsize=(13, 4.4)):
    """T5. Descriptive statistics per class.

    Purpose: show that no SINGLE hand-crafted scalar cleanly separates the
    classes. If one did, the whole metric-learning exercise would be
    unnecessary, and the benchmark would be measuring nothing.
    Statistics are elementary on purpose (mean, sd, CV, 90th percentile) -- the
    point is that even together they leave the classes overlapping.
    """
    labels = np.asarray(labels, dtype=int)
    stats = {"mean IFR": [], "sd IFR": [], "CV = sd/mean": [], "p90 IFR": []}
    for x in traces:
        x = np.asarray(x, dtype=float)
        mu, sd = float(x.mean()), float(x.std())
        stats["mean IFR"].append(mu)
        stats["sd IFR"].append(sd)
        stats["CV = sd/mean"].append(sd / mu if mu > 0 else np.nan)
        stats["p90 IFR"].append(float(np.percentile(x, 90)))

    names = list(stats)
    fig, axarr = plt.subplots(1, len(names), figsize=figsize)
    axarr = np.atleast_1d(axarr).ravel()
    classes = sorted(set(labels.tolist()))
    rng = np.random.default_rng(0)
    for ax, nm in zip(axarr, names):
        v = np.asarray(stats[nm], dtype=float)
        data = [v[labels == k] for k in classes]
        bp = ax.boxplot(data, positions=np.arange(len(classes)), widths=0.6,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="k", linewidth=1.3))
        for patch, k in zip(bp["boxes"], classes):
            patch.set_facecolor(_c(k))
            patch.set_alpha(0.5)
        for pos, k in enumerate(classes):
            y = v[labels == k]
            ax.scatter(np.full(y.shape, pos) + rng.uniform(-0.13, 0.13, y.size),
                       y, s=12, color="k", alpha=0.45, linewidths=0)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(["ph %d" % k for k in classes], fontsize=8)
        ax.set_title(nm, fontsize=9)
        ax.tick_params(labelsize=7)
        _style(ax)
    fig.suptitle("T5  Elementary per-culture statistics: no single scalar "
                 "cleanly separates the phenotypes", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig
