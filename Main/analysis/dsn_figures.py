"""
dsn_figures.py -- PLOTTING LAYER ONLY for the l3c_joint_full joint search.

Separation of concerns (directive 2). This module draws. It does NOT read the
trial logs (dsn_load does that) and does NOT compute statistics (dsn_analyze
does that). Every function takes an already-prepared pandas object and returns
a matplotlib Figure. No function here calls read/parse/groupby-for-inference on
raw records, and none of them saves: saving is the driver's job (make_report).

HPC rules (hpc-python-compat):
  - pure ASCII source;
  - headless: matplotlib.use("Agg") is set at import, before pyplot;
  - never plt.show(); the driver calls savefig.

Figure numbering follows HANDOFF_next_steps_refit_tightening_figures.md, S7.2:
  F1 convergence traces .............. fig_convergence
  F2 per-cell distribution ........... fig_cell_distribution
  F3 cross-lane rank agreement ....... fig_rank_agreement
  F4 axis-vs-objective scatter ....... fig_axis_scatter
  F5 confirmatory re-fit summary ..... NOT IMPLEMENTED: needs re-fit results.json,
                                       which do not exist yet for any lane.
  F6 selected-epoch histogram ........ fig_selected_epochs
Three diagnostics the handoff did not list, added because the segment-1
analysis turned them into results in their own right:
  D1 objective (silhouette) vs ARI ... fig_objective_vs_ari
  D2 random vs GP phase .............. fig_phase_comparison
  D3 categorical marginals ........... fig_categorical_marginals

CONVENTION CARRIED THROUGHOUT: the objective is MINIMISED and NEGATIVE-GOOD.
Every axis showing J is drawn with more-negative UPWARD or LEFTWARD as marked in
the axis label, and every ordering is ascending. A reader who reads these as
"higher is better" reads them backwards, so each J axis is labelled explicitly.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")                      # MUST precede the pyplot import

import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402

__all__ = [
    "LANE_COLORS", "J_LABEL",
    "fig_convergence", "fig_cell_distribution", "fig_rank_agreement",
    "fig_axis_scatter", "fig_selected_epochs", "fig_objective_vs_ari",
    "fig_phase_comparison", "fig_categorical_marginals",
]

LANE_COLORS = {0: "#1f77b4", 1: "#d62728", 2: "#2ca02c", 3: "#9467bd"}
J_LABEL = "objective J = -silhouette   (MINIMISED: lower is better)"

_GRID = dict(alpha=0.3, linewidth=0.5)


def _style(ax):
    ax.grid(True, **_GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


# --------------------------------------------------------------- F1 --------- #
def fig_convergence(traces, n_init=150, figsize=(11, 6)):
    """F1. Running best J against trial index, one line per lane.

    Parameters
    ----------
    traces : DataFrame from dsn_analyze.running_best, columns
             lane, trial, objective, running_best.
    n_init : the cold-start initial-design size, drawn as the random/GP boundary.
    """
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    for lane, g in traces.groupby("lane"):
        c = LANE_COLORS.get(lane, None)
        ax0.step(g["trial"], g["running_best"], where="post", color=c,
                 label="lane %d (k=%d)" % (lane, len(g)), linewidth=1.8)
        ax0.scatter(g["trial"], g["objective"], s=5, color=c, alpha=0.25,
                    linewidths=0)
    ax0.axvline(n_init, color="k", linestyle="--", linewidth=1.2)
    ax0.text(n_init + 3, ax0.get_ylim()[1], "N_init = %d\nrandom | GP-driven" % n_init,
             va="top", ha="left", fontsize=8)
    ax0.set_ylabel(J_LABEL, fontsize=9)
    ax0.legend(fontsize=8, loc="lower left", frameon=False)
    ax0.set_title("F1  Convergence: running best objective per lane "
                  "(dots = individual trials)", fontsize=10)
    _style(ax0)

    # lower panel: rolling median, which shows the reliability gain the
    # running-best curve hides
    for lane, g in traces.groupby("lane"):
        g = g.sort_values("trial")
        roll = g["objective"].rolling(20, min_periods=5).median()
        ax1.plot(g["trial"], roll, color=LANE_COLORS.get(lane), linewidth=1.5)
    ax1.axvline(n_init, color="k", linestyle="--", linewidth=1.2)
    ax1.set_xlabel("trial index t (cumulative within lane)")
    ax1.set_ylabel("rolling median J\n(window 20)", fontsize=9)
    ax1.set_title("The GP's real contribution is the median, not the minimum",
                  fontsize=9)
    _style(ax1)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------- F2 --------- #
def fig_cell_distribution(pooled_groups, random_groups, top_n=18,
                          figsize=(13, 8)):
    """F2. Per-cell distribution of J, pooled vs random-design-only.

    BOTH panels are drawn deliberately. The pooled view mixes the uniform
    random design with the GP's exploitation sample, so a cell the GP visited
    heavily gets a median computed mostly inside the exploited basin. The
    random-only view is the unbiased comparison. Their divergence is a result,
    not a nuisance, so the figure shows it rather than picking one.

    Parameters
    ----------
    pooled_groups, random_groups : dict {cell_name: 1-D array of J}, already
        ordered by the caller (best median first).
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
    for ax, groups, title in (
            (axes[0], pooled_groups,
             "pooled: random design + GP phase\n(CONFOUNDED -- see caption)"),
            (axes[1], random_groups,
             "random design only (t < N_init)\n(unbiased over cells)")):
        names = list(groups.keys())[:top_n]
        data = [np.asarray(groups[n], dtype=float) for n in names]
        pos = np.arange(len(names))[::-1]
        bp = ax.boxplot(data, positions=pos, vert=False, widths=0.65,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="k", linewidth=1.5))
        for patch in bp["boxes"]:
            patch.set_facecolor("#c6dbef")
            patch.set_edgecolor("#3182bd")
        for p, d in zip(pos, data):
            ax.scatter(d, np.full(d.shape, p), s=6, color="#08519c",
                       alpha=0.45, linewidths=0, zorder=3)
            # n_kappa annotation is NOT optional (handoff S7.2 item 2)
            ax.text(ax.get_xlim()[1], p, " n=%d" % len(d), va="center",
                    fontsize=7, color="#444444")
        ax.set_yticks(pos)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel(J_LABEL, fontsize=9)
        ax.set_title(title, fontsize=9)
        _style(ax)
    fig.suptitle("F2  Objective by experimental cell, top %d by median "
                 "(more negative = better)" % top_n, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# --------------------------------------------------------------- F3 --------- #
def fig_rank_agreement(ranks, corr, top_n=12, figsize=(12, 5.5)):
    """F3. Cross-lane rank agreement: per-lane cell rank + pairwise Spearman.

    Parameters
    ----------
    ranks : DataFrame from dsn_analyze.rank_agreement (index = cell).
    corr  : DataFrame from dsn_analyze.cross_lane_rank_corr.
    """
    lane_cols = [c for c in ranks.columns if c.startswith("rank_lane")]
    sub = ranks.head(top_n)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize,
                                   gridspec_kw={"width_ratios": [2.2, 1]})

    M = sub[lane_cols].to_numpy(dtype=float)
    im = ax0.imshow(M, cmap="viridis_r", aspect="auto")
    ax0.set_xticks(range(len(lane_cols)))
    ax0.set_xticklabels([c.replace("rank_", "") for c in lane_cols], fontsize=8)
    ax0.set_yticks(range(len(sub)))
    ax0.set_yticklabels(sub.index, fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax0.text(j, i, "-" if np.isnan(v) else "%d" % v, ha="center",
                     va="center", fontsize=7,
                     color="w" if (not np.isnan(v) and v < np.nanmax(M) * 0.6)
                     else "k")
    ax0.set_title("F3a  within-lane rank of each cell (1 = best)\n"
                  "'-' = lane visited that cell too few times to rank",
                  fontsize=9)
    fig.colorbar(im, ax=ax0, shrink=0.8, label="rank (1 = best)")

    C = corr.to_numpy(dtype=float)
    im1 = ax1.imshow(C, cmap="RdYlGn", vmin=-1, vmax=1)
    labels = [c.replace("rank_", "") for c in corr.columns]
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=8)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if not np.isnan(C[i, j]):
                ax1.text(j, i, "%.2f" % C[i, j], ha="center", va="center",
                         fontsize=8)
    ax1.set_title("F3b  pairwise Spearman of\ncell rankings between lanes",
                  fontsize=9)
    fig.colorbar(im1, ax=ax1, shrink=0.8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------- F4 --------- #
def fig_axis_scatter(df_ok, bounds, axes_spec, top_frac=0.10, ncols=4,
                     figsize=(15, 9)):
    """F4. One panel per continuous axis: J against the raw sampled value.

    axes_spec : list of (axis_name, active_hp_or_None, clip_high_or_None).
        active_hp: if not None, only trials whose `active_loss_hps` contains it
        are plotted -- an inert axis's sampled value is noise (handoff S5.2).
        clip_high: an effective upper limit imposed at runtime that differs from
        the configured bound (sep_warmup_frac is clipped at patience/max_epochs).
    """
    n = len(axes_spec)
    nrows = int(np.ceil(n / float(ncols)))
    fig, axarr = plt.subplots(nrows, ncols, figsize=figsize)
    axarr = np.atleast_1d(axarr).ravel()
    m_top = max(5, int(np.ceil(top_frac * len(df_ok))))
    top_idx = set(df_ok.nsmallest(m_top, "objective").index)

    for ax, (name, active_hp, clip_high) in zip(axarr, axes_spec):
        d = df_ok
        note = ""
        if active_hp is not None:
            mask = d["active_loss_hps"].apply(lambda L: active_hp in L)
            d = d[mask]
            note = "\nactive only (n=%d)" % len(d)
        col = "raw_" + name
        x = d[col].to_numpy(dtype=float)
        y = d["objective"].to_numpy(dtype=float)
        is_top = np.array([i in top_idx for i in d.index])
        b = bounds.get(name, {})
        logx = bool(b.get("log"))
        ax.scatter(x[~is_top], y[~is_top], s=7, color="#999999", alpha=0.45,
                   linewidths=0, label="all")
        ax.scatter(x[is_top], y[is_top], s=18, color="#d62728", alpha=0.9,
                   linewidths=0, label="top %d%%" % int(round(100 * top_frac)))
        if b.get("low") is not None:
            ax.axvline(b["low"], color="k", linestyle=":", linewidth=1.0)
            ax.axvline(b["high"], color="k", linestyle=":", linewidth=1.0)
        if clip_high is not None:
            ax.axvline(clip_high, color="#ff7f0e", linestyle="--", linewidth=1.2)
            note += "\nruntime clip at %.2f" % clip_high
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(name + (" [log]" if logx else ""), fontsize=8)
        ax.set_ylabel("J", fontsize=8)
        ax.set_title(name + note, fontsize=8)
        ax.tick_params(labelsize=7)
        _style(ax)
    for ax in axarr[n:]:
        ax.set_visible(False)
    axarr[0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("F4  Objective against each continuous axis; dotted = configured "
                 "bounds, red = best %d%% of trials (lower J is better)"
                 % int(round(100 * top_frac)), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# --------------------------------------------------------------- F6 --------- #
def fig_selected_epochs(df_ok, e_max=100, patience=40, eta0=0.55,
                        figsize=(11, 4.5)):
    """F6. Distribution of the selected epoch e*, with the cost-model markers.

    Two histograms, because the cost model needs the second one and the
    handoff's Eq. (7) only measures the first:
      left  : e*, the SELECTED epoch -> eta_bar (what Eq. 7 measures)
      right : min(e* + patience, E_max), the epochs actually TRAINED -> the
              fraction walltime really scales with.
    """
    e = df_ok["sel_epoch"].to_numpy(dtype=float)
    trained = np.minimum(e + patience, e_max)
    eta_bar = e.mean() / e_max
    eta_cost = trained.mean() / e_max

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)
    ax0.hist(e, bins=np.arange(0, e_max + 5, 5), color="#6baed6",
             edgecolor="#2171b5")
    ax0.axvline(e.mean(), color="#d62728", linewidth=1.8,
                label=r"mean $e^{\star}$ = %.1f ($\bar{\eta}$ = %.3f)"
                      % (e.mean(), eta_bar))
    ax0.axvline(eta0 * e_max, color="k", linestyle="--", linewidth=1.5,
                label=r"hard-coded $\eta_0$ = %.2f" % eta0)
    ax0.set_xlabel(r"selected epoch $e^{\star}$")
    ax0.set_ylabel("trials")
    ax0.set_title("F6a  selected epoch (early stopping is binding:\n%.1f%% reach "
                  "E_max = %d)" % (100.0 * (e >= e_max).mean(), e_max), fontsize=9)
    ax0.legend(fontsize=8, frameon=False)
    _style(ax0)

    ax1.hist(trained, bins=np.arange(0, e_max + 5, 5), color="#fdae6b",
             edgecolor="#e6550d")
    ax1.axvline(trained.mean(), color="#d62728", linewidth=1.8,
                label=r"mean trained = %.1f ($\eta_{cost}$ = %.3f)"
                      % (trained.mean(), eta_cost))
    ax1.axvline(eta0 * e_max, color="k", linestyle="--", linewidth=1.5,
                label=r"hard-coded $\eta_0$ = %.2f" % eta0)
    ax1.set_xlabel(r"epochs TRAINED = min($e^{\star}$ + P, $E_{max}$), P = %d"
                   % patience)
    ax1.set_title("F6b  what walltime actually scales with\n"
                  "(cost model understates by %.0f%%)"
                  % (100.0 * (eta_cost / eta0 - 1.0)), fontsize=9)
    ax1.legend(fontsize=8, frameon=False)
    _style(ax1)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------- D1 --------- #
def fig_objective_vs_ari(df_ok, figsize=(6.5, 5.5)):
    """D1. Silhouette (the metric optimised) against ARI (the stated target)."""
    fig, ax = plt.subplots(figsize=figsize)
    for lane, g in df_ok.groupby("lane"):
        ax.scatter(g["sil_mean"], g["ari_mean"], s=10, alpha=0.55,
                   color=LANE_COLORS.get(lane), linewidths=0,
                   label="lane %d" % lane)
    lo = min(df_ok["sil_mean"].min(), df_ok["ari_mean"].min())
    hi = max(df_ok["sil_mean"].max(), df_ok["ari_mean"].max())
    ax.plot([lo, hi], [lo, hi], color="k", linestyle=":", linewidth=1.0)
    ax.set_xlabel("mean cosine silhouette vs TRUE labels  (the OPTIMISED metric)")
    ax.set_ylabel("ARI  (the metric the study is stated in)")
    ax.set_title("D1  The search minimised -silhouette, not -ARI.\n"
                 "They are tightly coupled ON THIS BENCHMARK.", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _style(ax)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------- D2 --------- #
def fig_phase_comparison(df_ok, n_init=150, figsize=(10, 4.5)):
    """D2. Objective distribution, random initial design vs GP-driven phase."""
    d = df_ok.copy()
    d["phase"] = np.where(d["trial"] < n_init, "random", "gp")
    lanes = sorted(d["lane"].unique())
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)

    data, labels, colors = [], [], []
    for lane in lanes:
        for ph, col in (("random", "#bdbdbd"), ("gp", "#4292c6")):
            v = d[(d["lane"] == lane) & (d["phase"] == ph)]["objective"].to_numpy()
            if v.size:
                data.append(v)
                labels.append("L%d\n%s\nn=%d" % (lane, ph, v.size))
                colors.append(col)
    bp = ax0.boxplot(data, patch_artist=True, showfliers=False, widths=0.6,
                     medianprops=dict(color="k", linewidth=1.4))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    ax0.set_xticklabels(labels, fontsize=6.5)
    ax0.set_ylabel(J_LABEL, fontsize=8)
    ax0.set_title("D2a  random design vs GP-driven, per lane\n"
                  "(lane 1 has NO gp box: it never reached N_init)", fontsize=9)
    _style(ax0)

    for ph, col in (("random", "#bdbdbd"), ("gp", "#4292c6")):
        v = d[d["phase"] == ph]["objective"].to_numpy()
        ax1.hist(v, bins=30, alpha=0.65, color=col, edgecolor="none",
                 label="%s (n=%d, median %+.3f)" % (ph, v.size, np.median(v)))
    ax1.set_xlabel(J_LABEL, fontsize=8)
    ax1.set_ylabel("trials")
    ax1.set_title("D2b  pooled over lanes", fontsize=9)
    ax1.legend(fontsize=8, frameon=False)
    _style(ax1)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------- D3 --------- #
def fig_categorical_marginals(marginals, figsize=(12, 7)):
    """D3. Objective marginal over each categorical axis.

    Parameters
    ----------
    marginals : dict {axis_name: DataFrame indexed by level with columns
        n, J_median, J_q25, J_best} -- from dsn_analyze.marginal_by_level,
        computed on the RANDOM design so the GP's exploitation does not bias it.
    """
    n = len(marginals)
    ncols = 3
    nrows = int(np.ceil(n / float(ncols)))
    fig, axarr = plt.subplots(nrows, ncols, figsize=figsize)
    axarr = np.atleast_1d(axarr).ravel()
    for ax, (name, tab) in zip(axarr, marginals.items()):
        pos = np.arange(len(tab))
        ax.barh(pos, tab["J_median"].to_numpy(), color="#74c476",
                edgecolor="#238b45", height=0.6)
        for p, (lvl, row) in zip(pos, tab.iterrows()):
            ax.text(0, p, "  n=%d" % int(row["n"]), va="center", fontsize=7)
        ax.set_yticks(pos)
        ax.set_yticklabels([str(i) for i in tab.index], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("median J (more negative = better)", fontsize=8)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
        _style(ax)
    for ax in axarr[n:]:
        ax.set_visible(False)
    fig.suptitle("D3  Categorical marginals on the RANDOM design only "
                 "(unbiased by GP exploitation)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
