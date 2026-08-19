"""
dsn_analyze.py -- analysis layer ONLY. Consumes the tidy frames from dsn_load.
No file I/O of raw logs, no plotting.

Implements, in the numbering of HANDOFF_interpreting_joint_search_results.md:
  Eq. (1) k_j                       -> lane_summary
  Eq. (4) n_kappa, per-cell pooling -> cell_table
  Eq. (5) cross-lane rank agreement -> rank_agreement
  Eq. (6) failure rate              -> lane_summary
  Eq. (7) eta_bar                   -> lane_summary / eta_summary
  sec 7.3 random vs GP half         -> phase_comparison
Plus two checks the handoff did not anticipate:
  objective_identity  -- what the objective actually IS in these logs
  objective_vs_ari    -- whether the optimised metric tracks the stated scientific target
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

N_CALLS = 300
N_INIT = 150
E_MAX = 100
ETA0 = 0.55


# ---------------------------------------------------------------- identity ---
def objective_identity(df: pd.DataFrame) -> pd.DataFrame:
    """Establish empirically which metric the logged `objective` encodes.

    Returns per-lane counts of the three candidate identities. This is a check,
    not an assumption: the handoff asserts J = -(ARI + eps*sil), the logs may
    say otherwise.
    """
    out = []
    for lane, g in df.groupby("lane"):
        out.append({
            "lane": lane,
            "n": len(g),
            "obj == -mean": int(np.sum(np.isclose(g["objective"], -g["mean"], atol=1e-12))),
            "mean == sil_mean": int(np.sum(np.isclose(g["mean"], g["sil_mean"], atol=1e-12))),
            "mean == ari_mean": int(np.sum(np.isclose(g["mean"], g["ari_mean"], atol=1e-12))),
            "epsilon set": int(g["epsilon"].notna().sum()),
            "selection_primary": "/".join(sorted(set(map(str, g["selection_primary"])))),
        })
    return pd.DataFrame(out)


def objective_vs_ari(df: pd.DataFrame) -> pd.DataFrame:
    """Rank and linear association between the OPTIMISED score and ARI.

    If the search minimised -silhouette while the scientific question is stated
    in ARI, this table quantifies how much the optimisation transfers.
    """
    ok = df[~df["failed"].astype(bool)]
    out = []
    for lane, g in ok.groupby("lane"):
        rho, p_rho = stats.spearmanr(g["sil_mean"], g["ari_mean"])
        r, p_r = stats.pearsonr(g["sil_mean"], g["ari_mean"])
        out.append({"lane": lane, "n": len(g), "spearman": rho, "p_spearman": p_rho,
                    "pearson": r, "p_pearson": p_r})
    rho, p_rho = stats.spearmanr(ok["sil_mean"], ok["ari_mean"])
    r, p_r = stats.pearsonr(ok["sil_mean"], ok["ari_mean"])
    out.append({"lane": "POOLED", "n": len(ok), "spearman": rho, "p_spearman": p_rho,
                "pearson": r, "p_pearson": p_r})
    return pd.DataFrame(out)


# ------------------------------------------------------------ lane summary ---
def lane_summary(df: pd.DataFrame, states: dict) -> pd.DataFrame:
    """Eq. (1), Eq. (6), Eq. (7) per lane, plus resume status and timing."""
    rows = []
    for lane, g in df.groupby("lane"):
        st = states.get(lane, {})
        k = len(g)
        failed = g["failed"].astype(bool)
        ok = g[~failed]
        idx = g["trial"].to_numpy()
        contiguous = bool(np.array_equal(idx, np.arange(idx.min(), idx.max() + 1)))
        wall = g["wall_elapsed_s"].to_numpy()
        rows.append({
            "lane": lane,
            "k_j": k,
            "completed": k >= N_CALLS,
            "trial_offset": st.get("trial_offset"),
            "n_trials_total(state)": st.get("n_trials_total"),
            "n_this_segment(state)": st.get("n_trials_this_segment"),
            "idx_contiguous": contiguous,
            "n_failed": int(failed.sum()),
            "failure_rate": float(failed.mean()),
            "best_obj(log)": float(ok["objective"].min()),
            "best_obj(state)": st.get("best_objective"),
            "best_trial(state)": st.get("best_trial"),
            "best_cell(state)": st.get("best_cell"),
            "eta_bar": float(ok["sel_epoch"].mean() / E_MAX),
            "mean_sel_epoch": float(ok["sel_epoch"].mean()),
            "median_sel_epoch": float(ok["sel_epoch"].median()),
            "frac_at_E_max": float((ok["sel_epoch"] >= E_MAX).mean()),
            "wall_total_h": float(wall.max() / 3600.0),
            "mean_h_per_trial": float(np.diff(np.r_[0.0, wall]).mean() / 3600.0),
            "projected_frac": float(g["projected"].astype(bool).mean()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------- per cell ----
def cell_table(df: pd.DataFrame, min_n: int = 1) -> pd.DataFrame:
    """Eq. (4): pool non-failed trials over lanes, group by post-Pi `cell`.

    Reports n_kappa, the median (a statistic that does NOT scale with n_kappa),
    the best-in-cell (which does), and how many lanes visited the cell.
    """
    ok = df[~df["failed"].astype(bool)]
    g = ok.groupby("cell")
    tab = pd.DataFrame({
        "n_kappa": g.size(),
        "J_median": g["objective"].median(),
        "J_q25": g["objective"].quantile(0.25),
        "J_best": g["objective"].min(),
        "J_mean": g["objective"].mean(),
        "J_std": g["objective"].std(),
        "ari_median": g["ari_mean"].median(),
        "ari_best": g["ari_mean"].max(),
        "n_lanes": g["lane"].nunique(),
    }).reset_index()
    tab = tab[tab["n_kappa"] >= min_n]
    return tab.sort_values("J_median").reset_index(drop=True)


def rank_agreement(df: pd.DataFrame, min_n_per_lane: int = 3) -> pd.DataFrame:
    """Eq. (5): per-lane rank of each cell by within-lane median objective.

    A cell is ranked in a lane only if that lane visited it at least
    `min_n_per_lane` times; otherwise the lane's rank is NaN, because a median
    over one or two trials is not a median.
    """
    ok = df[~df["failed"].astype(bool)]
    per_lane = {}
    for lane, g in ok.groupby("lane"):
        med = g.groupby("cell")["objective"].median()
        cnt = g.groupby("cell").size()
        med = med[cnt >= min_n_per_lane]
        per_lane[lane] = med.rank(method="average")  # ascending: 1 = best
    ranks = pd.DataFrame(per_lane)
    ranks.columns = [f"rank_lane{c}" for c in ranks.columns]
    ranks["n_lanes_ranked"] = ranks.notna().sum(axis=1)
    ranks["mean_rank"] = ranks[[c for c in ranks.columns if c.startswith("rank_lane")]].mean(axis=1)
    ranks["rank_spread"] = (
        ranks[[c for c in ranks.columns if c.startswith("rank_lane")]].max(axis=1)
        - ranks[[c for c in ranks.columns if c.startswith("rank_lane")]].min(axis=1)
    )
    return ranks.sort_values("mean_rank")


def cross_lane_rank_corr(df: pd.DataFrame, min_n_per_lane: int = 3) -> pd.DataFrame:
    """Pairwise Spearman correlation between lanes' cell rankings.

    This is the quantitative version of 'do the four lanes agree?'. Near-zero
    correlations mean the per-lane rankings are noise.
    """
    ranks = rank_agreement(df, min_n_per_lane)
    cols = [c for c in ranks.columns if c.startswith("rank_lane")]
    out = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a in cols:
        for b in cols:
            sub = ranks[[a, b]].dropna()
            if len(sub) >= 4 and a != b:
                out.loc[a, b] = stats.spearmanr(sub[a], sub[b]).statistic
            elif a == b:
                out.loc[a, b] = 1.0
    return out


# ----------------------------------------------------- random vs GP phase ----
def phase_comparison(df: pd.DataFrame, n_init: int = N_INIT) -> pd.DataFrame:
    """Section 7.3: did the GP-driven half beat the random design?

    Compares the objective distribution of trials t < n_init (random) against
    t >= n_init (GP-driven), per lane and pooled. Mann-Whitney U, one-sided
    (GP better = more negative objective).
    """
    ok = df[~df["failed"].astype(bool)].copy()
    ok["phase"] = np.where(ok["trial"] < n_init, "random", "gp")
    rows = []
    for lane, g in list(ok.groupby("lane")) + [("POOLED", ok)]:
        r = g[g["phase"] == "random"]["objective"].to_numpy()
        p = g[g["phase"] == "gp"]["objective"].to_numpy()
        rec = {"lane": lane, "n_random": len(r), "n_gp": len(p),
               "median_random": np.median(r) if len(r) else np.nan,
               "median_gp": np.median(p) if len(p) else np.nan,
               "best_random": r.min() if len(r) else np.nan,
               "best_gp": p.min() if len(p) else np.nan}
        if len(r) >= 5 and len(p) >= 5:
            u = stats.mannwhitneyu(p, r, alternative="less")  # gp more negative?
            rec["U"] = u.statistic
            rec["p_gp_better"] = u.pvalue
        else:
            rec["U"] = np.nan
            rec["p_gp_better"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def running_best(df: pd.DataFrame) -> pd.DataFrame:
    """Convergence trace: running min of the objective vs trial index, per lane."""
    ok = df[~df["failed"].astype(bool)].copy()
    ok = ok.sort_values(["lane", "trial"])
    ok["running_best"] = ok.groupby("lane")["objective"].cummin()
    return ok[["lane", "trial", "objective", "running_best", "ari_mean", "cell"]]


# ------------------------------------------------------- categorical axes ----
def marginal_by_level(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Pooled marginal of the objective over one categorical axis."""
    ok = df[~df["failed"].astype(bool)]
    g = ok.groupby(col)
    return pd.DataFrame({
        "n": g.size(),
        "J_median": g["objective"].median(),
        "J_q25": g["objective"].quantile(0.25),
        "J_best": g["objective"].min(),
        "ari_median": g["ari_mean"].median(),
    }).sort_values("J_median")


def boundary_check(df: pd.DataFrame, axis: str, lo: float, hi: float,
                   top_frac: float = 0.1, log_space: bool = False) -> dict:
    """Section 5.2 of the next-steps handoff: quantiles of the top trials on one
    continuous axis, and whether they pile up against a configured bound."""
    ok = df[~df["failed"].astype(bool)]
    m = max(5, int(np.ceil(top_frac * len(ok))))
    top = ok.nsmallest(m, "objective")
    x = top[f"raw_{axis}"].to_numpy(dtype=float)
    lo_, hi_, x_ = (np.log10(lo), np.log10(hi), np.log10(x)) if log_space else (lo, hi, x)
    q10, q50, q90 = np.percentile(x_, [10, 50, 90])
    rng = hi_ - lo_
    return {
        "axis": axis, "m_top": m, "log_space": log_space,
        "Q10": q10, "Q50": q50, "Q90": q90,
        "spread_frac_of_range": (q90 - q10) / rng if rng else np.nan,
        "pileup_upper": bool(q90 >= hi_ - 0.05 * rng),
        "pileup_lower": bool(q10 <= lo_ + 0.05 * rng),
    }
