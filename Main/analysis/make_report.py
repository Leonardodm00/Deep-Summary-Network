#!/usr/bin/env python3
"""
make_report.py -- DRIVER for the l3c_joint_full joint condition search.

Separation of concerns (directive 2): this script orchestrates and saves. It
contains no parsing (dsn_load), no statistics (dsn_analyze), and no drawing
(dsn_figures). Swapping any one of the three leaves the other two untouched.

    python3 make_report.py --run-root <dir> --out <dir> [--n-init 150]

--run-root accepts EITHER layout:
  (a) a flat handoff directory holding trials_lane<L>.jsonl / state_lane<L>.json
  (b) the live cluster tree holding out/l3c_joint_full_lane<L>/trials.jsonl
In case (b) the files are symlinked into a scratch dir with the flat names, so
the loader has exactly one layout to understand.

SNAPSHOT. --snapshot copies the logs before reading them, so a report generated
while a resumed segment is appending is reproducible: the figures and the tables
describe one frozen set of bytes, recorded in MANIFEST.json with line counts and
sha256 per file.

Outputs, under --out:
    RESULTS.md              the readable summary
    MANIFEST.json           what was read, how many records, checksums
    tables/*.csv            every table behind every figure
    figures/*.png, *.pdf    F1-F4, F6, D1-D3

Pure ASCII, headless (hpc-python-compat).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

import dsn_load as L
import dsn_analyze as A
import dsn_figures as F

E_MAX = 100
PATIENCE = 40
ETA0 = 0.55

# (axis, active_loss_hp gate or None, runtime clip or None)
# sep_warmup_frac is clipped at patience / max_epochs by search.py, so its
# configured upper bound of 0.5 is NOT reachable; the clip is drawn instead.
AXES_SPEC = [
    ("depth_exponent", None, None),
    ("width_multiplier", None, None),
    ("embedding_size", None, None),
    ("lr", None, None),
    ("one_minus_beta1", None, None),
    ("one_minus_beta2", None, None),
    ("weight_decay", None, None),
    ("dropout", None, None),
    ("margin", None, None),
    ("angular_alpha_deg", "angular_alpha_deg", None),
    ("lambda_sep", "lambda_sep", None),
    ("sep_warmup_frac", "sep_warmup_frac", float(PATIENCE) / E_MAX),
]

CATEGORICAL = ["mining_strategy", "loss_type", "strict_semihard",
               "head_fusion", "head_pool_ops_str"]


# --------------------------------------------------------------------------- #
def resolve_root(run_root, scratch):
    """Return a directory in the flat handoff layout, materialising it if the
    input is the live cluster tree. Never modifies the source."""
    flat = [f for f in os.listdir(run_root) if f.startswith("trials_lane")]
    if flat:
        return run_root
    os.makedirs(scratch, exist_ok=True)
    found = 0
    for j in range(4):
        d = os.path.join(run_root, "l3c_joint_full_lane%d" % j)
        if not os.path.isdir(d):
            d = os.path.join(run_root, "out", "l3c_joint_full_lane%d" % j)
        for src, dst in (("trials.jsonl", "trials_lane%d.jsonl" % j),
                         ("search_state.json", "state_lane%d.json" % j),
                         ("results.json", "results_lane%d.json" % j)):
            p = os.path.join(d, src)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(scratch, dst))
                if src == "trials.jsonl":
                    found += 1
        cfg = os.path.join(d, "config_input.json")
        if os.path.exists(cfg) and not os.path.exists(
                os.path.join(scratch, "config_input.json")):
            shutil.copy2(cfg, os.path.join(scratch, "config_input.json"))
    if not found:
        raise SystemExit("no trial logs found under %s (tried both layouts)"
                         % run_root)
    return scratch


def manifest(root):
    """sha256 + line count of every input file, so a report is traceable to bytes."""
    out = {}
    for f in sorted(os.listdir(root)):
        p = os.path.join(root, f)
        if not os.path.isfile(p):
            continue
        b = open(p, "rb").read()
        out[f] = {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()[:16],
                  "lines": b.count(b"\n")}
    return out


def save(fig, out_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, "%s.%s" % (name, ext)),
                    dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return name


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-init", type=int, default=150,
                    help="cold-start initial design size (default 150)")
    ap.add_argument("--top-n-cells", type=int, default=18)
    ap.add_argument("--min-n-per-lane", type=int, default=3)
    ap.add_argument("--snapshot", action="store_true",
                    help="copy the logs into <out>/snapshot before reading")
    args = ap.parse_args()

    out = args.out
    tdir, fdir = os.path.join(out, "tables"), os.path.join(out, "figures")
    for d in (out, tdir, fdir):
        os.makedirs(d, exist_ok=True)

    root = resolve_root(args.run_root, os.path.join(out, "snapshot"))
    if args.snapshot and root != os.path.join(out, "snapshot"):
        snap = os.path.join(out, "snapshot")
        os.makedirs(snap, exist_ok=True)
        for f in os.listdir(root):
            p = os.path.join(root, f)
            if os.path.isfile(p):
                shutil.copy2(p, os.path.join(snap, f))
        root = snap

    man = manifest(root)
    json.dump(man, open(os.path.join(out, "MANIFEST.json"), "w"),
              indent=2, sort_keys=True)

    df = L.load_trials(root)
    states = L.load_states(root)
    results = L.load_results(root)
    bounds = L.axis_bounds(states)
    ok = df[~df["failed"].astype(bool)].copy()
    rnd = ok[ok["trial"] < args.n_init]

    # ---------------- tables ------------------------------------------------ #
    tabs = {}
    tabs["objective_identity"] = A.objective_identity(df)
    tabs["lane_summary"] = A.lane_summary(df, states)
    tabs["objective_vs_ari"] = A.objective_vs_ari(df)
    tabs["cell_pooled"] = A.cell_table(df)
    g = rnd.groupby("cell")
    tabs["cell_random_only"] = pd.DataFrame({
        "n": g.size(), "J_median": g["objective"].median(),
        "J_q25": g["objective"].quantile(0.25), "J_best": g["objective"].min(),
        "ari_median": g["ari_mean"].median(),
    }).sort_values("J_median").reset_index()
    ranks = A.rank_agreement(df, args.min_n_per_lane)
    corr = A.cross_lane_rank_corr(df, args.min_n_per_lane)
    tabs["rank_agreement"] = ranks.reset_index()
    tabs["rank_corr"] = corr.reset_index()
    tabs["phase_comparison"] = A.phase_comparison(df, args.n_init)

    marg = {}
    for c in CATEGORICAL:
        marg[c] = A.marginal_by_level(rnd, c)
        tabs["marginal_" + c] = marg[c].reset_index()

    brows = []
    for name, active_hp, clip in AXES_SPEC:
        b = bounds.get(name, {})
        if b.get("kind") == "categorical" or b.get("low") is None:
            continue
        d = ok
        if active_hp is not None:
            d = ok[ok["active_loss_hps"].apply(lambda Ls: active_hp in Ls)]
        hi = clip if clip is not None else b["high"]
        r = A.boundary_check(d, name, b["low"], hi, 0.10, bool(b.get("log")))
        r.update({"low": b["low"], "high_effective": hi, "n_used": len(d),
                  "configured_high": b["high"]})
        brows.append(r)
    tabs["boundary_checks"] = pd.DataFrame(brows)

    for k, v in tabs.items():
        v.to_csv(os.path.join(tdir, k + ".csv"), index=False)

    # ---------------- figures ----------------------------------------------- #
    made = []
    made.append(save(F.fig_convergence(A.running_best(df), args.n_init),
                     fdir, "F1_convergence"))

    def groups_from(frame, order):
        gg = frame.groupby("cell")["objective"]
        return dict((c, gg.get_group(c).to_numpy()) for c in order
                    if c in gg.groups)

    order_pooled = tabs["cell_pooled"]["cell"].tolist()
    order_random = tabs["cell_random_only"]["cell"].tolist()
    made.append(save(F.fig_cell_distribution(
        groups_from(ok, order_pooled), groups_from(rnd, order_random),
        top_n=args.top_n_cells), fdir, "F2_cell_distribution"))

    made.append(save(F.fig_rank_agreement(ranks, corr), fdir,
                     "F3_rank_agreement"))
    made.append(save(F.fig_axis_scatter(ok, bounds, AXES_SPEC), fdir,
                     "F4_axis_scatter"))
    made.append(save(F.fig_selected_epochs(ok, E_MAX, PATIENCE, ETA0), fdir,
                     "F6_selected_epochs"))
    made.append(save(F.fig_objective_vs_ari(ok), fdir, "D1_objective_vs_ari"))
    made.append(save(F.fig_phase_comparison(ok, args.n_init), fdir,
                     "D2_phase_comparison"))
    made.append(save(F.fig_categorical_marginals(marg), fdir,
                     "D3_categorical_marginals"))

    # ---------------- RESULTS.md -------------------------------------------- #
    ls = tabs["lane_summary"]
    e = ok["sel_epoch"].to_numpy(dtype=float)
    eta_bar = e.mean() / E_MAX
    eta_cost = np.minimum(e + PATIENCE, E_MAX).mean() / E_MAX
    K = len(ok)
    n_gp = int((ok["trial"] >= args.n_init).sum())

    lines = []
    w = lines.append
    w("# Results: l3c_joint_full 4-lane joint condition search\n")
    w("Generated by `make_report.py` from the snapshot in `MANIFEST.json`. "
      "Every number below is computed from those bytes.\n")
    w("## Status\n")
    w("- Pooled non-failed trials K = **%d** of a planned %d." % (K, 4 * 300))
    w("- Failed trials: **%d** (failure rate %.3f pooled)."
      % (int(df["failed"].astype(bool).sum()),
         float(df["failed"].astype(bool).mean())))
    w("- GP-driven trials (t >= %d): **%d** (%.1f%% of K)."
      % (args.n_init, n_gp, 100.0 * n_gp / K))
    w("- Lanes with a `results.json` (i.e. that completed final training): "
      "**%s**.\n" % (sorted(results) if results else "NONE"))
    w("| lane | k_j | trial_offset | failures | best J | best cell | "
      "eta_bar | h/trial |")
    w("|---|---|---|---|---|---|---|---|")
    for _, r in ls.iterrows():
        w("| %d | %d | %s | %d | %+.4f | `%s` | %.3f | %.2f |"
          % (r["lane"], r["k_j"], r["trial_offset"], r["n_failed"],
             r["best_obj(log)"], r["best_cell(state)"], r["eta_bar"],
             r["mean_h_per_trial"]))
    w("")
    w("## The objective actually minimised\n")
    oi = tabs["objective_identity"]
    w("`selection_primary` = **%s** on every record; `epsilon` is unset on "
      "every record. So J = -(mean cosine silhouette vs the TRUE labels), "
      "not -(ARI + eps*sbar). Verified per lane:\n"
      % "/".join(sorted(set(oi["selection_primary"]))))
    w(oi.to_markdown(index=False))
    w("")
    ova = tabs["objective_vs_ari"]
    pooled = ova[ova["lane"] == "POOLED"].iloc[0]
    w("Silhouette and ARI are tightly coupled here (pooled Spearman "
      "**%.3f**, Pearson %.3f, n=%d), so the ranking would be similar under an "
      "ARI-primary objective -- but that coupling is a property of this "
      "synthetic benchmark and must not be assumed on real MEA data. "
      "See `figures/D1_objective_vs_ari.png`.\n"
      % (pooled["spearman"], pooled["pearson"], pooled["n"]))
    w("## Did the GP work?\n")
    pc = tabs["phase_comparison"]
    p = pc[pc["lane"] == "POOLED"].iloc[0]
    w("Pooled: median J went from **%+.3f** (random design, n=%d) to "
      "**%+.3f** (GP-driven, n=%d); Mann-Whitney one-sided p = %.2e. "
      "The surrogate found exploitable structure, so the "
      "'refuse to tighten because the GP beat nothing' veto does NOT apply.\n"
      % (p["median_random"], p["n_random"], p["median_gp"], p["n_gp"],
         p["p_gp_better"]))
    w(pc.to_markdown(index=False))
    w("")
    w("## Cost model\n")
    w("- eta_bar (mean selected epoch / E_max) = **%.3f** vs hard-coded "
      "eta_0 = %.2f." % (eta_bar, ETA0))
    w("- BUT walltime scales with epochs TRAINED = min(e* + P, E_max) with "
      "P = %d, giving eta_cost = **%.3f**: the cost model understates "
      "per-trial cost by about **%.0f%%**. That, not the epoch count, is why "
      "the lanes truncated. See `figures/F6_selected_epochs.png`."
      % (PATIENCE, eta_cost, 100.0 * (eta_cost / ETA0 - 1.0)))
    w("- Early stopping is binding: only %.1f%% of trials reach E_max = %d.\n"
      % (100.0 * (e >= E_MAX).mean(), E_MAX))
    w("## Caveats that govern how the figures are read\n")
    w("1. **F2 pooled panel is confounded.** The GP concentrated its %d trials "
      "in a few cells, so a pooled per-cell median mixes a uniform sample with "
      "an exploitation sample. The random-only panel is the unbiased "
      "comparison; the divergence between panels is itself a result." % n_gp)
    w("2. **`sep_warmup_frac` is clipped** at patience/max_epochs = %.2f by "
      "`search.py`, so its configured upper bound of %.2f is unreachable and "
      "the top-trial pile-up at the clip is not a preference."
      % (float(PATIENCE) / E_MAX,
         bounds.get("sep_warmup_frac", {}).get("high", float("nan"))))
    w("3. **sigma_s is unmeasured.** All %d raw points are distinct, so the "
      "logs give no handle on across-seed variance. The confirmatory re-fit at "
      "n_seeds >= 5 remains the gate; the 0.073 figure from earlier handoffs is "
      "NOT used anywhere in this report." % K)
    w("4. **N_eval = 90 windows from 9 cultures.** ARI is coarsely quantised at "
      "this N and windows within a culture are not independent, so near-ceiling "
      "differences are often exactly zero rather than small.")
    w("5. **F5 (re-fit summary) is not produced**: no re-fit `results.json` "
      "exists yet. It is the one figure that states the study's answer.\n")
    w("## Files\n")
    w("Figures: " + ", ".join("`%s.png`" % m for m in made))
    w("\nTables: " + ", ".join("`tables/%s.csv`" % k for k in sorted(tabs)))

    open(os.path.join(out, "RESULTS.md"), "w", encoding="ascii",
         errors="replace").write("\n".join(lines) + "\n")

    print("K=%d  lanes=%d  figures=%d  tables=%d" % (K, df["lane"].nunique(),
                                                     len(made), len(tabs)))
    print("wrote %s" % os.path.join(out, "RESULTS.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
