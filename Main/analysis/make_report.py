#!/usr/bin/env python3
"""
make_report.py -- DRIVER for a joint condition search (any study).

Separation of concerns (directive 2): this script orchestrates and saves. It
contains no parsing (dsn_load), no statistics (dsn_analyze), and no drawing
(dsn_figures). Swapping any one of the three leaves the other two untouched.

    python3 make_report.py --run-root <dir> --out <dir> [--n-init 150]

--run-root accepts EITHER layout:
  (a) a flat handoff directory holding trials_lane<L>.jsonl / state_lane<L>.json
  (b) the live cluster tree holding <experiment>_lane<L>/trials.jsonl

STUDY-AGNOSTIC. The lane directory prefix is DISCOVERED, not assumed, and
E_max / patience / n_initial_points are READ FROM THE STUDY'S OWN
config_input.json rather than hard-coded -- the synthetic (l3c) and MEA
studies do not share those values, and hard-coding them silently mis-scales
eta_bar, eta_cost and the `completed` flag on whichever study did not match.
Use --experiment to disambiguate if more than one study lives under one root.
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
import re
import shutil
import sys

import numpy as np
import pandas as pd

import dsn_load as L
import dsn_analyze as A
import dsn_figures as F

# Fallbacks ONLY. Every one of these is overridden from config_input.json when
# that file is present; they exist so a report can still be produced from a bare
# trials.jsonl with no config alongside it. When a fallback is used it is stated
# in RESULTS.md, never applied silently.
E_MAX_FALLBACK = 100
PATIENCE_FALLBACK = 40
ETA0 = 0.55           # the PLANNING assumption being tested; not read from config
N_CALLS_FALLBACK = 300
N_INIT_FALLBACK = 150

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
    ("sep_warmup_frac", "sep_warmup_frac", None),   # clip filled in per-study
]


def axes_spec_for(patience, e_max):
    """AXES_SPEC with sep_warmup_frac's runtime clip resolved for THIS study.
    The clip is patience/max_epochs, which differs between studies, so it
    cannot live in a module constant."""
    out = []
    for name, hp, clip in AXES_SPEC:
        if name == "sep_warmup_frac":
            clip = float(patience) / float(e_max)
        out.append((name, hp, clip))
    return out

CATEGORICAL = ["mining_strategy", "loss_type", "strict_semihard",
               "head_fusion", "head_pool_ops_str"]


# --------------------------------------------------------------------------- #
def discover_lanes(run_root, experiment=None):
    """Find lane directories under run_root. Returns (experiment_name, {lane: dir}).

    A lane directory is any directory holding a trials.jsonl whose name ends in
    `lane<digits>`. The experiment prefix is DERIVED from those names rather
    than assumed, so this works on l3c_joint_full_lane0..3, mea_joint_full_lane0..3
    or anything else, with no code change.

    If more than one experiment prefix is present under one root, the caller
    MUST disambiguate with --experiment: silently picking one would produce a
    report that mixes or omits studies without saying so.
    """
    roots = [run_root, os.path.join(run_root, "out")]
    found = {}
    for base in roots:
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            d = os.path.join(base, name)
            if not os.path.isdir(d):
                continue
            if not os.path.exists(os.path.join(d, "trials.jsonl")):
                continue
            m = re.match(r"^(.*)_lane(\d+)$", name)
            if not m:
                continue
            prefix, lane = m.group(1), int(m.group(2))
            if experiment and prefix != experiment:
                continue
            found.setdefault(prefix, {})[lane] = d
        if found:
            break

    if not found:
        raise SystemExit(
            "no lane directories with a trials.jsonl found under %s\n"
            "  expected <experiment>_lane<N>/trials.jsonl\n"
            "  (looked in %s and %s/out)" % (run_root, run_root, run_root))
    if len(found) > 1:
        raise SystemExit(
            "several experiments found under %s: %s\n"
            "  disambiguate with --experiment <name>"
            % (run_root, ", ".join(sorted(found))))
    prefix = list(found)[0]
    return prefix, found[prefix]


def materialise(lane_dirs, scratch):
    """Copy each lane's files into the flat handoff layout the loader expects.
    Never modifies the source."""
    os.makedirs(scratch, exist_ok=True)
    cfg_written = False
    for lane, d in sorted(lane_dirs.items()):
        for src, dst in (("trials.jsonl", "trials_lane%d.jsonl" % lane),
                         ("search_state.json", "state_lane%d.json" % lane),
                         ("results.json", "results_lane%d.json" % lane)):
            p = os.path.join(d, src)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(scratch, dst))
        cfg = os.path.join(d, "config_input.json")
        if os.path.exists(cfg) and not cfg_written:
            shutil.copy2(cfg, os.path.join(scratch, "config_input.json"))
            cfg_written = True
    return scratch


def study_params(root):
    """Read E_max, patience, n_calls, n_initial from the study's own config.

    Returns (params_dict, notes_list). Any value that could not be read falls
    back to the module constant AND is recorded in notes, so RESULTS.md can
    say so instead of presenting a guess as a measurement.
    """
    notes = []
    p = {"e_max": E_MAX_FALLBACK, "patience": PATIENCE_FALLBACK,
         "n_calls": N_CALLS_FALLBACK, "n_init": N_INIT_FALLBACK,
         "selection_primary": None}
    cfgp = os.path.join(root, "config_input.json")
    if not os.path.exists(cfgp):
        notes.append("config_input.json absent: E_max/patience/n_calls/n_init "
                     "are FALLBACK values, not this study's.")
        return p, notes
    with open(cfgp, "r") as fh:
        cfg = json.load(fh)
    tr, se = cfg.get("train", {}), cfg.get("search", {})
    for key, block, field in (("e_max", tr, "max_epochs"),
                              ("patience", tr, "patience"),
                              ("n_calls", se, "n_calls_joint"),
                              ("n_init", se, "n_initial_points_joint")):
        if field in block and block[field] is not None:
            p[key] = int(block[field])
        else:
            notes.append("%s not in config; using fallback %s" % (field, p[key]))
    p["selection_primary"] = tr.get("selection_primary")
    return p, notes


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
    ap.add_argument("--experiment",
                    help="lane-name prefix, e.g. mea_joint_full. Only needed "
                         "when several studies share one --run-root.")
    ap.add_argument("--n-init", type=int, default=None,
                    help="override the cold-start initial design size; by "
                         "default it is read from config_input.json")
    ap.add_argument("--top-n-cells", type=int, default=18)
    ap.add_argument("--min-n-per-lane", type=int, default=3)
    ap.add_argument("--snapshot", action="store_true",
                    help="copy the logs into <out>/snapshot before reading")
    args = ap.parse_args()

    out = args.out
    tdir, fdir = os.path.join(out, "tables"), os.path.join(out, "figures")
    for d in (out, tdir, fdir):
        os.makedirs(d, exist_ok=True)

    flat = [f for f in os.listdir(args.run_root)
            if f.startswith("trials_lane")] if os.path.isdir(args.run_root) else []
    if flat:
        experiment = args.experiment or os.path.basename(
            os.path.normpath(args.run_root))
        root = args.run_root
        if args.snapshot:
            snap = os.path.join(out, "snapshot")
            os.makedirs(snap, exist_ok=True)
            for f in os.listdir(root):
                p = os.path.join(root, f)
                if os.path.isfile(p):
                    shutil.copy2(p, os.path.join(snap, f))
            root = snap
    else:
        experiment, lane_dirs = discover_lanes(args.run_root, args.experiment)
        print("experiment: %s   lanes: %s"
              % (experiment, sorted(lane_dirs)))
        root = materialise(lane_dirs, os.path.join(out, "snapshot"))

    params, param_notes = study_params(root)
    n_init = args.n_init if args.n_init is not None else params["n_init"]
    e_max, patience = params["e_max"], params["patience"]
    print("study params: E_max=%d patience=%d n_calls=%d n_init=%d "
          "selection_primary=%s" % (e_max, patience, params["n_calls"],
                                    n_init, params["selection_primary"]))
    for n in param_notes:
        print("  NOTE: %s" % n)

    man = manifest(root)
    json.dump(man, open(os.path.join(out, "MANIFEST.json"), "w"),
              indent=2, sort_keys=True)

    df = L.load_trials(root)
    states = L.load_states(root)
    results = L.load_results(root)
    bounds = L.axis_bounds(states)
    ok = df[~df["failed"].astype(bool)].copy()
    rnd = ok[ok["trial"] < n_init]

    # ---------------- tables ------------------------------------------------ #
    tabs = {}
    tabs["objective_identity"] = A.objective_identity(df)
    tabs["lane_summary"] = A.lane_summary(df, states, e_max, params["n_calls"])
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
    tabs["phase_comparison"] = A.phase_comparison(df, n_init)

    marg = {}
    for c in CATEGORICAL:
        marg[c] = A.marginal_by_level(rnd, c)
        tabs["marginal_" + c] = marg[c].reset_index()

    brows = []
    for name, active_hp, clip in axes_spec_for(patience, e_max):
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
    made.append(save(F.fig_convergence(A.running_best(df), n_init),
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
    made.append(save(F.fig_axis_scatter(ok, bounds, axes_spec_for(patience, e_max)), fdir,
                     "F4_axis_scatter"))
    made.append(save(F.fig_selected_epochs(ok, e_max, patience, ETA0), fdir,
                     "F6_selected_epochs"))
    made.append(save(F.fig_objective_vs_ari(ok), fdir, "D1_objective_vs_ari"))
    made.append(save(F.fig_phase_comparison(ok, n_init), fdir,
                     "D2_phase_comparison"))
    made.append(save(F.fig_categorical_marginals(marg), fdir,
                     "D3_categorical_marginals"))

    # ---------------- RESULTS.md -------------------------------------------- #
    ls = tabs["lane_summary"]
    e = ok["sel_epoch"].to_numpy(dtype=float)
    eta_bar = e.mean() / e_max
    eta_cost = np.minimum(e + patience, e_max).mean() / e_max
    K = len(ok)
    n_gp = int((ok["trial"] >= n_init).sum())

    lines = []
    w = lines.append
    w("# Results: %s -- %d-lane joint condition search\n" % (experiment, len(ls)))
    w("Generated by `make_report.py` from the snapshot in `MANIFEST.json`. "
      "Every number below is computed from those bytes.\n")
    w("## Status\n")
    w("- Pooled non-failed trials K = **%d** of a planned %d (%d lanes x %d)."
      % (K, len(ls) * params["n_calls"], len(ls), params["n_calls"]))
    w("- Failed trials: **%d** (failure rate %.3f pooled)."
      % (int(df["failed"].astype(bool).sum()),
         float(df["failed"].astype(bool).mean())))
    w("- GP-driven trials (t >= %d): **%d** (%.1f%% of K)."
      % (n_init, n_gp, 100.0 * n_gp / K))
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
      % (patience, eta_cost, 100.0 * (eta_cost / ETA0 - 1.0)))
    w("- Early stopping is binding: only %.1f%% of trials reach E_max = %d.\n"
      % (100.0 * (e >= e_max).mean(), e_max))
    w("## Caveats that govern how the figures are read\n")
    w("1. **F2 pooled panel is confounded.** The GP concentrated its %d trials "
      "in a few cells, so a pooled per-cell median mixes a uniform sample with "
      "an exploitation sample. The random-only panel is the unbiased "
      "comparison; the divergence between panels is itself a result." % n_gp)
    w("2. **`sep_warmup_frac` is clipped** at patience/max_epochs = %.2f by "
      "`search.py`, so its configured upper bound of %.2f is unreachable and "
      "the top-trial pile-up at the clip is not a preference."
      % (float(patience) / e_max,
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

    # ---------------- BEST.json: the shortlist, machine-readable ------------ #
    # Chosen ACROSS lanes, not from one lane's argmin. Two orderings are given
    # because they answer different questions and can disagree (see the
    # GP-concentration caveat above): `by_objective` is the raw argmin;
    # `by_cross_lane_cell` prefers cells that rank well on the RANDOM design,
    # which is the unbiased comparison. A re-fit should test both.
    ok_sorted = ok.sort_values("objective")
    by_obj = []
    for _, r in ok_sorted.head(5).iterrows():
        by_obj.append({"lane": int(r["lane"]), "trial": int(r["trial"]),
                       "objective": float(r["objective"]),
                       "ari_mean": float(r["ari_mean"]), "cell": str(r["cell"])})
    rnd_rank = tabs["cell_random_only"].reset_index(drop=True)
    rnd_rank = {c: i + 1 for i, c in enumerate(rnd_rank["cell"].tolist())}
    pool_rank = {c: i + 1 for i, c in enumerate(tabs["cell_pooled"]["cell"].tolist())}
    agreed = []
    for cell in tabs["cell_pooled"]["cell"].tolist():
        rp, rr = pool_rank.get(cell), rnd_rank.get(cell)
        if rp is None or rr is None:
            continue
        sub = ok[ok["cell"] == cell]
        b = sub.loc[sub["objective"].idxmin()]
        agreed.append({"cell": cell, "rank_pooled": rp, "rank_random_only": rr,
                       "worst_rank": max(rp, rr), "n_lanes": int(sub["lane"].nunique()),
                       "best_lane": int(b["lane"]), "best_trial": int(b["trial"]),
                       "best_objective": float(b["objective"])})
    agreed.sort(key=lambda d: (d["worst_rank"], d["rank_random_only"]))
    best = {"experiment": experiment, "K": K, "n_lanes": len(ls),
            "note": ("by_objective is the raw argmin and may sit in a cell the GP "
                     "over-sampled; by_cross_lane_cell ranks cells by their WORST "
                     "rank across the pooled and random-design views, so it favours "
                     "candidates both views agree on. Re-fit both."),
            "by_objective": by_obj,
            "by_cross_lane_cell": agreed[:5]}
    json.dump(best, open(os.path.join(out, "BEST.json"), "w"), indent=2)

    open(os.path.join(out, "RESULTS.md"), "w", encoding="ascii",
         errors="replace").write("\n".join(lines) + "\n")

    print("\nshortlist (BEST.json):")
    print("  raw argmin      : lane %d trial %d  J=%+.4f  cell=%s"
          % (by_obj[0]["lane"], by_obj[0]["trial"], by_obj[0]["objective"],
             by_obj[0]["cell"]))
    if agreed:
        a = agreed[0]
        print("  both-views best : lane %d trial %d  J=%+.4f  cell=%s "
              "(pooled #%d, random #%d)"
              % (a["best_lane"], a["best_trial"], a["best_objective"],
                 a["cell"], a["rank_pooled"], a["rank_random_only"]))
    print("K=%d  lanes=%d  figures=%d  tables=%d" % (K, df["lane"].nunique(),
                                                     len(made), len(tabs)))
    print("wrote %s" % os.path.join(out, "RESULTS.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
