"""
make_joint_search_config.py
===========================

Emit THE ONE config that replaces the 52-cell factorial.

    python3 hpc/make_joint_search_config.py \\
        --base hpc/Config/config_l3c_h_multimean_005.json \\
        --out  hpc/Config/config_l3c_joint_search.json

WHY A GENERATOR AND NOT A CHECKED-IN JSON
-----------------------------------------
Two of the numbers this file writes are NOT settled, and a checked-in JSON
would freeze them silently:

  * max_epochs / patience. The design document computes that N_calls = 300 at
    E_max = 100 and n_batches = 100 needs roughly 157 h against a 144 h
    walltime, and RECOMMENDS E_max = 60, P = 20 -- explicitly recorded as "not
    yet accepted by the user". The defaults here are therefore the STATED
    values (100 / 40), and --max-epochs / --patience change them. Either way
    this script prints the walltime arithmetic, so the choice is made with the
    number in front of you rather than by whichever value happened to be
    committed.

  * the architecture size-mix multiplier, which is UNMEASURED. The estimate
    below is the depth-4 rate; a search that samples depth-5 backbones (about
    20 M parameters against 2.6 M at depth 4) will exceed it by an unknown
    factor. Stage 7 (--search-dry-run, then timing 10 sampled points end to
    end) is what resolves this. DO NOT SUBMIT on the strength of the number
    printed here.

WHAT IS SET, AND WHY
--------------------
  search.search_mode = "joint_conditions"   the 18-axis space
  the six *_choices lists                   the full factorial, as levels
  search ranges                             the design document's table
  train.n_seeds = 1                         deliberate: see the note printed
  train.batches_per_epoch = 100             6.7x the screening's step budget,
                                            and the single most important
                                            change in the design
  train.sep_warmup_frac = 0.3               tau, FIXED not searched
  train.lambda_sep = 0.1                    NOT a value that will be used --
                                            it is the CLAMP CONSTANT for the
                                            trials where lambda_sep is
                                            inactive. It must be exactly the
                                            TrainConfig default, or every
                                            triplet/joint trial fires the
                                            "INERT lambda_sep" RuntimeWarning
                                            and the reader learns to ignore
                                            warnings.
  train.sep_gate_* left as they are         inert; the gate is gone
  search.tie_break_gamma = 0.0              inert under a continuous primary
                                            anyway; 0 is the quiet way to say
                                            so (smoke_test [6-H])

WHAT IS NOT SET
---------------
make_factorial_configs.py is NOT superseded: it stays for the SCREENING tier,
which is how the 52 cells were produced and is still the cheap way to ask which
regions are worth searching at all.

HPC note (hpc-python-compat): pure ASCII.
"""

import argparse
import copy
import json
import os
import sys


# the design document's search-space table (section 3.5)
RANGES = {
    "depth_exponent_range": [2, 5],
    "width_multiplier_range": [1.5, 5.0],
    "embedding_size_range": [8, 16],
    "lr_range": [1e-4, 0.2],
    "one_minus_beta1_range": [0.01, 0.1],
    "one_minus_beta2_range": [1e-4, 1e-2],
    "margin_range": [0.1, 1.0],
    "angular_alpha_deg_range": [2.0, 20.0],
    "lambda_sep_range": [1e-2, 20.0],
}
# weight_decay and dropout come from the REGULARIZATION block in the joint
# space (it takes the wider weight_decay range), so they are set there
REG_RANGES = {
    "weight_decay_range": [1e-5, 1e-2],
    "dropout_range": [0.0, 0.3],
}

# measured: 8.74 s/epoch at batches_per_epoch = 25, of which ~97% is training,
# so the cost is near-linear in the batch count
_SEC_PER_EPOCH_AT_25 = 8.74
_TRAIN_FRACTION = 0.97
# the screening ran, on average, 55% of its epoch cap before early stopping
_MEAN_EPOCH_FRACTION = 0.55


def seconds_per_epoch(batches_per_epoch):
    """Near-linear extrapolation of the MEASURED 8.74 s/epoch at 25 batches."""
    return _SEC_PER_EPOCH_AT_25 * (
        _TRAIN_FRACTION * float(batches_per_epoch) / 25.0
        + (1.0 - _TRAIN_FRACTION))


def build(base, args):
    cfg = copy.deepcopy(base)

    # ---- data: three label axes, non-collinear class centres (as the 52) ----
    cfg["data"]["latent"]["label_axes"] = [0, 1, 2]
    cfg["data"]["latent"]["class_center_mode"] = "simplex"

    # ---- the search ---------------------------------------------------------
    s = cfg["search"]
    s["search_mode"] = "joint_conditions"
    s.update(RANGES)
    s["block_family_choices"] = [0, 1]
    s["mining_strategy_choices"] = ["hard", "easy_positive",
                                    "easy_pos_semihard_neg"]
    s["loss_type_choices"] = ["triplet", "joint", "joint_sep"]
    s["strict_semihard_choices"] = [0, 1]
    s["head_fusion_choices"] = [0, 1]
    s["head_pool_ops_choices"] = [0, 1]
    s["sep_centre_means_choices"] = [0, 1]
    s["n_calls_joint"] = int(args.n_calls)
    s["n_initial_points_joint"] = int(args.n_initial_points)
    s["tie_break_gamma"] = 0.0
    s["gp_random_state"] = int(args.gp_random_state)
    cfg["regularization"].update(REG_RANGES)

    # ---- the trainer --------------------------------------------------------
    t = cfg["train"]
    t["n_seeds"] = int(args.n_seeds)
    t["batches_per_epoch"] = int(args.batches_per_epoch)
    t["max_epochs"] = int(args.max_epochs)
    t["patience"] = int(args.patience)
    t["sep_warmup_frac"] = float(args.warmup_frac)
    # the CLAMP CONSTANT for inactive trials -- must be the TrainConfig default
    t["lambda_sep"] = 0.1
    # searched, so the written value is only the clamp for non-joint_sep trials
    t["sep_centre_means"] = None
    t["selection_primary"] = "silhouette"
    t["min_delta_sil_mode"] = "floor_scale"
    t["min_delta_sil_kappa"] = 2.0
    t["sil_floor_permutations"] = 200

    cfg["runtime"]["experiment_name"] = str(args.name)
    return cfg


def report(cfg, args):
    t = cfg["train"]
    nb = int(t["batches_per_epoch"])
    E = int(t["max_epochs"])
    n_calls = int(cfg["search"]["n_calls_joint"])
    ns = int(t["n_seeds"])
    tau = float(t["sep_warmup_frac"])

    spe = seconds_per_epoch(nb)
    mean_epochs = _MEAN_EPOCH_FRACTION * E
    hours = n_calls * ns * mean_epochs * spe / 3600.0
    T = E * nb

    out = []
    out.append("BUDGET")
    out.append("  n_calls = %d x n_seeds = %d = %d training runs"
               % (n_calls, ns, n_calls * ns))
    out.append("  T = max_epochs x batches_per_epoch = %d x %d = %d steps"
               % (E, nb, T))
    out.append("    (the screening's maximum was 1500 steps -> %.1fx)"
               % (T / 1500.0))
    out.append("  warm-up: tau = %.3g -> full lambda_sep at step %d, i.e. "
               "epoch %.1f of %d" % (tau, int(tau * T), tau * E, E))
    if float(t["patience"]) / float(E) < tau:
        out.append("    WARNING: patience/max_epochs = %.2f < tau = %.2f, so a "
                   "run stopping at its patience floor never reaches full "
                   "lambda_sep." % (float(t["patience"]) / float(E), tau))
    out.append("")
    out.append("WALL CLOCK (depth-4 rate, size mix UNMEASURED)")
    out.append("  %.1f s/epoch at %d batches (extrapolated from a MEASURED "
               "8.74 s at 25)" % (spe, nb))
    out.append("  mean %.0f epochs/run (the screening averaged %.0f%% of its "
               "cap)" % (mean_epochs, 100 * _MEAN_EPOCH_FRACTION))
    out.append("  -> %.0f h at the depth-4 rate, against a %d h walltime"
               % (hours, args.walltime))
    if hours > args.walltime:
        out.append("  DOES NOT FIT. gp_minimize is SEQUENTIAL and cannot be "
                   "split across lanes.")
        out.append("  Try --max-epochs 60 --patience 20 (still %d steps, %.1fx "
                   "the screening cap)" % (60 * nb, 60 * nb / 1500.0))
    else:
        out.append("  fits at the depth-4 rate, with %.0f h of margin -- BUT "
                   "the architecture size mix is not in this number."
                   % (args.walltime - hours))
    out.append("  STAGE 7 MUST MEASURE THE SIZE MIX BEFORE SUBMISSION.")
    out.append("")
    out.append("SINGLE-SEED NOISE (n_seeds = %d)" % ns)
    if ns == 1:
        out.append("  MEASURED within-cell seed sd s = 0.073; between-cell sd "
                   "0.117. The probability a single seed misranks two configs")
        out.append("  differing by d is Phi(-d / (s*sqrt(2))): d = 0.05 -> 31%, "
                   "0.10 -> 17%, 0.20 -> 2.6%.")
        out.append("  Coarse structure is recoverable; FINE RANKING IS NOT. The "
                   "reported winner is a top-10 candidate, and the Stage 8")
        out.append("  confirmatory re-fit of the top 5 at 5 seeds is part of "
                   "the plan, not optional.")
    out.append("")
    out.append("WHAT THIS SEARCH DOES NOT ANSWER")
    out.append("  The GP allocates trials adaptively, so 'triplet' may receive "
               "very few. This returns a tuned configuration; it does NOT")
    out.append("  establish that the composite objective beats the triplet "
               "baseline at matched budget. That needs its own run.")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--base", required=True,
                    help="an existing config JSON used as the data/eval skeleton")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="l3c_joint_search")
    ap.add_argument("--n-calls", type=int, default=300)
    ap.add_argument("--n-initial-points", type=int, default=100,
                    help="pre-surrogate random draws. MEASURED by "
                         "search_dry_run.py: 40 draws leave 24 of the 52 "
                         "condition x head cells unvisited before the "
                         "surrogate takes over, 100 leaves 8. The legacy rule "
                         "caps at 10, which is far too thin in 22 columns.")
    ap.add_argument("--n-seeds", type=int, default=1)
    ap.add_argument("--batches-per-epoch", type=int, default=100)
    ap.add_argument("--max-epochs", type=int, default=100,
                    help="the STATED value. The design recommends 60 but "
                         "records that as not yet accepted.")
    ap.add_argument("--patience", type=int, default=40,
                    help="the STATED value. The design recommends 20.")
    ap.add_argument("--warmup-frac", type=float, default=0.3, help="tau")
    ap.add_argument("--gp-random-state", type=int, default=0)
    ap.add_argument("--walltime", type=float, default=144.0,
                    help="requested walltime in hours, for the arithmetic")
    ap.add_argument("--validate", action="store_true",
                    help="load the result through ExperimentConfig (needs torch)")
    args = ap.parse_args(argv)

    with open(args.base, "r", encoding="utf-8") as fh:
        base = json.load(fh)
    cfg = build(base, args)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    with open(args.out, "w", encoding="ascii") as fh:
        json.dump(cfg, fh, indent=2)
    print("wrote %s  (ONE config, replacing 52)" % args.out)
    print("")
    print(report(cfg, args))

    if args.validate:
        sys.path.insert(0, os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        from config import ExperimentConfig
        loaded = ExperimentConfig.from_json(args.out)
        assert loaded.search.search_mode == "joint_conditions"
        print("")
        print("VALIDATED: the file loads as an ExperimentConfig, "
              "search_mode = %r" % loaded.search.search_mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
