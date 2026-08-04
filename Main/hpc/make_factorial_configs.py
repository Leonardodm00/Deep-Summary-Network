"""
make_factorial_configs.py
=========================

Emit the full loss x mining x filter x head factorial as config JSONs.

    python3 make_factorial_configs.py --base hpc/Config/config_l3c_h_multimean_005.json \\
                                      --out-dir hpc/Config/factorial \\
                                      --tier screen

THE GRID
--------
  switch 1  mining_strategy : hard | easy_positive | easy_pos_semihard_neg
  switch 2  loss_type       : triplet | joint | joint_sep
  switch 3  strict_semihard : on | off   (meaningless for "triplet")
  switch 4  head geometry   : head_fusion in {False, True}
                              head_pool_ops in {("mean",), ("mean","max","std")}

  3 x 5 = 15 loss/mining/filter cells MINUS 2 that are provably empty
  = 13, times 4 head geometries = 52 configurations.

THE TWO EXCLUDED CELLS
----------------------
mining_strategy="hard" with strict_semihard=True, under loss_type joint or
joint_sep. TripletMarginMiner(type_of_triplets="hard") returns only triplets
with D_an < D_ap; the strict semi-hard filter keeps only D_ap < D_an. The
intersection is empty by construction -- measured on a real batch: 16814 mined,
0 surviving -- so train_loss stays exactly 0.0 forever and the run looks stable
rather than broken. TrainConfig raises on it; this generator skips it.

TIERS
-----
  --tier screen  (default)  n_calls 0, --skip-search at run time, n_seeds 3.
                            Fixed hyper-parameters, so every cell trains ONE
                            configuration. This is the tier that makes 52 cells
                            affordable: roughly 1/140th of a full search each.
                            Use it to find which cells are even worth searching.
  --tier full               the archived budget (60 + 60 + 20 trials, n_seeds
                            2). Only run this on cells that survived screening.

WHAT IS CHANGED FROM THE BASE CONFIG
------------------------------------
  data.latent.label_axes        -> [0, 1, 2]   (three label axes)
  data.latent.class_center_mode -> "simplex"   (non-collinear class centres)
  search.depth_exponent_range   -> [2, 6]      (see the note below)
  search.width_multiplier_range -> [1.5, 5.0]
  train.selection_primary       -> "silhouette", floor-calibrated
  search.tie_break_gamma        -> 0.0         (inert under a continuous primary)

DEPTH 7 IS DELIBERATELY NOT INCLUDED. Measured parameter counts on this
backbone: depth 5 -> 18-20 M, depth 6 -> 139-250 M (about 3.0 GB of weights
plus AdamW state at width 5.0), depth 7 -> the process is OOM-killed before the
model finishes building. Extrapolating the x8-per-step trend puts depth 7 near
1-2 billion parameters for a 3-class benchmark with a few hundred training
windows. Pass --max-depth 7 to override, knowingly.

HPC note (hpc-python-compat): pure ASCII.
"""

import argparse
import copy
import json
import os
import sys

MINING = ("hard", "easy_positive", "easy_pos_semihard_neg")
LOSS = ("triplet", "joint", "joint_sep")
HEADS = (
    ("single", "mean", False, ["mean"]),
    ("single", "all", False, ["mean", "max", "std"]),
    ("multi", "mean", True, ["mean"]),
    ("multi", "all", True, ["mean", "max", "std"]),
)
_MINING_TAG = {"hard": "h", "easy_positive": "ep",
               "easy_pos_semihard_neg": "epsh"}
_LOSS_TAG = {"triplet": "trip", "joint": "joint", "joint_sep": "jsep"}


def grid():
    """Every valid (mining, loss, filter, head) cell, with the empty ones dropped."""
    out = []
    for m in MINING:
        for l in LOSS:
            filters = (None,) if l == "triplet" else (False, True)
            for f in filters:
                if l != "triplet" and f and m == "hard":
                    continue                      # provably empty; see header
                for head_h, head_p, fusion, ops in HEADS:
                    out.append({
                        "mining": m, "loss": l, "filter": f,
                        "head_fusion": fusion, "head_pool_ops": ops,
                        "name": "%s_%s%s_%s%s" % (
                            _MINING_TAG[m], _LOSS_TAG[l],
                            "" if f is None else ("_filtON" if f else "_filtOFF"),
                            head_h, head_p),
                    })
    return out


def build(base, cell, tier, max_depth, width_hi):
    cfg = copy.deepcopy(base)

    # ---- data: three label axes, non-collinear class centres ----
    cfg["data"]["latent"]["label_axes"] = [0, 1, 2]
    cfg["data"]["latent"]["class_center_mode"] = "simplex"

    # ---- backbone: the head geometry factor ----
    cfg["backbone"]["head_fusion"] = bool(cell["head_fusion"])
    cfg["backbone"]["head_pool_ops"] = list(cell["head_pool_ops"])

    # ---- the objective factors ----
    t = cfg["train"]
    t["mining_strategy"] = cell["mining"]
    t["loss_type"] = cell["loss"]
    t["strict_semihard"] = bool(cell["filter"]) if cell["filter"] is not None else False
    t["swap"] = True
    t["margin"] = 0.3
    t["angular_alpha_deg"] = 12.0
    # lambda_sep and the gate are only built for joint_sep. Setting them on the
    # other 32 cells would be inert AND would fire the "INERT lambda_sep"
    # warning on every one of them, training the reader to ignore warnings.
    if cell["loss"] == "joint_sep":
        t["lambda_sep"] = 0.05
        t["sep_gate_threshold"] = 0.20
        t["sep_gate_momentum"] = 0.05
        t["sep_gate_min_batches"] = 20

    # ---- selection: silhouette primary, floor-calibrated ----
    t["selection_primary"] = "silhouette"
    t["min_delta_sil_mode"] = "floor_scale"
    t["min_delta_sil_kappa"] = 2.0
    t["sil_floor_permutations"] = 200
    cfg["search"]["tie_break_gamma"] = 0.0        # inert under a continuous primary

    # ---- search space ----
    s = cfg["search"]
    s["depth_exponent_range"] = [2, int(max_depth)]
    s["width_multiplier_range"] = [1.5, float(width_hi)]
    s["angular_alpha_deg_range"] = [2.0, 20.0]
    s["lambda_sep_range"] = [0.001, 1.0]

    # ---- tier ----
    if tier == "screen":
        t["n_seeds"] = 3
        t["max_epochs"] = 60
        t["patience"] = 15
        # The screening tier is RUN WITH --skip-search, so these budgets are
        # never spent. They are still set to 1 so that a screening config
        # accidentally run WITHOUT --skip-search costs one trial rather than
        # 140. n_initial_points must drop to 0 (the legacy rule) or config
        # validation refuses: 20 initial points cannot fit inside 1 call.
        s["n_calls_arch"] = 1
        s["n_calls_train"] = 1
        s["n_initial_points"] = 0
        if "regularization" in cfg:
            cfg["regularization"]["n_calls"] = 1
    else:
        t["n_seeds"] = 2
        t["max_epochs"] = 100
        t["patience"] = 15
        s["n_calls_arch"] = 60
        s["n_calls_train"] = 60
        if "regularization" in cfg:
            cfg["regularization"]["n_calls"] = 20

    cfg["runtime"]["experiment_name"] = "l3c_%s_%s" % (tier, cell["name"])
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--base", required=True,
                    help="an existing l3c config JSON to inherit data/runtime from")
    ap.add_argument("--out-dir", default="hpc/Config/factorial")
    ap.add_argument("--tier", choices=("screen", "full"), default="screen")
    ap.add_argument("--max-depth", type=int, default=6,
                    help="depth_exponent_range high. 7 is OOM on this backbone.")
    ap.add_argument("--width-hi", type=float, default=5.0)
    ap.add_argument("--list", action="store_true",
                    help="print the grid and write nothing")
    a = ap.parse_args(argv)

    cells = grid()
    if a.list:
        print("%d configurations" % len(cells))
        for c in cells:
            print("  %-40s %-22s %-9s filter=%s" % (
                c["name"], c["mining"], c["loss"],
                "-" if c["filter"] is None else c["filter"]))
        return 0

    if a.max_depth >= 7:
        sys.stderr.write(
            "WARNING: --max-depth %d. Depth 7 could not be CONSTRUCTED in a\n"
            "  multi-GB container (OOM-killed); depth 6 already reaches 139-250 M\n"
            "  parameters. Proceeding because you asked explicitly.\n" % a.max_depth)

    with open(a.base) as f:
        base = json.load(f)

    os.makedirs(a.out_dir, exist_ok=True)
    written = []
    for c in cells:
        cfg = build(base, c, a.tier, a.max_depth, a.width_hi)
        path = os.path.join(a.out_dir, "config_l3c_%s_%s.json" % (a.tier, c["name"]))
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
        written.append(path)

    print("tier=%s  wrote %d configs -> %s" % (a.tier, len(written), a.out_dir))
    print("")
    print("  by mining strategy:")
    for m in MINING:
        n = sum(1 for c in cells if c["mining"] == m)
        print("    %-22s %d" % (m, n))
    print("  excluded: hard x {joint, joint_sep} x strict_semihard=True")
    print("            (empty by construction -- see the module header)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
