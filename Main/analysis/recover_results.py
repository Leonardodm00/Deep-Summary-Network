#!/usr/bin/env python3
"""
recover_results.py -- rebuild results.json from completed per-seed checkpoints.

WHY. run_final writes checkpoints/final_seed_<n>.pt (config + full epoch history
+ that seed's TEST metrics) immediately after each seed finishes, but assembles
results.json only after ALL seeds are done. A walltime kill therefore leaves
every completed seed's data on disk with no aggregate to read it from.

This reads whatever final_seed_*.pt files exist and writes the same structure
run_final would have written, over however many seeds actually completed. The
aggregate is honest about that: n_seeds is the number RECOVERED, and the output
is marked `recovered: true` so it can never be mistaken for a full run.

It does NOT retrain, does not re-evaluate, and does not touch the checkpoints.
Every number comes from what the training run itself already computed and
stored, so a recovered results.json agrees with a complete one seed-for-seed.

USAGE
  python3 recover_results.py --run-dir out/refit_mea_r1_l2_t65
  python3 recover_results.py --run-dir out/... --out results_partial.json
  python3 recover_results.py --run-dir out/... --dry-run

Pure ASCII (hpc-python-compat).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np


def _ensure_repo(main_dir):
    main_dir = os.path.abspath(main_dir)
    if not os.path.isfile(os.path.join(main_dir, "config.py")):
        raise SystemExit("not a repository Main/ (no config.py): %s" % main_dir)
    if main_dir not in sys.path:
        sys.path.insert(0, main_dir)
    return main_dir


def _agg(values):
    """mean / std / values over seeds. Population std, matching run_optimization
    ._agg exactly so a recovered file is comparable with a complete one."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "values": [float(v) for v in values]}
    return {"mean": float(arr.mean()), "std": float(arr.std()),
            "values": [float(v) for v in values]}


def _best_finite(history, key):
    vals = [h.get(key) for h in history if h.get(key) is not None
            and np.isfinite(h.get(key))]
    return float(max(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--main-dir", default=".")
    ap.add_argument("--out", default=None,
                    help="output path (default: <run-dir>/results_recovered.json)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    _ensure_repo(args.main_dir)
    import torch

    run_dir = os.path.abspath(args.run_dir)
    ckpts = sorted(glob.glob(os.path.join(run_dir, "checkpoints",
                                          "final_seed_*.pt")),
                   key=lambda p: int(re.search(r"final_seed_(\d+)", p).group(1)))
    if not ckpts:
        raise SystemExit(
            "no checkpoints/final_seed_*.pt in %s.\n"
            "  Nothing completed yet: a seed writes its checkpoint only after it\n"
            "  finishes training AND evaluating." % run_dir)

    complete = os.path.join(run_dir, "results.json")
    if os.path.exists(complete):
        print("NOTE: %s already exists -- the run finished. This recovery is "
              "probably unnecessary." % complete)

    per_seed, cfg_dict = [], None
    for p in ckpts:
        n = int(re.search(r"final_seed_(\d+)", p).group(1))
        ck = torch.load(p, map_location="cpu", weights_only=False)
        extra = ck.get("extra") or {}
        test = extra.get("test") or {}
        hist = extra.get("history") or []
        best = ck.get("best_metric") or {}
        if cfg_dict is None:
            cfg_dict = ck.get("config")

        row = {
            "seed": extra.get("seed"),
            "seed_index": n,
            "epochs_run": int(ck.get("epoch") or len(hist)),
            "best_val_ari": float(best.get("val_ari", _best_finite(hist, "ari"))),
            "best_val_silhouette": float(
                best.get("val_silhouette", _best_finite(hist, "silhouette"))),
            "test_ari": float(test.get("ari", float("nan"))),
            "test_ami": float(test.get("ami", float("nan"))),
            "test_silhouette": float(test.get("silhouette", float("nan"))),
            "test_n_windows": test.get("n_windows"),
            "checkpoint": p,
            "figure": os.path.join(run_dir, "figures",
                                   "embedding_test_seed_%d.png" % n),
        }
        # eff_rank is not always in the checkpoint's test dict (it lives under
        # ev["health"] at write time). Report it only if present; never invent it.
        if "eff_rank" in test:
            row["test_eff_rank"] = float(test["eff_rank"])
        per_seed.append(row)
        print("  seed %d: epochs=%s test_ari=%.4f test_sil=%.4f  %s"
              % (n, row["epochs_run"], row["test_ari"],
                 row["test_silhouette"], os.path.basename(p)))

    n_rec = len(per_seed)
    expected = None
    if cfg_dict:
        try:
            expected = int(cfg_dict["train"]["n_seeds"])
        except (KeyError, TypeError, ValueError):
            expected = None

    out = {
        "experiment": (cfg_dict or {}).get("runtime", {}).get("experiment_name"),
        "recovered": True,
        "recovered_note": (
            "Rebuilt by recover_results.py from per-seed checkpoints after an "
            "incomplete run. n_seeds below is the number of seeds that COMPLETED, "
            "not the number requested. Every value is read from what the training "
            "run itself computed; nothing was retrained or re-evaluated."),
        "n_seeds_requested": expected,
        "n_seeds_recovered": n_rec,
        "config_best": cfg_dict,
        "test": {
            "ari": _agg([r["test_ari"] for r in per_seed]),
            "ami": _agg([r["test_ami"] for r in per_seed]),
            "silhouette": _agg([r["test_silhouette"] for r in per_seed]),
            "n_seeds": n_rec,
            "per_seed": per_seed,
        },
    }
    if all("test_eff_rank" in r for r in per_seed):
        out["test"]["eff_rank"] = _agg([r["test_eff_rank"] for r in per_seed])

    t = out["test"]
    print("\nrecovered %d seed(s)%s"
          % (n_rec, "" if expected is None else " of %d requested" % expected))
    print("  TEST ari        : %.4f +/- %.4f" % (t["ari"]["mean"], t["ari"]["std"]))
    print("  TEST ami        : %.4f +/- %.4f" % (t["ami"]["mean"], t["ami"]["std"]))
    print("  TEST silhouette : %.4f +/- %.4f"
          % (t["silhouette"]["mean"], t["silhouette"]["std"]))
    if n_rec == 1:
        print("  (std is 0.0 BY CONSTRUCTION with one seed, not zero uncertainty)")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    dst = args.out or os.path.join(run_dir, "results_recovered.json")
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("\nwrote %s" % dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())