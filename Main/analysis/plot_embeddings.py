#!/usr/bin/env python3
"""
plot_embeddings.py -- DRIVER: checkpoint(s) -> embedding figures + metrics.

Separation of concerns (directive 2): orchestrates and saves. Extraction lives
in dsn_embed, drawing in dsn_embed_figures. This file contains no scoring
mathematics and no matplotlib calls beyond savefig.

WHAT THIS NEEDS THAT YOU MAY NOT HAVE YET
An embedding requires TRAINED WEIGHTS. The search writes checkpoints only during
the FINAL training (run_final), so a lane killed at the walltime has
trials.jsonl but no checkpoints/. If --discover finds nothing, this script says
so and prints the exact command that produces one, rather than failing obscurely.

USAGE

  # 1. point it at one checkpoint
  python3 plot_embeddings.py \
      --main-dir "$HOME/Deep Summary Network/Deep_multich/Main" \
      --checkpoint out/refit_top1/checkpoints/seed_0/best.pt \
      --out ~/emb_top1

  # 2. or let it find every checkpoint under a run directory
  python3 plot_embeddings.py \
      --main-dir "$HOME/Deep Summary Network/Deep_multich/Main" \
      --discover out/refit_top1 \
      --out ~/emb_top1

  # 3. list what it would use, touching no data
  python3 plot_embeddings.py --main-dir ... --discover out/refit_top1 --dry-run

Outputs per checkpoint, under <out>/<tag>/:
    E1_embedding_pca, E2_split_comparison, E3_per_class_silhouette,
    E4_similarity_heatmap, E5_culture_structure, E6_embedding_health  (png+pdf)
    metrics.json     the numbers behind every panel
Plus <out>/EMBEDDING_SUMMARY.csv across all checkpoints.

COST. Embedding is forward-pass only: no training, no optimizer, no augmentation.
The expensive step is build_traces, and it reads the existing cache
(runtime.cache_dir) rather than regenerating. Minutes, not hours -- but it does
load the full dataset, so run it in a job or a short interactive allocation, not
on the login node.

Pure ASCII, headless (hpc-python-compat).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import dsn_embed as E
import dsn_embed_figures as EF


def discover_checkpoints(root, prefer=("best.pt", "final_seed", "best_model.pt")):
    """Every checkpoint under root, newest-relevant first, de-duplicated by dir.

    Prefers best.pt over last.pt inside the same seed directory: last.pt is the
    final epoch, best.pt is the epoch the score was actually taken at, and those
    differ whenever early stopping fired -- which is nearly always here.
    """
    hits = []
    for dirpath, _dirs, files in os.walk(root):
        pts = sorted(f for f in files if f.endswith(".pt"))
        if not pts:
            continue
        chosen = None
        for want in ("best.pt", "best_model.pt"):
            if want in pts:
                chosen = want
                break
        if chosen is None:
            finals = [f for f in pts if f.startswith("final_seed")]
            chosen = finals[0] if finals else pts[0]
        hits.append(os.path.join(dirpath, chosen))
    return sorted(hits)


def tag_for(path, root=None):
    """A short, filesystem-safe label identifying this checkpoint."""
    p = os.path.normpath(path)
    parts = [q for q in p.split(os.sep) if q not in ("", ".", "checkpoints")]
    keep = parts[-3:] if len(parts) >= 3 else parts
    t = "_".join(keep).replace(".pt", "")
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in t)


def save(fig, out_dir, name):
    if fig is None:
        return None
    import matplotlib.pyplot as plt
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, "%s.%s" % (name, ext)), dpi=150,
                    bbox_inches="tight")
    plt.close(fig)
    return name


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main-dir", required=True,
                    help="the repository's Main/ directory")
    ap.add_argument("--checkpoint", action="append", default=[],
                    help="a .pt checkpoint; repeatable")
    ap.add_argument("--discover",
                    help="walk this directory and use every checkpoint found")
    ap.add_argument("--out", help="output directory (required unless --dry-run)")
    ap.add_argument("--splits", default="train,val,test",
                    help="comma-separated subset of train,val,test")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--config-override",
                    help="config JSON whose runtime.cache_dir / out_dir replace "
                         "the checkpoint's. For RELOCATION ONLY -- it warns.")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the checkpoints and exit, touching no data")
    args = ap.parse_args()

    main_dir = os.path.abspath(args.main_dir)
    cks = list(args.checkpoint)
    if args.discover:
        d = args.discover if os.path.isabs(args.discover) \
            else os.path.join(main_dir, args.discover)
        cks += discover_checkpoints(d)
    cks = sorted(set(cks))

    if not cks:
        where = args.discover or "(none given)"
        sys.stderr.write(
            "No checkpoint found under %s.\n\n"
            "Embeddings need trained weights, and the SEARCH does not write "
            "them: checkpoints are produced only by the FINAL training. A lane "
            "killed at the walltime has trials.jsonl and no checkpoints/.\n\n"
            "To produce one, rebuild a config from the trial log and train it:\n"
            "  cd %s\n"
            "  python3 best_from_trials.py --run-dir out/l3c_joint_full_lane0 "
            "--top-k 5\n"
            "  # edit train.n_seeds -> 5 and runtime.experiment_name -> "
            "refit_lane0_top1\n"
            "  python3 run_optimization.py --config "
            "out/l3c_joint_full_lane0/config_top1.json --skip-search --verbose\n"
            "  # then re-run this script with --discover out/refit_lane0_top1\n"
            % (where, main_dir))
        return 2

    print("checkpoints (%d):" % len(cks))
    for c in cks:
        print("  %s   -> tag %s" % (c, tag_for(c)))
    if args.dry_run:
        print("\n--dry-run: nothing loaded, nothing written.")
        return 0
    if not args.out:
        sys.stderr.write("--out is required unless --dry-run\n")
        return 2

    which = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    for s in which:
        if s not in E.SPLIT_NAMES:
            sys.stderr.write("bad split %r; choose from %s\n"
                             % (s, ",".join(E.SPLIT_NAMES)))
            return 2

    os.makedirs(args.out, exist_ok=True)
    rows = []
    splits_cache = {}

    for ck in cks:
        tag = tag_for(ck)
        odir = os.path.join(args.out, tag)
        os.makedirs(odir, exist_ok=True)
        print("\n=== %s ===" % tag, flush=True)

        model, cfg, ckpt = E.load_model_and_config(
            ck, main_dir, map_location="cpu",
            config_override=args.config_override)

        # The splits depend only on the config, so build them ONCE per distinct
        # data configuration rather than per checkpoint: for a re-fit's seeds
        # that turns N rebuilds into one.
        key = json.dumps({"data": cfg.data.__dict__.get("__dict__", None)
                          or str(cfg.data), "seed": cfg.runtime.seed,
                          "cache": cfg.runtime.cache_dir}, default=str,
                         sort_keys=True)
        if key not in splits_cache:
            splits_cache[key] = E.build_split_bundle(cfg, main_dir, verbose=True)
        splits, fs = splits_cache[key]

        embs = E.embed_all_splits(model, splits, cfg, device=args.device,
                                  tag=tag, main_dir=main_dir, which=which,
                                  batch_size=args.batch_size)
        for nm, e in embs.items():
            print("  %-6s %r" % (nm, e))

        made = []
        made.append(save(EF.fig_embedding_pca(embs), odir, "E1_embedding_pca"))
        made.append(save(EF.fig_split_comparison(embs), odir,
                         "E2_split_comparison"))
        made.append(save(EF.fig_per_class_silhouette(embs), odir,
                         "E3_per_class_silhouette"))
        made.append(save(EF.fig_similarity_heatmap(embs), odir,
                         "E4_similarity_heatmap"))
        c5 = save(EF.fig_culture_structure(embs), odir, "E5_culture_structure")
        if c5 is None:
            print("  [E5 skipped: no per-window culture map on these datasets]")
        made.append(c5)
        made.append(save(EF.fig_embedding_health(embs), odir,
                         "E6_embedding_health"))

        payload = {"checkpoint": ck, "tag": tag,
                   "epoch": ckpt.get("epoch"), "fs": float(fs),
                   "embedding_dim": int(next(iter(embs.values())).dim),
                   "figures": [m for m in made if m],
                   "splits": dict(
                       (nm, {"n": e.n, "metrics": e.metrics,
                             "health": e.health,
                             "n_cultures": (None if e.culture_ids is None
                                            else int(len(e.culture_ids)))})
                       for nm, e in embs.items())}
        json.dump(payload, open(os.path.join(odir, "metrics.json"), "w"),
                  indent=2, sort_keys=True, default=float)

        for nm, e in embs.items():
            rows.append({"tag": tag, "split": nm, "n": e.n, "dim": e.dim,
                         "ari": e.metrics.get("ari"),
                         "ami": e.metrics.get("ami"),
                         "silhouette": e.metrics.get("silhouette"),
                         "eff_rank": e.health.get("eff_rank"),
                         "min_std": e.health.get("min_std"),
                         "checkpoint": ck})

    import pandas as pd
    df = pd.DataFrame(rows)
    csv = os.path.join(args.out, "EMBEDDING_SUMMARY.csv")
    df.to_csv(csv, index=False)
    print("\n%s" % csv)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
