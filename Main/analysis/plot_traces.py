#!/usr/bin/env python3
"""
plot_traces.py -- DRIVER: synthetic latent config -> trace figures.

Orchestrates and saves. Generation is delegated to the repository's own
latent_burst_generator (build_latent_spec, LatentBurstProvider,
latent_ground_truth_table); drawing to dsn_trace_figures. This file contains
no signal synthesis and no matplotlib calls beyond savefig.

    python3 plot_traces.py --main-dir "$PWD" \
        --config hpc/Config/config_l3c_joint_full_lane0.json \
        --out ~/trace_figs

NO TRAINING, NO CHECKPOINT NEEDED. This reads only the DATA section of a
config, so it works before any model exists -- unlike plot_embeddings.py,
which needs trained weights. It is cheap: generating a few dozen traces takes
seconds to a couple of minutes, so it is fine on a login node.

--n-per-class limits how many cultures are synthesised PER CLASS (default 5),
independent of what the config's synthetic_n_per_class says, so a quick look
does not pay for the full 45-culture dataset.

Outputs under <out>/:
    T1_trace_gallery, T2_class_overlay, T3_latent_space,
    T4_window_view, T5_summary_stats            (png + pdf)
    latents.csv     the phi coordinate of every synthesised culture
    spec.json       the resolved latent spec, for the record

Pure ASCII, headless (hpc-python-compat).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import dsn_trace_figures as TF


def _ensure_repo(main_dir):
    main_dir = os.path.abspath(main_dir)
    if not os.path.isfile(os.path.join(main_dir, "config.py")):
        raise SystemExit("not a repository Main/ directory (no config.py): %s"
                         % main_dir)
    if main_dir not in sys.path:
        sys.path.insert(0, main_dir)
    return main_dir


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
    ap.add_argument("--main-dir", required=True)
    ap.add_argument("--config", required=True,
                    help="any config JSON with data.data_mode = latent")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per-class", type=int, default=5,
                    help="cultures synthesised per class (default 5)")
    ap.add_argument("--overlay-t-max", type=float, default=None,
                    help="seconds to show in T2 (default: whole trace)")
    args = ap.parse_args()

    main_dir = _ensure_repo(args.main_dir)
    from config import ExperimentConfig
    import latent_burst_generator as LBG
    import run_optimization as R

    with open(args.config, "r") as fh:
        cfg = ExperimentConfig.from_dict(json.load(fh))
    if str(cfg.data.data_mode) != "latent":
        raise SystemExit("config has data_mode=%r; this tool needs 'latent'"
                         % cfg.data.data_mode)

    # Directive 2: the repo already owns the cfg -> LatentSpec adapter
    # (run_optimization.latent_spec_from_config). Re-unpacking cfg here would
    # be a second, silently-diverging copy of that mapping.
    spec = R.latent_spec_from_config(cfg)
    C = int(spec.n_classes)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "spec.json"), "w") as fh:
        json.dump(LBG.latent_ground_truth_table(spec), fh, indent=2,
                  default=str)

    provider = LBG.LatentBurstProvider(spec)
    traces, labels, phis, fs = [], [], [], None
    print("synthesising %d culture(s) per class, C = %d ..."
          % (args.n_per_class, C))
    for c in range(C):
        for r in range(int(args.n_per_class)):
            x, fs_i = provider(c, r)
            traces.append(x)
            labels.append(c)
            phis.append(provider.latents[(c, r)])
            fs = fs_i
    phis = np.asarray(phis, dtype=float)
    labels = np.asarray(labels, dtype=int)
    print("  %d trace(s), %d samples each, f_s = %.1f Hz"
          % (len(traces), traces[0].size, fs))

    names = [a.name for a in spec.axes]
    with open(os.path.join(args.out, "latents.csv"), "w") as fh:
        fh.write("culture_index,phenotype," + ",".join(names) + "\n")
        for i in range(phis.shape[0]):
            fh.write("%d,%d," % (i, labels[i]) +
                     ",".join("%.6f" % v for v in phis[i]) + "\n")

    centers = getattr(spec, "class_centers", None)
    if centers is None:
        try:
            centers = LBG._class_center_vectors(
                C, len(spec.label_axes), str(spec.class_center_mode))
            full = np.full((C, len(names)), 0.5)
            for j, k in enumerate(spec.label_axes):
                full[:, k] = centers[:, j]
            centers = full
        except Exception:
            centers = None

    made = []
    made.append(save(TF.fig_trace_gallery(traces, labels, fs, n_per_class=1),
                     args.out, "T1_trace_gallery"))
    made.append(save(TF.fig_class_overlay(traces, labels, fs,
                                          n_per_class=args.n_per_class,
                                          t_max_s=args.overlay_t_max),
                     args.out, "T2_class_overlay"))
    made.append(save(TF.fig_latent_space(phis, labels, names,
                                         list(spec.label_axes), centers),
                     args.out, "T3_latent_space"))
    made.append(save(TF.fig_window_view(traces[0], labels[0], fs,
                                        float(cfg.data.window_s),
                                        float(cfg.data.eval_stride_s),
                                        culture_id=0),
                     args.out, "T4_window_view"))
    made.append(save(TF.fig_summary_stats(traces, labels, fs),
                     args.out, "T5_summary_stats"))

    print("\nwrote %d figure(s) to %s" % (len([m for m in made if m]), args.out))
    for m in made:
        if m:
            print("  %s.png / .pdf" % m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
