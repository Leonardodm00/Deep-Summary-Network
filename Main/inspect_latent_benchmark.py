"""
inspect_latent_benchmark.py
===========================

Generate the latent-factor benchmark for a given config and LOOK at it: the
synthesized IFR traces, class by class, and the latent coordinates that produced
them.

Why this exists
---------------
For data_mode == "synthetic" the driver writes a traces figure next to the run
(run_optimization.save_synthetic_artifacts). For data_mode == "latent" it writes
only latent_ground_truth.json, because that table is what the factor-retention
analysis needs and signal synthesis is not free. That leaves no way to EYEBALL
the benchmark before spending cluster hours on it, which is what this script is
for. It is an INSPECTION tool, deliberately outside the pipeline: it trains
nothing, scores nothing, and writes nothing the pipeline reads back.

Separation of concerns (directive 2)
------------------------------------
  section 1 : config -> LatentSpec        (no synthesis, no plotting)
  section 2 : synthesis                   (no plotting)
  section 3 : reporting + figures         (no synthesis)
Signal generation is delegated entirely to latent_burst_generator, which is
itself a reparametrization of generate_burst_data. Nothing is reimplemented.

NO TORCH REQUIRED. It reads the config JSON directly rather than through
config.py, whose import chain pulls in backbone.py -> torch. That makes it
runnable on a login node, or on any machine, without the training environment.
The cost is that the config -> LatentSpec unpacking is written out here as well
as in run_optimization.latent_spec_from_config; the two must agree, which the
smoke test asserts field by field.

Notation (symbols introduced at first use; carried in full)
-----------------------------------------------------------
    n         : number of latent factors, n in N, n >= 1
    k         : latent axis index, 0-based in code (k in {0, ..., n-1})
    phi       : latent coordinate vector, phi in [0, 1]^n
    phi_k     : k-th latent coordinate, phi_k in [0, 1]
    C         : number of phenotype classes, class label c in {0, ..., C-1}
    n_c       : number of traces in class c, n_c in N, n_c >= 1
    r         : trace index within a class, r in {0, ..., n_c - 1}
    S         : label-carrying axis subset (the axes that determine the class)
    S^c       : the label-IRRELEVANT ("free") axes
    tau       : class overlap, tau >= 0, in normalized latent units
    x         : synthesized IFR trace, x in R_{>=0}^{K}
    K         : trace length in samples, K = round(T_rec * f_s)
    f_s       : IFR sampling rate [Hz]
    T_rec     : trace duration [s]

Usage
-----
    python3 inspect_latent_benchmark.py --config hpc/config_latent_3class_hard.json \
                                        --out-dir latent_inspect

    # or with no config at all, using the LatentSpec defaults (C = 3, n_c = 3):
    python3 inspect_latent_benchmark.py --out-dir latent_inspect

    # a quicker look: shorter traces, fewer neurons
    python3 inspect_latent_benchmark.py --duration-s 120 --n-neurons 40 \
                                        --out-dir latent_inspect

Outputs (all under --out-dir)
-----------------------------
    latent_traces.png       one row per class; every trace of that class overlaid
    latent_factors.png      phi scatter: label axes (separate) vs free axes (do not)
    latent_inspection.json  the per-trace latent + physical parameter table
    a printed table of phi and the physical values each phi maps to

HPC note (hpc-python-compat): pure ASCII. Its one local import,
latent_burst_generator, was byte-verified pure ASCII as well (Rule 6).
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")            # headless: must precede pyplot (HPC / no display)
import matplotlib.pyplot as plt  # noqa: E402

from latent_burst_generator import (        # noqa: E402
    DEFAULT_AXIS_NAMES,
    LatentBurstProvider,
    build_latent_spec,
    latent_ground_truth_table,
)

__all__ = [
    "latent_spec_from_config_dict",
    "synthesize_dataset",
    "plot_traces_by_class",
    "plot_latent_factors",
    "format_latent_table",
]


# --------------------------------------------------------------------------- #
# section 1 -- config -> LatentSpec (no synthesis, no plotting)
# --------------------------------------------------------------------------- #
def latent_spec_from_config_dict(cfg: Dict[str, object],
                                 duration_s: float = None,
                                 n_neurons: int = None,
                                 seed: int = None,
                                 n_per_class: Sequence[int] = None,
                                 class_overlap: float = None):
    """Build a LatentSpec from a parsed config JSON dict.

    MUST agree field-for-field with run_optimization.latent_spec_from_config.
    The duplication is deliberate (it buys torch-independence, see the module
    docstring) and is asserted by Smoke_Tests/smoke_test_inspect_latent.py.

    An absent "latent" block falls back to the LatentConfig defaults, and an
    absent "data" block falls back to the LatentSpec defaults, so the script is
    useful with no config file at all.

    Parameters
    ----------
    cfg        : the parsed config JSON (may be {}).
    duration_s : optional override of T_rec [s], for a quicker look.
    n_neurons  : optional override of the neuron count N.
    seed       : optional override of the base seed.
    """
    data = dict(cfg.get("data", {}) or {})
    lat = dict(data.get("latent", {}) or {})
    runtime = dict(cfg.get("runtime", {}) or {})

    n_per_class = tuple(int(v) for v in (
        n_per_class if n_per_class is not None
        else data.get("synthetic_n_per_class", (3, 3, 3))))
    fs = float(data.get("synthetic_fs", 50.0))
    T_rec = float(duration_s if duration_s is not None
                  else data.get("synthetic_duration_s", 600.0))
    base_seed = int(seed if seed is not None else runtime.get("seed", 0))
    n_neu = int(n_neurons if n_neurons is not None else lat.get("n_neurons", 100))

    return build_latent_spec(
        axis_names=tuple(lat.get("axis_names", DEFAULT_AXIS_NAMES)),
        label_axes=tuple(int(k) for k in lat.get("label_axes", (0, 1))),
        n_per_class=n_per_class,
        duration_s=T_rec,
        fs=fs,
        class_overlap=float(class_overlap if class_overlap is not None
                            else lat.get("class_overlap", 0.10)),
        class_center_mode=str(lat.get("class_center_mode", "interior")),
        n_neurons=n_neu,
        gaussian_window=float(lat.get("gaussian_window", 0.04)),
        seed=base_seed,
        axis_overrides=list(lat.get("axis_overrides", []) or []),
    )


# --------------------------------------------------------------------------- #
# section 2 -- synthesis (no plotting)
# --------------------------------------------------------------------------- #
def synthesize_dataset(spec) -> Tuple[List[np.ndarray], List[int], List[int],
                                      np.ndarray, float]:
    """Generate every trace the spec describes.

    Returns
    -------
    traces     : list of (K,) float32 IFR traces, in class-major order
    conditions : class label c of each trace
    trace_ids  : within-class index r of each trace
    Phi        : (n_traces, n) latent coordinates, Phi[t, k] = phi_k of trace t
    fs         : the sampling rate f_s [Hz] every trace shares
    """
    provider = LatentBurstProvider(spec)
    traces, conditions, trace_ids, phis = [], [], [], []
    fs_seen = None
    for c, n_c in enumerate(spec.n_per_class):
        for r in range(int(n_c)):
            x, fs = provider(c, r)
            if fs_seen is None:
                fs_seen = float(fs)
            elif abs(float(fs) - fs_seen) > 1e-9:
                raise RuntimeError("traces disagree on f_s: %r vs %r"
                                   % (fs, fs_seen))
            traces.append(np.asarray(x, dtype=np.float32))
            conditions.append(int(c))
            trace_ids.append(int(r))
            phis.append(provider.latents[(c, r)])
    return traces, conditions, trace_ids, np.asarray(phis, dtype=float), float(fs_seen)


# --------------------------------------------------------------------------- #
# section 3 -- reporting + figures (no synthesis)
# --------------------------------------------------------------------------- #
def format_latent_table(spec, conditions, trace_ids, Phi, table) -> str:
    """A printable per-trace table of phi and the physical values it maps to."""
    names = [a.name for a in spec.axes]
    label_set = set(int(k) for k in spec.label_axes)
    lines = []
    lines.append("latent coordinates phi (0 = low end of the axis, 1 = high end)")
    lines.append("  axes marked [S] carry the class label; the rest are FREE "
                 "(label-irrelevant)")
    header = "  %-5s %-4s" % ("class", "r")
    for k, nm in enumerate(names):
        header += " %14s" % (nm[:12] + ("[S]" if k in label_set else ""))
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for t, (c, r) in enumerate(zip(conditions, trace_ids)):
        row = "  %-5d %-4d" % (c, r)
        for k in range(Phi.shape[1]):
            row += " %14.4f" % Phi[t, k]
        lines.append(row)

    lines.append("")
    lines.append("physical values each phi maps to (units in parentheses)")
    units = {a.target: a.units for a in spec.axes}
    keys = ["lambda_b", "sigma_d", "median_duration_s", "lambda_burst",
            "participation_mean", "lambda_bg"]
    header = "  %-5s %-4s" % ("class", "r")
    for key in keys:
        header += " %14s" % key[:14]
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for row_d in table["rows"]:
        row = "  %-5d %-4d" % (int(row_d["condition"]), int(row_d["trace_id"]))
        for key in keys:
            row += " %14.4f" % float(row_d["physical"][key])
        lines.append(row)
    lines.append("")
    lines.append("  units: " + ", ".join(
        "%s (%s)" % (k, units.get(k, "?")) for k in keys if k in units))
    return "\n".join(lines)


def plot_traces_by_class(traces, conditions, fs, out_path, seconds=None,
                         dpi=130):
    """One row per class, every trace of that class overlaid.

    seconds : if given, plot only the first `seconds` of each trace, so bursts
              are actually resolvable rather than smeared into a solid band.
    """
    classes = sorted(set(int(c) for c in conditions))
    fig, axes = plt.subplots(len(classes), 1, figsize=(12, 2.6 * len(classes)),
                             sharex=True, sharey=True)
    if len(classes) == 1:
        axes = [axes]
    n_show = None if seconds is None else int(round(float(seconds) * float(fs)))

    for ax, c in zip(axes, classes):
        idx = [i for i, cc in enumerate(conditions) if int(cc) == c]
        for j, i in enumerate(idx):
            x = traces[i] if n_show is None else traces[i][:n_show]
            t = np.arange(x.shape[0]) / float(fs)
            ax.plot(t, x, lw=0.7, alpha=0.75, label="trace %d" % j)
        ax.set_ylabel("class %d\nIFR" % c)
        ax.legend(loc="upper right", fontsize=7, ncol=len(idx))
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("time (s)")
    ttl = "Latent benchmark: population IFR by phenotype class"
    if seconds is not None:
        ttl += "  (first %g s of each trace)" % float(seconds)
    axes[0].set_title(ttl)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_latent_factors(spec, conditions, Phi, out_path, dpi=130):
    """Scatter phi for a LABEL-axis pair and a FREE-axis pair, side by side.

    The point of the figure: the label axes separate the classes (that is what
    makes the task learnable at all), while the free axes do NOT (that is what
    makes the factor-retention question askable). If the right-hand panel shows
    class structure, something is wrong with the axis assignment.
    """
    names = [a.name for a in spec.axes]
    label_axes = [int(k) for k in spec.label_axes]
    free_axes = [int(k) for k in spec.free_axes]
    classes = sorted(set(int(c) for c in conditions))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(classes)))

    pairs, titles = [], []
    if len(label_axes) >= 2:
        pairs.append((label_axes[0], label_axes[1]))
        titles.append("LABEL axes S -- classes should separate")
    elif len(label_axes) == 1 and len(free_axes) >= 1:
        pairs.append((label_axes[0], free_axes[0]))
        titles.append("LABEL axis (x) vs FREE axis (y)")
    if len(free_axes) >= 2:
        pairs.append((free_axes[0], free_axes[1]))
        titles.append("FREE axes S^c -- classes should NOT separate")

    if not pairs:
        return None

    fig, axes = plt.subplots(1, len(pairs), figsize=(5.4 * len(pairs), 5.0))
    if len(pairs) == 1:
        axes = [axes]
    for ax, (kx, ky), title in zip(axes, pairs, titles):
        for ci, c in enumerate(classes):
            m = np.asarray([int(cc) == c for cc in conditions])
            ax.scatter(Phi[m, kx], Phi[m, ky], s=90, color=colors[ci],
                       edgecolor="k", linewidth=0.6, label="class %d" % c)
        ax.set_xlabel("phi[%d] = %s" % (kx, names[kx]))
        ax.set_ylabel("phi[%d] = %s" % (ky, names[ky]))
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Latent coordinates phi (tau = %.3g)" % spec.class_overlap)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate and inspect the latent-factor benchmark.")
    ap.add_argument("--config", default=None,
                    help="config JSON (e.g. hpc/config_latent_3class_hard.json). "
                         "Omit to use the LatentSpec defaults.")
    ap.add_argument("--out-dir", default="latent_inspect")
    ap.add_argument("--duration-s", type=float, default=None,
                    help="override T_rec [s] for a quicker look")
    ap.add_argument("--n-neurons", type=int, default=None,
                    help="override the neuron count N")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the base seed")
    ap.add_argument("--n-per-class", type=int, nargs="+", default=None,
                    metavar="N",
                    help="override the traces per class, one integer per class. "
                         "Its LENGTH sets C, so '--n-per-class 3 3 3 3' gives a "
                         "4-class benchmark. Note the class centres are "
                         "m_c = (c+1)/(C+1) under the default interior spacing, "
                         "so the adjacent-centre gap SHRINKS as C grows "
                         "(0.25 at C=3, 0.20 at C=4): at fixed tau a larger C "
                         "is a harder task, and tau should be re-chosen, not "
                         "carried over.")
    ap.add_argument("--tau", type=float, default=None,
                    help="override the class overlap tau")
    ap.add_argument("--show-seconds", type=float, default=60.0,
                    help="plot only the first N seconds of each trace "
                         "(0 = whole trace)")
    a = ap.parse_args(argv)

    cfg = {}
    if a.config:
        with open(a.config, "r", encoding="ascii") as fh:
            cfg = json.load(fh)

    spec = latent_spec_from_config_dict(
        cfg, duration_s=a.duration_s, n_neurons=a.n_neurons, seed=a.seed,
        n_per_class=a.n_per_class, class_overlap=a.tau)

    print("=" * 74)
    print("latent benchmark")
    print("  config        : %s" % (a.config or "(defaults)"))
    print("  n (axes)      : %d  %r" % (spec.n_latent, [x.name for x in spec.axes]))
    print("  label axes S  : %r (0-based)  ->  %r"
          % (list(spec.label_axes), [spec.axes[k].name for k in spec.label_axes]))
    print("  free axes S^c : %r (0-based)  ->  %r"
          % (list(spec.free_axes), [spec.axes[k].name for k in spec.free_axes]))
    print("  C (classes)   : %d,  n_c = %r  ->  %d traces total"
          % (spec.n_classes, list(spec.n_per_class), sum(spec.n_per_class)))
    print("  tau (overlap) : %.4g" % spec.class_overlap)
    from latent_burst_generator import _class_mean as _cm
    print("  class centres : mode=%r  ->  m_c = %r"
          % (spec.class_center_mode,
             [round(_cm(c, spec.n_classes, spec.class_center_mode), 4)
              for c in range(spec.n_classes)]))
    print("  T_rec, f_s    : %.4g s, %.4g Hz  ->  K = %d samples"
          % (spec.duration_s, spec.fs, int(round(spec.duration_s * spec.fs))))
    print("  N (neurons)   : %d" % spec.n_neurons)
    print("  seed          : %d" % spec.seed)
    print("=" * 74)

    print("\nsynthesizing %d traces ..." % sum(spec.n_per_class))
    traces, conditions, trace_ids, Phi, fs = synthesize_dataset(spec)
    table = latent_ground_truth_table(spec)
    print("done: %d traces, K = %d samples each, f_s = %.4g Hz"
          % (len(traces), traces[0].shape[0], fs))

    print()
    print(format_latent_table(spec, conditions, trace_ids, Phi, table))

    # per-class summary of the trace itself, as a coarse sanity check
    print("\nper-class trace statistics (mean over that class's traces)")
    print("  %-6s %10s %10s %10s" % ("class", "mean IFR", "max IFR", "std IFR"))
    print("  " + "-" * 40)
    for c in sorted(set(conditions)):
        idx = [i for i, cc in enumerate(conditions) if cc == c]
        print("  %-6d %10.4f %10.4f %10.4f"
              % (c,
                 float(np.mean([traces[i].mean() for i in idx])),
                 float(np.mean([traces[i].max() for i in idx])),
                 float(np.mean([traces[i].std() for i in idx]))))

    os.makedirs(a.out_dir, exist_ok=True)
    seconds = None if (a.show_seconds is None or a.show_seconds <= 0) else a.show_seconds
    p1 = plot_traces_by_class(traces, conditions, fs,
                              os.path.join(a.out_dir, "latent_traces.png"),
                              seconds=seconds)
    p2 = plot_latent_factors(spec, conditions, Phi,
                             os.path.join(a.out_dir, "latent_factors.png"))
    out_json = os.path.join(a.out_dir, "latent_inspection.json")
    with open(out_json, "w", encoding="ascii") as fh:
        json.dump(table, fh, indent=2, ensure_ascii=True)
        fh.write("\n")

    print("\nwrote:")
    print("  %s" % p1)
    if p2:
        print("  %s" % p2)
    print("  %s" % out_json)
    print("\nWhat to look for in latent_factors.png: the LEFT panel (label axes)")
    print("should show the three classes at separated positions; the RIGHT panel")
    print("(free axes) should show them fully intermingled. If the right panel")
    print("separates too, the axis assignment is wrong and the factor-retention")
    print("measurement would be meaningless.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
