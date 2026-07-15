#!/usr/bin/env python3
"""
estimate_walltime.py
====================

Measure the REAL per-epoch training time on THIS node (CPU or GPU) and
extrapolate it to the full optimization/search budget of a given config, so you
can size your PBS walltime correctly BEFORE submitting a multi-hour job.

Why this exists
---------------
On a GPU (Colab T4) the 243-run x 60-epoch search fits in ~hours. On davinci-1
CPU it can be many times slower, and requesting too little walltime means the
job is killed near the end with nothing to show; requesting way too much wastes
your queue priority. This script removes the guesswork by TIMING a short real
run and multiplying by the real budget.

Method (identical arithmetic to the Colab wrapper's time_one_epoch_and_project,
reproduced here as a standalone HPC tool since the Colab wrapper is not used on
the cluster):
    1. Run a SHORT real training: --skip-search, n_seeds=1,
       max_epochs=<epochs_to_time>. This measures seconds_per_epoch from the
       actual trainer on this actual node -- not a proxy.
    2. Read the REAL config's budget via run_optimization.estimate_budget:
       TOTAL_train_runs and max_epochs (the same numbers `run_optimization.py
       --dry-run` prints).
    3. projected_seconds = seconds_per_epoch * max_epochs * TOTAL_train_runs
       Report it in hours, and print a suggested PBS walltime with a safety
       margin.

The short timing run also confirms end to end that the env, the data caching,
the splits, and one full train() all work on this node -- so it doubles as a
final pre-flight.

Usage
-----
    conda activate <your_env_name>
    python estimate_walltime.py --config config_search_3class_hpc.json
    # CPU is the default; add --device cuda on a GPU node
    # tune the measurement length with --epochs-to-time (default 3)
    # tune the safety margin with --margin (default 1.3 = +30%)

HPC note (hpc-python-compat): pure ASCII.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace


def _fmt_hms(seconds):
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return "%02d:%02d:%02d" % (h, m, s)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True,
                   help="the SAME config json you will run the full search with")
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu",
                   help="device for the timing run (default cpu; use the same "
                        "device you will run the real job on)")
    p.add_argument("--epochs-to-time", type=int, default=3,
                   help="how many epochs to actually run for the measurement "
                        "(default 3; more = steadier estimate, slower)")
    p.add_argument("--margin", type=float, default=1.3,
                   help="safety multiplier applied to the projection for the "
                        "suggested walltime (default 1.3 = +30%%)")
    p.add_argument("--cache-dir", default="./timing_cache",
                   help="scratch cache dir for the timing run")
    p.add_argument("--out-dir", default="./timing_out",
                   help="scratch out dir for the timing run")
    args = p.parse_args(argv)

    # Imports deferred until after argparse so --help works even if a heavy
    # dependency is missing, and so an import error names itself clearly.
    try:
        import run_optimization as R
        from run_optimization import load_config, estimate_budget, run
    except Exception as ex:
        print("ERROR: could not import run_optimization (%s: %s)."
              % (type(ex).__name__, ex))
        print("Make sure you run this from the pipeline directory with the "
              "conda env activated.")
        sys.exit(2)

    cfg = load_config(args.config)

    # --- budget of the REAL config (full search, all phases) ---
    budget = estimate_budget(cfg, skip_search=False, skip_regularization=False)
    total_runs = int(budget["TOTAL_train_runs"])
    max_epochs = int(cfg.train.max_epochs)

    print("=" * 70)
    print("Walltime estimator")
    print("=" * 70)
    print("config              : %s" % args.config)
    print("device (timing)     : %s" % args.device)
    print("full-search budget  : TOTAL_train_runs=%d, max_epochs=%d" %
          (total_runs, max_epochs))
    print("budget breakdown    : %s" % budget)
    print("")
    print("Timing a short run (%d epoch(s), 1 seed, search skipped) on this "
          "node ..." % args.epochs_to_time)
    print("-" * 70)

    # --- build a timing config: skip search, 1 seed, few epochs ---
    timing_cfg = R._deep_copy_cfg(cfg)
    timing_cfg.train = replace(timing_cfg.train,
                               n_seeds=1,
                               max_epochs=int(args.epochs_to_time),
                               patience=int(args.epochs_to_time))
    timing_cfg.runtime = replace(
        timing_cfg.runtime,
        device=args.device,
        cache_dir=args.cache_dir,
        out_dir=args.out_dir,
        experiment_name=(cfg.runtime.experiment_name + "_timing"))

    # --- assemble the args Namespace run() expects (mirrors the CLI flags) ---
    timing_args = argparse.Namespace(
        config=args.config,
        data_mode=None, npz_specs=None, specs_json=None, window_s=None,
        overwrite_cache=True,           # force a clean cache for a fair measurement
        engine_module=None, engine_kwargs=None,
        skip_search=True,               # measure ONE architecture, not the search
        skip_regularization=False,
        dry_run=False, resume=False,
        device=args.device, seed=None, n_seeds=None, max_epochs=None,
        num_workers=None, train_stride_s=None, eval_stride_s=None,
        synthetic_duration_s=None,
        n_calls_arch=None, n_calls_train=None, n_calls_reg=None,
        out_dir=args.out_dir, cache_dir=args.cache_dir,
        experiment_name=timing_cfg.runtime.experiment_name,
        verbose=True)

    t0 = time.time()
    result = run(timing_cfg, timing_args)
    wall = time.time() - t0

    secs_per_epoch = float(result["test"]["per_seed"][0]["seconds_per_epoch"])

    projected_seconds = secs_per_epoch * max_epochs * total_runs
    projected_hours = projected_seconds / 3600.0
    suggested_seconds = projected_seconds * float(args.margin)

    print("-" * 70)
    print("RESULTS")
    print("-" * 70)
    print("measured seconds/epoch     : %.3f  (timing run wall-clock %.1f s)"
          % (secs_per_epoch, wall))
    print("full-search train() runs   : %d" % total_runs)
    print("epochs per run (max)       : %d" % max_epochs)
    print("")
    print("projected TOTAL compute    : %.3f s/epoch * %d epochs * %d runs"
          % (secs_per_epoch, max_epochs, total_runs))
    print("                           = %.0f s = %.1f h = %s"
          % (projected_seconds, projected_hours, _fmt_hms(projected_seconds)))
    print("")
    print("suggested PBS walltime     : %s   (projection x %.2f safety margin)"
          % (_fmt_hms(suggested_seconds), args.margin))
    print("  -> in your .pbs script:  #PBS -l walltime=%s"
          % _fmt_hms(suggested_seconds))
    print("")

    # --- practical guidance ---
    note_lines = []
    note_lines.append("NOTES")
    note_lines.append("  * max_epochs is an UPPER bound: early stopping (patience=%d) often"
                      % int(cfg.train.patience))
    note_lines.append("    ends runs sooner, so the real total is usually LESS than projected.")
    note_lines.append("    The projection is deliberately conservative -- good for sizing walltime.")
    if projected_hours > 24:
        note_lines.append("  * >24 h projected. If your queue caps walltime below this, either")
        note_lines.append("    request a longer-walltime queue, or reduce the BUDGET (n_calls_arch /")
        note_lines.append("    n_calls_train / regularization.n_calls / n_seeds) -- NOT max_epochs,")
        note_lines.append("    since cutting epochs reintroduces the non-convergence failure mode.")
    if args.device == "cpu":
        note_lines.append("  * This is a CPU timing. A GPU node would typically be much faster;")
        note_lines.append("    re-run with --device cuda on a GPU node to size a GPU job.")
    print("\n".join(note_lines))

    return {
        "seconds_per_epoch": secs_per_epoch,
        "total_train_runs": total_runs,
        "max_epochs": max_epochs,
        "projected_seconds": projected_seconds,
        "projected_hours": projected_hours,
        "suggested_walltime_seconds": suggested_seconds,
    }


if __name__ == "__main__":
    main()
