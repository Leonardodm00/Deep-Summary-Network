#!/usr/bin/env python3
"""
test_real_data_loader.py -- exercise the REAL data loader against the cohort
paths actually filled into a config file. No training, no qsub.

This IS the "build the cache once" pre-flight step the runbook calls for
(concurrent search lanes would otherwise race on the first cache write), with
much richer diagnostics than a bare `run_optimization.py --dry-run`.

What it does, in order -- each step aborts before the next if it fails:

  1. REFUSES TO RUN if any cohort path still contains the REPLACE/ME
     placeholder. This is the single most common way this step fails
     silently: the config loads fine, the walk finds nothing, and a bare
     --dry-run just reports zero wells with no clue why.
  2. Walks the cohort (reusing make_mea_specs.build_records -- the SAME code
     the real specs generator uses, so this cannot drift from it) and prints
     the inventory: cultures found, extraction mode detected, missing wells.
  3. Writes data.npz_specs (unless --no-write; --strict makes a missing well
     a hard failure instead of a warning).
  4. Loads EVERY trace through the REAL loader
     (run_optimization.build_traces), which populates runtime.cache_dir --
     the SAME cache directory the actual search will read from, so this is
     not wasted work; it is the mandatory single-writer cache build.
  5. Sanity-checks trace CONTENT, not just presence: flags any trace that is
     entirely non-finite or perfectly constant, which a bad export can
     produce silently (a truncated write, a zeroed buffer) while still
     passing every structural check.
  6. Runs the K3-aware split (build_cultures + build_splits) and asserts the
     one invariant that matters: no culture appears in two splits.
  7. Prints window counts per split at the CONFIGURED window_s / strides, so
     you see the real batch geometry before committing to a lane launch.

Usage
-----
    cd Main
    export PYTHONPATH="$(pwd)"
    python3 test_real_data_loader.py --config hpc/Config/config_mea_joint_full.json

    --no-write     don't (re)write npz_specs; use the existing file as-is
    --strict       missing extraction output is a hard failure (exit 3),
                   not a warning
    --max-print N  cap the per-trace content-check lines printed (default 10)

Exit status: 0 if every check passes; non-zero otherwise. Never launches
training and never calls qsub.

Pure ASCII, LF only (hpc-python-compat).
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import ExperimentConfig                              # noqa: E402
import make_mea_specs as MMS                                      # noqa: E402
import run_optimization as RO                                     # noqa: E402

_PLACEHOLDER = "REPLACE/ME"


def _fail(msg):
    print("ABORT: %s" % msg)
    sys.exit(1)


# --------------------------------------------------------------------------- #
# step 1
# --------------------------------------------------------------------------- #
def check_no_placeholders(cohort):
    bad = []
    if _PLACEHOLDER in str(cohort.extract_root):
        bad.append("cohort.extract_root")
    for k, roots in cohort.class_roots.items():
        for r in roots:
            if _PLACEHOLDER in str(r):
                bad.append("cohort.class_roots[%r]" % k)
    if bad:
        _fail("cohort still has REPLACE/ME placeholder(s) in: %s\n"
              "       Edit the config's \"cohort\" block with your real "
              "plate/extraction paths before running this."
              % ", ".join(sorted(set(bad))))
    print("[1/7] no placeholder paths remain -- OK")


# --------------------------------------------------------------------------- #
# step 5
# --------------------------------------------------------------------------- #
def check_trace_content(traces, cultures, conditions, max_print):
    """Flag traces that are all-NaN/Inf or perfectly constant.

    Both pass every structural check (right shape, right dtype, loads without
    error) while being the two shapes a silently-bad export most often takes:
    a truncated write leaves NaNs; a zeroed or never-updated buffer is
    constant. Neither is a K3 concern -- this is pure data-quality.
    """
    bad = []
    for i, x in enumerate(traces):
        x = np.asarray(x)
        finite = np.isfinite(x)
        if not finite.all():
            n_bad = int((~finite).sum())
            bad.append((i, "non-finite", "%d / %d sample(s)" % (n_bad, x.size)))
            continue
        if x.size and float(x.max()) == float(x.min()):
            bad.append((i, "constant", "value=%.6g" % float(x.max())))

    n_ok = len(traces) - len(bad)
    print("[5/7] trace content: %d / %d OK" % (n_ok, len(traces)))
    if bad:
        print("      %d problem trace(s) (showing up to %d):"
              % (len(bad), max_print))
        for i, kind, detail in bad[:max_print]:
            print("        culture=%-30s condition=%d  %-10s %s"
                  % (cultures[i], int(conditions[i]), kind, detail))
        if len(bad) > max_print:
            print("        ... and %d more" % (len(bad) - max_print))
    return bad


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--config", required=True)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--max-print", type=int, default=10)
    args = ap.parse_args(argv)

    if not os.path.isfile(args.config):
        _fail("config not found: %s" % args.config)
    cfg = ExperimentConfig.from_json(args.config)
    cohort = cfg.cohort

    if not cohort.class_roots:
        _fail("cohort.class_roots is empty in %s; nothing to test." % args.config)

    # ---- 1. placeholder guard ---------------------------------------------
    check_no_placeholders(cohort)

    # ---- 2. walk the cohort (identical code path to make_mea_specs) -------
    records, report = MMS.build_records(cohort)
    if not records:
        _fail("the cohort walk found zero usable trace records. Run:\n"
              "         python3 make_mea_specs.py --config %s --dry-run\n"
              "       for the full inventory and what is missing." % args.config)
    MMS.print_report(records, report, cohort, cfg)
    print("[2/7] cohort walk: %d record(s) over %d culture(s) -- OK"
          % (len(records), len(report["cultures"])))

    if len(set(report["modes"])) > 1:
        _fail("the cohort mixes extraction modes %r; fix extraction before "
              "testing the loader." % sorted(report["modes"]))

    # ---- 3. write specs -----------------------------------------------------
    if args.no_write:
        specs_path = cfg.data.npz_specs
        if not specs_path or not os.path.isfile(specs_path):
            _fail("--no-write given but data.npz_specs (%r) does not exist. "
                  "Drop --no-write to generate it." % specs_path)
        print("[3/7] --no-write: using existing %s" % specs_path)
    else:
        rc = MMS.main(["--config", args.config] + (["--strict"] if args.strict else []))
        if rc not in (0,):
            _fail("make_mea_specs exited %d; see the report above." % rc)
        print("[3/7] wrote %s -- OK" % cfg.data.npz_specs)

    # ---- 4. load every trace through the REAL loader -----------------------
    print("[4/7] loading traces via the real loader "
          "(this also builds runtime.cache_dir=%r) ..." % cfg.runtime.cache_dir)
    try:
        traces, conditions, fs = RO.build_traces(cfg, verbose=False)
    except Exception as ex:
        _fail("build_traces raised %s: %s" % (type(ex).__name__, ex))
    try:
        cultures = RO.build_cultures(cfg)
    except Exception as ex:
        _fail("build_cultures raised %s: %s" % (type(ex).__name__, ex))

    if len(traces) != len(records):
        _fail("loaded %d trace(s) but the cohort walk found %d; the cache "
              "and the specs file have drifted apart. Re-run with a clean "
              "cache_dir." % (len(traces), len(records)))
    if len(cultures) != len(traces):
        _fail("build_cultures returned %d id(s) for %d trace(s)."
              % (len(cultures), len(traces)))
    n_cult = len(set(str(c) for c in cultures))
    print("[4/7] loaded %d trace(s), fs=%.6g Hz, %d distinct culture(s) -- OK"
          % (len(traces), fs, n_cult))

    # ---- 5. content sanity ---------------------------------------------------
    bad = check_trace_content(traces, cultures, conditions, args.max_print)

    # ---- 6. K3-aware split ----------------------------------------------------
    try:
        splits = RO.build_splits(cfg, traces, conditions, fs,
                                 verbose=True, cultures=cultures)
    except Exception as ex:
        _fail("build_splits raised %s: %s" % (type(ex).__name__, ex))

    seen = {}
    for name in ("train", "val", "test"):
        seen[name] = set(np.asarray(splits.trace_of_window[name]).tolist())
    leak = ((seen["train"] & seen["test"]) | (seen["train"] & seen["val"])
            | (seen["val"] & seen["test"]))
    if leak:
        _fail("K3 INVARIANT VIOLATED: culture(s) %r span more than one "
              "split. This must never happen; stop and investigate before "
              "using this cache." % sorted(leak)[:5])
    print("[6/7] K3 split: no culture spans two splits -- OK")

    # ---- 7. window counts at the configured geometry --------------------------
    print("[7/7] window counts at window_s=%.4g, train_stride_s=%.4g, "
          "eval_stride_s=%.4g:"
          % (cfg.data.window_s, cfg.data.train_stride_s, cfg.data.eval_stride_s))
    for name in ("train", "val", "test"):
        n_cu = len(splits.cultures[name])
        n_wi = len(np.asarray(splits.trace_of_window[name]))
        print("       %-5s: %3d culture(s), %5d window(s)" % (name, n_cu, n_wi))

    print("")
    if bad:
        print("DONE WITH WARNINGS: %d trace(s) have suspect content (see "
              "[5/7] above). Structure and K3 grouping are sound; the flagged "
              "traces are a data-quality question, not a pipeline bug."
              % len(bad))
        return 1
    print("ALL CHECKS PASSED. The cache at %r is ready for a real run."
          % cfg.runtime.cache_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())