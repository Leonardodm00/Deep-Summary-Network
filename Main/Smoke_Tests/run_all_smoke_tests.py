#!/usr/bin/env python3
"""
run_all_smoke_tests.py
=======================

Run every smoke-test suite in the project, in dependency order, and report a
single pass/fail table. Exit code 0 = everything passed.

    python3 run_all_smoke_tests.py             # all 12 suites (~6 min on 1 CPU)
    python3 run_all_smoke_tests.py --quick     # ~3 min: passes --quick to the
                                               # three suites that accept it
    python3 run_all_smoke_tests.py --list      # show the order, run nothing
    python3 run_all_smoke_tests.py --only train search   # substring match

Why a runner exists at all
---------------------------
There are 12 suites spread across three topics, they have to be run from a FLAT
directory (the modules import each other by bare name), and three of them accept
a --quick flag while the other nine reject it. Running them by hand means
remembering all of that. On Colab it is the one command to run per fresh
runtime, before spending GPU time; on the cluster it is what you run on the
login node before qsub.

Dependency order
----------------
Suites are ordered so that a failure lands as close as possible to its cause: a
broken config.py fails smoke_test_config first, rather than surfacing as a
confusing failure inside smoke_test_search 5 minutes later. The order is
foundational -> compositional:

    config -> backbone -> augmentation -> data_pipeline -> inference
           -> checkpoint -> evaluate -> burst_pipeline
           -> train -> search
           -> run_optimization -> run_optimization_colab

KNOWN GAP (not this runner's doing)
------------------------------------
02_TECHNICAL.md (sec. "Test suites") and 03_USAGE.md both list three further
suites, with specific check counts and runtimes, that DO NOT EXIST in the
repository:

    smoke_test_data_splits.py   (documented: 9 checks -- "leakage-free by
                                 construction", provenance no-drift, stride
                                 semantics)
    smoke_test_metrics.py       (documented: 11 checks -- eff_rank vs a known
                                 participation ratio, closed-form cosine vs
                                 brute force)
    smoke_test_end_to_end.py    (documented: 12 checks -- full pipeline, all 3
                                 pre-flights, artifacts, resume)

The third is effectively SUPERSEDED by smoke_test_run_optimization.py, which
covers the full pipeline, all three pre-flights, the artifact tree, resume, and
"no interactive call reachable". The first two are genuinely missing, and they
cover the two most load-bearing modules in the project: data_splits.py carries
the LEAKAGE GUARANTEE, and metrics.py computes the OBJECTIVE the entire
hyper-parameter search maximises. Both are currently exercised only indirectly.
This runner prints that warning at the end rather than quietly reporting "all
green", because the docs' table would otherwise leave you believing those two
modules are directly tested when they are not.

HPC note (hpc-python-compat): pure ASCII.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# The CANONICAL order, foundational -> compositional (see module docstring).
# Nineteen suites are listed now. Three of them (data_splits, metrics,
# end_to_end) were written and passing but had never been committed to the
# repository; they were recovered and are included here.
# smoke_test_synthetic_config.py was added later (configurable burst-generator
# params + per-run saved traces) and is placed after
# smoke_test_run_optimization.py because it imports run_optimization directly
# (R._resolved_synthetic_params, R.save_synthetic_artifacts), so it depends on
# that suite's import chain already working.
# smoke_test_removed_modules.py runs FIRST: it needs no torch, no data and no
# import from the package, it takes milliseconds, and it is a pure regression
# guard against a deleted module reappearing (Change 2). Any suite listed but
# absent on disk is reported as a GAP at the end of the run rather than
# silently skipped.
# Four suites that existed on disk but had never been registered here were added
# on 29 July 2026, taking the count to twenty-three. They were placed by the same
# two rules the list already followed -- cheapest and most depended-upon first,
# and anything importing run_optimization after that suite's import chain is
# known to work:
#   objective_wiring  SECOND, right behind removed_modules: it needs no torch and
#                     no data either, and it checks the selection rule and the
#                     tie-break math that metrics, train and search all rest on,
#                     so a broken rule fails here in seconds instead of inside
#                     search much later.
#   selected_epoch    directly AFTER train and BEFORE search: it trains for real
#                     and asserts that objective_utils recomputes the same e*
#                     train.py recorded, which is the assumption search makes.
#   latent_wiring     AFTER run_optimization, which it imports. Guards the
#                     preprocessing cache FINGERPRINT: the spec list is identical
#                     for every tau, so without it a changed tau silently reuses
#                     old traces.
#   inspect_latent    likewise imports run_optimization; guards the duplicated
#                     config -> LatentSpec mapping against the driver's copy.
ORDER = [
    "smoke_test_removed_modules.py",        # Change 2 deletion guard (no torch)
    "smoke_test_objective_wiring.py",       # selection rule + tie-break MATH (no torch)
    "smoke_test_batch_geometry.py",         # [C4] Eq. (2)/(3) + the caps (no torch)
    "smoke_test_config.py",
    "smoke_test_backbone.py",
    "smoke_test_augmentation.py",
    "smoke_test_data_pipeline.py",
    "smoke_test_data_splits.py",            # the LEAKAGE GUARANTEE
    "smoke_test_trace_splits.py",           # WHOLE-CULTURE leakage guarantee
    "smoke_test_metrics.py",                # the SEARCH OBJECTIVE
    "smoke_test_inference.py",
    "smoke_test_checkpoint.py",
    "smoke_test_evaluate.py",
    "smoke_test_burst_pipeline.py",
    "smoke_test_adaptive_patience.py",      # growing patience + silhouette floor
    "smoke_test_train.py",
    "smoke_test_selected_epoch.py",         # train.py vs objective_utils: NO DRIFT
    "smoke_test_search.py",
    "smoke_test_end_to_end.py",             # full pipeline + REAL resume
    "smoke_test_run_optimization.py",
    "smoke_test_latent_wiring.py",          # latent cache FINGERPRINT + stale refusal
    "smoke_test_inspect_latent.py",         # inspection tool vs driver: NO DRIFT
    "smoke_test_synthetic_config.py",       # configurable burst params + saved artifacts
    "smoke_test_run_optimization_colab.py",
]

# Descriptions used only when a suite listed above is missing from disk.
MISSING = {
    "smoke_test_data_splits.py":
        "the LEAKAGE GUARANTEE (data_splits.py), provenance, stride semantics",
    "smoke_test_metrics.py":
        "the SEARCH OBJECTIVE (metrics.py): eff_rank, closed-form cosine",
    "smoke_test_end_to_end.py":
        "full pipeline, all 3 pre-flights, artifacts, REAL resume-from-last.pt",
    "smoke_test_trace_splits.py":
        "the WHOLE-CULTURE leakage guarantee (make_trace_splits): disjointness, "
        "stratification, tiling, trace_of_window, determinism",
    "smoke_test_adaptive_patience.py":
        "growing-patience termination bound and the label-shuffled silhouette "
        "floor that sets its threshold",
    "smoke_test_removed_modules.py":
        "the Change 2 deletion guard: the removed retention module, its suite "
        "and its references must stay removed",
    "smoke_test_batch_geometry.py":
        "[C4] batch geometry: Eq. (2) n_g and M, the Eq. (3) availability "
        "clamp, and the miner-gated group-size, degeneracy and resource caps",
    "smoke_test_objective_wiring.py":
        "the selection rule and tie-break MATH: lexicographic e*, Delta_min(y), "
        "role-ordered (u, v), and the tie-break dispatch on selection_primary",
    "smoke_test_selected_epoch.py":
        "the DRIFT TEST: on real training runs, objective_utils must recompute "
        "the same e* train.py itself recorded",
    "smoke_test_latent_wiring.py":
        "the latent cache FINGERPRINT: it must change with tau, label_axes and "
        "seed, and a stale cache must raise rather than be silently reused",
    "smoke_test_inspect_latent.py":
        "the inspection tool's config -> LatentSpec mapping must agree field by "
        "field with run_optimization's copy",
}


def accepts_quick(path):
    """True if the suite's own argparse defines --quick. Passing it to a suite
    that does not would make argparse exit(2) and look like a test failure."""
    try:
        return '"--quick"' in path.read_text(encoding="ascii")
    except Exception:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--quick", action="store_true",
                    help="pass --quick to the suites that accept it")
    ap.add_argument("--list", action="store_true",
                    help="print the run order and exit")
    ap.add_argument("--only", nargs="+", metavar="PAT",
                    help="run only suites whose filename contains a pattern")
    a = ap.parse_args(argv)

    here = Path(__file__).resolve().parent
    suites = [here / n for n in ORDER if (here / n).exists()]
    absent = [n for n in ORDER if not (here / n).exists()]

    if a.only:
        suites = [p for p in suites
                  if any(pat.lower() in p.name.lower() for pat in a.only)]

    if a.list:
        print("run order (%d suite(s)):" % len(suites))
        for p in suites:
            print("  %-40s %s" % (p.name,
                  "(accepts --quick)" if accepts_quick(p) else ""))
        return 0

    if not suites:
        print("no suites matched.")
        return 1

    print("=" * 78)
    print("run_all_smoke_tests.py  --  %d suite(s)%s"
          % (len(suites), "  [--quick]" if a.quick else ""))
    print("  cwd: %s" % here)
    print("=" * 78)

    results = []
    t_all = time.time()
    for p in suites:
        cmd = [sys.executable, "-W", "ignore::RuntimeWarning", str(p)]
        if a.quick and accepts_quick(p):
            cmd.append("--quick")
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(here), capture_output=True, text=True)
        dt = time.time() - t0
        passed = proc.returncode == 0
        results.append((p.name, passed, dt, proc))
        print("  %-40s %s  %5.1fs" % (p.name, "PASS" if passed else "FAIL", dt))
        if not passed:
            print("  " + "-" * 74)
            tail = (proc.stdout or "").strip().splitlines()[-15:]
            err = (proc.stderr or "").strip().splitlines()[-15:]
            for line in tail + err:
                print("    | %s" % line[:110])
            print("  " + "-" * 74)

    n_pass = sum(1 for _, ok, _, _ in results if ok)
    n_fail = len(results) - n_pass
    print("=" * 78)
    print("  %d/%d suites passed in %.1fs" % (n_pass, len(results),
                                              time.time() - t_all))

    if absent:
        print()
        print("  NOTE: %d suite(s) documented in 02_TECHNICAL.md / 03_USAGE.md are"
              % len(absent))
        print("  NOT in this repository, so 'all green' above does NOT mean they")
        print("  passed -- it means they were never run:")
        for n in absent:
            print("    %-32s %s" % (n, MISSING.get(n, "")))
        print("  data_splits.py (the leakage guarantee) and metrics.py (the search")
        print("  objective) are therefore exercised only INDIRECTLY, by the suites")
        print("  above. Treat that as a gap, not as coverage.")

    print("=" * 78)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
