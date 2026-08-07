#!/usr/bin/env python3
"""
make_mea_specs.py -- build data.npz_specs from a CohortConfig.

WHAT THIS IS FOR
----------------
The search consumes a FLAT list of .npz records (data_mode == "numpy"). The real
cohort is a nested tree of plates, wells and subregion archives. This script is
the bridge, and it is the ONLY place where "which well belongs to which class"
is decided:

    <root>/ptrain_A1/            a WELL == a recording == a CULTURE
        |
        | run_channel_subset_extraction.py   (already run, separately)
        v
    <extract_root>/.../ptrain_A1/
        trace_subregion_00.npz ... trace_subregion_08.npz     [per_region_single]
        or traces.npz                                          [multichannel]
        |
        | THIS SCRIPT
        v
    npz_specs.json    [{path, name, condition, culture}, ...]

CULTURE IS THE POINT (K3)
-------------------------
In per_region_single mode a well yields N_SUB archives that are N_SUB trace
RECORDS but ONE culture. Every record from a well is emitted with the same
`culture`, which is what keeps its subregions (i) inside a single split and
(ii) out of each other's positive pairs. Omit that field and the pipeline
silently treats them as N_SUB independent cultures -- so run_optimization.py
refuses per-subregion archives that arrive without one.

BOTH EXTRACTION MODES ARE RECOGNISED
------------------------------------
The layout found on disk decides the record count per well, and is REPORTED
rather than assumed:

    per_region_single  N_SUB records/well, shared culture, n_channels = 1
    multichannel       1 record/well,      culture == name, n_channels = N_SUB

Mixing the two across one cohort is refused: they imply different
data.n_channels, and the backbone stem cannot be both.

USAGE
-----
    python3 make_mea_specs.py --config hpc/Config/config_mea_joint_full.json

    --out PATH        where to write the specs (default: data.npz_specs from
                      the config; --out overrides it)
    --relative-to DIR write paths relative to DIR instead of absolute
    --strict          exit non-zero if ANY expected well is missing its
                      extraction output (default: report and continue with
                      what exists)
    --dry-run         print the inventory and the split feasibility check,
                      write nothing

EXIT STATUS
    0 specs written (or dry run clean); 1 nothing usable found;
    2 configuration/usage error; 3 --strict and something was missing.

Pure ASCII (hpc-python-compat).
"""

import argparse
import json
import os
import sys
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import ExperimentConfig                            # noqa: E402

SUBREGION_PREFIX = "trace_subregion_"
MULTICHANNEL_NAME = "traces.npz"


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def root_name_for(root):
    """The {root_name} value substituted into extract_layout/culture_template.

    THE single place this is computed -- both build_records() below and
    list_extraction_jobs.py's manifest builder call this, so the two can
    never disagree about a well's culture id or output path.

    Two path COMPONENTS (parent + leaf), not one. A bare basename collides
    the moment two batches reuse a leaf name -- e.g. a "SubBatch1" folder
    that exists under both a control batch and a pathological batch is a
    real, observed layout, not a hypothetical: DATA_C/Batch4/SubBatch1 and
    DATA_P/Batch3/SubBatch1 are two different cultures that a bare basename
    would fold into one, silently pointing both wells' extraction at the
    same output directory.

    This is a best-effort disambiguator, not a uniqueness proof: two roots
    that ALSO share their parent's name would still collide. That is exactly
    why the collision check at the call site (build_records here;
    list_extraction_jobs.py's own check) is a hard abort rather than a
    warning -- silent data loss is not an acceptable failure mode for either
    caller, so an unresolved collision must stop the run, not print past it.
    """
    root = os.path.normpath(str(root))
    leaf = os.path.basename(root)
    parent = os.path.basename(os.path.dirname(root))
    return "%s_%s" % (parent, leaf) if parent else leaf


def find_wells(root, well_glob):
    """Immediate child directories of `root` matching `well_glob`, sorted.

    Deliberately NOT recursive: the extractor reads one leaf well folder at a
    time, and recursing would sweep up intermediate Batch/ folders as if they
    were wells (see REAL_DATA_FINDINGS, "Command").
    """
    import fnmatch
    if not os.path.isdir(root):
        return None                                  # signals "root missing"
    out = []
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if os.path.isdir(full) and fnmatch.fnmatch(entry, well_glob):
            out.append(entry)
    return out


def classify_output_dir(out_dir):
    """('per_region_single', [paths]) | ('multichannel', [path]) | (None, [])."""
    if not os.path.isdir(out_dir):
        return None, []
    names = sorted(os.listdir(out_dir))
    subs = [n for n in names
            if n.startswith(SUBREGION_PREFIX) and n.endswith(".npz")]
    if subs:
        return "per_region_single", [os.path.join(out_dir, n) for n in subs]
    if MULTICHANNEL_NAME in names:
        return "multichannel", [os.path.join(out_dir, MULTICHANNEL_NAME)]
    return None, []


def expand(template, class_index, class_name, root_name, well):
    return template.format(class_index=class_index, class_name=class_name,
                           root_name=root_name, well=well)


# --------------------------------------------------------------------------- #
# the walk
# --------------------------------------------------------------------------- #
def build_records(cohort, verbose=True):
    """Walk the cohort and return (records, report).

    records : list of dicts {path, name, condition, culture}
    report  : dict of counters and problems, for the caller to print/act on
    """
    report = {
        "missing_roots": [], "empty_roots": [], "missing_output": [],
        "modes": {}, "wells_per_class": {}, "cultures": OrderedDict(),
        "n_sub_seen": set(),
    }
    records = []
    seen_names, seen_cultures = {}, {}

    for c in range(cohort.n_classes()):
        cname = cohort.name_of_class(c)
        roots = cohort.class_roots[str(c)]
        report["wells_per_class"][c] = 0

        for root in roots:
            root = str(root)
            root_name = root_name_for(root)
            wells = find_wells(root, cohort.well_glob)

            if wells is None:
                report["missing_roots"].append(root)
                continue
            if not wells:
                report["empty_roots"].append(root)
                continue

            for well in wells:
                rel = expand(cohort.extract_layout, c, cname, root_name, well)
                out_dir = os.path.join(str(cohort.extract_root), rel)
                mode, paths = classify_output_dir(out_dir)

                if mode is None:
                    report["missing_output"].append((root, well, out_dir))
                    continue

                report["modes"][mode] = report["modes"].get(mode, 0) + 1
                culture = expand(cohort.culture_template, c, cname,
                                 root_name, well)

                if culture in seen_cultures and seen_cultures[culture] != (c, root):
                    raise SystemExit(
                        "ABORT: culture id %r is produced by two different "
                        "wells (%r and %r). Culture ids must be unique across "
                        "the whole cohort -- adjust cohort.culture_template."
                        % (culture, seen_cultures[culture], (c, root)))
                seen_cultures[culture] = (c, root)
                report["cultures"][culture] = c
                report["wells_per_class"][c] += 1

                if mode == "per_region_single":
                    report["n_sub_seen"].add(len(paths))
                    for j, p in enumerate(sorted(paths)):
                        name = "%s__sub%02d" % (culture, j)
                        if name in seen_names:
                            raise SystemExit(
                                "ABORT: duplicate record name %r (from %s and "
                                "%s). Names are the cache primary key."
                                % (name, seen_names[name], p))
                        seen_names[name] = p
                        records.append({"path": p, "name": name,
                                        "condition": c, "culture": culture})
                else:                                   # multichannel
                    name = culture
                    if name in seen_names:
                        raise SystemExit(
                            "ABORT: duplicate record name %r." % name)
                    seen_names[name] = paths[0]
                    records.append({"path": paths[0], "name": name,
                                    "condition": c, "culture": culture})

    return records, report


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def apportion_preview(n, fractions=(0.6, 0.2, 0.2)):
    """Largest-remainder apportionment, mirroring data_splits.apportion."""
    raw = [n * f for f in fractions]
    base = [int(x) for x in raw]
    rem = n - sum(base)
    order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - base[i]), i))
    for i in range(rem):
        base[order[i % len(order)]] += 1
    return base


def print_report(records, report, cohort, cfg):
    n_cult = len(report["cultures"])
    print("")
    print("COHORT INVENTORY")
    print("  classes            : %d" % cohort.n_classes())
    for c in range(cohort.n_classes()):
        n_roots = len(cohort.class_roots[str(c)])
        n_wells = report["wells_per_class"].get(c, 0)
        print("    class %d (%-14s): %d root(s), %d well(s)/culture(s)"
              % (c, cohort.name_of_class(c), n_roots, n_wells))
    print("  cultures total     : %d" % n_cult)
    print("  trace records      : %d" % len(records))
    if report["modes"]:
        for m, k in sorted(report["modes"].items()):
            print("  extraction mode    : %s  (%d well(s))" % (m, k))
    if report["n_sub_seen"]:
        print("  subregions/well    : %s"
              % sorted(report["n_sub_seen"]))

    for key, label in (("missing_roots", "ROOT NOT FOUND"),
                       ("empty_roots", "ROOT HAS NO MATCHING WELL")):
        for r in report[key]:
            print("  WARNING %s: %s" % (label, r))
    if report["missing_output"]:
        print("  WARNING: %d well(s) have no extraction output:"
              % len(report["missing_output"]))
        for root, well, out_dir in report["missing_output"][:10]:
            print("    %s / %s  ->  %s" % (os.path.basename(root), well, out_dir))
        if len(report["missing_output"]) > 10:
            print("    ... and %d more" % (len(report["missing_output"]) - 10))

    # ---- split feasibility, the thing that aborts a run at startup ---------
    print("")
    print("SPLIT FEASIBILITY (data.split_fractions=%r, min_train=%d)"
          % (list(cfg.data.split_fractions),
             int(cfg.data.min_train_cultures_per_class)))
    min_train = int(cfg.data.min_train_cultures_per_class)
    ok = True
    u_avail = []
    for c in range(cohort.n_classes()):
        n_c = report["wells_per_class"].get(c, 0)
        tr, va, te = apportion_preview(n_c, tuple(cfg.data.split_fractions))
        u_avail.append(tr)
        flag = ""
        if n_c < min_train + 2:
            flag = "  <-- TOO FEW (need >= %d)" % (min_train + 2)
            ok = False
        print("  class %d: %2d culture(s) -> train %d / val %d / test %d%s"
              % (c, n_c, tr, va, te, flag))

    if u_avail:
        u_req = int(cfg.data.cultures_per_class_per_batch)
        u_eff = min([u_req] + u_avail)
        q = int(cfg.data.windows_per_culture_per_batch)
        n_s = int(cfg.data.augmentation.n_negatives)
        m = cohort.n_classes() * u_eff * q * (1 + n_s)
        print("  U_eff = min(request %d, avail %r) = %d"
              % (u_req, u_avail, u_eff))
        print("  batch M = C*U_eff*q*(1+N_s) = %d*%d*%d*%d = %d"
              % (cohort.n_classes(), u_eff, q, 1 + n_s, m))
        print("  cross-culture positives per anchor = U_eff - 1 = %d"
              % (u_eff - 1))
        if u_eff < u_req:
            print("  NOTE: the request %d was CLAMPED to %d by the cohort."
                  % (u_req, u_eff))
    return ok


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--relative-to", default=None)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.config):
        print("ABORT: config not found: %s" % args.config)
        return 2
    cfg = ExperimentConfig.from_json(args.config)
    cohort = cfg.cohort

    if not cohort.class_roots:
        print("ABORT: cohort.class_roots is empty in %s. Nothing to walk."
              % args.config)
        return 2
    if not cohort.extract_root:
        print("ABORT: cohort.extract_root is empty; this script reads the "
              "EXTRACTION OUTPUTS, not the raw ptrain_*.mat folders.")
        return 2

    records, report = build_records(cohort)

    if not records:
        print_report(records, report, cohort, cfg)
        print("\nABORT: no usable extraction output found under %r."
              % cohort.extract_root)
        return 1

    modes = set(report["modes"])
    if len(modes) > 1:
        print("ABORT: the cohort mixes extraction modes %r. They imply "
              "different data.n_channels (1 vs %d) and the backbone stem "
              "cannot be both. Re-extract the odd wells."
              % (sorted(modes), cohort.n_subsets))
        return 2
    mode = modes.pop()
    n_channels = 1 if mode == "per_region_single" else int(cohort.n_subsets)

    if args.relative_to:
        base = os.path.abspath(args.relative_to)
        for r in records:
            r["path"] = os.path.relpath(os.path.abspath(r["path"]), base)

    ok = print_report(records, report, cohort, cfg)

    print("")
    print("CONFIG CONSISTENCY")
    want = int(cfg.data.n_channels)
    print("  extraction mode implies data.n_channels = %d; config says %d%s"
          % (n_channels, want, "" if want == n_channels else "   <-- MISMATCH"))
    if want != n_channels:
        ok = False
    fs_ifr = cohort.fs_ifr()
    t_win = float(cfg.data.window_s)
    print("  f_s^IFR = 1/w_size = %.6g Hz -> window T = %.6g * %.6g = %d samples"
          % (fs_ifr, t_win, fs_ifr, int(round(t_win * fs_ifr))))
    print("  smoothing sigma = %.6g s = %.2f bin(s) at w_size = %.6g s"
          % (float(cohort.gaussian_window), cohort.sigma_bins(),
             float(cohort.w_size)))

    if args.strict and (report["missing_output"] or report["missing_roots"]
                        or report["empty_roots"]):
        print("\nABORT (--strict): the inventory is incomplete.")
        return 3

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0 if ok else 1

    out = args.out or cfg.data.npz_specs
    if not out:
        print("\nABORT: no output path. Set data.npz_specs in the config or "
              "pass --out.")
        return 2
    out = os.path.abspath(out)
    parent = os.path.dirname(out)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    # ensure_ascii=True keeps the artifact HPC-safe on disk.
    with open(out, "w", encoding="ascii") as fh:
        json.dump(records, fh, indent=2)
        fh.write("\n")
    print("\nwrote %d record(s) over %d culture(s) -> %s"
          % (len(records), len(report["cultures"]), out))
    if not ok:
        print("NOTE: the run would still fail the checks flagged above.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
