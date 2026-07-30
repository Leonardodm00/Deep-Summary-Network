"""
audit_factorial.py

Verify that a set of DSN JSON configs really is the factorial design you think
it is, BEFORE spending cluster hours on it.

Why this exists
---------------
A factorial ablation is only interpretable if every config differs in the
FACTOR fields and in nothing else. A single stray edit (a different seed, a
different margin, a different split fold) silently turns a controlled
comparison into an uncontrolled one, and the corruption is invisible in the
run logs: every cell still trains, still converges, still writes results.json.
The only cheap moment to catch it is before qsub.

This script therefore does four things, and fails loudly on any of them:

  1. STRUCTURE   every config parses and they all carry the SAME key set
                 (a key present in one file and absent in another is a
                 silent default, not a controlled setting).
  2. CONTRAST    the set of fields that differ across the configs is a subset
                 of (declared factor fields) UNION (declared nuisance fields).
                 Any other differing field is an ERROR.
  3. CROSSING    the observed (factor level) combinations are exactly the full
                 Cartesian product of the declared levels, each realised
                 exactly once -- no missing cell, no duplicate cell, no
                 undeclared level.
  4. HYGIENE     runtime.experiment_name is unique per config (otherwise two
                 cells overwrite each other's output directory), and fields
                 that are EXPECTED to be shared (cache_dir, out_dir,
                 torch_threads) are reported so the caller can act on them.

It is deliberately stdlib-only (no numpy, no torch, no sklearn): it must run
on a login node with no conda environment activated, in milliseconds.

Design specification
--------------------
The design being audited is data, not code. DEFAULT_DESIGN below encodes the
l3c 3 x 2 study; point --design at a JSON file with the same shape to audit a
different study without touching this file.

A "factor" is a named group of one or more config fields that move together.
The head geometry is one factor with three levels even though it is spelled
across two fields (backbone.head_fusion and backbone.head_pool_ops), because
those two fields are not independently varied here.

Canonicalisation
----------------
backbone.head_pool_ops is order-insensitive downstream: backbone.py reorders
the requested ops into the canonical order ("mean", "max", "std") before
concatenating, so ["max", "mean"] and ["mean", "max"] build the SAME network.
This script therefore compares pool-op lists as canonically ordered tuples,
so a harmless reordering is not reported as a design violation.

Usage
-----
    python3 audit_factorial.py <dir>                 # audit every *.json in dir
    python3 audit_factorial.py a.json b.json ...     # audit an explicit list
    python3 audit_factorial.py <dir> --design d.json # non-default design
    python3 audit_factorial.py <dir> --quiet         # exit code only

Exit codes: 0 = design verified, 1 = violation found, 2 = usage error.

HPC note (hpc-python-compat): pure ASCII, LF endings, stdlib only.
"""

import argparse
import glob
import json
import os
import sys

__all__ = [
    "DEFAULT_DESIGN",
    "flatten",
    "canonical_value",
    "load_configs",
    "audit",
    "format_report",
]


# ----------------------------------------------------------------------
# The design under audit (edit this, or pass --design, for a new study)
# ----------------------------------------------------------------------
DEFAULT_DESIGN = {
    "name": "l3c 3 x 2: head geometry x triplet miner",
    # Each factor: an ordered list of fields, and the levels those fields
    # jointly take. "values" is positionally aligned with "fields".
    "factors": {
        "head": {
            "fields": ["backbone.head_fusion", "backbone.head_pool_ops"],
            "levels": [
                {"label": "singlemean", "values": [False, ["mean"]]},
                {"label": "multimean", "values": [True, ["mean"]]},
                {"label": "multiall",
                 "values": [True, ["mean", "max", "std"]]},
            ],
        },
        "miner": {
            "fields": ["train.mining_strategy"],
            "levels": [
                {"label": "hard", "values": ["hard"]},
                {"label": "epsh", "values": ["easy_pos_semihard_neg"]},
            ],
        },
    },
    # Fields allowed to differ WITHOUT being factors (bookkeeping only).
    "nuisance_fields": ["runtime.experiment_name"],
    # Fields that MUST be identical, reported explicitly because the caller
    # needs to act on them (shared cache -> populate once; threads -> ncpus).
    "reported_shared_fields": [
        "runtime.cache_dir",
        "runtime.out_dir",
        "runtime.torch_threads",
        "runtime.seed",
        "train.n_seeds",
        "train.max_epochs",
    ],
    # Fields whose value must be unique across configs.
    "unique_fields": ["runtime.experiment_name"],
    # Fields compared as canonically ordered tuples (order-insensitive).
    "order_insensitive_fields": ["backbone.head_pool_ops"],
}

# Canonical order backbone.py imposes on the pooling ops.
_POOL_ORDER = ("mean", "max", "std")


# ----------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------
def flatten(obj, prefix=""):
    """Flatten a nested dict to {dotted.path: value}. Lists are LEAVES.

    Lists are deliberately not descended into: head_pool_ops is a single
    setting, not three independent ones, and reporting it as
    'backbone.head_pool_ops.0' would make the contrast report unreadable.
    """
    out = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = "%s.%s" % (prefix, key) if prefix else str(key)
            out.update(flatten(value, path))
    else:
        out[prefix] = obj
    return out


def canonical_value(field, value, order_insensitive_fields):
    """Return a hashable, comparison-ready form of a config value.

    For a field declared order-insensitive, a list is reduced to a tuple in
    the canonical pooling order (unknown entries are appended, sorted, so an
    invalid op still compares deterministically rather than being dropped).
    Every other value is rendered via json.dumps, which gives a stable string
    for dicts and lists and distinguishes True from 1 and None from "null".
    """
    if field in order_insensitive_fields and isinstance(value, list):
        known = [op for op in _POOL_ORDER if op in value]
        unknown = sorted(str(v) for v in value if v not in _POOL_ORDER)
        return tuple(known + unknown)
    if isinstance(value, list):
        return tuple(json.dumps(v, sort_keys=True) for v in value)
    return json.dumps(value, sort_keys=True)


def _level_key(values, fields, order_insensitive_fields):
    """Canonical key for one factor level, aligned field-by-field."""
    return tuple(canonical_value(f, v, order_insensitive_fields)
                 for f, v in zip(fields, values))


def load_configs(paths):
    """Read JSON configs. Returns (ordered list of (name, flat_dict), errors)."""
    loaded, errors = [], []
    for path in paths:
        name = os.path.basename(path)
        try:
            with open(path, "r") as handle:
                raw = json.load(handle)
        except (IOError, OSError) as exc:
            errors.append("cannot read %s: %s" % (name, exc))
            continue
        except ValueError as exc:
            errors.append("%s is not valid JSON: %s" % (name, exc))
            continue
        if not isinstance(raw, dict):
            errors.append("%s does not contain a JSON object" % name)
            continue
        loaded.append((name, flatten(raw)))
    return loaded, errors


# ----------------------------------------------------------------------
# The audit
# ----------------------------------------------------------------------
def audit(configs, design):
    """Run all four checks. Returns a result dict; never raises on a design
    violation (violations are data, not exceptions), only on a malformed
    design specification."""
    factors = design["factors"]
    oi_fields = list(design.get("order_insensitive_fields", []))
    nuisance = set(design.get("nuisance_fields", []))
    unique_fields = list(design.get("unique_fields", []))
    reported = list(design.get("reported_shared_fields", []))

    factor_fields = []
    for spec in factors.values():
        factor_fields.extend(spec["fields"])
    if len(set(factor_fields)) != len(factor_fields):
        raise ValueError("a field appears in more than one factor: %r"
                         % (factor_fields,))

    errors, warnings = [], []
    names = [name for name, _ in configs]

    # ---- expected cell count -----------------------------------------
    n_expected = 1
    for spec in factors.values():
        n_expected *= len(spec["levels"])
    if len(configs) != n_expected:
        errors.append(
            "expected %d configs (the full crossing), got %d: %s"
            % (n_expected, len(configs), ", ".join(names)))

    # ---- check 1: identical key sets ---------------------------------
    all_keys = set()
    for _, flat in configs:
        all_keys |= set(flat)
    for name, flat in configs:
        missing = sorted(all_keys - set(flat))
        if missing:
            errors.append("%s is MISSING %d key(s) present elsewhere: %s"
                          % (name, len(missing), ", ".join(missing[:8])))

    # ---- check 2: contrast set ---------------------------------------
    differing = []
    for key in sorted(all_keys):
        seen = set()
        for _, flat in configs:
            if key in flat:
                seen.add(canonical_value(key, flat[key], oi_fields))
        if len(seen) > 1:
            differing.append(key)

    allowed = set(factor_fields) | nuisance
    unexpected = [k for k in differing if k not in allowed]
    for key in unexpected:
        values = ["%s=%s" % (name, json.dumps(flat.get(key)))
                  for name, flat in configs]
        errors.append("UNCONTROLLED field differs across cells: %s  [%s]"
                      % (key, "; ".join(values)))

    inert = [f for f in factor_fields if f not in differing]
    for key in inert:
        warnings.append("declared factor field %s does not vary across the "
                        "configs (constant, so it is not a factor here)" % key)

    # ---- check 3: crossing -------------------------------------------
    level_lookup = {}
    for fname, spec in factors.items():
        for level in spec["levels"]:
            key = _level_key(level["values"], spec["fields"], oi_fields)
            level_lookup[(fname, key)] = level["label"]

    cells = {}
    for name, flat in configs:
        labels = {}
        for fname, spec in factors.items():
            observed = tuple(
                canonical_value(f, flat.get(f), oi_fields)
                for f in spec["fields"])
            label = level_lookup.get((fname, observed))
            if label is None:
                errors.append(
                    "%s has an UNDECLARED level of factor '%s': %s"
                    % (name, fname,
                       ", ".join("%s=%s" % (f, json.dumps(flat.get(f)))
                                 for f in spec["fields"])))
                label = "<undeclared>"
            labels[fname] = label
        cells[name] = labels

    order = list(factors)
    seen_cells = {}
    for name, labels in cells.items():
        key = tuple(labels[f] for f in order)
        seen_cells.setdefault(key, []).append(name)
    for key, owners in sorted(seen_cells.items()):
        if len(owners) > 1:
            errors.append("DUPLICATE cell %s realised by: %s"
                          % (" x ".join(key), ", ".join(sorted(owners))))

    expected_cells = [()]
    for fname in order:
        expected_cells = [c + (lvl["label"],)
                          for c in expected_cells
                          for lvl in factors[fname]["levels"]]
    for key in expected_cells:
        if key not in seen_cells:
            errors.append("MISSING cell: %s" % " x ".join(key))

    # ---- check 4: hygiene --------------------------------------------
    for field in unique_fields:
        buckets = {}
        for name, flat in configs:
            buckets.setdefault(json.dumps(flat.get(field)), []).append(name)
        for value, owners in sorted(buckets.items()):
            if len(owners) > 1:
                errors.append("%s must be unique per config; %s shared by: %s"
                              % (field, value, ", ".join(sorted(owners))))

    shared = {}
    for field in reported:
        values = set()
        for _, flat in configs:
            values.add(json.dumps(flat.get(field)))
        shared[field] = (sorted(values)[0] if len(values) == 1
                         else "<VARIES: %s>" % ", ".join(sorted(values)))

    return {
        "ok": not errors,
        "n_configs": len(configs),
        "n_expected": n_expected,
        "names": names,
        "factor_order": order,
        "cells": cells,
        "differing_fields": differing,
        "unexpected_fields": unexpected,
        "shared": shared,
        "errors": errors,
        "warnings": warnings,
    }


def format_report(result, design):
    """Render the audit result as an ASCII report."""
    lines = []
    lines.append("FACTORIAL AUDIT: %s" % design.get("name", "(unnamed design)"))
    lines.append("  configs audited : %d (expected %d)"
                 % (result["n_configs"], result["n_expected"]))
    lines.append("")

    lines.append("CELL MAP")
    order = result["factor_order"]
    width = max([len(n) for n in result["names"]] + [10])
    header = "  %-*s  %s" % (width, "config",
                             "  ".join("%-14s" % f for f in order))
    lines.append(header)
    for name in sorted(result["names"]):
        labels = result["cells"].get(name, {})
        lines.append("  %-*s  %s"
                     % (width, name,
                        "  ".join("%-14s" % labels.get(f, "?") for f in order)))
    lines.append("")

    lines.append("FIELDS THAT DIFFER ACROSS CELLS (%d)"
                 % len(result["differing_fields"]))
    for key in result["differing_fields"]:
        tag = "  UNEXPECTED" if key in result["unexpected_fields"] else "  ok"
        lines.append("  %-42s %s" % (key, tag))
    lines.append("")

    lines.append("SHARED SETTINGS (identical by design; act on these)")
    for field in sorted(result["shared"]):
        lines.append("  %-28s %s" % (field, result["shared"][field]))
    lines.append("")

    if result["warnings"]:
        lines.append("WARNINGS")
        for msg in result["warnings"]:
            lines.append("  ! %s" % msg)
        lines.append("")

    if result["errors"]:
        lines.append("ERRORS (%d)" % len(result["errors"]))
        for msg in result["errors"]:
            lines.append("  X %s" % msg)
        lines.append("")
        lines.append("AUDIT FAILED -- do not submit until these are resolved.")
    else:
        lines.append("AUDIT PASSED -- the contrast is clean.")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def resolve_paths(args_paths):
    """Expand a directory argument into its *.json files; pass files through."""
    paths = []
    for item in args_paths:
        if os.path.isdir(item):
            found = sorted(glob.glob(os.path.join(item, "*.json")))
            if not found:
                return [], "no *.json files in directory %s" % item
            paths.extend(found)
        else:
            paths.append(item)
    return paths, None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit a set of DSN configs as a factorial design.")
    parser.add_argument("paths", nargs="+",
                        help="a directory of configs, or explicit .json files")
    parser.add_argument("--design", default=None,
                        help="JSON file with an alternative design spec")
    parser.add_argument("--quiet", action="store_true",
                        help="print nothing; communicate via exit code only")
    args = parser.parse_args(argv)

    design = DEFAULT_DESIGN
    if args.design:
        try:
            with open(args.design, "r") as handle:
                design = json.load(handle)
        except (IOError, OSError, ValueError) as exc:
            sys.stderr.write("cannot read design %s: %s\n"
                             % (args.design, exc))
            return 2

    paths, err = resolve_paths(args.paths)
    if err:
        sys.stderr.write("%s\n" % err)
        return 2

    configs, load_errors = load_configs(paths)
    if load_errors:
        for msg in load_errors:
            sys.stderr.write("X %s\n" % msg)
        return 1
    if not configs:
        sys.stderr.write("no configs to audit\n")
        return 2

    result = audit(configs, design)
    if not args.quiet:
        print(format_report(result, design))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
