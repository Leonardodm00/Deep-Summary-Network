"""
smoke_test_audit_factorial.py

Standalone correctness harness for audit_factorial.py. Requires no data, no
conda environment, no cluster: it synthesises small config sets in a temporary
directory, mutates them one defect at a time, and asserts that the auditor
accepts the clean case and rejects each defect for the RIGHT reason.

The point of a smoke test for a checker is adversarial: a checker that never
fails is indistinguishable from a checker that always passes, so every check
below pairs a positive case with the specific corruption it must catch.

Checks
------
  [1]  clean 3 x 2 factorial passes
  [2]  head_pool_ops reordering is NOT a violation (canonical order)
  [3]  an uncontrolled field (train.margin) is caught
  [4]  a missing cell is caught
  [5]  a duplicate cell (and the missing cell it implies) is caught
  [6]  a key present in one config and absent in another is caught
  [7]  a duplicated runtime.experiment_name is caught
  [8]  an undeclared factor level is caught
  [9]  shared-field reporting distinguishes identical from varying
  [10] the CLI returns 0 on the clean set and 1 on the corrupted set

Usage
-----
    python3 smoke_test_audit_factorial.py            # all checks
    python3 smoke_test_audit_factorial.py --quick    # checks 1, 3, 10 only

Exit code 0 and "ALL SMOKE TESTS PASSED" on success; any failure raises with a
diagnostic naming the offending check.

HPC note (hpc-python-compat): pure ASCII, LF endings, stdlib only.
"""

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import audit_factorial as af                                   # noqa: E402


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
HEAD_LEVELS = [
    ("singlemean", False, ["mean"]),
    ("multimean", True, ["mean"]),
    ("multiall", True, ["mean", "max", "std"]),
]
MINER_LEVELS = [
    ("h", "hard"),
    ("epsh", "easy_pos_semihard_neg"),
]


def _base_config():
    """A minimal config with the same nesting shape as the real ones."""
    return {
        "data": {"window_s": 100.0, "max_group_size": 16,
                 "split_fractions": [0.6, 0.2, 0.2]},
        "backbone": {"depth_exponent": 4, "head_fusion": False,
                     "head_pool_ops": ["mean"], "embedding_size": 16},
        "train": {"margin": 0.3, "mining_strategy": "hard", "n_seeds": 2,
                  "max_epochs": 50},
        "runtime": {"seed": 0, "torch_threads": 48, "cache_dir": "/c",
                    "out_dir": "/o", "experiment_name": "x"},
    }


def make_clean_set():
    """The full 3 x 2 crossing as {filename: config-dict}."""
    out = {}
    for miner_tag, miner_value in MINER_LEVELS:
        for head_tag, fusion, ops in HEAD_LEVELS:
            cfg = _base_config()
            cfg["backbone"]["head_fusion"] = fusion
            cfg["backbone"]["head_pool_ops"] = list(ops)
            cfg["train"]["mining_strategy"] = miner_value
            name = "config_%s_%s.json" % (miner_tag, head_tag)
            cfg["runtime"]["experiment_name"] = "%s_%s" % (miner_tag, head_tag)
            out[name] = cfg
    return out


def write_set(configs, directory):
    for name, cfg in configs.items():
        with open(os.path.join(directory, name), "w") as handle:
            json.dump(cfg, handle, indent=2)
    return sorted(os.path.join(directory, n) for n in configs)


def run_audit(configs):
    """Audit an in-memory config set without touching the filesystem."""
    flat = [(name, af.flatten(cfg)) for name, cfg in sorted(configs.items())]
    return af.audit(flat, af.DEFAULT_DESIGN)


def _has_error(result, needle):
    return any(needle in msg for msg in result["errors"])


def _fail(check, message, result=None):
    detail = ""
    if result is not None:
        detail = "\n    errors: %s" % json.dumps(result["errors"], indent=6)
    raise AssertionError("[%s] %s%s" % (check, message, detail))


# ----------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------
def check_clean_passes():
    result = run_audit(make_clean_set())
    if not result["ok"]:
        _fail(1, "clean factorial was rejected", result)
    if result["n_configs"] != 6 or result["n_expected"] != 6:
        _fail(1, "wrong cell count: %r" % (result["n_configs"],), result)
    if sorted(result["differing_fields"]) != sorted([
            "backbone.head_fusion", "backbone.head_pool_ops",
            "runtime.experiment_name", "train.mining_strategy"]):
        _fail(1, "unexpected contrast set: %r" % (result["differing_fields"],))
    print("  [1] clean 3 x 2 factorial passes; contrast set is exactly the "
          "4 intended fields")


def check_pool_op_order_insensitive():
    configs = make_clean_set()
    for name, cfg in configs.items():
        if cfg["backbone"]["head_pool_ops"] == ["mean", "max", "std"]:
            cfg["backbone"]["head_pool_ops"] = ["std", "mean", "max"]
    result = run_audit(configs)
    if not result["ok"]:
        _fail(2, "reordered head_pool_ops was treated as a violation", result)
    labels = set(c["head"] for c in result["cells"].values())
    if "multiall" not in labels:
        _fail(2, "reordered ops lost the multiall label: %r" % (labels,))
    print("  [2] head_pool_ops reordering is canonicalised, not flagged")


def check_uncontrolled_field_caught():
    configs = make_clean_set()
    victim = sorted(configs)[0]
    configs[victim]["train"]["margin"] = 0.9
    result = run_audit(configs)
    if result["ok"]:
        _fail(3, "an uncontrolled train.margin was NOT caught")
    if not _has_error(result, "UNCONTROLLED field differs"):
        _fail(3, "caught, but not as an uncontrolled field", result)
    if "train.margin" not in result["unexpected_fields"]:
        _fail(3, "train.margin missing from unexpected_fields", result)
    print("  [3] an uncontrolled field (train.margin) is caught")


def check_missing_cell_caught():
    configs = make_clean_set()
    del configs[sorted(configs)[0]]
    result = run_audit(configs)
    if result["ok"]:
        _fail(4, "a missing cell was NOT caught")
    if not _has_error(result, "MISSING cell"):
        _fail(4, "caught, but not as a missing cell", result)
    if not _has_error(result, "expected 6 configs"):
        _fail(4, "cell count was not reported", result)
    print("  [4] a missing cell is caught")


def check_duplicate_cell_caught():
    configs = make_clean_set()
    names = sorted(configs)
    donor, target = names[0], names[1]
    configs[target]["backbone"] = copy.deepcopy(configs[donor]["backbone"])
    configs[target]["train"]["mining_strategy"] = \
        configs[donor]["train"]["mining_strategy"]
    result = run_audit(configs)
    if result["ok"]:
        _fail(5, "a duplicate cell was NOT caught")
    if not _has_error(result, "DUPLICATE cell"):
        _fail(5, "caught, but not as a duplicate cell", result)
    if not _has_error(result, "MISSING cell"):
        _fail(5, "duplicate did not also report the cell it displaced", result)
    print("  [5] a duplicate cell is caught, and the displaced cell reported")


def check_missing_key_caught():
    configs = make_clean_set()
    del configs[sorted(configs)[0]]["train"]["n_seeds"]
    result = run_audit(configs)
    if result["ok"]:
        _fail(6, "a config missing a key was NOT caught")
    if not _has_error(result, "is MISSING"):
        _fail(6, "caught, but not as a missing key", result)
    print("  [6] a key present in one config and absent in another is caught")


def check_duplicate_experiment_name_caught():
    configs = make_clean_set()
    names = sorted(configs)
    configs[names[1]]["runtime"]["experiment_name"] = \
        configs[names[0]]["runtime"]["experiment_name"]
    result = run_audit(configs)
    if result["ok"]:
        _fail(7, "a duplicated experiment_name was NOT caught")
    if not _has_error(result, "must be unique per config"):
        _fail(7, "caught, but not as a uniqueness violation", result)
    print("  [7] a duplicated runtime.experiment_name is caught "
          "(it would overwrite an output directory)")


def check_undeclared_level_caught():
    configs = make_clean_set()
    configs[sorted(configs)[0]]["train"]["mining_strategy"] = "easy_positive"
    result = run_audit(configs)
    if result["ok"]:
        _fail(8, "an undeclared factor level was NOT caught")
    if not _has_error(result, "UNDECLARED level"):
        _fail(8, "caught, but not as an undeclared level", result)
    print("  [8] an undeclared factor level (easy_positive) is caught")


def check_shared_field_reporting():
    result = run_audit(make_clean_set())
    if result["shared"]["runtime.torch_threads"] != "48":
        _fail(9, "identical shared field misreported: %r"
              % (result["shared"]["runtime.torch_threads"],))
    configs = make_clean_set()
    configs[sorted(configs)[0]]["runtime"]["torch_threads"] = 16
    result = run_audit(configs)
    if "<VARIES" not in result["shared"]["runtime.torch_threads"]:
        _fail(9, "a varying shared field was not flagged as varying")
    if not _has_error(result, "UNCONTROLLED field differs"):
        _fail(9, "a varying torch_threads should also be an uncontrolled "
                 "field error", result)
    print("  [9] shared-field reporting distinguishes identical from varying")


def check_cli_round_trip():
    tmp = tempfile.mkdtemp(prefix="audit_smoke_")
    try:
        clean_dir = os.path.join(tmp, "clean")
        dirty_dir = os.path.join(tmp, "dirty")
        os.makedirs(clean_dir)
        os.makedirs(dirty_dir)

        write_set(make_clean_set(), clean_dir)
        code = af.main([clean_dir, "--quiet"])
        if code != 0:
            _fail(10, "CLI returned %d on the clean set (expected 0)" % code)

        dirty = make_clean_set()
        dirty[sorted(dirty)[0]]["train"]["margin"] = 0.9
        write_set(dirty, dirty_dir)
        code = af.main([dirty_dir, "--quiet"])
        if code != 1:
            _fail(10, "CLI returned %d on the dirty set (expected 1)" % code)

        code = af.main([os.path.join(clean_dir, "nope.json"), "--quiet"])
        if code != 1:
            _fail(10, "CLI returned %d on a missing file (expected 1)" % code)

        code = af.main([os.path.join(tmp, "no_such_dir_or_file")])
        if code != 1:
            _fail(10, "CLI returned %d on a bad path (expected 1)" % code)
        print("  [10] CLI exit codes: 0 clean, 1 dirty, 1 unreadable")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


ALL_CHECKS = [
    ("clean", check_clean_passes, True),
    ("pool_order", check_pool_op_order_insensitive, False),
    ("uncontrolled", check_uncontrolled_field_caught, True),
    ("missing_cell", check_missing_cell_caught, False),
    ("duplicate_cell", check_duplicate_cell_caught, False),
    ("missing_key", check_missing_key_caught, False),
    ("dup_expname", check_duplicate_experiment_name_caught, False),
    ("undeclared", check_undeclared_level_caught, False),
    ("shared", check_shared_field_reporting, False),
    ("cli", check_cli_round_trip, True),
]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Smoke test for audit_factorial.py")
    parser.add_argument("--quick", action="store_true",
                        help="run only the three fastest structural checks")
    args = parser.parse_args(argv)

    print("smoke_test_audit_factorial: %s"
          % ("quick" if args.quick else "full"))
    ran = 0
    for _, func, in_quick in ALL_CHECKS:
        if args.quick and not in_quick:
            continue
        func()
        ran += 1
    print("ALL SMOKE TESTS PASSED (%d checks)" % ran)
    return 0


if __name__ == "__main__":
    sys.exit(main())
