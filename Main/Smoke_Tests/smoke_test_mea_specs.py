"""
smoke_test_mea_specs.py
=======================

Verification for CohortConfig + make_mea_specs.py -- the bridge from the real
MEA directory tree to `data.npz_specs`.

Checks:

    A. COHORT ROUND-TRIPS SILENTLY. CohortConfig survives to_dict/from_dict and
       emits NO warning. This matters because preflight_config.py treats "no
       warnings" as a pass signal, so a noisy config block would poison it.
    B. VALIDATION RAISES on non-contiguous class indices, a bare string instead
       of a list of roots, an empty class, a duplicated root, and one root
       claimed by two classes.
    C. INVENTORY IS CORRECT on a fixture with the real geometry
       (2 classes x 3 roots x 6 wells x 9 subregions): 36 cultures, 324 records,
       every record of a well sharing one culture id.
    D. BOTH EXTRACTION MODES are recognised, and a cohort that MIXES them is
       refused (they imply different data.n_channels).
    E. MISSING OUTPUT IS REPORTED, not silently dropped, and --strict turns it
       into a non-zero exit.
    F. THE SPECS FEED THE PIPELINE. The generated file is consumed by
       build_traces/build_splits and yields the culture count the inventory
       promised, with no culture spanning two splits.

Run
---
    cd Main
    PYTHONPATH="$(pwd)" python3 Smoke_Tests/smoke_test_mea_specs.py

Exit status 0 only if every check passes. No .mat files, no GPU, no network:
the fixture writes tiny .npz archives with the real key names.
"""

import json
import os
import shutil
import sys
import tempfile
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAIN = os.path.dirname(_HERE)
if _MAIN not in sys.path:
    sys.path.insert(0, _MAIN)

from config import CohortConfig, ExperimentConfig                # noqa: E402
import make_mea_specs as MMS                                     # noqa: E402

WELLS = ["ptrain_A1", "ptrain_A2", "ptrain_A3",
         "ptrain_B1", "ptrain_B2", "ptrain_B3"]
CLASS_NAMES = ["control", "pathological"]
N_ROOTS, N_SUB, K = 3, 9, 12000          # K small: nothing here needs 60k
FS = 50.0


def _fail(msg):
    raise AssertionError(msg)


def build_fixture(base, mode="per_region_single", skip_wells=(),
                  mixed_well=None):
    """Write a cohort tree. Returns the cohort dict for a config."""
    raw, ext = os.path.join(base, "raw"), os.path.join(base, "extracted")
    roots = {"0": [], "1": []}
    rng = np.random.default_rng(0)
    for c in (0, 1):
        for p in range(N_ROOTS):
            rn = "plate_%s_%d" % (CLASS_NAMES[c][:4], p)
            root = os.path.join(raw, rn)
            roots[str(c)].append(root)
            # [root_name_for] the SAME function build_records() calls, so this
            # fixture cannot silently diverge from where production code
            # actually looks the moment that function's convention changes.
            layout_rn = MMS.root_name_for(root)
            for w in WELLS:
                os.makedirs(os.path.join(root, w), exist_ok=True)
                if (rn, w) in skip_wells:
                    continue           # raw well exists, extraction output not
                od = os.path.join(ext, CLASS_NAMES[c], layout_rn, w)
                os.makedirs(od, exist_ok=True)
                this = "multichannel" if (rn, w) == mixed_well else mode
                if this == "per_region_single":
                    for j in range(N_SUB):
                        np.savez_compressed(
                            os.path.join(od, "trace_subregion_%02d.npz" % j),
                            ifr_trace=rng.gamma(2.0, 0.5, K).astype(np.float32),
                            fs_ifr=np.float64(FS), in_channels=1,
                            subregion_index=j, culture_id="%s__%s" % (rn, w))
                else:
                    np.savez_compressed(
                        os.path.join(od, "traces.npz"),
                        ifr_trace=rng.gamma(2.0, 0.5,
                                            (N_SUB, K)).astype(np.float32),
                        fs_ifr=np.float64(FS), in_channels=N_SUB,
                        culture_id="%s__%s" % (rn, w))
    return {
        "class_roots": roots, "class_names": list(CLASS_NAMES),
        "well_glob": "ptrain_*", "extract_root": ext,
        "extract_layout": "{class_name}/{root_name}/{well}",
        "culture_template": "{root_name}__{well}",
        "n_subsets": N_SUB, "electrodes_per_subset": 9,
        "fs_raw": 10110.09, "grid_width": 48, "index_base": 0,
        "mfr_threshold": 0.1, "w_size": 0.02,
    }


def write_config(base, cohort, n_channels=1, window_s=180.0):
    cfg = ExperimentConfig().to_dict()
    cfg["cohort"] = cohort
    cfg["data"].update({
        "data_mode": "numpy", "npz_specs": os.path.join(base, "specs.json"),
        "n_channels": n_channels, "window_s": window_s,
        "train_stride_s": window_s, "eval_stride_s": window_s,
        "split_fractions": [0.6, 0.2, 0.2], "split_mode": "trace",
        "min_train_cultures_per_class": 2,
        "positives_mode": "cross_culture",
        "cultures_per_class_per_batch": 9,
        "windows_per_culture_per_batch": 1,
        "exclude_same_culture_positives": True,
    })
    cfg["data"]["augmentation"]["fs"] = FS
    # cross_culture REQUIRES n_positives == 0: positives come from other
    # cultures, not from warps of the anchor's own window.
    cfg["data"]["augmentation"]["n_positives"] = 0
    cfg["data"]["augmentation"]["n_negatives"] = 2
    cfg["runtime"]["cache_dir"] = os.path.join(base, "cache")
    cfg["runtime"]["out_dir"] = os.path.join(base, "out")
    p = os.path.join(base, "cfg.json")
    with open(p, "w", encoding="ascii") as fh:
        json.dump(cfg, fh, indent=2)
    return p


# --------------------------------------------------------------------------- #
def check_A_round_trip_is_silent():
    c = CohortConfig(class_roots={"0": ["/a", "/b"], "1": ["/c"]},
                     class_names=["ctrl", "path"])
    cfg = ExperimentConfig(cohort=c)
    d = json.loads(json.dumps(cfg.to_dict()))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = ExperimentConfig.from_dict(d)
    if back.cohort.class_roots != c.class_roots:
        _fail("A: class_roots did not survive the round trip")
    if back.cohort.n_classes() != 2:
        _fail("A: n_classes wrong after round trip")
    if abs(back.cohort.fs_ifr() - 50.0) > 1e-12:
        _fail("A: fs_ifr should be 1/w_size = 50 Hz")
    # an EMPTY cohort must also be silent -- every pre-existing config has one
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ExperimentConfig.from_dict(json.loads(json.dumps(
            ExperimentConfig().to_dict())))
    print("  [A] PASS  cohort round-trips, no warnings, empty cohort silent")


def check_B_validation_raises():
    cases = [
        ({"0": ["/a"], "2": ["/b"]}, "non-contiguous class indices"),
        ({"0": "/a"}, "bare string instead of a list"),
        ({"0": []}, "empty class"),
        ({"0": ["/a", "/a"]}, "duplicate root in one class"),
        ({"0": ["/a"], "1": ["/a"]}, "root under two classes"),
    ]
    for roots, why in cases:
        try:
            CohortConfig(class_roots=roots)
        except ValueError:
            continue
        _fail("B: did NOT raise on %s" % why)
    try:
        CohortConfig(class_roots={"0": ["/a"], "1": ["/b"]},
                     class_names=["only_one"])
    except ValueError:
        pass
    else:
        _fail("B: did NOT raise on class_names of the wrong length")
    try:
        CohortConfig(class_roots={"0": ["/a"]}, index_base=7)
    except ValueError:
        pass
    else:
        _fail("B: did NOT raise on index_base outside {0,1}")
    print("  [B] PASS  all 7 malformed cohorts raise")


def check_C_inventory():
    base = tempfile.mkdtemp(prefix="mea_specs_C_")
    try:
        cohort = build_fixture(base)
        cfg_path = write_config(base, cohort)
        rc = MMS.main(["--config", cfg_path])
        if rc != 0:
            _fail("C: generator exited %d" % rc)
        recs = json.load(open(os.path.join(base, "specs.json")))
        n_wells = 2 * N_ROOTS * len(WELLS)
        if len(recs) != n_wells * N_SUB:
            _fail("C: %d records, expected %d" % (len(recs), n_wells * N_SUB))
        cults = set(r["culture"] for r in recs)
        if len(cults) != n_wells:
            _fail("C: %d cultures, expected %d" % (len(cults), n_wells))
        if len(set(r["name"] for r in recs)) != len(recs):
            _fail("C: record names are not unique")
        by_c = {}
        for r in recs:
            by_c.setdefault(r["culture"], set()).add(r["condition"])
        for cu, labels in by_c.items():
            if len(labels) != 1:
                _fail("C: culture %r spans conditions %r" % (cu, labels))
        sizes = set(len([r for r in recs if r["culture"] == cu]) for cu in cults)
        if sizes != {N_SUB}:
            _fail("C: cultures have %r records each, expected {%d}"
                  % (sizes, N_SUB))
        print("  [C] PASS  %d records over %d cultures, %d per well, labels "
              "consistent" % (len(recs), len(cults), N_SUB))
    finally:
        shutil.rmtree(base, ignore_errors=True)


def check_D_modes():
    base = tempfile.mkdtemp(prefix="mea_specs_D_")
    try:
        cohort = build_fixture(base, mode="multichannel")
        cfg_path = write_config(base, cohort, n_channels=N_SUB)
        if MMS.main(["--config", cfg_path]) != 0:
            _fail("D: multichannel cohort was rejected")
        recs = json.load(open(os.path.join(base, "specs.json")))
        n_wells = 2 * N_ROOTS * len(WELLS)
        if len(recs) != n_wells:
            _fail("D: multichannel should give 1 record/well, got %d for %d "
                  "wells" % (len(recs), n_wells))
    finally:
        shutil.rmtree(base, ignore_errors=True)

    base = tempfile.mkdtemp(prefix="mea_specs_D2_")
    try:
        cohort = build_fixture(base, mode="per_region_single",
                               mixed_well=("plate_cont_0", "ptrain_A2"))
        cfg_path = write_config(base, cohort)
        if MMS.main(["--config", cfg_path]) == 0:
            _fail("D: a cohort MIXING extraction modes was accepted")
    finally:
        shutil.rmtree(base, ignore_errors=True)
    print("  [D] PASS  both modes recognised; mixed cohort refused")


def check_E_missing_reported():
    base = tempfile.mkdtemp(prefix="mea_specs_E_")
    try:
        skip = {("plate_cont_1", "ptrain_B3"), ("plate_path_2", "ptrain_A1")}
        cohort = build_fixture(base, skip_wells=skip)
        cfg_path = write_config(base, cohort)
        recs_expected = (2 * N_ROOTS * len(WELLS) - len(skip)) * N_SUB
        if MMS.main(["--config", cfg_path]) != 0:
            _fail("E: non-strict run should succeed with what exists")
        recs = json.load(open(os.path.join(base, "specs.json")))
        if len(recs) != recs_expected:
            _fail("E: %d records, expected %d with %d well(s) skipped"
                  % (len(recs), recs_expected, len(skip)))
        if MMS.main(["--config", cfg_path, "--strict"]) != 3:
            _fail("E: --strict should exit 3 when wells are missing")
    finally:
        shutil.rmtree(base, ignore_errors=True)
    print("  [E] PASS  missing wells reported; --strict exits 3")


def check_F_feeds_the_pipeline():
    import run_optimization as RO
    base = tempfile.mkdtemp(prefix="mea_specs_F_")
    try:
        cohort = build_fixture(base)
        # K = 12000 at 50 Hz = 240 s, so a 60 s window still gives 4/trace
        cfg_path = write_config(base, cohort, window_s=60.0)
        if MMS.main(["--config", cfg_path]) != 0:
            _fail("F: generator failed")
        cfg = ExperimentConfig.from_json(cfg_path)
        traces, conds, fs = RO.build_traces(cfg, verbose=False)
        cultures = RO.build_cultures(cfg)
        n_wells = 2 * N_ROOTS * len(WELLS)
        if len(traces) != n_wells * N_SUB:
            _fail("F: %d traces cached, expected %d"
                  % (len(traces), n_wells * N_SUB))
        if len(set(cultures)) != n_wells:
            _fail("F: %d distinct cultures, expected %d"
                  % (len(set(cultures)), n_wells))
        splits = RO.build_splits(cfg, traces, conds, fs, cultures=cultures)
        seen = {}
        for nm in ("train", "val", "test"):
            seen[nm] = set(np.asarray(splits.trace_of_window[nm]).tolist())
        leak = ((seen["train"] & seen["test"]) | (seen["train"] & seen["val"])
                | (seen["val"] & seen["test"]))
        if leak:
            _fail("F: culture(s) %r span two splits" % sorted(leak)[:5])
        total = sum(len(splits.cultures[n]) for n in ("train", "val", "test"))
        if total != n_wells:
            _fail("F: %d cultures assigned, expected %d" % (total, n_wells))
        print("  [F] PASS  specs drive the pipeline: %d traces -> %d cultures, "
              "no leakage" % (len(traces), n_wells))
    finally:
        shutil.rmtree(base, ignore_errors=True)


def main():
    print("CohortConfig / make_mea_specs smoke test")
    print("  fixture: 2 classes x %d roots x %d wells x %d subregions"
          % (N_ROOTS, len(WELLS), N_SUB))
    checks = [check_A_round_trip_is_silent, check_B_validation_raises,
              check_C_inventory, check_D_modes, check_E_missing_reported,
              check_F_feeds_the_pipeline]
    for fn in checks:
        fn()
    print("ALL CHECKS PASSED (%d)" % len(checks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
