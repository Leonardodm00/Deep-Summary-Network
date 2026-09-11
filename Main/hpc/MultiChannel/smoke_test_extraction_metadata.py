"""Smoke test: the extractor records every preprocessing parameter it used.

Run:  python3 smoke_test_extraction_metadata.py

Runs run_channel_subset_extraction.py END TO END on a synthetic ptrain
folder (two 3x3 clusters, as smoke_test_channel_subsets_stage5 builds) in
per_region_single mode with NON-default parameters, then checks that:

  E1  traces.npz and every trace_subregion_XX.npz carry w_size,
      gaussian_window, sigma_sm_bins, electrodes_per_subset, n_subsets,
      mfr_threshold, fs_raw, extractor_version -- with the values PASSED,
      not the defaults;
  E2  traces_meta.json exists beside them and agrees with the npz;
  E3  fs_ifr in the archive equals 1 / w_size;
  E4  the pure extraction_metadata() records the pre-patch defaults
      (0.02 / 0.04) as such, so an unpinned run is at least visible;
  E5  the archives are still readable by the downstream consumer's key set
      (ifr_trace, fs_ifr, T_rec, culture_id, subregion_index).

Pure ASCII, LF only. numpy + scipy only.
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import scipy.io as sio

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import run_channel_subset_extraction as R  # noqa: E402

W = 48
RESULTS = []


def ok(name, cond, detail):
    RESULTS.append(bool(cond))
    print("[%s] %-58s %s" % ("PASS" if cond else "FAIL", name, detail))


def _write_ptrain(folder, idx, spike_samples, n_samples):
    raster = np.zeros((n_samples, 1), dtype=np.uint8)
    raster[np.asarray(spike_samples, dtype=np.int64), 0] = 1
    sio.savemat(os.path.join(folder, "ptrain_%d.mat" % idx),
                {"ptrain": raster}, do_compression=True)


def _block(r0, c0, half):
    return [(r0 + dr) * W + (c0 + dc)
            for dr in range(-half, half + 1) for dc in range(-half, half + 1)]


def make_folder(folder, n_samples=20000, seed=11):
    rng = np.random.default_rng(seed)
    for i in _block(10, 10, 1):
        _write_ptrain(folder, i, np.sort(rng.integers(0, n_samples, 60)), n_samples)
    for i in _block(30, 30, 1):
        _write_ptrain(folder, i, np.sort(rng.integers(0, n_samples, 30)), n_samples)


PARAMS = dict(w_size=0.01, gaussian_window=0.02, n_subsets=2,
              electrodes_per_subset=9, mfr_threshold=0.1, fs_raw=1000.0)


def test_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "ptrain_TEST")
        out = os.path.join(td, "out")
        os.makedirs(src)
        make_folder(src)
        cmd = [sys.executable, os.path.join(_HERE, "run_channel_subset_extraction.py"),
               src, "--out-dir", out, "--mode", "per_region_single",
               "--w-size", str(PARAMS["w_size"]),
               "--gaussian-window", str(PARAMS["gaussian_window"]),
               "--n-subsets", str(PARAMS["n_subsets"]),
               "--electrodes-per-subset", str(PARAMS["electrodes_per_subset"]),
               "--mfr-threshold", str(PARAMS["mfr_threshold"]),
               "--fs-raw", str(PARAMS["fs_raw"]), "--base", "0", "--no-plots"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        ok("E0 the extractor runs end to end on a synthetic folder",
           r.returncode == 0, (r.stderr.strip().splitlines() or ["rc=0"])[-1])
        if r.returncode != 0:
            return

        files = sorted(f for f in os.listdir(out) if f.endswith(".npz"))
        subs = [f for f in files if f.startswith("trace_subregion_")]
        ok("E1a traces.npz + one archive per subregion were written",
           "traces.npz" in files and len(subs) == PARAMS["n_subsets"],
           "%s" % files)

        keys = ("w_size", "gaussian_window", "sigma_sm_bins",
                "electrodes_per_subset", "n_subsets", "mfr_threshold",
                "fs_raw", "extractor_version")
        bad = []
        for f in files:
            with np.load(os.path.join(out, f), allow_pickle=False) as d:
                for k in keys:
                    if k not in d.files:
                        bad.append("%s lacks %s" % (f, k))
                if "w_size" in d.files:
                    got = (float(d["w_size"]), float(d["gaussian_window"]),
                           int(d["electrodes_per_subset"]), float(d["mfr_threshold"]))
                    want = (PARAMS["w_size"], PARAMS["gaussian_window"],
                            PARAMS["electrodes_per_subset"], PARAMS["mfr_threshold"])
                    if got != want:
                        bad.append("%s: %r != %r" % (f, got, want))
                    if abs(float(d["sigma_sm_bins"]) - 2.0) > 1e-9:
                        bad.append("%s: sigma_sm_bins %r" % (f, float(d["sigma_sm_bins"])))
        ok("E1b every archive carries the parameters PASSED, not the defaults",
           not bad, bad[:3] or "%d files x %d keys" % (len(files), len(keys)))

        mp = os.path.join(out, "traces_meta.json")
        ok("E2a traces_meta.json exists beside the archives", os.path.isfile(mp), mp)
        if os.path.isfile(mp):
            j = json.load(open(mp))
            with np.load(os.path.join(out, "traces.npz"), allow_pickle=False) as d:
                agree = all(abs(float(j[k]) - float(d[k])) < 1e-12
                            for k in ("w_size", "gaussian_window", "fs_ifr", "T_rec"))
            ok("E2b traces_meta.json agrees with the npz and carries provenance",
               agree and "argv" in j and "source_folder" in j
               and j["extractor_version"] == R.EXTRACTOR_VERSION,
               "keys: %s" % sorted(j)[:6])

        with np.load(os.path.join(out, "traces.npz"), allow_pickle=False) as d:
            ok("E3 fs_ifr == 1 / w_size in the archive",
               abs(float(d["fs_ifr"]) * float(d["w_size"]) - 1.0) < 1e-9,
               "fs_ifr=%.3f w_size=%.4f" % (float(d["fs_ifr"]), float(d["w_size"])))

        with np.load(os.path.join(out, subs[0]), allow_pickle=False) as d:
            need = ("ifr_trace", "fs_ifr", "T_rec", "culture_id", "subregion_index")
            ok("E5 the consumer's key set is intact on a subregion archive",
               all(k in d.files for k in need) and d["ifr_trace"].ndim == 1,
               "culture_id=%s subregion=%d" % (d["culture_id"], int(d["subregion_index"])))


def test_defaults_visible():
    class A(object):
        folder = "/x"; w_size = 0.02; gaussian_window = 0.04; fs_raw = 10110.09
        n_subsets = 9; electrodes_per_subset = 9; mfr_threshold = 0.1
    m = R.extraction_metadata(A(), fs_ifr=50.0, argv=["run"])
    ok("E4 an unpinned run records the defaults 0.02 / 0.04 explicitly",
       m["w_size"] == 0.02 and m["gaussian_window"] == 0.04
       and abs(m["sigma_sm_bins"] - 2.0) < 1e-12,
       "what a run without --w-size/--gaussian-window silently used")


def main():
    print("=" * 84)
    print("Smoke test: extraction metadata (run_channel_subset_extraction v2, end to end)")
    print("=" * 84)
    test_end_to_end()
    test_defaults_visible()
    print("-" * 84)
    n_fail = RESULTS.count(False)
    print("%d passed, %d failed" % (RESULTS.count(True), n_fail))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
