#!/usr/bin/env python3
"""
augmentation_diagnostics.py
===========================

Input-space diagnostic for the contrastive data-augmentation pipeline
(augmentation.py -> build_triplet_instance). It answers, quantitatively, the
question "are the augmentations good?", where "good" is defined operationally by
three checks that matter for triplet-metric learning:

  (1) SEPARATION. In distortion space, are negatives actually more distorted
      away from the anchor than positives? Measured with a threshold-free AUC
      (probability a random negative is more distorted than a random positive)
      plus d-prime, an optimal-threshold confusion, and a leakage fraction.
      The head-line metric is a SHAPE/TIMING distortion (correlation distance),
      which is invariant to pure amplitude rescaling -- so a negative that only
      changed amplitude (magnitude warp) but kept the burst profile is correctly
      seen as "not separated". A pure-MSE distortion is reported alongside for
      continuity with the percentile_mse split (augmentation.py option 2).

  (2) BURST-SCALE REACH. The guiding rule (project theory doc, section 5.2) is:
      to destroy the global activity profile the time distortion must be on the
      network-burst time scale tau_burst; positives a magnitude less. This tool
      estimates tau_burst from the traces (mean observed burst width) and reports
      the fraction of the NEGATIVE time band that sits BELOW tau_burst -- i.e.
      the fraction of negatives drawn too weak to destroy the profile. That is
      the concrete "negatives too similar to the anchor" failure rate, and it is
      computed analytically from the uniform band, so it needs no per-surrogate
      instrumentation.

  (3) HEALTH. Finiteness, non-negativity (physical firing rate >= 0), and that
      the circular shift is label-preserving (a circular roll must preserve the
      multiset of sample values row-by-row).

Separation of concerns (directive 2): this module does data acquisition, metric
computation, and reporting only. It imports the augmentation transforms from the
pipeline rather than reimplementing them, so it always measures the SAME code the
training run consumes. It never trains a model; the embedding-space evaluation
(do positives embed near the anchor under the trained encoder) is a separate,
post-training check and is intentionally out of scope here.

HPC / encoding note (hpc-python-compat): this file is pure ASCII on purpose, so
it survives any Windows / MobaXterm / scp transfer without a non-UTF-8
SyntaxError. Greek letters are spelled out (sigma, tau) in code, comments, and
plot labels. The imported pipeline modules (augmentation.py, and, for the
synthetic data path, generate_burst_data.py) must ALSO be pure ASCII for the
import to compile on a strict-locale node; the project keeps them ASCII already.

Outputs (all written to --out-dir, all machine- and human-readable):
    aug_diag_summary.json      -- every number, plus a PASS / WARN / FAIL verdict
    aug_diag_separation.png    -- distortion histograms, ECDF, band-vs-tau_burst
    aug_diag_examples.png      -- anchor vs example positive / negative traces
A formatted text report is also printed to stdout (captured in the PBS .out log).

Usage (see the __main__ block and the companion .pbs for full examples):
    # self-test (no pipeline data needed beyond augmentation.py):
    python3 augmentation_diagnostics.py --smoke --pipeline-dir /path/to/pipeline

    # real diagnostic on synthetic traces, faithful to a search config:
    python3 augmentation_diagnostics.py \
        --pipeline-dir /path/to/pipeline \
        --config config_search_3class_hpc.json \
        --synthetic --n-control 3 --n-patho 3 \
        --out-dir ./aug_diag_out
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

# Matplotlib is imported lazily inside make_plots() with the Agg backend, so the
# core numeric path (and --smoke) never depends on a display or on matplotlib.

EPS = 1e-12


# =========================================================================== #
# Pipeline import (the tool measures the pipeline's OWN transforms)
# =========================================================================== #
def import_pipeline(pipeline_dir: str):
    """Put pipeline_dir on sys.path and import the symbols we need.

    Returns a small namespace object with:
        AugmentationConfig, build_triplet_instance   (always)
        closest_power_of_2                            (from data_pipeline, or a
                                                       local fallback)
        gen                                           (generate_burst_data module
                                                       or None if unavailable)
    """
    pipeline_dir = os.path.abspath(pipeline_dir)
    if pipeline_dir not in sys.path:
        sys.path.insert(0, pipeline_dir)

    try:
        from augmentation import AugmentationConfig, build_triplet_instance
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Could not import augmentation.py from --pipeline-dir="
            + pipeline_dir
            + "\n  underlying error: "
            + repr(exc)
            + "\n  Pass the directory that contains augmentation.py (the "
            "unzipped dsn_pipeline)."
        )

    # closest_power_of_2 keeps the window length identical to the training run.
    try:
        from data_pipeline import closest_power_of_2
    except Exception:
        def closest_power_of_2(n: float) -> int:
            if n < 1:
                raise ValueError("closest_power_of_2 needs n >= 1, got %r" % (n,))
            return int(2 ** round(math.log2(n)))

    try:
        import generate_burst_data as gen
    except Exception:
        gen = None

    ns = argparse.Namespace(
        AugmentationConfig=AugmentationConfig,
        build_triplet_instance=build_triplet_instance,
        closest_power_of_2=closest_power_of_2,
        gen=gen,
    )
    return ns


# =========================================================================== #
# Configuration assembly
# =========================================================================== #
def load_config_overrides(path: Optional[str]) -> Dict:
    """Read the pipeline JSON config and pull the augmentation-relevant keys.

    Only keys that are actually present are returned; missing keys fall back to
    the AugmentationConfig defaults (this is reported to the user, because the
    sigma bands are NOT in the JSON -- they live as code defaults).
    """
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    aug = cfg.get("data", {}).get("augmentation", {})
    out: Dict = {}
    for key in ("fs", "n_positives", "n_negatives", "shift_magnitude_s"):
        if key in aug:
            out[key] = aug[key]
    # window / stride live at the data level
    data = cfg.get("data", {})
    for src, dst in (("window_s", "window_s"), ("train_stride_s", "stride_s")):
        if src in data:
            out[dst] = data[src]
    # split_method is a code default in AugmentationConfig; honour it only if the
    # config author added it explicitly.
    if "split_method" in aug:
        out["split_method"] = aug["split_method"]
    return out


def build_aug_config(pipe, args, overrides: Dict):
    """Construct the AugmentationConfig actually used, tracking provenance.

    Returns (cfg, provenance) where provenance maps each field to
    'cli' | 'config-json' | 'AugmentationConfig-default'.
    """
    AugmentationConfig = pipe.AugmentationConfig
    prov: Dict[str, str] = {}

    def pick(field, cli_val, default_placeholder=object()):
        # priority: explicit CLI > config JSON > dataclass default
        if cli_val is not None:
            prov[field] = "cli"
            return cli_val
        if field in overrides:
            prov[field] = "config-json"
            return overrides[field]
        prov[field] = "AugmentationConfig-default"
        return default_placeholder  # sentinel -> let the dataclass default apply

    # Start from the dataclass defaults, then override selectively.
    kwargs: Dict = {}

    fs = pick("fs", args.fs)
    if prov["fs"] != "AugmentationConfig-default":
        kwargs["fs"] = float(fs)
    else:
        # fs is REQUIRED by AugmentationConfig (no default). Use the runtime fs
        # of the loaded data; the caller passes it in via args._resolved_fs.
        kwargs["fs"] = float(args._resolved_fs)
        prov["fs"] = "data-fs"

    for field, cli_val, cast in (
        ("n_positives", args.n_positives, int),
        ("n_negatives", args.n_negatives, int),
        ("shift_magnitude_s", args.shift_magnitude_s, float),
        ("split_method", args.split_method, str),
        ("percentile_q", args.percentile_q, float),
        ("intra_knot_dist", args.intra_knot_dist, float),
    ):
        val = pick(field, cli_val)
        if prov[field] != "AugmentationConfig-default":
            kwargs[field] = cast(val)

    # sigma bands: CLI-only overrides (they are not in the JSON).
    band_specs = (
        ("sigma_mag_pos", args.sigma_mag_pos),
        ("sigma_mag_neg", args.sigma_mag_neg),
        ("sigma_time_pos_s", args.sigma_time_pos_s),
        ("sigma_time_neg_s", args.sigma_time_neg_s),
    )
    for field, cli_pair in band_specs:
        if cli_pair is not None:
            kwargs[field] = (float(cli_pair[0]), float(cli_pair[1]))
            prov[field] = "cli"
        else:
            prov[field] = "AugmentationConfig-default"

    cfg = AugmentationConfig(**kwargs)
    return cfg, prov


# =========================================================================== #
# Data acquisition
# =========================================================================== #
def make_bump_trace(fs: float, duration_s: float, burst_width_s: float,
                    inter_burst_s: float, rng: np.random.Generator) -> np.ndarray:
    """A dependency-light synthetic activity trace: a baseline plus Gaussian
    'bursts' of a KNOWN width, used by --smoke so the self-test does not depend
    on scipy or on generate_burst_data. Not used for real diagnostics.
    """
    T = int(round(duration_s * fs))
    t = np.arange(T, dtype=np.float64) / fs
    x = 0.05 * np.ones(T, dtype=np.float64)
    centre = inter_burst_s * 0.5
    sigma = burst_width_s / 2.355  # FWHM -> std
    while centre < duration_s:
        amp = 1.0 + 0.5 * rng.random()
        x += amp * np.exp(-0.5 * ((t - centre) / sigma) ** 2)
        centre += inter_burst_s * (0.8 + 0.4 * rng.random())
    return x.astype(np.float32)


def get_traces(pipe, args) -> Tuple[List[np.ndarray], List[int], float]:
    """Return (traces, conditions, fs).

    Modes:
      --smoke or --bump : numpy-only bump traces (known burst width).
      --synthetic       : generate_burst_data (CONTROL_PARAMS / PATHO_PARAMS).
      --npz-glob PATTERN: load pre-computed IFR .npz (keys ifr_trace, fs_ifr).
    """
    rng = np.random.default_rng(args.seed)

    if args.bump or args.smoke:
        fs = float(args.fs) if args.fs else 50.0
        traces, conditions = [], []
        n = max(1, args.n_control)
        for _ in range(n):
            traces.append(make_bump_trace(fs, args.synthetic_duration_s,
                                          args.bump_burst_width_s,
                                          args.bump_inter_burst_s, rng))
            conditions.append(0)
        return traces, conditions, fs

    if args.npz_glob:
        paths = sorted(glob.glob(args.npz_glob))
        if not paths:
            raise SystemExit("No .npz files matched --npz-glob=%r" % (args.npz_glob,))
        traces, conditions, fs = [], [], None
        for p in paths:
            data = np.load(p, allow_pickle=True)
            traces.append(np.ascontiguousarray(data["ifr_trace"], dtype=np.float32))
            fs_p = float(data["fs_ifr"])
            fs = fs_p if fs is None else fs
            # condition from filename hint, else 0 (augmentation is
            # condition-agnostic, so this only affects tau_burst grouping).
            name = os.path.basename(p).lower()
            conditions.append(1 if "patho" in name else 0)
        return traces, conditions, float(fs)

    # default: synthetic via the project generator
    if pipe.gen is None:
        raise SystemExit(
            "--synthetic requested but generate_burst_data could not be imported "
            "from --pipeline-dir. Use --npz-glob, or add generate_burst_data.py "
            "to the pipeline directory."
        )
    gen = pipe.gen
    traces, conditions = [], []
    for tid in range(max(0, args.n_control)):
        params = gen.CONTROL_PARAMS
        st = gen.generate_spike_times(params, np.random.default_rng(args.seed + tid))
        ifr, fs = gen.compute_ifr_trace(st, params)
        traces.append(np.ascontiguousarray(ifr, dtype=np.float32))
        conditions.append(0)
    for tid in range(max(0, args.n_patho)):
        params = gen.PATHO_PARAMS
        st = gen.generate_spike_times(params, np.random.default_rng(args.seed + 1000 + tid))
        ifr, fs = gen.compute_ifr_trace(st, params)
        traces.append(np.ascontiguousarray(ifr, dtype=np.float32))
        conditions.append(1)
    if not traces:
        raise SystemExit("No synthetic traces generated (set --n-control / --n-patho >= 1).")
    return traces, conditions, float(fs)


def window_traces(traces: List[np.ndarray], conditions: List[int],
                  window_length: int, stride: int, max_anchors: int,
                  seed: int) -> List[Tuple[np.ndarray, int]]:
    """Window the traces exactly as MEAWindowDataset does (start=0, step=stride,
    while start + W <= L), then evenly subsample down to max_anchors windows.

    Returns a list of (window float32 array, condition).
    """
    index: List[Tuple[int, int, int]] = []
    for ti, (tr, cond) in enumerate(zip(traces, conditions)):
        L = int(tr.shape[0])
        if L < window_length:
            continue
        s = 0
        while s + window_length <= L:
            index.append((ti, s, int(cond)))
            s += stride
    if not index:
        raise SystemExit(
            "No windows produced: window_length=%d exceeds every trace length. "
            "Lower --window-s or lengthen the traces." % (window_length,)
        )
    if max_anchors > 0 and len(index) > max_anchors:
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(len(index), size=max_anchors, replace=False))
        index = [index[i] for i in sel]
    out = []
    for ti, s, cond in index:
        w = np.ascontiguousarray(traces[ti][s:s + window_length], dtype=np.float32)
        out.append((w, cond))
    return out


# =========================================================================== #
# tau_burst estimation (mean observed burst width in the trace)
# =========================================================================== #
def estimate_tau_burst(traces: List[np.ndarray], fs: float, method: str,
                       k_std: float, frac: float,
                       min_run_s: float = 0.02) -> Dict:
    """Estimate tau_burst as the mean duration of supra-threshold excursions.

    method='std'  : threshold_c = mean(x_c) + k_std * std(x_c) per trace c.
    method='frac' : threshold_c = frac * max(x_c) per trace c.

    A 'burst' is a maximal run of consecutive samples with x >= threshold,
    longer than min_run_s seconds (drops single-sample blips). The reported
    tau_burst is the mean run duration pooled over all traces; the median and
    run count are reported too.

    Caveat (reported): the IFR trace is Gaussian-smoothed at the source
    (sigma_smooth ~ 0.04 s), so this measures the OBSERVED burst width in the
    trace the augmentation acts on -- appropriate for band tuning, but wider
    than the underlying spike-burst duration.
    """
    min_run = max(1, int(round(min_run_s * fs)))
    durations_s: List[float] = []
    per_trace: List[float] = []
    for x in traces:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0 or not np.all(np.isfinite(x)):
            per_trace.append(float("nan"))
            continue
        if method == "frac":
            thr = frac * float(np.max(x))
        else:
            thr = float(np.mean(x)) + k_std * float(np.std(x))
        above = x >= thr
        # find run boundaries
        runs = []
        idx = 0
        n = above.size
        while idx < n:
            if above[idx]:
                j = idx
                while j < n and above[j]:
                    j += 1
                if (j - idx) >= min_run:
                    runs.append((j - idx) / fs)
                idx = j
            else:
                idx += 1
        if runs:
            durations_s.extend(runs)
            per_trace.append(float(np.mean(runs)))
        else:
            per_trace.append(float("nan"))

    if durations_s:
        value = float(np.mean(durations_s))
        median = float(np.median(durations_s))
        n_runs = int(len(durations_s))
    else:
        value = float("nan")
        median = float("nan")
        n_runs = 0

    method_str = ("threshold = mean + %.2f*std" % k_std) if method != "frac" \
        else ("threshold = %.2f*max" % frac)
    return {
        "value_s": value,
        "median_s": median,
        "n_runs": n_runs,
        "per_trace_mean_s": per_trace,
        "method": method_str,
        "min_run_s": float(min_run_s),
    }


# =========================================================================== #
# Distortion metrics (pure numpy, vectorized over a surrogate matrix)
# =========================================================================== #
def distortion_metrics(anchor: np.ndarray, surrogates: np.ndarray) -> Dict[str, np.ndarray]:
    """Given anchor (T,) and surrogates (m, T), return per-surrogate:
        mse       : mean_t (s(t) - a(t))^2                 (amplitude+misalign)
        nrmse     : sqrt(mse) / (std(a) + eps)             (energy-normalized)
        corr_dist : 1 - pearson(s, a)   in [0, 2]          (shape/timing only,
                                                            amplitude-invariant)
    corr_dist is the head-line 'profile distortion': it is invariant to
    s -> alpha*s + beta, so a surrogate that only rescaled amplitude
    (magnitude warp) but preserved the burst timing has corr_dist ~ 0.
    """
    a = np.asarray(anchor, dtype=np.float64).ravel()
    S = np.asarray(surrogates, dtype=np.float64)
    if S.ndim == 1:
        S = S[None, :]
    diff = S - a[None, :]
    mse = np.mean(diff * diff, axis=1)
    std_a = float(np.std(a))
    nrmse = np.sqrt(np.maximum(mse, 0.0)) / (std_a + EPS)

    a_c = a - float(np.mean(a))
    S_c = S - np.mean(S, axis=1, keepdims=True)
    num = S_c @ a_c
    den = (np.linalg.norm(S_c, axis=1) * (np.linalg.norm(a_c) + EPS)) + EPS
    corr = num / den
    corr = np.clip(corr, -1.0, 1.0)
    corr_dist = 1.0 - corr
    return {"mse": mse, "nrmse": nrmse, "corr_dist": corr_dist}


def auc_separation(d_pos: np.ndarray, d_neg: np.ndarray) -> float:
    """Threshold-free separability = P(distortion_neg > distortion_pos).

    Equivalent to the Mann-Whitney U statistic normalized to [0, 1] (this is the
    same quantity as roc_auc_score with negatives as the positive class and
    distortion as the score). 1.0 = perfectly separated, 0.5 = indistinguishable.
    Ties contribute 0.5. Pure numpy (no sklearn dependency at run time).
    """
    d_pos = np.asarray(d_pos, dtype=np.float64).ravel()
    d_neg = np.asarray(d_neg, dtype=np.float64).ravel()
    nP, nN = d_pos.size, d_neg.size
    if nP == 0 or nN == 0:
        return float("nan")
    alld = np.concatenate([d_pos, d_neg])
    order = np.argsort(alld, kind="mergesort")
    ranks = np.empty(alld.size, dtype=np.float64)
    ranks[order] = np.arange(1, alld.size + 1, dtype=np.float64)
    # average ranks for ties
    _, inv, counts = np.unique(alld, return_inverse=True, return_counts=True)
    # cumulative sum of ranks per unique value / count gives the average rank
    sum_ranks = np.zeros(counts.size, dtype=np.float64)
    np.add.at(sum_ranks, inv, ranks)
    avg_rank_per_val = sum_ranks / counts
    ranks = avg_rank_per_val[inv]
    rank_neg_sum = float(np.sum(ranks[nP:]))
    u_neg = rank_neg_sum - nN * (nN + 1) / 2.0
    return float(u_neg / (nP * nN))


def dprime(d_pos: np.ndarray, d_neg: np.ndarray) -> float:
    d_pos = np.asarray(d_pos, dtype=np.float64).ravel()
    d_neg = np.asarray(d_neg, dtype=np.float64).ravel()
    if d_pos.size == 0 or d_neg.size == 0:
        return float("nan")
    pooled = 0.5 * (np.var(d_pos) + np.var(d_neg))
    if pooled <= EPS:
        return float("nan")
    return float((np.mean(d_neg) - np.mean(d_pos)) / math.sqrt(pooled))


def threshold_confusion(d_pos: np.ndarray, d_neg: np.ndarray) -> Dict:
    """Pick the distortion threshold t* that maximizes balanced accuracy
    (predict 'negative' when distortion > t*), and report the confusion.

    neg_below_star_frac is the head-line 'negatives that look like positives'
    rate -- exactly the failure the user is worried about.
    """
    d_pos = np.asarray(d_pos, dtype=np.float64).ravel()
    d_neg = np.asarray(d_neg, dtype=np.float64).ravel()
    if d_pos.size == 0 or d_neg.size == 0:
        return {"threshold": float("nan"), "balanced_acc": float("nan"),
                "neg_below_star_frac": float("nan"),
                "pos_above_star_frac": float("nan"),
                "leakage_frac": float("nan")}
    cands = np.unique(np.concatenate([d_pos, d_neg]))
    if cands.size > 1:
        mids = 0.5 * (cands[:-1] + cands[1:])
        thresholds = np.concatenate([[cands[0] - 1e-9], mids, [cands[-1] + 1e-9]])
    else:
        thresholds = np.array([cands[0] - 1e-9, cands[0] + 1e-9])
    best_t, best_bacc = float("nan"), -1.0
    for t in thresholds:
        tpr = float(np.mean(d_neg > t))   # negatives correctly called negative
        tnr = float(np.mean(d_pos <= t))  # positives correctly called positive
        bacc = 0.5 * (tpr + tnr)
        if bacc > best_bacc:
            best_bacc, best_t = bacc, float(t)
    neg_below = float(np.mean(d_neg <= best_t))
    pos_above = float(np.mean(d_pos > best_t))
    # threshold-independent leakage: negatives falling within the bulk (<= p95)
    # of the positive distortion.
    p95_pos = float(np.percentile(d_pos, 95))
    leakage = float(np.mean(d_neg <= p95_pos))
    return {"threshold": best_t, "balanced_acc": best_bacc,
            "neg_below_star_frac": neg_below, "pos_above_star_frac": pos_above,
            "leakage_frac": leakage}


# =========================================================================== #
# Analytic band-vs-tau_burst diagnostics
# =========================================================================== #
def analytic_band_diagnostics(cfg, tau_burst_s: float) -> Dict:
    """Fraction of each uniform band that is 'too weak', and band placement in
    units of tau_burst. For sigma_time ~ Uniform[a, b]:
        P(sigma_time < tau) = clip((tau - a) / (b - a), 0, 1).
    Also re-checks that the positive and negative sigma ranges do not overlap.
    """
    smp = tuple(cfg.sigma_mag_pos)
    smn = tuple(cfg.sigma_mag_neg)
    stp = tuple(cfg.sigma_time_pos_s)
    stn = tuple(cfg.sigma_time_neg_s)

    def frac_below(band, thr):
        a, b = float(band[0]), float(band[1])
        if b <= a:
            return float("nan")
        return float(min(max((thr - a) / (b - a), 0.0), 1.0))

    def overlaps(band1, band2):
        a1, b1 = float(band1[0]), float(band1[1])
        a2, b2 = float(band2[0]), float(band2[1])
        return (min(b1, b2) - max(a1, a2)) > 0.0

    tau = float(tau_burst_s)
    result = {
        "tau_burst_s": tau,
        "sigma_time_neg_frac_sub_burst": frac_below(stn, tau) if np.isfinite(tau) else float("nan"),
        "sigma_time_pos_frac_sub_burst": frac_below(stp, tau) if np.isfinite(tau) else float("nan"),
        "sigma_time_neg_band_in_tau_units": [float(stn[0] / tau), float(stn[1] / tau)] if (np.isfinite(tau) and tau > 0) else [float("nan"), float("nan")],
        "sigma_time_pos_band_in_tau_units": [float(stp[0] / tau), float(stp[1] / tau)] if (np.isfinite(tau) and tau > 0) else [float("nan"), float("nan")],
        "sigma_time_bands_overlap": bool(overlaps(stp, stn)),
        "sigma_mag_bands_overlap": bool(overlaps(smp, smn)),
        "bands": {
            "sigma_mag_pos": [float(smp[0]), float(smp[1])],
            "sigma_mag_neg": [float(smn[0]), float(smn[1])],
            "sigma_time_pos_s": [float(stp[0]), float(stp[1])],
            "sigma_time_neg_s": [float(stn[0]), float(stn[1])],
        },
    }
    return result


# =========================================================================== #
# Core driver: generate surrogates, collect distortions, run health checks
# =========================================================================== #
def run_collection(pipe, cfg, windows: List[Tuple[np.ndarray, int]],
                   seed: int, shift_tol: float = 1e-3) -> Dict:
    """For each anchor window, build the triplet (pre-shift), compute distortion
    metrics for positives (excluding the clean anchor row) and negatives, and
    accumulate. Also run per-instance health checks.
    """
    build = pipe.build_triplet_instance
    rng = np.random.default_rng(seed)

    pooled = {"mse": {"pos": [], "neg": []},
              "nrmse": {"pos": [], "neg": []},
              "corr_dist": {"pos": [], "neg": []}}
    per_anchor_auc = []  # corr_dist AUC per anchor (consistency across anchors)

    n_pos_surr = 0
    n_neg_surr = 0
    n_nonfinite = 0
    min_value = math.inf
    shift_max_sorted_diff = 0.0
    anchor_row_max_dev = 0.0

    # keep a few example triples for plotting (min / median / max corr_dist neg)
    examples = {"anchor": None, "pos": None, "neg_low": None, "neg_high": None,
                "neg_low_val": None, "neg_high_val": None}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # empty-class re-draws
        for (w, _cond) in windows:
            out = build(w, cfg, rng, return_pre_shift=True)
            anchor, positives, negatives, pos_pre, neg_pre = out
            a = np.asarray(anchor, dtype=np.float64).ravel()
            pos_pre = np.asarray(pos_pre, dtype=np.float64)
            neg_pre = np.asarray(neg_pre, dtype=np.float64)
            positives = np.asarray(positives, dtype=np.float64)
            negatives = np.asarray(negatives, dtype=np.float64)

            # anchor must be row 0 of pos_pre and equal the clean window
            anchor_row_max_dev = max(anchor_row_max_dev,
                                     float(np.max(np.abs(pos_pre[0] - a))))

            pos_surr = pos_pre[1:] if pos_pre.shape[0] > 1 else pos_pre[:0]
            neg_surr = neg_pre

            # health: finiteness + non-negativity on the surrogates
            for block in (pos_surr, neg_surr):
                if block.size:
                    n_nonfinite += int(np.sum(~np.isfinite(block)))
                    finite = block[np.isfinite(block)]
                    if finite.size:
                        min_value = min(min_value, float(np.min(finite)))

            # health: circular shift is a permutation of the pre-shift row
            for post, pre in ((positives[1:], pos_pre[1:]), (negatives, neg_pre)):
                m = min(post.shape[0], pre.shape[0])
                for i in range(m):
                    d = float(np.max(np.abs(np.sort(post[i]) - np.sort(pre[i]))))
                    shift_max_sorted_diff = max(shift_max_sorted_diff, d)

            # distortions
            if pos_surr.size:
                mp = distortion_metrics(a, pos_surr)
                n_pos_surr += pos_surr.shape[0]
            else:
                mp = {"mse": np.array([]), "nrmse": np.array([]), "corr_dist": np.array([])}
            if neg_surr.size:
                mn = distortion_metrics(a, neg_surr)
                n_neg_surr += neg_surr.shape[0]
            else:
                mn = {"mse": np.array([]), "nrmse": np.array([]), "corr_dist": np.array([])}

            for key in ("mse", "nrmse", "corr_dist"):
                pooled[key]["pos"].append(mp[key])
                pooled[key]["neg"].append(mn[key])

            if mp["corr_dist"].size and mn["corr_dist"].size:
                per_anchor_auc.append(auc_separation(mp["corr_dist"], mn["corr_dist"]))

            # capture example traces once
            if examples["anchor"] is None and neg_surr.shape[0] >= 1:
                examples["anchor"] = a.copy()
                if pos_surr.shape[0] >= 1:
                    examples["pos"] = pos_surr[0].copy()
                cd = mn["corr_dist"]
                lo_i = int(np.argmin(cd))
                hi_i = int(np.argmax(cd))
                examples["neg_low"] = neg_surr[lo_i].copy()
                examples["neg_high"] = neg_surr[hi_i].copy()
                examples["neg_low_val"] = float(cd[lo_i])
                examples["neg_high_val"] = float(cd[hi_i])

    # concatenate pooled distortions
    agg: Dict = {}
    for key in ("mse", "nrmse", "corr_dist"):
        dpos = np.concatenate(pooled[key]["pos"]) if pooled[key]["pos"] else np.array([])
        dneg = np.concatenate(pooled[key]["neg"]) if pooled[key]["neg"] else np.array([])
        stats = {
            "auc": auc_separation(dpos, dneg),
            "dprime": dprime(dpos, dneg),
            "mean_pos": float(np.mean(dpos)) if dpos.size else float("nan"),
            "mean_neg": float(np.mean(dneg)) if dneg.size else float("nan"),
            "std_pos": float(np.std(dpos)) if dpos.size else float("nan"),
            "std_neg": float(np.std(dneg)) if dneg.size else float("nan"),
        }
        stats.update(threshold_confusion(dpos, dneg))
        agg[key] = stats
        agg[key]["_dpos"] = dpos  # kept for plotting; stripped before JSON
        agg[key]["_dneg"] = dneg

    if min_value is math.inf:
        min_value = float("nan")

    health = {
        "n_anchors": len(windows),
        "n_pos_surrogates": int(n_pos_surr),
        "n_neg_surrogates": int(n_neg_surr),
        "n_nonfinite": int(n_nonfinite),
        "all_finite": bool(n_nonfinite == 0),
        "min_surrogate_value": float(min_value),
        "nonneg_ok": bool(np.isnan(min_value) or (min_value >= -1e-4)),
        "shift_max_sorted_diff": float(shift_max_sorted_diff),
        "shift_invariance_ok": bool(shift_max_sorted_diff <= shift_tol),
        "anchor_row_max_dev": float(anchor_row_max_dev),
        "anchor_is_clean_ok": bool(anchor_row_max_dev <= 1e-5),
        "per_anchor_corr_auc_mean": float(np.mean(per_anchor_auc)) if per_anchor_auc else float("nan"),
        "per_anchor_corr_auc_std": float(np.std(per_anchor_auc)) if per_anchor_auc else float("nan"),
    }
    return {"agg": agg, "health": health, "examples": examples}


# =========================================================================== #
# Verdict
# =========================================================================== #
def build_verdict(agg: Dict, health: Dict, analytic: Dict, thr: Dict) -> Dict:
    """Turn the numbers into PASS / WARN / FAIL lines. The thresholds are
    heuristic and exposed via CLI so the user can tighten or relax them.
    """
    warns: List[str] = []
    fails: List[str] = []

    corr_auc = agg["corr_dist"]["auc"]
    neg_below = agg["corr_dist"]["neg_below_star_frac"]
    frac_sub = analytic.get("sigma_time_neg_frac_sub_burst", float("nan"))

    if np.isfinite(corr_auc) and corr_auc < thr["min_auc"]:
        warns.append(
            "profile separability low: corr_dist AUC=%.3f < %.2f "
            "(positives and negatives overlap in shape/timing space)"
            % (corr_auc, thr["min_auc"]))
    if np.isfinite(neg_below) and neg_below > thr["max_neg_below"]:
        warns.append(
            "negatives-look-like-positives rate high: %.1f%% > %.1f%% "
            "(fraction of negatives below the optimal distortion threshold)"
            % (100 * neg_below, 100 * thr["max_neg_below"]))
    if np.isfinite(frac_sub) and frac_sub > 0.0:
        sev = "STRONG " if frac_sub > thr["max_frac_sub_burst"] else ""
        warns.append(
            "%sunder-reach: %.1f%% of the negative time band is BELOW tau_burst "
            "-> those negatives cannot destroy the profile (raise "
            "sigma_time_neg lower endpoint above tau_burst)"
            % (sev, 100 * frac_sub))
    if analytic.get("sigma_time_bands_overlap"):
        warns.append("sigma_time positive and negative bands OVERLAP")
    if analytic.get("sigma_mag_bands_overlap"):
        warns.append("sigma_mag positive and negative bands OVERLAP")

    if not health["all_finite"]:
        fails.append("non-finite surrogate values detected (%d)" % health["n_nonfinite"])
    if not health["nonneg_ok"]:
        fails.append("negative surrogate values beyond tolerance (min=%.4g)"
                     % health["min_surrogate_value"])
    if not health["shift_invariance_ok"]:
        fails.append("circular shift is NOT label-preserving "
                     "(max sorted-value diff=%.4g)" % health["shift_max_sorted_diff"])
    if not health["anchor_is_clean_ok"]:
        fails.append("anchor row is not the clean window (max dev=%.4g)"
                     % health["anchor_row_max_dev"])

    if fails:
        status = "FAIL"
    elif warns:
        status = "WARN"
    else:
        status = "PASS"
    return {"status": status, "warnings": warns, "failures": fails}


# =========================================================================== #
# Reporting
# =========================================================================== #
def format_report(cfg, prov: Dict, tau: Dict, analytic: Dict, agg: Dict,
                  health: Dict, verdict: Dict) -> str:
    L: List[str] = []
    def line(s=""):
        L.append(s)

    line("=" * 74)
    line("AUGMENTATION DIAGNOSTICS")
    line("=" * 74)
    line("split_method            : %s  (source: %s)"
         % (cfg.split_method, prov.get("split_method", "?")))
    line("n_positives/n_negatives : %d / %d" % (cfg.n_positives, cfg.n_negatives))
    line("fs / shift_magnitude_s  : %.3f Hz / %.3f s" % (cfg.fs, cfg.shift_magnitude_s))
    line("anchors x surrogates    : %d anchors, %d pos + %d neg surrogates"
         % (health["n_anchors"], health["n_pos_surrogates"], health["n_neg_surrogates"]))
    line("")
    line("SIGMA BANDS  (provenance in brackets)")
    line("  sigma_mag_pos   = [%.4f, %.4f]   [%s]"
         % (cfg.sigma_mag_pos[0], cfg.sigma_mag_pos[1], prov.get("sigma_mag_pos", "?")))
    line("  sigma_mag_neg   = [%.4f, %.4f]   [%s]"
         % (cfg.sigma_mag_neg[0], cfg.sigma_mag_neg[1], prov.get("sigma_mag_neg", "?")))
    line("  sigma_time_pos  = [%.4f, %.4f] s [%s]"
         % (cfg.sigma_time_pos_s[0], cfg.sigma_time_pos_s[1], prov.get("sigma_time_pos_s", "?")))
    line("  sigma_time_neg  = [%.4f, %.4f] s [%s]"
         % (cfg.sigma_time_neg_s[0], cfg.sigma_time_neg_s[1], prov.get("sigma_time_neg_s", "?")))
    line("  sigma_time bands overlap : %s   sigma_mag bands overlap : %s"
         % (analytic["sigma_time_bands_overlap"], analytic["sigma_mag_bands_overlap"]))
    line("")
    line("TAU_BURST  (mean observed burst width in the trace)")
    line("  tau_burst        = %.4f s   (median %.4f s over %d runs)"
         % (tau["value_s"], tau["median_s"], tau["n_runs"]))
    line("  estimator        = %s ; min_run = %.3f s" % (tau["method"], tau["min_run_s"]))
    if np.isfinite(analytic["sigma_time_neg_band_in_tau_units"][0]):
        line("  neg time band    = [%.2f, %.2f] x tau_burst"
             % (analytic["sigma_time_neg_band_in_tau_units"][0],
                analytic["sigma_time_neg_band_in_tau_units"][1]))
        line("  pos time band    = [%.2f, %.2f] x tau_burst"
             % (analytic["sigma_time_pos_band_in_tau_units"][0],
                analytic["sigma_time_pos_band_in_tau_units"][1]))
    line("  fraction of NEG time band below tau_burst : %.1f%%   "
         "(these negatives cannot destroy the profile)"
         % (100 * analytic["sigma_time_neg_frac_sub_burst"]
            if np.isfinite(analytic["sigma_time_neg_frac_sub_burst"]) else float("nan")))
    line("")
    line("SEPARATION  (positives vs negatives in distortion space)")
    header = "  %-12s %7s %8s %10s %10s %12s %10s" % (
        "metric", "AUC", "d-prime", "mean_pos", "mean_neg", "neg<thr(%)", "leak(%)")
    line(header)
    for key in ("corr_dist", "mse", "nrmse"):
        s = agg[key]
        line("  %-12s %7.3f %8.3f %10.4g %10.4g %12.1f %10.1f" % (
            key, s["auc"], s["dprime"], s["mean_pos"], s["mean_neg"],
            100 * s["neg_below_star_frac"], 100 * s["leakage_frac"]))
    line("  (corr_dist is the head-line PROFILE metric: amplitude-invariant, so")
    line("   magnitude-only changes do not count as profile destruction.)")
    line("  per-anchor corr_dist AUC : mean %.3f  std %.3f"
         % (health["per_anchor_corr_auc_mean"], health["per_anchor_corr_auc_std"]))
    line("")
    line("HEALTH")
    line("  all surrogates finite    : %s   (%d non-finite)"
         % (health["all_finite"], health["n_nonfinite"]))
    line("  non-negativity ok        : %s   (min value %.4g)"
         % (health["nonneg_ok"], health["min_surrogate_value"]))
    line("  shift label-preserving   : %s   (max sorted diff %.2e)"
         % (health["shift_invariance_ok"], health["shift_max_sorted_diff"]))
    line("  anchor row is clean      : %s   (max dev %.2e)"
         % (health["anchor_is_clean_ok"], health["anchor_row_max_dev"]))
    line("")
    line("VERDICT : %s" % verdict["status"])
    for w in verdict["warnings"]:
        line("  [WARN] " + w)
    for f in verdict["failures"]:
        line("  [FAIL] " + f)
    if verdict["status"] == "PASS":
        line("  no issues at the configured thresholds.")
    line("=" * 74)
    return "\n".join(L)


def strip_arrays_for_json(agg: Dict) -> Dict:
    out = {}
    for key, stats in agg.items():
        out[key] = {k: v for k, v in stats.items() if not k.startswith("_")}
    return out


def make_plots(out_dir: str, cfg, tau: Dict, analytic: Dict, agg: Dict,
               examples: Dict, fs: float) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless: mandatory on HPC compute nodes
        import matplotlib.pyplot as plt
    except Exception as exc:
        warnings.warn("matplotlib unavailable, skipping plots: %r" % (exc,),
                      RuntimeWarning)
        return []

    paths: List[str] = []

    # ---- Figure 1: separation + band placement ---------------------------- #
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    def hist_pair(a, dpos, dneg, title, xlabel):
        if dpos.size == 0 and dneg.size == 0:
            a.set_title(title + " (no data)")
            return
        alld = np.concatenate([d for d in (dpos, dneg) if d.size])
        lo, hi = float(np.min(alld)), float(np.max(alld))
        if hi <= lo:
            hi = lo + 1e-6
        bins = np.linspace(lo, hi, 40)
        if dpos.size:
            a.hist(dpos, bins=bins, alpha=0.6, label="positives", density=True,
                   color="#2c7fb8")
        if dneg.size:
            a.hist(dneg, bins=bins, alpha=0.6, label="negatives", density=True,
                   color="#d95f0e")
        a.set_title(title)
        a.set_xlabel(xlabel)
        a.set_ylabel("density")
        a.legend()

    cd = agg["corr_dist"]
    hist_pair(ax[0, 0], cd["_dpos"], cd["_dneg"],
              "profile distortion (corr_dist)  AUC=%.3f" % cd["auc"],
              "1 - pearson(surrogate, anchor)")
    ms = agg["mse"]
    hist_pair(ax[0, 1], ms["_dpos"], ms["_dneg"],
              "MSE distortion  AUC=%.3f" % ms["auc"], "mean squared error")

    # ECDF of corr_dist
    a2 = ax[1, 0]
    for d, lab, col in ((cd["_dpos"], "positives", "#2c7fb8"),
                        (cd["_dneg"], "negatives", "#d95f0e")):
        if d.size:
            xs = np.sort(d)
            ys = np.arange(1, xs.size + 1) / xs.size
            a2.plot(xs, ys, label=lab, color=col)
    a2.set_title("corr_dist ECDF")
    a2.set_xlabel("1 - pearson(surrogate, anchor)")
    a2.set_ylabel("cumulative fraction")
    a2.legend()

    # band placement vs tau_burst (time axis)
    a3 = ax[1, 1]
    stp = cfg.sigma_time_pos_s
    stn = cfg.sigma_time_neg_s
    a3.barh(1.0, stp[1] - stp[0], left=stp[0], height=0.3, color="#2c7fb8",
            label="sigma_time_pos")
    a3.barh(0.5, stn[1] - stn[0], left=stn[0], height=0.3, color="#d95f0e",
            label="sigma_time_neg")
    tval = tau["value_s"]
    if np.isfinite(tval):
        a3.axvline(tval, color="black", linestyle="--", label="tau_burst")
        # shade sub-burst part of the negative band
        sub_hi = min(stn[1], tval)
        if sub_hi > stn[0]:
            a3.barh(0.5, sub_hi - stn[0], left=stn[0], height=0.3,
                    color="none", edgecolor="red", hatch="////",
                    label="neg sub-burst")
    a3.set_ylim(0.0, 1.6)
    a3.set_yticks([])
    a3.set_xlabel("sigma_time [s]")
    a3.set_title("time-warp bands vs tau_burst")
    a3.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    p1 = os.path.join(out_dir, "aug_diag_separation.png")
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    paths.append(p1)

    # ---- Figure 2: example traces ---------------------------------------- #
    if examples.get("anchor") is not None:
        a = examples["anchor"]
        t = np.arange(a.size) / fs
        fig2, ax2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        ax2[0].plot(t, a, color="black", label="anchor")
        if examples.get("pos") is not None:
            ax2[0].plot(t, examples["pos"], color="#2c7fb8", alpha=0.8,
                        label="a positive (pre-shift)")
        ax2[0].set_title("anchor vs positive")
        ax2[0].legend()
        ax2[0].set_ylabel("activity")

        ax2[1].plot(t, a, color="black", label="anchor")
        if examples.get("neg_low") is not None:
            ax2[1].plot(t, examples["neg_low"], color="#d95f0e", alpha=0.8,
                        label="LOW-distortion negative (corr_dist=%.3f)"
                        % examples["neg_low_val"])
        ax2[1].set_title("anchor vs the WEAKEST negative "
                         "(the dangerous 'too similar' case)")
        ax2[1].legend()
        ax2[1].set_ylabel("activity")

        ax2[2].plot(t, a, color="black", label="anchor")
        if examples.get("neg_high") is not None:
            ax2[2].plot(t, examples["neg_high"], color="#d95f0e", alpha=0.8,
                        label="HIGH-distortion negative (corr_dist=%.3f)"
                        % examples["neg_high_val"])
        ax2[2].set_title("anchor vs the STRONGEST negative")
        ax2[2].legend()
        ax2[2].set_xlabel("time [s]")
        ax2[2].set_ylabel("activity")

        fig2.tight_layout()
        p2 = os.path.join(out_dir, "aug_diag_examples.png")
        fig2.savefig(p2, dpi=120)
        plt.close(fig2)
        paths.append(p2)

    return paths


# =========================================================================== #
# Argument parsing
# =========================================================================== #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Input-space diagnostic for the augmentation pipeline.")
    p.add_argument("--pipeline-dir", default=os.path.dirname(os.path.abspath(__file__)),
                   help="directory containing augmentation.py (unzipped dsn_pipeline).")
    p.add_argument("--config", default=None,
                   help="pipeline JSON config; augmentation params are read from it.")
    p.add_argument("--out-dir", default="./aug_diag_out")

    # data source
    p.add_argument("--synthetic", action="store_true",
                   help="generate traces via generate_burst_data (default source).")
    p.add_argument("--npz-glob", default=None,
                   help="glob of pre-computed IFR .npz files (keys ifr_trace, fs_ifr).")
    p.add_argument("--bump", action="store_true",
                   help="numpy-only bump traces (no scipy / generator needed).")
    p.add_argument("--n-control", type=int, default=3)
    p.add_argument("--n-patho", type=int, default=3)
    p.add_argument("--synthetic-duration-s", type=float, default=200.0)
    p.add_argument("--bump-burst-width-s", type=float, default=0.25)
    p.add_argument("--bump-inter-burst-s", type=float, default=6.0)

    # windowing
    p.add_argument("--window-s", type=float, default=30.0)
    p.add_argument("--stride-s", type=float, default=15.0)
    p.add_argument("--n-anchors", type=int, default=64,
                   help="max anchor windows to sample (0 = all).")

    # augmentation config overrides (all optional; None -> config/JSON/default)
    p.add_argument("--fs", type=float, default=None)
    p.add_argument("--n-positives", type=int, default=None)
    p.add_argument("--n-negatives", type=int, default=None)
    p.add_argument("--shift-magnitude-s", type=float, default=None)
    p.add_argument("--split-method", default=None,
                   choices=[None, "warp_bands", "percentile_mse"])
    p.add_argument("--percentile-q", type=float, default=None)
    p.add_argument("--intra-knot-dist", type=float, default=None)
    p.add_argument("--sigma-mag-pos", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"))
    p.add_argument("--sigma-mag-neg", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"))
    p.add_argument("--sigma-time-pos-s", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"))
    p.add_argument("--sigma-time-neg-s", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"))

    # tau_burst estimator
    p.add_argument("--tau-method", default="std", choices=["std", "frac"])
    p.add_argument("--tau-k-std", type=float, default=1.0)
    p.add_argument("--tau-frac", type=float, default=0.2)
    p.add_argument("--tau-burst-s", type=float, default=None,
                   help="override the estimated tau_burst with a known value.")

    # verdict thresholds (heuristic, tunable)
    p.add_argument("--min-auc", type=float, default=0.90)
    p.add_argument("--max-neg-below", type=float, default=0.10)
    p.add_argument("--max-frac-sub-burst", type=float, default=0.25)

    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true",
                   help="run the built-in self-test and exit.")
    return p


# =========================================================================== #
# Main
# =========================================================================== #
def run_main(args) -> int:
    pipe = import_pipeline(args.pipeline_dir)
    overrides = load_config_overrides(args.config)

    # window/stride can come from the config JSON
    if "window_s" in overrides and args.window_s == 30.0:
        args.window_s = float(overrides["window_s"])
    if "stride_s" in overrides and args.stride_s == 15.0:
        args.stride_s = float(overrides["stride_s"])

    traces, conditions, fs = get_traces(pipe, args)
    args._resolved_fs = fs

    cfg, prov = build_aug_config(pipe, args, overrides)
    # keep cfg.fs consistent with the actual data fs
    if abs(float(cfg.fs) - float(fs)) > 1e-9:
        warnings.warn("cfg.fs=%.4f differs from data fs=%.4f; using data fs."
                      % (cfg.fs, fs), RuntimeWarning)
        from dataclasses import replace as _dc_replace
        cfg = _dc_replace(cfg, fs=float(fs))

    W = pipe.closest_power_of_2(args.window_s * fs)
    stride = max(1, int(args.stride_s * fs))
    windows = window_traces(traces, conditions, W, stride, args.n_anchors, args.seed)

    # tau_burst
    if args.tau_burst_s is not None:
        tau = {"value_s": float(args.tau_burst_s), "median_s": float(args.tau_burst_s),
               "n_runs": 0, "per_trace_mean_s": [], "method": "user-supplied",
               "min_run_s": 0.0}
    else:
        tau = estimate_tau_burst(traces, fs, args.tau_method, args.tau_k_std,
                                 args.tau_frac)

    analytic = analytic_band_diagnostics(cfg, tau["value_s"])

    collected = run_collection(pipe, cfg, windows, args.seed)
    agg = collected["agg"]
    health = collected["health"]
    examples = collected["examples"]

    thr = {"min_auc": args.min_auc, "max_neg_below": args.max_neg_below,
           "max_frac_sub_burst": args.max_frac_sub_burst}
    verdict = build_verdict(agg, health, analytic, thr)

    report = format_report(cfg, prov, tau, analytic, agg, health, verdict)
    print(report, flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {
        "config": {
            "split_method": cfg.split_method,
            "n_positives": cfg.n_positives,
            "n_negatives": cfg.n_negatives,
            "fs": float(cfg.fs),
            "shift_magnitude_s": float(cfg.shift_magnitude_s),
            "intra_knot_dist": float(cfg.intra_knot_dist),
            "window_length": int(W),
            "provenance": prov,
        },
        "tau_burst": tau,
        "analytic": analytic,
        "separation": strip_arrays_for_json(agg),
        "health": health,
        "verdict": verdict,
        "thresholds": thr,
    }
    summary_path = os.path.join(args.out_dir, "aug_diag_summary.json")
    with open(summary_path, "w", encoding="ascii") as fh:
        json.dump(summary, fh, indent=2)
    print("wrote %s" % summary_path, flush=True)

    if not args.no_plots:
        plots = make_plots(args.out_dir, cfg, tau, analytic, agg, examples, fs)
        for pth in plots:
            print("wrote %s" % pth, flush=True)

    return 0 if verdict["status"] != "FAIL" else 2


# =========================================================================== #
# Built-in self-test (directive 4: ship a smoke test with the implementation)
# =========================================================================== #
def _check(name: str, cond: bool, detail: str = "") -> bool:
    tag = "PASS" if cond else "FAIL"
    msg = "[%s] %s" % (tag, name)
    if detail:
        msg += "  (%s)" % detail
    print(msg, flush=True)
    return cond


def smoke_test(pipeline_dir: str) -> int:
    """Self-test the metrics, the analytic arithmetic, the tau_burst estimator,
    and the end-to-end collection against the REAL augmentation.py. Uses
    numpy-only bump traces so it does not depend on scipy or the generator.
    """
    print("=" * 60)
    print("SMOKE TEST: augmentation_diagnostics")
    print("=" * 60)
    pipe = import_pipeline(pipeline_dir)
    AugmentationConfig = pipe.AugmentationConfig
    ok = True

    # ---- analytic arithmetic (hand-computed expectations) ----------------- #
    class _Cfg:
        sigma_mag_pos = (0.01, 0.10)
        sigma_mag_neg = (0.20, 0.50)
        sigma_time_pos_s = (0.005, 0.050)
        sigma_time_neg_s = (0.100, 0.400)
    a1 = analytic_band_diagnostics(_Cfg(), 0.25)
    ok &= _check("analytic frac_sub_burst @tau=0.25 == 0.5",
                 abs(a1["sigma_time_neg_frac_sub_burst"] - 0.5) < 1e-9,
                 "got %.4f" % a1["sigma_time_neg_frac_sub_burst"])
    a2 = analytic_band_diagnostics(_Cfg(), 0.05)
    ok &= _check("analytic frac_sub_burst @tau=0.05 == 0.0",
                 abs(a2["sigma_time_neg_frac_sub_burst"] - 0.0) < 1e-9,
                 "got %.4f" % a2["sigma_time_neg_frac_sub_burst"])
    a3 = analytic_band_diagnostics(_Cfg(), 0.60)
    ok &= _check("analytic frac_sub_burst @tau=0.60 == 1.0",
                 abs(a3["sigma_time_neg_frac_sub_burst"] - 1.0) < 1e-9,
                 "got %.4f" % a3["sigma_time_neg_frac_sub_burst"])
    ok &= _check("analytic detects disjoint sigma_time bands",
                 a1["sigma_time_bands_overlap"] is False)

    # ---- metric sanity ---------------------------------------------------- #
    rng = np.random.default_rng(0)
    base = rng.random(256) + 0.1
    same = base.copy()
    scaled = 3.0 * base + 2.0          # pure affine -> corr_dist ~ 0
    shifted_shape = np.roll(base, 30)  # different shape at lag 0 -> corr_dist>0
    m = distortion_metrics(base, np.stack([same, scaled, shifted_shape]))
    ok &= _check("corr_dist(identical) ~ 0", m["corr_dist"][0] < 1e-9,
                 "got %.3g" % m["corr_dist"][0])
    ok &= _check("corr_dist(affine-rescaled) ~ 0 (amplitude-invariant)",
                 m["corr_dist"][1] < 1e-6, "got %.3g" % m["corr_dist"][1])
    ok &= _check("corr_dist(shape change) > 0.1", m["corr_dist"][2] > 0.1,
                 "got %.3g" % m["corr_dist"][2])
    ok &= _check("mse(affine-rescaled) > mse(identical)",
                 m["mse"][1] > m["mse"][0])

    # ---- AUC / dprime on constructed distributions ------------------------ #
    dp = rng.normal(0.0, 1.0, 500)
    dn = rng.normal(4.0, 1.0, 500)
    auc = auc_separation(dp, dn)
    ok &= _check("AUC well-separated > 0.98", auc > 0.98, "got %.3f" % auc)
    auc_same = auc_separation(dp, dp.copy())
    ok &= _check("AUC identical ~ 0.5", abs(auc_same - 0.5) < 0.02,
                 "got %.3f" % auc_same)
    ok &= _check("dprime well-separated > 2", dprime(dp, dn) > 2.0,
                 "got %.3f" % dprime(dp, dn))

    # ---- tau_burst estimator on a known-width bump trace ------------------ #
    fs = 50.0
    bump = make_bump_trace(fs, 120.0, burst_width_s=0.30, inter_burst_s=6.0,
                           rng=np.random.default_rng(1))
    tau = estimate_tau_burst([bump], fs, "std", 1.0, 0.2)
    ok &= _check("tau_burst finite and in (0.05, 2.0) s",
                 np.isfinite(tau["value_s"]) and 0.05 < tau["value_s"] < 2.0,
                 "got %.3f s over %d runs" % (tau["value_s"], tau["n_runs"]))

    # ---- end-to-end: degenerate bands -> AUC ~ 0.5 ------------------------ #
    tiny = (1e-6, 1e-6)
    cfg_deg = AugmentationConfig(fs=fs, n_positives=8, n_negatives=8,
                                 shift_magnitude_s=5.0, split_method="warp_bands",
                                 sigma_mag_pos=tiny, sigma_mag_neg=tiny,
                                 sigma_time_pos_s=tiny, sigma_time_neg_s=tiny)
    W = pipe.closest_power_of_2(20.0 * fs)
    wins = window_traces([bump], [0], W, int(10 * fs), 12, 0)
    col_deg = run_collection(pipe, cfg_deg, wins, 0)
    auc_deg = col_deg["agg"]["corr_dist"]["auc"]
    ok &= _check("degenerate bands -> corr_dist AUC ~ 0.5",
                 (not np.isfinite(auc_deg)) or abs(auc_deg - 0.5) < 0.15,
                 "got %.3f" % auc_deg)
    ok &= _check("degenerate bands -> health all finite + nonneg",
                 col_deg["health"]["all_finite"] and col_deg["health"]["nonneg_ok"])
    ok &= _check("shift is label-preserving (degenerate)",
                 col_deg["health"]["shift_invariance_ok"],
                 "max sorted diff %.2e" % col_deg["health"]["shift_max_sorted_diff"])

    # ---- end-to-end: default disjoint bands -> negatives more distorted ---- #
    cfg_sep = AugmentationConfig(fs=fs, n_positives=12, n_negatives=12,
                                 shift_magnitude_s=5.0, split_method="warp_bands")
    col_sep = run_collection(pipe, cfg_sep, wins, 0)
    sep = col_sep["agg"]["corr_dist"]
    ok &= _check("default bands -> mean_neg corr_dist > mean_pos",
                 sep["mean_neg"] > sep["mean_pos"],
                 "pos=%.3g neg=%.3g" % (sep["mean_pos"], sep["mean_neg"]))
    ok &= _check("default bands -> corr_dist AUC > 0.7",
                 sep["auc"] > 0.7, "got %.3f" % sep["auc"])
    ok &= _check("default bands -> health all finite + nonneg + shift ok",
                 col_sep["health"]["all_finite"] and col_sep["health"]["nonneg_ok"]
                 and col_sep["health"]["shift_invariance_ok"])

    # ---- monotonicity: stronger negative time band -> more distortion ------ #
    from dataclasses import replace as _dc_replace
    cfg_strong = _dc_replace(cfg_sep, sigma_time_neg_s=(0.4, 1.2))
    col_strong = run_collection(pipe, cfg_strong, wins, 0)
    ok &= _check("stronger sigma_time_neg -> higher mean neg corr_dist",
                 col_strong["agg"]["corr_dist"]["mean_neg"] >= sep["mean_neg"],
                 "weak=%.3g strong=%.3g"
                 % (sep["mean_neg"], col_strong["agg"]["corr_dist"]["mean_neg"]))

    print("=" * 60)
    print("SMOKE RESULT: %s" % ("ALL PASSED" if ok else "FAILURES ABOVE"))
    print("=" * 60)
    return 0 if ok else 1


def main() -> int:
    args = build_parser().parse_args()
    if args.smoke:
        return smoke_test(args.pipeline_dir)
    return run_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
