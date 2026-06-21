"""
generate_burst_data.py
======================

Synthesize a realistic in-vitro MEA bursting recording for N neurons over
T_rec seconds and save it as a .npz archive for end-to-end testing of the
Topic-1 contrastive pipeline.

Three traces are generated (2 control, 1 pathological) and a
``burst_specs.json`` index is written so ``run_data_pipeline.py --data-mode
numpy`` can consume them without any .mat files.

Model (notation introduced at first use)
-----------------------------------------
Let N ∈ ℕ be the number of neurons and T_rec ∈ ℝ_{>0} the recording duration
[s].  The spike train of neuron i ∈ {1,…,N} is the superposition of a
background process and J burst-driven processes.

Burst count
    J | T_rec, λ_b  ~  Poisson(λ_b · T_rec),
where λ_b [bursts/s] is the population-burst rate.

Burst onsets
    t_b^{(j)} | T_rec  ~  Uniform([0, T_rec)),  j = 1,…,J  (i.i.d., sorted).

Burst durations
    D^{(j)}  ~  LogNormal(μ_D, σ_D^2),  j = 1,…,J  (i.i.d.),
so that median(D^{(j)}) = exp(μ_D) [s] and σ_D controls the log-scale spread.

Per-neuron participation probability (drawn ONCE per neuron, persistent)
    p_i  ~  Beta(α_p, β_p),  i = 1,…,N  (i.i.d.).

Participation indicator (per neuron, per burst)
    Z_i^{(j)} | p_i  ~  Bernoulli(p_i),  i = 1,…,N,  j = 1,…,J  (i.i.d.).

Intra-burst spikes  (given Z_i^{(j)} = 1)
    {t_{i,k}^{(j)}} from a Poisson process with rate λ_{burst} [spikes/s]
    on the interval [t_b^{(j)}, min(t_b^{(j)} + D^{(j)}, T_rec)).

Background spikes  (always active)
    {t_{i,k}^{bg}} from a Poisson process with rate λ_{bg} [spikes/s]
    on [0, T_rec).

Total spike set for neuron i:
    S_i = {t_{i,k}^{bg}} ∪ ⋃_{j : Z_i^{(j)}=1} {t_{i,k}^{(j)}}  ⊆ [0, T_rec).

IFR computation  (matching the Neuronal_traces convention)
    Bin width    Δt = w_size  [s],  so that f_s^{IFR} = 1/Δt  [Hz].
    Bin index    k ∈ {0,…,K-1},  K = ⌊T_rec / Δt⌋.
    Population spike count in bin k:
        C[k] = Σ_{i=1}^{N} |S_i ∩ [kΔt, (k+1)Δt)|.
    Gaussian-smoothed cumulative IFR:
        R̃[k] = (C ∗ g_{σ_s})[k],
    where g_{σ_s} is a discrete Gaussian kernel with σ_s = σ_{smooth}/Δt
    [bins] and ∗ denotes discrete convolution.  R̃[k] ≥ 0 for all k because
    C[k] ≥ 0 and the Gaussian kernel is non-negative.

Pathological vs control
    Control  : moderate burst rate, moderate intra-burst rate, high but
               variable participation → clear but moderate IFR peaks.
    Patho    : higher burst rate, higher intra-burst rate, more irregular
               durations, lower average participation → denser, more erratic
               IFR peaks.  Both differ observably in burst statistics, which
               is what the pipeline must learn to separate.

Run
---
    python generate_burst_data.py            # generates ./burst_data/
    python generate_burst_data.py --smoke    # runs internal smoke test only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

__all__ = [
    "BurstParams",
    "CONTROL_PARAMS",
    "PATHO_PARAMS",
    "generate_spike_times",
    "compute_ifr_trace",
    "save_burst_npz",
    "load_burst_npz",
    "plot_raster",
]

# --------------------------------------------------------------------------- #
# model parameters
# --------------------------------------------------------------------------- #
@dataclass
class BurstParams:
    """Parameters of the network-burst generative model.

    Every field is annotated with its mathematical symbol and physical unit as
    defined in the module docstring.
    """
    # recording geometry
    n_neurons:       int   = 100                   # N          [count]
    duration_s:      float = 200.0                 # T_rec      [s]
    fs_spike:        float = 1000.0                # f_s^{sp}  [Hz] — spike resolution

    # burst process
    lambda_b:        float = 0.15                  # λ_b        [bursts/s]
    mu_d:            float = -1.3862943611198906   # μ_D = ln(0.25) [log s]
    sigma_d:         float = 0.40                  # σ_D        [dimensionless]

    # participation  Beta(α_p, β_p) per neuron
    alpha_p:         float = 3.0                   # α_p
    beta_p:          float = 1.0                   # β_p

    # firing rates
    lambda_burst:    float = 80.0                  # λ_{burst}  [spikes/s]
    lambda_bg:       float = 0.02                  # λ_{bg}     [spikes/s]

    # IFR parameters
    w_size:          float = 0.02                  # Δt         [s]  → 50 Hz
    gaussian_window: float = 0.04                  # σ_{smooth} [s]

    # metadata
    condition:       int   = 0                     # 0=control, 1=patho
    tag:             str   = "control"


# canonical parameter sets  (exported so test scripts don't hard-code values)
CONTROL_PARAMS = BurstParams(
    lambda_b=0.15,  mu_d=-1.3862943611198906, sigma_d=0.40,   # median burst ~250 ms
    alpha_p=3.0,    beta_p=1.0,                                # high participation
    lambda_burst=80.0, lambda_bg=0.02,
    condition=0,    tag="control",
)

PATHO_PARAMS = BurstParams(
    lambda_b=0.30,  mu_d=-1.6094379124341003, sigma_d=0.80,   # higher rate, irregular
    alpha_p=2.0,    beta_p=2.0,                                # lower, uniform participation
    lambda_burst=120.0, lambda_bg=0.05,
    condition=1,    tag="patho",
)

# --------------------------------------------------------------------------- #
# generative model
# --------------------------------------------------------------------------- #
def generate_spike_times(
    params: BurstParams,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Return S_i ⊂ [0, T_rec) for each neuron i ∈ {1,…,N}.

    Parameters
    ----------
    params : BurstParams — all model parameters (see module docstring).
    rng    : np.random.Generator — explicit RNG for reproducibility.

    Returns
    -------
    spike_times : list of N sorted float64 arrays [s].
    """
    N, T = params.n_neurons, params.duration_s

    # J ~ Poisson(λ_b · T_rec) burst onsets, sorted ascending
    J = int(rng.poisson(params.lambda_b * T))
    burst_onsets    = np.sort(rng.uniform(0.0, T, size=J))             # t_b^{(j)} [s]
    burst_durations = rng.lognormal(params.mu_d, params.sigma_d, size=J)  # D^{(j)} [s]

    # p_i ~ Beta(α_p, β_p), drawn once per neuron
    p_part = rng.beta(params.alpha_p, params.beta_p, size=N)           # (N,)

    spike_times: List[np.ndarray] = []
    for i in range(N):
        spikes: List[float] = []

        # background: Poisson(λ_{bg}) on [0, T_rec)
        n_bg = int(rng.poisson(params.lambda_bg * T))
        if n_bg > 0:
            spikes.extend(rng.uniform(0.0, T, size=n_bg).tolist())

        # burst contribution
        for j in range(J):
            if rng.random() > p_part[i]:    # Z_i^{(j)} = 0
                continue
            t0  = float(burst_onsets[j])
            t1  = min(t0 + float(burst_durations[j]), T)   # clamp to [0, T_rec)
            if t1 <= t0:
                continue
            n_b = int(rng.poisson(params.lambda_burst * (t1 - t0)))
            if n_b > 0:
                spikes.extend(rng.uniform(t0, t1, size=n_b).tolist())

        spike_times.append(np.sort(np.array(spikes, dtype=np.float64)))

    return spike_times


def compute_ifr_trace(
    spike_times: List[np.ndarray],
    params: BurstParams,
) -> Tuple[np.ndarray, float]:
    """Compute the smoothed cumulative IFR R̃ ∈ ℝ_{≥0}^K at f_s^{IFR} Hz.

    Parameters
    ----------
    spike_times : list of N sorted float64 arrays [s].
    params      : BurstParams  (uses w_size = Δt and gaussian_window = σ_smooth).

    Returns
    -------
    ifr_trace : (K,) float32,  K = ⌊T_rec / Δt⌋.
    fs_ifr    : float = 1 / Δt  [Hz].

    Notes
    -----
    C[k] = Σ_{i=1}^N |S_i ∩ [kΔt, (k+1)Δt)|  is a non-negative integer array.
    R̃[k] = (C ∗ g_{σ_s})[k] with σ_s = σ_smooth / Δt bins; gaussian_filter1d
    applies a separable Gaussian, which preserves R̃[k] ≥ 0 since C[k] ≥ 0.
    np.clip(..., 0, None) removes the negligible negative floating-point noise
    that can arise from the Gaussian's tails at machine precision.
    """
    Dt    = params.w_size
    T     = params.duration_s
    K     = int(T / Dt)                       # K = ⌊T_rec / Δt⌋
    fs_ifr = 1.0 / Dt

    # bin edges: [0, Δt, 2Δt, …, KΔt]  — length K+1
    edges = np.arange(K + 1, dtype=np.float64) * Dt

    # C[k] = population spike count in bin k
    C = np.zeros(K, dtype=np.float64)
    for st in spike_times:
        if st.size > 0:
            c, _ = np.histogram(st, bins=edges)
            C += c

    # R̃ = C ∗ g_{σ_s},  σ_s in bins
    sigma_bins = params.gaussian_window / Dt
    R_tilde    = gaussian_filter1d(C, sigma=sigma_bins)
    R_tilde    = np.clip(R_tilde, 0.0, None)   # non-negativity guard
    return R_tilde.astype(np.float32), float(fs_ifr)


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def save_burst_npz(
    spike_times: List[np.ndarray],
    ifr_trace:   np.ndarray,
    fs_ifr:      float,
    params:      BurstParams,
    path:        str,
) -> None:
    """Save all generated artefacts to a .npz archive.

    Saved arrays
    ------------
    ifr_trace        : (K,) float32 — smoothed cumulative IFR at fs_ifr Hz.
    spike_times_obj  : (N,) object  — S_i [s] as float64 arrays (allow_pickle=True
                       required when loading).
    fs_ifr           : scalar float64 [Hz].
    n_neurons        : scalar int64   [count].
    duration_s       : scalar float64 [s].
    n_bursts_expected: scalar float64 = λ_b · T_rec.
    condition        : scalar int64   — 0=control, 1=patho.
    params_json      : 0-d object     — JSON-serialised BurstParams.
    """
    spike_times_obj = np.empty(len(spike_times), dtype=object)
    for i, st in enumerate(spike_times):
        spike_times_obj[i] = st.astype(np.float64)

    # JSON-safe copy of params (all numpy scalars cast to Python built-ins)
    d = asdict(params)
    d = {k: (float(v) if isinstance(v, (np.floating, float)) else
             int(v)   if isinstance(v, (np.integer, int))    else v)
         for k, v in d.items()}

    np.savez(
        path,
        ifr_trace         = ifr_trace,
        spike_times_obj   = spike_times_obj,
        fs_ifr            = np.float64(fs_ifr),
        n_neurons         = np.int64(params.n_neurons),
        duration_s        = np.float64(params.duration_s),
        n_bursts_expected = np.float64(params.lambda_b * params.duration_s),
        condition         = np.int64(params.condition),
        params_json       = np.array(json.dumps(d), dtype=object),
    )


def load_burst_npz(path: str) -> Dict:
    """Load a .npz archive produced by save_burst_npz.

    Returns a dict with keys: ifr_trace, spike_times, fs_ifr, n_neurons,
    duration_s, n_bursts_expected, condition, params (BurstParams).
    """
    raw = np.load(path, allow_pickle=True)
    params_d = json.loads(str(raw["params_json"]))
    return {
        "ifr_trace"        : raw["ifr_trace"].astype(np.float32),
        "spike_times"      : list(raw["spike_times_obj"]),
        "fs_ifr"           : float(raw["fs_ifr"]),
        "n_neurons"        : int(raw["n_neurons"]),
        "duration_s"       : float(raw["duration_s"]),
        "n_bursts_expected": float(raw["n_bursts_expected"]),
        "condition"        : int(raw["condition"]),
        "params"           : BurstParams(**params_d),
    }


# --------------------------------------------------------------------------- #
# visualization (headless, Agg backend)
# --------------------------------------------------------------------------- #
def plot_raster(
    spike_times: List[np.ndarray],
    ifr_trace:   np.ndarray,
    fs_ifr:      float,
    params:      BurstParams,
    out_path:    str,
) -> str:
    """Save a two-panel figure: spike raster (top) and R̃ trace (bottom).

    Panel 1 — raster: neuron index i vs spike time t [s].
    Panel 2 — smoothed cumulative IFR: R̃[k] vs k/f_s^{IFR} [s].

    Parameters
    ----------
    spike_times : list of N arrays [s].
    ifr_trace   : (K,) float32, R̃[k].
    fs_ifr      : float [Hz].
    params      : BurstParams — used only for plot annotations.
    out_path    : destination PNG path (parent dir created if missing).

    Returns
    -------
    out_path    : the written path (for chaining / reporting).
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    N      = len(spike_times)
    t_ifr  = np.arange(len(ifr_trace), dtype=np.float32) / float(fs_ifr)

    fig, axes = plt.subplots(
        2, 1, figsize=(15, 7),
        gridspec_kw={"height_ratios": [3, 1.5]},
        sharex=True,
    )
    fig.suptitle(
        f"Synthetic MEA — N={N} neurons, T_rec={params.duration_s:.0f} s, "
        f"condition={params.tag!r}  "
        f"(λ_b={params.lambda_b} bursts/s, "
        f"median D={np.exp(params.mu_d)*1e3:.0f} ms, "
        f"λ_{{burst}}={params.lambda_burst} Hz)",
        fontsize=10,
    )

    # --- panel 1: raster ---
    ax = axes[0]
    for i, st in enumerate(spike_times):
        if st.size > 0:
            ax.scatter(st, np.full(st.size, i, dtype=np.float32),
                       s=0.35, c="black", linewidths=0, rasterized=True)
    ax.set_xlim(0.0, params.duration_s)
    ax.set_ylim(-0.5, N - 0.5)
    ax.invert_yaxis()
    ax.set_ylabel("Neuron index  i")
    ax.set_title("Spike raster  (S_i  ⊂  [0, T_rec))")
    ax.grid(axis="x", alpha=0.20)

    # --- panel 2: smoothed IFR ---
    ax = axes[1]
    ax.fill_between(t_ifr, ifr_trace, alpha=0.50, color="tab:blue")
    ax.plot(t_ifr, ifr_trace, lw=0.8, color="tab:blue")
    ax.set_xlim(0.0, params.duration_s)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Time  t  [s]")
    ax.set_ylabel("R̃[k]  [spikes]")
    ax.set_title(
        f"Smoothed cumulative IFR  (Δt={params.w_size*1e3:.0f} ms, "
        f"σ_smooth={params.gaussian_window*1e3:.0f} ms)"
    )
    ax.grid(axis="x", alpha=0.20)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# generation script
# --------------------------------------------------------------------------- #
def generate_all(out_dir: str = "./burst_data", seed: int = 0) -> str:
    """Generate control_0, control_1, patho_0 and write burst_specs.json.

    Parameters
    ----------
    out_dir : destination directory (created if missing).
    seed    : base seed for the global RNG; each trace consumes a contiguous
              portion of the RNG stream, so all three are jointly reproducible
              from this single seed.

    Returns
    -------
    specs_path : absolute path to the written burst_specs.json.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # three traces: 2 control (different rng draws), 1 patho
    d_ctrl = asdict(CONTROL_PARAMS)
    d_path = asdict(PATHO_PARAMS)
    trace_defs = [
        BurstParams(**{**d_ctrl, "tag": "control_0", "condition": 0}),
        BurstParams(**{**d_ctrl, "tag": "control_1", "condition": 0}),
        BurstParams(**{**d_path, "tag": "patho_0",   "condition": 1}),
    ]

    specs_list = []
    for params in trace_defs:
        print(f"\n[generate] {params.tag!r}  (condition={params.condition})", flush=True)

        spike_times        = generate_spike_times(params, rng)
        ifr_trace, fs_ifr  = compute_ifr_trace(spike_times, params)

        n_spikes = sum(len(s) for s in spike_times)
        print(f"           spikes={n_spikes:,}  IFR shape={ifr_trace.shape}"
              f"  fs_ifr={fs_ifr:.1f} Hz  "
              f"max(R̃)={ifr_trace.max():.2f}  mean(R̃)={ifr_trace.mean():.4f}",
              flush=True)

        npz_path = os.path.abspath(os.path.join(out_dir, f"{params.tag}.npz"))
        png_path = os.path.abspath(os.path.join(out_dir, f"{params.tag}_raster.png"))

        save_burst_npz(spike_times, ifr_trace, fs_ifr, params, npz_path)
        plot_raster(spike_times, ifr_trace, fs_ifr, params, png_path)

        print(f"           -> {npz_path}", flush=True)
        print(f"           -> {png_path}", flush=True)

        specs_list.append({
            "npz_path" : npz_path,
            "condition": params.condition,
            "tag"      : params.tag,
        })

    specs_path = os.path.join(out_dir, "burst_specs.json")
    with open(specs_path, "w") as fh:
        json.dump(specs_list, fh, indent=2)
    print(f"\n[generate] burst_specs.json -> {specs_path}", flush=True)
    return os.path.abspath(specs_path)


# --------------------------------------------------------------------------- #
# smoke test
# --------------------------------------------------------------------------- #
_RESULTS: List[bool] = []


def _check(name: str, cond: bool, detail: str = "") -> bool:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"   | {detail}" if detail else ""))
    _RESULTS.append(bool(cond))
    return bool(cond)


def smoke_test() -> int:
    """Verify the generative model and persistence on a short recording.

    Checks (32 total, run twice):
        * spike-train properties (sorted, in-range, non-empty)
        * IFR shape, dtype, non-negativity, fs_ifr
        * save/load round-trip (all arrays identical)
        * control vs patho IFR statistics differ in the expected direction
        * burst count ≈ expected
        * raster plot written

    Returns 0 on full pass, 1 on any failure.
    """
    print("\n========== smoke_test_generate_burst_data ==========")
    _RESULTS.clear()

    # --- minimal params for speed ---
    p = BurstParams(n_neurons=20, duration_s=100.0, lambda_b=0.20,
                    condition=0, tag="smoke")
    rng = np.random.default_rng(42)

    # 1. generate
    spike_times = generate_spike_times(p, rng)
    _check("N spike trains returned", len(spike_times) == p.n_neurons,
           f"got {len(spike_times)}")

    all_ok = all(
        st.size == 0 or (
            np.all(np.diff(st) >= 0)          # sorted
            and float(st[0]) >= 0.0            # lower bound
            and float(st[-1]) < p.duration_s  # upper bound
        )
        for st in spike_times
    )
    _check("all spike trains sorted and in [0, T_rec)", all_ok)

    _check("at least one neuron fired",
           any(len(s) > 0 for s in spike_times))

    # 2. IFR
    ifr, fs_ifr = compute_ifr_trace(spike_times, p)
    K_expected   = int(p.duration_s / p.w_size)
    _check("IFR length K = ⌊T_rec/Δt⌋",
           len(ifr) == K_expected, f"got {len(ifr)}, expected {K_expected}")
    _check("IFR dtype float32", ifr.dtype == np.float32)
    _check("IFR non-negative", bool(np.all(ifr >= 0.0)),
           f"min(R̃)={ifr.min():.6f}")
    _check("fs_ifr = 1/w_size",
           np.isclose(fs_ifr, 1.0 / p.w_size), f"got {fs_ifr}")

    # 3. save / load round-trip
    with tempfile.TemporaryDirectory() as td:
        npz = os.path.join(td, "test.npz")
        save_burst_npz(spike_times, ifr, fs_ifr, p, npz)
        _check("npz file written", os.path.exists(npz))

        d = load_burst_npz(npz)
        _check("roundtrip: ifr_trace identical",
               np.array_equal(d["ifr_trace"], ifr))
        _check("roundtrip: fs_ifr",
               np.isclose(d["fs_ifr"], fs_ifr))
        _check("roundtrip: n_neurons",
               d["n_neurons"] == p.n_neurons)
        _check("roundtrip: condition",
               d["condition"] == p.condition)
        _check("roundtrip: n spike trains",
               len(d["spike_times"]) == p.n_neurons)
        _check("roundtrip: first train identical",
               np.array_equal(d["spike_times"][0], spike_times[0]))
        _check("roundtrip: params tag",
               d["params"].tag == p.tag)

        # 4. raster plot
        png = os.path.join(td, "raster.png")
        plot_raster(spike_times, ifr, fs_ifr, p, png)
        _check("raster PNG written", os.path.exists(png))

    # 5. burst count sanity  (J ~ Poisson(λ_b·T_rec) → expected = λ_b·T_rec)
    rng2 = np.random.default_rng(7)
    counts = [
        int(np.random.default_rng(s).poisson(p.lambda_b * p.duration_s))
        for s in range(200)
    ]
    expected_J = p.lambda_b * p.duration_s
    # empirical Poisson mean should be within 3·std = 3·sqrt(expected_J)
    _check("burst count distribution: mean ≈ λ_b·T_rec",
           abs(np.mean(counts) - expected_J) < 3 * np.sqrt(expected_J),
           f"mean={np.mean(counts):.2f}, expected={expected_J:.2f}")

    # 6. control vs patho IFR differ in expected direction
    rng3 = np.random.default_rng(99)
    pc   = BurstParams(n_neurons=40, duration_s=120.0,
                       **{k: v for k, v in asdict(CONTROL_PARAMS).items()
                          if k not in ("n_neurons","duration_s","tag","condition")},
                       condition=0, tag="c")
    pp   = BurstParams(n_neurons=40, duration_s=120.0,
                       **{k: v for k, v in asdict(PATHO_PARAMS).items()
                          if k not in ("n_neurons","duration_s","tag","condition")},
                       condition=1, tag="p")
    ifr_c, _ = compute_ifr_trace(generate_spike_times(pc, rng3), pc)
    ifr_p, _ = compute_ifr_trace(generate_spike_times(pp, rng3), pp)
    _check("patho mean IFR > control mean IFR",
           float(ifr_p.mean()) > float(ifr_c.mean()),
           f"patho={ifr_p.mean():.4f}, control={ifr_c.mean():.4f}")
    _check("patho max IFR > control max IFR",
           float(ifr_p.max()) > float(ifr_c.max()),
           f"patho={ifr_p.max():.2f}, control={ifr_c.max():.2f}")

    # 7. reproducibility: same seed -> same spike times
    r1 = generate_spike_times(BurstParams(n_neurons=5, duration_s=20.0),
                               np.random.default_rng(0))
    r2 = generate_spike_times(BurstParams(n_neurons=5, duration_s=20.0),
                               np.random.default_rng(0))
    r3 = generate_spike_times(BurstParams(n_neurons=5, duration_s=20.0),
                               np.random.default_rng(1))
    _check("reproducibility: same seed -> identical",
           all(np.array_equal(a, b) for a, b in zip(r1, r2)))
    _check("reproducibility: diff seed -> differs",
           not all(np.array_equal(a, b) for a, b in zip(r1, r3)))

    n_pass = sum(_RESULTS)
    n_tot  = len(_RESULTS)
    print(f"\n{'='*52}")
    print(f"  {n_pass}/{n_tot} checks passed")
    return 0 if n_pass == n_tot else 1


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic MEA burst data")
    parser.add_argument("--out-dir", default="./burst_data")
    parser.add_argument("--seed",    type=int, default=0)
    parser.add_argument("--smoke",   action="store_true",
                        help="run internal smoke test and exit")
    args = parser.parse_args()

    if args.smoke:
        # run the test twice (directive 4)
        print("=== CONTROL RUN 1 ==="); rc1 = smoke_test()
        print("\n=== CONTROL RUN 2 ==="); rc2 = smoke_test()
        sys.exit(max(rc1, rc2))

    generate_all(out_dir=args.out_dir, seed=args.seed)
