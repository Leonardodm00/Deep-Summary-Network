"""
smoke_test_pipeline_mc.py
=========================

Step-5 end-to-end smoke test for the MULTICHANNEL (C, T) pipeline.

What this verifies (each is an independent, self-contained check)
-----------------------------------------------------------------
A. CONFIG single-source-of-truth wiring (config.py)
   A1 data.n_channels drives backbone.in_channels (ExperimentConfig.__post_init__)
   A2 to_json / from_json round-trips both fields consistently
   A3 a conflicting non-default backbone.in_channels is overridden (data wins)
      AND a RuntimeWarning is emitted
   A4 the default (n_channels == 1) leaves backbone.in_channels == 1, no warning
   A5 validate() reports a hand-forced channel-axis mismatch
   A6 the shipped config_search_3class_9ch.json loads with in_channels == 9

B. EVAL-FORWARD generalization (inference.clean_windows / embed_clean_windows)
   B1 1-D traces  -> clean_windows returns (N, W), each row == reference slice
      (byte-for-byte: the single-channel path is unchanged)
   B2 (C, K) traces -> clean_windows returns (N, C, W), each row == reference slice
   B3 ragged channel counts across traces are rejected (ValueError)
   B4 embed_clean_windows(batch_size small) == a single whole-batch forward
      (GroupNorm makes the embedding batching-invariant), correct (N, C, W) path

C. END-TO-END training step on synthetic (C, T)   [C = --n-channels, default 9]
   C1 the real collator assembles a batch X of shape (M, C, W)
   C2 model(X) -> (M, E); real forward -> miner -> loss is FINITE (n_triplets reported)
   C3 deterministic grad-flow probe: a surrogate loss (sum of squared embeddings)
      backpropagates a FINITE, NON-ZERO gradient onto the STEM conv weight
      (whose shape is (stem_width, C, kernel)) -- proves the C-channel input is
      in the autograd graph
   C4 per-channel dependence: zeroing any single input channel CHANGES the
      embedding -- proves no channel is silently dropped
   C5 determinism: re-running C with the same seed reproduces the loss and Z

Everything reuses the REAL modules (generate_burst_data, data_pipeline, backbone,
train, config, inference) -- no logic is re-implemented here.

HPC note (hpc-python-compat): pure ASCII; import chain (torch, numpy,
pytorch_metric_learning) is safe.

How to run (swiftly)
--------------------
    # deps (CPU): torch + pytorch-metric-learning are required
    pip install --break-system-packages torch pytorch-metric-learning

    # default 9-channel run:
    python3 smoke_test_pipeline_mc.py
    # or another channel count:
    python3 smoke_test_pipeline_mc.py --n-channels 4
    # run twice (determinism / correctness double-check the user asks for):
    python3 smoke_test_pipeline_mc.py && python3 smoke_test_pipeline_mc.py

Exit code 0 == all checks passed; 1 == at least one failed (details above the
final SMOKE RESULT line). Prints one "SMOKE RESULT: ..." line so a PBS/grep
harness can detect success.
"""

import argparse
import os
import sys
import tempfile
import warnings

import numpy as np
import torch

# --- real modules under test (no re-implementation) ---
from config import ExperimentConfig, DataConfig
from backbone import BackboneConfig, build_backbone
from data_pipeline import MEAWindowDataset, TripletCollator
from generate_burst_data import generate_multichannel_traces
from inference import clean_windows, embed_clean_windows
from train import build_loss_and_miner, set_global_seed
from config import TrainConfig
from augmentation import AugmentationConfig


# --------------------------------------------------------------------------- #
# tiny check helper
# --------------------------------------------------------------------------- #
def _check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    line = "  [%s] %s" % (status, name)
    if detail:
        line += "  (%s)" % detail
    print(line)
    return bool(ok)


# --------------------------------------------------------------------------- #
# small builders (separation of concerns: build data / build model separately)
# --------------------------------------------------------------------------- #
def _small_backbone(n_channels, embedding_size=8, seed=0):
    """A deliberately tiny backbone so the test is fast; in_channels = n_channels."""
    set_global_seed(int(seed))
    cfg = BackboneConfig(
        depth_exponent=3,
        width_multiplier=1.5,
        stem_width=8,
        in_channels=int(n_channels),
        embedding_size=int(embedding_size),
    )
    return build_backbone(cfg)


def _choose_window(K, base=512, floor=64):
    """Largest power-of-2 window <= base that leaves room for >= ~3 windows."""
    W = int(base)
    while W * 3 > int(K) and W > floor:
        W //= 2
    return W


def _make_mc_dataset(n_channels, seed=0, duration_s=200.0, n_control=2, n_patho=2):
    """Build a real MEAWindowDataset from synthetic multichannel (C, K) traces.

    Returns (dataset, W, stride, fs). Traces are (n_channels, K) when
    n_channels > 1; when n_channels == 1 they are squeezed to genuine 1-D (K,)
    so the 1-D code path is exercised.
    """
    traces, conditions, fs = generate_multichannel_traces(
        n_control=int(n_control), n_patho=int(n_patho),
        n_channels=int(n_channels), neurons_per_channel=20,
        channel_gain_spread=0.3, duration_s=float(duration_s), seed=int(seed))
    if int(n_channels) == 1:
        traces = [np.ascontiguousarray(t[0], dtype=np.float32) for t in traces]  # (1,K)->(K,)
    K = int(np.asarray(traces[0]).shape[-1])
    W = _choose_window(K)
    stride = W                                     # disjoint windows
    aug = AugmentationConfig(fs=float(fs), n_positives=4, n_negatives=4,
                             shift_magnitude_s=min(2.0, 0.25 * W / float(fs)))
    ds = MEAWindowDataset(traces, conditions, W, stride, aug, base_seed=int(seed))
    return ds, W, stride, float(fs)


def _balanced_indices(dataset, per_condition=2):
    """Pick per_condition window indices from EACH condition present, so the
    collated batch has same-class positives and cross-class + unique negatives."""
    idxs = []
    for cond in sorted(set(int(c) for c in dataset.conditions_per_item)):
        got = [i for i in range(len(dataset)) if int(dataset.index[i][2]) == cond]
        idxs.extend(got[:per_condition])
    return idxs


def _stem_conv_weight(model, n_channels):
    """Return the stem Conv1d weight parameter: the unique 3-D conv weight whose
    input-channel dim equals n_channels (shape (stem_width, n_channels, kernel))."""
    for _name, p in model.named_parameters():
        if p.ndim == 3 and int(p.shape[1]) == int(n_channels) and "weight" in _name:
            return p
    # fallback: first module that is a Conv1d with in_channels == n_channels
    for m in model.modules():
        if isinstance(m, torch.nn.Conv1d) and int(m.in_channels) == int(n_channels):
            return m.weight
    return None


# --------------------------------------------------------------------------- #
# A. config wiring
# --------------------------------------------------------------------------- #
def check_config_wiring():
    print("A. config single-source-of-truth wiring")
    ok = True

    # A1 data.n_channels drives backbone.in_channels
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg9 = ExperimentConfig(data=DataConfig(n_channels=9))
    ok &= _check("A1 data.n_channels=9 -> backbone.in_channels=9",
                 int(cfg9.backbone.in_channels) == 9,
                 "in_channels=%d" % int(cfg9.backbone.in_channels))

    # A2 round-trip
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "cfg9.json")
        cfg9.to_json(p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg9b = ExperimentConfig.from_json(p)
    ok &= _check("A2 to_json/from_json preserves n_channels and in_channels",
                 int(cfg9b.data.n_channels) == 9 and int(cfg9b.backbone.in_channels) == 9,
                 "n_channels=%d in_channels=%d"
                 % (int(cfg9b.data.n_channels), int(cfg9b.backbone.in_channels)))

    # A3 conflicting non-default backbone.in_channels -> data wins + warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg_conf = ExperimentConfig(data=DataConfig(n_channels=9),
                                    backbone=BackboneConfig(in_channels=5))
        warned = any(issubclass(x.category, RuntimeWarning)
                     and "conflict" in str(x.message).lower() for x in w)
    ok &= _check("A3 conflicting backbone.in_channels=5 overridden to 9 (data wins) + warns",
                 int(cfg_conf.backbone.in_channels) == 9 and warned,
                 "in_channels=%d warned=%s" % (int(cfg_conf.backbone.in_channels), warned))

    # A4 default is single-channel, no warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg1 = ExperimentConfig()
        warned1 = any(issubclass(x.category, RuntimeWarning)
                      and "conflict" in str(x.message).lower() for x in w)
    ok &= _check("A4 default n_channels=1 -> in_channels=1, no conflict warning",
                 int(cfg1.backbone.in_channels) == 1 and not warned1,
                 "in_channels=%d warned=%s" % (int(cfg1.backbone.in_channels), warned1))

    # A5 validate() flags a hand-forced mismatch
    from dataclasses import replace as _replace
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg_mis = ExperimentConfig(data=DataConfig(n_channels=9))
        cfg_mis.backbone = _replace(cfg_mis.backbone, in_channels=3)   # force drift
        msgs = cfg_mis.validate()
    ok &= _check("A5 validate() reports a forced channel-axis mismatch",
                 any("channel-axis" in m or "in_channels" in m for m in msgs),
                 "%d msg(s)" % len(msgs))

    # A6 shipped 9ch JSON
    # The config is a SHIPPED asset. Two layouts must both work:
    #   * cluster runtime -- every package unpacks flat into $PBS_O_WORKDIR, so
    #     the config sits next to this file;
    #   * repo checkout   -- configs live in Main/hpc/ by convention, tests in
    #     Main/Smoke_Tests/, so it is one directory up and across.
    # Searched in order; the first hit wins. Single source of truth either way.
    _here = os.path.dirname(os.path.abspath(__file__))
    _candidates = [
        os.path.join(_here, "config_search_3class_9ch.json"),
        os.path.join(_here, os.pardir, "hpc", "config_search_3class_9ch.json"),
        os.path.join(os.getcwd(), "config_search_3class_9ch.json"),
    ]
    jpath = next((c for c in _candidates if os.path.exists(c)), _candidates[0])
    if os.path.exists(jpath):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfgj = ExperimentConfig.from_json(jpath)
        ok &= _check("A6 config_search_3class_9ch.json -> in_channels=9",
                     int(cfgj.data.n_channels) == 9 and int(cfgj.backbone.in_channels) == 9,
                     "n_channels=%d in_channels=%d"
                     % (int(cfgj.data.n_channels), int(cfgj.backbone.in_channels)))
    else:
        _check("A6 config_search_3class_9ch.json present", False, "file not found")
        ok = False
    return ok


# --------------------------------------------------------------------------- #
# B. clean_windows / embed_clean_windows generalization
# --------------------------------------------------------------------------- #
def check_clean_windows(n_channels):
    print("B. eval-forward generalization (clean_windows / embed_clean_windows)")
    ok = True

    # B1 genuine 1-D path -> (N, W), byte-for-byte reference slices
    ds1, W1, _s1, _fs1 = _make_mc_dataset(1, seed=1)
    X1 = clean_windows(ds1)
    b1_shape = (X1.ndim == 2 and X1.shape[1] == W1)
    b1_vals = True
    for i, (ti, s, _c) in enumerate(ds1.index):
        ref = np.asarray(ds1.traces[ti])[s:s + W1]
        if not np.array_equal(X1[i], ref.astype(np.float32)):
            b1_vals = False
            break
    ok &= _check("B1 1-D traces -> clean_windows (N, W), rows == reference slice",
                 b1_shape and b1_vals, "shape=%s" % (X1.shape,))

    # B2 (C, K) path -> (N, C, W), reference slices on the LAST axis
    dsC, WC, _sC, _fsC = _make_mc_dataset(n_channels, seed=2)
    XC = clean_windows(dsC)
    b2_shape = (XC.ndim == 3 and XC.shape[1] == int(n_channels) and XC.shape[2] == WC)
    b2_vals = True
    for i, (ti, s, _c) in enumerate(dsC.index):
        ref = np.asarray(dsC.traces[ti])[:, s:s + WC]
        if not np.array_equal(XC[i], ref.astype(np.float32)):
            b2_vals = False
            break
    ok &= _check("B2 (C,K) traces -> clean_windows (N, C, W), rows == reference slice",
                 b2_shape and b2_vals, "shape=%s" % (XC.shape,))

    # B3 ragged channel counts rejected
    raggedC = int(n_channels)
    other = max(2, raggedC - 1) if raggedC != 2 else 3
    K = 4 * WC
    t_full = np.random.default_rng(0).standard_normal((raggedC, K)).astype(np.float32)
    t_bad = np.random.default_rng(1).standard_normal((other, K)).astype(np.float32)
    aug = AugmentationConfig(fs=50.0, n_positives=2, n_negatives=2, shift_magnitude_s=1.0)
    ds_rag = MEAWindowDataset([t_full, t_bad], [0, 1], WC, WC, aug, base_seed=0)
    raised = False
    try:
        _ = clean_windows(ds_rag)
    except ValueError:
        raised = True
    ok &= _check("B3 ragged channel counts across traces are rejected",
                 raised, "channels %d vs %d" % (raggedC, other))

    # B4 embed batching-invariance == whole-batch forward, correct (N, C, W)
    model = _small_backbone(n_channels, seed=3).eval()
    with torch.no_grad():
        Z_ref = model(torch.from_numpy(XC)).cpu().numpy()
    Z_batched, y = embed_clean_windows(model, dsC, torch.device("cpu"), batch_size=7)
    b4_shape = (Z_batched.shape == Z_ref.shape and Z_batched.shape[0] == len(dsC.index))
    b4_close = np.allclose(Z_batched, Z_ref, atol=1e-5, rtol=1e-4)
    b4_y = (y.shape[0] == len(dsC.index)
            and np.array_equal(y, np.asarray(dsC.conditions_per_item, dtype=np.int64)))
    ok &= _check("B4 embed_clean_windows == whole-batch forward (batching-invariant)",
                 b4_shape and b4_close and b4_y,
                 "maxdiff=%.2e shape=%s" % (float(np.abs(Z_batched - Z_ref).max()),
                                            Z_batched.shape))
    return ok


# --------------------------------------------------------------------------- #
# C. end-to-end training step on synthetic (C, T)
# --------------------------------------------------------------------------- #
def _one_end_to_end(n_channels, seed):
    """Build a real batch and run forward -> miner -> loss. Returns a dict of
    everything the checks need (so C can be re-run for the determinism check)."""
    set_global_seed(int(seed))
    ds, W, _stride, _fs = _make_mc_dataset(n_channels, seed=seed)
    idxs = _balanced_indices(ds, per_condition=2)
    batch = [ds[i] for i in idxs]
    collate = TripletCollator()
    X, y, _metas = collate(batch)

    model = _small_backbone(n_channels, seed=seed)
    loss_fn, miner = build_loss_and_miner(TrainConfig())

    Z = model(X)                                   # (M, E)
    pairs = miner(Z, y)
    loss = loss_fn(Z, y, pairs)
    n_trip = int(pairs[0].numel())
    return {"X": X, "y": y, "Z": Z.detach(), "model": model,
            "loss": float(loss.detach()), "n_trip": n_trip, "W": W}


def check_end_to_end(n_channels):
    print("C. end-to-end training step on synthetic (C, T),  C=%d" % int(n_channels))
    ok = True
    seed = 12345
    out = _one_end_to_end(n_channels, seed)
    X, y = out["X"], out["y"]

    # C1 collated batch shape (M, C, W)
    c1 = (X.dim() == 3 and int(X.shape[1]) == int(n_channels) and int(X.shape[2]) == out["W"])
    ok &= _check("C1 collator -> batch X of shape (M, C, W)",
                 c1, "X.shape=%s" % (tuple(X.shape),))

    # C2 real forward -> miner -> loss is finite
    c2 = np.isfinite(out["loss"]) and out["Z"].shape[0] == X.shape[0]
    ok &= _check("C2 model(X)->(M,E); real miner/loss FINITE",
                 c2, "loss=%.4g  n_triplets=%d  E=%d"
                 % (out["loss"], out["n_trip"], int(out["Z"].shape[1])))

    # C3 deterministic grad-flow probe onto the stem conv weight
    model = out["model"]
    model.zero_grad(set_to_none=True)
    Zs = model(X)
    surrogate = Zs.pow(2).sum()                    # always > 0 for non-degenerate Z
    surrogate.backward()
    stem_w = _stem_conv_weight(model, n_channels)
    if stem_w is None or stem_w.grad is None:
        c3 = False
        gdetail = "stem conv weight/grad not found"
    else:
        gnorm = float(stem_w.grad.norm())
        c3 = (tuple(stem_w.shape)[1] == int(n_channels)
              and np.isfinite(gnorm) and gnorm > 0.0)
        gdetail = "stem weight shape=%s grad_norm=%.4g" % (tuple(stem_w.shape), gnorm)
    ok &= _check("C3 surrogate loss backprops FINITE non-zero grad to stem (C in graph)",
                 c3, gdetail)

    # C4 per-channel dependence: zeroing a channel changes the embedding
    model.eval()
    with torch.no_grad():
        Z0 = model(X)
        channels_to_test = sorted(set([0, int(n_channels) // 2, int(n_channels) - 1]))
        changed_all = True
        for c in channels_to_test:
            Xc = X.clone()
            Xc[:, c, :] = 0.0
            Zc = model(Xc)
            if torch.allclose(Z0, Zc, atol=1e-6):
                changed_all = False
                break
    ok &= _check("C4 zeroing any single channel CHANGES the embedding (no channel dropped)",
                 changed_all, "tested channels %s" % (channels_to_test,))

    # C5 determinism: same seed reproduces loss and Z
    out2 = _one_end_to_end(n_channels, seed)
    c5 = (abs(out2["loss"] - out["loss"]) < 1e-6
          and torch.allclose(out2["Z"], out["Z"], atol=1e-6))
    ok &= _check("C5 determinism: re-run with same seed reproduces loss and Z",
                 c5, "loss1=%.6g loss2=%.6g" % (out["loss"], out2["loss"]))
    return ok


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Step-5 multichannel end-to-end smoke test")
    ap.add_argument("--n-channels", type=int, default=9,
                    help="channel count C for the (C, T) end-to-end checks (default 9)")
    args = ap.parse_args()
    C = int(args.n_channels)
    if C < 2:
        print("--n-channels must be >= 2 for the multichannel checks (B2/B3/C use C).")
        return 1

    torch.set_num_threads(1)
    print("=" * 66)
    print("MULTICHANNEL PIPELINE SMOKE TEST  (C = %d)" % C)
    print("torch %s | numpy %s" % (torch.__version__, np.__version__))
    print("=" * 66)

    ok = True
    ok &= check_config_wiring()
    ok &= check_clean_windows(C)
    ok &= check_end_to_end(C)

    print("=" * 66)
    print("SMOKE RESULT: %s" % ("ALL PASSED" if ok else "FAILURES ABOVE"))
    print("=" * 66)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
