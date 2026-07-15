#!/usr/bin/env python3
"""
verify_env_hpc.py
=================

Fast, honest verification that the conda env can actually run the 1D-CNN MEA
contrastive optimization pipeline. It does NOT just import packages -- it
exercises ONE real sub-API of each dependency the pipeline actually calls, so a
subtly broken build (e.g. a scipy that imports but crashes on CubicSpline
because of the libstdc++ / GLIBCXX issue) is caught here in seconds instead of
20 minutes into a search job.

This is a SEPARATE file from the repo's own verify_env.py on purpose: the repo
one also checks optuna, which the current pipeline does not import.
environment_hpc.yml intentionally omits optuna, so the repo verifier would
report spurious optuna failures. Use THIS file with the HPC setup.

Run it right after setup_env_davinci.sh, and again at the top of any session
before you qsub:

    conda activate <your_env_name>
    python verify_env_hpc.py

Exit code 0 and "ALL CHECKS PASSED" means the env is ready. Non-zero lists
exactly which check failed and the underlying error, so you know whether it is
a missing package, a libstdc++ problem, or a version mismatch.

HPC note (hpc-python-compat): pure ASCII.
"""

from __future__ import annotations

import sys

PASSED = 0
FAILED = 0
FAILURES = []


def check(label, fn):
    """Run fn(); record pass/fail. fn must raise on failure and return a short
    status string (or None) on success."""
    global PASSED, FAILED
    try:
        detail = fn()
        PASSED += 1
        suffix = (" -- %s" % detail) if detail else ""
        print("PASS  %s%s" % (label, suffix))
    except Exception as ex:
        FAILED += 1
        FAILURES.append((label, "%s: %s" % (type(ex).__name__, ex)))
        print("FAIL  %s  (%s: %s)" % (label, type(ex).__name__, ex))


# --------------------------------------------------------------------------- #
# 1. Python version
# --------------------------------------------------------------------------- #
def _python_version():
    v = sys.version_info
    if v < (3, 9):
        raise RuntimeError("python %d.%d too old; pipeline targets 3.11"
                           % (v.major, v.minor))
    return "python %d.%d.%d" % (v.major, v.minor, v.micro)


# --------------------------------------------------------------------------- #
# 2. numpy: default_rng + a real array op
# --------------------------------------------------------------------------- #
def _numpy():
    import numpy as np
    rng = np.random.default_rng(0)
    x = rng.normal(size=1000)
    _ = np.histogram(x, bins=10)
    return "numpy %s" % np.__version__


# --------------------------------------------------------------------------- #
# 3. scipy: the three APIs the pipeline uses. CubicSpline is the libstdc++
#    canary -- it lives in a compiled extension that crashes if GLIBCXX is
#    wrong, exactly the davinci-1 failure mode.
# --------------------------------------------------------------------------- #
def _scipy():
    import numpy as np
    from scipy.interpolate import CubicSpline
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    t = np.linspace(0, 1, 50)
    y = np.sin(2 * np.pi * t)
    cs = CubicSpline(t, y)
    _ = cs(0.5)
    _ = gaussian_filter1d(y, sigma=2.0)
    _ = find_peaks(y)
    import scipy
    return "scipy %s (CubicSpline OK -- libstdc++ fine)" % scipy.__version__


# --------------------------------------------------------------------------- #
# 4. matplotlib: Agg backend renders headless
# --------------------------------------------------------------------------- #
def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.canvas.draw()             # forces the Agg renderer to actually run
    plt.close(fig)
    return "matplotlib %s (Agg OK)" % matplotlib.__version__


# --------------------------------------------------------------------------- #
# 5. scikit-learn: KMeans + ARI, the search objective's core
# --------------------------------------------------------------------------- #
def _sklearn():
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (30, 4)), rng.normal(5, 1, (30, 4))])
    labels = np.array([0] * 30 + [1] * 30)
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
    ari = adjusted_rand_score(labels, km.labels_)
    _ = PCA(n_components=2).fit_transform(X)
    if ari < 0.9:
        raise RuntimeError("KMeans ARI %.3f on trivially separable data "
                           "(expected ~1.0)" % ari)
    import sklearn
    return "scikit-learn %s (KMeans ARI=%.3f)" % (sklearn.__version__, ari)


# --------------------------------------------------------------------------- #
# 6. scikit-optimize: gp_minimize on a trivial 1-D function
# --------------------------------------------------------------------------- #
def _skopt():
    from skopt import gp_minimize
    from skopt.space import Real
    res = gp_minimize(lambda p: (p[0] - 0.3) ** 2,
                      [Real(-1.0, 1.0)], n_calls=8, n_initial_points=4,
                      random_state=0)
    import skopt
    return "scikit-optimize %s (gp_minimize ran, x*=%.3f)" % (
        skopt.__version__, res.x[0])


# --------------------------------------------------------------------------- #
# 7. torch: tensor op + a backward pass; report CPU/CUDA
# --------------------------------------------------------------------------- #
def _torch():
    import torch
    x = torch.randn(8, 4, requires_grad=True)
    y = (x ** 2).sum()
    y.backward()
    if x.grad is None:
        raise RuntimeError("autograd produced no gradient")
    cuda = torch.cuda.is_available()
    dev = (" (CUDA: %s)" % torch.cuda.get_device_name(0)) if cuda \
        else " (CPU-only build)"
    return "torch %s, cuda_available=%s%s" % (torch.__version__, cuda, dev)


# --------------------------------------------------------------------------- #
# 8. pytorch-metric-learning: a real TripletMarginLoss forward pass, which is
#    exactly what train.py uses. This is the package Colab was missing.
# --------------------------------------------------------------------------- #
def _pml():
    import torch
    from pytorch_metric_learning import losses, miners, distances, reducers
    emb = torch.randn(12, 8, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    loss_fn = losses.TripletMarginLoss(
        margin=0.3,
        distance=distances.CosineSimilarity(),
        reducer=reducers.AvgNonZeroReducer())
    miner = miners.TripletMarginMiner(margin=0.3,
                                      distance=distances.CosineSimilarity(),
                                      type_of_triplets="all")
    hard = miner(emb, labels)
    loss = loss_fn(emb, labels, hard)
    loss.backward()
    import pytorch_metric_learning as pml
    return "pytorch-metric-learning %s (TripletMarginLoss fwd+bwd OK)" % pml.__version__


# --------------------------------------------------------------------------- #
# 9. tqdm: importable and constructs
# --------------------------------------------------------------------------- #
def _tqdm():
    from tqdm import tqdm
    for _ in tqdm(range(3), disable=True):
        pass
    import tqdm as _t
    return "tqdm %s" % _t.__version__


def main():
    print("=" * 70)
    print("verify_env_hpc.py -- checking the 1D-CNN MEA pipeline environment")
    print("=" * 70)
    checks = [
        ("python version", _python_version),
        ("numpy", _numpy),
        ("scipy (+ libstdc++ canary)", _scipy),
        ("matplotlib (Agg)", _matplotlib),
        ("scikit-learn", _sklearn),
        ("scikit-optimize", _skopt),
        ("torch", _torch),
        ("pytorch-metric-learning", _pml),
        ("tqdm", _tqdm),
    ]
    for label, fn in checks:
        check(label, fn)

    print("=" * 70)
    print("PASSED: %d   FAILED: %d" % (PASSED, FAILED))
    if FAILED:
        print("")
        print("FAILURES:")
        for label, detail in FAILURES:
            print("  - %s: %s" % (label, detail))
        print("")
        print("Common fixes:")
        print("  * scipy/CubicSpline fails with GLIBCXX -> the libstdc++ hook did")
        print("    not take effect. Check: echo $LD_LIBRARY_PATH (should start with")
        print("    your env's lib dir), and re-activate the env.")
        print("  * pytorch-metric-learning missing -> pip install pytorch-metric-learning")
        print("  * torch missing -> re-run setup_env_davinci.sh")
        print("ENVIRONMENT NOT READY")
        sys.exit(1)
    print("ALL CHECKS PASSED -- environment is ready.")
    sys.exit(0)


if __name__ == "__main__":
    main()
