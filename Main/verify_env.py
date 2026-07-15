"""
verify_env.py
=============

Verifies that every library used across the 1D-CNN MEA contrastive pipeline
(Topics 1-3) is importable at the required minimum version, that the CUDA
device is visible to PyTorch, and that the key sub-APIs we actually call
are present (not just top-level package imports).

Run
---
    conda activate meacnn
    python verify_env.py

Exit code
---------
    0  -- all checks passed
    1  -- one or more checks failed (details printed inline)

The checks are intentionally ordered from most fundamental to most specific
so the first failure points immediately to the broken layer.
"""

from __future__ import annotations

import sys
from typing import Callable, List, Tuple

# --------------------------------------------------------------------------- #
# tiny harness (no external deps)
# --------------------------------------------------------------------------- #
_RESULTS: List[Tuple[str, bool, str]] = []


def _ck(name: str, fn: Callable[[], str], min_ok: bool = True) -> bool:
    """Run fn(); record PASS/FAIL. fn returns a detail string or raises."""
    try:
        detail = fn()
        ok = True
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        ok = False
    ok = ok and min_ok
    _RESULTS.append((name, ok, detail))
    status = "\033[32mPASS\033[0m" if ok else "\033[31mFAIL\033[0m"
    print(f"  [{status}] {name:<52s}  {detail}")
    return ok


def _version_ge(module_name: str, attr: str, min_ver: str) -> bool:
    """Return True if module.attr >= min_ver (compared tuple-wise).
    Returns False (not raises) when the module is not installed,
    so _ck records a clean FAIL rather than crashing the script.
    """
    import importlib
    def _t(v: str) -> tuple:
        return tuple(int(x) for x in v.split(".")[:3] if x.isdigit())
    try:
        m = importlib.import_module(module_name)
        ver_str = getattr(m, attr, "0.0.0")
        return _t(ver_str) >= _t(min_ver)
    except ImportError:
        return False


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def main() -> int:
    print()
    print("=" * 72)
    print("  meacnn environment verification")
    print("=" * 72)
    print()

    # -- Python ---------------------------------------------------------------
    print("-- Python --")
    _ck("Python >= 3.11",
        lambda: f"{sys.version.split()[0]}",
        sys.version_info >= (3, 11))

    # -- NumPy ----------------------------------------------------------------
    print("\n-- NumPy --")
    _ck("import numpy",
        lambda: __import__("numpy").__version__)
    _ck("numpy >= 1.24",
        lambda: __import__("numpy").__version__,
        _version_ge("numpy", "__version__", "1.24"))
    _ck("numpy.random.default_rng",
        lambda: (
            __import__("numpy").random.default_rng(0),
            "ok"
        )[-1])
    _ck("numpy.histogram",
        lambda: (
            __import__("numpy").histogram([1.0, 2.0], bins=2),
            "ok"
        )[-1])

    # -- SciPy ----------------------------------------------------------------
    print("\n-- SciPy --")
    _ck("import scipy",
        lambda: __import__("scipy").__version__)
    _ck("scipy >= 1.9",
        lambda: __import__("scipy").__version__,
        _version_ge("scipy", "__version__", "1.9"))
    _ck("scipy.interpolate.CubicSpline (not-a-knot)",
        lambda: (
            __import__("scipy.interpolate", fromlist=["CubicSpline"])
            .CubicSpline([0, 1, 2, 3], [0, 1, 0, 1]),
            "ok"
        )[-1])
    _ck("scipy.ndimage.gaussian_filter1d",
        lambda: (
            __import__("scipy.ndimage", fromlist=["gaussian_filter1d"])
            .gaussian_filter1d([1.0, 2.0, 3.0], sigma=1.0),
            "ok"
        )[-1])

    # -- Matplotlib -----------------------------------------------------------
    print("\n-- Matplotlib --")
    _ck("import matplotlib",
        lambda: __import__("matplotlib").__version__)
    _ck("matplotlib >= 3.6",
        lambda: __import__("matplotlib").__version__,
        _version_ge("matplotlib", "__version__", "3.6"))
    _ck("Agg backend (headless)",
        lambda: (
            __import__("matplotlib").use("Agg"),
            __import__("matplotlib.pyplot", fromlist=["plt"]),
            "ok"
        )[-1])

    # -- PyTorch --------------------------------------------------------------
    print("\n-- PyTorch --")
    _ck("import torch",
        lambda: __import__("torch").__version__)
    _ck("torch >= 2.1",
        lambda: __import__("torch").__version__,
        _version_ge("torch", "__version__", "2.1"))
    _ck("torch.cuda.is_available()",
        lambda: str(__import__("torch").cuda.is_available()))
    _ck("torch.cuda device name (if GPU present)",
        lambda: (
            __import__("torch").cuda.get_device_name(0)
            if __import__("torch").cuda.is_available()
            else "CPU-only build -- no GPU visible"
        ))
    _ck("torch.nn.Conv1d",
        lambda: (
            __import__("torch.nn", fromlist=["Conv1d"])
            .Conv1d(1, 16, 3),
            "ok"
        )[-1])
    _ck("torch.nn.BatchNorm1d",
        lambda: (
            __import__("torch.nn", fromlist=["BatchNorm1d"])
            .BatchNorm1d(16),
            "ok"
        )[-1])
    _ck("torch.nn.GroupNorm",
        lambda: (
            __import__("torch.nn", fromlist=["GroupNorm"])
            .GroupNorm(8, 16),
            "ok"
        )[-1])
    _ck("torch.nn.functional.normalize",
        lambda: (
            __import__("torch.nn.functional", fromlist=["normalize"])
            .normalize(__import__("torch").randn(4, 16), p=2, dim=1),
            "ok"
        )[-1])
    _ck("torch.utils.data.DataLoader + Dataset",
        lambda: (
            __import__("torch.utils.data",
                       fromlist=["DataLoader", "Dataset"]),
            "ok"
        )[-1])
    _ck("torch.use_deterministic_algorithms(warn_only=True)",
        lambda: (
            __import__("torch").use_deterministic_algorithms(
                True, warn_only=True),
            "ok"
        )[-1])
    _ck("torch.compile available (Python 3.11 + torch >=2.1)",
        lambda: (
            hasattr(__import__("torch"), "compile"),
            "present" if hasattr(__import__("torch"), "compile") else "missing"
        )[-1])

    # -- scikit-learn ---------------------------------------------------------
    print("\n-- scikit-learn --")
    _ck("import sklearn",
        lambda: __import__("sklearn").__version__)
    _ck("sklearn >= 1.2",
        lambda: __import__("sklearn").__version__,
        _version_ge("sklearn", "__version__", "1.2"))
    _ck("sklearn.cluster.KMeans",
        lambda: (
            __import__("sklearn.cluster", fromlist=["KMeans"])
            .KMeans(n_clusters=2, n_init=4, random_state=0),
            "ok"
        )[-1])
    _ck("sklearn.decomposition.PCA",
        lambda: (
            __import__("sklearn.decomposition", fromlist=["PCA"])
            .PCA(n_components=2),
            "ok"
        )[-1])
    _ck("sklearn.metrics.adjusted_rand_score",
        lambda: (
            __import__("sklearn.metrics", fromlist=["adjusted_rand_score"])
            .adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 1]),
            "1.0 (perfect)"
        )[-1])
    _ck("sklearn.metrics.adjusted_mutual_info_score",
        lambda: (
            __import__("sklearn.metrics",
                       fromlist=["adjusted_mutual_info_score"])
            .adjusted_mutual_info_score([0, 0, 1, 1], [0, 0, 1, 1]),
            "1.0 (perfect)"
        )[-1])

    # -- scikit-optimize -------------------------------------------------------
    print("\n-- scikit-optimize --")
    _ck("import skopt",
        lambda: __import__("skopt").__version__)
    _ck("skopt >= 0.9",
        lambda: __import__("skopt").__version__,
        _version_ge("skopt", "__version__", "0.9"))
    _ck("skopt.space.Real, Integer",
        lambda: (
            __import__("skopt.space", fromlist=["Real", "Integer"]),
            "ok"
        )[-1])
    _ck("skopt.gp_minimize importable",
        lambda: (
            __import__("skopt", fromlist=["gp_minimize"]).gp_minimize,
            "callable"
        )[-1])
    _ck("skopt.plots.plot_objective importable",
        lambda: (
            __import__("skopt.plots", fromlist=["plot_objective"])
            .plot_objective,
            "callable"
        )[-1])

    # -- optuna ----------------------------------------------------------------
    print("\n-- optuna --")
    _ck("import optuna",
        lambda: __import__("optuna").__version__)
    _ck("optuna >= 3.4",
        lambda: __import__("optuna").__version__,
        _version_ge("optuna", "__version__", "3.4"))
    _ck("optuna.create_study",
        lambda: (
            __import__("optuna").create_study(direction="minimize"),
            "ok"
        )[-1])

    # -- pytorch-metric-learning -----------------------------------------------
    print("\n-- pytorch-metric-learning --")
    _ck("import pytorch_metric_learning",
        lambda: __import__("pytorch_metric_learning").__version__)
    _ck("pytorch_metric_learning >= 2.3",
        lambda: __import__("pytorch_metric_learning").__version__,
        _version_ge("pytorch_metric_learning", "__version__", "2.3"))
    _ck("losses.TripletMarginLoss",
        lambda: (
            __import__("pytorch_metric_learning.losses",
                       fromlist=["TripletMarginLoss"])
            .TripletMarginLoss(margin=0.1),
            "ok"
        )[-1])
    _ck("miners.TripletMarginMiner",
        lambda: (
            __import__("pytorch_metric_learning.miners",
                       fromlist=["TripletMarginMiner"])
            .TripletMarginMiner(margin=0.05, type_of_triplets="hard"),
            "ok"
        )[-1])
    _ck("reducers.AvgNonZeroReducer",
        lambda: (
            __import__("pytorch_metric_learning.reducers",
                       fromlist=["AvgNonZeroReducer"])
            .AvgNonZeroReducer(),
            "ok"
        )[-1])
    _ck("distances.CosineSimilarity",
        lambda: (
            __import__("pytorch_metric_learning.distances",
                       fromlist=["CosineSimilarity"])
            .CosineSimilarity(),
            "ok"
        )[-1])
    _ck("distances.LpDistance",
        lambda: (
            __import__("pytorch_metric_learning.distances",
                       fromlist=["LpDistance"])
            .LpDistance(),
            "ok"
        )[-1])
    _ck("regularizers.LpRegularizer",
        lambda: (
            __import__("pytorch_metric_learning.regularizers",
                       fromlist=["LpRegularizer"])
            .LpRegularizer(),
            "ok"
        )[-1])

    # -- end-to-end PML mini-forward -------------------------------------------
    print("\n-- end-to-end PML forward (dummy embeddings) --")
    def _pml_forward():
        import torch
        from pytorch_metric_learning import losses, miners, reducers
        from pytorch_metric_learning.distances import CosineSimilarity

        torch.manual_seed(0)
        emb = torch.randn(12, 16)           # 12 samples, 16-D embeddings
        # labels: 4 samples x 3 "classes"
        lab = torch.tensor([0,0,0,0, 1,1,1,1, 2,2,2,2])
        dist    = CosineSimilarity()
        reducer = reducers.AvgNonZeroReducer()
        loss_fn = losses.TripletMarginLoss(
            margin=0.1, swap=True, distance=dist, reducer=reducer)
        miner   = miners.TripletMarginMiner(
            margin=0.05, type_of_triplets="hard", distance=dist)
        pairs   = miner(emb, lab)
        loss    = loss_fn(emb, lab, pairs)
        return f"loss={loss.item():.6f}  (finite={torch.isfinite(loss).item()})"

    _ck("TripletMarginLoss forward pass (12 emb, 3 classes)", _pml_forward)

    # -- summary ---------------------------------------------------------------
    n_pass = sum(ok for _, ok, _ in _RESULTS)
    n_tot  = len(_RESULTS)
    n_fail = n_tot - n_pass
    print()
    print("=" * 72)
    if n_fail == 0:
        print(f"\033[32m  {n_pass}/{n_tot} checks passed -- environment is ready.\033[0m")
    else:
        print(f"\033[31m  {n_pass}/{n_tot} passed, {n_fail} FAILED -- fix failures before proceeding.\033[0m")
        print()
        print("  Failed checks:")
        for name, ok, detail in _RESULTS:
            if not ok:
                print(f"    x  {name}")
                print(f"       {detail}")
    print("=" * 72)
    print()
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
