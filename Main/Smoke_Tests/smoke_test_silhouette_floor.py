"""
smoke_test_silhouette_floor.py

Correctness checks for silhouette_floor.py: the label-shuffled silhouette FLOOR
and the improvement threshold it calibrates. Change 6 reverted the growing-
patience counter that used to share this module (AdaptivePatience) back to a
fixed integer patience whose counter now lives inline in train.py; the groups
that exercised that counter ([A]-[D], [G] of the old smoke_test_adaptive_patience)
are gone, and the fixed-patience RULE they guarded is exercised on a real run in
smoke_test_train.py [E]. What remains here is the floor itself.

The one thing most worth catching
---------------------------------
A THRESHOLD THAT SILENTLY DISABLES EARLY STOPPING. delta = 2 * mu_floor is the
literal reading of the "2 times the floor" request, and mu_floor for a
permutation null sits at or just below zero (a(i) involves no minimum, b(i)
takes a minimum over the C - 1 other classes and a minimum of noisy quantities
is biased downward), so that threshold is approximately 0 or negative -- and a
negative threshold makes a DECREASE count as an improvement. [E] measures
mu_floor and sigma_floor on real embeddings to show where they actually sit, and
[F] checks that mode='floor_location' REFUSES rather than returning such a
threshold, while the default mode='floor_scale' uses kappa * sigma_floor, which
is strictly positive.

Run:
    cd Main && PYTHONPATH=. python3 Smoke_Tests/smoke_test_silhouette_floor.py

Checks:
  E.  silhouette_floor on structured and unstructured embeddings: mu near zero
      under the null, sigma strictly positive, the C > 2 sign asymmetry, and a
      real signal above the floor.
  F.  resolve_min_delta_sil: floor_scale positive, floor_location refuses a
      non-positive mu, absolute passes through, bad modes rejected.
  G'. randomized property (fuzz) test of the floor -> threshold pipeline over
      many random embeddings: the returned null summary's invariants hold and
      floor_scale delta = kappa * sigma_floor stays positive. Ported (in spirit)
      from the deleted growing-patience compatibility fuzz.
"""

import os
import sys

import numpy as np
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from silhouette_floor import (                                   # noqa: E402
    resolve_min_delta_sil,
    silhouette_floor,
)


def _expect(exc_type, fn, must_mention=()):
    try:
        fn()
    except exc_type as ex:
        for token in must_mention:
            assert token in str(ex), (
                "%s raised but message lacks %r: %s"
                % (exc_type.__name__, token, ex))
        return str(ex)
    raise AssertionError("expected %s, none was raised" % (exc_type.__name__,))


def check_floor_statistics():
    rng = np.random.default_rng(1)

    # (i) pure noise, C = 2 and C = 4: mu near 0, sigma > 0
    for C in (2, 4):
        N, E = 96, 12
        Z = rng.normal(size=(N, E))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)   # L2-normalised, as in the pipeline
        y = np.repeat(np.arange(C), N // C)
        floor = silhouette_floor(Z, y, n_permutations=120, metric="cosine",
                                 seed=0)
        assert floor["sigma"] > 0.0, floor
        assert abs(floor["mu"]) < 0.15, (
            "mu_floor = %.5f is not near zero at C = %d" % (floor["mu"], C))
        assert floor["sigma"] < 0.1, (
            "sigma_floor = %.5f is implausibly large for a permutation null"
            % floor["sigma"])
        assert floor["n_valid"] == 120
        print("      C = %d, N = %d noise: mu_floor = %+.5f, sigma_floor = "
              "%.5f, q95 = %+.5f" % (C, N, floor["mu"], floor["sigma"],
                                     floor["q95"]))

    # (ii) the sign asymmetry the module docstring claims: with C > 2, b(i) takes
    #      a minimum over C - 1 classes and is biased downward, so mu_floor tends
    #      NEGATIVE. Averaged over several draws to avoid asserting on one sample.
    mus = []
    for rep in range(6):
        Z = rng.normal(size=(80, 10))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        y = np.repeat(np.arange(4), 20)
        mus.append(silhouette_floor(Z, y, n_permutations=80, seed=rep)["mu"])
    mean_mu = float(np.mean(mus))
    assert mean_mu < 0.0, (
        "expected mu_floor < 0 at C = 4 (min-over-classes bias in b(i)); got "
        "mean %.6f over %d draws: %r" % (mean_mu, len(mus), mus))
    print("      C = 4, 6 independent draws: mean mu_floor = %+.5f (negative, "
          "as the min-over-classes bias in b(i) predicts) -- this is why "
          "2 * mu_floor is not a usable threshold" % mean_mu)

    # (iii) a REAL signal must sit above the floor, or the floor is useless
    centres = np.array([[3.0, 0.0], [-3.0, 0.0]])
    Z = np.vstack([centres[0] + 0.25 * rng.normal(size=(40, 2)),
                   centres[1] + 0.25 * rng.normal(size=(40, 2))])
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    y = np.repeat([0, 1], 40)
    real = float(silhouette_score(Z, y, metric="cosine"))
    floor = silhouette_floor(Z, y, n_permutations=200, seed=0)
    assert real > floor["q95"], (real, floor["q95"])
    z_score = (real - floor["mu"]) / floor["sigma"]
    assert z_score > 5.0, z_score
    print("      two well-separated clusters: s_bar = %.4f vs floor q95 = "
          "%+.4f (%.1f sigma above mu_floor)" % (real, floor["q95"], z_score))


def check_resolve_delta():
    floor_pos = {"mu": 0.02, "sigma": 0.01}
    floor_zero = {"mu": 0.0, "sigma": 0.01}
    floor_neg = {"mu": -0.008, "sigma": 0.01}

    assert abs(resolve_min_delta_sil(floor_pos, kappa=2.0,
                                     mode="floor_scale") - 0.02) < 1e-12
    assert abs(resolve_min_delta_sil(floor_neg, kappa=2.0,
                                     mode="floor_scale") - 0.02) < 1e-12
    assert abs(resolve_min_delta_sil(floor_pos, kappa=2.0,
                                     mode="floor_location") - 0.04) < 1e-12
    # the literal mode must REFUSE rather than hand back <= 0
    for bad in (floor_zero, floor_neg):
        msg = _expect(ValueError,
                      lambda f=bad: resolve_min_delta_sil(
                          f, kappa=2.0, mode="floor_location"),
                      ["floor_scale"])
        assert "disables early stopping" in msg
    # absolute passes through; a zero threshold is legal there (current default)
    assert resolve_min_delta_sil(None, mode="absolute", absolute=0.0) == 0.0
    assert resolve_min_delta_sil(None, mode="absolute", absolute=0.03) == 0.03
    _expect(ValueError, lambda: resolve_min_delta_sil(None, mode="nonsense"),
            ["mode"])
    _expect(ValueError, lambda: resolve_min_delta_sil(None,
                                                      mode="floor_scale"),
            ["floor"])
    _expect(ValueError, lambda: resolve_min_delta_sil(floor_pos, kappa=0.0,
                                                      mode="floor_scale"),
            ["kappa"])
    _expect(ValueError, lambda: resolve_min_delta_sil(
        {"mu": 0.1, "sigma": 0.0}, mode="floor_scale"), ["collapsed"])
    print("      floor_scale = kappa * sigma (positive whatever mu does); "
          "floor_location refuses mu <= 0 and names the alternative; absolute "
          "passes through unchanged")


def check_floor_pipeline_fuzz():
    """Randomized invariants of the floor -> threshold pipeline.

    Ported (in spirit) from the deleted growing-patience compatibility fuzz:
    instead of fuzzing the removed counter, this fuzzes the KEPT floor pipeline
    over many random embeddings and asserts its structural invariants, which no
    single hand-built case in [E]/[F] covers.
    """
    rng = np.random.default_rng(20260729)
    n_cases = 30
    for _ in range(n_cases):
        C = int(rng.integers(2, 6))               # 2..5 classes
        per = int(rng.integers(12, 30))           # rows per class
        E = int(rng.integers(4, 20))              # embedding dimension
        N = C * per
        if rng.random() < 0.5:                    # structured half the time
            centres = rng.normal(size=(C, E)) * 3.0
            Z = np.vstack([centres[c] + 0.5 * rng.normal(size=(per, E))
                           for c in range(C)])
        else:                                     # pure noise the other half
            Z = rng.normal(size=(N, E))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        y = np.repeat(np.arange(C), per)
        R = int(rng.integers(30, 90))
        floor = silhouette_floor(Z, y, n_permutations=R, metric="cosine",
                                 seed=int(rng.integers(0, 10_000)))
        # structural invariants of the returned null summary
        assert floor["n_permutations"] == R, floor
        assert 2 <= floor["n_valid"] <= R, floor
        assert floor["n_eval"] == N and floor["n_classes"] == C, floor
        assert floor["sigma"] > 0.0 and np.isfinite(floor["sigma"]), floor
        assert np.isfinite(floor["mu"]), floor
        assert (floor["minimum"] <= floor["q05"] <= floor["q50"]
                <= floor["q95"] <= floor["maximum"]), floor
        # the kept threshold rule is exactly kappa * sigma, always positive
        for kappa in (1.0, 2.0, 3.5):
            d = resolve_min_delta_sil(floor, kappa=kappa, mode="floor_scale")
            assert abs(d - kappa * floor["sigma"]) < 1e-12, (kappa, d, floor)
            assert d > 0.0, (kappa, d)
    print("      %d random embeddings (C=2..5, E=4..19, structured & noise): "
          "null-summary invariants hold and floor_scale delta = kappa*sigma > 0"
          % n_cases)


def main():
    groups = [
        ("E", "silhouette floor statistics", check_floor_statistics),
        ("F", "threshold resolution", check_resolve_delta),
        ("G'", "floor -> threshold pipeline fuzz", check_floor_pipeline_fuzz),
    ]
    print("smoke_test_silhouette_floor.py  [label-shuffled silhouette floor]")
    failures = []
    for letter, title, fn in groups:
        try:
            fn()
        except Exception as ex:                    # noqa: BLE001
            failures.append((letter, title, ex))
            print("  [%s] %-40s FAIL" % (letter, title))
            print("      %s: %s" % (type(ex).__name__, ex))
        else:
            print("  [%s] %-40s PASS" % (letter, title))
    if failures:
        print("FAILED: %d of %d assertion group(s): %s"
              % (len(failures), len(groups),
                 ", ".join(f[0] for f in failures)))
        return 1
    print("ALL SILHOUETTE-FLOOR CHECKS PASSED (%d groups)" % len(groups))
    return 0


if __name__ == "__main__":
    sys.exit(main())
