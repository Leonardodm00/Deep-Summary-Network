"""
smoke_test_evaluate.py

Standalone correctness checks for evaluate.py (Stage 6). CPU only, synthetic data
only, headless. No data files, no display.

Run:
    python3 smoke_test_evaluate.py

Checks (C in {2, 3, 4} where noted):
  [A] evaluate() returns DISTINCT embedding rows: N == len(dataset.index) and the
      window provenance (trace_idx, start) is unique -- the legacy
      resample-with-replacement duplication cannot recur.
  [B] METRIC / PLOT LABEL IDENTITY (the headline consistency fix). The colours in
      the WRITTEN FIGURE are recovered from the rendered scatter collections and
      mapped back to cluster ids; the recovered per-point cluster assignment must
      equal results["labels_pred"] EXACTLY -- the same array that produced the
      reported ARI/AMI. This is asserted against the artefact, not by trusting that
      the right variable was passed.
  [C] evaluate()'s scores are reproducible and agree with a direct
      clustering_metrics call on the same (Z, y) -- no hidden re-clustering.
  [D] The figure FILE is written (non-empty PNG), the parent directory is created,
      and no figure is left open (matplotlib figure count returns to 0).
  [E] HEADLESS: the backend is Agg and plt.show is NEVER called (asserted by
      patching plt.show to raise). Also, evaluate.py's source contains no
      "plt.show" and no "Visible" flag (the legacy NameError path is gone).
  [F] plot_embedding does NOT re-cluster: monkeypatching sklearn KMeans inside the
      evaluate module to raise proves the figure is drawn WITHOUT fitting any
      K-means (the legacy plot fitted a second, unseeded one on PCA data).
  [G] K = C is honoured from the caller's full label set even when the evaluated
      split is missing a phenotype (same invariant the trainer guards).
  [H] A TRAINED model scores better than an untrained one on separable synthetic
      phenotypes (ARI rises) -- the evaluator is measuring something real.
  [I] Held-out TEST split: evaluating SplitBundle.test uses windows DISJOINT from
      train and val (verified against the bundle's coverage), i.e. leakage-free.
  [J] evaluate_and_plot returns the figure path and the same scores as evaluate().
  [K] Degenerate inputs do not crash: a single-class split yields nan ARI/AMI (not
      an exception) and still writes a figure.
"""

import glob
import os
import sys
import tempfile
import warnings
from dataclasses import replace

import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt

from config import (
    ExperimentConfig, DataConfig, TrainConfig, EvalConfig, RuntimeConfig,
    BackboneConfig, AugmentationConfig,
)
from backbone import build_backbone
from preprocessing_cache import cache_traces, load_cached_traces
from data_splits import (
    MultiClassSyntheticProvider, make_synthetic_specs, make_time_segment_splits,
)
from metrics import clustering_metrics
from train import train
import evaluate as ev
from evaluate import evaluate, plot_embedding, evaluate_and_plot

_DURATION_S = 160.0
_FS = 50.0
_WINDOW_S = 8.0
_TRAIN_STRIDE_S = 2.0
_EVAL_STRIDE_S = 8.0
_FRACTIONS = (0.6, 0.2, 0.2)
_EMB = 8


def _make_cfg(C, max_epochs=2, seed=0):
    aug = replace(AugmentationConfig(fs=_FS),
                  n_positives=3, n_negatives=3, shift_magnitude_s=2.0)
    data = DataConfig(
        data_mode="synthetic", synthetic_n_per_class=tuple([3] * C),
        synthetic_duration_s=_DURATION_S, synthetic_fs=_FS,
        window_s=_WINDOW_S, train_stride_s=_TRAIN_STRIDE_S,
        eval_stride_s=_EVAL_STRIDE_S, split_fractions=_FRACTIONS,
        augmentation=aug,
    )
    bb = BackboneConfig(depth_exponent=3, width_multiplier=1.5, stem_width=8,
                        embedding_size=_EMB)
    tr = TrainConfig(margin=0.3, lr=3e-3, max_epochs=max_epochs,
                     patience=max_epochs, windows_per_condition=4,
                     batches_per_epoch=6, n_seeds=1)
    rt = RuntimeConfig(seed=seed, device="cpu", num_workers=0, torch_threads=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return ExperimentConfig(data=data, backbone=bb, train=tr,
                                eval=EvalConfig(), runtime=rt)


def _make_splits(C, cfg, cache_dir):
    provider = MultiClassSyntheticProvider(
        n_classes=C, duration_s=_DURATION_S, fs=_FS, seed=0)
    specs = make_synthetic_specs(cfg.data.synthetic_n_per_class)
    cache_traces(specs, provider, cache_dir)
    traces, conditions, fs = load_cached_traces(cache_dir)
    return make_time_segment_splits(traces, conditions, fs, cfg.data, base_seed=0)


def _fresh_model(cfg, seed=0):
    torch.manual_seed(seed)
    return build_backbone(cfg.backbone)


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(*a, **kw)


# --------------------------------------------------------------------------- #
def check_distinct_rows(cache_dir):
    C = 2
    cfg = _make_cfg(C)
    splits = _make_splits(C, cfg, cache_dir)
    model = _fresh_model(cfg)
    res = _quiet(evaluate, model, splits.test, "cpu", seed=0, n_clusters=C,
                 eval_cfg=cfg.eval)
    N = len(splits.test.index)
    assert res["n_windows"] == N == res["Z"].shape[0] == res["y"].shape[0]
    prov = [(ti, s) for (ti, s, _c) in splits.test.index]
    assert len(set(prov)) == N, "duplicated window provenance in the TEST split"
    assert res["Z"].shape[1] == _EMB
    assert res["labels_pred"].shape == (N,)
    print("  [A] distinct rows OK: N=%d embeddings, %d unique (trace,start) pairs "
          "(no resample-with-replacement)" % (N, len(set(prov))))


def _recover_point_clusters_from_figure(Z, y, labels_pred, cfg):
    """Read the COLOURS back out of the RENDERED scatter collections and map each
    point to the cluster id its colour encodes.

    plot_embedding closes its figure, so we intercept plt.subplots to capture the
    live Axes as the module draws, then inspect ax.collections: for each scatter
    call we recover (a) the facecolour -> which cluster it encodes, and (b) the
    plotted offsets -> which embedding rows they are (matched against the PCA
    coordinates). This asserts the identity against the ARTEFACT, rather than
    trusting that the right variable was passed in.

    Returns an (N,) array of recovered cluster ids in the original row order.
    """
    from sklearn.decomposition import PCA

    N = Z.shape[0]
    P_eff = int(min(int(cfg.eval.pca_components), N, Z.shape[1]))
    Z_pca = PCA(n_components=P_eff).fit_transform(
        np.ascontiguousarray(np.asarray(Z, dtype=np.float64)))
    if Z_pca.shape[1] == 1:
        Z_pca = np.hstack([Z_pca, np.zeros((N, 1))])

    captured = {}
    orig_subplots = ev.plt.subplots

    def capturing_subplots(*a, **kw):
        fig, ax = orig_subplots(*a, **kw)
        captured["fig"] = fig
        captured["ax"] = ax
        return fig, ax

    ev.plt.subplots = capturing_subplots
    try:
        with tempfile.TemporaryDirectory() as d:
            plot_embedding(Z, y, labels_pred, os.path.join(d, "probe2.png"),
                           pca_components=int(cfg.eval.pca_components))
    finally:
        ev.plt.subplots = orig_subplots

    ax = captured["ax"]
    # colour -> cluster id, exactly as the module builds it
    pred_ids = np.unique(np.asarray(labels_pred))
    cmap = plt.get_cmap("tab10" if pred_ids.shape[0] <= 10 else "tab20")
    colour_of = {int(p): np.asarray(cmap(i % cmap.N), dtype=float)
                 for i, p in enumerate(pred_ids)}

    recovered = np.full(N, -1, dtype=np.int64)
    for coll in ax.collections:
        offs = np.asarray(coll.get_offsets(), dtype=float)      # (n_i, 2)
        fc = np.asarray(coll.get_facecolors(), dtype=float)
        if offs.shape[0] == 0 or fc.shape[0] == 0:
            continue
        rgba = fc[0]                                             # one colour per call
        # which cluster does this colour encode?
        cid = None
        for p, c in colour_of.items():
            if np.allclose(rgba[:3], c[:3], atol=1e-6):
                cid = int(p)
                break
        assert cid is not None, "a plotted colour does not match any cluster colour"
        # map each plotted point back to its row by matching PCA coordinates
        for pt in offs:
            d2 = ((Z_pca[:, 0] - pt[0]) ** 2 + (Z_pca[:, 1] - pt[1]) ** 2)
            i = int(np.argmin(d2))
            assert d2[i] < 1e-12, "plotted point does not match any embedding row"
            recovered[i] = cid
    plt.close(captured["fig"])
    assert np.all(recovered >= 0), "some rows were never plotted"
    return recovered


def check_metric_plot_identity():
    """[B] The colours in the RENDERED figure must encode exactly the labels_pred
    that produced the reported ARI/AMI. Asserted against the artefact."""
    for C in (2, 3, 4):
        with tempfile.TemporaryDirectory() as d:
            cfg = _make_cfg(C)
            splits = _make_splits(C, cfg, d)
            model = _fresh_model(cfg)
            out = os.path.join(d, "figs", "emb.png")
            res = _quiet(evaluate_and_plot, model, splits.test, "cpu", out,
                         seed=cfg.eval.kmeans_seed, n_clusters=C,
                         eval_cfg=cfg.eval)
            recovered = _recover_point_clusters_from_figure(
                res["Z"], res["y"], res["labels_pred"], cfg)
            assert np.array_equal(recovered, res["labels_pred"]), \
                ("the figure's colours do NOT encode labels_pred (metric/plot "
                 "mismatch -- the legacy bug!)",
                 recovered[:12], res["labels_pred"][:12])
        print("  [B] C=%d metric/plot label IDENTITY OK: colours recovered from "
              "the rendered figure == labels_pred behind the reported ARI/AMI"
              % C)


def check_scores_consistent(cache_dir):
    C = 3
    cfg = _make_cfg(C)
    splits = _make_splits(C, cfg, cache_dir)
    model = _fresh_model(cfg)
    r1 = _quiet(evaluate, model, splits.test, "cpu", seed=0, n_clusters=C,
                eval_cfg=cfg.eval)
    r2 = _quiet(evaluate, model, splits.test, "cpu", seed=0, n_clusters=C,
                eval_cfg=cfg.eval)
    # reproducible
    assert r1["ari"] == r2["ari"] and r1["ami"] == r2["ami"]
    assert np.array_equal(r1["labels_pred"], r2["labels_pred"])
    assert np.array_equal(r1["Z"], r2["Z"])
    # and identical to a DIRECT clustering_metrics call on the same (Z, y):
    # evaluate adds no scoring mathematics of its own
    m = _quiet(clustering_metrics, r1["Z"], r1["y"], seed=0, n_clusters=C,
               n_init=cfg.eval.kmeans_n_init,
               silhouette_metric=cfg.eval.silhouette_metric)
    assert abs(m["ari"] - r1["ari"]) < 1e-12
    assert abs(m["ami"] - r1["ami"]) < 1e-12
    assert np.array_equal(m["labels_pred"], r1["labels_pred"])
    print("  [C] scores reproducible + identical to a direct clustering_metrics "
          "call (no hidden re-clustering) OK")


def check_figure_written(cache_dir):
    C = 2
    cfg = _make_cfg(C)
    splits = _make_splits(C, cfg, cache_dir)
    model = _fresh_model(cfg)
    nested = os.path.join(cache_dir, "a", "b", "c")     # non-existent dirs
    out = os.path.join(nested, "embedding.png")
    before = len(plt.get_fignums())
    res = _quiet(evaluate_and_plot, model, splits.test, "cpu", out, seed=0,
                 n_clusters=C, eval_cfg=cfg.eval)
    after = len(plt.get_fignums())
    assert os.path.exists(out), "figure not written"
    assert os.path.getsize(out) > 1000, os.path.getsize(out)
    assert res["figure"] == out
    assert after == before, \
        "figure left open (%d -> %d): long HPO runs would leak memory" % (before, after)
    print("  [D] figure written OK (%d bytes, nested dir created, no figure left "
          "open)" % os.path.getsize(out))


def check_headless(cache_dir):
    """[E] Headless by construction.

    Note on method: a naive grep of the source for "plt.show()" is the WRONG
    instrument -- it cannot tell CODE from PROSE, and evaluate.py's docstrings
    legitimately mention plt.show() and the Visible flag while documenting that
    both were REMOVED. We therefore parse the module's AST and assert there are no
    .show() CALL sites and no `Visible` identifiers in executable code, which is a
    strictly stronger claim than the grep, and then additionally patch plt.show to
    raise and confirm it is never reached at runtime.
    """
    import ast

    assert matplotlib.get_backend().lower() == "agg", matplotlib.get_backend()

    tree = ast.parse(open(ev.__file__, "r", encoding="ascii").read())
    show_calls, visible_names = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == "show":
                show_calls.append(node.lineno)
        if isinstance(node, ast.Name) and node.id == "Visible":
            visible_names.append(node.lineno)
        if isinstance(node, ast.arg) and node.arg == "Visible":
            visible_names.append(node.lineno)
    assert not show_calls, \
        "evaluate.py has .show() CALL sites at lines %r" % (show_calls,)
    assert not visible_names, \
        "a Visible identifier survives in evaluate.py code at lines %r" % (visible_names,)

    C = 2
    cfg = _make_cfg(C)
    splits = _make_splits(C, cfg, cache_dir)
    model = _fresh_model(cfg)

    # patch plt.show to EXPLODE: any interactive call is now a hard failure
    orig_show = ev.plt.show

    def boom(*a, **kw):
        raise AssertionError("plt.show() was called -- evaluate.py is not headless")

    ev.plt.show = boom
    try:
        out = os.path.join(cache_dir, "headless.png")
        _quiet(evaluate_and_plot, model, splits.test, "cpu", out, seed=0,
               n_clusters=C, eval_cfg=cfg.eval)
        assert os.path.exists(out)
    finally:
        ev.plt.show = orig_show
    print("  [E] headless OK: backend=Agg; AST shows ZERO .show() call sites and "
          "ZERO Visible identifiers in code; a patched exploding plt.show is never "
          "reached at runtime")


def check_plot_does_not_recluster(cache_dir):
    """[F] The legacy plot fitted a SECOND, unseeded K-means on PCA data. Prove the
    new plot fits NONE: make KMeans raise inside the evaluate module and draw."""
    C = 3
    cfg = _make_cfg(C)
    splits = _make_splits(C, cfg, cache_dir)
    model = _fresh_model(cfg)
    res = _quiet(evaluate, model, splits.test, "cpu", seed=0, n_clusters=C,
                 eval_cfg=cfg.eval)          # scoring may cluster; drawing must not

    import sklearn.cluster as skc
    orig_kmeans = skc.KMeans

    class ExplodingKMeans:
        def __init__(self, *a, **kw):
            raise AssertionError(
                "plot_embedding fitted a K-means -- it must NOT re-cluster "
                "(the legacy metric/plot mismatch)")

    skc.KMeans = ExplodingKMeans
    try:
        out = os.path.join(cache_dir, "norecluster.png")
        plot_embedding(res["Z"], res["y"], res["labels_pred"], out,
                       pca_components=cfg.eval.pca_components)
        assert os.path.exists(out)
    finally:
        skc.KMeans = orig_kmeans
    print("  [F] plot does NOT re-cluster OK (drawing succeeds with KMeans "
          "patched to raise; PCA is display-only)")


def check_k_equals_C_when_class_missing(cache_dir):
    """[G] K = C from the caller's FULL label set, even if the evaluated split is
    missing a phenotype (the same invariant train() guards)."""
    C = 3
    cfg = _make_cfg(C)
    splits = _make_splits(C, cfg, cache_dir)
    test = splits.test
    keep = [i for i, (_t, _s, c) in enumerate(test.index) if int(c) != 2]
    assert len(keep) < len(test.index)
    test.index = [test.index[i] for i in keep]
    test.conditions_per_item = np.asarray(
        [c for (_t, _s, c) in test.index], dtype=int)
    assert set(int(v) for v in np.unique(test.conditions_per_item)) == {0, 1}

    model = _fresh_model(cfg)
    res = _quiet(evaluate, model, test, "cpu", seed=0, n_clusters=C,   # K = C = 3
                 eval_cfg=cfg.eval)
    assert res["n_clusters"] == 3, res["n_clusters"]
    assert len(np.unique(res["labels_pred"])) <= 3
    # and the fallback (n_clusters=None) would have used only the 2 present classes
    res2 = _quiet(evaluate, model, test, "cpu", seed=0, n_clusters=None,
                  eval_cfg=cfg.eval)
    assert res2["n_clusters"] == 2, res2["n_clusters"]
    print("  [G] K=C from the caller's full label set OK (split missing a "
          "phenotype -> K=3 when told C=3; the None fallback would give K=2)")


def check_trained_beats_untrained(cache_dir):
    """[H] The evaluator measures something real: a trained model separates the
    synthetic phenotypes better than an untrained one."""
    C = 2
    cfg = _make_cfg(C, max_epochs=14)
    cfg.train.batches_per_epoch = 8
    splits = _make_splits(C, cfg, cache_dir)

    untrained = _fresh_model(cfg, seed=0)
    r_before = _quiet(evaluate, untrained, splits.test, "cpu",
                      seed=cfg.eval.kmeans_seed, n_clusters=C, eval_cfg=cfg.eval)

    trained, _hist = _quiet(train, cfg, splits.train, splits.val, "cpu", seed=0)
    r_after = _quiet(evaluate, trained, splits.test, "cpu",
                     seed=cfg.eval.kmeans_seed, n_clusters=C, eval_cfg=cfg.eval)

    assert r_after["ari"] > r_before["ari"], (r_before["ari"], r_after["ari"])
    assert r_after["ari"] > 0.5, r_after["ari"]
    print("  [H] trained > untrained on the HELD-OUT TEST split OK: ARI %.3f "
          "(untrained) -> %.3f (trained)" % (r_before["ari"], r_after["ari"]))
    return splits, trained, cfg


def check_test_split_is_leakage_free(cache_dir):
    """[I] The TEST windows evaluated are DISJOINT from every train and val window
    (verified against the bundle's own coverage, in ORIGINAL trace coordinates)."""
    C = 2
    cfg = _make_cfg(C)
    splits = _make_splits(C, cfg, cache_dir)
    cov = splits.coverage

    def overlap(a, b):
        return a[0] < b[1] and b[0] < a[1]

    n_checked = 0
    for (ti, s, e, _c) in cov["test"]:
        for other in ("train", "val"):
            for (tj, s2, e2, _c2) in cov[other]:
                if tj != ti:
                    continue
                assert not overlap((s, e), (s2, e2)), \
                    "TEST window %r overlaps a %s window %r" % ((s, e), other, (s2, e2))
                n_checked += 1
    # and the evaluated dataset is exactly the test coverage
    assert len(splits.test.index) == len(cov["test"])
    print("  [I] TEST split leakage-free OK (%d test windows, %d cross-split "
          "interval comparisons, zero overlaps)"
          % (len(cov["test"]), n_checked))


def check_evaluate_and_plot_agrees(cache_dir):
    C = 2
    cfg = _make_cfg(C)
    splits = _make_splits(C, cfg, cache_dir)
    model = _fresh_model(cfg)
    r1 = _quiet(evaluate, model, splits.test, "cpu", seed=0, n_clusters=C,
                eval_cfg=cfg.eval)
    out = os.path.join(cache_dir, "combined.png")
    r2 = _quiet(evaluate_and_plot, model, splits.test, "cpu", out, seed=0,
                n_clusters=C, eval_cfg=cfg.eval)
    assert r2["figure"] == out and os.path.exists(out)
    for k in ("ari", "ami", "silhouette", "n_clusters", "n_windows"):
        assert r1[k] == r2[k] or (np.isnan(r1[k]) and np.isnan(r2[k])), k
    assert np.array_equal(r1["labels_pred"], r2["labels_pred"])
    print("  [J] evaluate_and_plot agrees with evaluate + writes the figure OK")


def check_degenerate_single_class(cache_dir):
    """[K] A split with a single phenotype -> nan ARI/AMI (warned, not raised), and
    the figure is still written."""
    C = 2
    cfg = _make_cfg(C)
    splits = _make_splits(C, cfg, cache_dir)
    test = splits.test
    keep = [i for i, (_t, _s, c) in enumerate(test.index) if int(c) == 0]
    test.index = [test.index[i] for i in keep]
    test.conditions_per_item = np.asarray([0] * len(keep), dtype=int)

    model = _fresh_model(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = os.path.join(cache_dir, "degenerate.png")
        res = evaluate_and_plot(model, test, "cpu", out, seed=0, n_clusters=1,
                                eval_cfg=cfg.eval)
    assert np.isnan(res["ari"]) and np.isnan(res["ami"]), (res["ari"], res["ami"])
    assert os.path.exists(out)
    print("  [K] degenerate single-class split OK: ARI/AMI = nan (no crash), "
          "figure still written")


def _colours_by_cluster(Z, y, labels_pred, pca_components=2):
    """Draw and read back {cluster_id -> RGB} from the rendered collections."""
    captured = {}
    orig_subplots = ev.plt.subplots

    def capturing_subplots(*a, **kw):
        fig, ax = orig_subplots(*a, **kw)
        captured["fig"], captured["ax"] = fig, ax
        return fig, ax

    ev.plt.subplots = capturing_subplots
    try:
        with tempfile.TemporaryDirectory() as d:
            plot_embedding(Z, y, labels_pred, os.path.join(d, "p.png"),
                           pca_components=pca_components)
    finally:
        ev.plt.subplots = orig_subplots

    ax = captured["ax"]
    out = {}
    for coll, (t, p) in zip(ax.collections, _cells(y, labels_pred)):
        fc = np.asarray(coll.get_facecolors(), dtype=float)
        if fc.shape[0]:
            out.setdefault(int(p), tuple(np.round(fc[0][:3], 6)))
    plt.close(captured["fig"])
    return out


def _cells(y, labels_pred):
    """The (true, pred) cells in the SAME order plot_embedding draws them."""
    cells = []
    for t in np.unique(y):
        for p in np.unique(labels_pred):
            if np.any((y == t) & (labels_pred == p)):
                cells.append((int(t), int(p)))
    return cells


def check_colour_keyed_on_id():
    """[L] The colour of cluster k (and the marker of phenotype k) must be a
    function of k ALONE, not of which other ids happen to be present.

    Why this matters: K-means can leave a cluster empty. If the colour were keyed
    on the id's POSITION among the present ids, cluster 2 would render in one colour
    when cluster 1 is empty and a DIFFERENT colour when it is not -- so the same
    cluster would change colour between the validation and test figures, and across
    HPO trials, silently destroying visual comparability.
    """
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((40, 6))
    y = np.array([0] * 20 + [1] * 20)

    # (i) all three clusters present
    pred_full = np.array([0] * 13 + [1] * 13 + [2] * 14)
    cols_full = _colours_by_cluster(Z, y, pred_full)

    # (ii) cluster 1 EMPTY -> present ids are [0, 2]; a positional colour map would
    #      hand cluster 2 the colour that cluster 1 had.
    pred_gap = np.where(pred_full == 1, 0, pred_full)      # ids present: {0, 2}
    cols_gap = _colours_by_cluster(Z, y, pred_gap)

    assert set(cols_gap.keys()) == {0, 2}, cols_gap.keys()
    for cid in (0, 2):
        assert cols_full[cid] == cols_gap[cid], \
            ("cluster %d changed colour when cluster 1 became empty -> the colour "
             "map is POSITIONAL, not id-keyed" % cid, cols_full[cid], cols_gap[cid])
    assert cols_full[0] != cols_full[2], "distinct clusters share a colour"
    print("  [L] colour/marker keyed on the ID OK: cluster colours are stable when "
          "another cluster is empty (comparable across figures and HPO trials)")


def main():
    print("Running evaluate smoke tests...")
    with tempfile.TemporaryDirectory() as d:
        check_distinct_rows(d)
    check_metric_plot_identity()
    with tempfile.TemporaryDirectory() as d:
        check_scores_consistent(d)
    with tempfile.TemporaryDirectory() as d:
        check_figure_written(d)
    with tempfile.TemporaryDirectory() as d:
        check_headless(d)
    with tempfile.TemporaryDirectory() as d:
        check_plot_does_not_recluster(d)
    with tempfile.TemporaryDirectory() as d:
        check_k_equals_C_when_class_missing(d)
    with tempfile.TemporaryDirectory() as d:
        check_trained_beats_untrained(d)
    with tempfile.TemporaryDirectory() as d:
        check_test_split_is_leakage_free(d)
    with tempfile.TemporaryDirectory() as d:
        check_evaluate_and_plot_agrees(d)
    with tempfile.TemporaryDirectory() as d:
        check_degenerate_single_class(d)
    check_colour_keyed_on_id()
    print("ALL EVALUATE SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
