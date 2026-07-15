"""
smoke_test_end_to_end.py

Stage 8: the WHOLE pipeline at toy scale on synthetic data. CPU only, headless, no
data files, no display, no cluster.

Run:
    python3 smoke_test_end_to_end.py

Checks:
  [A] --dry-run prints the budget and trains NOTHING (asserted by patching train to
      explode: it must never be called).
  [B] Budget arithmetic: each phase costs n_calls * n_seeds train() runs, plus
      n_seeds for the final fit.
  [C] PRE-FLIGHT: window feasibility. The SHIPPED DEFAULTS are infeasible (200 s
      window vs a 120 s val segment -> 0 windows), and the driver must say so BEFORE
      any expensive work, naming the concrete fix.
  [D] PRE-FLIGHT: model size. The parameter count is ~exponential in depth_exponent
      (~700x across the default range, up to ~214 M params / ~2.6 GB). The driver
      must report the corners and warn when the space reaches a size that would be
      SIGKILLed (uncatchable) rather than raising.
  [E] PRE-FLIGHT: batch rows. M = C * B_c * (1 + P + N). The augmentation expands
      every source window into 1+P+N rows, so the shipped defaults make a 2-class
      batch 976 rows, not 16 -- a 61x multiplier that windows_per_condition alone
      does not reveal.
  [F] FULL PIPELINE: phase1 -> phase2 -> regularization -> final train -> held-out
      TEST eval, end to end, emitting every artifact.
  [G] Artifacts: config_input.json, config_best.json, results.json, per-seed and
      final checkpoints, PDP figures, embedding figures -- all present and non-empty.
  [H] results.json carries TEST ARI/AMI +/- SEED std (the spread is over TRAINING
      seeds, not over K-means restarts), with one value per seed.
  [I] The final checkpoint is SELF-DESCRIBING: the model rebuilds from the EMBEDDED
      config alone, with zero hyper-parameters remembered in code.
  [J] RESUME: an interrupted final train resumes from last.pt.
  [K] NO INTERACTIVE CALL is reachable: plt.show is patched to explode and the whole
      pipeline still completes; the backend is Agg.
  [L] --skip-search fits the config AS GIVEN (no search phases in results.json).
"""

import json
import os
import sys
import tempfile
import warnings

import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt

import run_optimization as R
from run_optimization import (
    build_parser, resolve_config, estimate_budget, estimate_model_sizes,
    estimate_batch_rows, check_window_feasibility, run, main,
)
from checkpoint import rebuild_model_from_checkpoint
from config import ExperimentConfig


def _toy_cfg_json(path):
    """A config whose arch space, augmentation, and batching all FIT a small box."""
    cfg = {
        "search": {"depth_exponent_range": [3, 4],
                   "width_multiplier_range": [1.5, 2.0],
                   "embedding_size_range": [8, 12],
                   "block_family_choices": [0, 1]},
        "backbone": {"stem_width": 8},
        "data": {"augmentation": {"fs": 50.0, "n_positives": 3, "n_negatives": 3,
                                  "shift_magnitude_s": 2.0}},
        "train": {"windows_per_condition": 2, "batches_per_epoch": 3},
    }
    with open(path, "w", encoding="ascii") as fh:
        json.dump(cfg, fh)
    return path


def _toy_argv(cfg_path, out_dir, cache_dir, extra=None):
    argv = [
        "--config", cfg_path, "--data-mode", "synthetic", "--device", "cpu",
        "--num-workers", "0", "--out-dir", out_dir, "--cache-dir", cache_dir,
        "--experiment-name", "toy",
        "--window-s", "8", "--train-stride-s", "4", "--eval-stride-s", "8",
        "--synthetic-duration-s", "160",
        "--n-seeds", "2", "--max-epochs", "3",
        "--n-calls-arch", "4", "--n-calls-train", "4", "--n-calls-reg", "4",
    ]
    return argv + list(extra or [])


def _resolved(argv):
    return resolve_config(build_parser().parse_args(argv))


# --------------------------------------------------------------------------- #
def check_dry_run_trains_nothing():
    with tempfile.TemporaryDirectory() as d:
        cfg_path = _toy_cfg_json(os.path.join(d, "cfg.json"))
        argv = _toy_argv(cfg_path, os.path.join(d, "out"),
                         os.path.join(d, "cache"), ["--dry-run"])
        orig = R.train

        def boom(*a, **kw):
            raise AssertionError("--dry-run TRAINED something")

        R.train = boom
        try:
            rc = main(argv)
        finally:
            R.train = orig
        assert rc == 0, rc
        assert not os.path.exists(os.path.join(d, "out", "toy", "results.json"))
    print("  [A] --dry-run prints the budget and trains NOTHING (train() patched to "
          "explode is never reached) OK")


def check_budget_arithmetic():
    with tempfile.TemporaryDirectory() as d:
        cfg_path = _toy_cfg_json(os.path.join(d, "cfg.json"))
        cfg = _resolved(_toy_argv(cfg_path, d, d))
    b = estimate_budget(cfg)
    ns = cfg.train.n_seeds
    assert b["phase1_arch"] == cfg.search.n_calls_arch * ns, b
    assert b["phase2_train"] == cfg.search.n_calls_train * ns, b
    assert b["regularization"] == cfg.regularization.n_calls * ns, b
    assert b["final"] == ns, b
    assert b["TOTAL_train_runs"] == sum(v for k, v in b.items()
                                        if k != "TOTAL_train_runs"), b
    # --skip-search costs only the final fits
    b2 = estimate_budget(cfg, skip_search=True)
    assert b2["TOTAL_train_runs"] == ns, b2
    print("  [B] budget arithmetic OK (%d train() runs; --skip-search -> %d)"
          % (b["TOTAL_train_runs"], b2["TOTAL_train_runs"]))


def check_window_feasibility_preflight():
    """[C] The SHIPPED DEFAULTS are infeasible; the check must catch it early."""
    cfg = ExperimentConfig()                       # defaults: window_s = 200 s
    L = int(cfg.data.synthetic_duration_s * cfg.data.synthetic_fs)   # 600 s @ 50 Hz
    raised = False
    try:
        check_window_feasibility(cfg, L, cfg.data.synthetic_fs)
    except ValueError as ex:
        raised = True
        msg = str(ex)
        assert "val" in msg and "test" in msg, msg
        assert "reduce data.window_s" in msg, msg
        assert "120.0 s" in msg, msg                # the actual shortest segment
    assert raised, "the infeasible DEFAULT geometry was not caught"

    # a feasible geometry passes and reports the segments
    cfg.data.window_s = 8.0
    cfg.data.synthetic_duration_s = 160.0
    segs = check_window_feasibility(cfg, int(160.0 * 50.0), 50.0)
    assert set(segs) == {"train", "val", "test"}
    assert all(v > 0 for v in segs.values()), segs
    print("  [C] window-feasibility pre-flight OK: the SHIPPED DEFAULT (200 s window "
          "vs a 120 s val segment) is caught BEFORE any training, and names the fix")


def check_model_size_preflight():
    """[D] ~700x parameter spread across the default arch space; the big corner is a
    SIGKILL risk that Python cannot catch, so it must be reported up front."""
    cfg = ExperimentConfig()
    s = estimate_model_sizes(cfg)
    corners = {k: v for k, v in s["corners"].items() if isinstance(v, int)}
    assert len(corners) == 4, corners
    small, big = min(corners.values()), max(corners.values())
    assert big > 50 * small, (small, big)          # the explosion is real
    assert s["max_params"] == big
    assert s["max_ram_gb_weights_and_optimizer"] > 1.0, s
    print("  [D] model-size pre-flight OK: %s .. %s params across the DEFAULT arch "
          "space (%.0fx spread, worst ~%.1f GB weights+AdamW)"
          % (format(small, ","), format(big, ","), big / float(small),
             s["max_ram_gb_weights_and_optimizer"]))


def check_batch_rows_preflight():
    """[E] M = C * B_c * (1 + P + N): the augmentation multiplier."""
    cfg = ExperimentConfig()                        # defaults: P = N = 30, B_c = 8
    br = estimate_batch_rows(cfg, n_classes=2)
    assert br["rows_per_source_window"] == 1 + 30 + 30 == 61, br
    assert br["source_windows_per_batch"] == 2 * 8 == 16, br
    assert br["rows_per_batch_M"] == 2 * 8 * 61 == 976, br
    print("  [E] batch-rows pre-flight OK: the DEFAULTS make one batch %d rows, not "
          "%d -- a %dx multiplier that windows_per_condition alone hides"
          % (br["rows_per_batch_M"], br["source_windows_per_batch"],
             br["rows_per_source_window"]))


def check_full_pipeline(root):
    """[F] + [G] + [H] + [I] + [K]: the whole thing, end to end."""
    cfg_path = _toy_cfg_json(os.path.join(root, "cfg.json"))
    out_dir = os.path.join(root, "out")
    argv = _toy_argv(cfg_path, out_dir, os.path.join(root, "cache"))
    args = build_parser().parse_args(argv)
    cfg = resolve_config(args)

    # [K] any interactive call is a hard failure
    assert matplotlib.get_backend().lower() == "agg", matplotlib.get_backend()
    orig_show = plt.show

    def boom(*a, **kw):
        raise AssertionError("plt.show() reached -- the driver is not headless")

    plt.show = boom
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = run(cfg, args)
    finally:
        plt.show = orig_show
    print("  [K] no interactive call reachable OK (backend=Agg; a patched exploding "
          "plt.show is never hit across the WHOLE pipeline)")

    exp = os.path.join(out_dir, "toy")
    # [G] artifacts
    must_exist = [
        "config_input.json", "config_best.json", "results.json",
        "figures/pdp_phase1_arch.png", "figures/pdp_phase2_train.png",
        "figures/pdp_regularization.png",
        "figures/embedding_test_seed_0.png", "figures/embedding_test_seed_1.png",
        "checkpoints/final_seed_0.pt", "checkpoints/final_seed_1.pt",
        "checkpoints/seed_0/last.pt", "checkpoints/seed_1/last.pt",
    ]
    for rel in must_exist:
        p = os.path.join(exp, rel)
        assert os.path.exists(p), "missing artifact: %s" % rel
        assert os.path.getsize(p) > 0, "empty artifact: %s" % rel
    print("  [G] artifacts OK: %d files (configs, results, 3 PDPs, 2 embedding "
          "figures, per-seed + final checkpoints)" % len(must_exist))

    # [F] every phase ran
    for k in ("phase1_arch", "phase2_train", "regularization"):
        assert k in results, k
        assert len(results[k]["trial_log"]) == 4, (k, len(results[k]["trial_log"]))
    print("  [F] full pipeline OK: phase1 (obj %+.4f) -> phase2 (obj %+.4f) -> "
          "regularization (obj %+.4f) -> final train -> held-out TEST eval"
          % (results["phase1_arch"]["best_objective"],
             results["phase2_train"]["best_objective"],
             results["regularization"]["best_objective"]))

    # [H] TEST metrics +/- SEED std, one value per seed
    t = results["test"]
    assert t["n_seeds"] == 2, t["n_seeds"]
    for key in ("ari", "ami", "silhouette"):
        assert len(t[key]["values"]) == 2, (key, t[key])
        assert np.isfinite(t[key]["mean"]) and np.isfinite(t[key]["std"]), (key, t[key])
        # the std must be the POPULATION std over the seed values
        assert abs(t[key]["std"] - float(np.std(t[key]["values"]))) < 1e-9, key
    assert len(t["per_seed"]) == 2
    for ps in t["per_seed"]:
        assert "test_ari" in ps and "best_val_ari" in ps and "figure" in ps
    print("  [H] TEST ARI %.4f +/- %.4f (mean +/- SEED std over %d training seeds, "
          "not over K-means restarts) OK"
          % (t["ari"]["mean"], t["ari"]["std"], t["n_seeds"]))

    # [I] the final checkpoint rebuilds the model from its EMBEDDED config alone
    ck = os.path.join(exp, "checkpoints", "final_seed_0.pt")
    model, ckpt = rebuild_model_from_checkpoint(ck)
    emb = int(ckpt["config"]["backbone"]["embedding_size"])
    with torch.no_grad():
        z = model(torch.randn(2, int(results["window_length"])))
    assert z.shape == (2, emb), (z.shape, emb)
    # and the embedded config IS the winning config
    assert ckpt["config"]["backbone"]["depth_exponent"] == \
        results["config_best"]["backbone"]["depth_exponent"]
    print("  [I] final checkpoint self-describing OK: model rebuilt from the EMBEDDED "
          "config alone -> (2, %d) embeddings; zero HPs remembered in code" % emb)
    return exp, cfg_path, out_dir, root


def check_resume(cfg_path, out_dir, root):
    """[J] An interrupted final train resumes from last.pt."""
    # a fresh experiment, trained for only 2 epochs
    argv = _toy_argv(cfg_path, out_dir, os.path.join(root, "cache"),
                     ["--skip-search", "--n-seeds", "1", "--max-epochs", "2",
                      "--experiment-name", "resume_test"])
    args = build_parser().parse_args(argv)
    cfg = resolve_config(args)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r1 = run(cfg, args)
    assert r1["test"]["per_seed"][0]["epochs_run"] == 2, r1["test"]["per_seed"][0]

    last = os.path.join(out_dir, "resume_test", "checkpoints", "seed_0", "last.pt")
    assert os.path.exists(last), "no last.pt to resume from"

    # now RESUME with a higher ceiling: it must continue, not restart
    argv2 = _toy_argv(cfg_path, out_dir, os.path.join(root, "cache"),
                      ["--skip-search", "--n-seeds", "1", "--max-epochs", "4",
                       "--experiment-name", "resume_test", "--resume"])
    args2 = build_parser().parse_args(argv2)
    cfg2 = resolve_config(args2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r2 = run(cfg2, args2)
    # the resumed history spans epochs 1..4 (2 carried through the checkpoint + 2 new)
    assert r2["test"]["per_seed"][0]["epochs_run"] == 4, \
        ("resume did not continue from epoch 2", r2["test"]["per_seed"][0])
    print("  [J] resume OK: 2 epochs -> interrupted -> resumed to 4 (history carried "
          "through last.pt, not restarted)")


def check_skip_search(cfg_path, out_dir, root):
    """[L] --skip-search fits the config AS GIVEN: no search phases at all."""
    argv = _toy_argv(cfg_path, out_dir, os.path.join(root, "cache"),
                     ["--skip-search", "--n-seeds", "1", "--max-epochs", "2",
                      "--experiment-name", "skip_test"])
    args = build_parser().parse_args(argv)
    cfg = resolve_config(args)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run(cfg, args)
    for k in ("phase1_arch", "phase2_train", "regularization", "retune_arch"):
        assert k not in res, "--skip-search still ran %s" % k
    assert res["budget"]["TOTAL_train_runs"] == 1, res["budget"]
    assert np.isfinite(res["test"]["ari"]["mean"])
    print("  [L] --skip-search OK: no search phases, 1 train() run, TEST ARI %.4f"
          % res["test"]["ari"]["mean"])


def main_():
    print("Running end-to-end smoke tests...")
    check_dry_run_trains_nothing()
    check_budget_arithmetic()
    check_window_feasibility_preflight()
    check_model_size_preflight()
    check_batch_rows_preflight()
    with tempfile.TemporaryDirectory() as root:
        exp, cfg_path, out_dir, root = check_full_pipeline(root)
        check_resume(cfg_path, out_dir, root)
        check_skip_search(cfg_path, out_dir, root)
    print("ALL END-TO-END SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main_())
