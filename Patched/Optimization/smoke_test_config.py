"""
smoke_test_config.py

Standalone correctness checks for config.py (Stage 1). No external data, CPU only.
Self-contained: imports config (which pulls the real backbone / augmentation
configs), torch is only an indirect dependency of those modules.

Run:
    python3 smoke_test_config.py

Checks:
  1. Defaults construct; the nested configs are the REAL BackboneConfig /
     AugmentationConfig (not local copies), and the placeholder fs is present.
  2. JSON round-trip equality: ExperimentConfig == from_json(to_json(cfg)) for a
     fully non-default config (the acid test of the tuple coercion).
  3. Tuple coercion: list-valued JSON fields come back as tuples, never lists.
  4. Nested reconstruction types after round-trip.
  5. Partial JSON fills defaults for omitted sections.
  6. Validation errors fire on illegal values (both this module's dataclasses and
     the reused BackboneConfig).
  7. Soft cross-field warnings fire (patience >= max_epochs; eval-window overlap).
  8. resolved_augmentation(fs) injects fs and leaves every other field unchanged.
  9. Real file write / read: nested output dir is created and the JSON on disk is
     pure ASCII (HPC-safe artifact).
"""

import json
import os
import sys
import tempfile
import warnings
from dataclasses import replace

import config as C
from config import (
    ExperimentConfig, DataConfig, TrainConfig, SearchConfig,
    RegularizationConfig, EvalConfig, RuntimeConfig,
    BackboneConfig, AugmentationConfig,
)


def _build_nondefault():
    """A config in which every sub-config differs from its default, including
    several tuple-valued fields, so serialization is genuinely exercised."""
    aug = replace(
        AugmentationConfig(fs=50.0),
        n_positives=20, n_negatives=25,
        sigma_mag_pos=(0.02, 0.20),
        split_method="percentile_mse", percentile_q=0.4,
    )
    data = DataConfig(
        data_mode="numpy", npz_specs="burst_specs.json",
        synthetic_n_per_class=(3, 2, 1),
        window_s=150.0, train_stride_s=50.0, eval_stride_s=150.0,
        split_fractions=(0.7, 0.15, 0.15), augmentation=aug,
    )
    bb = BackboneConfig(
        depth_exponent=5, width_multiplier=2.3, block_family=1, group_width=8,
        embedding_size=12, head_fusion=True,
        head_pool_ops=("mean", "max", "std"), dropout=0.1,
    )
    tr = TrainConfig(
        margin=0.5, mining_strategy="easy_positive", lr=5e-4,
        beta1=0.95, beta2=0.9995, weight_decay=5e-4,
        max_epochs=80, patience=8, n_seeds=5, selection_primary="silhouette",
        use_scheduler=True, use_amp=True,
        windows_per_condition=6, batches_per_epoch=12,   # Stage-5 batching fields
    )
    se = SearchConfig(
        depth_exponent_range=(3, 5), width_multiplier_range=(1.5, 2.5),
        block_family_choices=(0, 1), embedding_size_range=(8, 12),
        one_minus_beta1_range=(5e-3, 5e-2), n_calls_arch=20, n_calls_train=20,
        do_refine=True, do_retune_arch=True,
    )
    rg = RegularizationConfig(
        dropout_range=(0.0, 0.25), weight_decay_range=(1e-6, 1e-3), n_calls=15,
    )
    ev = EvalConfig(kmeans_seed=7, kmeans_n_init=8, pca_components=3)
    rt = RuntimeConfig(
        seed=123, device="cuda", torch_threads=4, num_workers=8,
        out_dir="out/exp", cache_dir="cache", experiment_name="unit",
    )
    return ExperimentConfig(data=data, backbone=bb, train=tr, search=se,
                            regularization=rg, eval=ev, runtime=rt)


def check_defaults():
    cfg = ExperimentConfig()
    assert type(cfg.backbone) is BackboneConfig, type(cfg.backbone)
    assert type(cfg.data.augmentation) is AugmentationConfig, type(cfg.data.augmentation)
    assert type(cfg.train) is TrainConfig
    assert type(cfg.search) is SearchConfig
    assert cfg.data.augmentation.fs == C._PLACEHOLDER_FS
    # Stage-5 batching fields exist with their documented defaults (0 -> derive
    # n_batches = ceil(N_train / (C * windows_per_condition)) at trainer build time)
    assert cfg.train.windows_per_condition == 8, cfg.train.windows_per_condition
    assert cfg.train.batches_per_epoch == 0, cfg.train.batches_per_epoch
    print("  [1] defaults construct; nested real configs + placeholder fs + "
          "Stage-5 batching defaults OK")


def check_roundtrip_equality():
    cfg = _build_nondefault()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "cfg.json")
        cfg.to_json(p)
        cfg2 = ExperimentConfig.from_json(p)
    assert cfg == cfg2, "round-trip inequality"
    print("  [2] JSON round-trip equality OK")


def check_tuple_coercion():
    cfg = _build_nondefault()
    cfg2 = ExperimentConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    for obj, name in [
        (cfg2.backbone.head_pool_ops, "backbone.head_pool_ops"),
        (cfg2.data.split_fractions, "data.split_fractions"),
        (cfg2.data.synthetic_n_per_class, "data.synthetic_n_per_class"),
        (cfg2.data.augmentation.sigma_mag_pos, "augmentation.sigma_mag_pos"),
        (cfg2.search.block_family_choices, "search.block_family_choices"),
        (cfg2.search.width_multiplier_range, "search.width_multiplier_range"),
        (cfg2.regularization.dropout_range, "regularization.dropout_range"),
    ]:
        assert isinstance(obj, tuple), "%s is %s, expected tuple" % (name, type(obj))
        assert not isinstance(obj, list), "%s came back as list" % name
    print("  [3] tuple coercion OK (JSON lists -> tuples across all nested configs)")


def check_nested_types():
    cfg = _build_nondefault()
    cfg2 = ExperimentConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert type(cfg2.backbone) is BackboneConfig
    assert type(cfg2.data.augmentation) is AugmentationConfig
    assert type(cfg2.search) is SearchConfig
    assert type(cfg2.regularization) is RegularizationConfig
    print("  [4] nested reconstruction types OK")


def check_partial_defaults():
    cfg = ExperimentConfig.from_dict({"runtime": {"seed": 7}})
    assert cfg.runtime.seed == 7
    assert cfg.data == DataConfig()
    assert cfg.backbone == BackboneConfig()
    assert cfg.train == TrainConfig()
    # nested partial: data present but augmentation omitted -> default augmentation
    cfg2 = ExperimentConfig.from_dict({"data": {"window_s": 111.0}})
    assert cfg2.data.window_s == 111.0
    assert cfg2.data.augmentation == AugmentationConfig(fs=C._PLACEHOLDER_FS)
    print("  [5] partial JSON fills defaults (top-level and nested) OK")


def check_validation_errors():
    cases = [
        lambda: DataConfig(split_fractions=(0.6, 0.2, 0.3)),   # sums to 1.1
        lambda: DataConfig(data_mode="bogus"),
        lambda: TrainConfig(mining_strategy="bogus"),
        lambda: TrainConfig(beta1=1.0),                        # not in (0,1)
        lambda: TrainConfig(margin=0.0),
        lambda: RuntimeConfig(device="tpu"),
        lambda: RegularizationConfig(dropout_range=(0.1, 1.5)),  # >= 1
        lambda: SearchConfig(width_multiplier_range=(0.9, 3.0)),  # <= 1
        lambda: SearchConfig(one_minus_beta1_range=(0.01, 1.0)),  # high >= 1
        lambda: EvalConfig(pca_components=0),
        lambda: BackboneConfig(dropout=1.2),                   # reused-class validator
        lambda: BackboneConfig(block_family=2),                # reused-class validator
        lambda: TrainConfig(windows_per_condition=0),          # Stage-5 batching field
        lambda: TrainConfig(batches_per_epoch=-1),             # Stage-5 batching field
    ]
    for i, fn in enumerate(cases):
        raised = False
        try:
            fn()
        except (ValueError, TypeError):
            raised = True
        assert raised, "case %d did not raise" % i
    print("  [6] validation errors fire (%d cases, incl. reused configs) OK" % len(cases))


def check_soft_warnings():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        TrainConfig(max_epochs=5, patience=10)
    assert any("early stopping" in str(x.message) for x in w), "no patience warning"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        DataConfig(eval_stride_s=50.0, window_s=200.0)
    assert any("OVERLAP" in str(x.message) for x in w), "no overlap warning"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ExperimentConfig(
            train=TrainConfig(max_epochs=5, patience=10)).validate()
    assert any("early stopping cannot fire" in str(x.message) for x in w), \
        "validate() did not warn"
    print("  [7] soft cross-field warnings fire OK")


def check_resolved_augmentation():
    cfg = ExperimentConfig()
    aug = cfg.data.resolved_augmentation(137.0)
    assert aug.fs == 137.0
    base = cfg.data.augmentation
    # everything except fs must be identical
    assert replace(aug, fs=base.fs) == base
    print("  [8] resolved_augmentation injects fs, leaves other fields intact OK")


def check_file_io():
    cfg = _build_nondefault()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "sub", "nested", "cfg.json")   # non-existent dirs
        cfg.to_json(p)
        assert os.path.exists(p), "nested output dir not created"
        raw = open(p, "rb").read()
        assert all(b <= 127 for b in raw), "config JSON on disk is not pure ASCII"
        cfg2 = ExperimentConfig.from_json(p)
    assert cfg == cfg2
    print("  [9] file write / read (nested dir, ASCII JSON) OK")


def main():
    print("Running config smoke tests...")
    check_defaults()
    check_roundtrip_equality()
    check_tuple_coercion()
    check_nested_types()
    check_partial_defaults()
    check_validation_errors()
    check_soft_warnings()
    check_resolved_augmentation()
    check_file_io()
    print("ALL CONFIG SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
