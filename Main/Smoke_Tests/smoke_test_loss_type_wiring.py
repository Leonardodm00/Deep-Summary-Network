"""
smoke_test_loss_type_wiring.py
==============================

Wiring checks for the loss_type option: config fields, the CONDITIONAL phase-2
and joint search spaces, the point -> config writer, the winner reader, and the
trainer's loss dispatch.

Run:
    python3 smoke_test_loss_type_wiring.py

Exit code 0 and "ALL CHECKS PASSED" mean the wiring is consistent.

The thing this file exists to catch
-----------------------------------
The phase-2 space is now a different DIMENSION and a different ORDER depending
on train.loss_type. Coordinates are matched to names by POSITION
(dict(zip(names, point))), so a space and a name list that disagree would not
raise -- it would silently write the learning rate into the margin. Every check
below that compares a space against train_names() is guarding that failure.

Backwards compatibility is a first-class requirement here: loss_type defaults
to "triplet" and must reproduce the pre-existing 5-dimensional margin space
exactly, so every archived config keeps running unchanged.

HPC note (hpc-python-compat): pure ASCII.
"""

import math
import sys
import traceback
import warnings

import torch

import search as S
from config import ExperimentConfig, SearchConfig, TrainConfig


def base_cfg(**train_kw):
    cfg = ExperimentConfig()
    cfg.train = TrainConfig(**train_kw)
    return cfg


def names_of(space):
    return tuple(d.name for d in space)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def check_defaults_are_backwards_compatible():
    t = TrainConfig()
    assert t.loss_type == "triplet", "default loss_type changed: %r" % t.loss_type
    assert t.margin == 0.3 and t.swap is True
    s = SearchConfig()
    assert s.margin_range == (0.1, 1.0), "margin_range default changed"
    assert t.strict_semihard is False, \
        "strict_semihard must default False: True is incompatible with the " \
        "default 'hard' miner"
    assert t.mining_strategy == "hard", \
        "mining_strategy default is not the collapse-seeking 'hard'"
    # the LOW end of the angular range is the collapse-forcing end: the implied
    # silhouette floor is 1 - 4 sin^2(alpha), so 2 deg demands S >= 0.995
    assert s.angular_alpha_deg_range == (2.0, 20.0), \
        "angular range is %r; the low end is the collapse knob" \
        % (s.angular_alpha_deg_range,)
    floor_lo = 1.0 - 4.0 * math.sin(math.radians(s.angular_alpha_deg_range[0])) ** 2
    floor_hi = 1.0 - 4.0 * math.sin(math.radians(s.angular_alpha_deg_range[1])) ** 2
    assert floor_lo > 0.99, "low-alpha end no longer forces collapse (%.4f)" % floor_lo
    assert floor_hi > 0.0, "high-alpha end is vacuous (%.4f)" % floor_hi
    assert s.lambda_sep_range == (1e-3, 1.0)
    return "defaults unchanged; alpha range [%.0f, %.0f] deg spans silhouette " \
        "floors %.3f down to %.3f" % (s.angular_alpha_deg_range[0],
                                      s.angular_alpha_deg_range[1],
                                      floor_lo, floor_hi)


def check_config_validation():
    for kw, frag in (
            ({"loss_type": "nope"}, "loss_type"),
            ({"loss_type": "joint", "angular_alpha_deg": 0.0}, "angular"),
            ({"loss_type": "joint", "angular_alpha_deg": 90.0}, "angular"),
            ({"loss_type": "joint_sep", "lambda_sep": -1.0}, "lambda_sep"),
            ({"loss_type": "joint_sep", "sep_gate_momentum": 1.5}, "momentum"),
            ({"loss_type": "joint_sep", "sep_gate_min_batches": 0}, "min_batches"),
            ({"loss_type": "joint_sep", "sep_gate_threshold": 3.0}, "threshold"),
    ):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                TrainConfig(**kw)
        except ValueError as ex:
            assert frag in str(ex), "wrong error for %r: %s" % (kw, ex)
        else:
            raise AssertionError("TrainConfig(%r) did not raise" % (kw,))
    try:
        SearchConfig(lambda_sep_range=(1.0, 0.1)).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("inverted lambda_sep_range did not raise")
    return "7 invalid TrainConfigs and 1 invalid range all rejected"


def check_inert_lambda_sep_warns_but_defaults_are_silent():
    """An inert NON-DEFAULT lambda_sep must warn. The DEFAULT config must be
    completely silent: loss_type='triplet' is the default, so a warning there
    would fire on every archived run and train the reader to ignore warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TrainConfig(loss_type="triplet", lambda_sep=0.5)
    msgs = " ".join(str(w.message) for w in caught)
    assert "lambda_sep" in msgs, "no warning for an inert lambda_sep"

    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        TrainConfig()
        TrainConfig(loss_type="joint")
        TrainConfig(loss_type="joint_sep")
    assert not caught2, \
        "default configs are not silent: %r" % [str(w.message) for w in caught2]
    return "inert lambda_sep warns; all three default configs are silent"


# --------------------------------------------------------------------------- #
# spaces
# --------------------------------------------------------------------------- #
def check_triplet_space_is_the_legacy_space():
    cfg = base_cfg()
    sp = S.train_space(cfg.search, cfg.train)
    got = names_of(sp)
    want = ("margin", "lr", "one_minus_beta1", "one_minus_beta2", "weight_decay")
    assert got == want, "phase-2 space changed for loss_type='triplet': %r" % (got,)
    assert got == S.train_names(cfg.train), "space and train_names disagree"
    assert names_of(S.train_space(cfg.search)) == want, \
        "train_space(search_cfg) alone no longer reproduces the legacy space"
    assert sp[0].prior != "log-uniform", "margin is no longer a plain Real"
    return "loss_type='triplet' reproduces the legacy 5-dim space exactly"


def check_joint_space_swaps_margin_for_alpha():
    cfg = base_cfg(loss_type="joint")
    got = names_of(S.train_space(cfg.search, cfg.train))
    assert got[0] == "angular_alpha_deg", "alpha is not the first dimension"
    assert "margin" not in got, "margin is still searched under 'joint'"
    assert len(got) == 5, "expected 5 dims, got %d" % len(got)
    assert got == S.train_names(cfg.train), "space and train_names disagree"
    return "loss_type='joint': alpha replaces margin, still 5 dims"


def check_joint_sep_space_adds_lambda_sep():
    cfg = base_cfg(loss_type="joint_sep")
    sp = S.train_space(cfg.search, cfg.train)
    got = names_of(sp)
    assert got[:2] == ("angular_alpha_deg", "lambda_sep"), "wrong leading dims: %r" % (got,)
    assert "margin" not in got, "margin is still searched under 'joint_sep'"
    assert len(got) == 6, "expected 6 dims, got %d" % len(got)
    assert got == S.train_names(cfg.train), "space and train_names disagree"
    assert sp[1].prior == "log-uniform", \
        "lambda_sep must be log-uniform: its useful range spans C = 2 and C >= 3"
    lo, hi = sp[1].bounds
    assert (lo, hi) == (1e-3, 1.0), "lambda_sep bounds are %r" % ((lo, hi),)
    return "loss_type='joint_sep': 6 dims, lambda_sep log-uniform over [1e-3, 1]"


def check_full_search_space_is_covered():
    """Every space must expose exactly the names its reader expects, for all
    three loss types, in both staged and joint modes."""
    for lt in ("triplet", "joint", "joint_sep"):
        cfg = base_cfg(loss_type=lt)
        assert names_of(S.train_space(cfg.search, cfg.train)) \
            == S.train_names(cfg.train), "staged mismatch at %r" % lt
        jsp = S.joint_space(cfg.search, cfg.regularization, cfg.train)
        assert names_of(jsp) == S.joint_names(cfg.train), \
            "joint mismatch at %r: %r vs %r" \
            % (lt, names_of(jsp), S.joint_names(cfg.train))
        assert len(set(names_of(jsp))) == len(names_of(jsp)), \
            "duplicate dimension name in the joint space at %r" % lt
    return "staged and joint spaces match their name lists for all 3 loss types"


# --------------------------------------------------------------------------- #
# point -> config -> winner round trip
# --------------------------------------------------------------------------- #
def check_point_writes_the_right_fields():
    cfg = base_cfg(loss_type="joint_sep")
    point = [12.5, 0.037, 1e-3, 0.02, 0.001, 1e-4]      # alpha, lam, lr, u1, u2, wd
    out = S.config_from_train_point(cfg, point)
    assert abs(out.train.angular_alpha_deg - 12.5) < 1e-12, "alpha not written"
    assert abs(out.train.lambda_sep - 0.037) < 1e-12, "lambda_sep not written"
    assert abs(out.train.lr - 1e-3) < 1e-12, "lr landed in the wrong field"
    assert abs(out.train.beta1 - (1.0 - 0.02)) < 1e-12, "beta1 conversion broken"
    assert abs(out.train.margin - cfg.train.margin) < 1e-12, \
        "margin was overwritten although it is not searched under 'joint_sep'"

    cfg_t = base_cfg()
    out_t = S.config_from_train_point(cfg_t, [0.5, 1e-3, 0.02, 0.001, 1e-4])
    assert abs(out_t.train.margin - 0.5) < 1e-12, "legacy margin write broken"
    return "each loss type writes its own HPs and leaves the others alone"


def check_winner_reader_round_trips():
    class _Res(object):
        def __init__(self, x):
            self.x = x

    for lt, point, want in (
            ("triplet", [0.42, 1e-3, 0.02, 0.001, 1e-4], {"margin": 0.42}),
            ("joint", [11.0, 1e-3, 0.02, 0.001, 1e-4],
             {"angular_alpha_deg": 11.0}),
            ("joint_sep", [11.0, 0.02, 1e-3, 0.02, 0.001, 1e-4],
             {"angular_alpha_deg": 11.0, "lambda_sep": 0.02}),
    ):
        cfg = base_cfg(loss_type=lt)
        got = S.best_train_dict(_Res(point), cfg.train)
        for k, v in want.items():
            assert abs(got[k] - v) < 1e-12, \
                "%s: %s read as %r, expected %r" % (lt, k, got.get(k), v)
        assert abs(got["lr"] - 1e-3) < 1e-12, "%s: lr misread" % lt
        assert abs(got["beta1"] - 0.98) < 1e-12, "%s: beta1 misread" % lt
        assert set(got) == set(S.loss_hp_names(cfg.train)) | {
            "lr", "beta1", "beta2", "weight_decay"}, \
            "%s: unexpected keys %r" % (lt, sorted(got))
    assert "margin" in S.best_train_dict(_Res([0.42, 1e-3, 0.02, 0.001, 1e-4])), \
        "best_train_dict() without a train_cfg lost the legacy reading"
    return "winner reader round trips for all 3 loss types, legacy call intact"


# --------------------------------------------------------------------------- #
# trainer dispatch
# --------------------------------------------------------------------------- #
def check_trainer_builds_each_loss():
    from dsn_joint_loss import CompositeDSNLoss, JointTripletLoss
    from pytorch_metric_learning import losses as pml_losses
    from train import build_loss_and_miner

    want = (("triplet", pml_losses.TripletMarginLoss),
            ("joint", JointTripletLoss),
            ("joint_sep", CompositeDSNLoss))
    for lt, cls in want:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = TrainConfig(loss_type=lt, angular_alpha_deg=18.0,
                            lambda_sep=0.1)
        loss_fn, miner = build_loss_and_miner(t, n_classes=3)
        assert isinstance(loss_fn, cls), \
            "%s built %s, expected %s" % (lt, type(loss_fn).__name__, cls.__name__)
        assert miner is not None, "%s built no miner" % lt

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = TrainConfig(loss_type="joint_sep")
    try:
        build_loss_and_miner(t, n_classes=None)
    except ValueError as ex:
        assert "n_classes" in str(ex), "wrong error: %s" % ex
    else:
        raise AssertionError("'joint_sep' without n_classes did not raise")
    return "all 3 loss types build; 'joint_sep' demands n_classes"


def check_margin_conversion_is_applied():
    """The config states the margin in COSINE distance and the composite loss
    works in SQUARED EUCLIDEAN. The factor 2 must be applied exactly once."""
    from train import build_loss_and_miner
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = TrainConfig(loss_type="joint", margin=0.3)
    loss_fn, miner = build_loss_and_miner(t, n_classes=3)
    assert abs(loss_fn.margin - 0.6) < 1e-12, \
        "loss margin is %r, expected 2 * 0.3" % loss_fn.margin
    assert abs(float(miner.margin) - 0.3) < 1e-12, \
        "miner margin is %r; the MINER must keep the cosine value" % miner.margin
    return "margin 0.3 cosine -> 0.6 squared-Euclidean in the loss, 0.3 in the miner"


def check_all_miners_work_under_every_loss():
    """loss_type and mining_strategy must be independent: 3 x 3 = 9 cells."""
    from train import build_loss_and_miner
    torch.manual_seed(0)
    z = torch.randn(27, 8)
    z = z / z.norm(dim=1, keepdim=True)
    y = torch.arange(3).repeat_interleave(9)
    n_ok = 0
    for lt in ("triplet", "joint", "joint_sep"):
        for ms in ("hard", "easy_positive", "easy_pos_semihard_neg"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t = TrainConfig(loss_type=lt, mining_strategy=ms,
                                strict_semihard=False)
            loss_fn, miner = build_loss_and_miner(t, n_classes=3)
            out = loss_fn(z, y, miner(z, y))
            assert torch.isfinite(out), "%s x %s gave %r" % (lt, ms, out)
            n_ok += 1
    return "%d of 9 loss x mining cells produce a finite loss" % n_ok


def check_gate_momentum_zero_means_cumulative():
    from train import build_loss_and_miner
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = TrainConfig(loss_type="joint_sep", sep_gate_momentum=0.0)
    loss_fn, _ = build_loss_and_miner(t, n_classes=3)
    assert loss_fn.gate.momentum is None, \
        "sep_gate_momentum=0.0 did not select the cumulative mean"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t2 = TrainConfig(loss_type="joint_sep", sep_gate_momentum=0.05)
    loss_fn2, _ = build_loss_and_miner(t2, n_classes=3)
    assert abs(loss_fn2.gate.momentum - 0.05) < 1e-12, "EMA momentum not passed"
    return "sep_gate_momentum 0.0 -> cumulative mean, 0.05 -> EMA"


def check_census_keys_are_plain_floats():
    """train.py converts loss_fn.stats() to floats ONCE per epoch. Everything in
    that dict must survive float() so the history entry stays JSON-serialisable."""
    import json
    from train import build_loss_and_miner
    torch.manual_seed(0)
    z = torch.randn(27, 8)
    z = z / z.norm(dim=1, keepdim=True)
    y = torch.arange(3).repeat_interleave(9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = TrainConfig(loss_type="joint_sep")
    loss_fn, miner = build_loss_and_miner(t, n_classes=3)
    loss_fn(z, y, miner(z, y))
    census = {k: float(v) for k, v in loss_fn.stats().items()}
    json.dumps(census)                       # raises if anything is not a number
    for key in ("n_mined", "n_strict", "n_active", "sep_active", "latch_step"):
        assert key in census, "census is missing %r" % key
    return "census has %d JSON-serialisable fields" % len(census)


def check_hard_mining_and_strict_filter_are_rejected():
    """PROVABLY EMPTY combination, and silent. TripletMarginMiner
    type_of_triplets="hard" mines D_an < D_ap; the strict semi-hard filter
    keeps only D_ap < D_an. Nothing survives, every batch gives n_active = 0
    and train_loss = 0.0, and the run looks STABLE rather than broken -- the
    silhouette simply never moves. Measured on a real batch: 16814 mined, 0
    surviving. The config must refuse it rather than let it burn cluster
    hours."""
    for lt in ("joint", "joint_sep"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                TrainConfig(loss_type=lt, mining_strategy="hard",
                            strict_semihard=True)
        except ValueError as ex:
            assert "mutually exclusive" in str(ex), "wrong error: %s" % ex
        else:
            raise AssertionError("%s + hard + strict_semihard was accepted" % lt)

    # the three compatible combinations must still be accepted
    ok = [dict(loss_type="joint", mining_strategy="hard", strict_semihard=False),
          dict(loss_type="joint", mining_strategy="easy_pos_semihard_neg",
               strict_semihard=True),
          dict(loss_type="triplet", mining_strategy="hard",
               strict_semihard=True)]
    for kw in ok:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            TrainConfig(**kw)

    # and prove the emptiness empirically, not just by assertion
    import torch
    from pytorch_metric_learning import distances, miners
    from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
    torch.manual_seed(0)
    z = torch.randn(54, 8)
    z = z / z.norm(dim=1, keepdim=True)
    y = torch.arange(3).repeat_interleave(18)
    m_cos = 0.3
    kept = {}
    for tot in ("hard", "semihard"):
        mi = miners.TripletMarginMiner(margin=m_cos, type_of_triplets=tot,
                                       distance=distances.CosineSimilarity())
        a, p, n = lmu.convert_to_triplets(mi(z, y), y)
        d_ap = ((z[a] - z[p]) ** 2).sum(1)
        d_an = ((z[a] - z[n]) ** 2).sum(1)
        d_pn = ((z[p] - z[n]) ** 2).sum(1)
        hi = d_ap + 2.0 * m_cos
        keep = (d_ap < d_an) & (d_an < hi) & (d_ap < d_pn) & (d_pn < hi)
        kept[tot] = (int(a.numel()), int(keep.sum()))
    assert kept["hard"][1] == 0, \
        "hard mining unexpectedly survived the filter: %r" % (kept["hard"],)
    assert kept["semihard"][1] > 0, \
        "semihard mining should survive the filter: %r" % (kept["semihard"],)
    return "hard+strict rejected (%d mined, %d survive); semihard survives " \
        "(%d of %d)" % (kept["hard"][0], kept["hard"][1],
                        kept["semihard"][1], kept["semihard"][0])


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
CHECKS = [
    ("config defaults backwards compatible", check_defaults_are_backwards_compatible),
    ("config validation rejects bad values", check_config_validation),
    ("inert lambda_sep warns, defaults silent",
     check_inert_lambda_sep_warns_but_defaults_are_silent),
    ("triplet space == legacy space", check_triplet_space_is_the_legacy_space),
    ("joint space swaps margin for alpha", check_joint_space_swaps_margin_for_alpha),
    ("joint_sep space adds lambda_sep", check_joint_sep_space_adds_lambda_sep),
    ("all spaces match their name lists", check_full_search_space_is_covered),
    ("point writes the right fields", check_point_writes_the_right_fields),
    ("winner reader round trips", check_winner_reader_round_trips),
    ("trainer builds each loss", check_trainer_builds_each_loss),
    ("margin cosine -> squared Euclidean", check_margin_conversion_is_applied),
    ("3 x 3 loss x mining cells", check_all_miners_work_under_every_loss),
    ("hard + strict filter is rejected",
     check_hard_mining_and_strict_filter_are_rejected),
    ("gate momentum 0 = cumulative", check_gate_momentum_zero_means_cumulative),
    ("census is JSON-serialisable", check_census_keys_are_plain_floats),
]


def main():
    torch.set_num_threads(1)
    width = max(len(n) for n, _ in CHECKS)
    failures = 0
    for name, fn in CHECKS:
        try:
            print("PASS  %-*s  %s" % (width, name, fn()))
        except Exception:
            failures += 1
            print("FAIL  %-*s" % (width, name))
            traceback.print_exc()
    print("")
    if failures:
        print("%d of %d CHECKS FAILED" % (failures, len(CHECKS)))
        return 1
    print("ALL CHECKS PASSED (%d/%d)" % (len(CHECKS), len(CHECKS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
