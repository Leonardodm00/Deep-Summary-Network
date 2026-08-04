"""
smoke_test_sep_warmup.py
========================

Acceptance tests for Stage 5: the deterministic warm-up

    lambda_sep(t) = lambda_sep * min(1, t / (tau * T)),
    T = max_epochs * batches_per_epoch,

which REPLACES the latching silhouette gate on the joint_sep path.

THE FIVE ACCEPTANCE CHECKS OF THE DESIGN DOCUMENT
-------------------------------------------------
  [5-A] lambda_sep(0) = 0
  [5-B] lambda_sep(tau * T) = lambda_sep, exactly
  [5-C] monotone non-decreasing in t
  [5-D] tau = 0 reduces to a constant weight
  [5-E] the term contributes EXACTLY zero for loss_type != "joint_sep"

AND THE CHECKS THAT GUARD THE WIRING AROUND THEM
------------------------------------------------
  [5-F] the module agrees with the pure schedule at EVERY step of a full run
  [5-G] no device sync: the ramp never reads the step buffer back to the host
  [5-H] the gate is GONE from CompositeDSNLoss, and SilhouetteGate survives as
        a class for the archived analysis tooling
  [5-I] the ramp is auditable: sep_lambda_t and sep_warmup_scale reach .stats()
  [5-J] state_dict round-trip resumes the ramp where it stopped
  [5-K] end to end through build_loss_and_miner: tau and T arrive from the
        config; the removed centring is unreachable and an inert
        sep_centre_means or gate threshold warns
  [5-L] the gradient actually scales with the ramp (the term is not merely
        multiplied by a number that never reaches the graph)
  [5-M] tau is now a SEARCHED axis: a sampled tau round-trips
        point -> cfg.train.sep_warmup_frac -> SepWarmup.warmup_frac
  [5-N] tau = 0 survives that path and still means constant full weight

HOW TO RUN
----------
    cd Main
    PYTHONPATH=. python3 Smoke_Tests/smoke_test_sep_warmup.py

Exit 0 and "ALL SMOKE TESTS PASSED" on success.

HPC note (hpc-python-compat): pure ASCII.
"""

import os
import sys
import warnings

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAIN = os.path.dirname(_HERE)
if _MAIN not in sys.path:
    sys.path.insert(0, _MAIN)

import dsn_joint_loss as D
from config import TrainConfig
from train import build_loss_and_miner


def _expect(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _batch(n_per_class=6, n_classes=3, dim=8, seed=0):
    """A small labelled batch of unit-norm embeddings."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n_per_class * n_classes, dim, generator=g)
    z = z / z.norm(dim=1, keepdim=True)
    y = torch.arange(n_classes).repeat_interleave(n_per_class)
    return z.requires_grad_(True), y


def _triplets(y):
    """Every (a, p, n) triple in the batch, as the miner would hand them over."""
    a, p, n = [], [], []
    m = y.numel()
    for i in range(m):
        for j in range(m):
            for k in range(m):
                if y[i] == y[j] and i != j and y[k] != y[i]:
                    a.append(i); p.append(j); n.append(k)
    return (torch.tensor(a), torch.tensor(p), torch.tensor(n))


# --------------------------------------------------------------------------- #
# the five acceptance checks
# --------------------------------------------------------------------------- #
def test_5a_zero_at_zero():
    for T, tau in ((1000, 0.3), (60, 1.0), (10000, 0.05)):
        g = D.sep_warmup_scale(0, T, tau)
        _expect(g == 0.0,
                "g(0) = %r, expected exactly 0 at T=%d tau=%g" % (g, T, tau))
    print("  [5-A] lambda_sep(0) = 0 exactly OK")


def test_5b_full_at_tau_T():
    for T, tau, lam in ((1000, 0.3, 4.0), (6000, 0.3, 0.05), (60, 1.0, 20.0)):
        t_full = int(round(tau * T))
        g = D.sep_warmup_scale(t_full, T, tau)
        _expect(g == 1.0,
                "g(tau*T) = %r at T=%d tau=%g, expected exactly 1" % (g, T, tau))
        _expect(lam * g == lam,
                "lambda_sep(tau*T) = %r, expected %r" % (lam * g, lam))
        # and it stays there
        _expect(D.sep_warmup_scale(T, T, tau) == 1.0, "g(T) must be 1")
    print("  [5-B] lambda_sep(tau*T) = lambda_sep exactly, and stays OK")


def test_5c_monotone():
    T, tau = 997, 0.3               # prime T, so no lucky round numbers
    prev = -1.0
    for t in range(0, T + 1):
        g = D.sep_warmup_scale(t, T, tau)
        _expect(g >= prev - 1e-15,
                "g decreased at t=%d: %r < %r" % (t, g, prev))
        _expect(0.0 <= g <= 1.0, "g(%d) = %r left [0, 1]" % (t, g))
        prev = g
    print("  [5-C] monotone non-decreasing on all %d steps, and in [0, 1] OK" % T)


def test_5d_tau_zero_is_constant():
    T = 500
    vals = set(D.sep_warmup_scale(t, T, 0.0) for t in range(0, T + 1))
    _expect(vals == {1.0},
            "tau = 0 must give a CONSTANT weight of 1; saw %r" % vals)
    # and through the module, which is what actually runs
    w = D.SepWarmup(total_steps=T, warmup_frac=0.0)
    seen = set(float(w()) for _ in range(50))
    _expect(seen == {1.0}, "SepWarmup at tau = 0 gave %r" % seen)
    print("  [5-D] tau = 0 reduces to a constant weight (pure fn and module) OK")


def test_5e_zero_for_other_loss_types():
    """The separation term must not exist at all off joint_sep."""
    z, y = _batch()
    idx = _triplets(y)
    for lt in ("triplet", "joint"):
        t = TrainConfig(loss_type=lt, mining_strategy="easy_pos_semihard_neg",
                        strict_semihard=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss_fn, _miner = build_loss_and_miner(t, n_classes=3,
                                                   total_steps=1000)
        _expect(not isinstance(loss_fn, D.CompositeDSNLoss),
                "loss_type=%s built a CompositeDSNLoss" % lt)
        _expect(not hasattr(loss_fn, "sep"),
                "loss_type=%s carries a separation term" % lt)
        _expect(not hasattr(loss_fn, "warmup"),
                "loss_type=%s carries a warm-up" % lt)
        stats = loss_fn.stats() if hasattr(loss_fn, "stats") else {}
        for k in ("sep_lambda_t", "sep_warmup_scale", "sep_mean_cos"):
            _expect(k not in stats,
                    "loss_type=%s leaked %r into the census" % (lt, k))
    # and the joint loss value is exactly the JointTripletLoss value
    t = TrainConfig(loss_type="joint", mining_strategy="easy_pos_semihard_neg",
                    strict_semihard=False, margin=0.3, angular_alpha_deg=12.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built, _m = build_loss_and_miner(t, n_classes=3, total_steps=1000)
    ref = D.JointTripletLoss(margin=2.0 * 0.3, alpha_deg=12.0,
                             use_angular=True, strict_semihard=False,
                             swap=True, reduce_nonzero=True)
    _expect(torch.allclose(built(z, y, idx), ref(z, y, idx)),
            "loss_type='joint' does not reproduce JointTripletLoss exactly")
    print("  [5-E] the separation term contributes exactly zero (is absent) "
          "for triplet and joint OK")


# --------------------------------------------------------------------------- #
# wiring guards
# --------------------------------------------------------------------------- #
def test_5f_module_matches_pure_schedule():
    T, tau = 240, 0.3
    w = D.SepWarmup(total_steps=T, warmup_frac=tau)
    for t in range(T):
        want = D.sep_warmup_scale(t, T, tau)
        got = float(w())
        _expect(abs(got - want) < 1e-6,
                "step %d: module gave %r, pure schedule gives %r"
                % (t, got, want))
    _expect(int(w.step) == T, "step counter ended at %d, expected %d"
            % (int(w.step), T))
    print("  [5-F] the module matches the pure schedule at all %d steps OK" % T)


def test_5g_no_device_sync():
    """The ramp must not read the step buffer back to the host per batch.

    Detected structurally: the ramp length is a Python float fixed at
    construction, and forward() touches only tensor ops. A regression that
    replaced it with int(self.step) would make warmup_steps unnecessary, so
    assert both the type and that forward returns a TENSOR sharing storage with
    the buffer rather than a fresh Python float.
    """
    w = D.SepWarmup(total_steps=1000, warmup_frac=0.3)
    _expect(isinstance(w.warmup_steps, float),
            "warmup_steps must be a Python float, got %s"
            % type(w.warmup_steps).__name__)
    out = w()
    _expect(torch.is_tensor(out), "forward must return a tensor, got %s"
            % type(out).__name__)
    _expect(out.data_ptr() == w.scale.data_ptr(),
            "forward must return the scale BUFFER (no fresh allocation, no "
            "host round-trip)")
    _expect(out.shape == torch.Size([]), "the scale must be 0-dim")
    _expect(not out.requires_grad, "the ramp must not carry gradient")
    print("  [5-G] the ramp is computed on-device with no per-batch sync OK")


def test_5h_gate_removed_class_kept():
    _expect(hasattr(D, "SilhouetteGate"),
            "SilhouetteGate was DELETED; the archived analysis tooling reads "
            "its buffers and the design says to keep the class")
    g = D.SilhouetteGate(threshold=0.2, n_classes=3)
    _expect(hasattr(g, "latch_step"), "SilhouetteGate lost its buffers")
    loss = D.CompositeDSNLoss(n_classes=3, margin=0.6, alpha_deg=12.0,
                              lambda_sep=1.0, total_steps=100, warmup_frac=0.3)
    _expect(not hasattr(loss, "gate"),
            "CompositeDSNLoss still constructs a gate")
    _expect(isinstance(loss.warmup, D.SepWarmup),
            "CompositeDSNLoss does not carry a SepWarmup")
    for bad in ("gate_threshold", "gate_momentum", "gate_min_batches"):
        try:
            D.CompositeDSNLoss(n_classes=3, margin=0.6, **{bad: 0.2})
        except TypeError:
            continue
        raise AssertionError(
            "CompositeDSNLoss still accepts %r -- a silently inert argument is "
            "worse than a loud one" % bad)
    print("  [5-H] the gate is gone from the composite; SilhouetteGate survives "
          "as a class OK")


def test_5i_auditable():
    T, tau, lam = 100, 0.3, 4.0
    loss = D.CompositeDSNLoss(n_classes=3, margin=0.6, alpha_deg=12.0,
                              lambda_sep=lam, total_steps=T, warmup_frac=tau)
    z, y = _batch()
    idx = _triplets(y)
    seen = []
    for _ in range(40):
        loss(z, y, idx)
        st = loss.stats()
        seen.append(float(st["sep_lambda_t"]))
    for k in ("sep_lambda_t", "sep_warmup_scale", "sep_step",
              "sep_warmup_steps", "sep_total_steps"):
        _expect(k in loss.stats(), "%r missing from stats() -> it never reaches "
                "history and the ramp is not auditable" % k)
    _expect(seen[0] == 0.0, "first recorded lambda_sep(t) = %r, expected 0"
            % seen[0])
    _expect(abs(seen[-1] - lam * D.sep_warmup_scale(39, T, tau)) < 1e-6,
            "recorded lambda_sep(t) does not track the schedule")
    _expect(all(b >= a - 1e-9 for a, b in zip(seen, seen[1:])),
            "recorded lambda_sep(t) is not monotone")
    _expect(max(seen) <= lam + 1e-9, "recorded weight exceeded lambda_sep")
    # every stats value must survive float() -- that is what train.py does
    for k, v in loss.stats().items():
        float(v)
    print("  [5-I] sep_lambda_t / sep_warmup_scale reach stats(), monotone, "
          "capped at lambda_sep OK")


def test_5j_state_dict_resumes():
    T, tau = 200, 0.3
    a = D.SepWarmup(total_steps=T, warmup_frac=tau)
    for _ in range(25):
        a()
    b = D.SepWarmup(total_steps=T, warmup_frac=tau)
    b.load_state_dict(a.state_dict())
    _expect(int(b.step) == 25, "resumed step = %d, expected 25" % int(b.step))
    _expect(abs(float(b()) - float(D.sep_warmup_scale(25, T, tau))) < 1e-6,
            "the resumed ramp did not continue where it stopped")
    fresh = D.SepWarmup(total_steps=T, warmup_frac=tau)
    _expect(float(fresh()) == 0.0,
            "a checkpoint WITHOUT this state_dict must restart the ramp at 0 "
            "(documented behaviour, asserted so it stays documented)")
    print("  [5-J] state_dict round-trip resumes the ramp where it stopped OK")


def test_5k_end_to_end_from_config():
    t = TrainConfig(loss_type="joint_sep", mining_strategy="easy_pos_semihard_neg",
                    strict_semihard=True, lambda_sep=4.0, sep_warmup_frac=0.3,
                     max_epochs=60,
                    batches_per_epoch=100)
    T = int(t.max_epochs) * int(t.batches_per_epoch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loss_fn, _m = build_loss_and_miner(t, n_classes=3, total_steps=T)
    _expect(isinstance(loss_fn, D.CompositeDSNLoss), "joint_sep did not build "
            "a CompositeDSNLoss")
    _expect(loss_fn.warmup.warmup_frac == 0.3,
            "sep_warmup_frac did not reach the loss: %r"
            % loss_fn.warmup.warmup_frac)
    _expect(loss_fn.warmup.total_steps == T,
            "total_steps did not reach the loss: %r" % loss_fn.warmup.total_steps)
    _expect(abs(loss_fn.warmup.warmup_steps - 0.3 * T) < 1e-9,
            "the ramp length is wrong: %r" % loss_fn.warmup.warmup_steps)
    _expect(not hasattr(loss_fn.sep, "centre_means"),
            "CentroidSeparationLoss still carries a centre_means attribute; "
            "the centred formulation was removed outright")
    _expect(int(loss_fn.sep.last_centred) == 0,
            "sep_centred must be pinned to 0 for archived history readers")
    # a config still setting sep_centre_means must WARN, not be honoured
    t2 = TrainConfig(loss_type="joint_sep",
                     mining_strategy="easy_pos_semihard_neg",
                     sep_centre_means=True)
    with warnings.catch_warnings(record=True) as rec2:
        warnings.simplefilter("always")
        build_loss_and_miner(t2, n_classes=3, total_steps=100)
    _expect(any("INERT" in str(w.message) for w in rec2),
            "sep_centre_means=True was honoured, or ignored silently")
    # an inert gate threshold must WARN, not pass silently
    t3 = TrainConfig(loss_type="joint_sep",
                     mining_strategy="easy_pos_semihard_neg",
                     sep_gate_threshold=0.20)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        build_loss_and_miner(t3, n_classes=3, total_steps=100)
    _expect(any("INERT" in str(w.message) for w in rec),
            "a config carrying sep_gate_threshold built silently; the 20 "
            "archived joint_sep cells all set it, so this must be loud")
    print("  [5-K] tau and T arrive from the config; centring is unreachable; "
          "inert sep_centre_means and gate threshold both warn OK")


def test_5l_gradient_scales_with_ramp():
    """The ramp must actually reach the graph, not just the reported number."""
    T, tau, lam = 100, 0.3, 50.0
    z, y = _batch(seed=3)
    idx = _triplets(y)
    early = D.CompositeDSNLoss(n_classes=3, margin=0.6, alpha_deg=12.0,
                               lambda_sep=lam, total_steps=T, warmup_frac=tau)
    late = D.CompositeDSNLoss(n_classes=3, margin=0.6, alpha_deg=12.0,
                              lambda_sep=lam, total_steps=T, warmup_frac=tau)
    late.warmup.step.fill_(T)                  # past the end of the ramp
    z1 = z.detach().clone().requires_grad_(True)
    z2 = z.detach().clone().requires_grad_(True)
    early(z1, y, idx).backward()
    late(z2, y, idx).backward()
    g1 = z1.grad.norm().item()
    g2 = z2.grad.norm().item()
    _expect(float(early.stats()["sep_lambda_t"]) == 0.0,
            "the first step should carry zero separation weight")
    _expect(abs(float(late.stats()["sep_lambda_t"]) - lam) < 1e-6,
            "past tau*T the weight should be lambda_sep")
    _expect(abs(g1 - g2) > 1e-8,
            "the gradient is IDENTICAL at weight 0 and weight %g: the ramp is "
            "being reported but not applied (grad norms %r vs %r)"
            % (lam, g1, g2))
    print("  [5-L] the ramp reaches the gradient, not just the census "
          "(|grad| %.4g at t=0 vs %.4g at t>=tau*T) OK" % (g1, g2))


def _joint_sep_point(tau, seed=17):
    """A joint_sep point of the 18-axis space, with tau pinned to a value.

    Imported lazily: this file is otherwise a loss-module test and needs no
    skopt. If the search stack is unavailable the two checks below SKIP rather
    than fail, because they are about wiring, not about the ramp itself.
    """
    import warnings as _w
    from skopt.space import Space
    import search as S
    from config import ExperimentConfig

    base = ExperimentConfig()
    names = S.joint_condition_names()
    with _w.catch_warnings():
        _w.simplefilter("ignore")           # the expected tau-cap clip
        space = S.joint_condition_space(base.search, base.regularization,
                                        base.train)
    pt = [list(x) for x in Space(space).rvs(n_samples=1, random_state=seed)][0]
    pt[names.index("loss_type")] = "joint_sep"
    pt[names.index("mining_strategy")] = "easy_pos_semihard_neg"
    pt[names.index("strict_semihard")] = 0
    pt[names.index("sep_warmup_frac")] = float(tau)
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        cfg = S.config_from_joint_condition_point(base, pt)
    return cfg


def test_5m_searched_tau_reaches_the_loss():
    """ACCEPTANCE: the sampled tau reaches cfg.train AND then the loss module.

    tau is the 18th searched axis, so the path
        point -> cfg.train.sep_warmup_frac -> SepWarmup.warmup_frac
    is now load-bearing. A break anywhere along it is silent: the run trains
    perfectly well, just on a schedule nobody chose.
    """
    try:
        cfg = _joint_sep_point(0.17)
    except ImportError:
        print("  [5-M] SKIPPED: the search stack (skopt) is not importable")
        return
    _expect(abs(float(cfg.train.sep_warmup_frac) - 0.17) < 1e-12,
            "the sampled tau did not reach cfg.train: %r"
            % (cfg.train.sep_warmup_frac,))
    _expect(cfg.train.loss_type == "joint_sep",
            "the test point did not build a joint_sep config")
    T = int(cfg.train.max_epochs) * 100
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loss_fn, _m = build_loss_and_miner(cfg.train, n_classes=3,
                                           total_steps=T)
    _expect(abs(float(loss_fn.warmup.warmup_frac) - 0.17) < 1e-12,
            "tau reached the config but NOT the loss: %r"
            % (loss_fn.warmup.warmup_frac,))
    _expect(abs(loss_fn.warmup.warmup_steps - 0.17 * T) < 1e-9,
            "the ramp length is wrong: %r" % (loss_fn.warmup.warmup_steps,))
    print("  [5-M] a SEARCHED tau round-trips point -> config -> "
          "SepWarmup.warmup_frac OK")


def test_5n_searched_tau_zero_is_constant_full_weight():
    """ACCEPTANCE: tau = 0 survives the search path as the control arm.

    tau = 0 must be REACHABLE (it is why the prior is uniform, not
    log-uniform) and must still mean "full weight from step 0" once it has
    travelled through the point -> config -> loss path.
    """
    try:
        cfg = _joint_sep_point(0.0)
    except ImportError:
        print("  [5-N] SKIPPED: the search stack (skopt) is not importable")
        return
    _expect(float(cfg.train.sep_warmup_frac) == 0.0,
            "tau = 0 did not survive to the config: %r"
            % (cfg.train.sep_warmup_frac,))
    T = int(cfg.train.max_epochs) * 100
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loss_fn, _m = build_loss_and_miner(cfg.train, n_classes=3,
                                           total_steps=T)
    _expect(loss_fn.warmup.warmup_steps == 0.0,
            "tau = 0 must disable the ramp entirely; warmup_steps = %r"
            % (loss_fn.warmup.warmup_steps,))
    for _ in range(5):
        _expect(float(loss_fn.warmup()) == 1.0,
                "tau = 0 must give constant FULL weight from step 0")
    print("  [5-N] tau = 0 survives the search path and gives constant full "
          "weight from step 0 OK")


def main():
    print("Stage 5 -- the warm-up that replaces the gate")
    test_5a_zero_at_zero()
    test_5b_full_at_tau_T()
    test_5c_monotone()
    test_5d_tau_zero_is_constant()
    test_5e_zero_for_other_loss_types()
    print("Stage 5 -- wiring guards")
    test_5f_module_matches_pure_schedule()
    test_5g_no_device_sync()
    test_5h_gate_removed_class_kept()
    test_5i_auditable()
    test_5j_state_dict_resumes()
    test_5k_end_to_end_from_config()
    test_5l_gradient_scales_with_ramp()
    print("Stage 5 -- tau as a SEARCHED axis")
    test_5m_searched_tau_reaches_the_loss()
    test_5n_searched_tau_zero_is_constant_full_weight()
    print("\nALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
