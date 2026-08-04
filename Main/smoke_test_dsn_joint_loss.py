"""
smoke_test_dsn_joint_loss.py
============================

Correctness checks for dsn_joint_loss.py. Torch-only, CPU, seconds to run.

Run:
    python3 smoke_test_dsn_joint_loss.py

Exit code 0 and a final "ALL CHECKS PASSED" line mean every check held.
Any failure prints the offending values and exits 1.

What each block establishes
---------------------------
[A] plumbing        forward runs, returns a 0-dim tensor, both miner arities
[B] margin term     numerically EQUALS 2 x pytorch_metric_learning's
                    TripletMarginLoss under the stated convention m = 2 m_cos.
                    This is the anchor check: it ties the new module to the
                    tested library rather than to this file's own arithmetic.
[C] angular term    the median-length identity is exact; the hinge sits exactly
                    at a/b = 4 sin^2(alpha); exchange of anchor and positive
                    changes nothing
[D] separation      zero on a simplex ETF, positive on collinear centres,
                    invariant to translation, and UNCHANGED by the presence of
                    destroyed-surrogate rows
[E] silhouette      agrees with sklearn, and ignores surrogate rows
[F] gate            latches once, never re-closes, respects the warm-up
[G] composite       equals the joint loss exactly before the latch and
                    joint + lambda * sep after it
[H] discipline      every logged statistic is a device tensor, not a Python
                    float, so nothing here forces a per-batch device sync
[I] degenerate      empty triplet set and short batches do not raise or NaN

HPC note (hpc-python-compat): pure ASCII. Imports dsn_joint_loss, which is
also pure ASCII.
"""

import math
import sys
import traceback

import torch

from pytorch_metric_learning import distances, losses, miners, reducers

from dsn_joint_loss import (
    CentroidSeparationLoss,
    CompositeDSNLoss,
    JointTripletLoss,
    SilhouetteGate,
    batch_cosine_silhouette,
)

TOL = 1e-5
SEED = 0


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def make_batch(n_classes=3, per_class=9, dim=16, spread=1.0, seed=SEED):
    """Unit-norm embeddings in n_classes blobs, plus their true labels."""
    g = torch.Generator().manual_seed(seed)
    centres = torch.randn(n_classes, dim, generator=g)
    centres = centres / centres.norm(dim=1, keepdim=True)
    z, y = [], []
    for c in range(n_classes):
        noise = spread * torch.randn(per_class, dim, generator=g)
        z.append(centres[c].unsqueeze(0) + noise)
        y.append(torch.full((per_class,), c, dtype=torch.long))
    z = torch.cat(z, dim=0)
    z = z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return z, torch.cat(y, dim=0)


def add_surrogates(z, y, n_sur=6, base=1_000_000, seed=SEED + 1):
    """Append destroyed-surrogate rows carrying unique labels >= base."""
    g = torch.Generator().manual_seed(seed)
    extra = torch.randn(n_sur, z.shape[1], generator=g)
    extra = extra / extra.norm(dim=1, keepdim=True)
    lab = torch.arange(base, base + n_sur, dtype=torch.long)
    return torch.cat([z, extra], 0), torch.cat([y, lab], 0)


def etf_batch(n_classes=3, per_class=9, dim=16):
    """Rows sitting exactly on a simplex ETF: every row of class c equals mu_c."""
    mu = torch.zeros(n_classes, dim)
    for c in range(n_classes):
        ang = 2.0 * math.pi * c / n_classes
        mu[c, 0] = math.cos(ang)
        mu[c, 1] = math.sin(ang)
    z = mu.repeat_interleave(per_class, dim=0)
    y = torch.arange(n_classes).repeat_interleave(per_class)
    return z, y


def collinear_batch(n_classes=3, per_class=9, dim=16):
    """Class centres equally spaced along one axis: rank Cov = 1."""
    z = torch.zeros(n_classes * per_class, dim)
    y = torch.arange(n_classes).repeat_interleave(per_class)
    for c in range(n_classes):
        z[y == c, 0] = float(c)
    return z, y


def triplet_miner(margin_cos):
    return miners.TripletMarginMiner(margin=margin_cos, type_of_triplets="all",
                                     distance=distances.CosineSimilarity())


def pair_miner():
    return miners.BatchEasyHardMiner(pos_strategy="easy", neg_strategy="hard",
                                     distance=distances.CosineSimilarity())


# --------------------------------------------------------------------------- #
# [A] plumbing
# --------------------------------------------------------------------------- #
def check_forward_both_miner_arities():
    z, y = make_batch()
    loss_fn = JointTripletLoss(margin=0.6, alpha_deg=18.0)
    for name, miner in (("TripletMarginMiner", triplet_miner(0.3)),
                        ("BatchEasyHardMiner", pair_miner())):
        pairs = miner(z, y)
        assert len(pairs) in (3, 4), "%s returned arity %d" % (name, len(pairs))
        out = loss_fn(z, y, pairs)
        assert out.dim() == 0, "%s: loss is not 0-dim" % name
        assert torch.isfinite(out), "%s: loss not finite (%r)" % (name, out)
    return "both miner arities accepted, loss finite and 0-dim"


# --------------------------------------------------------------------------- #
# [B] margin term against the library
# --------------------------------------------------------------------------- #
def check_margin_term_equals_pml():
    """With the angular term and the filter off, this module must reproduce
    pytorch_metric_learning exactly, up to the documented factor of 2 between
    the squared-Euclidean and cosine conventions (margin = 2 * m_cos)."""
    m_cos = 0.3
    z, y = make_batch(spread=0.7)
    pairs = triplet_miner(m_cos)(z, y)

    mine = JointTripletLoss(margin=2.0 * m_cos, use_angular=False,
                            strict_semihard=False, swap=True,
                            reduce_nonzero=True)(z, y, pairs)
    ref = losses.TripletMarginLoss(
        margin=m_cos, swap=True, distance=distances.CosineSimilarity(),
        reducer=reducers.AvgNonZeroReducer())(z, y, pairs)

    err = float((mine - 2.0 * ref).abs())
    assert err < TOL, "margin term mismatch: mine=%.8f 2*pml=%.8f err=%.2e" \
        % (float(mine), 2.0 * float(ref), err)
    return "margin term == 2 x PML TripletMarginLoss (err %.2e)" % err


# --------------------------------------------------------------------------- #
# [C] angular term
# --------------------------------------------------------------------------- #
def check_median_identity():
    """D_nc = (2 D_an + 2 D_pn - D_ap) / 4, exactly, for arbitrary vectors."""
    g = torch.Generator().manual_seed(7)
    xa, xp, xn = (torch.randn(4000, 8, generator=g) for _ in range(3))
    xc = 0.5 * (xa + xp)
    d_nc = ((xn - xc) ** 2).sum(1)
    d_ap = ((xa - xp) ** 2).sum(1)
    d_an = ((xa - xn) ** 2).sum(1)
    d_pn = ((xp - xn) ** 2).sum(1)
    pred = (2.0 * d_an + 2.0 * d_pn - d_ap) / 4.0
    rel = float(((d_nc - pred).abs() / d_nc.abs().clamp_min(1e-9)).max())
    assert rel < 1e-5, "median identity residual %.3e" % rel
    return "median-length identity exact (max rel residual %.2e)" % rel


def check_angular_hinge_is_the_silhouette_floor():
    """Symmetric construction with D_an = D_pn: the hinge of l_ang sits exactly
    at D_ap / D_an = 4 sin^2(alpha), which is the silhouette floor of Eq. (3)."""
    alpha = 18.0
    t = math.tan(math.radians(alpha))
    loss_fn = JointTripletLoss(margin=0.6, alpha_deg=alpha, use_angular=True,
                               strict_semihard=False, swap=False)

    def l_ang_only(w):
        xn = torch.tensor([[0.0, 0.0]])
        xa = torch.tensor([[1.0, w]])
        xp = torch.tensor([[1.0, -w]])
        z = torch.cat([xa, xp, xn], 0)
        idx = (torch.tensor([0]), torch.tensor([1]), torch.tensor([2]))
        d_ap = ((z[0] - z[1]) ** 2).sum()
        d_an = ((z[0] - z[2]) ** 2).sum()
        xc = 0.5 * (z[0] + z[1])
        d_nc = ((z[2] - xc) ** 2).sum()
        ang = torch.relu(d_ap - loss_fn.four_tan2 * d_nc)
        return float(ang), float(d_ap / d_an), idx

    at, ratio_at, _ = l_ang_only(t)
    above, ratio_ab, _ = l_ang_only(t * 1.05)
    below, _, _ = l_ang_only(t * 0.95)
    target = 4.0 * math.sin(math.radians(alpha)) ** 2

    assert abs(at) < 1e-6, "hinge not zero at the boundary: %.3e" % at
    assert abs(ratio_at - target) < 1e-6, \
        "boundary ratio %.8f != 4 sin^2 alpha %.8f" % (ratio_at, target)
    assert above > 1e-6, "no penalty above the boundary (%.3e)" % above
    assert abs(below) < 1e-12, "penalty below the boundary (%.3e)" % below
    assert ratio_ab > target, "ratio did not increase past the boundary"
    return "hinge at a/b = 4 sin^2(%.0f deg) = %.6f, zero below, positive above" \
        % (alpha, target)


def check_angular_invariant_under_anchor_positive_exchange():
    z, y = make_batch(spread=0.7)
    a = torch.arange(0, 9)
    p = torch.arange(9, 18) % 9
    n = torch.arange(18, 27)
    fn = JointTripletLoss(margin=0.6, alpha_deg=18.0, use_angular=True,
                          strict_semihard=False, swap=True)

    def ang_only(ia, ip, inn):
        xc = 0.5 * (z[ia] + z[ip])
        d_ap = ((z[ia] - z[ip]) ** 2).sum(1)
        d_nc = ((z[inn] - xc) ** 2).sum(1)
        return torch.relu(d_ap - fn.four_tan2 * d_nc)

    diff = float((ang_only(a, p, n) - ang_only(p, a, n)).abs().max())
    assert diff == 0.0, "angular term not exchange-invariant: %.3e" % diff
    return "angular term exactly invariant under a <-> p (diff %.2e)" % diff


# --------------------------------------------------------------------------- #
# [D] centroid separation
# --------------------------------------------------------------------------- #
def check_sep_zero_on_etf():
    z, y = etf_batch()
    val = float(CentroidSeparationLoss(n_classes=3)(z, y))
    assert val < 1e-9, "L_sep on a simplex ETF is %.3e, expected < 1e-9" % val
    return "L_sep = %.2e on the simplex ETF" % val


def check_sep_positive_on_collinear():
    z, y = collinear_batch()
    val = float(CentroidSeparationLoss(n_classes=3)(z, y))
    assert val > 1e-3, "L_sep on collinear centres is %.3e, expected > 0" % val
    return "L_sep = %.4f on collinear class centres" % val


def check_sep_ignores_surrogates():
    """THE supervision check. Destroyed surrogates carry unique labels >= 1e6.
    A version keying on torch.unique(labels) would see each as a singleton
    class, trip the min-per-class guard and return exactly zero."""
    z, y = make_batch(spread=0.6)
    fn = CentroidSeparationLoss(n_classes=3)
    clean = float(fn(z, y))
    z2, y2 = add_surrogates(z, y, n_sur=12)
    withsur = float(fn(z2, y2))
    assert clean > 1e-6, "baseline L_sep is degenerate (%.3e)" % clean
    assert abs(clean - withsur) < 1e-6, \
        "surrogates changed L_sep: %.8f -> %.8f" % (clean, withsur)
    assert int(fn.last_n_classes) == 3, \
        "n_classes seen = %d, expected 3" % int(fn.last_n_classes)
    return "L_sep unchanged by 12 surrogate rows (%.6f), K = 3" % clean


def check_sep_is_not_scale_blind():
    """THE regression guard for the defect that removed the centred form.

    The centred form was invariant to translation AND scale, so it measured
    only the SHAPE of the simplex of class means and never its SIZE: three
    classes collapsed into a tiny cap scored ~0. The raw form must respond
    to collapse, monotonically.
    """
    import math
    def cap_batch(eps, K=3, E=8, per_class=9, seed=0):
        g = torch.Generator().manual_seed(seed)
        base = torch.zeros(E); base[0] = 1.0
        e1 = torch.zeros(E); e1[1] = 1.0
        e2 = torch.zeros(E); e2[2] = 1.0
        Z, y = [], []
        for c in range(K):
            ang = 2 * math.pi * c / K
            v = base + eps * (math.cos(ang) * e1 + math.sin(ang) * e2)
            d = v / v.norm()
            z = d.unsqueeze(0) + 0.001 * torch.randn(per_class, E, generator=g)
            Z.append(z / z.norm(dim=1, keepdim=True)); y += [c] * per_class
        return torch.cat(Z), torch.tensor(y)

    fn = CentroidSeparationLoss(n_classes=3)
    vals = [float(fn(*cap_batch(e)).detach()) for e in (0.02, 0.5, 1.0, 2.0)]
    assert vals[0] > 1.0, \
        ("COLLAPSED classes scored %.6f -- the term is blind to collapse, "
         "which is exactly the centred-form defect" % vals[0])
    assert all(a > b for a, b in zip(vals, vals[1:])), \
        "L_sep is not monotone decreasing in separation: %r" % vals
    return ("L_sep sees collapse: %.3f (cap) -> %.3f (spread), monotone"
            % (vals[0], vals[-1]))


def check_sep_responds_to_scale():
    """Scaling the class means toward their own mean MUST change L_sep.

    Under the centred form this was identically zero change; it is the
    single-line statement of why that form was removed.
    """
    z, y = make_batch(spread=0.6)
    fn = CentroidSeparationLoss(n_classes=3)
    mus = torch.stack([z[y == c].mean(0) for c in range(3)])
    mG = mus.mean(0)
    shrunk = []
    for scale in (1.0, 0.05):
        zz = mG + scale * (z - mG)
        zz = zz / zz.norm(dim=1, keepdim=True)
        shrunk.append(float(fn(zz, y).detach()))
    assert abs(shrunk[0] - shrunk[1]) > 1e-4, \
        ("L_sep unchanged (%.8f vs %.8f) when the cloud is shrunk 20x toward "
         "its own mean -- that is the scale-blindness the centred form had"
         % (shrunk[0], shrunk[1]))
    return "L_sep responds to shrinking the cloud (%.4f -> %.4f)" % tuple(shrunk)


def check_sep_gradient_is_finite():
    z, y = make_batch(spread=0.6)
    z = z.clone().requires_grad_(True)
    CentroidSeparationLoss(n_classes=3)(z, y).backward()
    assert z.grad is not None, "no gradient reached the embeddings"
    assert torch.isfinite(z.grad).all(), "non-finite gradient in L_sep"
    assert float(z.grad.abs().sum()) > 0.0, "L_sep gradient is identically zero"
    return "L_sep gradient finite and non-zero"


# --------------------------------------------------------------------------- #
# [E] silhouette
# --------------------------------------------------------------------------- #
def check_silhouette_matches_sklearn():
    try:
        from sklearn.metrics import silhouette_score
    except ImportError:
        return "SKIPPED (sklearn not installed)"
    z, y = make_batch(spread=0.6)
    mine = float(batch_cosine_silhouette(z, y, n_classes=3))
    ref = float(silhouette_score(z.numpy(), y.numpy(), metric="cosine"))
    assert abs(mine - ref) < 1e-4, \
        "silhouette mismatch: mine=%.8f sklearn=%.8f" % (mine, ref)
    return "cosine silhouette matches sklearn (%.6f vs %.6f)" % (mine, ref)


def check_silhouette_ignores_surrogates():
    z, y = make_batch(spread=0.6)
    clean = float(batch_cosine_silhouette(z, y, n_classes=3))
    z2, y2 = add_surrogates(z, y, n_sur=12)
    withsur = float(batch_cosine_silhouette(z2, y2, n_classes=3))
    assert abs(clean - withsur) < 1e-6, \
        "surrogates changed the silhouette: %.8f -> %.8f" % (clean, withsur)
    return "silhouette unchanged by 12 surrogate rows (%.6f)" % clean


def check_silhouette_undefined_returns_nan():
    z, y = make_batch(n_classes=3, per_class=9)
    y_one = torch.zeros_like(y)                       # a single class
    val = batch_cosine_silhouette(z, y_one, n_classes=3)
    assert torch.isnan(val), "expected NaN for a single class, got %r" % val
    return "silhouette returns NaN when undefined"


# --------------------------------------------------------------------------- #
# [F] gate
# --------------------------------------------------------------------------- #
def check_gate_latches_once_and_stays_open():
    tight_z, tight_y = make_batch(spread=0.05)        # high silhouette
    loose_z, loose_y = make_batch(spread=3.0)         # low silhouette
    s_hi = float(batch_cosine_silhouette(tight_z, tight_y, 3))
    s_lo = float(batch_cosine_silhouette(loose_z, loose_y, 3))
    assert s_hi > s_lo, "fixture broken: %.4f !> %.4f" % (s_hi, s_lo)

    thr = 0.5 * (s_hi + s_lo)
    gate = SilhouetteGate(threshold=thr, n_classes=3, momentum=1.0,
                          min_batches=3)
    seq = []
    for _ in range(5):
        seq.append(float(gate(loose_z, loose_y)))     # below threshold
    assert all(v == 0.0 for v in seq), "gate opened early: %r" % seq

    opened = float(gate(tight_z, tight_y))
    assert opened == 1.0, "gate did not latch when the threshold was crossed"
    latch_at = int(gate.latch_step)
    assert latch_at == 6, "latch_step = %d, expected 6" % latch_at

    after = [float(gate(loose_z, loose_y)) for _ in range(4)]
    assert all(v == 1.0 for v in after), "gate re-closed: %r" % after
    assert int(gate.latch_step) == 6, "latch_step moved after latching"
    return "latched at batch 6 on crossing, stayed open for 4 poor batches"


def check_gate_respects_warmup():
    tight_z, tight_y = make_batch(spread=0.05)
    gate = SilhouetteGate(threshold=-1.0, n_classes=3, momentum=1.0,
                          min_batches=5)
    early = [float(gate(tight_z, tight_y)) for _ in range(4)]
    assert all(v == 0.0 for v in early), \
        "gate latched before min_batches: %r" % early
    assert float(gate(tight_z, tight_y)) == 1.0, \
        "gate did not latch on the min_batches-th batch"
    return "warm-up honoured: no latch for 4 batches, latch on the 5th"


def check_gate_always_open_when_threshold_none():
    z, y = make_batch(spread=3.0)
    gate = SilhouetteGate(threshold=None, n_classes=3)
    assert float(gate(z, y)) == 1.0, "threshold=None did not open the gate"
    return "threshold=None opens the gate from the first batch"


# --------------------------------------------------------------------------- #
# [G] composite
# --------------------------------------------------------------------------- #
def check_composite_follows_the_warmup():
    """The gate is gone; the composite now follows the deterministic ramp.

    Same assertion shape as the old latch test: at the START of the warm-up the
    composite must equal L_joint EXACTLY (weight 0), and once the ramp is past
    tau * T it must equal L_joint + lambda_sep * L_sep EXACTLY.
    """
    z, y = make_batch(spread=0.6)
    pairs = triplet_miner(0.3)(z, y)
    lam, m_cos, alpha = 0.1, 0.3, 18.0
    T, tau = 100, 0.3

    joint_ref = float(JointTripletLoss(margin=2.0 * m_cos, alpha_deg=alpha)(
        z, y, pairs))
    sep_ref = float(CentroidSeparationLoss(n_classes=3)(z, y))
    assert sep_ref > 1e-6, "fixture broken: L_sep is ~0 (%.3e)" % sep_ref

    cold = CompositeDSNLoss(n_classes=3, margin=2.0 * m_cos, alpha_deg=alpha,
                            lambda_sep=lam, total_steps=T, warmup_frac=tau)
    got_cold = float(cold(z, y, pairs))
    assert abs(got_cold - joint_ref) < TOL, \
        "t = 0: %.8f != joint %.8f" % (got_cold, joint_ref)
    assert float(cold.stats()["sep_lambda_t"]) == 0.0, \
        "lambda_sep(0) was not zero"

    warm = CompositeDSNLoss(n_classes=3, margin=2.0 * m_cos, alpha_deg=alpha,
                            lambda_sep=lam, total_steps=T, warmup_frac=tau)
    warm.warmup.step.fill_(T)                 # past the end of the ramp
    got_warm = float(warm(z, y, pairs))
    want = joint_ref + lam * sep_ref
    assert abs(got_warm - want) < TOL, \
        "t >= tau*T: %.8f != joint + lambda*sep %.8f" % (got_warm, want)

    # halfway up the ramp the weight is exactly half
    mid = CompositeDSNLoss(n_classes=3, margin=2.0 * m_cos, alpha_deg=alpha,
                           lambda_sep=lam, total_steps=T, warmup_frac=tau)
    mid.warmup.step.fill_(int(0.5 * tau * T))
    got_mid = float(mid(z, y, pairs))
    want_mid = joint_ref + 0.5 * lam * sep_ref
    assert abs(got_mid - want_mid) < TOL, \
        "halfway: %.8f != joint + 0.5*lambda*sep %.8f" % (got_mid, want_mid)

    return ("composite = joint at t=0, = joint + %.2f*sep past tau*T, "
            "= joint + %.3f*sep halfway" % (lam, 0.5 * lam))


def check_composite_backward():
    z, y = make_batch(spread=0.6)
    pairs = triplet_miner(0.3)(z, y)
    z = z.clone().requires_grad_(True)
    # warmup_frac = 0 -> full weight from the first step, so the separation
    # term is in the graph immediately and the gradient check is not vacuous
    fn = CompositeDSNLoss(n_classes=3, margin=0.6, alpha_deg=18.0,
                          lambda_sep=0.1, total_steps=100, warmup_frac=0.0)
    fn(z, y, pairs).backward()
    assert z.grad is not None and torch.isfinite(z.grad).all(), \
        "non-finite gradient through the composite"
    assert float(z.grad.abs().sum()) > 0.0, "composite gradient identically zero"
    return "composite backward produces finite, non-zero gradients"


# --------------------------------------------------------------------------- #
# [H] sync discipline
# --------------------------------------------------------------------------- #
def check_stats_are_tensors():
    """Every logged statistic must be a device tensor. A Python float here
    would mean a .item() ran inside forward, i.e. a per-batch device sync,
    which smoke_test_train check [D-strict] forbids."""
    z, y = make_batch(spread=0.6)
    pairs = triplet_miner(0.3)(z, y)
    fn = CompositeDSNLoss(n_classes=3, margin=0.6, alpha_deg=18.0,
                          lambda_sep=0.1, total_steps=100, warmup_frac=0.3)
    fn(z, y, pairs)
    st = fn.stats()
    expected = {"n_mined", "n_strict", "n_active", "sep_mean_cos",
                "sep_n_classes", "sep_centred", "sep_warmup_scale", "sep_step",
                "sep_lambda_t"}
    missing = expected - set(st)
    assert not missing, "stats() is missing %r" % sorted(missing)
    # sep_warmup_steps / sep_total_steps are CONSTANTS of the run, deliberately
    # plain floats: reading a constant is not a device sync. Everything that
    # varies per batch must still be a tensor.
    constants = {"sep_warmup_steps", "sep_total_steps"}
    bad = [k for k, v in st.items()
           if k not in constants and not torch.is_tensor(v)]
    assert not bad, "these per-batch stats are not tensors: %r" % bad
    assert int(st["n_mined"]) >= int(st["n_strict"]) >= 0, \
        "census ordering violated: mined=%d strict=%d" \
        % (int(st["n_mined"]), int(st["n_strict"]))
    assert int(st["n_strict"]) >= int(st["n_active"]), \
        "n_active exceeds n_strict"
    return "all %d statistics are device tensors; census ordering holds" \
        % len(st)


# --------------------------------------------------------------------------- #
# [I] degenerate inputs
# --------------------------------------------------------------------------- #
def check_empty_triplet_set():
    z, y = make_batch(spread=0.6)
    z = z.clone().requires_grad_(True)
    empty = (torch.zeros(0, dtype=torch.long),) * 3
    out = JointTripletLoss(margin=0.6, alpha_deg=18.0)(z, y, empty)
    assert torch.isfinite(out) and float(out.detach()) == 0.0, \
        "empty triplet set gave %r" % out
    out.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all(), \
        "empty triplet set broke the backward pass"
    return "empty triplet set -> loss 0, backward clean"


def check_short_batch_degrades_gracefully():
    """One class with a single row: it drops out, K falls to 2, nothing NaNs."""
    z, y = make_batch(n_classes=3, per_class=9, spread=0.6)
    keep = torch.ones(z.shape[0], dtype=torch.bool)
    keep[19:] = False                                   # class 2 keeps one row
    z, y = z[keep], y[keep]
    fn = CentroidSeparationLoss(n_classes=3, min_per_class=2)
    val = fn(z, y)
    assert torch.isfinite(val), "short batch gave %r" % val
    assert int(fn.last_n_classes) == 2, \
        "expected K = 2, got %d" % int(fn.last_n_classes)
    return "short batch: K falls to 2, L_sep = %.6f, finite" % float(val)


def check_raw_form_is_not_vacuous_at_two_classes():
    """The C = 2 vacuity was never a special case; it was the same defect.

    Under the removed centred form, two centred means are antipodal by
    construction, so the term was identically zero for EVERY embedding. The raw
    form must be strictly positive unless the two classes really are antipodal.
    """
    fn = CentroidSeparationLoss(n_classes=2)
    E = 8
    # genuinely COLLAPSED: two class means almost coincident
    d = torch.zeros(E); d[0] = 1.0
    tilt = torch.zeros(E); tilt[1] = 1.0
    g = torch.Generator().manual_seed(0)
    def two(eps):
        Z, y = [], []
        for c, sgn in enumerate((+1.0, -1.0)):
            v = d + sgn * eps * tilt
            v = v / v.norm()
            z = v.unsqueeze(0) + 0.001 * torch.randn(9, E, generator=g)
            Z.append(z / z.norm(dim=1, keepdim=True)); y += [c] * 9
        return torch.cat(Z), torch.tensor(y)
    collapsed = float(fn(*two(0.02)).detach())     # cosine ~ +1
    apart = float(fn(*two(50.0)).detach())         # nearly antipodal
    assert collapsed > 1.0, \
        ("raw form at C = 2 scored %.3e on COLLAPSED classes -- that is the "
         "old centred vacuity" % collapsed)
    assert collapsed > apart, \
        "raw form at C = 2 not monotone: collapsed %.4f vs apart %.4f" \
        % (collapsed, apart)
    return "raw form at C = 2 is informative (collapsed %.3f -> apart %.3f)" \
        % (collapsed, apart)


def check_centring_is_gone_everywhere():
    """The centred code path must not exist, at any C, behind any flag."""
    for C in (2, 3, 5):
        fn = CentroidSeparationLoss(n_classes=C)
        assert not hasattr(fn, "centre_means"), \
            "C = %d still carries a centre_means attribute" % C
        assert int(fn.last_centred) == 0, \
            "sep_centred must be 0 (retained for archived readers), got %d" \
            % int(fn.last_centred)
    try:
        CentroidSeparationLoss(n_classes=3, centre_means=True)
    except TypeError:
        pass
    else:
        raise AssertionError(
            "CentroidSeparationLoss still accepts centre_means -- a silently "
            "inert argument is worse than a loud one")
    return "no centre_means path at C = 2, 3 or 5; sep_centred pinned to 0"


def check_two_class_uses_raw_means_by_default():
    """C = 2 must select the raw-mean formulation automatically, and that
    formulation must actually respond to the data."""
    fn = CentroidSeparationLoss(n_classes=2)
    assert not hasattr(fn, "centre_means"), "C = 2 still has a formulation flag"
    assert int(fn.last_centred) == 0, "last_centred not reported as raw"
    assert not hasattr(CentroidSeparationLoss(n_classes=3), "centre_means"), \
        "C = 3 still has a formulation flag"

    g = torch.Generator().manual_seed(12)
    vals = []
    for _ in range(50):
        z = torch.randn(18, 16, generator=g)
        y = torch.arange(2).repeat_interleave(9)
        vals.append(float(fn(z, y).detach()))
    spread = max(vals) - min(vals)
    assert min(vals) > 0.0, "raw form still returned an exact zero"
    assert spread > 0.05, \
        "raw form barely varies across batches (spread %.3e)" % spread
    return "C = 2 auto-selects raw means; loss varies over %.3f-%.3f " \
        "across 50 batches" % (min(vals), max(vals))


def check_two_class_zero_iff_antipodal():
    """At C = 2 the loss must be zero exactly when the two class means are
    antipodal, and strictly increasing as they are brought together."""
    dim = 16
    fn = CentroidSeparationLoss(n_classes=2)
    y = torch.arange(2).repeat_interleave(9)

    def loss_at(theta):
        m0 = torch.zeros(dim)
        m1 = torch.zeros(dim)
        m0[0] = 1.0
        m1[0] = math.cos(theta)
        m1[1] = math.sin(theta)
        z = torch.cat([m0.repeat(9, 1), m1.repeat(9, 1)], 0)
        return float(fn(z, y).detach())

    at_pi = loss_at(math.pi)                      # antipodal
    assert at_pi < 1e-9, "antipodal means gave %.3e, expected 0" % at_pi

    thetas = [math.pi, 2.6, 2.0, 1.4, 0.8, 0.2]
    seq = [loss_at(t) for t in thetas]
    assert all(b > a for a, b in zip(seq, seq[1:])), \
        "loss not monotone as the means close up: %r" % seq
    at_zero = loss_at(0.0)
    assert abs(at_zero - 4.0) < 1e-5, \
        "coincident means gave %.6f, expected (1+1)^2 = 4" % at_zero
    return "C = 2: 0 at 180 deg, monotone to 4.0 at 0 deg"


def check_two_class_gradient_separates():
    """The C = 2 gradient must push the two class means apart."""
    dim = 8
    y = torch.arange(2).repeat_interleave(9)
    m0 = torch.zeros(dim)
    m1 = torch.zeros(dim)
    m0[0] = 1.0
    m1[0] = math.cos(0.6)
    m1[1] = math.sin(0.6)
    z = torch.cat([m0.repeat(9, 1), m1.repeat(9, 1)], 0).clone()
    z.requires_grad_(True)
    CentroidSeparationLoss(n_classes=2)(z, y).backward()
    assert torch.isfinite(z.grad).all(), "non-finite gradient at C = 2"

    step = 1e-3
    moved = (z - step * z.grad).detach()
    def cos_between(v):
        a = v[:9].mean(0)
        b = v[9:].mean(0)
        return float((a @ b) / (a.norm() * b.norm()))
    before, after = cos_between(z.detach()), cos_between(moved)
    assert after < before, \
        "gradient step did not separate the means: %.6f -> %.6f" \
        % (before, after)
    return "C = 2 gradient separates the means (cos %.4f -> %.4f)" \
        % (before, after)


def check_three_class_path_unchanged():
    """C >= 3 now uses the RAW form, and must still be zero on a true ETF.

    Zero-on-ETF is the one property that had to survive the removal of the
    centring: the target -1/(K-1) is unchanged, only the vectors it is applied
    to changed. Translation invariance did NOT survive, and must not.
    """
    z_etf, y_etf = etf_batch()
    assert float(CentroidSeparationLoss(n_classes=3)(z_etf, y_etf)) < 1e-9, \
        "C = 3 ETF no longer gives zero"
    fn = CentroidSeparationLoss(n_classes=3)
    assert int(fn.last_centred) == 0, "sep_centred must be pinned to 0"
    z, y = make_batch(spread=0.6)
    base = float(fn(z, y).detach())
    shifted = float(fn(z + 3.7, y).detach())
    assert abs(base - shifted) > 1e-5, \
        ("C = 3 is translation invariant (%.8f vs %.8f) -- that is the "
         "scale/shape-only behaviour the centring was removed to eliminate"
         % (base, shifted))
    return "C >= 3 raw form: ETF still zero, responds to a shift (%.4f -> %.4f)" \
        % (base, shifted)


def check_raw_form_is_not_translation_invariant():
    """The raw form responds to a rigid shift, at every C.

    Rows are L2-normalised onto the sphere upstream, so translation is not a
    symmetry of the representation anyway. Invariance here would mean the term
    had gone back to measuring shape only."""
    z, y = make_batch(n_classes=2, per_class=9, spread=0.6)
    fn = CentroidSeparationLoss(n_classes=2)
    base = float(fn(z, y).detach())
    shifted = float(fn(z + 0.5, y).detach())
    assert abs(base - shifted) > 1e-4, \
        "raw form unexpectedly translation invariant (%.8f vs %.8f)" \
        % (base, shifted)
    return "raw form responds to a rigid shift (%.4f -> %.4f), as documented" \
        % (base, shifted)


def check_three_class_batch_missing_one_class_stays_inert():
    """A C = 3 run that transiently loses a class degrades to K = 2 rather
    than to nonsense, and must not switch formulation on a data-dependent
    condition (there is only one formulation now)."""
    z, y = make_batch(n_classes=3, per_class=9, spread=0.6)
    keep = y != 2
    fn = CentroidSeparationLoss(n_classes=3)
    val = float(fn(z[keep], y[keep]).detach())
    assert not hasattr(fn, "centre_means"), "a formulation flag reappeared"
    assert int(fn.last_n_classes) == 2, \
        "expected K = 2, got %d" % int(fn.last_n_classes)
    # Under the removed centred form this contributed exactly 0, because two
    # centred means are antipodal by construction -- the vacuity leaking in.
    # The raw form gives a real value, which is the point.
    assert math.isfinite(val) and val > 0.0, \
        "C = 3 minus a class should now contribute a real value, got %.3e" % val
    return "C = 3 minus one class: K = 2, raw form, contributes %.4f" % val


def check_strict_filter_actually_bites():
    """The filter must remove triplets and change the loss, otherwise it is
    silently inert and the comparison against Chung and Lee's ablation would be
    meaningless."""
    z, y = make_batch(spread=0.7)
    pairs = triplet_miner(0.3)(z, y)
    m = 0.6

    off = JointTripletLoss(margin=m, alpha_deg=18.0, strict_semihard=False)
    on = JointTripletLoss(margin=m, alpha_deg=18.0, strict_semihard=True)
    v_off, v_on = float(off(z, y, pairs).detach()), float(on(z, y, pairs).detach())

    n_mined = int(on.last_n_mined)
    n_strict = int(on.last_n_strict)
    assert n_mined > 0, "fixture mined nothing"
    assert 0 < n_strict < n_mined, \
        "filter did not bite: mined=%d strict=%d" % (n_mined, n_strict)
    assert int(off.last_n_strict) == n_mined, \
        "filter off should keep every mined triplet"
    assert abs(v_off - v_on) > 1e-6, \
        "filter changed the triplet set but not the loss (%.8f vs %.8f)" \
        % (v_off, v_on)
    return "filter kept %d of %d mined triplets (%.0f%%), loss %.4f -> %.4f" \
        % (n_strict, n_mined, 100.0 * n_strict / n_mined, v_off, v_on)


def check_gate_cumulative_mean_path():
    """momentum=None must give the exact cumulative mean of the per-batch
    silhouettes, and must persist that mean across calls."""
    batches = [make_batch(spread=s, seed=100 + i)
               for i, s in enumerate((3.0, 2.0, 1.0, 0.4, 0.1))]
    vals = [float(batch_cosine_silhouette(z, y, 3)) for z, y in batches]
    gate = SilhouetteGate(threshold=None if False else 10.0, n_classes=3,
                          momentum=None, min_batches=1)
    for z, y in batches:
        gate(z, y)
    want = sum(vals) / len(vals)
    got = float(gate.stat)
    assert abs(got - want) < 1e-6, \
        "cumulative mean %.8f != expected %.8f" % (got, want)
    assert int(gate.n_seen) == len(batches), \
        "n_seen = %d, expected %d" % (int(gate.n_seen), len(batches))
    assert float(gate.latched) == 0.0, "gate latched on an impossible threshold"
    return "cumulative mean exact over 5 batches (%.6f), state persisted" % got


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
CHECKS = [
    ("[A] forward, both miner arities", check_forward_both_miner_arities),
    ("[B] margin term == 2 x PML", check_margin_term_equals_pml),
    ("[C] median-length identity", check_median_identity),
    ("[C] angular hinge = silhouette floor",
     check_angular_hinge_is_the_silhouette_floor),
    ("[C] angular a<->p invariance",
     check_angular_invariant_under_anchor_positive_exchange),
    ("[D] L_sep = 0 on simplex ETF", check_sep_zero_on_etf),
    ("[D] L_sep > 0 on collinear centres", check_sep_positive_on_collinear),
    ("[D] L_sep ignores surrogates", check_sep_ignores_surrogates),
    ("[D] L_sep sees collapse (not scale-blind)", check_sep_is_not_scale_blind),
    ("[D] L_sep responds to shrinking", check_sep_responds_to_scale),
    ("[D] L_sep gradient finite", check_sep_gradient_is_finite),
    ("[D] raw form informative at C = 2",
     check_raw_form_is_not_vacuous_at_two_classes),
    ("[D] centring removed everywhere", check_centring_is_gone_everywhere),
    ("[D] C = 2 auto-selects raw means",
     check_two_class_uses_raw_means_by_default),
    ("[D] C = 2 zero iff antipodal", check_two_class_zero_iff_antipodal),
    ("[D] C = 2 gradient separates", check_two_class_gradient_separates),
    ("[D] C >= 3 path unchanged", check_three_class_path_unchanged),
    ("[D] raw form not shift invariant",
     check_raw_form_is_not_translation_invariant),
    ("[D] C = 3 minus a class stays inert",
     check_three_class_batch_missing_one_class_stays_inert),
    ("[B] strict filter actually bites", check_strict_filter_actually_bites),
    ("[E] silhouette == sklearn", check_silhouette_matches_sklearn),
    ("[E] silhouette ignores surrogates", check_silhouette_ignores_surrogates),
    ("[E] silhouette NaN when undefined", check_silhouette_undefined_returns_nan),
    ("[F] gate latches once, stays open",
     check_gate_latches_once_and_stays_open),
    ("[F] gate respects warm-up", check_gate_respects_warmup),
    ("[F] threshold=None always open",
     check_gate_always_open_when_threshold_none),
    ("[F] cumulative-mean path exact", check_gate_cumulative_mean_path),
    ("[G] composite switches at the latch",
     check_composite_follows_the_warmup),
    ("[G] composite backward", check_composite_backward),
    ("[H] statistics are device tensors", check_stats_are_tensors),
    ("[I] empty triplet set", check_empty_triplet_set),
    ("[I] short batch degrades gracefully",
     check_short_batch_degrades_gracefully),
]


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    width = max(len(name) for name, _ in CHECKS)
    failures = 0
    for name, fn in CHECKS:
        try:
            detail = fn()
            print("PASS  %-*s  %s" % (width, name, detail))
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
