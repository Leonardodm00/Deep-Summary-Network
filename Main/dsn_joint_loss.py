"""
dsn_joint_loss.py
=================

Composite metric-learning objective for the Deep Summary Network.

    L(t) = L_joint  +  lambda_sep(t) * L_sep

where

    L_joint    = ( 1 / |T_active| ) * sum_{(a,p,n) in T_strict} (l_trip + l_ang)
    L_sep      = per-batch Gram penalty driving the class means to a simplex
                 equiangular tight frame (ETF). Which class directions it is
                 built from the RAW normalised class means; see
                 CentroidSeparationLoss.
    lambda_sep(t) = lambda_sep * min(1, t / (tau * T)), a DETERMINISTIC linear
                 warm-up over the first tau * T optimiser steps, with
                 T = max_epochs * batches_per_epoch the PLANNED step budget.

The warm-up REPLACES a latching silhouette gate
-----------------------------------------------
Until this change the third term was switched on by a gate: a running estimate
of the TRAINING cosine silhouette, latched permanently the first time it
crossed a threshold. That is gone from this path. SilhouetteGate itself is
KEPT, because the archived analysis tooling still reads its buffers, but
CompositeDSNLoss no longer constructs one.

Four reasons, three of them measured on the 52-cell screening:

  1. It deletes three hyper-parameters (threshold, momentum, min_batches) and
     adds one that is FIXED rather than searched.
  2. The switch is no longer data-dependent, so it cannot fire on a
     fluctuation. MEASURED: the gate latched in 56 of 60 seeds, 70% of which
     peaked in a running silhouette of 0.15 to 0.21 -- i.e. on an excursion --
     at a median held-out silhouette of -0.054, at or below the run's own
     label-shuffled null.
  3. It is deterministic, so seed variance no longer mixes optimisation noise
     with switch timing. Two seeds of one config now share an objective at
     every step, which is what makes their spread interpretable.
  4. lambda_sep becomes active in EVERY run, so a search over it can identify
     it. Under a gate it was inactive in most trials, i.e. a flat direction the
     surrogate learns nothing from.

tau is FIXED, never searched: the schedule integrates to
lambda_sep * T * (1 - tau/2), so tau and lambda_sep trade off almost
multiplicatively and searching both would move along a ridge.

STEP CONVENTION (a plus-or-minus-one that matters for reproducibility). The
ramp is evaluated at t = the number of optimiser steps COMPLETED so far, so the
FIRST batch sees t = 0 and therefore weight exactly 0, and full weight is
reached after exactly tau * T batches. The alternative convention (t = 1 on the
first batch) would start the run with a small but nonzero separation weight,
which is precisely what a warm-up exists to avoid.

The class means are never centred
---------------------------------
L_sep is built from the RAW normalised class means mu_hat_c = mu_c/||mu_c||_2
at every C. An earlier version centred them (mu_c - mu_G) at C >= 3, following
the NC2 literature, and used the raw form only at C = 2 as a special case. Both
the centring and the special case are gone, for one reason:

    centring then normalising is invariant to translation AND scale, so the
    penalty measures only the SHAPE of the simplex of class means, never its
    SIZE.

Any equilateral arrangement satisfies it, including an arbitrarily tiny one.
MEASURED here: three classes collapsed into a cap whose raw pairwise cosine is
+0.9994 -- essentially on top of one another -- score L_sep = 0.000035 centred
and 2.248 raw. The centred form cannot see collapse, which is the single
failure mode this term exists to prevent.

The raw form does not impose an extra constraint. For UNIT vectors {v_c} with
all pairwise inner products rho, || sum_c v_c ||^2 = K + K(K-1) rho, which is
exactly 0 at rho = -1/(K-1). Equiangularity at the ETF target therefore ALREADY
implies sum_c v_hat_c = 0; the two are one condition. On a sphere centred at
the origin, where L2-normalised embeddings live, "the class directions balance
about the origin" IS maximal spread, not a statement about the cloud's absolute
position. mu_G is a nuisance parameter only where features carry an arbitrary
offset -- the NC2 setting, not this one.

The C = 2 vacuity is then not a special case needing a carve-out but the same
defect at its extreme: with two points every configuration is degenerately
"equilateral", so the centred form is identically zero everywhere. At K >= 3 the
collapse is partial rather than total, which is worse, because it is silent.

    L_sep = mean over pairs c != c' of ( <mu_hat_c, mu_hat_c'> + 1/(K-1) )^2

with mu_hat_c = mu_c / ||mu_c||_2 at every C, and -1/(K-1) still the correct
ETF target. K is recomputed per batch, so a batch that transiently loses a
class degrades to a smaller ETF rather than to nonsense; watch sep_n_classes.

sep_centred remains in the census, always 0, so archived readers do not break.

The ramp is evaluated per batch, so the weight changes MID-EPOCH; history
records lambda_sep(t) as it stood at the END of each epoch (sep_lambda_t).

Distance conventions (never mixed silently)
-------------------------------------------
Everything inside this module is SQUARED EUCLIDEAN distance,
D_uv = ||u - v||_2^2. The pipeline config states the margin in COSINE distance,
m_cos. On L2-normalised rows the two are related exactly by

    ||u - v||_2^2 = 2 * d_cos(u, v)        =>        margin = 2 * m_cos

so the caller must pass margin = 2.0 * cfg.train.margin. No conversion is done
implicitly here.

The silhouette used by the gate is the COSINE silhouette, matching
cfg.eval.silhouette_metric and the geometry in which the angular hinge implies
a silhouette floor (S >= 1 - 4 sin^2 alpha).

Label convention (load-bearing)
-------------------------------
The training batch contains BOTH true condition labels (0 .. C-1) AND
destroyed-surrogate labels (>= TripletCollator.unique_label_base, default
1e6). Every class statistic in this module is formed by one-hot encoding the
labels against arange(n_classes), so ANY label outside [0, C-1] -- surrogates
included -- contributes to no class and is silently excluded. This is what
makes L_sep and the silhouette SUPERVISED on the true labels.

A version that instead iterated torch.unique(labels) would see every surrogate
as its own singleton class, trip any min-per-class guard, and return exactly
zero on every batch with no error and no warning.

Device-sync discipline
----------------------
The trainer calls .item() exactly ONCE per epoch (guarded by
smoke_test_train check [D-strict]). Nothing in this module forces a
device->host sync: the gate decision is carried as a 0.0/1.0 TENSOR multiplying
L_sep rather than as a Python bool branch, and every statistic exposed for
logging is a device tensor to be read once at the epoch boundary via .stats().

Early stopping and the planned horizon (recorded, not argued)
-------------------------------------------------------------
T is the PLANNED budget max_epochs * batches_per_epoch, not the number of steps
a run actually takes. A run that early-stops at half its cap therefore spends a
LARGER fraction of its life at full weight than the schedule nominally implies,
but still reaches full weight, provided tau < (actual epochs) / max_epochs. At
tau = 0.3 that holds for any run reaching 30% of its cap. Runs stopping earlier
than that never see full lambda_sep; sep_lambda_t in history is what makes this
visible rather than assumed.

Checkpoint note
---------------
The step counter lives in a registered buffer, so a checkpoint that carries
this module's state_dict resumes the ramp where it stopped. One that does NOT
resets the counter to 0 and re-runs the warm-up from the beginning.

HPC note (hpc-python-compat): pure ASCII, no local imports.
"""

import math

import torch
import torch.nn as nn

from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

__all__ = [
    "class_onehot",
    "batch_cosine_silhouette",
    "JointTripletLoss",
    "CentroidSeparationLoss",
    "SilhouetteGate",
    "sep_warmup_scale",
    "SepWarmup",
    "CompositeDSNLoss",
]

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def class_onehot(labels, n_classes):
    """(M, C) float32 one-hot over TRUE condition labels 0 .. C-1.

    Rows whose label lies outside [0, C-1] -- destroyed surrogates, or any
    other bookkeeping label -- match no column and come back all zero. They
    therefore contribute to no class mean, no class count and no silhouette.
    """
    ids = torch.arange(int(n_classes), device=labels.device, dtype=labels.dtype)
    return (labels.reshape(-1, 1) == ids.reshape(1, -1)).to(torch.float32)


@torch.no_grad()
def batch_cosine_silhouette(emb, labels, n_classes, min_per_class=2):
    """Mean COSINE silhouette of one batch against the TRUE labels.

    Returns a 0-dim float32 tensor. Returns NaN when fewer than two classes
    have at least min_per_class rows, i.e. when the silhouette is undefined.
    Never differentiable: the gate must not backpropagate.

    a(i) = mean cosine distance from i to the other rows of its own class
    b(i) = min over other valid classes of the mean cosine distance from i
    s(i) = (b(i) - a(i)) / max(a(i), b(i)),  averaged over admissible rows i
    """
    z = emb.detach().float()
    z = z / z.norm(dim=1, keepdim=True).clamp_min(_EPS)

    oh = class_onehot(labels, n_classes)                    # (M, C)
    counts = oh.sum(dim=0)                                  # (C,)
    valid_c = counts >= float(min_per_class)                # (C,)
    n_valid = valid_c.sum()

    dist = (1.0 - z @ z.t()).clamp(0.0, 2.0)                # (M, M)
    sums = dist @ oh                                        # (M, C)

    own_count = (counts.reshape(1, -1) * oh).sum(dim=1)     # (M,)
    # dist[i, i] = 0, so the self term already contributes nothing to the sum
    a_i = (sums * oh).sum(dim=1) / (own_count - 1.0).clamp_min(1.0)

    mean_to_c = sums / counts.clamp_min(1.0).reshape(1, -1)  # (M, C)
    other = (oh < 0.5) & valid_c.reshape(1, -1)
    inf = torch.full_like(mean_to_c, float("inf"))
    b_i = torch.where(other, mean_to_c, inf).min(dim=1).values

    row_ok = (oh.sum(dim=1) > 0.5) & (own_count >= float(min_per_class)) \
        & torch.isfinite(b_i)
    s_i = (b_i - a_i) / torch.maximum(a_i, b_i).clamp_min(_EPS)
    s_i = torch.where(row_ok, s_i, torch.zeros_like(s_i))

    n_ok = row_ok.sum()
    s = s_i.sum() / n_ok.clamp_min(1).float()
    ok = (n_valid >= 2) & (n_ok > 0)
    return torch.where(ok, s, torch.full_like(s, float("nan")))


# --------------------------------------------------------------------------- #
# term 1 + 2: margin and angular, on the mined triplets
# --------------------------------------------------------------------------- #
class JointTripletLoss(nn.Module):
    """Margin hinge plus angular hinge, on strict-semi-hard-filtered triplets.

    For each triplet (x_a, x_p, x_n), with x_c = (x_a + x_p) / 2 and all
    distances SQUARED EUCLIDEAN:

        l_trip = [ D_ap - min(D_an, D_pn) + margin ]_+          (swap = True)
        l_ang  = [ D_ap - 4 tan^2(alpha) * D_nc ]_+

    and the strict semi-hard filter keeps only

        D_ap < D_an < D_ap + margin  AND  D_ap < D_pn < D_ap + margin

    The angular hinge is exactly invariant under exchange of x_a and x_p, since
    it is built from D_ap and from the midpoint x_c; it therefore needs no
    switching counterpart. The margin hinge gets its symmetry from swap = True,
    which is why use_switching is not offered here.

    indices_tuple is whatever the miner returned. It is passed through
    pytorch_metric_learning's own lmu.convert_to_triplets, so a 3-tuple from a
    triplet miner and a 4-tuple from a pair miner are handled by exactly the
    same rule the library already applies inside TripletMarginLoss: pairs are
    joined on a shared anchor row, and pairs whose anchor appears on only one
    side are dropped.

    Args:
        margin          : SQUARED-EUCLIDEAN margin. Pass 2.0 * m_cos.
        alpha_deg       : angular constraint in degrees, in (0, 90).
        use_angular     : include l_ang.
        strict_semihard : apply the filter above.
        swap            : use min(D_an, D_pn) in the margin term.
        reduce_nonzero  : divide by |T_active| (matches AvgNonZeroReducer)
                          rather than by |T_strict|.
    """

    def __init__(self, margin, alpha_deg=18.0, use_angular=True,
                 strict_semihard=True, swap=True, reduce_nonzero=True):
        super().__init__()
        margin = float(margin)
        alpha_deg = float(alpha_deg)
        if margin <= 0.0:
            raise ValueError("margin must be > 0 (squared-Euclidean); got %r"
                             % (margin,))
        if not (0.0 < alpha_deg < 90.0):
            raise ValueError("alpha_deg must lie in (0, 90); got %r"
                             % (alpha_deg,))
        self.margin = margin
        self.alpha_deg = alpha_deg
        self.four_tan2 = 4.0 * math.tan(math.radians(alpha_deg)) ** 2
        self.use_angular = bool(use_angular)
        self.strict_semihard = bool(strict_semihard)
        self.swap = bool(swap)
        self.reduce_nonzero = bool(reduce_nonzero)

        self.register_buffer("last_n_mined", torch.zeros((), dtype=torch.long))
        self.register_buffer("last_n_strict", torch.zeros((), dtype=torch.long))
        self.register_buffer("last_n_active", torch.zeros((), dtype=torch.long))

    def forward(self, emb, labels, indices_tuple):
        a, p, n = lmu.convert_to_triplets(indices_tuple, labels)
        z = emb.float()

        d_ap = ((z[a] - z[p]) ** 2).sum(dim=1)
        d_an = ((z[a] - z[n]) ** 2).sum(dim=1)
        d_pn = ((z[p] - z[n]) ** 2).sum(dim=1)

        neg = torch.minimum(d_an, d_pn) if self.swap else d_an
        l_trip = torch.relu(d_ap - neg + self.margin)

        if self.use_angular:
            x_c = 0.5 * (z[a] + z[p])
            d_nc = ((z[n] - x_c) ** 2).sum(dim=1)
            l_ang = torch.relu(d_ap - self.four_tan2 * d_nc)
        else:
            l_ang = torch.zeros_like(l_trip)

        if self.strict_semihard:
            hi = d_ap + self.margin
            keep = (d_ap < d_an) & (d_an < hi) & (d_ap < d_pn) & (d_pn < hi)
        else:
            keep = torch.ones_like(d_ap, dtype=torch.bool)

        # the filter is applied as a WEIGHT, not by boolean indexing, so no
        # extra nonzero()/device sync is introduced
        w = keep.to(z.dtype)
        per = (l_trip + l_ang) * w

        n_strict = keep.sum()
        n_active = ((per > 0.0) & keep).sum()
        denom = n_active if self.reduce_nonzero else n_strict
        loss = per.sum() / denom.clamp_min(1).to(z.dtype)

        self.last_n_mined.fill_(a.numel())
        self.last_n_strict.copy_(n_strict.detach())
        self.last_n_active.copy_(n_active.detach())
        return loss

    def stats(self):
        """Device tensors for the per-epoch history entry. Read ONCE per epoch."""
        return {"n_mined": self.last_n_mined,
                "n_strict": self.last_n_strict,
                "n_active": self.last_n_active}


# --------------------------------------------------------------------------- #
# term 3: centroid separation (supervised, true labels only)
# --------------------------------------------------------------------------- #
class CentroidSeparationLoss(nn.Module):
    """Per-batch Gram penalty driving the class means to a simplex ETF.

        mu_c        = mean embedding of the rows of class c IN THIS BATCH
        mu_G        = mean of the mu_c over the classes present
        mu_hat_c    = mu_c / ||mu_c||_2      RAW directions, always
        mu_hat_c    = mu_dot_c / ||mu_dot_c||_2

        L_sep = mean over pairs c != c' of ( <mu_hat_c, mu_hat_c'> + 1/(K-1) )^2

    where K is the number of classes present in the batch with at least
    min_per_class rows. Zero exactly at the simplex ETF, strictly positive
    otherwise. The target -1/(K-1) is a geometric constant computed from K; it
    is never a hyperparameter.

    WHY THE MEANS ARE **NOT** CENTRED
    ---------------------------------
    An earlier version subtracted mu_G before normalising, on the grounds that
    NC2 is defined on globally centred class means. That was imported from a
    setting this pipeline is not in, and it BROKE the term.

    The defect: subtracting mu_G and then normalising makes the result
    invariant to BOTH translation and scale, so the penalty measures only the
    SHAPE of the simplex of class means and never its SIZE. Any equilateral
    arrangement satisfies it, including an arbitrarily tiny one. MEASURED on
    this implementation: three classes collapsed into a cap so tight that their
    raw pairwise cosine is +0.9994 -- essentially on top of one another --
    score L_sep = 0.000035 centred against 2.248 raw. Separation is precisely
    what centring discards.

    The raw form is NOT "ETF plus an extra constraint". For UNIT vectors {v_c}
    with all pairwise inner products equal to rho,

        || sum_c v_c ||^2  =  K + K(K-1) rho ,

    which at the ETF target rho = -1/(K-1) is exactly zero. Equiangularity at
    the target THEREFORE IMPLIES sum_c v_hat_c = 0: the two conditions are one
    package, and the raw form smuggles nothing in. On a sphere centred at the
    origin -- where L2-normalised embeddings live -- "the class directions
    balance about the origin" IS the statement that they are maximally spread,
    not a claim about where the cloud sits. mu_G is a nuisance parameter only
    when features carry an arbitrary offset, which is the NC2 literature's
    setting and not this one.

    This UNIFIES the C = 2 case rather than carving it out. At K = 2 every
    configuration is a degenerate equilateral one, so the centred form is
    identically zero everywhere: the same defect at its extreme. At K >= 3 the
    collapse is partial rather than total, which is worse, because it is silent
    instead of obvious.

    One thing this gets right that is easy to get wrong:

      * The means are SUPERVISED and per-batch. Class membership comes from the
        true condition labels only; destroyed surrogates are excluded by
        construction (see class_onehot). No running centroids are kept across
        batches.

    With the condition-balanced sampler every class is present in every batch,
    so K = n_classes normally; K is nonetheless recomputed per batch so a short
    or degenerate batch degrades to a smaller ETF rather than to nonsense.

    Args:
        n_classes     : C, the configured number of true condition classes.
        min_per_class : rows a class needs in a batch to enter the statistic.

    There is no centre_means argument. The centred formulation is not a
    supported option and its code path has been removed; see above.
    """

    def __init__(self, n_classes, min_per_class=2):
        super().__init__()
        n_classes = int(n_classes)
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2; got %r" % (n_classes,))
        self.n_classes = n_classes
        self.min_per_class = int(min_per_class)
        self.register_buffer("last_mean_cos",
                             torch.tensor(float("nan"), dtype=torch.float32))
        self.register_buffer("last_n_classes", torch.zeros((), dtype=torch.long))
        # retained and ALWAYS 0, so archived history readers keying on
        # "sep_centred" keep working; the centred form no longer exists
        self.register_buffer("last_centred", torch.zeros((), dtype=torch.long))

    def forward(self, emb, labels):
        z = emb.float()
        oh = class_onehot(labels, self.n_classes)               # (M, C)
        counts = oh.sum(dim=0)                                  # (C,)
        valid = counts >= float(self.min_per_class)             # (C,)
        n_valid = valid.sum()
        vf = valid.to(z.dtype).reshape(-1, 1)                   # (C, 1)

        mu = (oh.t() @ z) / counts.clamp_min(1.0).reshape(-1, 1)   # (C, E)
        # RAW class means, at every C. Centring would make the term invariant
        # to scale and so blind to collapse; see the class docstring.
        mu_dot = mu * vf
        mu_hat = mu_dot / mu_dot.norm(dim=1, keepdim=True).clamp_min(_EPS)
        gram = mu_hat @ mu_hat.t()                              # (C, C)

        eye = torch.eye(self.n_classes, dtype=torch.bool, device=z.device)
        pair = (valid.reshape(1, -1) & valid.reshape(-1, 1)) & (~eye)
        pw = pair.to(z.dtype)
        n_pairs = pw.sum().clamp_min(1.0)

        target = -1.0 / (n_valid.to(z.dtype) - 1.0).clamp_min(1.0)
        loss = (((gram - target) ** 2) * pw).sum() / n_pairs

        active = (n_valid >= 2).to(z.dtype)
        mean_cos = (gram * pw).sum().detach() / n_pairs
        self.last_mean_cos.copy_(
            torch.where(active > 0.0, mean_cos,
                        torch.full_like(mean_cos, float("nan"))))
        self.last_n_classes.copy_(n_valid.detach())
        return loss * active

    def stats(self):
        """Device tensors for the per-epoch history entry. Read ONCE per epoch."""
        return {"sep_mean_cos": self.last_mean_cos,
                "sep_n_classes": self.last_n_classes,
                "sep_centred": self.last_centred}


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #
class SilhouetteGate(nn.Module):
    """Latching gate on a running estimate of the TRAINING cosine silhouette.

    Per batch: compute the batch silhouette against the true labels, fold it
    into a running statistic that PERSISTS ACROSS EPOCH BOUNDARIES, and latch
    once that statistic reaches the threshold. Once latched the gate stays open
    for the rest of training; it never re-closes.

    The returned gate is a 0-dim tensor in {0.0, 1.0}, never a Python bool, so
    the caller can multiply by it without a device sync and the switch is exact
    at a batch boundary.

    Args:
        threshold   : running-silhouette level at which the gate latches.
                      threshold = None means "always open", which is the
                      control arm with no gate at all.
        n_classes   : C, for the true-label one-hot.
        momentum    : EMA coefficient in (0, 1]. None means a cumulative mean
                      over every batch seen. The EMA is the default because a
                      cumulative mean carries the earliest, worst batches at
                      full weight forever and so latches very late or never.
        min_batches : batches that must be seen before latching is permitted.
                      Guards against latching on one lucky early batch, which
                      matters at 9 rows per class.
    """

    def __init__(self, threshold, n_classes, momentum=0.05, min_batches=20,
                 min_per_class=2):
        super().__init__()
        self.always_open = threshold is None
        self.threshold = float("-inf") if threshold is None else float(threshold)
        if momentum is not None:
            momentum = float(momentum)
            if not (0.0 < momentum <= 1.0):
                raise ValueError("momentum must lie in (0, 1] or be None; got %r"
                                 % (momentum,))
        self.momentum = momentum
        self.n_classes = int(n_classes)
        self.min_batches = int(min_batches)
        self.min_per_class = int(min_per_class)

        self.register_buffer("stat",
                             torch.tensor(float("nan"), dtype=torch.float32))
        self.register_buffer("run_sum", torch.zeros((), dtype=torch.float32))
        self.register_buffer("n_seen", torch.zeros((), dtype=torch.long))
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.register_buffer("latched", torch.zeros((), dtype=torch.float32))
        self.register_buffer("latch_step",
                             torch.full((), -1, dtype=torch.long))

    @torch.no_grad()
    def forward(self, emb, labels):
        self.step.add_(1)
        if self.always_open:
            self.latched.fill_(1.0)
            return self.latched

        s = batch_cosine_silhouette(emb, labels, self.n_classes,
                                    self.min_per_class).to(torch.float32)
        valid = torch.isfinite(s)
        s_safe = torch.where(valid, s, torch.zeros_like(s))

        self.n_seen.add_(valid.to(torch.long))
        if self.momentum is None:
            self.run_sum.add_(s_safe)
            new_stat = self.run_sum / self.n_seen.clamp_min(1).to(torch.float32)
        else:
            first = ~torch.isfinite(self.stat)
            blended = (1.0 - self.momentum) * torch.where(
                first, s_safe, self.stat) + self.momentum * s_safe
            new_stat = torch.where(first, s_safe, blended)
        self.stat.copy_(torch.where(valid, new_stat, self.stat))

        ready = self.n_seen >= self.min_batches
        crossed = (valid & ready & torch.isfinite(self.stat)
                   & (self.stat >= self.threshold))
        newly = crossed & (self.latched < 0.5)
        self.latch_step.copy_(torch.where(newly, self.step, self.latch_step))
        self.latched.copy_(torch.maximum(self.latched, crossed.to(torch.float32)))
        return self.latched

    def stats(self):
        """Device tensors for the per-epoch history entry. Read ONCE per epoch."""
        return {"sil_running": self.stat,
                "sep_active": self.latched,
                "latch_step": self.latch_step,
                "gate_batches_seen": self.n_seen}


# --------------------------------------------------------------------------- #
# the warm-up (replaces the gate)
# --------------------------------------------------------------------------- #
def sep_warmup_scale(step, total_steps, warmup_frac):
    """The dimensionless ramp factor g(t) in [0, 1], as a PLAIN PYTHON float.

        g(t) = min(1, t / (tau * T)),    tau > 0
        g(t) = 1                          tau = 0   (no warm-up at all)

    so that lambda_sep(t) = lambda_sep * g(t).

    Pure, importable without a config and without a trainer, and it is what the
    smoke test checks: the module below must agree with it at every step.

    Arguments
    ---------
    step        : t, the number of optimiser steps COMPLETED so far, t >= 0.
                  The first batch of a run passes t = 0, so g(0) = 0 exactly.
    total_steps : T = max_epochs * batches_per_epoch, the PLANNED budget.
    warmup_frac : tau in [0, 1]. Full weight is reached at t = tau * T.

    Boundary cases, all deliberate:
      tau = 0  -> constant 1. This is the CONTROL ARM and the default, and it
                  reproduces the pre-existing ungated behaviour exactly.
      tau = 1  -> the ramp finishes exactly at the last planned step.
      T <= 0   -> constant 1. A ramp needs a horizon; without one the only
                  honest thing is to apply the full weight rather than to
                  silently spend the whole run near zero.
    """
    t = int(step)
    T = int(total_steps)
    tau = float(warmup_frac)
    if t < 0:
        raise ValueError("step must be >= 0; got %r" % (step,))
    if not (0.0 <= tau <= 1.0):
        raise ValueError("warmup_frac (tau) must lie in [0, 1]; got %r"
                         % (warmup_frac,))
    if tau == 0.0 or T <= 0:
        return 1.0
    w = tau * float(T)                      # the ramp length, in steps
    if t >= w:
        return 1.0
    return float(t) / w


class SepWarmup(nn.Module):
    """Deterministic linear ramp on the separation weight. Drop-in for the gate.

    Deliberately the SAME SHAPE as SilhouetteGate: it owns a step counter in a
    registered buffer, it returns a 0-dim tensor the caller multiplies by, and
    it exposes .stats() as device tensors to be read once per epoch. That is
    what lets CompositeDSNLoss.forward change by one line.

    READING THE HISTORY (a plus-or-minus-one worth knowing). The scale is
    computed for the CURRENT step and the counter advances afterwards, so the
    value recorded at the END of epoch k is g(k * n_batches - 1), not
    g(k * n_batches): history lags the live state by exactly one step. With
    tau * T = 20 and n_batches = 5, epoch 4 therefore logs 0.95 and epoch 5 is
    the first to log 1.0, even though full weight is reached at step 20, the
    first batch of epoch 5. VERIFIED against a real run. Nothing is wrong with
    the schedule; the epoch boundary simply does not coincide with the ramp
    boundary. Read sep_step alongside sep_warmup_scale if the exact step
    matters.

    NO DEVICE SYNC. The ramp is computed with tensor ops on the step buffer;
    the ramp LENGTH tau * T is a Python float fixed at construction. Reading
    int(self.step) instead would force a device->host read on every batch and
    break the one-.item()-per-epoch discipline that smoke_test_train [D-strict]
    guards.

    Args:
        total_steps : T = max_epochs * batches_per_epoch. None or <= 0 disables
                      the ramp (constant full weight) with a warning, since a
                      warm-up without a horizon is not well posed.
        warmup_frac : tau in [0, 1]. 0 (the default) means no warm-up.
    """

    def __init__(self, total_steps, warmup_frac=0.0):
        super().__init__()
        tau = float(warmup_frac)
        if not (0.0 <= tau <= 1.0):
            raise ValueError("warmup_frac (tau) must lie in [0, 1]; got %r"
                             % (warmup_frac,))
        T = 0 if total_steps is None else int(total_steps)
        if tau > 0.0 and T <= 0:
            warnings.warn(
                "SepWarmup: warmup_frac=%g was requested but total_steps=%r, "
                "so there is no horizon to ramp over -> the weight is constant "
                "at full lambda_sep. Pass total_steps = max_epochs * "
                "batches_per_epoch." % (tau, total_steps), RuntimeWarning)
        self.total_steps = T
        self.warmup_frac = tau
        # ramp length in steps, as a PYTHON float: multiplying a device tensor
        # by it introduces no sync
        self.warmup_steps = (tau * float(T)) if (tau > 0.0 and T > 0) else 0.0

        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.register_buffer("scale", torch.zeros((), dtype=torch.float32))

    @torch.no_grad()
    def forward(self):
        """g(t) for the CURRENT step, then advance. Returns a 0-dim tensor."""
        if self.warmup_steps <= 0.0:
            self.scale.fill_(1.0)
        else:
            t = self.step.to(torch.float32)
            self.scale.copy_(torch.clamp(t / self.warmup_steps, max=1.0))
        self.step.add_(1)
        return self.scale

    def stats(self):
        """Device tensors for the per-epoch history entry. Read ONCE per epoch.

        sep_warmup_steps and sep_total_steps are Python floats, not tensors:
        they are constants of the run, so reading them costs nothing.
        """
        return {"sep_warmup_scale": self.scale,
                "sep_step": self.step,
                "sep_warmup_steps": float(self.warmup_steps),
                "sep_total_steps": float(self.total_steps)}


# --------------------------------------------------------------------------- #
# the composite
# --------------------------------------------------------------------------- #
class CompositeDSNLoss(nn.Module):
    """L(t) = L_joint + lambda_sep(t) * L_sep.

    L_sep is evaluated on EVERY batch and multiplied by the ramp factor, rather
    than being skipped behind a Python branch. That costs one C x C Gram per
    batch -- negligible at C = 3 -- and buys no device sync and a weight that is
    exact at every batch boundary.

    THE GATE IS GONE. This class no longer constructs a SilhouetteGate; the
    separation term is scheduled deterministically by SepWarmup instead (see the
    module header for why, and for what was measured). SilhouetteGate remains
    defined and exported for the archived analysis tooling that reads its
    buffers; it is simply not on this path.

    Call signature mirrors the current trainer:  loss_fn(Z, y, pairs).

    Args:
        n_classes       : C, for the ETF target -1/(K-1) and the class masking.
        margin          : SQUARED-EUCLIDEAN margin. Pass 2.0 * m_cos.
        alpha_deg       : angular constraint in degrees, in (0, 90).
        lambda_sep      : the ASYMPTOTIC weight on L_sep, reached at t = tau*T.
        total_steps     : T = max_epochs * batches_per_epoch, the planned budget.
        warmup_frac     : tau in [0, 1]. 0 (default) = full weight from step 0,
                          which reproduces the pre-existing ungated behaviour.
        There is no centre_means argument: the class means are never centred.
    """

    def __init__(self, n_classes, margin, alpha_deg=18.0, lambda_sep=0.1,
                 total_steps=None, warmup_frac=0.0, use_angular=True,
                 strict_semihard=True, swap=True, reduce_nonzero=True,
                 min_per_class=2):
        super().__init__()
        lambda_sep = float(lambda_sep)
        if lambda_sep < 0.0:
            raise ValueError("lambda_sep must be >= 0; got %r" % (lambda_sep,))
        self.lambda_sep = lambda_sep
        self.joint = JointTripletLoss(
            margin=margin, alpha_deg=alpha_deg, use_angular=use_angular,
            strict_semihard=strict_semihard, swap=swap,
            reduce_nonzero=reduce_nonzero)
        self.sep = CentroidSeparationLoss(n_classes=n_classes,
                                          min_per_class=min_per_class)
        self.warmup = SepWarmup(total_steps=total_steps,
                                warmup_frac=warmup_frac)

    def forward(self, emb, labels, indices_tuple):
        joint = self.joint(emb, labels, indices_tuple)
        scale = self.warmup()                  # g(t) in [0, 1], no grad, no sync
        sep = self.sep(emb, labels)
        return joint + self.lambda_sep * scale * sep

    def stats(self):
        """Everything the per-epoch history entry needs, as device tensors.

        Read ONCE per epoch, at the epoch boundary, to preserve the
        one-.item()-per-epoch discipline. sep_lambda_t is the WEIGHT that was
        in force at the end of the epoch, which is what makes the ramp auditable
        after the fact rather than merely intended.
        """
        out = {}
        out.update(self.joint.stats())
        out.update(self.sep.stats())
        out.update(self.warmup.stats())
        out["sep_lambda_t"] = self.lambda_sep * self.warmup.scale
        return out
