"""
dsn_joint_loss.py
=================

Composite metric-learning objective for the Deep Summary Network.

    L = L_joint  +  lambda_sep * g_t * L_sep

where

    L_joint = ( 1 / |T_active| ) * sum_{(a,p,n) in T_strict} ( l_trip + l_ang )
    L_sep   = per-batch Gram penalty driving the class means to a simplex
              equiangular tight frame (ETF). At C >= 3 the means are CENTRED;
              at C = 2 they are RAW -- see "The two-class case" below.
    g_t     = a LATCHING gate in {0, 1}, driven by a running estimate of the
              TRAINING cosine silhouette

The two-class case
------------------
At C = 2 the CENTRED class means are always exactly antipodal, because
mu_dot_1 = -mu_dot_2 is forced by the centring itself. Their cosine is then
identically -1, which is also the ETF target -1/(C-1) = -1, so the centred
form of L_sep is IDENTICALLY ZERO for every embedding, good or bad. It is not
wrong, it is vacuous: with only two points there is no arrangement to
constrain, only a separation.

The fix is to drop the centring, not to change the target. Two antipodal unit
vectors ARE the simplex ETF for K = 2, so -1/(K-1) = -1 remains the correct
target; applied to the RAW normalised class means it is the ordinary statement
"push the two class means to opposite poles", i.e. maximise the cosine
distance between them. The functional form is unchanged:

    L_sep = mean over pairs c != c' of ( <mu_hat_c, mu_hat_c'> + 1/(K-1) )^2

with mu_hat_c built from mu_c - mu_G at C >= 3 and from mu_c at C = 2.

The switch is keyed on the CONFIGURED n_classes, which is fixed for a run, and
NOT on the number of classes present in a given batch. A C = 3 run that
transiently loses a class keeps the centred form and contributes zero for that
batch, exactly as before; it does not silently switch objectives mid-training
on a data-dependent condition. Watch sep_n_classes in the census for that case.

One property is lost at C = 2 and is worth knowing: the centred form is
invariant to a rigid translation of the whole batch, the raw form is not.
Since the backbone L2-normalises every row onto the sphere, translation is not
a symmetry of the representation anyway, but the two formulations are not
interchangeable and the census records which one ran.

Before the gate latches the objective is the joint triplet loss alone; from the
batch at which it latches onward, the centroid-separation term is included. The
gate is evaluated per batch, so the switch can and does happen MID-EPOCH.

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

Consequence of a trajectory-dependent gate (recorded, not argued)
-----------------------------------------------------------------
Because the latch epoch is a function of the training trajectory, different
seeds and different search trials switch objectives at different points. Seed
std therefore mixes optimisation noise with switch timing, and a search over
alpha compares trials that did not share an objective for the same number of
batches. Log latch_step and read it alongside any alpha ranking.

Checkpoint note
---------------
The gate state (running statistic, counter, latch) lives in registered buffers.
If training is resumed from a checkpoint that does not carry this module's
state_dict, the gate resets and will re-warm from scratch.

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
        mu_dot_c    = mu_c - mu_G      (C >= 3)   or   mu_c      (C = 2)
        mu_hat_c    = mu_dot_c / ||mu_dot_c||_2

        L_sep = mean over pairs c != c' of ( <mu_hat_c, mu_hat_c'> + 1/(K-1) )^2

    where K is the number of classes present in the batch with at least
    min_per_class rows. Zero exactly at the simplex ETF, strictly positive
    otherwise. The target -1/(K-1) is a geometric constant computed from K; it
    is never a hyperparameter.

    Three things this gets right that are easy to get wrong:

      * At C >= 3 the means are CENTRED. Neural collapse NC2 is a statement
        about mu_c - mu_G. Penalising the raw class means toward -1/(K-1)
        would silently impose the extra, unstated constraint mu_G -> 0.
      * At C = 2 the means are RAW. Centring two means makes them antipodal by
        construction, so the centred form is identically zero for every
        embedding. Dropping the centring turns the same expression into
        "drive the cosine between the two class means to -1", i.e. maximise
        the cosine distance between them. See the module header.
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
        centre_means  : None (default) selects centring automatically, i.e.
                        True at C >= 3 and False at C = 2. Pass True or False
                        to force one formulation, which is useful only for
                        testing that the two differ.
    """

    def __init__(self, n_classes, min_per_class=2, centre_means=None):
        super().__init__()
        n_classes = int(n_classes)
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2; got %r" % (n_classes,))
        self.n_classes = n_classes
        self.min_per_class = int(min_per_class)
        # keyed on the CONFIGURED class count, so the formulation is fixed for
        # the whole run and never switches on a data-dependent condition
        self.centre_means = (n_classes >= 3) if centre_means is None \
            else bool(centre_means)
        self.register_buffer("last_mean_cos",
                             torch.tensor(float("nan"), dtype=torch.float32))
        self.register_buffer("last_n_classes", torch.zeros((), dtype=torch.long))
        self.register_buffer(
            "last_centred",
            torch.tensor(1 if self.centre_means else 0, dtype=torch.long))

    def forward(self, emb, labels):
        z = emb.float()
        oh = class_onehot(labels, self.n_classes)               # (M, C)
        counts = oh.sum(dim=0)                                  # (C,)
        valid = counts >= float(self.min_per_class)             # (C,)
        n_valid = valid.sum()
        vf = valid.to(z.dtype).reshape(-1, 1)                   # (C, 1)

        mu = (oh.t() @ z) / counts.clamp_min(1.0).reshape(-1, 1)   # (C, E)
        if self.centre_means:
            mu_g = (mu * vf).sum(dim=0, keepdim=True) \
                / n_valid.clamp_min(1).to(z.dtype)
            mu_dot = (mu - mu_g) * vf
        else:
            # C = 2: centring would force the pair antipodal and make the term
            # vacuous, so the RAW class means carry the separation
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
# the composite
# --------------------------------------------------------------------------- #
class CompositeDSNLoss(nn.Module):
    """L = L_joint + lambda_sep * g_t * L_sep.

    L_sep is evaluated on every batch and multiplied by the gate tensor, rather
    than being skipped behind a Python branch. That costs one C x C Gram per
    batch -- negligible at C = 3 -- and buys two things: no device sync, and a
    switch that is exact at the batch where the gate latches.

    Call signature mirrors the current trainer:  loss_fn(Z, y, pairs).
    """

    def __init__(self, n_classes, margin, alpha_deg=18.0, lambda_sep=0.1,
                 gate_threshold=None, use_angular=True, strict_semihard=True,
                 swap=True, reduce_nonzero=True, gate_momentum=0.05,
                 gate_min_batches=20, min_per_class=2, centre_means=None):
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
                                          min_per_class=min_per_class,
                                          centre_means=centre_means)
        self.gate = SilhouetteGate(threshold=gate_threshold,
                                   n_classes=n_classes,
                                   momentum=gate_momentum,
                                   min_batches=gate_min_batches,
                                   min_per_class=min_per_class)

    def forward(self, emb, labels, indices_tuple):
        joint = self.joint(emb, labels, indices_tuple)
        gate = self.gate(emb, labels)          # {0, 1}, no grad, no sync
        sep = self.sep(emb, labels)
        return joint + self.lambda_sep * gate * sep

    def stats(self):
        """Everything the per-epoch history entry needs, as device tensors.

        Read ONCE per epoch, at the epoch boundary, to preserve the
        one-.item()-per-epoch discipline.
        """
        out = {}
        out.update(self.joint.stats())
        out.update(self.sep.stats())
        out.update(self.gate.stats())
        return out
