# The Joint Condition Search: Theory of a Conditional Bayesian Search Space with a Deterministically Scheduled Composite Objective

**Author:** prepared for the Deep Summary Network project
**Date:** 3 August 2026 — **Revision 2**

> **Revision 2 supersedes a wrong argument in Revision 1.** §3.7.4 previously
> claimed that the uncentred (raw) form of the separation term imposes "the ETF
> condition *plus* the unstated constraint $\mu_G \to 0$". That is false. For
> unit vectors, equiangularity at $-1/(K-1)$ *already implies* the vectors sum
> to zero — the two are one condition, not two. The centring has been **removed
> from the code entirely**, the search axis $\kappa$ (`sep_centre_means`) has
> been **deleted**, and the space became **17 axes / 21 columns**. §3.7.3–§3.7.4
>
> **Revision 3:** $\tau$ (`sep_warmup_frac`) has been **added** as axis 18, so
> the space is now **18 axes / 22 columns**. §3.6.2 is superseded: its dose
> integral is correct but the ridge it infers does not close, because equal
> dose does not imply equal terminal weight. The upper bound of the new axis is
> **derived**, $\tau_{\max} = \min(1, P/E_{\max})$. Sentences below that read
> "18 axes / 22 columns became 17 / 21" describe the Revision-2 *transition* and
> are left as written; the live counts are 18 / 22.
> are rewritten below; §3.5's coverage arithmetic is unaffected, since $\kappa$
> was never part of the cell definition.
**Repository / branch:** `Leonardodm00/Deep-Summary-Network`, `feat/composite-dsn-loss`, commit `fab2053`
**Predecessor documents:** `HANDOFF_joint_gp_search.md` (design), `HANDOFF_factorial52_analysis.md` (screening)
**Companion document:** `Main/Documentation/CHANGES_joint_condition_search.md` (implementation record)

## Abstract

The scientific question behind this work is whether a composite metric-learning
objective — a triplet hinge with an angular constraint, optionally augmented by
a Neural-Collapse-inspired centroid-separation term — outperforms a plain
triplet baseline on a three-class synthetic latent benchmark, and under which
mining strategy, triplet-filter setting and head geometry. The engineering
question, which this document treats theoretically, is how to *search* for that
answer when four of the factors are categorical, when some of their
combinations are not merely uninteresting but **provably degenerate**, and when
the budget permits only one seed per configuration.

This document establishes the mathematics of the search object that was built.
It covers: the passage from a partitioned factorial to a single conditional
measure on a product space; the legality projection $\Pi$ and why a projection
is the correct device where a penalty is not; the activity mask $A(\ell)$,
inactive coordinates, and the identifiability argument that forbids searching
two particular hyper-parameters together; the encoding and the exact
axis-versus-column arithmetic; an **exact expression for the expected coverage
of the induced cell partition as a function of the random-design size**,
validated against the sampler; the deterministic warm-up that replaces a
data-dependent gate, with its ridge argument and its epoch-boundary lag; a
proof that the centroid-separation term is **near-vacuous at three classes** in
its centred form, and the sense in which the uncentred form is a different
objective rather than a corrected one; the scoring objective under a single
seed, including the misranking probability and the exact reason a variance
estimator's degrees-of-freedom choice is load-bearing; and the cost model that
converts a measured architecture size mix into a wall-clock estimate, together
with the condition under which that estimate is admissible.

**Deliberately excluded:** the screening-tier empirical findings (predecessor
handoff); the data-generating process of the synthetic benchmark; the backbone
architecture, which enters only through its parameter count; and any claim
about which objective wins, which is what the search exists to determine.

---

## 1. Notation and Symbols

Every symbol appearing anywhere in this document, including indices,
decorations and operators.

| Symbol | Name / Meaning | Type & domain | Units | First used in § |
|---|---|---|---|---|
| $C$ | number of true condition classes in the benchmark | $C \in \mathbb{N}$, $C \ge 2$; here $C = 3$ | dimensionless | §3.7 |
| $K$ | number of classes present in a batch with at least $n_{\min}$ rows; $K \le C$ | $K \in \mathbb{N}$, $K \ge 2$ | dimensionless | §3.7 |
| $c, c'$ | class indices | $c, c' \in \{1, \dots, C\}$ | dimensionless | §3.7 |
| $n_{\min}$ | rows a class needs in a batch to enter a batch statistic | $n_{\min} \in \mathbb{N}$, here $2$ | rows | §3.7 |
| $E$ | embedding dimension | $E \in \mathbb{N}$, searched over $[8, 16]$ | dimensionless | §3.7 |
| $M$ | rows in one batch | $M \in \mathbb{N}$ | rows | §3.7 |
| $z_i$ | embedding of row $i$ of a batch, unit-norm | $z_i \in \mathbb{S}^{E-1} \subset \mathbb{R}^{E}$ | dimensionless | §3.7 |
| $\mu_c$ | batch mean embedding of class $c$; **not** re-normalised | $\mu_c \in \mathbb{R}^{E}$ | dimensionless | §3.7 |
| $\mu_G$ | mean of the class means, $\mu_G = \tfrac{1}{K}\sum_{c=1}^{K}\mu_c$ | $\mu_G \in \mathbb{R}^{E}$ | dimensionless | §3.7 |
| $\hat\mu_c$ | unit-norm class direction; **two definitions**, selected by $\kappa$ | $\hat\mu_c \in \mathbb{S}^{E-1}$ | dimensionless | §3.7 |
| $\kappa$ | `sep_centre_means`: selects centred ($\kappa = 1$) or raw ($\kappa = 0$) class directions | $\kappa \in \{0, 1\}$, or `None` for the automatic rule | dimensionless | §3.7 |
| $\rho_{\mathrm{ETF}}$ | simplex-ETF target cosine, $\rho_{\mathrm{ETF}} = -1/(K-1)$ | $\rho_{\mathrm{ETF}} \in (-1, 0)$ | dimensionless | §3.7 |
| $\mathcal{L}_{\mathrm{sep}}$ | centroid-separation (NC2-inspired) loss term | $\mathcal{L}_{\mathrm{sep}} \in [0, \infty)$ | loss units | §3.7 |
| $\mathcal{L}_{\mathrm{joint}}$ | triplet hinge plus angular hinge, before $\mathcal{L}_{\mathrm{sep}}$ | $\mathcal{L}_{\mathrm{joint}} \in [0, \infty)$ | loss units | §3.6 |
| $\mathcal{L}(t)$ | total loss at optimiser step $t$ | $\mathcal{L}(t) \in [0, \infty)$ | loss units | §3.6 |
| $\lambda_{\mathrm{sep}}$ | asymptotic weight on $\mathcal{L}_{\mathrm{sep}}$; a **searched** axis | $\lambda_{\mathrm{sep}} \in [10^{-2}, 20]$, log-uniform prior | dimensionless | §3.6 |
| $\lambda_{\mathrm{sep}}(t)$ | warm-up-scaled weight in force at step $t$ | $\lambda_{\mathrm{sep}}(t) \in [0, \lambda_{\mathrm{sep}}]$ | dimensionless | §3.6 |
| $g(t)$ | dimensionless ramp factor, $\lambda_{\mathrm{sep}}(t) = \lambda_{\mathrm{sep}}\, g(t)$ | $g(t) \in [0, 1]$ | dimensionless | §3.6 |
| $t$ | global optimiser-step index, **steps completed**, persisting across epochs | $t \in \{0, 1, \dots, T\}$ | steps | §3.6 |
| $T$ | planned total steps, $T = E_{\max}\, n_{\mathrm{b}}$ | $T \in \mathbb{N}$ | steps | §3.6 |
| $E_{\max}$ | `max_epochs`, the epoch cap | $E_{\max} \in \mathbb{N}$ | epochs | §3.6 |
| $n_{\mathrm{b}}$ | `batches_per_epoch`, optimiser steps per epoch | $n_{\mathrm{b}} \in \mathbb{N}$ | steps/epoch | §3.6 |
| $k$ | epoch index | $k \in \{1, \dots, E_{\max}\}$ | epochs | §3.6 |
| $\tau$ | warm-up fraction; $\lambda_{\mathrm{sep}}(t)$ reaches full value at $t = \tau T$. **Fixed, not searched** | $\tau \in [0, 1]$, here $0.3$ | dimensionless | §3.6 |
| $P$ | `patience`, epochs without improvement before early stopping | $P \in \mathbb{N}$ | epochs | §3.9 |
| $m_{\cos}$ | triplet margin in cosine-distance units | $m_{\cos} \in [0.1, 1.0]$ | cosine distance | §3.3 |
| $\alpha$ | angular-constraint half-angle | $\alpha \in [2°, 20°]$ | degrees | §3.3 |
| $D_{ap}, D_{an}, D_{pn}$ | squared-Euclidean distances anchor-positive, anchor-negative, positive-negative | $\in [0, 4]$ on $\mathbb{S}^{E-1}$ | squared distance | §3.2 |
| $\ell$ | the `loss_type` categorical value | $\ell \in \mathcal{T} = \{\texttt{triplet}, \texttt{joint}, \texttt{joint\_sep}\}$ | — | §3.1 |
| $m$ | the `mining_strategy` categorical value | $m \in \mathcal{M} = \{\texttt{hard}, \texttt{easy\_positive}, \texttt{easy\_pos\_semihard\_neg}\}$ | — | §3.1 |
| $s$ | the `strict_semihard` binary value | $s \in \{0, 1\}$ | — | §3.1 |
| $h$ | head geometry, the pair (`head_fusion`, `head_pool_ops`) | $h \in \mathcal{H}$, $\lvert\mathcal{H}\rvert = 4$ | — | §3.1 |
| $\omega$ | a **condition**, $\omega = (m, \ell, s)$ | $\omega \in \mathcal{M} \times \mathcal{T} \times \{0,1\}$ | — | §3.2 |
| $\mathcal{W}$ | the set of raw conditions, $\lvert\mathcal{W}\rvert = 18$ | finite set | — | §3.2 |
| $\mathcal{W}^{*}$ | the set of **legal** conditions, $\lvert\mathcal{W}^{*}\rvert = 13$ | $\mathcal{W}^{*} \subset \mathcal{W}$ | — | §3.2 |
| $\Pi$ | legality projection, $\Pi : \mathcal{X} \to \mathcal{X}$ | idempotent map | — | §3.2 |
| $\mathcal{C}$ | the set of **cells**, $\mathcal{C} = \mathcal{W}^{*} \times \mathcal{H}$, $\lvert\mathcal{C}\rvert = 52$ | finite set | — | §3.5 |
| $\gamma_j$ | the $j$-th cell of $\mathcal{C}$ | $\gamma_j \in \mathcal{C}$, $j \in \{1, \dots, 52\}$ | — | §3.5 |
| $p_j$ | probability that one random draw lands in cell $\gamma_j$ after $\Pi$ | $p_j \in (0, 1)$, $\sum_j p_j = 1$ | dimensionless | §3.5 |
| $A(\ell)$ | activity mask: the loss hyper-parameters that $\ell$ reads | $A(\ell) \subseteq \mathcal{P}$ | — | §3.3 |
| $\mathcal{P}$ | the loss-hyper-parameter superset, $\lvert\mathcal{P}\rvert = 4$ | finite set | — | §3.3 |
| $x$ | a point of the search space (one GP trial) | $x \in \mathcal{X}$, the 17-axis space of §3.4 | mixed | §3.1 |
| $x^{(j)}$ | the $j$-th coordinate of $x$; parenthesised superscript, **never a power** | see §3.4 | mixed | §3.4 |
| $d_{\mathrm{ax}}$ | declared axis count | $d_{\mathrm{ax}} = 18$ | dimensionless | §3.4 |
| $d_{\mathrm{col}}$ | surrogate-facing column count after one-hot expansion | $d_{\mathrm{col}} = 21$ | dimensionless | §3.4 |
| $N$ | number of sampled points in a dry run | $N \in \mathbb{N}$ | draws | §3.5 |
| $N_{\mathrm{init}}$ | size of the random initial design (`n_initial_points_joint`) | $N_{\mathrm{init}} \in \mathbb{N}$, here $100$ | trials | §3.5 |
| $N_{\mathrm{calls}}$ | GP trial budget (`n_calls_joint`) | $N_{\mathrm{calls}} \in \mathbb{N}$, here $300$ | trials | §3.5 |
| $N_{\mathrm{seeds}}$ | seeds per trial | $N_{\mathrm{seeds}} \in \mathbb{N}$, here $1$ | seeds | §3.8 |
| $U_N$ | number of cells unvisited after $N$ draws | $U_N \in \{0, \dots, 52\}$ | cells | §3.5 |
| $S_{\mathrm{val}}$ | validation cosine silhouette on held-out data | $S_{\mathrm{val}} \in [-1, 1]$ | dimensionless | §3.8 |
| $u, v$ | primary and secondary selection metrics, **by role** | $u, v \in [-1, 1]$ | dimensionless | §3.8 |
| $e^{*}$ | selected epoch index for a run | $e^{*} \in \{1, \dots, E_{\max}\}$ | epochs | §3.8 |
| $J_{\varepsilon}$ | the search objective (minimised) | $J_{\varepsilon} \in \mathbb{R}$ | dimensionless | §3.8 |
| $\varepsilon$ | tie-break weight | $\varepsilon \in [0, \infty)$, or undefined when disabled | dimensionless | §3.8 |
| $\gamma_{\mathrm{tb}}$ | tie-break coefficient (`tie_break_gamma`) | $\gamma_{\mathrm{tb}} \in [0, 1]$ | dimensionless | §3.8 |
| $\Delta_{\min}(y)$ | smallest strictly-positive gap below 1 that ARI can take on labels $y$ | $\Delta_{\min}(y) > 0$ | dimensionless | §3.8 |
| $y$ | validation label vector | $y \in \{1,\dots,C\}^{N_{\mathrm{eval}}}$ | — | §3.8 |
| $\sigma_{\mathrm{s}}$ | within-cell standard deviation of the selection metric across seeds | $\sigma_{\mathrm{s}} \ge 0$; measured $0.073$ | silhouette units | §3.8 |
| $\sigma_{\mathrm{b}}$ | between-cell standard deviation of cell means | measured $0.117$ | silhouette units | §3.8 |
| $d$ | true difference in the selection metric between two configurations | $d \in \mathbb{R}$ | silhouette units | §3.8 |
| $\Phi(\cdot)$ | standard normal cumulative distribution function | $\Phi : \mathbb{R} \to (0,1)$ | — | §3.8 |
| $F$ | the failure penalty (`FAILED_OBJECTIVE`) | $F = +1.0$, finite | dimensionless | §3.8 |
| $\theta$ | parameter count of a sampled backbone | $\theta \in \mathbb{N}$ | parameters | §3.9 |
| $\Theta$ | the random parameter count induced by sampling $x$ | $\Theta$ a random variable on $\mathbb{N}$ | parameters | §3.9 |
| $r(\theta)$ | cost model, seconds per epoch at parameter count $\theta$ | $r : \mathbb{N} \to (0,\infty)$ | s/epoch | §3.9 |
| $a, b$ | intercept and slope of the fitted cost model | $a > 0$, $b > 0$ | s/epoch, s/epoch/param | §3.9 |
| $R^2$ | coefficient of determination of that fit | $R^2 \le 1$ | dimensionless | §3.9 |
| $\eta$ | size-mix multiplier | $\eta > 0$ | dimensionless | §3.9 |
| $W$ | requested walltime | $W > 0$ | hours | §3.9 |
| $\beta$ | required unused fraction of the walltime (margin) | $\beta \in [0, 1)$, here $0.15$ | dimensionless | §3.9 |
| $\bar{e}$ | mean epochs actually run per trial | $\bar{e} \in (0, E_{\max}]$ | epochs | §3.9 |
| $\lVert\cdot\rVert_2$ | Euclidean norm | operator on $\mathbb{R}^{E}$ | — | §3.7 |
| $\langle\cdot,\cdot\rangle$ | Euclidean inner product | operator on $\mathbb{R}^{E}\times\mathbb{R}^{E}$ | — | §3.7 |
| $\hat{\ }$ (hat) | unit-normalisation of the decorated vector | decoration | — | §3.7 |
| $\mathbb{1}[\cdot]$ | indicator function | $\to \{0,1\}$ | — | §3.5 |
| $\mathbb{E}[\cdot]$ | expectation | operator | — | §3.5 |

### 1.1 Conventions

- Class indices run $c = 1, \dots, C$ and are never zero-based in the
  mathematics, although the code is zero-based; the two are distinguished by
  context and never mixed inside one expression.
- All embeddings are L2-normalised, so every inner product
  $\langle z_i, z_j \rangle$ is a cosine. "Distance" without qualification means
  cosine distance $1 - \langle z_i, z_j \rangle$. Squared-Euclidean distances,
  which is what the loss internals use, satisfy
  $\lVert z_i - z_j \rVert_2^2 = 2\,(1 - \langle z_i, z_j\rangle)$ and are always
  named as such.
- $\log$ without a base means the natural logarithm. "Log-uniform prior" refers
  to the search library's `prior="log-uniform"`, which is base-independent.
- Vectors are column vectors; $\mu_c \in \mathbb{R}^{E}$ is a point, not a row.
- "Step", "batch" and "optimiser step" are synonyms, counted by $t$. One epoch
  is $n_{\mathrm{b}}$ steps. **$t$ counts steps *completed*,** so the first
  batch of a run is evaluated at $t = 0$ (§3.6).
- "Axis" means a declared search dimension; "column" means a surrogate-facing
  dimension after one-hot expansion. The two differ and both are always
  reported (§3.4).
- Measured quantities are stated as measured and attributed. Anything inferred
  is labelled as inference in §5. Claims taken from a source are attributed at
  the point of use, and §6 states for each source whether its full text was
  read.

---

## 2. Glossary / Jargon

Ordered **by first appearance**, because the concepts build on each other
rather than being independent lookups.

**Metric learning.** Learning an embedding in which semantically similar inputs
lie close together and dissimilar inputs lie far apart, rather than learning a
classifier directly. Operative from §3.1.

**Factorial (screening) design.** The predecessor approach: enumerate every
legal combination of the categorical factors as a separate configuration file,
run each independently, compare. §3.1.

**Cell.** One combination of the categorical factors — condition plus head
geometry. There are 52. §3.1.

**Condition.** The triple $\omega = (m, \ell, s)$ of mining strategy, loss type
and strict-filter flag, excluding the head geometry. §3.2.

**Mining strategy.** The rule choosing which (anchor, positive, negative)
triples the loss is evaluated on. §3.2.

**Hard mining.** Selecting the triples that most violate the desired ordering,
i.e. those whose negative is closer to the anchor than the positive is. §3.2.

**Easy-positive mining.** Selecting, for each anchor, the *most similar*
same-class example as the positive. Its purpose is the opposite of hard
positive mining: it constrains only the nearest same-class neighbour and so
permits a class to occupy a manifold rather than contracting to a point
(Xuan, Stylianou and Pless; full text in the project knowledge base). §3.2.

**Semi-hard negative.** A negative that is farther from the anchor than the
positive is, but still inside the margin band. The source defines it by the
condition $d(f(x_a), f(x)) > d(f(x_a), f(x_p))$ together with the margin
constraint. This definition is what makes §3.2's emptiness proof exact. §3.2.

**Strict semi-hard filter.** A post-mining filter retaining only triples with
$D_{ap} < D_{an}$ (and symmetrically for $D_{pn}$) inside the margin band. Note
that the everyday reading of "strict" as merely "stricter" understates it:
combined with hard mining the surviving set is provably empty. §3.2.

**Provably empty cell.** A condition whose surviving triple set is empty for
every batch by construction, so training proceeds with identically zero loss
and no error is raised. §3.2.

**Legality projection ($\Pi$).** A deterministic idempotent map applied to a
sampled point *before* any configuration is constructed, sending illegal
combinations onto legal ones. §3.2.

**Idempotent.** Satisfying $\Pi \circ \Pi = \Pi$: applying the map twice gives
the same result as applying it once. §3.2.

**Pushforward measure.** The distribution induced on the target space by
applying a map to a random draw from the source space. What $\Pi$ does to the
sampling distribution, and the reason cell probabilities are unequal. §3.5.

**Inactive coordinate.** An axis present in every point $x$ but not read for
the sampled loss type, e.g. $m_{\cos}$ when $\ell = \texttt{joint}$. Required
because the optimiser demands a fixed-length vector. §3.3.

**Activity mask $A(\ell)$.** The set of loss hyper-parameters that loss type
$\ell$ actually reads. §3.3.

**Clamping.** Fixing an inactive coordinate to a constant before building the
configuration, so that points differing only in inactive coordinates produce
identical configurations. §3.3.

**Identifiability.** The property that distinct parameter values produce
distinguishable objective values. Two parameters that are not jointly
identifiable trace a *ridge*. §3.3.

**Ridge (in a search space).** A direction along which the objective is nearly
flat because two parameters trade off against each other, so trials spent
moving along it purchase nothing. §3.3.

**One-hot expansion.** The encoding of a $k$-level categorical axis as $k$
binary surrogate columns. Why axis count and column count differ. §3.4.

**ARD (automatic relevance determination).** A Gaussian-process
kernel parameterisation with one length-scale per input dimension, so that an
irrelevant dimension can be assigned a large length-scale and effectively
ignored. §3.3.

**Nugget / observation noise.** The Gaussian process's model of irreducible
noise in the observed objective at a repeated input. What absorbs single-seed
variance, at a cost in sample efficiency. §3.8.

**Random initial design.** The trials drawn quasi-randomly *before* the
surrogate is first fitted. Its size is $N_{\mathrm{init}}$. §3.5.

**Coverage.** The number of distinct cells visited by a set of draws. §3.5.

**Coupon-collector problem.** The classical question of how many draws are
needed to see every category at least once; the coverage analysis of §3.5 is
its unequal-probability variant.

**Neural Collapse (NC).** The empirical phenomenon, documented by Papyan, Han
and Donoho, in which a classifier's last-layer features and classifier
converge to a rigid, highly symmetric configuration during the terminal phase
of training. §3.7.

**Terminal phase of training (TPT).** The regime beginning at the epoch where
training error first reaches zero, during which the loss continues to be driven
toward zero. NC is a phenomenon *of* this phase. §3.7.

**NC2.** The specific sub-property that the globally centred class means
converge to a simplex equiangular tight frame. §3.7.

**Simplex equiangular tight frame (simplex ETF).** A configuration of $K$
vectors that are equinormed and equiangular with all pairwise cosines equal to
$-1/(K-1)$, the maximum separation achievable by centred equiangular vectors.
§3.7.

**Equinormness.** All centred class means having equal $\lVert\cdot\rVert_2$.
One of the conditions constituting NC2, and the pivotal quantity in the $C = 3$
degeneracy. §3.7.

**Equiangularity.** All pairwise angles between centred class means being
equal. §3.7.

**Latching silhouette gate.** The mechanism this design **removes**: a switch
turning $\mathcal{L}_{\mathrm{sep}}$ on permanently the first time a running
estimate of the training silhouette exceeded a threshold. §3.6.

**Warm-up.** Its replacement: a deterministic ramp of the weight from zero to
full value over the first $\tau T$ steps, independent of the data. §3.6.

**Silhouette coefficient.** A clustering-quality index in $[-1, 1]$ comparing,
for each point, the mean distance to its own cluster against the mean distance
to the nearest other cluster. Used here in its cosine form. §3.8.

**Adjusted Rand Index (ARI).** A chance-corrected measure of agreement between
two partitions of a set. Discrete-valued on a fixed label vector, which is why
$\Delta_{\min}(y)$ exists for it and not for the silhouette. §3.8.

**Lexicographic tie-break.** Using a secondary metric to order only those
configurations that the primary metric cannot separate. §3.8.

**Degrees of freedom (`ddof`).** The divisor correction in a variance
estimator: $\mathrm{ddof} = 0$ divides by $n$ (population), $\mathrm{ddof} = 1$
by $n - 1$ (sample). At $n = 1$ the latter is undefined. §3.8.

**Size mix.** The distribution of model sizes the search actually draws, as
opposed to the extreme corners of the architecture space. §3.9.

**Submission gate.** A predicate that must hold before a cluster job may be
submitted. §3.9.

---

## 3. Main body

### 3.1 From a partitioned factorial to a single conditional measure

*This section establishes what changed at the level of the object being
optimised, and why the change is not merely organisational.*

The predecessor design treated the four categorical factors as a **partition**.
Writing $\mathcal{C}$ for the set of cells, it solved, for each cell
$\gamma \in \mathcal{C}$ independently,

$$
x^{*}(\gamma) \;=\; \arg\min_{x \,\in\, \mathcal{X}_{\mathrm{cont}}} \; J\!\left(x \,\middle|\, \gamma\right),
\tag{1}
$$

where $\mathcal{X}_{\mathrm{cont}}$ is the continuous hyper-parameter space and
$J(\cdot \mid \gamma)$ is the objective **conditional on the cell**. Note the
conditioning bar: under the factorial, the objective was never a function of
$\gamma$; it was a family of functions indexed by $\gamma$, each estimated from
its own trials, sharing no information.

The present design replaces (1) with a single problem on the product space,

$$
x^{*} \;=\; \arg\min_{x \,\in\, \mathcal{X}} \; J(x), \qquad \mathcal{X} \;=\; \mathcal{X}_{\mathrm{cont}} \times \mathcal{C},
\tag{2}
$$

in which the cell is a coordinate of $x$ rather than an index on $J$.

Three consequences follow, and only the first is obvious.

**(i) Information is shared across cells.** In (1), a trial run in cell
$\gamma_1$ tells the search nothing about cell $\gamma_2$. In (2), the
surrogate's kernel couples them: two points that agree on the continuous
coordinates and differ in one categorical coordinate are near-neighbours, so an
observation at one informs the posterior at the other. This is the entire
mechanism by which 300 trials can say anything about 52 cells.

**(ii) The budget is allocated adaptively rather than uniformly.** The
factorial spent an equal share on every cell, including cells that are visibly
hopeless after a handful of trials. Formulation (2) concentrates trials where
the acquisition function expects improvement. This is a benefit and a cost;
the cost is stated squarely in §5, item 3, because it is the reason this
search does **not** answer the original scientific question.

**(iii) The separability assumption is abandoned deliberately.** A staged
search — architecture first, then optimiser, then loss — assumes that the best
architecture is the best architecture whatever it is later paired with. For
this benchmark that assumption is known to be false: the screening found that
head geometries differ chiefly in *generalisation gap*, and that the
interaction between the strict filter and the head was the largest effect
measured. A staged search would fix the head before the loss type existed as a
variable at all, and could not represent the interaction it most needs to
resolve.

The remainder of this document is concerned with what must be true of
$\mathcal{X}$, of the sampling measure on it, and of $J$, for (2) to be
well posed.

### 3.2 Degeneracy, and the legality projection $\Pi$

*This section establishes that the raw product space is not merely inefficient
but ill-posed, and constructs the correction.*

#### 3.2.1 The raw product contains a provably degenerate region

Consider the raw condition set
$\mathcal{W} = \mathcal{M} \times \mathcal{T} \times \{0,1\}$, of cardinality
$3 \times 3 \times 2 = 18$. Two sub-regions are pathological, for different
reasons.

**Inert region.** For $\ell = \texttt{triplet}$ the strict filter does not
exist in the loss at all: the flag is written to the configuration and never
read. Every point with $\ell = \texttt{triplet}$ therefore builds the same
trainer whatever $s$ is. This is wasteful but harmless in itself.

**Degenerate region.** For $\ell \in \{\texttt{joint}, \texttt{joint\_sep}\}$
with $m = \texttt{hard}$ and $s = 1$, the surviving triple set is **empty for
every batch**. The argument is short and worth stating exactly, because the
whole projection rests on it.

Hard mining returns precisely the triples whose negative is closer to the
anchor than the positive is, that is, for each returned triple $(a, p, n)$,

$$
D_{an} \;<\; D_{ap}.
\tag{3}
$$

The strict semi-hard filter retains precisely the triples satisfying, for each
retained triple $(a, p, n)$,

$$
D_{ap} \;<\; D_{an} \;<\; D_{ap} + 2 m_{\cos}
\quad\text{and}\quad
D_{ap} \;<\; D_{pn} \;<\; D_{ap} + 2 m_{\cos}.
\tag{4}
$$

The first conjunct of (4) is $D_{ap} < D_{an}$, which is the exact negation of
(3). Hence for every batch the intersection is empty, the retained set has
cardinity zero, and the reduced loss is identically $0$ with identically zero
gradient. The semi-hard condition used in (4) is the one given in the source
that introduced it in this form (Xuan, Stylianou and Pless, full text read),
where the semi-hard negative is defined by requiring the anchor-negative
distance to exceed the anchor-positive distance while remaining inside the
margin.

Empirically this is confirmed rather than merely argued: on one real batch,
16 814 triples were mined and **zero** survived the filter.

The danger is not that this region is bad. It is that it is **silently
stable**. A search sampling it receives a perfectly reproducible objective
value from a run that trained nothing, with no exception raised and no warning
emitted. The surrogate would then model that value as a genuine property of the
region.

#### 3.2.2 The projection

Define $\Pi : \mathcal{X} \to \mathcal{X}$ acting on the condition coordinates
of $x$ and leaving every other coordinate fixed, by

$$
\Pi\bigl(m, \ell, s\bigr) \;=\;
\begin{cases}
(m, \ell, 0) & \text{if } \ell = \texttt{triplet},\\[2pt]
(m, \ell, 0) & \text{if } \ell \ne \texttt{triplet} \text{ and } m = \texttt{hard},\\[2pt]
(m, \ell, s) & \text{otherwise.}
\end{cases}
\tag{5}
$$

Three properties, each of which is asserted by a test rather than assumed.

**Well-definedness and range.** $\Pi$ never alters $m$ or $\ell$, only $s$.
Therefore no trial is silently converted into a different experiment; only its
filter setting moves. Counting the image: the $\texttt{triplet}$ branch
contributes $\lvert\mathcal{M}\rvert = 3$ conditions; the $m = \texttt{hard}$
branch contributes $2$ (one per non-triplet loss type); the remaining branch
contributes $2 \text{ mining} \times 2 \text{ loss} \times 2 \text{ filter} = 8$.
Hence

$$
\lvert \mathcal{W}^{*} \rvert \;=\; 3 + 2 + 8 \;=\; 13,
\tag{6}
$$

and with $\lvert\mathcal{H}\rvert = 4$ head geometries,
$\lvert\mathcal{C}\rvert = 13 \times 4 = 52$, exactly the historical cell
count. This is verified against the 104 shipped configuration files of the two
factorial tiers, not merely against the generator that produced them.

**Idempotence.** For every $\omega \in \mathcal{W}$,
$\Pi(\Pi(\omega)) = \Pi(\omega)$, because the output always has $s = 0$ in the
two branches that modify $s$, and those branches are conditioned only on $m$
and $\ell$, which $\Pi$ does not change. Consequently $\Pi$ is a projection in
the algebraic sense, and $\mathcal{W}^{*} = \{\omega \in \mathcal{W} : \Pi(\omega) = \omega\}$
is exactly its fixed-point set. The implementation defines legality *as*
fixed-pointhood, so the two notions cannot drift apart.

**Order of application.** $\Pi$ acts on the *point*, before any configuration
object is constructed. This matters: the configuration constructor validates
its arguments and raises on the degenerate combination, so a projection applied
afterwards would be repairing an object that could not have been built.

#### 3.2.3 Why a projection and not a penalty

An alternative treatment is to sample the raw space and assign the degenerate
region a penalty value. This is **wrong here**, and the reason is worth
stating precisely because it is a statement about what the surrogate learns.

Let $J$ be the objective and suppose the degenerate region were assigned
$J = F$ (a large finite penalty). The surrogate then fits a function that is
large whenever $m = \texttt{hard}$ and $s = 1$. Because the kernel is smooth in
the categorical coordinates, that elevation *leaks* onto neighbouring points —
in particular onto $m = \texttt{hard}$ with $s = 0$, which is a perfectly good
condition. The search would learn "hard mining is bad" from evidence that says
only "hard mining with a filter that contradicts it produces no triples". The
penalty encodes a fact about an interaction as if it were a fact about a main
effect.

Under $\Pi$, by contrast, two distinct raw points may map to the same
configuration and hence to the **same** objective value. The surrogate receives
a *duplicated observation*, not a false one. Duplicate observations are exactly
what a Gaussian process with a noise term is built to handle: they sharpen the
posterior at that input rather than distorting it elsewhere. The cost is a
mild loss of sample efficiency, since some draws are spent re-measuring a point
already measured; the benefit is that nothing false is learned. The projection
is recorded per trial, so a duplicated observation is *readable as* a
projection rather than appearing as unexplained noise.

### 3.3 Inactive coordinates, the activity mask, and identifiability

*This section establishes how a fixed-length vector can encode a space in which
different points have different numbers of meaningful parameters.*

#### 3.3.1 The fixed-length constraint

Gaussian-process optimisation requires that every sampled point be a vector of
the same length. But the meaningful parameters genuinely differ by loss type.
Define the loss-hyper-parameter superset

$$
\mathcal{P} \;=\; \bigl\{\, m_{\cos},\; \alpha,\; \lambda_{\mathrm{sep}} \,\bigr\},
\tag{7}
$$

and the activity mask, for each $\ell \in \mathcal{T}$,

$$
A(\texttt{triplet}) = \{m_{\cos}\}, \quad
A(\texttt{joint}) = \{\alpha\}, \quad
A(\texttt{joint\_sep}) = \{\alpha,\; \lambda_{\mathrm{sep}}\}.
\tag{8}
$$

Every point carries all three coordinates of $\mathcal{P}$; for a given trial
only those in $A(\ell)$ are read.

#### 3.3.2 Clamping, and why the clamp constant is the base configuration

Let $x_1, x_2 \in \mathcal{X}$ agree on every coordinate except some
$p \notin A(\ell)$, where $\ell$ is the loss type both sampled. If the
configuration builder wrote $p$ regardless, the two points would build
*different* configurations that nonetheless train identically, and the search
would receive two observations of the same underlying value at two different
inputs — noise, from the surrogate's perspective, manufactured by the encoding.

The correction is to clamp: for every $p \notin A(\ell)$, the built
configuration takes a constant value for $p$, independent of $x$. The
implementation realises this by **leaving the base configuration's value
untouched**, which has three merits over inventing constants. It is uniform
(one rule, not four); the base configuration is fixed for the entire study, so
the value genuinely is constant; and it preserves the documented semantics that
under $\ell \in \{\texttt{joint}, \texttt{joint\_sep}\}$ the margin is *fixed*
rather than unused — the composite loss really does read $m_{\cos}$, through
the conversion $\text{margin}_{\text{sq}} = 2 m_{\cos}$.

There is a corollary the base configuration must respect: since it defines the
clamp constants, a base value that triggers a validation warning will trigger
it on every trial where that coordinate is inactive. Concretely,
$\lambda_{\mathrm{sep}}$ must equal its default, or every $\texttt{triplet}$
and $\texttt{joint}$ trial warns that $\lambda_{\mathrm{sep}}$ is inert. The
preflight checks this.

The surrogate then sees, for each trial, up to two coordinates that are
exactly flat. ARD length-scales absorb flat directions by construction: the
marginal likelihood is maximised by sending the length-scale of an irrelevant
input to a large value, after which the kernel is insensitive to it. What that
absorption *costs* in sample efficiency at $N_{\mathrm{calls}} = 300$ is not
estimated anywhere in this work and is recorded as an open point (§5, item 7).

#### 3.3.3 The identifiability argument: why $m_{\cos}$ and $\alpha$ are never both active

Observe from (8) that $m_{\cos} \in A(\ell)$ and $\alpha \in A(\ell)$ never
hold simultaneously, for any $\ell \in \mathcal{T}$. This is not an
implementation convenience; it is a statement about the geometry of the loss.

Both parameters act on the same scalar functional of the embedding: the ratio
of within-class to between-class distance. The margin hinge is inactive once
$D_{an} - D_{ap} > 2 m_{\cos}$; the angular hinge is inactive once
$D_{ap} < 4\tan^2(\alpha)\, D_{nc}$, where $x_c$ is the anchor-positive
midpoint and $D_{nc} = \lVert z_n - x_c\rVert_2^2$. For a configuration at a
given within/between ratio, an increase in $m_{\cos}$ and a decrease in
$\alpha$ produce nearly the same change in which triples are active and by how
much. The objective is therefore nearly constant along a curve in the
$(m_{\cos}, \alpha)$ plane: a ridge. Trials that move along a ridge purchase
no information, because the likelihood is flat there; formally, the Fisher
information matrix of the pair is near-singular, and the pair is close to
non-identifiable from the objective alone.

Restricting each loss type to at most one of them removes the ridge by
construction rather than hoping the search avoids it.

### 3.4 Encoding and dimension: 18 axes, 22 columns

*This section establishes exactly what the surrogate sees, and why that is not
what the configuration declares.*

The space $\mathcal{X}$ has $d_{\mathrm{ax}} = 17$ declared axes:

| # | Axis | Type | Range / levels | Active when |
|---|---|---|---|---|
| 1 | `depth_exponent` | Integer | $[2, 5]$ | always |
| 2 | `width_multiplier` | Real | $[1.5, 5.0]$ | always |
| 3 | `block_family` | Integer | $\{0, 1\}$ | always |
| 4 | `embedding_size` | Integer | $[8, 16]$ | always |
| 5 | `lr` | Real, log | $[10^{-4}, 0.2]$ | always |
| 6 | `one_minus_beta1` | Real, log | $[10^{-2}, 10^{-1}]$ | always |
| 7 | `one_minus_beta2` | Real, log | $[10^{-4}, 10^{-2}]$ | always |
| 8 | `weight_decay` | Real, log | $[10^{-5}, 10^{-2}]$ | always |
| 9 | `dropout` | Real | $[0, 0.3]$ | always |
| 10 | `margin` ($m_{\cos}$) | Real | $[0.1, 1.0]$ | $\ell = \texttt{triplet}$ |
| 11 | `angular_alpha_deg` ($\alpha$) | Real | $[2, 20]$ | $\ell \ne \texttt{triplet}$ |
| 12 | `lambda_sep` | Real, log | $[10^{-2}, 20]$ | $\ell = \texttt{joint\_sep}$ |
| 13 | `mining_strategy` | Categorical | 3 levels | always |
| 14 | `loss_type` | Categorical | 3 levels | always |
| 15 | `strict_semihard` | Integer | $\{0, 1\}$ | $\ell \ne \texttt{triplet}$; projected by $\Pi$ |
| 16 | `head_fusion` | Integer | $\{0, 1\}$ | always |
| 17 | `head_pool_ops` | Integer | $\{0, 1\}$ | always |

The surrogate does not see 18 inputs. A $k$-level categorical axis is one-hot
expanded into $k$ binary columns, so

$$
d_{\mathrm{col}} \;=\; \underbrace{9}_{\text{Real}} \;+\; \underbrace{6}_{\text{Integer}} \;+\; \underbrace{3 + 3}_{\text{two 3-level Categorical}} \;=\; 21 .
\tag{9}
$$

Two remarks.

**Binary axes as Integer, not Categorical.** For a two-level factor the
one-hot encoding is *exactly redundant*: the second column satisfies
$x^{(2)} = 1 - x^{(1)}$, so it adds a dimension without adding information, and
the resulting design matrix is rank-deficient in those columns. Encoding the
five binaries as integer axes therefore saves five columns at no cost in
expressiveness, and a binary imposes no false ordering (there is only one
possible ordering of two levels, and it carries no metric content beyond
"different"). This is why $d_{\mathrm{col}} = 21$ rather than $25$.

**A caveat discovered, not assumed.** The repository carries an explicit
warning that the block-family axis must not be a real-valued dimension, since a
sampled value such as $0.37$ cannot index a list. It was tempting to justify
the integer encoding by asserting that integer axes yield native Python
integers. **They do not**: the sampler returns NumPy integer scalars. The
substance of the warning survives — a NumPy integer indexes a list correctly
where a float raises — so the encoding decision stands, but the stated
justification was false, and NumPy scalars are not JSON-serialisable, which
would have failed downstream where the trial log is written. Points are
therefore normalised to native types at the projection boundary. This is
recorded because a design decision defended by a false premise is fragile even
when its conclusion is right.

### 3.5 Coverage of the cell partition: an exact result

*This section establishes how many random draws are needed before the surrogate
has seen the cells it will be asked to extrapolate over, and is the principal
original derivation of this document.*

#### 3.5.1 The induced cell measure

The sampler draws each categorical coordinate uniformly and independently:
$m \sim \mathrm{Unif}(\mathcal{M})$, $\ell \sim \mathrm{Unif}(\mathcal{T})$,
$s \sim \mathrm{Unif}\{0,1\}$, and $h \sim \mathrm{Unif}(\mathcal{H})$. The
projection $\Pi$ then pushes this measure forward onto $\mathcal{W}^{*}$. Since
$\Pi$ merges exactly those raw conditions that differ only in $s$ within its
two collapsing branches, the induced condition probabilities are

$$
\mathbb{P}\bigl[\Pi(\omega) = (m, \ell, s)\bigr] =
\begin{cases}
\tfrac{1}{3}\cdot\tfrac{1}{3}\cdot 1 = \tfrac{1}{9}, & \ell = \texttt{triplet},\ s = 0,\\[4pt]
\tfrac{1}{9}, & \ell \ne \texttt{triplet},\ m = \texttt{hard},\ s = 0,\\[4pt]
\tfrac{1}{3}\cdot\tfrac{1}{3}\cdot\tfrac{1}{2} = \tfrac{1}{18}, & \ell \ne \texttt{triplet},\ m \ne \texttt{hard}.
\end{cases}
\tag{10}
$$

The total mass checks: $3 \cdot \tfrac{1}{9} + 2 \cdot \tfrac{1}{9} + 8 \cdot \tfrac{1}{18} = \tfrac{3}{9} + \tfrac{2}{9} + \tfrac{4}{9} = 1$.

Because the head geometry is independent and uniform on four values, each cell
probability is the corresponding condition probability divided by four. Hence
the 52 cells carry exactly **two** distinct probabilities:

$$
p_j =
\begin{cases}
\dfrac{1}{36} \approx 0.02778, & \text{for } 20 \text{ cells (the merged conditions)},\\[8pt]
\dfrac{1}{72} \approx 0.01389, & \text{for } 32 \text{ cells (the unmerged ones)}.
\end{cases}
\tag{11}
$$

Note the structural consequence, which is not obvious in advance: **the
projection makes the cells it merges twice as likely as the cells it does not**.
Collapsing two raw conditions onto one legal condition doubles that condition's
mass. The thin cells are precisely the eight
$(m \ne \texttt{hard}) \times (\ell \ne \texttt{triplet}) \times s$ conditions,
crossed with the four heads.

#### 3.5.2 Expected coverage

Let $U_N$ denote the number of cells unvisited after $N$ independent draws.
Writing $\mathbb{1}[\cdot]$ for the indicator, $U_N = \sum_{j=1}^{52} \mathbb{1}[\text{cell } j \text{ unvisited}]$,
and by linearity of expectation — which requires no independence between the
indicators, and they are indeed dependent —

$$
\mathbb{E}[U_N] \;=\; \sum_{j=1}^{52} (1 - p_j)^{N} \;=\; 20\left(1 - \tfrac{1}{36}\right)^{N} + 32\left(1 - \tfrac{1}{72}\right)^{N}.
\tag{12}
$$

Expected coverage is $52 - \mathbb{E}[U_N]$. Evaluating (12), and comparing
against the sampler over 30 independent sampler seeds:

| $N$ | $\mathbb{E}[U_N]$ | predicted cells seen | measured mean seen | measured sd |
|---|---|---|---|---|
| 40 | 24.77 | 27.23 | 27.43 | 1.71 |
| 100 | 9.10 | 42.90 | 43.10 | 2.31 |
| 248 | 1.02 | 50.98 | 50.97 | 0.87 |
| 300 | 0.49 | 51.51 | 51.43 | 0.67 |

The agreement is within sampling error at every $N$, which validates both the
derivation and the claim that the implemented sampler realises the intended
measure.

#### 3.5.3 Consequences for the random initial design

Three quantitative statements follow directly, and together they justify the
choice $N_{\mathrm{init}} = 100$.

First, solving $\mathbb{E}[U_N] < 1$ gives $N = 250$. **Pure random sampling
needs roughly 250 draws before it expects to have seen every cell once** —
five sixths of the entire trial budget. Complete coverage by the random design
alone is therefore not affordable, and the question is only how much of the
partition the surrogate is asked to extrapolate over.

Second, at $N_{\mathrm{init}} = 40$, $\mathbb{E}[U_{40}] = 24.8$: the surrogate
would begin proposing points having never observed **roughly half** the
partition. At $N_{\mathrm{init}} = 100$, $\mathbb{E}[U_{100}] = 9.1$.
Increasing the design from 40 to 100 therefore converts about sixteen
never-observed cells into observed ones, at a cost of sixty trials.

Third, the risk is concentrated where it is least visible. For any *specific*
thin cell, $\mathbb{P}[\text{unseen at } N = 100] = (1 - 1/72)^{100} = 0.247$.
A given thin cell has close to a one-in-four chance of being invisible to the
surrogate when it takes over.

Two caveats on this analysis, both material. It concerns *cell* coverage, not
coverage of the continuous coordinates; a cell visited once has been sampled at
one arbitrary point of an eleven-dimensional continuous space, which is not the
same as having been characterised. And it treats coverage as the quantity of
interest, whereas the surrogate can in principle interpolate to an unvisited
cell from its neighbours precisely because of the information-sharing property
of §3.1 — the expected coverage is thus a conservative diagnostic, not a
requirement. What it rules out is the surrogate extrapolating over a *large*
unobserved fraction, which at $N_{\mathrm{init}} = 40$ it would have been.

### 3.6 The composite objective and its deterministic schedule

*This section establishes the loss actually optimised and the schedule on its
third term.*

#### 3.6.1 The objective

The total loss at optimiser step $t$ is

$$
\mathcal{L}(t) \;=\; \mathcal{L}_{\mathrm{joint}} \;+\; \lambda_{\mathrm{sep}}(t)\,\mathcal{L}_{\mathrm{sep}},
\tag{13}
$$

with

$$
\lambda_{\mathrm{sep}}(t) \;=\; \lambda_{\mathrm{sep}}\, g(t), \qquad
g(t) \;=\; \min\!\left(1, \frac{t}{\tau T}\right) \ \ \text{for } \tau > 0, \qquad
g(t) \equiv 1 \ \ \text{for } \tau = 0,
\tag{14}
$$

for all $t \in \{0, 1, \dots, T\}$, where

$$
T \;=\; E_{\max}\, n_{\mathrm{b}}.
\tag{15}
$$

Note that $\tau = 0$ is defined by continuation rather than by the formula,
since $t/(\tau T)$ is undefined there; the convention is that no warm-up means
full weight from the first step, which reproduces the pre-existing ungated
behaviour exactly.

#### 3.6.2 Why $\tau$ was fixed rather than searched — **SUPERSEDED**

> **Revision 3 supersedes this subsection.** The dose argument below is
> reproduced unchanged because it is wrong in an instructive way, and deleting
> it would leave the reversal unexplained. Equation (16) is correct; the
> *inference* drawn from it is not. Two settings with equal dose have different
> **terminal** weights $\lambda_{\mathrm{sep}}\,g(T)$, and since the epoch
> selector usually picks a late epoch, the terminal weight plausibly governs the
> converged geometry more than the integral does. The dose is therefore **not a
> sufficient statistic** for the pair $(\lambda_{\mathrm{sep}}, \tau)$, the
> ridge does not close, and $\tau$ carries a genuine second degree of freedom.
> It is now the **18th searched axis**, with an upper bound
> $\tau_{\max} = \min(1, P/E_{\max})$ **derived** rather than configured, so
> that the ramp always completes before the earliest stop the rule permits.
> Above that cap "large $\tau$ wins" would be indistinguishable from "the
> separation term was off" — a statement about `joint` versus `joint_sep`, which
> is *already* a searched axis, reached by a confounded route. See
> `TUNING_1_searched_axes.md` §3.17. The claim that the terminal weight matters
> more than the integral remains an **argument, not a measurement**; the
> per-trial log now records both channels so it can be checked.


Integrating the schedule over the planned horizon, for $\tau \in (0, 1]$,

$$
\int_{0}^{T} \lambda_{\mathrm{sep}}(t)\,\mathrm{d}t
= \lambda_{\mathrm{sep}}\left[\int_{0}^{\tau T}\frac{t}{\tau T}\,\mathrm{d}t + \int_{\tau T}^{T} 1\,\mathrm{d}t\right]
= \lambda_{\mathrm{sep}}\left[\frac{\tau T}{2} + T - \tau T\right]
= \lambda_{\mathrm{sep}}\, T\left(1 - \frac{\tau}{2}\right).
\tag{16}
$$

The total separation "dose" delivered over training is thus a product of
$\lambda_{\mathrm{sep}}$ and a factor depending only on $\tau$. To first order
the two parameters trade off multiplicatively: any dose achievable with
$(\lambda_{\mathrm{sep}}, \tau)$ is achievable with
$(\lambda_{\mathrm{sep}}', \tau')$ satisfying
$\lambda_{\mathrm{sep}}(1 - \tau/2) = \lambda_{\mathrm{sep}}'(1 - \tau'/2)$.
That is precisely the ridge structure of §3.3.3, and the same remedy applies:
fix one, search the other. The argument justifies fixing *some* value of $\tau$;
it does not justify $0.3$ specifically, and §5 records that.

#### 3.6.3 Why a deterministic schedule replaces a data-dependent gate

The predecessor mechanism switched $\mathcal{L}_{\mathrm{sep}}$ on when a
running estimate of the training silhouette first crossed a threshold. Three
arguments against it, of which the third is the one that matters for the
search.

*It cannot fire on a fluctuation if it does not read the data.* Measured: the
gate latched in 56 of 60 seeds, and in about 70% of those the running statistic
peaked in a narrow band consistent with an excursion rather than a trend, at a
median held-out silhouette of $-0.054$ — at or below the run's own
label-shuffled null. The switch was firing on noise.

*It confounds two sources of variance.* Let $\mathrm{Var}[S_{\mathrm{val}}]$ be
the across-seed variance of the selection metric within a fixed configuration.
Under a data-dependent gate this decomposes, by the law of total variance
conditioning on the latch step $L$, as

$$
\mathrm{Var}\bigl[S_{\mathrm{val}}\bigr]
= \mathbb{E}\Bigl[\mathrm{Var}\bigl[S_{\mathrm{val}} \,\big|\, L\bigr]\Bigr]
+ \mathrm{Var}\Bigl[\mathbb{E}\bigl[S_{\mathrm{val}} \,\big|\, L\bigr]\Bigr],
\tag{17}
$$

that is, optimisation noise at fixed switch time *plus* the variance
contributed by the switch time itself. Under the deterministic schedule $L$ is
degenerate, the second term vanishes identically, and the measured seed spread
becomes an estimate of optimisation noise alone. This is what makes a seed
spread interpretable, and it is the property the search needs, because the
surrogate's noise model is fitted to exactly that spread.

*It destroys identifiability of $\lambda_{\mathrm{sep}}$.* If the gate never
opens, $\lambda_{\mathrm{sep}}$ is not merely unimportant, it is **unobservable**:
the objective is constant in it. A search over an axis that is inactive in most
trials learns nothing about that axis, and the surrogate's posterior over it
remains the prior. Under the deterministic schedule the axis is active in every
$\texttt{joint\_sep}$ trial and can be identified.

#### 3.6.4 The step convention, and a lag that is real

Two conventions are available: evaluate $g$ at the number of steps *completed*
(so the first batch sees $t = 0$), or at the number *begun* (so it sees
$t = 1$). The former is used, because $g(0) = 0$ exactly, so the first batch
carries no separation weight at all — which is what a warm-up is for. Under
the latter convention the first batch would already carry weight
$\lambda_{\mathrm{sep}}/(\tau T)$, small but nonzero, and the ramp would not
actually start from rest.

This convention has a consequence in the logs that is easy to misread. History
is written at epoch boundaries, and the weight recorded at the end of epoch $k$
is the last one *computed*, namely

$$
g\bigl(k\, n_{\mathrm{b}} - 1\bigr), \qquad \text{not} \qquad g\bigl(k\, n_{\mathrm{b}}\bigr).
\tag{18}
$$

The logged value therefore lags the live state by exactly one step. Verified on
a real run with $\tau T = 20$ and $n_{\mathrm{b}} = 5$: epoch 4 logs $0.95$ and
epoch 5 is the first to log $1.0$, even though full weight is reached at step
20, which is the first batch of epoch 5. Nothing is wrong with the schedule;
the epoch boundary simply does not coincide with the ramp boundary. The first
epoch at which full weight appears in the log is

$$
k_{\mathrm{full}} \;=\; \left\lceil \frac{\tau T + 1}{n_{\mathrm{b}}} \right\rceil,
\tag{19}
$$

which is the quantity to predict when auditing a run, and which the naive
$\lceil \tau E_{\max}\rceil$ gets wrong by one epoch.

#### 3.6.5 The planned horizon is not the realised one

$T$ in (15) is the *planned* budget. Early stopping means a run may take fewer
than $E_{\max}$ epochs. If a run stops after $\tilde{e}$ epochs, it reaches full
weight if and only if

$$
\tilde{e}\, n_{\mathrm{b}} \;\ge\; \tau T
\qquad\Longleftrightarrow\qquad
\frac{\tilde{e}}{E_{\max}} \;\ge\; \tau .
\tag{20}
$$

At $\tau = 0.3$ any run reaching 30% of its cap sees full
$\lambda_{\mathrm{sep}}$; a run stopping earlier never does, and its
$\lambda_{\mathrm{sep}}$ is effectively a smaller number than the one the search
believes it evaluated. Since patience $P$ bounds how early a run can stop, the
preflight warns when $P / E_{\max} < \tau$. The per-epoch record of
$\lambda_{\mathrm{sep}}(t)$ makes the realised dose auditable rather than
assumed.

### 3.7 The centroid-separation term, and its degeneracy at three classes

*This section establishes what the third term actually penalises, and shows
that in its faithful form it is near-vacuous for this benchmark.*

#### 3.7.1 What NC2 asserts

According to PubMed, Papyan, Han and Donoho document that during the terminal
phase of training — the regime beginning when training error first vanishes —
last-layer features and classifiers converge to a rigid geometry, of which the
second component, NC2, is that the class means converge to the vertices of a
simplex equiangular tight frame ([DOI](https://doi.org/10.1073/pnas.2015509117)).
Reading the full text, three points are load-bearing for what follows.

**NC2 is a statement about *globally centred* class means.** The paper states
explicitly that where it refers to class means in the text it means the
globally centred ones, and its measurements of equinormness and equiangularity
are computed on centred means.

**NC2 is measured as more than one condition.** The evidence is assembled from
separate measurements: the coefficient of variation of the centred class-mean
norms (equinormness), the standard deviation of the pairwise cosines
(equiangularity), and the convergence of those cosines to $-1/(K-1)$. The paper
is explicit that maximal equiangularity *combined with* equinormness is what
implies the simplex ETF. A loss that penalises only the third of these
implements one condition out of three.

**NC is a long-training phenomenon.** The reported protocol trains for 300
epochs on one dataset and 350 on the others, with the terminal phase beginning
only after training error vanishes. Any expectation that an NC2-derived penalty
will behave as it does in that literature must reckon with a budget of
$E_{\max} = 60$ to $100$ epochs on a metric-learning objective that never
drives a classification error to zero, because there is no classifier.

#### 3.7.2 The implemented term

For each fixed batch, with $K$ the number of classes present with at least
$n_{\min}$ rows,

$$
\mathcal{L}_{\mathrm{sep}} \;=\; \frac{1}{K(K-1)} \sum_{c \ne c'} \Bigl( \bigl\langle \hat\mu_c, \hat\mu_{c'} \bigr\rangle - \rho_{\mathrm{ETF}} \Bigr)^{2},
\qquad \rho_{\mathrm{ETF}} = -\frac{1}{K-1},
\tag{21}
$$

where the sum runs over ordered pairs of distinct valid classes, and where the
class directions are selected by $\kappa$:

$$
\kappa = 1: \quad \hat\mu_c = \frac{\mu_c - \mu_G}{\lVert \mu_c - \mu_G \rVert_2},
\qquad\qquad
\kappa = 0: \quad \hat\mu_c = \frac{\mu_c}{\lVert \mu_c \rVert_2}.
\tag{22}
$$

$\kappa = 1$ is the faithful reading of NC2. $\kappa = 0$ is not.

#### 3.7.3 The degeneracy at $K = 3$ (a special case of §3.7.4)

**Proposition.** Fix a batch and suppose $\kappa = 1$ and $K = 3$. If the three
centred class means have equal norms,
$\lVert \mu_1 - \mu_G \rVert_2 = \lVert \mu_2 - \mu_G \rVert_2 = \lVert \mu_3 - \mu_G \rVert_2$,
then $\mathcal{L}_{\mathrm{sep}} = 0$ exactly, for **any** arrangement at
**any** scale.

*Proof.* By definition of $\mu_G$ as the mean of the $K$ class means,
$\sum_{c=1}^{3} (\mu_c - \mu_G) = 0$ identically. Under the equal-norm
hypothesis, dividing by that common norm preserves the identity, so the unit
vectors satisfy $\hat\mu_1 + \hat\mu_2 + \hat\mu_3 = 0$. Then
$\hat\mu_3 = -(\hat\mu_1 + \hat\mu_2)$, and taking squared norms,

$$
1 = \lVert \hat\mu_3 \rVert_2^2 = \lVert\hat\mu_1\rVert_2^2 + 2\langle \hat\mu_1, \hat\mu_2\rangle + \lVert\hat\mu_2\rVert_2^2 = 2 + 2\langle \hat\mu_1, \hat\mu_2\rangle ,
$$

so $\langle \hat\mu_1, \hat\mu_2 \rangle = -\tfrac{1}{2} = \rho_{\mathrm{ETF}}$
at $K = 3$. By symmetry the same holds for every pair, so every summand of (21)
vanishes. $\square$

Two corollaries, and they are the substance of the matter.

**The term is scale-free.** Nothing in the proof constrains the common norm. A
*fully collapsed* configuration, in which all three class means are within
$\epsilon$ of $\mu_G$ but symmetrically so, gives
$\mathcal{L}_{\mathrm{sep}} = 0$ just as a well-separated one does. The term
cannot distinguish a good embedding from a degenerate one.

**What it actually penalises is norm imbalance.** By the proposition, deviation
from zero requires unequal centred norms. So at $K = 3$ with $\kappa = 1$ the
term is a proxy for the *equinormness* component of NC2 and carries essentially
no information about separation — which is the opposite of what its name and
its motivation suggest.

That this is a degeneracy of the same family as the known $K = 2$ case is worth
noting: at $K = 2$ centring makes the two means exactly antipodal by
construction, their cosine is identically $-1 = \rho_{\mathrm{ETF}}$, and the
centred form is identically zero for every embedding. At $K = 3$ it is not
identically zero, but the proposition shows the non-zero part measures the
wrong thing.

Consistent measurements: on the repository's own implementation,
$\mathcal{L}_{\mathrm{sep}} = 1.4 \times 10^{-3}$ for a well-separated batch
(batch silhouette $+0.95$) against $3.3 \times 10^{-2}$ for fully overlapping
classes (batch silhouette $-0.09$) — a factor of 24 on a term that never
exceeds $0.045$. On the cluster, the final mean pairwise cosine was $-0.485$
median across all 60 seeds (range $[-0.500, -0.338]$), with seven seeds at or
below $-0.45$ while their best held-out silhouette was at or below zero; the
correlation between attaining the ETF target and the embedding being any good
was $r = -0.15$.

#### 3.7.4 Why the centring was removed outright

Revision 1 argued that the raw form imposes the ETF condition *plus* an extra
constraint $\mu_G \to 0$, and therefore that $\kappa$ should be searched to
settle empirically which form behaved better. **That argument was wrong**, and
its correction removes the axis rather than resolving it.

**The algebraic fact.** For unit vectors $\{v_c\}_{c=1}^{K}$ whose pairwise
inner products all equal $\rho$,

$$
\Bigl\lVert \sum_{c=1}^{K} v_c \Bigr\rVert_2^2 \;=\; K + K(K-1)\rho ,
\tag{21a}
$$

which at the ETF target $\rho = \rho_{\mathrm{ETF}} = -1/(K-1)$ evaluates to
$K - K = 0$. Hence **equiangularity at the ETF target already implies
$\sum_c \hat\mu_c = 0$**: the sum-to-zero property is not an extra constraint
the raw form smuggles in, it is *half of what the ETF target means*. Verified
numerically at $K = 2, 3, 4, 6$.

**Why that makes centring positively harmful, not merely redundant.**
Subtracting $\mu_G$ and then normalising renders $\mathcal{L}_{\mathrm{sep}}$
invariant to translation **and to scale**. The term then measures only the
*shape* of the simplex of class means and never its *size*. Every equilateral
arrangement scores zero, including an arbitrarily small one — so a configuration
in which all three classes have collapsed into a tiny cap is scored as perfect.

Measured on the repository's own implementation, three classes placed in a cap
so tight that their raw pairwise cosine is $+0.9994$ — visually a single blob:

| $\varepsilon$ | raw pairwise cosine | $\mathcal{L}_{\mathrm{sep}}$ centred | $\mathcal{L}_{\mathrm{sep}}$ raw |
|---|---|---|---|
| 0.02 | $+0.9994$ | **0.000035** | 2.248 |
| 0.10 | $+0.9852$ | 0.000001 | 2.206 |
| 0.50 | $+0.7001$ | 0.000000 | 1.440 |
| 2.00 | $-0.1997$ | 0.000000 | 0.090 |

and directly, scaling a fixed configuration by $100\times$ or $0.01\times$
leaves the centred pairwise cosines identical at $-0.4999$. **Separation is
precisely the quantity the centring discards.**

**Where Revision 1 went wrong.** The centring was imported from the neural
collapse literature, where features live in $\mathbb{R}^{E}$ with an arbitrary
offset and $\mu_G$ genuinely is a nuisance parameter. This pipeline
L2-normalises every row onto $\mathbb{S}^{E-1}$, so the origin is already
meaningful. On a sphere centred at the origin, "the class directions balance
about the origin" *is* the statement that they are maximally spread — not a
claim about where the cloud sits. The centring solved a problem this
representation does not have, and destroyed information it does need.

**This unifies the $K = 2$ case rather than carving it out.** Revision 1 treated
the $K = 2$ vacuity as a special-case degeneracy needing a special-case
fallback. It is not special: it is the same defect at its extreme. At $K = 2$
every configuration is a degenerate "equilateral" one, so the centred form is
identically zero *everywhere*; at $K \ge 3$ the collapse is partial rather than
total, which is **worse**, because it is silent instead of obvious. The
proposition of §3.7.3 is the $K = 3$ instance of a general fact.

**Consequences for the design.** The centred code path is deleted;
`CentroidSeparationLoss` no longer accepts a formulation argument and raises
`TypeError` if given one. `sep_centre_means` survives in `TrainConfig` only so
archived configurations parse, and warns if set. The search space loses axis 18,
becoming 17 axes and 21 columns, and $A(\texttt{joint\_sep})$ becomes
$\{\alpha, \lambda_{\mathrm{sep}}\}$.

**One consequence to weigh before launch.** The raw form is a *much* stronger
constraint than the centred one it replaces — it demands a real ETF, not merely
an equilateral shape. Every archived `joint_sep` result was produced under the
shape-only penalty and says nothing about the term as it now stands, and the
searched range of $\lambda_{\mathrm{sep}}$ was chosen against the weaker term.
See §5, item 13.

#### 3.7.5 A tension the search will have to resolve

There is a structural conflict inside the composite objective that is worth
stating plainly, because the search may well be measuring it.

The separation term and a small angular constraint are **collapse-seeking**:
the ETF target drives class means apart while NC1-style behaviour drives
within-class variance down, and the angular constraint's implied silhouette
floor rises monotonically as $\alpha$ falls, demanding within-class collapse in
the limit.

Two of the three mining strategies are **anti-collapse by construction**. The
source that introduced easy-positive mining motivates it precisely as a remedy
for over-clustering: constraining only the nearest same-class neighbour lets a
class occupy a manifold instead of contracting to a point, and the paper
reports that easy-positive embeddings are less tightly clustered on training
data and generalise better to unseen classes than batch-all, N-pair or
hard-positive alternatives (full text in the project knowledge base).

So the space contains configurations that pair a collapse-forcing loss with an
anti-collapse miner. These are not illegal — nothing about them is degenerate
in the sense of §3.2 — but they are internally opposed, and if such a
configuration wins, the interpretation is not obvious. This is an argument for
reading the winning *condition* alongside the winning score, which is why the
trial log records the condition and the cell name for every trial.

### 3.8 The scoring objective under a single seed

*This section establishes what number the search minimises, and what a single
seed does and does not buy.*

#### 3.8.1 The objective

For a trial $t$ with seeds $\sigma = 1, \dots, N_{\mathrm{seeds}}$, let
$u(t, \sigma, e^{*})$ and $v(t, \sigma, e^{*})$ be the primary and secondary
metrics **by role**, both read at the *same* selected epoch
$e^{*}(t, \sigma)$. The objective minimised is

$$
J_{\varepsilon}(t) \;=\; -\frac{1}{N_{\mathrm{seeds}}} \sum_{\sigma=1}^{N_{\mathrm{seeds}}} \Bigl[\, u(t, \sigma, e^{*}) \;+\; \varepsilon\, v(t, \sigma, e^{*}) \,\Bigr],
\tag{23}
$$

with $\varepsilon$ undefined (and the bracket reducing to $u$ alone) whenever
the tie-break is disabled. Reading both metrics at one epoch is deliberate:
reading them at separately optimal epochs would report a configuration that
never existed at any point in training.

#### 3.8.2 The tie-break and its guarantee

When enabled,

$$
\varepsilon \;=\; \frac{\gamma_{\mathrm{tb}}\, \Delta_{\min}(y)}{s_{\mathrm{hi}} - s_{\mathrm{lo}}},
\tag{24}
$$

where $[s_{\mathrm{lo}}, s_{\mathrm{hi}}]$ bounds the secondary metric. The
guarantee is immediate: the total influence the secondary metric can exert on
(23) is at most $\varepsilon (s_{\mathrm{hi}} - s_{\mathrm{lo}}) = \gamma_{\mathrm{tb}} \Delta_{\min}(y)$,
which for every $\gamma_{\mathrm{tb}} \in (0,1)$ is strictly less than
$\Delta_{\min}(y)$, the smallest genuine primary difference the evaluation set
can express. Hence the secondary metric can reorder configurations **only
inside an exact primary tie**, which is the property that makes it safe.

Two conditions disable it, and both apply here. $\gamma_{\mathrm{tb}} = 0$ is
the documented off switch. More subtly, (24) presupposes that
$\Delta_{\min}(y)$ *exists*: ARI on a fixed label vector of finite length takes
finitely many values, so it has a smallest positive gap, whereas the mean
silhouette is continuous in the embedding, exact ties have probability zero,
and a weight whose whole justification is "it acts only inside an exact tie"
has nothing to act on. This study uses the silhouette as primary; the tie-break
is therefore inapplicable in principle, not merely switched off, and the
configuration sets $\gamma_{\mathrm{tb}} = 0$ so that the fact is stated rather
than warned about on every phase.

#### 3.8.3 The degrees-of-freedom question, and why it is load-bearing

The per-trial record reports a spread across seeds. Two estimators are
available:

$$
\hat\sigma^2_{\mathrm{ddof}=0} = \frac{1}{n}\sum_{i=1}^{n}(a_i - \bar{a})^2,
\qquad
\hat\sigma^2_{\mathrm{ddof}=1} = \frac{1}{n-1}\sum_{i=1}^{n}(a_i - \bar{a})^2 .
\tag{25}
$$

At $n = N_{\mathrm{seeds}} = 1$ the second is $0/0$ and evaluates to NaN. This
is not a cosmetic difference. The optimiser's surrogate cannot be fitted to a
non-finite observation, so a NaN propagating into the trial record would abort
the study — after however many hours of training had already been spent. The
implementation uses the population estimator, so a single seed yields a spread
of exactly $0$, and the objective remains finite. This was verified rather than
assumed, and cross-checked at three seeds so that the check is not a special
case.

The same principle governs failures: a trial that cannot be built or that
crashes returns a large **finite** penalty $F = +1.0$, strictly worse than any
achievable value of the negated metric (which is bounded in magnitude by 1) and
therefore learnable as "avoid this region", without ever introducing a
non-finite value. A trial in which some but not all seeds complete is scored as
failed rather than averaged over survivors, since a configuration scored on
fewer seeds is not comparable with one scored on all of them — and, worse, a
configuration that crashed on two of three seeds but was lucky on the third
would report a high mean with zero spread, which is exactly the signature the
surrogate finds most attractive.

#### 3.8.4 What one seed costs

Let two configurations have true selection-metric means differing by $d$, and
let the across-seed standard deviation be $\sigma_{\mathrm{s}}$. Under a normal
approximation, the difference of two independent single-seed draws has standard
deviation $\sigma_{\mathrm{s}}\sqrt{2}$, so the probability that a single seed
each misranks them is

$$
\mathbb{P}[\text{wrong order}] \;=\; \Phi\!\left(-\frac{d}{\sigma_{\mathrm{s}}\sqrt{2}}\right).
\tag{26}
$$

With the measured $\sigma_{\mathrm{s}} = 0.073$:

| $d$ | 0.02 | 0.05 | 0.10 | 0.15 | 0.20 |
|---|---|---|---|---|---|
| $\mathbb{P}[\text{wrong order}]$ | 42% | 31% | 17% | 7.2% | 2.6% |

The measured between-cell standard deviation is $\sigma_{\mathrm{b}} = 0.117$,
so $\sigma_{\mathrm{s}}/\sigma_{\mathrm{b}} = 0.62$. The interpretation is
sharp: **coarse structure is recoverable, fine ranking is not.** Effects of
order $0.2$ — the scale on which mining strategy and loss type appear to
differ — are resolved at a few percent error. Differences of $0.05$ are close
to a coin flip.

This has a direct consequence for what the search output means. The reported
argument-minimum is a *top-$k$ candidate*, not a winner, and a confirmatory
re-fit of the leading points at several seeds each is part of the design rather
than an optional extra. It also bears on the surrogate: with one seed, the
observation noise the Gaussian process must absorb through its nugget is the
full $\sigma_{\mathrm{s}}$, not $\sigma_{\mathrm{s}}/\sqrt{N_{\mathrm{seeds}}}$.

### 3.9 The cost model, the size mix, and the submission gate

*This section establishes how a wall-clock estimate is obtained and under what
condition it may be believed.*

#### 3.9.1 Why the corners of the space are the wrong statistic

Parameter count is roughly exponential in the depth exponent, so the
architecture space spans orders of magnitude. The natural instinct is to report
its corners. That is the wrong quantity: what determines total cost is the
**expectation over the sampling distribution**, and the sampled distribution of
sizes is strongly right-skewed, so its mean and its median differ by a large
factor.

Measured over 300 draws at the configured ranges, with depth near-uniform on
$\{2,3,4,5\}$ (26%, 22%, 23%, 28%):

| statistic | min | p25 | median | p75 | max | **mean** |
|---|---|---|---|---|---|---|
| parameters | 0.01 M | 0.07 M | 0.71 M | 6.23 M | 31.58 M | **4.46 M** |

The ratio mean/median is $6.26$. A practitioner reasoning from the *typical*
sampled model — 0.71 M parameters — would underestimate the cost by roughly
that factor, because cost is (approximately) linear in size and expectation
commutes with a linear map, not with the median.

#### 3.9.2 The cost model and the multiplier

Let $r(\theta) = a + b\,\theta$ be the per-epoch cost at parameter count
$\theta$, fitted by least squares to $k$ timed points. Let $\Theta$ be the
random parameter count induced by sampling a point of $\mathcal{X}$. Then the
expected per-epoch cost over the design is $\mathbb{E}[r(\Theta)] = a + b\,\mathbb{E}[\Theta]$,
estimated by the empirical mean over the dry-run sample, and the total wall
clock is

$$
\widehat{H} \;=\; \frac{N_{\mathrm{calls}}\, N_{\mathrm{seeds}}\, \bar{e}\; \mathbb{E}[r(\Theta)]}{3600},
\tag{27}
$$

in hours, with $\bar{e}$ the mean epochs actually run per trial (historically
about $0.55\,E_{\max}$). The **size-mix multiplier** is

$$
\eta \;=\; \frac{\mathbb{E}[r(\Theta)]}{r(\theta_{\mathrm{ref}})},
\tag{28}
$$

the ratio of the expected cost to the cost at the reference architecture the
original estimate implicitly assumed. $\eta$ is the number the predecessor
design recorded as unmeasured; (27) and (28) are how it is measured.

#### 3.9.3 The admissibility condition on the estimate

Equation (27) is only meaningful if $r$ actually describes the timed points.
This is not automatic and its failure is treacherous, because least squares
returns a confident-looking slope regardless of fit quality. The failure mode
was observed rather than anticipated: on small models the per-epoch cost is
dominated by fixed overhead — data loading, mining, interpreter time — so the
$b\theta$ term explains little of the variance, and a fit with $R^2 = 0.29$
still produced a precise-looking hour count.

The gate therefore requires

$$
R^2 \;\ge\; R^2_{\min} \;=\; 0.70,
\tag{29}
$$

and treats a fit below that, or a NaN, as a **failure** rather than a
footnote. The reasoning is that an unreliable estimate used as a submission
gate is worse than no estimate at all, because it carries authority it has not
earned. When (29) fails the remedy is to time more points spanning a wider
range of depths, or to model cost on something other than parameter count.

#### 3.9.4 The gate

Submission is authorised if and only if all of the following hold:

$$
\underbrace{U_N = 0 \ \text{over the observed draws}}_{\text{coverage}}
\ \wedge\
\underbrace{n_{\mathrm{fail}} = 0}_{\text{every point builds}}
\ \wedge\
\underbrace{R^2 \ge R^2_{\min}}_{\text{the model fits}}
\ \wedge\
\underbrace{\widehat{H} \le (1 - \beta)\,W}_{\text{it fits the walltime}} .
\tag{30}
$$

The margin $\beta$ is not decoration. The size mix is an estimate, $\bar{e}$ is
a historical average from a different configuration, and the optimiser is
**sequential**, so the work cannot be split across parallel lanes if it
overruns; a job killed at the wall loses everything after its last checkpoint.
A study that fits with zero margin does not fit.

Finally, the dry run builds every sampled point through the *same* constructor
the real search uses, rather than a reimplementation. This is what makes it
capable of catching a coordinate that decodes wrongly — a failure that would
otherwise appear only after a job had been queued.

---

## 4. Summary of results

The key statements, each with a cross-reference to where it was established.

**S1. The search object.** The factorial's family of conditional problems
$\{J(\cdot \mid \gamma)\}_{\gamma \in \mathcal{C}}$ is replaced by one problem
on a product space in which the cell is a coordinate, Eq. (2), §3.1. This buys
information sharing across cells and adaptive budget allocation, at the price
of abandoning matched-budget comparison.

**S2. The degenerate region is provably empty, not merely unpromising.** Hard
mining returns $D_{an} < D_{ap}$, Eq. (3); the strict filter requires
$D_{ap} < D_{an}$, Eq. (4); the intersection is empty for every batch, so the
loss and its gradient are identically zero while the run appears stable.
§3.2.1.

**S3. $\Pi$ is a projection onto 13 legal conditions,** Eq. (5), idempotent,
altering only the filter coordinate, with $\lvert\mathcal{W}^{*}\rvert = 13$ by
Eq. (6) and hence 52 cells. §3.2.2.

**S4. A projection is correct where a penalty is not.** A penalty on the
degenerate region leaks, through the kernel, onto the legal $m = \texttt{hard}$
region and teaches a false main effect; a projection produces duplicated
observations, which a Gaussian process handles natively. §3.2.3.

**S5. Inactive coordinates must be clamped,** or the encoding manufactures
noise by mapping identical trainers to distinct inputs; the base configuration
supplies the clamp constants. §3.3.2.

**S6. $m_{\cos}$ and $\alpha$ are never jointly active** because both bind on
the within/between distance ratio, making the pair near-non-identifiable and
tracing a ridge. §3.3.3. That argument was also applied to $\tau$, and is
**superseded** there: equal dose does not imply equal terminal weight, so
$\tau$ is now searched under a derived cap. §3.6.2.

**S7. 18 axes, 22 surrogate columns,** Eq. (9); the four binaries are integer
axes because a two-level one-hot is exactly redundant, $x^{(2)} = 1 - x^{(1)}$.
§3.4.

**S8. The induced cell measure has exactly two values,** Eq. (11): the
projection *doubles* the mass of the conditions it merges, so 20 cells carry
$1/36$ and 32 carry $1/72$. §3.5.1.

**S9. Expected coverage is exact,**
$\mathbb{E}[U_N] = 20(1 - 1/36)^N + 32(1 - 1/72)^N$, Eq. (12), and matches the
sampler within sampling error at $N \in \{40, 100, 248, 300\}$. §3.5.2.

**S10. Pure random sampling needs $N = 250$ draws for $\mathbb{E}[U_N] < 1$.**
At $N_{\mathrm{init}} = 40$ the surrogate begins with $24.8$ cells unobserved;
at $100$, with $9.1$; any given thin cell is unseen at $N = 100$ with
probability $0.247$. §3.5.3.

**S11. The warm-up dose is $\lambda_{\mathrm{sep}} T (1 - \tau/2)$,** Eq. (16),
which is why $\tau$ and $\lambda_{\mathrm{sep}}$ trade off multiplicatively and
only one may be searched. §3.6.2 — but note that the $\tau$ case is
**superseded**; the ridge closes for $(m_{\cos}, \alpha)$ and does not close
for $(\lambda_{\mathrm{sep}}, \tau)$.

**S12. A deterministic schedule removes a variance component.** Under a
data-dependent gate the seed spread decomposes as Eq. (17) into optimisation
noise plus switch-timing variance; making the switch deterministic annihilates
the second term and makes the spread interpretable — which matters because the
surrogate's noise model is fitted to it. §3.6.3.

**S13. The logged weight lags the live weight by one step,** Eq. (18), so full
weight first appears in the log at epoch
$\lceil (\tau T + 1)/n_{\mathrm{b}} \rceil$, Eq. (19), not
$\lceil \tau E_{\max} \rceil$. §3.6.4.

**S14. A run reaches full weight iff $\tilde{e}/E_{\max} \ge \tau$,** Eq. (20),
so early stopping can silently reduce the realised $\lambda_{\mathrm{sep}}$.
§3.6.5.

**S15. The centred separation term is scale-invariant and therefore blind to
collapse.** Measured: three classes at raw pairwise cosine $+0.9994$ score
$0.000035$ centred against $2.248$ raw. The $K = 3$ equal-norm proposition is
the special case. §3.7.3, §3.7.4.

**S16. Equiangularity at $-1/(K-1)$ already implies $\sum_c \hat\mu_c = 0$,**
by Eq. (21a), so the raw form imposes no extra constraint and the centring is
not a correction but a defect. It has been removed; the space became 17 axes / 21
columns. §3.7.4.

**S17. The composite objective contains an internal tension:** the separation
term and small $\alpha$ are collapse-seeking, while two of the three mining
strategies are anti-collapse by construction. §3.7.5.

**S18. The tie-break can only act inside an exact primary tie,** by Eq. (24),
and is inapplicable in principle under a continuous primary metric — which is
the case here. §3.8.2.

**S19. The variance estimator's `ddof` is load-bearing at one seed:** the
sample estimator gives NaN, which cannot be fitted and would abort the study;
the population estimator gives $0$. §3.8.3.

**S20. Single-seed misranking probability is $\Phi(-d/(\sigma_{\mathrm{s}}\sqrt{2}))$,**
Eq. (26); at the measured $\sigma_{\mathrm{s}} = 0.073$, differences of $0.2$
are resolved at 2.6% error and differences of $0.05$ at 31%. The output is a
top-$k$ candidate set, not a winner. §3.8.4.

**S21. Wall clock scales with the mean, not the median, of the size mix,**
Eq. (27); measured mean 4.46 M against median 0.71 M parameters, a factor of
6.26. §3.9.1.

**S22. The wall-clock estimate is inadmissible unless the cost model fits,**
Eq. (29); an $R^2$ of 0.29 was observed to produce a confident-looking and
meaningless hour count. §3.9.3.

**S23. The submission gate is the conjunction Eq. (30)** of coverage, build
success, model fit, and walltime with margin. §3.9.4.

---

## 5. Open points, caveats, and assumptions

1. **This search does not answer the original scientific question.** Under the
   adaptive allocation of §3.1, the trial budget is not matched across loss
   types, so $\texttt{triplet}$ may receive very few trials. The search returns
   a tuned configuration; it does **not** establish that the composite
   objective beats the triplet baseline at matched budget. That comparison
   requires its own run. Accepted deliberately.

2. **$\tau = 0.3$ is asserted, not derived.** Equation (16) justifies fixing
   *some* value; it says nothing about which. If the winning configuration uses
   $\texttt{joint\_sep}$, a one-dimensional sweep over $\tau$ at the winning
   point is the cheap check.

3. **The wall clock has not been measured on the target cluster.** The
   arithmetic reproduces the design document's 157 h estimate against a 144 h
   walltime, but that is the reference-architecture rate; $\eta$ of Eq. (28) is
   unmeasured on real hardware. The gate, Eq. (30), refuses to authorise
   submission until it is. The epoch cap and patience remain at their stated
   values rather than the recommended reductions, which were explicitly not
   accepted.

4. **The coverage analysis concerns cells, not the continuous space.** A cell
   visited once has been sampled at one arbitrary point of an
   eleven-dimensional continuous space. Equation (12) is a conservative
   diagnostic on the categorical partition, not a statement that the space has
   been characterised. §3.5.3.

5. **The cost of flat directions is unquantified.** Up to three of the 22
   columns are exactly flat for any given trial. ARD absorbs them in principle;
   how much of a 300-trial budget that absorption costs is not estimated
   anywhere here. §3.3.2.

6. **The $K = 3$ degeneracy proof is my own,** and is now understood as the
   $K = 3$ instance of the general scale-invariance defect of §3.7.4. It is a two-line argument,
   verified numerically against the repository's implementation, but it was not
   found stated in the literature consulted, and it should be checked
   independently before it appears in a thesis. §3.7.3.

7. **The single-seed noise figures come from a different configuration.**
   $\sigma_{\mathrm{s}} = 0.073$ and $\sigma_{\mathrm{b}} = 0.117$ were measured
   on the screening tier at a smaller step budget. Whether they transfer to
   6 000–10 000 optimiser steps is an assumption, and if the noise scales
   differently the misranking table of §3.8.4 is optimistic or pessimistic by
   an unknown amount.

8. **The normal approximation in Eq. (26)** is exactly that. The across-seed
   distribution of the selection metric was not tested for normality, and the
   measured maximum within-cell spread ($0.226$) is three times the median,
   which is not what a well-behaved normal sample looks like.

9. **NC2 is invoked outside its documented regime.** The source documents
   neural collapse in classification networks trained with cross-entropy for
   300–350 epochs into a terminal phase defined by vanishing training error.
   This work applies an NC2-derived penalty to a metric-learning objective with
   no classifier, no cross-entropy, and a budget an order of magnitude shorter.
   Nothing in the source licenses that extrapolation; it is a design
   hypothesis, and §3.7.3 gives a specific reason to doubt it at $K = 3$.

10. **The benchmark's difficulty scale is uncalibrated.** The class-overlap
    parameter of the synthetic generator was never calibrated against the
    class-centre construction, so every absolute silhouette number sits on an
    uncalibrated scale. Unchanged by this work, and inherited from the
    predecessor design.

11. **Equation (16) treats $t$ as continuous.** The schedule is applied at
    integer steps, so the integral is an approximation to a sum; the error is
    $O(1/T)$ and negligible at $T \ge 6\,000$, but the statement is
    approximate, not exact.

13. **The searched range of $\lambda_{\mathrm{sep}}$ predates the removal of
    the centring.** $[10^{-2}, 20]$ was chosen against the weaker, shape-only
    term. The raw form is a substantially harder constraint, so the useful
    range may now sit lower; the log-uniform prior over three decades can reach
    down to $10^{-2}$, but the range has not been revisited against the
    corrected term. Flagged, not changed.

14. **Every archived `joint_sep` result is about the shape-only penalty.** All
    20 `joint_sep` cells of the screening factorial, and any earlier joint run,
    used the centred form. Whatever they show about the separation term does
    not transfer.

12. **The gate's threshold $R^2_{\min} = 0.70$ is a judgement,** not a derived
    quantity. It was chosen to exclude the observed failure ($R^2 = 0.29$) with
    margin; no analysis relates it to the resulting error in $\widehat{H}$.

---

## 6. References and further reading

**Read in full text and relied upon.**

- Papyan, V., Han, X. Y., Donoho, D. L. (2020). *Prevalence of neural collapse
  during the terminal phase of deep learning training.* PNAS 117(40),
  24652–24663. Retrieved via PubMed, full text from PubMed Central
  (PMC7547234), [DOI](https://doi.org/10.1073/pnas.2015509117). Source for: the
  definition of the terminal phase of training; NC2 as convergence of class
  means to a simplex ETF; the fact that the paper's class means are the
  *globally centred* ones; the decomposition of the NC2 evidence into
  equinormness, equiangularity and convergence of cosines to $-1/(K-1)$; and
  the training protocol of 300–350 epochs that defines the regime in which NC
  was observed. Used in §3.7.1, and the basis of the caveat at §5, item 9.

- Xuan, H., Stylianou, A., Pless, R. *Improved Embeddings with Easy Positive
  Triplet Mining.* Full text in the project knowledge base. Source for: the
  definition of easy-positive mining as selection of the nearest same-class
  example; the semi-hard negative condition requiring the anchor-negative
  distance to exceed the anchor-positive distance within the margin, which
  makes the emptiness argument of §3.2.1 exact; and the motivation of
  easy-positive mining as a remedy for over-clustering, with the reported
  finding that easy-positive embeddings are less tightly clustered and
  generalise better to unseen classes. Used in §3.2.1 and §3.7.5.

**Searched, with the outcome reported rather than assumed.**

- PubMed was queried for the neural-collapse literature (2 records, one of them
  the source above) and for Gaussian-process Bayesian optimisation over
  conditional and categorical search spaces (**0 records**). The second
  outcome is expected but is reported rather than presumed: that index covers
  biomedical and life-sciences literature and does not index the
  machine-learning methodology venues where the Bayesian-optimisation results
  live. Consequently **no claim in §3.1, §3.3, §3.4 or §3.5 about Gaussian
  processes, ARD, acquisition functions or one-hot encoding is supported by a
  retrieved source**; those statements are standard background reasoning or
  derivation, and are labelled as such here rather than dressed in a citation.

- The bioRxiv/medRxiv connector was queried. It supports filtering by subject
  category and date **only — it has no keyword search**, so it cannot be
  targeted at metric learning, neural collapse or Bayesian optimisation at all.
  A category query returned recent bioinformatics preprints, none on topic. No
  preprint is cited in this document, and none was used.

**Read directly from the repository** (branch `feat/composite-dsn-loss`), not
from memory: `condition_space.py`, `search.py`, `dsn_joint_loss.py`,
`train.py`, `config.py`, `run_optimization.py`, `search_dry_run.py`,
`backbone.py`, `hpc/make_factorial_configs.py`, `hpc/preflight_config.py`, and
the 104 configuration files of the two factorial tiers.

**From the project knowledge base.** `HANDOFF_joint_gp_search.md` (the frozen
design, including decisions D1–D6 and the budget arithmetic);
`HANDOFF_factorial52_analysis.md` (the screening findings and their
limitations); `03_USAGE.md` and `02_TECHNICAL.md` (the objective, the failure
policy, and the search-space bug history).

**Measured in this work, and stated as measurement rather than citation:** the
cell-probability decomposition of Eq. (11) and its coverage consequences
(§3.5); the parameter-count distribution of §3.9.1; the epoch-lag verification
of §3.6.4; the $R^2 = 0.29$ cost-model failure of §3.9.3; and the numerical
verification of the $K = 3$ degeneracy against the repository's own
implementation.

**Stated from the predecessor documents' measurements, not re-measured here:**
the 16 814-mined/0-surviving figure (§3.2.1); the gate-latching statistics
(§3.6.3); the separation-term magnitudes and the $r = -0.15$ correlation
(§3.7.3); and $\sigma_{\mathrm{s}} = 0.073$, $\sigma_{\mathrm{b}} = 0.117$
(§3.8.4).
