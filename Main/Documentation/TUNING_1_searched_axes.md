# Tuning Reference I — The Seventeen Searched Axes

**Document 1 of 2.** Companion: *Tuning Reference II — The Knobs You Set*.
**Applies to:** `Main/hpc/Config/config_l3c_joint_search.json`,
`search_mode = "joint_conditions"`, branch `feat/composite-dsn-loss`.
**Date:** 3 August 2026 — **Revision 2**

> **Revision 2:** axis 18, `sep_centre_means`, has been **removed**. The centred
> form of $\mathcal{L}_{\mathrm{sep}}$ is invariant to scale and so cannot see
> collapse (measured: 0.000035 against 2.248 raw on collapsed classes), and
> equiangularity at $-1/(K-1)$ already implies the class directions sum to zero,
> so the raw form imposes nothing extra. Centring is deleted from the code, not
> made selectable.
>
> **Revision 3:** `sep_warmup_frac` ($\tau$) has been **added** as axis 18,
> appended last. It was previously a fixed knob (Document II §3.3) on a dose
> argument that does not hold — equal dose does not imply equal terminal
> weight. Its upper bound is **derived**, $\tau_{\max} = \min(1, P/E_{\max})$,
> not configured. The space is now **18 axes / 22 surrogate columns**. §3.17.

## Abstract

This document is the reference for the eighteen dimensions that the Gaussian
process moves on its own. For each axis it states the meaning, the type and
range as configured, the prior, the condition under which the axis is *active*,
the theory behind the range choice, the consequences of widening or narrowing
it, and the failure mode it can produce. The scientific question it serves is
practical: given a fixed trial budget, which of these ranges are load-bearing,
which are cosmetic, and which will silently cost the study if set wrongly.

**Covered:** the seventeen axes; their encodings; the activity mask; how a range
change propagates into surrogate columns, into cost, and into the coverage
arithmetic.

**Deliberately excluded:** everything the search does *not* move — budget,
schedule, selection metric, batch geometry, runtime, and the verification gate.
Those are Document 2. Also excluded: the derivations themselves, which live in
`THEORY_joint_condition_search.md` and are cross-referenced here rather than
repeated.

---

## 1. Notation and Symbols

| Symbol | Name / Meaning | Type & domain | Units | First used in § |
|---|---|---|---|---|
| $x$ | one sampled point of the search space | $x \in \mathcal{X}$ | mixed | §3 |
| $x^{(j)}$ | the $j$-th coordinate of $x$; parenthesised superscript, **never a power** | mixed | mixed | §3 |
| $d_{\mathrm{ax}}$ | declared axis count | $d_{\mathrm{ax}} = 17$ | dimensionless | §3 |
| $d_{\mathrm{col}}$ | surrogate columns after one-hot expansion | $d_{\mathrm{col}} = 21$ | dimensionless | §3 |
| $\ell$ | `loss_type` | $\ell \in \{\texttt{triplet}, \texttt{joint}, \texttt{joint\_sep}\}$ | — | §3.13 |
| $m$ | `mining_strategy` | $m \in \{\texttt{hard}, \texttt{easy\_positive}, \texttt{easy\_pos\_semihard\_neg}\}$ | — | §3.12 |
| $s$ | `strict_semihard` | $s \in \{0, 1\}$ | — | §3.14 |
| $h$ | head geometry, the pair (`head_fusion`, `head_pool_ops`) | $h \in \mathcal{H}$, $\lvert\mathcal{H}\rvert = 4$ | — | §3.15 |
| $A(\ell)$ | activity mask: loss hyper-parameters that $\ell$ reads | $A(\ell) \subseteq \mathcal{P}$ | — | §2 |
| $\mathcal{P}$ | loss-hyper-parameter superset | $\lvert\mathcal{P}\rvert = 4$ | — | §2 |
| $\Pi$ | legality projection | idempotent map | — | §3.14 |
| $d$ | `depth_exponent` | $d \in \{2,3,4,5\}$ | dimensionless | §3.1 |
| $B$ | total residual blocks, $B = 2^{d}$ | $B \in \{4, 8, 16, 32\}$ | blocks | §3.1 |
| $w$ | `width_multiplier` | $w \in [1.5, 5.0]$ | dimensionless | §3.2 |
| $E$ | `embedding_size` | $E \in \{8, \dots, 16\}$ | dimensionless | §3.4 |
| $\theta$ | parameter count of a sampled backbone | $\theta \in \mathbb{N}$ | parameters | §3.1 |
| $\eta_{\mathrm{lr}}$ | `lr`, the Adam learning rate | $\eta_{\mathrm{lr}} \in [10^{-4}, 0.2]$ | dimensionless | §3.5 |
| $\beta_1, \beta_2$ | Adam exponential decay rates | $\beta_1, \beta_2 \in (0,1)$ | dimensionless | §3.6 |
| $u_1, u_2$ | `one_minus_beta1`, `one_minus_beta2`; $\beta_i = 1 - u_i$ | $u_1 \in [10^{-2}, 10^{-1}]$, $u_2 \in [10^{-4}, 10^{-2}]$ | dimensionless | §3.6 |
| $\lambda_{\mathrm{wd}}$ | `weight_decay` | $\lambda_{\mathrm{wd}} \in [10^{-5}, 10^{-2}]$ | dimensionless | §3.8 |
| $p_{\mathrm{drop}}$ | `dropout` | $p_{\mathrm{drop}} \in [0, 0.3]$ | probability | §3.9 |
| $m_{\cos}$ | `margin`, in cosine-distance units | $m_{\cos} \in [0.1, 1.0]$ | cosine distance | §3.10 |
| $\alpha$ | `angular_alpha_deg`, the angular half-angle | $\alpha \in [2, 20]$ | degrees | §3.11 |
| $\lambda_{\mathrm{sep}}$ | asymptotic weight on the separation term | $\lambda_{\mathrm{sep}} \in [10^{-2}, 20]$ | dimensionless | §3.12 |
| $C$ | number of condition classes | $C = 3$ here | dimensionless | §3.11 |
| $K$ | valid classes present in a batch, $K \le C$ | $K \ge 2$ | dimensionless | §3.16 |
| $S_{\min}(\alpha)$ | silhouette floor implied by the angular constraint | $S_{\min} \in [0, 1)$ | dimensionless | §3.11 |
| $S_{\min}(m_{\cos})$ | silhouette floor implied by the margin | $S_{\min} \in [0, 1)$ | dimensionless | §3.10 |
| $N_{\mathrm{calls}}$ | trial budget | $N_{\mathrm{calls}} = 300$ | trials | §4 |
| $\mathbb{E}[\Theta]$ | mean parameter count over the sampled mix | — | parameters | §3.1 |

### 1.1 Conventions

- "Active" means the configured trainer *reads* the axis on that trial. An
  inactive axis is still present in $x$ and still occupies a surrogate column;
  it is simply clamped before the configuration is built.
- Ranges are written as configured in the JSON. Where a range differs from the
  `SearchConfig` dataclass default, both are given.
- "Log prior" means the search library samples uniformly in $\log$ of the
  coordinate. Base is irrelevant.
- Cosine distance is $1 - \langle z_i, z_j\rangle$ for unit-norm embeddings;
  squared-Euclidean distance is $2(1 - \langle z_i, z_j\rangle)$, so a margin
  quoted in cosine units becomes $2m_{\cos}$ in the loss internals.
- Axis numbering is the order the space is built in. **It is load-bearing:** at
  fixed `gp_random_state` the trial sequence is a function of it, so a study
  meant to be compared with an earlier one must not reorder axes.

---

## 2. Glossary

Ordered by first appearance.

**Axis.** One declared search dimension. There are 18.

**Surrogate column.** One input dimension as the Gaussian process sees it,
after a $k$-level categorical has been expanded into $k$ binary columns. There
are 22. Axis count and column count differ, and the second is what governs
sample-efficiency.

**Active / inactive axis.** Whether the configured trainer reads the axis on
this trial. Three of the four loss hyper-parameters are inactive on any given
trial.

**Activity mask $A(\ell)$.** The set of loss hyper-parameters that loss type
$\ell$ reads.

**Clamping.** Fixing an inactive axis to a constant before the configuration is
built, so that two points differing only in inactive coordinates build
byte-identical configurations. The constant is whatever the *base
configuration* states.

**Legality projection ($\Pi$).** The deterministic idempotent map applied to
the categorical coordinates before a configuration is built, sending the
degenerate combinations onto legal ones.

**Log-uniform prior.** Sampling uniformly in the logarithm, appropriate when a
parameter's effect is multiplicative rather than additive.

**Block family.** Which residual block the backbone stacks: ResNet or ResNeXt.

**Head fusion.** Whether the embedding head reads only the last backbone stage
or fuses all stages.

**Pooling ops.** Which per-stage statistics the head pools — mean alone, or
mean, max and standard deviation.

**Silhouette floor.** The lower bound on the achievable silhouette that a loss
hyper-parameter *implies*, obtained by asking what geometry satisfies the hinge
everywhere. A diagnostic, not a target.

**Simplex ETF.** The equinormed, equiangular configuration with pairwise cosine
$-1/(K-1)$ that the separation term targets.

**Size mix.** The distribution of parameter counts the search actually draws.

---

## 2.1 Which JSON field to edit

The sections below name each axis by its **axis name**, which is what appears in
the trial log and the winner report. The **configuration field** you edit has a
different name. This table is the mapping; getting it wrong is the most likely
way to edit a range and see no effect.

| # | Axis name (in logs) | JSON field to edit | Block |
|---|---|---|---|
| 1 | `depth_exponent` | `depth_exponent_range` | `search` |
| 2 | `width_multiplier` | `width_multiplier_range` | `search` |
| 3 | `block_family` | `block_family_choices` | `search` |
| 4 | `embedding_size` | `embedding_size_range` | `search` |
| 5 | `lr` | `lr_range` | `search` |
| 6 | `one_minus_beta1` | `one_minus_beta1_range` | `search` |
| 7 | `one_minus_beta2` | `one_minus_beta2_range` | `search` |
| 8 | `weight_decay` | `weight_decay_range` | **`regularization`** (not `search`) |
| 9 | `dropout` | `dropout_range` | **`regularization`** |
| 10 | `margin` | `margin_range` | `search` |
| 11 | `angular_alpha_deg` | `angular_alpha_deg_range` | `search` |
| 12 | `lambda_sep` | `lambda_sep_range` | `search` |
| 13 | `mining_strategy` | `mining_strategy_choices` | `search` |
| 14 | `loss_type` | `loss_type_choices` | `search` |
| 15 | `strict_semihard` | `strict_semihard_choices` | `search` |
| 16 | `head_fusion` | `head_fusion_choices` | `search` |
| 17 | `head_pool_ops` | `head_pool_ops_choices` | `search` |

Two entries are in the `regularization` block rather than `search`. For
`weight_decay` this is a genuine trap, because a same-named field also exists in
`search` and is **inert** — see §3.7.

The preferred way to change any of these is to regenerate the configuration with
`hpc/make_joint_search_config.py` rather than hand-editing, so that the budget
arithmetic is recomputed and printed alongside the change.

---

## 3. The seventeen axes

Each subsection states, in order: what the axis is, how it is encoded, when it
is active, the theory of the range, and the consequences of changing it.

### 3.1 Axis 1 — `depth_exponent` ($d$)

**Meaning.** The backbone stacks $B = 2^{d}$ residual blocks in total,
allocated across stages by the width schedule. This is a RegNet-style
parameterisation: depth is not fixed at four stages but grows with $d$.

**Encoding.** `Integer(2, 5)`, one surrogate column. **Always active.**

**Configured range** $[2, 5]$. **Dataclass default** $[3, 6]$ — the config
*lowers* both ends.

**Theory of the range.** $B = 2^{d}$ means the configured range spans
$B \in \{4, 8, 16, 32\}$: an eightfold span in block count, and because
parameter count grows faster than linearly in depth once the width schedule
compounds, a much larger span in $\theta$. Measured over 300 draws at the
configured ranges, $\theta$ spans 0.01 M to 31.6 M.

**This is the single most consequential axis for cost.** The measured size mix
is strongly right-skewed — mean 4.46 M against median 0.71 M parameters, a
factor of 6.26 — and wall clock scales with the *mean*, not the median
(THEORY §3.9.1). Depth is near-uniformly sampled (26 / 22 / 23 / 28 % over
$d = 2,3,4,5$), so roughly a quarter of all trials will draw the most expensive
corner.

**Consequences of changing it.**
- *Raising the upper bound to 6* doubles the maximum block count to 64 and will
  push $\mathbb{E}[\Theta]$ up by a large and non-linear factor. Do not do this
  without re-running the Stage 7 gate.
- *Lowering the upper bound to 4* is the cheapest single intervention if the
  wall-clock gate fails. It removes the entire right tail of the size mix,
  which is where the expectation lives.
- *Raising the lower bound* to 3 removes the 4-block models. These are cheap
  and may be genuinely competitive on a three-class problem with a 6-dimensional
  latent space; removing them narrows the study for little saving.

**Failure mode.** None internal — every value builds. The failure is
budgetary, and it is silent until the job hits the wall.

### 3.2 Axis 2 — `width_multiplier` ($w$)

**Meaning.** The RegNet-style slope of the per-block width schedule. Combined
with the stem width and the group width, it determines the width of every
block and hence most of $\theta$ at fixed depth.

**Encoding.** `Real(1.5, 5.0)`, uniform prior, one column. **Always active.**

**Configured range** $[1.5, 5.0]$. **Dataclass default** $[1.5, 3.0]$ — the
config substantially *widens* the top.

**Theory of the range.** The constructor requires $w > 1.0$ strictly, so the
lower bound cannot go to 1. The upper bound was widened because the screening
tier explored only up to 3.0 and the joint search is meant to be less
constrained than the phase it replaces.

**Prior choice.** Uniform, not log-uniform, which is a defensible but not
obvious call: width enters the parameter count multiplicatively per stage, so
an argument exists for a log prior. The uniform prior places half its mass
above $w = 3.25$, i.e. above the entire screening range. Combined with a
uniform depth prior, this is part of why the size mix is skewed.

**Consequences.** Narrowing the top to 3.0 recovers the screening range and
cuts $\mathbb{E}[\Theta]$ appreciably. Widening further compounds with depth
multiplicatively and is the second-fastest way to break the wall-clock gate.

### 3.3 Axis 3 — `block_family`

**Meaning.** $0$ selects the ResNet block, $1$ selects the ResNeXt block
(grouped convolution).

**Encoding.** `Integer(0, 1)`, one column. **Always active.**

**Why Integer and not Categorical.** A two-level one-hot is exactly redundant —
the second column satisfies $x^{(2)} = 1 - x^{(1)}$ — so encoding as an integer
saves one surrogate column at no cost in expressiveness and imposes no false
ordering (there is only one ordering of two levels and it carries no metric
content).

**A caveat you should know about.** The repository carries a warning that this
axis must never be a real-valued dimension, since a sampled $0.37$ cannot index
a block table. The design justified the integer encoding by asserting that
integer axes return native Python integers. **They do not** — the sampler
returns NumPy integer scalars. The substance survives (a NumPy integer indexes
correctly where a float raises) so the encoding stands, but the stated reason
was wrong, and NumPy scalars are not JSON-serialisable. Points are normalised
to native types at the projection boundary. If you ever change this axis's
type, the smoke test asserts index-usability and JSON round-trip, not the exact
type.

**Consequences.** Freezing to one family — `[0]` or `[1]` — is legal: the
implementation pins a single-level choice list as a one-value categorical,
because the search library rejects a degenerate `Integer(v, v)`. Freezing saves
one column and halves the architecture space.

### 3.4 Axis 4 — `embedding_size` ($E$)

**Meaning.** Dimension of the L2-normalised output embedding.

**Encoding.** `Integer(8, 16)`, one column. **Always active.**

**Theory of the range.** Two lower bounds constrain it. Geometrically, $K$
class means can only form a simplex ETF if $E \ge K - 1$; at $C = 3$ that is
$E \ge 2$, not binding. Practically, the range's lower end must leave room for
the head's pooled statistics; $E = 8$ is comfortably above the geometric floor.
The upper end is modest because the latent generative space has six axes, of
which three carry the labels — an embedding much larger than the intrinsic
dimension buys capacity to memorise rather than to separate.

**Interaction worth knowing.** $E$ interacts with `head_pool_ops` (§3.15): the
three-statistic head produces three times the pooled features before the final
projection, so at fixed $E$ the projection is doing more compression.

### 3.5 Axis 5 — `lr` ($\eta_{\mathrm{lr}}$)

**Meaning.** Adam learning rate.

**Encoding.** `Real(1e-4, 0.2)`, **log-uniform** prior, one column. **Always
active.**

**Theory of the range.** Learning rate acts multiplicatively on the update, so
a log prior is correct: it places equal mass on $[10^{-4}, 10^{-3}]$ and on
$[10^{-2}, 10^{-1}]$. The range spans three and a third decades, which is wide.
The upper end, $0.2$, is aggressive for Adam and will produce divergent or
degenerate runs.

**This is intentional and it is why the failure policy matters.** A trial that
diverges does not crash the study: it scores the finite penalty and the
surrogate learns to avoid that region. Widening a range is cheap precisely
because failures are absorbed. What it costs is trials.

**Consequences.** Narrowing to $[10^{-4}, 10^{-2}]$ would concentrate the
budget on the plausible region at the cost of assuming the answer. At 300
trials over 22 columns the argument for narrowing is stronger than it would be
at 1000.

**Interaction.** With `use_scheduler = False` (Document 2), the learning rate is
constant for the whole run. There is no warm-up and no decay, so $\eta_{lr}$
must be simultaneously small enough to be stable at step 1 and large enough to
make progress by step $T$. That is a real constraint, and it argues for the
lower half of the range.

### 3.6 Axes 6–7 — `one_minus_beta1`, `one_minus_beta2` ($u_1, u_2$)

**Meaning.** Adam's exponential decay rates, parameterised as
$\beta_i = 1 - u_i$.

**Encoding.** `Real`, **log-uniform**, one column each. **Always active.**
$u_1 \in [10^{-2}, 10^{-1}]$, so $\beta_1 \in [0.9, 0.99]$.
$u_2 \in [10^{-4}, 10^{-2}]$, so $\beta_2 \in [0.99, 0.9999]$.

**Why the $1 - \beta$ parameterisation, and why it matters.** The quantity with
physical meaning is the averaging timescale, roughly $1/(1-\beta)$ steps. In
$\beta$ coordinates the interesting region is compressed against 1 —
$0.999$ and $0.9999$ are visually adjacent but differ tenfold in timescale — and
a uniform prior on $\beta$ would put almost no mass there. Searching
$u = 1 - \beta$ on a log scale makes the prior uniform *in timescale*, which is
the correct geometry. This is the single most defensible prior choice in the
space.

**Consequences.** $\beta_2$ near $0.9999$ means a 10 000-step second-moment
memory, which is comparable to the whole planned run of $T = 10\,000$ steps.
That is not necessarily wrong, but it means the second-moment estimate never
fully forgets initialisation. If runs at the top of the $\beta_2$ range behave
oddly, that is the reason to suspect.

### 3.7 Axis 8 — `weight_decay` ($\lambda_{\mathrm{wd}}$) — **and a trap**

**Meaning.** Decoupled weight decay.

**Encoding.** `Real`, **log-uniform**, one column. **Always active.**

**Which range is used.** This is the trap. The configuration contains **two**
weight-decay ranges:

- `search.weight_decay_range = [1e-4, 1e-2]`
- `regularization.weight_decay_range = [1e-5, 1e-2]`

The joint condition space deliberately takes the **regularization** range,
because it is the wider of the two and because in the staged pipeline that
block owned this parameter. **`search.weight_decay_range` is therefore INERT
under `search_mode = "joint_conditions"`.** Editing it will change nothing and
will look like it should have. Edit `regularization.weight_decay_range`.

**Theory.** The realised range $[10^{-5}, 10^{-2}]$ spans three decades on a
log prior. In a metric-learning setting with L2-normalised outputs, weight
decay acts mostly on the interior layers; its effect on the embedding geometry
is indirect.

### 3.8 Axis 9 — `dropout` ($p_{\mathrm{drop}}$)

**Meaning.** Dropout probability in the backbone.

**Encoding.** `Real(0.0, 0.3)`, uniform prior, one column. **Always active.**
Read from `regularization.dropout_range`.

**Theory of the position in the design.** In the staged pipeline, dropout and
weight decay were tuned *last*, in a separate regularisation phase, after the
architecture and loss were frozen. The joint search makes dropout free from
trial 0. This matters because the screening's central finding was that head
geometries differ chiefly in **generalisation gap** — and regularisation
strength is exactly the knob that trades train fit against generalisation gap.
Freezing the head before the regulariser was tuned would confound the two.

**Consequences.** The uniform prior on $[0, 0.3]$ places a third of its mass
below $0.1$. Zero dropout is reachable and is a legitimate outcome.

### 3.9 Axis 10 — `margin` ($m_{\cos}$)

**Meaning.** The triplet hinge margin, in cosine-distance units. The loss
internals use squared-Euclidean distance, so the configured value enters as
$2m_{\cos}$.

**Encoding.** `Real(0.1, 1.0)`, uniform, one column.
**Active only when $\ell = \texttt{triplet}$.** Under `joint` and `joint_sep`
it is *fixed*, not unused: the composite loss still reads it for its hinge, at
whatever the base configuration states (0.3).

**Theory: the implied silhouette floor.** A margin asserts a geometry. If every
triple satisfies the hinge, the embedding must satisfy a lower bound on its
cosine silhouette; the preflight computes it. At $C = 3$ the configured base
$m_{\cos} = 0.3$ implies a silhouette floor of $0.2$, and the **top of the
searched range**, $m_{\cos} = 1.0$, implies a floor of $0.667$.

**Read that number carefully.** A silhouette of $0.667$ on this benchmark is
very high. A trial drawing $m_{\cos}$ near the top of the range is asserting a
geometry that may be unreachable, in which case the hinge is active on
essentially every triple for the entire run and the loss never approaches zero.
That is not a crash — it is a legitimately explorable region — but if the
search concentrates near $m_{\cos} = 1$ and the results are poor, this is the
explanation.

**Why it is never searched together with $\alpha$.** Both bind on the same
scalar functional, the within/between distance ratio, so the pair is close to
non-identifiable and traces a ridge. See THEORY §3.3.3. The activity mask
enforces the separation structurally.

### 3.10 Axis 11 — `angular_alpha_deg` ($\alpha$)

**Meaning.** The half-angle of the angular constraint, in degrees. The hinge is
satisfied when $D_{ap} < 4\tan^2(\alpha)\, D_{nc}$, with $x_c$ the
anchor-positive midpoint.

**Encoding.** `Real(2, 20)`, uniform, one column.
**Active when $\ell \in \{\texttt{joint}, \texttt{joint\_sep}\}$.**

**Theory of the range, and the important asymmetry.** The implied silhouette
floor is *monotonically decreasing* in $\alpha$: a **smaller** $\alpha$ is a
**stronger** constraint. At the bottom of the range, $\alpha = 2°$, the
constraint demands near-total within-class collapse. At $\alpha = 20°$ it is
mild.

This inverts the intuition that a bigger number means a stronger setting, and
it is worth internalising before reading any result: a trial that wins at
$\alpha = 3°$ is a trial that succeeded under a *severe* constraint.

**Consequences.** The uniform prior over $[2, 20]$ places equal mass on
$[2, 4]$ and on $[18, 20]$, which are very different regimes. If you want to
concentrate on the mild region, raise the lower bound; the effect on the
implied floor is strongly non-linear near the bottom because
$\tan^2$ varies fast there.

**Interaction with mining.** Small $\alpha$ is collapse-seeking, while
easy-positive mining exists specifically to *prevent* over-clustering. The
space therefore contains configurations that are internally opposed. They are
legal and may even win; see THEORY §3.7.5 before interpreting such a winner.

### 3.11 Axis 12 — `lambda_sep` ($\lambda_{\mathrm{sep}}$)

**Meaning.** The **asymptotic** weight on the centroid-separation term — the
value reached at the end of the warm-up, not the value applied at step 0.

**Encoding.** `Real(1e-2, 20)`, **log-uniform**, one column.
**Active only when $\ell = \texttt{joint\_sep}$.**

**Configured range** $[10^{-2}, 20]$ — three and a third decades. **Dataclass
default** $[10^{-3}, 1.0]$. The config both raises the top twentyfold and
narrows the bottom.

**Theory of the top of the range.** $\lambda_{\mathrm{sep}} = 20$ is large
relative to $\mathcal{L}_{\mathrm{joint}}$, so the top of the range represents
"the separation term dominates". That is deliberate: the whole point of making
it searchable is to find out whether the term helps at all, and a range that
cannot express dominance cannot answer that.

**The warm-up interaction you must not forget.** The weight actually applied is
$\lambda_{\mathrm{sep}}(t) = \lambda_{\mathrm{sep}}\min(1, t/(\tau T))$. The
total dose over training is $\lambda_{\mathrm{sep}} T (1 - \tau/2)$, so
$\lambda_{\mathrm{sep}}$ and $\tau$ trade off almost multiplicatively. This is
why $\tau$ is **fixed** and only $\lambda_{\mathrm{sep}}$ is searched — a
search over both would move along a ridge. See THEORY §3.6.2.

**The clamp constraint.** When $\ell \ne \texttt{joint\_sep}$ this axis is
inactive and the configuration keeps its base value. That base value **must** be
`0.1`, the dataclass default, or every `triplet` and `joint` trial fires an
"inert lambda_sep" warning. The preflight checks this. See Document 2, §3.3.

### 3.12 Axis 13 — `mining_strategy` ($m$)

**Meaning.** Which triples the loss is evaluated on.

- `hard` — the triples that most violate the desired ordering, i.e. those with
  $D_{an} < D_{ap}$.
- `easy_positive` — for each anchor, the *most similar* same-class example as
  positive. Motivated as a remedy for over-clustering: constraining only the
  nearest same-class neighbour lets a class occupy a manifold rather than
  contracting to a point.
- `easy_pos_semihard_neg` — easy positive paired with a semi-hard negative.

**Encoding.** `Categorical`, 3 levels, **3 surrogate columns.** **Always
active.**

**Theory.** These are not three points on a scale; they are qualitatively
opposed strategies. Hard mining pushes toward tight clusters; easy-positive
mining explicitly resists that. The categorical encoding is correct precisely
because no ordering exists.

**Cost note.** This axis and `loss_type` are the only two three-level
categoricals, and together they account for 6 of the 22 surrogate columns —
nearly a third of the surrogate's input dimension for two of seventeen axes.
Freezing either to a single level is therefore the largest available saving in
column count.

**Interaction with $\Pi$.** `hard` is the mining strategy whose combination
with the strict filter is degenerate, so drawing `hard` forces $s = 0$
(§3.14). One consequence is that the conditions containing `hard` receive
*double* the sampling mass of the others — see §4.

### 3.13 Axis 14 — `loss_type` ($\ell$)

**Meaning.** Which objective is optimised.

- `triplet` — the plain hinge. Reads $m_{\cos}$ only.
- `joint` — triplet hinge plus angular constraint. Reads $\alpha$; margin is
  fixed.
- `joint_sep` — the above plus the centroid-separation term. Reads $\alpha$,
  $\lambda_{\mathrm{sep}}$, $\kappa$.

**Encoding.** `Categorical`, 3 levels, **3 surrogate columns.** **Always
active.** It is the axis that determines every other axis's activity, via
$A(\ell)$.

**The most important caveat in this document.** The Gaussian process allocates
trials adaptively. There is **no guarantee** that `triplet` receives a
meaningful share of the 300 trials. Consequently this search returns a *tuned
configuration*; it does **not** establish that the composite objective beats the
triplet baseline at matched budget. If the baseline comparison is the
scientific claim you need, it requires its own matched-budget run. This is
recorded as open point 1 in the theory document and it is not a detail.

### 3.14 Axis 15 — `strict_semihard` ($s$)

**Meaning.** Whether the strict semi-hard filter is applied after mining,
retaining only triples with $D_{ap} < D_{an}$ inside the margin band.

**Encoding.** `Integer(0, 1)`, one column.
**Active only when $\ell \ne \texttt{triplet}$** — the plain triplet loss has
no such filter, so the flag is written and never read.

**This is the only axis $\Pi$ ever moves.** The projection sets $s \leftarrow 0$
in two cases: when $\ell = \texttt{triplet}$ (inert), and when
$m = \texttt{hard}$ (degenerate — hard mining returns $D_{an} < D_{ap}$ and the
filter demands the exact opposite, so the surviving set is provably empty;
measured, 16 814 mined and 0 surviving). $\Pi$ never alters $m$ or $\ell$, so a
projected trial is never silently converted into a different experiment.

**How to read a projected trial.** Every trial record carries `projected`,
`raw_condition` and `condition`. Roughly a third of random draws are projected.
A projected trial is a *duplicate observation*, not a wasted one — the
surrogate handles duplicates natively — but the log is what lets you tell the
difference between a duplicate and noise.

### 3.15 Axes 16–17 — `head_fusion`, `head_pool_ops` ($h$)

**Meaning.**
- `head_fusion`: $0$ = the head reads **only the last backbone stage**; $1$ =
  it **fuses all stages**.
- `head_pool_ops`: $0$ = pool the **mean** only; $1$ = pool **mean, max and
  standard deviation**. The head reorders the ops canonically, so only the set
  matters.

**Encoding.** `Integer(0, 1)` each, one column each. **Always active.**

**Theory, and why these are searched rather than fixed.** The screening's
central finding was that the four head geometries differ chiefly in
**generalisation gap** rather than in training fit, and that the interaction
between the strict filter and the head was the largest effect measured. An
interaction cannot be found by a staged search that fixes the head before the
filter is a variable. This is the primary justification for the whole
joint-condition design.

**Consequences.** `head_fusion = 1` with `head_pool_ops = 1` is the largest
head: all stages, three statistics each, so the pre-projection feature count is
(stages × 3). At high `depth_exponent` this is where the parameter count peaks.
The four combinations are equiprobable, so a quarter of trials draw the largest.

### 3.16 *Formerly* axis 18 — `sep_centre_means` — **REMOVED**

> Axis number 18 is now occupied by `sep_warmup_frac` (§3.17). This
> section documents the axis that vacated the slot, not the one that
> holds it.

This axis existed in Revision 1 and has been deleted. It selected whether
$\mathcal{L}_{\mathrm{sep}}$ was built from centred class means
($\hat\mu_c \propto \mu_c - \mu_G$) or raw ones
($\hat\mu_c \propto \mu_c$).

**Why it is gone rather than frozen.** Centring then normalising is invariant to
translation *and* scale, so the term measured only the *shape* of the simplex of
class means and never its *size*. Every equilateral arrangement scored zero,
including an arbitrarily tiny one — so three classes collapsed into a cap at raw
pairwise cosine $+0.9994$ scored $0.000035$, while the raw form scored $2.248$.
The centred form is structurally blind to the one failure the term exists to
prevent.

And the raw form imposes nothing extra in exchange: for unit vectors with all
pairwise inner products $\rho$, $\lVert\sum_c v_c\rVert^2 = K + K(K-1)\rho$,
which is exactly zero at $\rho = -1/(K-1)$. Equiangularity at the ETF target
*already implies* the directions sum to zero. There was therefore never a
trade-off to search — one option was simply broken.

**What this means in practice.**
`CentroidSeparationLoss` takes no formulation argument and raises `TypeError` if
given one. `train.sep_centre_means` and `search.sep_centre_means_choices` remain
in the configuration schema so archived files parse, are read by nothing, and
warn if set. $A(\texttt{joint\_sep})$ is now
$\{\alpha, \lambda_{\mathrm{sep}}, \tau\}$ (§3.17), and `sep_centred` stays in
the per-epoch history pinned to 0 so archived readers do not break.

**One thing to weigh:** the raw form is a considerably *stronger* constraint
than the shape-only term it replaces, and the searched range of
$\lambda_{\mathrm{sep}}$ (§3.11) was chosen against the weaker one. See §5.

---

### 3.17 Axis 18 — `sep_warmup_frac` ($\tau$)

| | |
|---|---|
| **Config field** | `search.sep_warmup_frac_range` |
| **Type / prior** | `Real`, **uniform** |
| **Requested range** | $(0.0,\ 0.5)$ — a *request*, not a guarantee |
| **Effective range** | $(0.0,\ \tau_{\max})$, $\tau_{\max}$ **derived** |
| **Active when** | $\ell = \texttt{joint\_sep}$ only |
| **Clamp constant** | `train.sep_warmup_frac = 0.0` (the `TrainConfig` default) |

**Meaning.** $\tau$ is the fraction of the planned step budget over which the
separation weight ramps linearly from $0$ to $\lambda_{\mathrm{sep}}$:

$$
\lambda_{\mathrm{sep}}(t) = \lambda_{\mathrm{sep}}\, g(t),
\qquad
g(t) = \min\!\left(1,\ \frac{t}{\tau T}\right),
\qquad
T = E_{\max}\, n_{\mathrm{b}},
\tag{17}
$$

for all $t \in \{0, \dots, T\}$, with $g(t) \equiv 1$ at $\tau = 0$. Full
weight is reached at step $\tau T$.

**Why it is searched, having been fixed.** Document II §3.3 fixed $\tau = 0.3$
because the schedule integrates to $\lambda_{\mathrm{sep}} T (1 - \tau/2)$, so
$\tau$ and $\lambda_{\mathrm{sep}}$ appeared to trade off multiplicatively and
trace a ridge. **The integral is right; the inference is not.** Two settings
with equal dose have different *terminal* weights $\lambda_{\mathrm{sep}} g(T)$,
and the epoch selector usually picks a late epoch, so the terminal weight
plausibly governs the converged geometry more than the integral does. The dose
is not a sufficient statistic for $(\lambda_{\mathrm{sep}}, \tau)$, and the
ridge does not close. THEORY §3.6.2 carries the superseded derivation.

**Why the prior is uniform, not log-uniform.** $\tau = 0$ must be reachable: it
is the no-warm-up control arm, handled by `sep_warmup_scale` as constant full
weight, and a log prior cannot contain zero. This is the same reason `dropout`
is uniform (§3.9).

**Why the upper bound is derived rather than configured.** A run reaches full
weight only if it completes at least $\tau T$ of its planned steps, i.e. only if

$$
\frac{\tilde{e}}{E_{\max}} \;\ge\; \tau ,
\tag{18}
$$

where $\tilde{e}$ is the number of epochs the run actually completes. Early
stopping with patience $P$ makes the earliest possible stop roughly $P+1$
epochs, so $\tilde{e} \ge P$ is the conservative worst case. Requiring (18) to
hold **even for the shortest run the stopping rule permits** gives

$$
\boxed{\ \tau_{\max} \;=\; \min\!\left(1,\ \frac{P}{E_{\max}}\right)\ }
\tag{19}
$$

$\tau_{\max} = 0.40$ at the shipped $100/40$, and $0.333$ at the alternative
$60/20$. Above the cap, "large $\tau$ wins" would be indistinguishable from
"the separation term was effectively off" — a finding about `joint` versus
`joint_sep`, which is *already* axis 14, reached by a confounded route. The cap
makes that region **unsamplable** rather than merely discouraged. And because
$\tau_{\max}$ depends on $P$ and $E_{\max}$, a static config field would go
stale the moment either changed, which is exactly what the wall-clock gate may
force.

Implementation: `condition_space.sep_warmup_frac_cap` owns the formula;
`search._loss_dims` clips the requested range to it and emits a `RuntimeWarning`
naming the clip; a lower bound at or above the cap raises `ValueError` rather
than building a degenerate dimension.

**Position in the axis order.** Appended **last**, not slotted in beside the
other loss hyper-parameters at 10–12. Axis order is load-bearing at a fixed
`gp_random_state` — the initial design is a function of the ordering — and no
study had been run against the 17-axis order, so appending is the minimal-diff
choice.

**What it costs.** 18 axes over 22 columns at $N_{\mathrm{calls}} = 300$ is
thinner than 17 over 21. No analysis quantifies the loss. $\tau$ is active in
only about a third of trials (`joint_sep` only), so the marginal cost is
probably small — but that is inference, not measurement. Cells, $\Pi$ and the
coverage arithmetic are **unaffected**: $\tau$ is not part of the cell
definition, and coverage is still 52/52 at $N = 300$.

**What is logged.** The per-trial record carries the realised dose
$\lambda_{\mathrm{sep}} T (1 - \tau/2)$ *and* the terminal weight
$\lambda_{\mathrm{sep}} g(T)$, because without both the post-hoc question "did
$\tau$ matter through the integral or through the shape?" cannot be answered.
When $n_{\mathrm{b}} = 0$ the trainer derives $T$ at build time, so the dose is
recorded as `null` and the $T$-free ratio $\lambda_{\mathrm{sep}}(1 - \tau/2)$
is recorded instead.

**Boundary caveat.** At exactly $\tau = \tau_{\max}$ a worst-case run reaches
full weight at its final step and spends *zero* steps there. A safety factor
$\tau_{\max} = s\,P/E_{\max}$ with $s \approx 0.8$ would remove this; $s = 1$
is shipped because that is what "fully open at the last training point" states,
but the choice is not derived.

---

## 4. Summary of results

**R1. Axis and column budget.** 18 declared axes; 22 surrogate columns —
10 Real + 6 Integer + two 3-level Categoricals expanding to 6. The four binaries
are Integer rather than Categorical, saving four columns, because a two-level
one-hot is exactly redundant. §3, §3.3.

**R2. Activity.** Nine axes are always active. Four loss hyper-parameters are
governed by $A(\ell)$: `margin` under `triplet` only; `angular_alpha_deg` under
the two composite losses; `lambda_sep` and `sep_warmup_frac` under `joint_sep`
only; `strict_semihard` under non-`triplet` only. Up to **three** columns are
exactly flat on any given trial. §3.10–§3.17.

**R3. `margin` and `angular_alpha_deg` are never both active,** because both
bind on the within/between distance ratio and would trace a ridge. Enforced
structurally by $A(\ell)$. §3.9, §3.10.

**R4. Cost is dominated by two axes.** `depth_exponent` ($B = 2^d$, so $[2,5]$
spans 4 to 32 blocks) and `width_multiplier` (widened to 5.0) together produce
a size mix with mean 4.46 M against median 0.71 M parameters. Wall clock scales
with the mean. §3.1, §3.2.

**R5. Two ranges invert intuition.** Smaller $\alpha$ is a *stronger*
constraint (§3.10); $m_{\cos}$ at the top of its range implies a silhouette
floor of 0.667, which may be unreachable (§3.9).

**R6. One range in the configuration is inert.**
`search.weight_decay_range` is not read under `joint_conditions`; the joint
space takes `regularization.weight_decay_range`. §3.7.

**R7. $\Pi$ moves only `strict_semihard`,** never the mining strategy or the
loss type, and it fires on roughly a third of random draws. §3.14.

**R8. The projection makes the conditions it merges twice as likely.** Cells
containing `triplet`, or `hard` with a composite loss, carry probability
$1/36$; the other 32 carry $1/72$. Expected coverage after $N$ draws is
$20(1-1/36)^N + 32(1-1/72)^N$ unvisited cells. THEORY §3.5.

**R9. Freezing a three-level categorical is the largest available saving**, at
3 columns each; `mining_strategy` and `loss_type` account for 6 of 22 columns.
§3.12.

**R10. The search does not answer the baseline question.** Adaptive allocation
means `triplet` may receive very few trials. §3.13.

---

## 5. Open points, caveats and assumptions

1. **The uniform prior on `width_multiplier` is arguable.** Width enters the
   parameter count multiplicatively, which is an argument for a log prior. The
   uniform prior places half its mass above $w = 3.25$, above the entire
   screening range, and contributes materially to the skew of the size mix.
   Not tested either way.

2. **The implied silhouette floors are derived from the hinge, not measured.**
   They say what geometry *would* satisfy the constraint everywhere. Whether a
   run actually attains it is a separate question, and the floors should be read
   as diagnostics rather than predictions.

3. **`lr` upper bound of 0.2 is aggressive** and will produce failed or
   degenerate trials. This is absorbed by the finite failure penalty rather
   than being a problem, but it consumes trials, and no analysis says how many.

4. **`lr` is constant for the whole run** (`use_scheduler = False`), so a single
   value must serve both the initial transient and the late phase. Whether the
   search compensates by preferring the low end is an empirical question this
   study will answer incidentally.

5. **$\beta_2$ near the top of its range gives a memory comparable to the whole
   run** ($1/u_2 = 10^4$ steps against $T = 10^4$). Not necessarily wrong;
   worth suspecting if the top of that range behaves oddly.

6. **The searched range of `lambda_sep` predates the removal of the centring**
   (§3.16). $[10^{-2}, 20]$ was chosen against the shape-only term; the raw
   form is a harder constraint and the useful range may sit lower. Flagged,
   not changed. Every archived `joint_sep` result is likewise about the
   shape-only penalty and does not transfer.

7. **The $K = 3$ vacuity proof is my own** — a two-line
   argument, verified numerically against the repository implementation, but
   not found stated in the literature consulted. Check it independently before
   it appears in a thesis.

7. **The cost of inactive (flat) columns is unquantified.** Up to 2 of 21 are
   flat per trial; ARD length-scales absorb flat directions in principle, but
   how much of a 300-trial budget that absorption costs is not estimated.

8. **Ranges were widened relative to the dataclass defaults without
   measurement.** `depth_exponent`, `width_multiplier` and `lambda_sep` all
   differ from their defaults. The widening is a design judgement about
   exploring less constrained regions than the screening did, not a conclusion
   from data.

---

## 6. References

- **THEORY_joint_condition_search.md** (this project) — derivations for the
  legality projection, the activity mask and the identifiability argument, the
  coverage arithmetic, the warm-up dose integral, and the $K = 3$ degeneracy
  proof. All theoretical claims in this document are cross-referenced there
  rather than re-derived.
- **Papyan, V., Han, X. Y., Donoho, D. L. (2020).** *Prevalence of neural
  collapse during the terminal phase of deep learning training.* PNAS 117(40),
  24652–24663. Retrieved via PubMed; full text read from PubMed Central
  (PMC7547234), [DOI](https://doi.org/10.1073/pnas.2015509117). Source for the
  simplex-ETF definition and for NC2 being a statement about globally centred
  class means — the premise of §3.16.
- **Xuan, H., Stylianou, A., Pless, R.** *Improved Embeddings with Easy Positive
  Triplet Mining.* Full text in the project knowledge base. Source for the
  easy-positive mining definition and its anti-over-clustering motivation
  (§3.12), and for the semi-hard negative condition used in §3.14.
- **Repository source**, read directly: `backbone.py` (block family, head
  fusion, pooling ops, $B = 2^d$), `search.py` (space construction and the
  activity mask), `config.py` (ranges and validation), `dsn_joint_loss.py`
  (the composite loss and the warm-up), `hpc/preflight_config.py` (implied
  silhouette floors).
- **Measured in this work:** the size-mix distribution of §3.1 and the depth
  sampling frequencies; the projection rate of §3.14; the column arithmetic.
- **From the predecessor handoffs, not re-measured:** the 16 814-mined /
  0-surviving figure (§3.14) and the screening finding on generalisation gap
  and the filter × head interaction (§3.15).
