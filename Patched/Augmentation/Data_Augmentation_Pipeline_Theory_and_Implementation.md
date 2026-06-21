# Data Augmentation & Contrastive Pipeline — Theory and Implementation

*Reference document for Topic 1 of the 1D-CNN summary-network refactor (MEA neuronal-activity
embeddings). Covers the mathematical formulation of every transform, the positive/negative split,
the condition-level contrastive label scheme (option b), the software architecture, the HPC and
reproducibility design, usage, and validation.*

**Scope.** This document describes the *data/augmentation* stage only. It produces a labelled
contrastive batch $(\mathbf{X}, \mathbf{y})$ that is consumed by the encoder (Topic 2) and the
triplet loss / optimization (Topic 3). Those consumers are summarized where needed for grounding,
but their full treatment lives in their own documents.

A symbol glossary is given in §12; every symbol is also introduced at first use.

---

## 1. Purpose and scientific goal

The upstream pipeline converts each microelectrode-array (MEA) recording of a neuronal culture into
a one-dimensional **smoothed cumulative instantaneous firing rate (IFR)** trace. We denote such a
trace by

$$
r_c[n], \qquad n = 0, 1, \dots, L_c - 1,
$$

where $c$ indexes the source recording (a *well/culture*), $L_c$ is its length in samples, and the
trace is sampled at rate $f_s$ (Hz) after downsampling. Each recording carries a **condition label**

$$
\ell_c \in \{0, 1\}, \qquad 0 \equiv \text{control}, \quad 1 \equiv \text{pathological}.
$$

The scientific objective is to learn an **embedding map**

$$
f_\theta : \mathbb{R}^{T} \to \mathbb{R}^{E}, \qquad x \mapsto f_\theta(x),
$$

parameterized by $\theta$, that maps a fixed-length **window** $x \in \mathbb{R}^{T}$ ($T$ = window
length in samples, $E$ = embedding dimension) into a space where windows of the *same condition*
cluster together and windows of *different conditions* are pushed apart — so that downstream
clustering (e.g. control vs pathological) is well separated, and the embedding generalizes to unseen
cultures.

Because per-window condition labels are the *only* supervision available, the learning signal is
provided by **metric learning** (a triplet objective) over a batch whose structure is manufactured
by **data augmentation**. The augmentation must produce, for each window, variants that the encoder
should treat as *the same* (positives) and variants it should treat as *different* (negatives),
encoding the right invariances and the right sensitivities.

---

## 2. Design principles

The pipeline is built on four principles (the project's coding directives), which the rest of the
document refers back to:

1. **Leverage the ecosystem.** Spline interpolation uses `scipy.interpolate.CubicSpline`; batching,
   gathering, and tensor ops use PyTorch; nothing numerical is re-implemented from scratch.
2. **Separation of concerns.** Transforms, batch assembly, plotting, and the front-end entry point
   live in distinct modules; the encoder/loss never appear in the data path.
3. **Alignment before logic.** The two scientific forks — the positive/negative *split rule* and the
   contrastive *label scheme* — were fixed with the user before implementation (§5, §6).
4. **Validated algorithms.** Every module ships with a self-contained smoke test, and each script is
   controlled twice (§10).

A fifth, cross-cutting constraint is **HPC deployment** (davinci-1, Leonardo): augmentation runs on
CPU inside DataLoader worker processes, the pipeline is config-driven and headless, and
reproducibility is guaranteed for a fixed `(seed, num_workers)` (§8).

---

## 3. From trace to window

A window is a contiguous slice of a trace of fixed length

$$
T = \mathrm{cp2}(\,\lfloor \tau_w f_s \rceil\,),
$$

where $\tau_w$ is the window length in seconds, $\lfloor\cdot\rceil$ denotes rounding, and
$\mathrm{cp2}(\cdot)$ returns the power of two closest (in $\log_2$) to its argument. Snapping $T$ to
a power of two keeps the strided downsampling inside the encoder clean. *Flagged dependency:* the
choice of $\mathrm{cp2}$ (nearest vs floor) must match the encoder engine; the implementation uses
*nearest* and documents how to switch to *floor*.

From a trace $r_c$ of length $L_c$, windows are extracted at stride $S_w$ (samples):

$$
x^{(c, j)}(t) = r_c\big[\,j S_w + t\,\big], \qquad t = 0, \dots, T-1, \qquad
j = 0, 1, \dots, \Big\lfloor \tfrac{L_c - T}{S_w} \Big\rfloor,
$$

and each window inherits the condition label $\ell_c$. Choosing $S_w < T$ produces **overlapping**
windows, which raises the number of distinct windows (a deliberate remedy for the low-diversity
problem identified earlier, where non-overlapping windowing yielded only a handful of windows per
trace). For each fixed trace $c$, all of its windows share the same $\ell_c$.

Throughout the rest of the document we suppress the $(c,j)$ superscript and write a generic window as
$x(t)$, $t = 0, \dots, T-1$, with the understanding that $x(t) \ge 0$ for all $t$ (a firing rate is
non-negative) and that $x$ carries a known condition label $\ell \in \{0,1\}$.

---

## 4. The augmentation transforms

A *surrogate* of a window is produced by composing a **magnitude warp** and a **time warp**, then
(at the batch-assembly stage) a **circular shift**. All three transforms share a common knot grid.

**Knot grid.** For a window of length $T$ we place $K$ knots at the integer-rounded uniform positions

$$
t_k = (k-1)\,\frac{T-1}{K-1}, \qquad k = 1, \dots, K,
$$

so that $t_1 = 0$ and $t_K = T-1$. The number of knots is

$$
K = \max\!\Big(K_{\min},\; \big\lfloor \kappa_{ps}\, \tfrac{T}{f_s} \big\rfloor\Big), \qquad
\kappa_{ps} = \big\lfloor 1/\Delta_{\text{knot}} \big\rfloor,
$$

where $\Delta_{\text{knot}}$ is the target inter-knot spacing in seconds, $\kappa_{ps}$ is the
resulting (integer) number of knots per second, and $K_{\min} = 4$ guards against degenerate splines
(a cubic spline requires at least two knots; four gives a well-posed not-a-knot fit). The guard fires
only for sub-second windows.

### 4.1 Magnitude warp (log-space, strictly positive)

The magnitude warp perturbs the *amplitude* of the window while preserving its *timing*. To
guarantee that the warped firing rate stays non-negative, the multiplicative curve is built in
**log-space**.

Draw knot log-gains

$$
g_k \overset{\text{iid}}{\sim} \mathcal{N}\!\big(0,\ \sigma_{\text{mag}}^2\big), \qquad k = 1, \dots, K,
$$

where $\sigma_{\text{mag}} > 0$ is the (dimensionless) log-amplitude standard deviation. Let
$s_{\sigma_{\text{mag}}}(\cdot)$ be the cubic spline determined by the interpolation conditions
$s_{\sigma_{\text{mag}}}(t_k) = g_k$ for each $k = 1,\dots,K$. The **scaling curve** is

$$
c_{\sigma_{\text{mag}}}(t) = \exp\!\big(s_{\sigma_{\text{mag}}}(t)\big), \qquad t = 0, \dots, T-1,
$$

and the magnitude-warped window is

$$
\big(\mathrm{MagWarp}_{\sigma_{\text{mag}}} x\big)(t) \;=\; x(t)\,\cdot\,c_{\sigma_{\text{mag}}}(t)
\;=\; x(t)\,\exp\!\big(s_{\sigma_{\text{mag}}}(t)\big), \qquad t = 0, \dots, T-1.
$$

**Positivity.** For each fixed $t$, $c_{\sigma_{\text{mag}}}(t) = \exp(\,\cdot\,) > 0$ regardless of
$\sigma_{\text{mag}}$, hence $(\mathrm{MagWarp}_{\sigma_{\text{mag}}} x)(t) \ge 0$ whenever
$x(t) \ge 0$. This is the central property that the previous additive-in-amplitude construction
($c = 1 + s$) lacked: there, large $\sigma_{\text{mag}}$ let $c$ dip below zero and flip the sign of
the rate.

**Distributional note (flagged abuse).** At a knot, $c_{\sigma_{\text{mag}}}(t_k) = \exp(g_k)$ is
log-normal, so

$$
\operatorname{median}\big[c_{\sigma_{\text{mag}}}(t_k)\big] = 1, \qquad
\mathbb{E}\big[c_{\sigma_{\text{mag}}}(t_k)\big] = \exp\!\big(\sigma_{\text{mag}}^2/2\big) > 1.
$$

The curve therefore has unit *median* but a slight upward *mean* bias (≈ 2% at
$\sigma_{\text{mag}} = 0.2$). We keep the median-1, strictly-positive behaviour and **flag** the mean
bias rather than removing it; it could be removed by centering $g_k \sim
\mathcal{N}(-\sigma_{\text{mag}}^2/2,\ \sigma_{\text{mag}}^2)$, which would force
$\mathbb{E}[c_{\sigma_{\text{mag}}}(t_k)] = 1$ for each fixed $k$.

### 4.2 Time warp (pinned endpoints, no edge plateaus)

The time warp perturbs the *timing* while preserving overall amplitude. Draw knot temporal offsets,
in samples,

$$
\delta_k \sim \mathcal{N}\!\big(0,\ (\sigma_{\text{time}}\, f_s)^2\big), \quad k = 2, \dots, K-1,
\qquad \delta_1 = \delta_K = 0,
$$

where $\sigma_{\text{time}} > 0$ is the temporal standard deviation **in seconds** (so
$\sigma_{\text{time}} f_s$ is in samples). The endpoints are *pinned* to zero offset. Let
$w_{\sigma_{\text{time}}}(\cdot)$ be the cubic spline with $w_{\sigma_{\text{time}}}(t_k) = \delta_k$.
The **warped index map** is

$$
\phi_{\sigma_{\text{time}}}(t) = t + w_{\sigma_{\text{time}}}(t), \qquad t = 0, \dots, T-1,
$$

which by the pinning satisfies $\phi_{\sigma_{\text{time}}}(0) = 0$ and
$\phi_{\sigma_{\text{time}}}(T-1) = T-1$ exactly. Let $\psi_x(\cdot)$ be the cubic spline that
interpolates the *signal*, i.e. $\psi_x(t) = x(t)$ for each integer $t = 0,\dots,T-1$. The
time-warped window resamples the signal at the warped indices:

$$
\big(\mathrm{TimeWarp}_{\sigma_{\text{time}}} x\big)(t) \;=\; \psi_x\!\big(\phi_{\sigma_{\text{time}}}(t)\big),
\qquad t = 0, \dots, T-1.
$$

Two deliberate properties:

- **Endpoints untouched.** Because $\phi_{\sigma_{\text{time}}}(0)=0$ and
  $\phi_{\sigma_{\text{time}}}(T-1)=T-1$, the first and last samples are preserved exactly
  ($\tilde x(0) = x(0)$, $\tilde x(T-1) = x(T-1)$). There is therefore no clipping at the boundary and
  no clip-induced flat plateau. Interior $\phi_{\sigma_{\text{time}}}(t) \notin [0, T-1]$ (rare) is
  handled by the resampling spline's own polynomial extrapolation, again without edge clipping.
- **Folds allowed.** $\phi_{\sigma_{\text{time}}}$ is **not** constrained to be monotonic. A
  non-monotonic $\phi$ (a "fold", i.e. locally reversed time) is permitted; such heavy distortions
  are intended to fall in the negative class (§5), so they need not be prevented.

### 4.3 Circular shift (label-preserving translation)

The shift is a circular translation applied to whole surrogates. Let $\tau_{\text{shift}}$ be the
maximum shift magnitude in seconds and

$$
S = \big\lfloor \tau_{\text{shift}}\, f_s \big\rfloor
$$

the maximum shift in samples. For a surrogate $\tilde x$, draw an integer shift
$s \sim \mathcal{U}\{-S, -S+1, \dots, S\}$ (uniform over that integer range) and apply

$$
\big(\mathrm{Shift}_s\,\tilde x\big)(t) = \tilde x\big((t - s)\ \mathrm{mod}\ T\big), \qquad t = 0, \dots, T-1,
$$

which is a circular roll (`torch.roll`). Each surrogate receives its own independent $s$. If
$S = 0$ (i.e. $\tau_{\text{shift}} f_s$ rounds to zero) a warning is emitted and the input is returned
unchanged. The circular wrap is kept by design; for arbitrarily-cut windows it stitches the tail onto
the head, which is an accepted property here.

### 4.4 Composition into a surrogate

A single surrogate draws a pair $(\sigma_{\text{mag}}, \sigma_{\text{time}})$ from a band (positive or
negative, §5.2) and composes the two warps, then clamps to enforce non-negativity:

$$
\tilde x \;=\; \max\!\Big(0,\ \big(\mathrm{TimeWarp}_{\sigma_{\text{time}}} \circ
\mathrm{MagWarp}_{\sigma_{\text{mag}}}\big)\,x\Big),
$$

where $\max(0,\cdot)$ acts pointwise. The magnitude warp alone never produces negatives (§4.1); the
clamp only catches the rare negative excursions a cubic *resampling* spline can introduce as
overshoot near sharp bursts. The shift is applied later (§5.3), after the split.

Note that $\sigma_{\text{mag}}$ and $\sigma_{\text{time}}$ are **physically distinct** — a
dimensionless log-amplitude and a temporal standard deviation in seconds, respectively — and are
drawn from independent bands; they are never tied to a single shared strength.

---

## 5. The positive / negative split

For each anchor window $x$, the augmentation produces a **positive** set (variants the encoder should
embed near $x$) and a **negative** set (variants it should embed far from $x$). Two split rules are
implemented and selected by a single parameter; both compute the split on the **unshifted**
surrogates and then apply the shift to both classes.

### 5.1 Option 2 — per-anchor percentile of a dissimilarity

Generate a pool of $N$ surrogates $\{\tilde x_m\}_{m=1}^{N}$ with $(\sigma_{\text{mag}},
\sigma_{\text{time}})$ drawn uniformly from a **broad** band that spans both the positive and the
negative ranges (so the pool exhibits a spread of distortion strengths). Measure each surrogate's
dissimilarity to the anchor by the mean squared error,

$$
d_m \;=\; \frac{1}{T} \sum_{t=0}^{T-1} \big(\tilde x_m(t) - x(t)\big)^2, \qquad m = 1, \dots, N,
$$

set the **per-anchor threshold** to the empirical $q$-quantile of these dissimilarities,

$$
\tau_x \;=\; Q_q\big(\{d_m\}_{m=1}^{N}\big), \qquad q \in (0,1),
$$

and split

$$
\mathcal{P} = \{\,m : d_m \le \tau_x\,\}, \qquad \mathcal{N} = \{\,m : d_m > \tau_x\,\}.
$$

Because the threshold is a quantile of *this anchor's own* dissimilarities, the positive:negative
ratio is $\approx q : (1-q)$ for every window regardless of its energy or content — which is precisely
what a single *absolute* threshold failed to provide. For each fixed $q \in (0,1)$ and $N \ge 2$,
both $\mathcal{P}$ and $\mathcal{N}$ are non-empty.

*Limitation (stated honestly).* MSE is dominated by amplitude and by peak misalignment, so a mild
*time* shift can register a large $d_m$ even though it is intended as a positive. The percentile makes
the split robust to window energy but does not fix this metric–intent mismatch; that motivates
Option 3.

### 5.2 Option 3 — labels by construction (warp-strength bands)

Here the label is assigned by *which band a surrogate was drawn from*, with no post-hoc metric.
Define positive and negative bands for each warp strength:

$$
\sigma_{\text{mag}} \sim \mathcal{U}\big[\sigma_{\text{mag}}^{+,\mathrm{lo}}, \sigma_{\text{mag}}^{+,\mathrm{hi}}\big],\quad
\sigma_{\text{time}} \sim \mathcal{U}\big[\sigma_{\text{time}}^{+,\mathrm{lo}}, \sigma_{\text{time}}^{+,\mathrm{hi}}\big]
\qquad\text{(positives, sub-burst scale),}
$$
$$
\sigma_{\text{mag}} \sim \mathcal{U}\big[\sigma_{\text{mag}}^{-,\mathrm{lo}}, \sigma_{\text{mag}}^{-,\mathrm{hi}}\big],\quad
\sigma_{\text{time}} \sim \mathcal{U}\big[\sigma_{\text{time}}^{-,\mathrm{lo}}, \sigma_{\text{time}}^{-,\mathrm{hi}}\big]
\qquad\text{(negatives, $\gtrsim$ burst scale).}
$$

Generate $N^{+}$ positive and $N^{-}$ negative surrogates from the respective bands; the sets
$\mathcal{P}$ and $\mathcal{N}$ are then exactly those generated sets. This encodes directly the
survey's guiding rule — *to destroy the global activity profile, the distortion must be on the
network-burst time scale $\tau_{\text{burst}}$; for positives, a magnitude less* — by anchoring the
negative bands to $\tau_{\text{burst}}$ (estimable from the data via the mean burst duration). There
is no threshold and no empty-class possibility.

The split rule (Option 2 vs Option 3) is selected by the `split_method` parameter; the broad band of
Option 2 is taken to be the union $[\sigma^{+,\mathrm{lo}}, \sigma^{-,\mathrm{hi}}]$ of the two
Option-3 bands, so the same tunable band endpoints serve both methods.

### 5.3 Split-before-shift, and anchor inclusion

In both options the split is computed *before* the circular shift. The shift is then applied
independently to each class:

$$
\mathcal{P}^{\text{sh}} = \big\{\,\mathrm{Shift}_{s_m}\,\tilde x_m : m \in \mathcal{P}\,\big\}, \qquad
\mathcal{N}^{\text{sh}} = \big\{\,\mathrm{Shift}_{s_m}\,\tilde x_m : m \in \mathcal{N}\,\big\},
$$

with each $s_m$ drawn independently. Since a circular shift preserves the identity of a window
(translation is a nuisance, not a class change), applying it after the split leaves the labels intact
while teaching the encoder translation invariance: positives remain positives and negatives remain
negatives under translation.

Finally, the **clean, unshifted anchor** is appended to the positive set:

$$
P = \{x\} \cup \mathcal{P}^{\text{sh}}, \qquad N = \mathcal{N}^{\text{sh}}.
$$

Including $x$ itself (clean and unshifted) ensures the training distribution contains the exact inputs
embedded at inference time (where windows are embedded raw), removing a train/inference distribution
mismatch.

### 5.4 Empty-class guard

If a split yields an empty positive or empty negative set, the instance is **re-drawn** (up to a
fixed number of retries) with a warning; persistent failure raises. Under Option 2 with $0 < q < 1$
and $N \ge 2$, and under Option 3 with $N^{+}, N^{-} \ge 1$, the guard essentially never fires; it
remains as a defensive safeguard against pathological inputs (e.g. NaNs).

The per-anchor builder thus returns the triple

$$
\big(\,a,\ P,\ N\,\big), \qquad a = x \in \mathbb{R}^{1\times T},\quad
P \in \mathbb{R}^{(1+|\mathcal{P}|)\times T},\quad N \in \mathbb{R}^{|\mathcal{N}|\times T},
$$

with $a$ being the first row of $P$.

---

## 6. The contrastive batch and the condition-level label scheme (option b)

### 6.1 Batch construction

A training batch is assembled from $B$ source windows. To guarantee that every batch supports
cross-condition comparisons, a **condition-balanced** sampler draws $m$ windows per condition, so
$B = m \cdot |\{\text{conditions}\}| = 2m$ (for two conditions). Denote the batch's source windows and
their condition labels by

$$
\big\{ \big(x^{(b)}, \ell^{(b)}\big) \big\}_{b=1}^{B}, \qquad \ell^{(b)} \in \{0,1\}.
$$

Each source window $x^{(b)}$ is expanded by §4–§5 into $\big(a^{(b)}, P^{(b)}, N^{(b)}\big)$ with
$a^{(b)} = x^{(b)}$, $P^{(b)} = \{a^{(b)}\} \cup \{\tilde p^{(b)}_j\}_{j=1}^{P_b}$ (profile-preserving,
shifted), and $N^{(b)} = \{\tilde n^{(b)}_k\}_{k=1}^{N_b}$ (profile-destroying, shifted). The embedding
batch is the concatenation

$$
\mathbf{X} \;=\; \bigsqcup_{b=1}^{B} \big( P^{(b)} \cup N^{(b)} \big) \;\in\; \mathbb{R}^{M \times T},
\qquad M = \sum_{b=1}^{B} \big(1 + P_b + N_b\big).
$$

### 6.2 The label assignment (option b)

The labels $\mathbf{y} \in \mathbb{Z}^{M}$ are assigned as follows. For each source window $b$:

$$
y(p) = \ell^{(b)} \quad \text{for every } p \in P^{(b)}
\qquad\text{(condition label for all positives, incl. the anchor),}
$$
$$
y\big(\tilde n^{(b)}_k\big) = u_{b,k} \quad \text{for every } \tilde n^{(b)}_k \in N^{(b)},
$$

where each $u_{b,k}$ is drawn from a label space **disjoint from $\{0,1\}$ and unique per surrogate**,
$u_{b,k} \ge u_{\text{base}}$ with $u_{b,k} \neq u_{b',k'}$ whenever $(b,k) \neq (b',k')$
(implemented as a running counter starting at $u_{\text{base}}$).

### 6.3 What this scheme induces (with quantifiers)

The downstream miner forms triplets $(i,j,k)$ requiring $y_i = y_j$ and $y_i \neq y_k$. Under the
labels above:

- **Cross-window, same-condition positives.** For any two source windows $b \neq b'$ with
  $\ell^{(b)} = \ell^{(b')}$, every $p \in P^{(b)}$ and $p' \in P^{(b')}$ satisfy $y(p) = y(p')$, hence
  form a positive pair. The encoder is therefore asked to pull together *different windows of the same
  condition* — the cross-window contrast that one-window batches lacked.
- **Cross-condition negatives.** For any $b, b'$ with $\ell^{(b)} \neq \ell^{(b')}$, positives from the
  two windows have different labels and form negative pairs. This is the control-vs-pathological
  contrast that is the scientific goal, and the balanced sampler guarantees it is present in *every*
  batch.
- **Negatives-only distractors (option b).** Each destroyed surrogate $\tilde n^{(b)}_k$ has a unique
  label $u_{b,k}$, which matches no other sample. Therefore, *for every* such surrogate, there exists
  no $j$ with $y_j = u_{b,k}$ and $j \neq$ that surrogate; it can never be the anchor or the positive
  of a valid triplet. It can only ever appear as the negative $k$ in a triplet whose anchor has label
  $\ell \in \{0,1\}$. It is thus a pure distractor.
- **Per-anchor behaviour under hard mining.** A hard miner selects, for each anchor, the negatives
  that most violate the margin. For an anchor $a^{(b)}$, its own destroyed surrogates
  $\{\tilde n^{(b)}_k\}_k$ originate from the same window and are typically the *closest* (hardest)
  distractors, so they dominate the selected negatives for $a^{(b)}$. Hence the unique-label
  construction realizes the intended **per-anchor hard negatives** (option b) using only standard
  label-based mining — no custom miner.

*Honest caveat.* By construction a destroyed surrogate is a valid negative for *every* condition
anchor (its unique label differs from both $0$ and $1$), not strictly only for its own source window.
Under hard mining this collapses to per-anchor behaviour in practice, and it arguably generalizes
better ("any destroyed profile lies outside the valid-condition manifold, for any anchor"). A
`destroyed_label_mode = "shared"` alternative is provided, under which all destroyed surrogates share
a single label and instead form one cluster; this is not the default.

### 6.4 The consumer: triplet objective (for grounding only)

The batch $(\mathbf{X}, \mathbf{y})$ feeds a triplet-margin objective (full treatment in Topic 3).
With embeddings $z_i = f_\theta(\mathbf{X}_i)$ and a distance $d(\cdot,\cdot)$ — here the cosine
distance $d(z_i, z_j) = 1 - \frac{\langle z_i, z_j\rangle}{\lVert z_i\rVert\, \lVert z_j\rVert}$ — the
loss over the miner-selected triplet set $\mathcal{T} = \{(i,j,k) : y_i = y_j,\ y_i \neq y_k\}$ is

$$
\mathcal{L}(\theta) \;=\; \frac{1}{|\mathcal{T}|} \sum_{(i,j,k) \in \mathcal{T}}
\Big[\, d(z_i, z_j) - d(z_i, z_k) + \beta \,\Big]_{+},
$$

where $\beta > 0$ is the margin, $[u]_{+} = \max(u, 0)$, and the average is over the *non-zero* terms
(the `AvgNonZeroReducer`). The label scheme of §6.2 is exactly what determines which $(i,j,k)$ are
admissible, and therefore what the encoder is pushed to do. This objective is included here only to
make the role of the labels concrete; its margin, miner, distance, and reducer are Topic-3 concerns.

---

## 7. Software architecture

The pipeline is split into four modules plus two smoke tests, with strict separation of concerns.

```
 raw trace r_c[n]                         provider:  SyntheticTraceProvider | NeuronalTracesProvider
        │
        ▼  window (T = cp2(τ_w·f_s), stride S_w)                         [data_pipeline.MEAWindowDataset]
   window x(t), condition ℓ
        │
        ▼  build_triplet_instance(x, cfg, rng)         [augmentation.py — runs on CPU in DataLoader workers]
        │     split BEFORE shift:
        │       warp_bands     : P-band → 𝒫 , N-band → 𝒩
        │       percentile_mse : broad pool → MSE q-quantile → 𝒫, 𝒩
        │     shift both classes; prepend clean anchor to 𝒫
        ▼
   (a = x ,  P = {a} ∪ shift(𝒫) ,  N = shift(𝒩))
        │
        ▼  condition-balanced batch of B source windows  [data_pipeline.ConditionBalancedBatchSampler]
        ▼  collate → option-(b) labels                   [data_pipeline.TripletCollator]
   (X ∈ ℝ^{M×T},  y ∈ ℤ^{M})
        │
        ▼   EXTENSION POINT  →  encoder f_θ  →  miner + triplet loss     [Topics 2–3]

 plotting (anchor / positives / negatives)  [augmentation_viz.plot_triplet_instance — separate module]
 entry point (config, seeding, DataLoader, sanity report, debug plots)   [run_data_pipeline.py]
```

| Module | Responsibility | Key public objects |
|---|---|---|
| `augmentation.py` | pure transforms + split + per-anchor triplet builder | `AugmentationConfig`, `magnitude_warp`, `time_warp`, `random_circular_shift`, `build_triplet_instance` |
| `data_pipeline.py` | data loading, windowing, balanced batching, collation | `SyntheticTraceProvider`, `NeuronalTracesProvider`, `MEAWindowDataset`, `ConditionBalancedBatchSampler`, `TripletCollator`, `seed_worker`, `closest_power_of_2` |
| `augmentation_viz.py` | headless visual-debug plotting only | `plot_triplet_instance` |
| `run_data_pipeline.py` | HPC front-end / entry point | `PipelineConfig`, `main` |

The transforms know nothing about conditions, batches, or the model; the collator knows nothing about
spline internals; the plotter imports nothing from the training path; the front-end wires them
together via configuration. Swapping the split rule, the band values, the label mode, the window
overlap, or the data source touches one place each.

### 7.1 Numerical type policy

A single dtype rule is enforced at every tensor boundary: tensors leaving the augmentation are CPU
`float32`; SciPy performs the spline solves transiently in `float64` (better conditioning) and the
result is cast back to `float32`. Condition labels and the unique negative labels are `int64`
(`torch.long`), as required by the metric-learning loss. This removes the earlier dtype-mismatch
failure mode and keeps the pipeline autocast-compatible.

---

## 8. Reproducibility and HPC design

**Where augmentation runs.** Augmentation is CPU work performed inside DataLoader worker processes
(`num_workers > 0`), so it overlaps with encoder compute on the accelerator and does not block it.

**Randomness.** Every transform takes an explicit `numpy.random.Generator`. Each worker is seeded
deterministically in `seed_worker` from the dataset's base seed and the worker id; the
condition-balanced sampler is seeded per epoch. The reproducibility guarantee is precise:

> For a **fresh** pipeline construction with a fixed `(seed, num_workers)`, the sequence of produced
> batches is identical across runs.

The dataset's generator is *stateful* — it advances as windows are drawn, which is what gives
augmentation diversity across epochs — so reproducibility is a property of a fresh construction, not
of re-iterating an already-used dataset object. (This distinction is exactly what the smoke test
verifies, §10.)

**HPC hygiene.** The front-end is a plain script (no notebook cells, no IPython magics, no hard-coded
paths); plotting uses a headless backend and writes PNGs to a folder; intra-op thread count is set
via `torch.set_num_threads` to avoid oversubscription when many workers run; the full configuration is
dumped to JSON for provenance. A `--data-mode synthetic` path lets the entire pipeline (and a SLURM
submission) be validated without any data files before switching to `--data-mode real`.

---

## 9. Usage (front-end)

```bash
pip install numpy scipy torch matplotlib

# (1) dry-run on HPC, no data needed (validates the pipeline + SLURM submission):
python -u run_data_pipeline.py --data-mode synthetic --num-workers 4 --n-debug-plots 6

# (2) real data: specs.json lists {folder, base, condition} per well
python -u run_data_pipeline.py --data-mode real --specs-json specs.json --num-workers 8 \
       --window-s 200 --stride-s 100 --split-method warp_bands
```

`specs.json` (real mode):

```json
[
  {"folder": "/path/ptrain_Control00_Well11", "base": "ptrain_Control00_Well11_", "condition": 0},
  {"folder": "/path/ptrain_Control00_Well17", "base": "ptrain_Control00_Well17_", "condition": 0},
  {"folder": "/path/pgroup02_Well14",         "base": "pgroup02_Well14_",         "condition": 1}
]
```

The front-end prints a per-batch sanity report ($\mathbf{X}$ shape, counts of control/pathological
positives, count of unique-labelled negatives, number of source windows) and writes visual-debug
figures. The marked **extension point** is where the Topic-2 encoder and Topic-3 loss attach,
consuming $\mathbf{X} \in \mathbb{R}^{M\times T}$ (unsqueezed to $\mathbb{R}^{M\times 1\times T}$ for
the 1-D convolution) and $\mathbf{y} \in \mathbb{Z}^{M}$.

---

## 10. Validation

Each module ships a self-contained smoke test that synthesizes data, asserts the invariants, and
writes debug figures. Each was **controlled twice** (a unit run and an independent second control).

**`smoke_test_augmentation.py`** — 32 checks: magnitude-warp positivity at $\sigma_{\text{mag}} \in
\{0.05, 0.2, 0.5\}$; time-warp endpoint preservation; circular-shift equivalence to `torch.roll`;
the zero-shift warning; percentile-split positive fraction $\approx q$; warp-band ordering
$\overline{\mathrm{MSE}}(\mathcal{P}) < \overline{\mathrm{MSE}}(\mathcal{N})$; anchor-as-first-positive;
`float32` dtype; the empty-class guard raising at $q=1$; and reproducibility. *Second control:* a
40-trial sweep over random signals/seeds — positivity held 40/40 (global min exactly $0$), endpoints
40/40 (exact), warp-band MSE ordering 40/40 (margin always positive), determinism confirmed.

**`smoke_test_data_pipeline.py`** — 18 checks: windowing with overlap; anchor identity; the option-(b)
label scheme (positives carry $\{0,1\}$; negatives have unique labels $\ge u_{\text{base}}$, all
distinct, disjoint from $\{0,1\}$); $M = \sum_b (1 + P_b + N_b)$; every batch condition-balanced;
fresh-construction determinism; the multi-worker path; both split methods; both label modes. *Second
control:* the front-end run end-to-end in synthetic mode, confirming balanced batches
(`control=patho` positives), unique negatives, the `percentile_mse` path, the config dump, and the
debug figures.

Run them with:

```bash
python smoke_test_augmentation.py        # 32/32
python smoke_test_data_pipeline.py       # 18/18
```

---

## 11. Parameters, tuning, and items to confirm

**Strength bands (must be tuned).** The positive/negative bands $\big[\sigma_{\text{mag}}^{\pm,\cdot}\big]$
and $\big[\sigma_{\text{time}}^{\pm,\cdot}\big]$ are placeholders. They should be tuned against the
measured network-burst time scale $\tau_{\text{burst}}$ (mean burst duration): negatives at or above
$\tau_{\text{burst}}$ (profile-destroying), positives well below it. As a positivity-safe starting
point, keep $\sigma_{\text{mag}} \lesssim 0.2$ for the positive band (so the log-normal amplitude stays
mild) and choose $\sigma_{\text{time}}$ (seconds) by reference to $\tau_{\text{burst}}$.

**Split fraction.** For `percentile_mse`, $q$ is the positive fraction (e.g. $q = 0.3$). For
`warp_bands`, the counts $N^{+}, N^{-}$ set the class sizes directly.

**Knot spacing.** $\Delta_{\text{knot}}$ (seconds) controls spline smoothness via $K$; coarser knots
give smoother, lower-frequency warps.

**Windowing.** $\tau_w$ (window length, s) and $S_w$ (stride, s) trade window count against window
correlation; $S_w < T$ raises diversity beyond the few non-overlapping windows per trace.

**Batch shape.** $m$ (windows per condition) and the per-window counts set $M$; larger $m$ gives more
cross-condition contrast per step at higher memory cost.

**Items to confirm before a real run.** (i) the $\mathrm{cp2}$ convention (nearest vs floor) must match
the encoder engine; (ii) the real-mode import must point at the actual `Neuronal_traces` module name;
(iii) `--num-workers` should equal the CPUs-per-task granted by the scheduler.

---

## 12. Symbol glossary

| Symbol | Meaning | First appears |
|---|---|---|
| $r_c[n],\ L_c$ | source trace $c$ and its length (samples) | §1 |
| $f_s$ | sampling rate (Hz) | §1 |
| $\ell_c,\ \ell^{(b)}$ | condition label of trace $c$ / source window $b$ ($0$ control, $1$ patho) | §1, §6 |
| $f_\theta,\ E$ | embedding map and embedding dimension | §1 |
| $x(t),\ T$ | a window and its length (samples) | §3 |
| $\tau_w,\ S_w,\ \mathrm{cp2}$ | window length (s), stride (samples), nearest-power-of-two | §3 |
| $t_k,\ K,\ K_{\min},\ \Delta_{\text{knot}},\ \kappa_{ps}$ | knot positions, count, minimum, spacing (s), knots/s | §4 |
| $g_k,\ \sigma_{\text{mag}},\ s_{\sigma_{\text{mag}}},\ c_{\sigma_{\text{mag}}}$ | magnitude knot log-gains, log-amplitude std, spline, scaling curve | §4.1 |
| $\delta_k,\ \sigma_{\text{time}},\ w_{\sigma_{\text{time}}},\ \phi_{\sigma_{\text{time}}},\ \psi_x$ | time knot offsets, temporal std (s), warp spline, index map, signal spline | §4.2 |
| $\tau_{\text{shift}},\ S,\ s,\ \mathrm{Shift}_s$ | max shift (s), max shift (samples), drawn shift, circular roll | §4.3 |
| $\tilde x,\ \tilde x_m$ | a surrogate / the $m$-th surrogate | §4.4, §5 |
| $\tau_{\text{burst}}$ | network-burst time scale (mean burst duration) | §5.2 |
| $N,\ d_m,\ q,\ Q_q,\ \tau_x$ | pool size, dissimilarity, quantile fraction, quantile, per-anchor threshold | §5.1 |
| $N^{+},N^{-},\ \sigma^{\pm,\mathrm{lo/hi}}$ | positive/negative counts and band endpoints | §5.2 |
| $\mathcal{P},\mathcal{N},\ \mathcal{P}^{\text{sh}},\mathcal{N}^{\text{sh}}$ | positive/negative index sets, before/after shift | §5 |
| $a,\ P,\ N$ | anchor, positive set (incl. anchor), negative set (one instance) | §5.4 |
| $B,\ m$ | source windows per batch, windows per condition | §6.1 |
| $a^{(b)},P^{(b)},N^{(b)},P_b,N_b$ | per-source-window anchor/sets and their sizes | §6.1 |
| $\mathbf{X},\ \mathbf{y},\ M$ | embedding batch, label vector, batch size $M=\sum_b(1+P_b+N_b)$ | §6.1–6.2 |
| $u_{b,k},\ u_{\text{base}}$ | unique negative label and its base offset | §6.2 |
| $z_i,\ d(\cdot,\cdot),\ \beta,\ \mathcal{T},\ \mathcal{L}(\theta)$ | embeddings, distance, margin, triplet set, loss | §6.4 |

---

## 13. References (project library)

- Time-series data-augmentation survey (taxonomy of magnitude/time warping; the "negative training"
  caution that the perturbation magnitude must match its intended role) — grounds §4–§5.
- Easy-positive triplet-mining paper (hard vs easy positive selection and its effect on
  generalization) — relevant to the miner choice in Topic 3, foreshadowed in §6.4.
