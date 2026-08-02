# Multichannel ($C \geq 1$) Support in the Deep Summary Network

**Date:** 31 July 2026
**Branch:** `multichannel` (merged onto `main` @ `ada539a`)
**Status:** merged and verified; one deliberate gap (Sec. 5, K3)

## Abstract

The Deep Summary Network learns an embedding of neuronal-activity traces from
multi-electrode array (MEA) recordings, in which windows of a smoothed
instantaneous firing rate (IFR) are mapped to a metric space by a 1-D CNN
trained with a triplet objective. Until this work, every trace was a single
scalar time series: one population IFR per recording, and the network stem was
hard-wired to one input channel. This document specifies the scientific question
*"does the spatial structure of a culture carry phenotype information that the
whole-culture population rate discards?"* and the engineering required to make
that question askable: a channel axis $C$ threaded coherently from electrode
partitioning, through IFR extraction, augmentation, windowing, splitting and
batching, to the convolutional stem. **Covered:** the shape contract and its
propagation, the augmentation semantics that preserve inter-channel structure,
the electrode-to-subregion partitioning, the three extraction modes, the merge
against the concurrent backbone/optimization overhaul, and the verification.
**Deliberately excluded:** any claim that multichannel input improves
classification (untested on real phenotype data); the sibling-subregion culture
grouping required by cross-culture positives (specified but not implemented,
Sec. 5 K3); latent-mode multichannel generation (refused by design, Sec. 3.6); and
tuning of $C$, $E$, or the subregion geometry.

---

## 1. Notation and Symbols

| Symbol | Name / Meaning | Type & domain | Units | First used in Sec. |
|---|---|---|---|---|
| $C$ | channel count: input traces per window | $C \in \mathbb{N}$, $C \geq 1$ | dimensionless | Sec. 3.1 |
| $c$ | channel index | $c \in \{0,\dots,C-1\}$ | dimensionless | Sec. 3.1 |
| $K$ | IFR samples per full trace | $K \in \mathbb{N}$ | samples | Sec. 3.1 |
| $k$ | discrete time index into a trace | $k \in \{0,\dots,K-1\}$ | dimensionless | Sec. 3.1 |
| $t_k$ | continuous time of sample $k$, $t_k = k / f_s^{\mathrm{IFR}}$ | $t_k \in \mathbb{R}_{\geq 0}$ | s | Sec. 3.1 |
| $W$ | window length | $W \in \mathbb{N}$, $W \leq K$ | samples | Sec. 3.3 |
| $T$ | generic window length in transform formulae; $T \equiv W$ | $T \in \mathbb{N}$ | samples | Sec. 3.4 |
| $s$ | window start offset within a trace | $s \in \{0,\dots,K-W\}$ | samples | Sec. 3.3 |
| $M$ | batch rows (anchors + positives + negatives) | $M \in \mathbb{N}$ | dimensionless | Sec. 3.3 |
| $m$ | batch row index | $m \in \{0,\dots,M-1\}$ | dimensionless | Sec. 3.3 |
| $u$ | global culture (recording) index | $u \in \{0,\dots,U-1\}$ | dimensionless | Sec. 3.7 |
| $U$ | number of cultures in the dataset | $U \in \mathbb{N}$ | dimensionless | Sec. 3.7 |
| $f_s^{\mathrm{raw}}$ | raw acquisition sampling rate | $f_s^{\mathrm{raw}} \in \mathbb{R}_{>0}$ | Hz | Sec. 3.2 |
| $f_s^{\mathrm{IFR}}$ | IFR sampling rate after smoothing/decimation | $f_s^{\mathrm{IFR}} \in \mathbb{R}_{>0}$ | Hz | Sec. 3.2 |
| $T_{\mathrm{rec}}$ | recording duration | $T_{\mathrm{rec}} \in \mathbb{R}_{>0}$ | s | Sec. 3.2 |
| $\mathcal{E}$ | set of electrodes present in a recording | $\mathcal{E} \subset \mathbb{N}$, finite | dimensionless | Sec. 3.2 |
| $E$ | electrodes per subregion | $E \in \mathbb{N}$, $E \geq 1$ | dimensionless | Sec. 3.2 |
| $\mathcal{M}_c$ | electrode set of subregion $c$ | $\mathcal{M}_c \subseteq \mathcal{E}$, $\lvert \mathcal{M}_c \rvert = E$ | dimensionless | Sec. 3.2 |
| $\theta$ | mean-firing-rate inclusion threshold | $\theta \in \mathbb{R}_{\geq 0}$ | spikes/s | Sec. 3.2 |
| $N_e(t_k)$ | spike count of electrode $e$ in the bin at $t_k$ | $N_e(t_k) \in \mathbb{Z}_{\geq 0}$ | spikes | Sec. 3.2 |
| $\mathcal{G}_\sigma$ | Gaussian smoothing operator, bandwidth $\sigma$ | linear operator on $\mathbb{R}^{K}$ | --  | Sec. 3.2 |
| $\sigma$ | smoothing kernel bandwidth | $\sigma \in \mathbb{R}_{>0}$ | s | Sec. 3.2 |
| $R_c(t_k)$ | IFR of channel $c$ at time $t_k$ | $R_c(t_k) \in \mathbb{R}_{\geq 0}$ | spikes/s | Sec. 3.2 |
| $X$ | full multichannel trace, rows indexed by $c$ | $X \in \mathbb{R}_{\geq 0}^{C \times K}$ | spikes/s | Sec. 3.1 |
| $x$ | a single window (generic; $(T,)$ or $(C,T)$) | $x \in \mathbb{R}^{T}$ or $\mathbb{R}^{C \times T}$ | spikes/s | Sec. 3.4 |
| $\sigma_{\mathrm{mag}}$ | log-amplitude warp standard deviation | $\sigma_{\mathrm{mag}} \in \mathbb{R}_{>0}$ | dimensionless | Sec. 3.4 |
| $\sigma_{\mathrm{time}}$ | temporal warp standard deviation | $\sigma_{\mathrm{time}} \in \mathbb{R}_{>0}$ | s | Sec. 3.4 |
| $K_{\mathrm{knot}}$ | number of spline knots per window | $K_{\mathrm{knot}} \in \mathbb{N}$, $\geq 4$ | dimensionless | Sec. 3.4 |
| $g_j$ | knot log-gain, $j$-th knot | $g_j \sim \mathcal{N}(0,\sigma_{\mathrm{mag}}^2)$ | dimensionless | Sec. 3.4 |
| $\varsigma(\cdot)$ | cubic spline through the knot log-gains | $\varsigma : [0,T-1] \to \mathbb{R}$ | dimensionless | Sec. 3.4 |
| $a(t)$ | multiplicative magnitude-warp curve, $a = \exp \varsigma$ | $a(t) \in \mathbb{R}_{>0}$ | dimensionless | Sec. 3.4 |
| $\varphi^{\mathrm{warp}}(t)$ | warped index map (time warp) | $\varphi^{\mathrm{warp}} : [0,T-1] \to \mathbb{R}$ | samples | Sec. 3.4 |
| $\psi_x(\cdot)$ | cubic spline interpolating the signal $x$ | $\psi_x : \mathbb{R} \to \mathbb{R}$ | spikes/s | Sec. 3.4 |
| $S$ | circular-shift magnitude bound | $S \in \mathbb{Z}_{\geq 0}$ | samples | Sec. 3.4 |
| $\varsigma_{\mathrm{sh}}$ | drawn circular shift for one surrogate | $\varsigma_{\mathrm{sh}} \in \{-S,\dots,S\}$ | samples | Sec. 3.4 |
| $P$ | positives generated per anchor window | $P \in \mathbb{Z}_{\geq 0}$ | dimensionless | Sec. 3.4 |
| $N_{\mathrm{s}}$ | surrogate negatives per anchor window | $N_{\mathrm{s}} \in \mathbb{Z}_{\geq 0}$ | dimensionless | Sec. 3.4 |
| $w_0$ | stem width (output channels of the stem conv) | $w_0 \in \mathbb{N}$ | dimensionless | Sec. 3.5 |
| $\kappa$ | stem kernel length | $\kappa \in \mathbb{N}$ | samples | Sec. 3.5 |
| $r$ | stem stride | $r \in \mathbb{N}$ | samples | Sec. 3.5 |
| $f(\cdot)$ | the embedding network (backbone + head) | $f : \mathbb{R}^{C \times T} \to \mathbb{S}^{d-1}$ | --  | Sec. 3.5 |
| $d$ | embedding dimensionality | $d \in \mathbb{N}$ | dimensionless | Sec. 3.5 |
| $Z$ | batch embedding matrix | $Z \in \mathbb{R}^{M \times d}$ | dimensionless | Sec. 3.5 |
| $y$ | per-window condition labels | $y \in \{0,\dots,L-1\}^{M}$ | dimensionless | Sec. 3.7 |
| $L$ | number of conditions (classes) | $L \in \mathbb{N}$ | dimensionless | Sec. 3.7 |
| $g$ | per-window culture id (`trace_of_window`) | $g \in \{0,\dots,U-1\}^{M}$ | dimensionless | Sec. 3.7 |
| $\mathcal{C}(x)$ | class label of sample $x$ | $\mathcal{C}(x) \in \{0,\dots,L-1\}$ | dimensionless | Sec. 3.7 |
| $\varphi^{\mathrm{lat}}$ | latent phenotype vector (latent data mode) | $\varphi^{\mathrm{lat}} \in \mathbb{R}^{n_{\mathrm{lat}}}$ | mixed | Sec. 3.6 |
| $n_{\mathrm{lat}}$ | latent-space dimensionality | $n_{\mathrm{lat}} \in \mathbb{N}$ | dimensionless | Sec. 3.6 |

### 1.1 Conventions

- **Index origin.** All array indices are 0-based ($c$, $k$, $m$, $u$), matching
  the implementation. The MEA electrode index `index_base` is a *separate*
  convention, configurable as 0 or 1, and is **not** resolved by this work (Sec. 5,
  O1).
- **Shape ordering.** Arrays are written in NumPy/PyTorch order. A full trace is
  $(C, K)$; a window is $(C, W)$; a batch is $(M, C, W)$. **The time axis is
  always last.** This is the single invariant on which Sec. 3.3 depends.
- **The $C = 1$ degenerate case.** A single-channel trace may be stored as
  $(K,)$ *or* $(1, K)$. Both are accepted everywhere. Where the distinction
  matters it is stated explicitly; otherwise "for $C = 1$" covers both.
- **Symbol collision, disambiguated.** The letter $\varphi$ carries two unrelated
  meanings in this codebase: the **warped index map** of the time-warp transform
  (Sec. 3.4) and the **latent phenotype vector** of the latent generator (Sec. 3.6).
  They are written $\varphi^{\mathrm{warp}}$ and $\varphi^{\mathrm{lat}}$
  throughout and are never abbreviated to bare $\varphi$. Similarly $\varsigma$
  denotes the knot spline in Sec. 3.4 and $\varsigma_{\mathrm{sh}}$ the circular
  shift; the subscript is never dropped.
- **Quantifiers.** Statements about traces hold *for each fixed culture $u$*;
  statements about channels hold *for each fixed $c \in \{0,\dots,C-1\}$*.
  These are written out rather than assumed.
- **Units.** $R_c(t_k)$ is a rate in spikes/s. The magnitude warp is
  dimensionless (a multiplicative gain); the time warp is in samples;
  $\sigma_{\mathrm{time}}$ is specified in seconds and converted by
  $\sigma_{\mathrm{time}} f_s^{\mathrm{IFR}}$.
- **Code references** are given as `file.py:line` against the merged branch.

---

## 2. Glossary / Jargon

Ordered **by first appearance**, because the concepts build on one another.

**MEA (multi-electrode array).** A grid of extracellular electrodes recording
spiking activity from a cultured neuronal network. Here a $48 \times 48$ grid,
2304 sites. Operative from Sec. 3.2.

**ptrain.** The per-electrode spike raster as stored by the acquisition
pipeline: one `.mat` file per electrode, holding a binary `uint8` array of shape
$(n, 1)$ with a 1 at each spike sample. *Not* a list of spike times --  a
distinction that silently breaks Stage 1 if violated. Operative from Sec. 3.2.

**IFR (instantaneous firing rate).** The spike train convolved with a smoothing
kernel and expressed as a rate. This, not the raw raster, is the network's
input. Operative from Sec. 3.2.

**Culture.** One recording of one biological preparation. Synonymous with
"trace" in the original single-channel design, where the mapping was
one-to-one --  an identification this work breaks (Sec. 3.7). Operative from Sec. 3.7.

**Subregion.** A spatially contiguous set of $E$ electrodes whose pooled IFR
forms one channel. Chosen around a seed electrode by nearest-neighbour
selection on the grid. Operative from Sec. 3.2.

**Channel.** One row of a multichannel trace, $R_c(\cdot)$. **Everyday-meaning
warning:** in MEA hardware jargon "channel" usually means *one electrode*. Here
it means *one subregion's pooled IFR*, aggregating $E$ electrodes. The two
senses differ by a factor of $E$ and are routinely confused. Operative from Sec. 3.1.

**Stem.** The first convolution of the CNN, which maps the raw input channel
count to the network's internal width $w_0$. The only place the input channel
count appears architecturally. Operative from Sec. 3.5.

**Window.** A contiguous slice of $W$ samples along the time axis, the unit the
network actually consumes. Operative from Sec. 3.3.

**Surrogate.** An augmented copy of a window, produced by composing a magnitude
warp, a time warp and a circular shift. Operative from Sec. 3.4.

**Magnitude warp.** A smooth, time-varying multiplicative gain applied to the
signal, built in log-space so the result stays non-negative. Operative from Sec. 3.4.

**Time warp.** A smooth deformation of the *time* axis, resampling the signal at
non-integer indices. Operative from Sec. 3.4.

**Positive / negative.** In metric learning, a sample that should embed *near*
the anchor (positive) or *far* from it (negative). Here positives are
profile-preserving surrogates (or, in cross-culture mode, windows from other
cultures of the same class) and negatives are profile-destroying surrogates.
Operative from Sec. 3.4.

**Easy-positive mining (EP / EPHN).** A triplet-selection rule that pairs the
anchor with the *most similar* same-class sample rather than the least similar,
to preserve intra-class variance. Operative from Sec. 3.7.

**Cross-culture positives.** A training mode in which positives come from
*different cultures* of the same class rather than from warps of the anchor's
own window, removing the augmentation as the shared carrier of similarity.
Operative from Sec. 3.7.

**Trace split / culture split.** Assigning whole cultures --  never windows --  to
train/val/test, so no culture contributes windows to two splits. Operative from
Sec. 3.7.

**Latent mode.** A synthetic data generator that samples a phenotype vector
$\varphi^{\mathrm{lat}}$ per recording, giving a controllable
higher-dimensional data manifold. Operative from Sec. 3.6.

---

## 3. Main body

### 3.1 The shape contract

*This section establishes the single invariant everything else depends on.*

A trace is a matrix

$$
X = \big[\,R_c(t_k)\,\big]_{c=0,\dots,C-1;\; k=0,\dots,K-1} \in \mathbb{R}_{\geq 0}^{C \times K},
\tag{1}
$$

with $C = 1$ permitted and, in that case, storable equivalently as a vector in
$\mathbb{R}_{\geq 0}^{K}$. The contract is:

> **Invariant I.** The time axis is the **last** axis, for every array, at every
> stage. A window is $(C, W)$; a batch is $(M, C, W)$.

The pre-existing code violated Invariant I in five places by reading
`shape[0]` as the time length --  correct for a $(K,)$ vector, but on a $(C,K)$
matrix that expression returns $C$. Worse, the accompanying slice `tr[s:e]` then
cuts the **channel** axis. This fails silently: it produces a well-formed array
of the wrong content. All five were corrected to `shape[-1]` and `tr[..., s:e]`
(Sec. 4, R2).

`DataConfig.n_channels` is the single source of truth for $C$ and *drives*
`BackboneConfig.in_channels` in `config.py`, so the data and the stem cannot
disagree by construction rather than by convention.

### 3.2 From electrodes to channels

*This section establishes how $C$ subregion IFRs are obtained from one MEA
recording.*

Let $\mathcal{E}$ be the set of electrodes present in a recording and let

$$
\mathrm{MFR}(e) = \frac{1}{T_{\mathrm{rec}}}\sum_{k=0}^{K_{\mathrm{raw}}-1} N_e(t_k),
\qquad e \in \mathcal{E},
\tag{2}
$$

be the mean firing rate of electrode $e$ over the whole recording. Electrodes
with $\mathrm{MFR}(e) < \theta$ are discarded as inactive. Subregion centres are
chosen greedily by descending $\mathrm{MFR}$, and for each fixed centre the
subregion $\mathcal{M}_c$ is the set of the $E$ nearest surviving electrodes on
the grid, $\lvert \mathcal{M}_c \rvert = E$, with $\mathcal{M}_c \cap
\mathcal{M}_{c'} = \emptyset$ for $c \neq c'$.

The channel IFR is the smoothed pooled count, normalised by the pool size:

$$
R_c(t_k) \;=\; \frac{1}{E}\,
\mathcal{G}_\sigma\!\Big[\textstyle\sum_{e \in \mathcal{M}_c} N_e(\cdot)\Big](t_k),
\qquad \text{for each fixed } c \text{ and each } k .
\tag{3}
$$

The whole-culture rate is the same construction with $\mathcal{M} = \mathcal{E}$
and normalisation $1/\lvert \mathcal{E} \rvert$.

Three extraction modes are provided, and they are **not** interchangeable:

| Mode | Output | `row_meaning` | `in_channels` | Meaning |
|---|---|---|---|---|
| `multichannel` | $(C, K)$ | `channels` | $C$ | one sample, $C$ channels |
| `per_region_single` | $(C, K)$ | `samples` | $1$ | $C$ independent single-channel samples |
| `whole_culture` | $(1, K)$ | `samples` | $1$ | the original design |

**The first two produce arrays of identical shape with opposite meaning.** Any
code that infers $C$ from `X.shape[0]` gets `per_region_single` exactly wrong,
treating $C$ unrelated samples as $C$ channels of one sample and feeding them to
a stem expecting $C$ channels. This is why the loader reads the archive's own
`in_channels` field and never infers from shape (Sec. 3.3).

### 3.3 Loading, windowing, splitting

*This section establishes that the channel axis survives the data path
untouched.*

`NumpyTraceProvider` validates **metadata-first**: it requires `ifr_trace`,
accepts `ndim` $\in \{1,2\}$, and where the archive carries `in_channels` it
cross-checks against the array shape, raising if they disagree. Where the field
is absent (hand-authored archives predating the convention) it falls back to the
`ndim` rule, preserving backward compatibility.

Windows are extracted for each fixed trace index and each admissible start $s$:

$$
x^{(u,s)} \;=\; X^{(u)}\big[\,\cdot\,,\; s : s+W\,\big] \in \mathbb{R}^{C \times W},
\qquad s \in \{0, \tau, 2\tau, \dots\},\; s + W \leq K,
\tag{4}
$$

with stride $\tau$ differing between train and eval splits. The ellipsis in the
implementation (`traces[ti][..., s:s+W]`) is what makes (4) simultaneously
correct for $(K,)$ and $(C,K)$.

Splitting operates on **whole cultures** (`make_trace_splits`) or on **disjoint
time segments** (`make_time_segment_splits`); in both, the channel axis rides
through untouched once Invariant I holds. The preprocessing cache stores
$(C,K)$ arrays and records `length` from `shape[-1]`.

### 3.4 Augmentation with a shared field

*This section establishes the one genuinely non-obvious design decision.*

For a window $x$ and each fixed $t \in \{0,\dots,T-1\}$, the magnitude warp is

$$
\big(\mathrm{MagWarp}_{\sigma_{\mathrm{mag}}}\, x\big)(c,t)
\;=\; x(c,t)\; a(t), \qquad a(t) = \exp\big(\varsigma(t)\big),
\tag{5}
$$

where $\varsigma$ is the cubic spline through knot log-gains $g_j \sim
\mathcal{N}(0, \sigma_{\mathrm{mag}}^2)$, $j = 1,\dots,K_{\mathrm{knot}}$. The
time warp resamples at the warped indices,

$$
\big(\mathrm{TimeWarp}_{\sigma_{\mathrm{time}}}\, x\big)(c,t)
\;=\; \psi_{x(c,\cdot)}\big(\varphi^{\mathrm{warp}}(t)\big),
\qquad \varphi^{\mathrm{warp}}(t) = t + w(t),
\tag{6}
$$

with $w$ the cubic spline through temporal knot offsets $\delta_j \sim
\mathcal{N}\big(0, (\sigma_{\mathrm{time}} f_s^{\mathrm{IFR}})^2\big)$ and
endpoints pinned, $\delta_1 = \delta_{K_{\mathrm{knot}}} = 0$. The circular
shift is

$$
\big(\mathrm{Shift}_{\varsigma_{\mathrm{sh}}}\, x\big)(c,t)
= x\big(c,\; (t - \varsigma_{\mathrm{sh}}) \bmod T\big),
\qquad \varsigma_{\mathrm{sh}} \sim \mathrm{Unif}\{-S,\dots,S\}.
\tag{7}
$$

**The decision.** In (5), (6) and (7), $a(t)$, $\varphi^{\mathrm{warp}}(t)$ and
$\varsigma_{\mathrm{sh}}$ carry **no $c$ index**. One field is drawn per
surrogate and applied identically to all $C$ channels:

$$
\text{for all } c, c' \in \{0,\dots,C-1\}: \quad
a_c(\cdot) = a_{c'}(\cdot), \quad
\varphi^{\mathrm{warp}}_c(\cdot) = \varphi^{\mathrm{warp}}_{c'}(\cdot), \quad
\varsigma_{\mathrm{sh},c} = \varsigma_{\mathrm{sh},c'} .
\tag{8}
$$

*Rationale.* Independent per-channel fields would randomise the relative timing
between subregions, destroying exactly the inter-channel synchrony and
propagation structure that is the reason to use multiple channels at all. A
positive must remain a plausible recording of the *same* spatial dynamics.
Property (8) is asserted directly by `smoke_test_augmentation_mc.py`, which
checks that channels stay identical when the input channels are identical
(max deviation $0.00$) and that the ratio invariant
$R_c = \alpha_c R_0$ is preserved (max deviation $1.9\times10^{-6}$).

*Consequence.* Because exactly one field is drawn per surrogate regardless of
$C$, the RNG draw order is unchanged from the single-channel implementation, so
the $C = 1$ path is **numerically identical** to before --  not merely equivalent.

The magnitude and time warps are the standard constructions catalogued in the
project's augmentation survey (Sec. 6, KB1, full text read), which defines magnitude
warping as a cubic-spline-interpolated scaling applied pointwise to the curve
and time warping as the analogous deformation applied along the temporal axis
rather than the amplitude. The survey also lists **channel permutation** among
multivariate techniques; it is *not* used here, and deliberately so --  permuting
subregion identity would destroy the spatial correspondence that (8) exists to
preserve.

One edge case required care. In cross-culture mode $P = 0$ (positives come from
other cultures, not warps), so the positive pool is legitimately empty. The
empty tensor must keep the channel axis: $(0, T)$ for $C = 1$ but $(0, C, T)$
for $C > 1$. The pre-existing code computed $T$ by flattening, which on a
$(C,T)$ window yields $C \cdot T$.

### 3.5 The stem

*This section establishes the architectural change, which is small.*

The stem convolution becomes

$$
\mathrm{Stem}(x) = \mathrm{ReLU}\Big(\mathrm{Norm}\big(\mathrm{Conv1d}_{C \to w_0,\,\kappa,\,r}(x)\big)\Big),
\qquad x \in \mathbb{R}^{M \times C \times T},
\tag{9}
$$

replacing $\mathrm{Conv1d}_{1 \to w_0}$. The forward shape contract is

$$
x \in \mathbb{R}^{M \times C \times T}, \quad\text{or}\quad
x \in \mathbb{R}^{M \times T} \text{ accepted only when } C = 1 .
\tag{10}
$$

Everything downstream --  the width schedule $w_0 \to$ block widths, the depth
$2^{d_e}$, the block family, the head --  is untouched. The parameter cost is
$(C-1)\,w_0\,\kappa$ additional weights in one layer.

`smoke_test_in_channels.py` verifies (9)-(10) and, critically,
`smoke_test_pipeline_mc.py` check C4 verifies that **zeroing any single channel
changes the embedding** --  i.e. no channel is silently dropped by a shape bug
that would otherwise pass every dimension assertion.

### 3.6 Mode-specific validation, and why latent is refused

*This section establishes which data modes support $C > 1$ and why one does not.*

The previous blanket `NotImplementedError` for `n_channels > 1` fired for *all*
modes, including synthetic. It is replaced by:

| `data_mode` | $C > 1$ | Reason |
|---|---|---|
| `numpy` | supported | pre-computed $(C,K)$ archives |
| `real` | supported | subject to the external engine module |
| `synthetic` | refused | provider emits one population IFR, no channel axis |
| `latent` | **refused by design** | see below |

`LatentBurstProvider` samples $\varphi^{\mathrm{lat}} \in
\mathbb{R}^{n_{\mathrm{lat}}}$ per (condition, trace) and returns a single
population IFR. Extending it to $C > 1$ is not a shape fix but a *generative
choice* with two defensible answers:

- **(a) Shared latent.** One $\varphi^{\mathrm{lat}}$ per recording; the $C$
  channels differ only by independent spike-level noise. Models $C$ noisy views
  of one phenotype.
- **(b) Per-subregion latent.** An independent $\varphi^{\mathrm{lat}}_c$ per
  channel. Models spatial heterogeneity *within* a culture.

These imply different data manifolds and different meanings for the resulting
embedding. The guard names both options in its error message rather than
silently picking one.

### 3.7 Interaction with cross-culture positives

*This section establishes what works, and identifies the one thing that does
not.*

Under `positives_mode="cross_culture"` the positive for an anchor is a window
from a *different* culture of the same class. The batch sampler receives
`trace_of_window`, an array $g$ with $g_m$ the culture id of batch row $m$, and
`exclude_same_culture_positives=True` forbids pairing rows with $g_m = g_{m'}$.

**Case $C > 1$ (`multichannel`): works.** One recording is one $(C,K)$ trace, so
the trace-to-culture map stays one-to-one and $g$ is unchanged. Verified: the
existing `smoke_test_cross_culture_batches.py` passes against the merged tree.

**Case $C = 1$ from subregions (`per_region_single`): does NOT work.** Each
subregion becomes its own trace. Since culture identity *is* the trace index in
the current implementation (`data_splits.py`, where `trace_of_window` is built
from the global trace index $u$), sibling subregions of one recording receive
**different** culture ids. The exclusion rule then permits

$$
\text{positive pair } \big(x^{(u,c)},\, x^{(u,c')}\big), \quad c \neq c',
\tag{11}
$$

i.e. two subregions of the *same* recording at the same time --  precisely the
near-duplicate pairing the mode exists to eliminate.

This is aggravated by easy-positive mining. The EP rule (Sec. 6, KB2, full text
read) selects
$$
x_{\mathrm{ep}} = \arg\min_{x \,:\, \mathcal{C}(x) = \mathcal{C}(x_a)} d\big(f(x_a), f(x)\big),
\tag{12}
$$
the *most similar* same-class sample. Under (11) that is very likely to be the
sibling subregion at the same time index --  a near-duplicate, giving near-zero
loss and a vanishing gradient. The authors' stated purpose for EP is preserving
genuine intra-class variance; near-duplicate siblings are the opposite of that.

**The fix is specified but not implemented (Sec. 5, K3):** a culture-of-trace
grouping vector, so that all $C$ sibling traces (i) land in the same split and
(ii) emit the recording id in $g$. The extractor already writes `culture_id`
into every per-subregion archive in preparation; nothing consumes it yet.

### 3.8 The merge

*This section establishes that the multichannel work and the concurrent
architecture overhaul were reconciled, not one overwritten by the other.*

The two branches diverged from a common ancestor
(`Main/Colab_zips/dsn_pipeline.zip`, confirmed by four non-Python assets being
byte-identical to both sides). Against it, `main` changed ~2100 lines and added
six modules (latent generation, batch geometry, objective utilities, silhouette
floor, latent inspection, trace splits); the multichannel branch changed ~750.
A three-way merge gave:

| Outcome | Files |
|---|---|
| identical | 7 |
| take `main` | 7 |
| take multichannel | 6 |
| clean auto-merge | 3 (`data_pipeline`, `data_splits`, `train`) |
| conflict, add-only | 2 (`config` x4 hunks, `run_optimization` x1) |
| conflict, semantic | 1 (`augmentation` x1) |

The single semantic conflict was in `build_triplet_instance`: `main` had
restructured the retry logic so that `warp_bands` no longer retries (an empty
pool is deliberate under cross-culture), while the multichannel branch had added
the $(C,T)$ handling to the *old* control flow. Resolution keeps `main`'s
control flow and ports the multichannel guard into it. Naive union resolution
was tried and **fails to compile**, which is how the conflict was confirmed
semantic rather than textual.

---

## 4. Summary of results

**R1 --  Shape contract.** Invariant I (time axis last) holds end to end; a trace
is $(C,K)$, a window $(C,W)$, a batch $(M,C,W)$. Derived Sec. 3.1, Eq. (1).

**R2 --  Axis corrections.** Five sites read `shape[0]` as time length and one
sliced the wrong axis; all corrected to `shape[-1]` / `[..., s:e]` across
`run_optimization.py`, `data_splits.py` (both `make_time_segment_splits` and the
newer `make_trace_splits`), `data_pipeline.py` and `preprocessing_cache.py`.
Derived Sec. 3.1, Sec. 3.3, Eq. (4).

**R3 --  Shared augmentation field.** One warp field and one shift per surrogate,
applied identically across channels, Eq. (8); inter-channel structure preserved;
$C = 1$ numerically identical to the previous implementation because the RNG
draw order is unchanged. Derived Sec. 3.4.

**R4 --  Stem.** $\mathrm{Conv1d}_{1 \to w_0} \mapsto \mathrm{Conv1d}_{C \to w_0}$,
Eq. (9); forward contract Eq. (10); downstream architecture untouched;
no channel silently dropped (verified). Derived Sec. 3.5.

**R5 --  Metadata-first channel resolution.** $C$ is read from the archive's
`in_channels`, never inferred from `shape[0]`, because `multichannel` and
`per_region_single` emit identically-shaped arrays with opposite meaning.
Derived Sec. 3.2, Sec. 3.3.

**R6 --  Mode-specific guard.** `numpy`/`real` proceed; `synthetic`/`latent`
refuse $C > 1$, the latter deliberately pending a generative decision between
(a) shared and (b) per-subregion latents. Derived Sec. 3.6.

**R7 --  Extraction.** Three modes; `per_region_single` additionally emits one
$(K,)$ archive per subregion carrying `in_channels=1` and a shared `culture_id`.
Derived Sec. 3.2.

**R8 --  Merge.** One genuine semantic conflict out of six hunks, resolved by
keeping `main`'s control flow; naive union provably fails. Derived Sec. 3.8.

**R9 --  Verification.** 20/20 pre-existing suites pass against the merged tree;
5/5 multichannel suites; 7/7 extractor suites standalone; end-to-end training on
real-format MEA data at $C = 9$ and $C = 1$; every check run **twice** with
identical results; all 68 `.py` files pure ASCII.

**R10 --  Cross-culture compatibility is partial.** $C > 1$ multichannel is
compatible unchanged; `per_region_single` + cross-culture is **not**, by
Eq. (11). Derived Sec. 3.7.

---

## 5. Open points, caveats, and assumptions

**K3 --  Sibling-subregion culture grouping. NOT IMPLEMENTED.** The gap of
Eq. (11)/Sec. 3.7. Requires a culture-of-trace vector threaded through
`assign_cultures` / `make_trace_splits`, a `culture` field in `npz_specs`
records, and a smoke test asserting no same-culture positive pair survives
mining. `culture_id` is already written by the extractor. **Until this lands,
`per_region_single` must not be combined with
`positives_mode="cross_culture"`.**

**A1 --  No efficacy claim.** Nothing here demonstrates that multichannel input
improves phenotype discrimination. The end-to-end runs used *fabricated*
recordings with class-dependent firing rates, so the reported $\mathrm{ARI} =
1.0$ at $C = 9$ measures **plumbing correctness, not scientific signal** --  the
classes were separable by mean rate alone. Treat it as a smoke test.

**A2 --  Scale parity, unresolved.** Real channels are normalised by pool size,
Eq. (3), while the synthetic generator's channels are not. The CNN is not
scale-invariant, so mixing real and synthetic traces in one experiment requires
reconciling this. Author's reasoning, not a literature result.

**A3 --  Subregion geometry is a choice, not a derivation.** $C = 9$, $E = 9$,
greedy MFR-ordered centres and nearest-neighbour membership were chosen for
convenience. No sensitivity analysis over $C$, $E$, the seeding rule, or the
resulting spatial coverage has been performed.

**A4 --  Disjointness assumed, coverage not.** The $\mathcal{M}_c$ are disjoint by
construction, but their union need not cover $\mathcal{E}$; electrodes outside
every subregion contribute to no channel. Whether that discards signal is
untested.

**O1 --  `index_base`.** 0 vs 1 for MEA electrode indices is configurable and
**unverified against real files**. One real folder settles it.

**O2 --  ptrain format.** The loader requires a binary `uint8` raster of shape
$(n,1)$ and fails loud otherwise. If the group's files store spike *indices*,
Stage 1 stops. Unverified against real data.

**O3 --  $f_s^{\mathrm{raw}}$.** Defaults to 10110.09 Hz; per-dataset correctness
unverified.

**O4 --  Smoothing bandwidth $\sigma$.** Inherited from the single-channel design.
Whether the bandwidth appropriate for a whole-culture rate (pooling
$\lvert\mathcal{E}\rvert \approx 10^2$ electrodes) is appropriate for a
subregion rate (pooling $E = 9$, hence far sparser and noisier) has **not** been
examined. This is a real statistical question, not a detail.

**C1 --  Duplicate module.** `generate_burst_data.py` ships in both the pipeline
and MultiChannel packages, byte-identical at merge time. They must be kept in
sync or consolidated onto `PYTHONPATH`.

**C2 --  Version skew.** Verified on torch 2.13.0+cu130 (CPU), numpy 2.4.4,
scipy 1.17.1; `environment.yml` pins numpy 1.26. Re-run on the cluster.

**C3 --  Pre-existing failures, not caused here.** `smoke_test_inspect_latent.py`
fails identically on untouched `main` (path dependency on `hpc/`). Five
`Main/hpc/Config/*.json` files are committed with CRLF against a
`.gitattributes` declaring `eol=lf`. `Main/hpc/run_all_smoke_tests.sh` could not
resolve `Main/` from its own committed location; fixed on this branch.

---

## 6. References / further reading

**Project knowledge base (full text retrieved and read):**

- **KB1.** Iglesias et al., *Data augmentation techniques in time series domain:
  a survey and taxonomy*, Neural Computing and Applications 35:10123-10145
  (2023). Grounds Sec. 3.4: the cubic-spline knot construction for magnitude and
  time warping, and the listing of channel permutation as a multivariate
  technique (not adopted here, Sec. 3.4).
- **KB2.** Xuan, Stylianou & Pless, *Improved Embeddings with Easy Positive
  Triplet Mining*. Grounds Sec. 3.7, Eq. (12): the easy-positive selection rule and
  the stated rationale of preserving intra-class variance rather than
  over-clustering.
- **KB3.** `Patched/Augmentation/Data_Augmentation_Pipeline_Theory_and_Implementation.md`
  (this project). Grounds Sec. 3.4: log-space positivity of the magnitude warp,
  endpoint pinning of the time warp, the knot-count rule, and the flagged
  log-normal mean bias $\mathbb{E}[a(t_k)] = \exp(\sigma_{\mathrm{mag}}^2/2) > 1$.

**Literature searches performed and their outcome:**

- **PubMed**, two query formulations on MEA subregion parcellation / multichannel
  CNN classification of neuronal cultures: **0 results each**. An earlier query
  on MEA population-rate normalisation returned one record (PMID 15044515) with
  **no PMC full text**, so per the full-text rule nothing from it is used here.
- **bioRxiv/medRxiv**: the connector supports category + date filtering only, not
  keyword search. A recent-neuroscience listing was retrieved and contained
  **nothing relevant** to this topic. **No preprint informs any claim in this
  document.**

**Stated from the codebase rather than from a source:** all `file.py:line`
references, the merge classification of Sec. 3.8, and every count in Sec. 4 R9 were
obtained by executing the code, not from memory or literature.

**Stated as author's reasoning, unsourced:** A2 (scale parity), the aggravation
argument in Sec. 3.7 combining Eq. (11) with Eq. (12), and the rationale for the
shared-field decision in Sec. 3.4. Eq. (12) itself is from KB2; the inference that
sibling subregions will dominate the EP selection is mine and is untested.
