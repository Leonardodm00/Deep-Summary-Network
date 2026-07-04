# Technical Report — `backbone.py`
## A RegNet-inspired 1D-CNN summary network for cumulative-IFR MEA traces

**Document type.** Technical (implementation) report. This is the first of two
companion documents; the second, the *Theoretical Documentation*, covers the
mathematical and neuroscientific rationale and the supporting literature. This
report documents *the code as written*: its contract, structure, algorithms,
numerical behaviour, verification, and integration surface.

**Artifacts under documentation.** `backbone.py` (the module) and
`smoke_test_backbone.py` (its correctness harness).

**Provenance of statements (please read first).** To honour the project's
transparency rule, every claim in this report falls into one of four
categories, and I try to make the category clear wherever it is not obvious:

- **[CODE]** — derived directly from the source file being documented. This is
  the primary source and the default for the API, configuration, data-flow,
  and verification sections. It is authoritative because it *is* the artifact.
- **[KB]** — grounded in a full-text PDF present in the project knowledge base
  and actually read: the RegNet paper (Radosavovic et al., *Designing Network
  Design Spaces*, CVPR 2020) and the ResNeXt paper (Xie et al., *Aggregated
  Residual Transformations for Deep Neural Networks*, CVPR 2017). Where I quote
  a specific equation or convention I cite it as [KB: RegNet] or [KB: ResNeXt].
- **[MEASURED]** — an empirical number obtained by running the code in this
  environment (PyTorch 2.12, CPU). These are properties of *this*
  implementation, not literature results.
- **[REASONING]** — my own engineering rationale, labelled as such.

**A note on the literature scope of this document.** The reference methods for
the *architecture* (RegNet, ResNeXt, GroupNorm, zero-γ residual init,
metric-learning L2 normalization) are computer-vision / machine-learning
papers. Two of them (RegNet, ResNeXt) are in the project knowledge base and are
cited from full text. The others (GroupNorm — Wu & He 2018; the zero-init-γ
"bag of tricks" — He et al. 2019 / Goyal et al. 2017) are **not** in the
project knowledge base, and I have **not** retrieved their full text here; they
are described only at the textbook/uncontested level and flagged accordingly.
PubMed was not queried for this report because these are not biomedical
sources and PubMed does not index them; the biomedical grounding (MEA, IFR,
neuronal-dynamics embeddings) belongs to the *Theoretical Documentation*, where
PubMed full-text retrieval will be used per the project rules.

---

## Table of contents

1. Purpose and scope
2. Public API and the I/O contract
3. Dependencies, environment, and portability constraints
4. Source-file map
5. Configuration object: `BackboneConfig`
6. Core algorithms (with full notation)
7. Component modules
8. Parameter initialization
9. End-to-end forward data flow (worked shape trace)
10. Numerical and correctness properties
11. Deviations and simplifications vs. reference methods
12. Verification — the smoke test
13. Empirical characteristics (measured)
14. Integration notes for the optimization stage (Topic 3)
15. Known limitations and open items
16. Sources and provenance

---

## 1. Purpose and scope

`backbone.py` implements the **summary network** of the pipeline: a
pure function

$$
f_\theta : \mathbb{R}^{M \times T} \;\longrightarrow\; \mathbb{R}^{M \times E},
$$

parameterized by weights $\theta$, that maps a batch of $M$ univariate
cumulative instantaneous-firing-rate (IFR) traces, each of length $T$, to $M$
embedding vectors of dimension $E$. When L2 normalization is enabled (the
default), each output row lies on the unit hypersphere
$\mathbb{S}^{E-1} = \{u \in \mathbb{R}^{E} : \lVert u \rVert_2 = 1\}$, which is
the natural geometry for the downstream **triplet metric loss with cosine
distance**. **[CODE]**

Symbols used throughout ($M$: batch size / number of windows; $T$: input
length in samples; $E$: embedding dimension) are introduced once here and
reused. Additional symbols are introduced at first use in §6.

**Design intent.** **[REASONING]** The module is deliberately a *pure
tensor-to-tensor* `nn.Module`: it contains no device management, no data
augmentation, no training `State` flag, and no I/O. This is the "simulation
logic" layer in the sense of the project's separation-of-concerns directive —
data loading (Topic 1) and training/optimization (Topic 3) live in other
modules. The purity buys three things: it is trivially unit-testable (see
§12), it is safe under `DistributedDataParallel`, `torch.compile`, and
automatic mixed precision (AMP), and its output for a given input never depends
on batch composition or on `train()`/`eval()` mode (see §10).

**What this report does not cover.** The optimization/search harness, the data
loader, the loss/miner configuration, and the biomedical justification for the
representation are out of scope here and are addressed elsewhere.

---

## 2. Public API and the I/O contract

The module exposes exactly three public names. **[CODE]**

| Name | Kind | Role |
|---|---|---|
| `BackboneConfig` | frozen dataclass | Immutable, validated hyperparameter bundle (§5). |
| `OneDCNNBackbone` | `nn.Module` | The assembled network (§7.5). |
| `build_backbone(cfg)` | factory function | Type-checks `cfg` and returns an initialized `OneDCNNBackbone`. |

Everything else (`Stem`, `ResNetBlock`, `ResNeXtBlock`, `MultiScaleHead`, and
the pure helpers `compute_block_widths`, `allocate_stages`,
`adjust_width_to_group`, `pick_G`, `conv_out_length`, `make_norm`, `_pool`,
`_pad_same_1d`) is public at the Python level (no name mangling beyond the two
`_pool`/`_pad_same_1d` helpers) and is intended to be importable for testing,
but the three names above are the supported surface.

**Input contract.** `forward` accepts either shape and normalizes internally:

- `(M, T)` — a batch of raw traces; a singleton channel axis is inserted.
- `(M, 1, T)` — a batch already carrying the explicit single input channel.

Any other rank, or a channel dimension $\neq 1$, raises `ValueError`. The single
channel encodes the univariate-trace assumption. Dtype is expected to be
`float32`; device is whatever the caller placed the tensor and module on.
**[CODE]**

**Output contract.** A tensor of shape `(M, E)` where `E = cfg.embedding_size`.
If `cfg.l2_normalize` is `True` (default), each row $u_i$ satisfies
$\lVert u_i \rVert_2 = 1$ up to floating-point tolerance (the normalization uses
`F.normalize(z, p=2, dim=1)`, i.e. $u = z / \max(\lVert z \rVert_2, \varepsilon)$
with PyTorch's default $\varepsilon = 10^{-12}$; the $\max(\cdot,\varepsilon)$
guards the zero vector). **[CODE]**

**Minimal usage.** **[CODE]**

```python
import torch
from backbone import BackboneConfig, build_backbone

cfg = BackboneConfig(depth_exponent=4, width_multiplier=2.0,
                     block_family=0, embedding_size=16)   # ResNet family
model = build_backbone(cfg)

x = torch.randn(32, 1024)          # (M=32, T=1024); (32, 1, 1024) also valid
z = model(x)                       # (32, 16), each row L2-normalized
assert z.shape == (32, 16)
```

Introspection attributes populated at construction: `model.stage_widths`
(list of per-stage widths $w^{(i)}$), `model.stage_depths` (list of per-stage
block counts $d^{(i)}$), and `model.head.in_features` (the flattened width fed
to the final `Linear`). These are convenient for logging and for the search
harness. **[CODE]**

---

## 3. Dependencies, environment, and portability constraints

**Runtime dependencies.** `torch` (tested on 2.12.1) and the Python standard
library (`math`, `dataclasses`, `typing`). No NumPy, no SciPy, no
`pytorch_metric_learning` — those belong to the training layer. This keeps the
module importable on a bare cluster node. **[CODE]**

**Establishing-ecosystem note.** **[REASONING]** In line with the project's
"leverage the ecosystem" directive, the module builds entirely on `torch.nn`
primitives (`Conv1d`, `GroupNorm`, `LayerNorm`, `Linear`, `Dropout`) and
`torch.nn.functional` (`pad`, `normalize`); the only hand-written numerics are
the RegNet width arithmetic and the padding length formula (§6), which have no
suitable library equivalent and are covered by the smoke test.

**HPC portability (ASCII).** The file is intentionally **pure ASCII** — no
Unicode in identifiers, strings, or comments — so that transfer to the
`davinci-1` cluster (via tools that may re-encode to cp1252, e.g. MobaXterm,
copy-paste, or Windows-side `scp`) cannot introduce a non-UTF-8 byte that
would raise a `SyntaxError` at import time. Greek letters in the design (e.g.
the normalization scale) are spelled out (`gamma`) in code; this report, being
a document rather than cluster-bound source, uses the mathematical symbols
($\gamma$, $\beta$) freely. **[CODE]**

**Numerical modes.** The module runs under CPU and CUDA identically (no
device-specific code paths) and is AMP-safe: an autocast forward in `bfloat16`
was exercised and returns a finite `(M, E)` tensor (§13). **[MEASURED]**

---

## 4. Source-file map

The file is organized top-to-bottom as a dependency-respecting stack, so it
reads linearly from primitives to assembly. **[CODE]**

| Section | Contents |
|---|---|
| Module docstring + constants | Contract summary; `_POOL_ORDER = ("mean","max","std")` (canonical pooling order); `_STD_EPS = 1e-5`. |
| Configuration | `BackboneConfig` (frozen dataclass) + `__post_init__` validation. |
| Pure helpers | `conv_out_length`, `adjust_width_to_group`, `pick_G`, `compute_block_widths`, `allocate_stages`, `_pad_same_1d`, `make_norm`. |
| Stem | `Stem` module. |
| Residual blocks | `ResNetBlock`, `ResNeXtBlock`. |
| Head | `_pool`, `MultiScaleHead`. |
| Assembly | `OneDCNNBackbone` (+ `_init_weights`, `forward`) and `build_backbone`. |

The pure helpers carry no `torch` state (they operate on Python ints/floats and
lists), which is why the smoke test can exercise the entire allocation logic —
the part most prone to off-by-one and quantization bugs — without constructing
any tensors. **[CODE / REASONING]**

---

## 5. Configuration object: `BackboneConfig`

`BackboneConfig` is a `@dataclass(frozen=True)`: immutable (hashable, safe to
share across processes and to use as a cache key) and validated in
`__post_init__`. Every field, its default, and its meaning: **[CODE]**

| Field | Type | Default | Meaning / admissible range |
|---|---|---|---|
| `depth_exponent` | `int` | `4` | $d$; total residual blocks $D = 2^{d}$. Search grid $\{3,4,5,6\}$. Must be $\geq 1$. |
| `width_multiplier` | `float` | `2.0` | $w_m$; RegNet width multiplier. Search range $[1.5, 3.0]$ (continuous). Must be $> 1.0$. |
| `stem_width` | `int` | `16` | $w_0 = w_a$; stem output width and the RegNet slope (simplified variant, §6.1). Must be $\geq 1$. |
| `block_family` | `int` | `0` | $0 =$ ResNet block, $1 =$ ResNeXt (grouped) block. Must be $\in\{0,1\}$. |
| `group_width` | `int` | `16` | $g_w$; ResNeXt channels per group (§6.2, §7.3). Must be $\geq 1$. |
| `embedding_size` | `int` | `16` | $E$; output dimension. Search grid e.g. $\{8,\dots,16\}$. Must be $\geq 1$. |
| `l2_normalize` | `bool` | `True` | If true, L2-normalize each output row onto $\mathbb{S}^{E-1}$. |
| `head_fusion` | `bool` | `False` | If false, pool the last stage only; if true, fuse all stages ("pseudo-dense", §7.4). |
| `head_pool_ops` | `tuple[str,...]` | `("mean",)` | Subset of `{mean,max,std}`; per-stage length statistics. Must be non-empty and a subset. |
| `head_prenorm` | `bool` | `True` | Per-stage `LayerNorm` before concat; **operative only when `head_fusion` is true**. |
| `norm_target_cpg` | `int` | `16` | $c_g^{\star}$; target channels per GroupNorm group. Search set $\{4,8,16,24,32\}$. Must be $\geq 1$. |
| `norm_g_max` | `int` | `32` | Cap on the number of GroupNorm groups $G$. Must be $\geq 1$. |
| `stem_kernel` | `int` | `5` | Stem convolution kernel. Must be $\geq 1$. |
| `stem_stride` | `int` | `4` | Stem downsampling stride. Must be $\geq 1$. |
| `stage_kernel` | `int` | `3` | Convolution kernel inside blocks. Must be $\geq 1$. |
| `downsampling_rate` | `int` | `2` | Stride of the first block of each stage. Must be $\geq 1$. |
| `dropout` | `float` | `0.0` | Dropout probability applied after each block's final ReLU. Must be $\in [0,1)$. |

**Validation behaviour.** `__post_init__` raises `ValueError` on any
out-of-range field, on an empty or out-of-vocabulary `head_pool_ops`, or on a
kernel/stride $< 1$; `build_backbone` additionally raises `TypeError` if handed
a non-`BackboneConfig`. Validation is eager (at construction), so a malformed
search point fails fast rather than mid-forward. **[CODE]**

**On `head_pool_ops` ordering.** The config stores the *set* of requested
statistics; the head re-orders them into the canonical `_POOL_ORDER`
(`mean`, then `max`, then `std`) so the concatenated feature columns are
deterministic regardless of the order the user wrote them. **[CODE]**

**Ablation axes vs. search axes.** **[REASONING]** `head_fusion`,
`head_pool_ops`, and `head_prenorm` are intended as **fixed-per-run ablation
knobs**, not Bayesian-optimization dimensions, at least initially; the search
dimensions are `depth_exponent`, `width_multiplier`, `block_family`,
`embedding_size`, `norm_target_cpg`, and optionally `group_width` (see §14).

---

## 6. Core algorithms (with full notation)

This section documents the four non-trivial numerical routines. I carry the
full notation throughout and flag every simplification relative to the source
method, per the project's notation-fidelity rule.

### 6.1 RegNet quantized-linear width allocation — `compute_block_widths`

**Reference.** [KB: RegNet] The RegNet design space parameterizes per-block
widths by a *quantized linear* function. In the paper's notation, with block
index $j$ and stage index $i$, one first sets an unquantized per-block width

$$
u_j = w_0 + w_a \cdot j, \qquad 0 \le j < d, \tag{RegNet Eq. 2}
$$

with depth $d$, initial width $w_0 > 0$, and slope $w_a > 0$. A multiplier
$w_m > 0$ controls quantization via $s_j$ defined by

$$
u_j = w_0 \cdot w_m^{\,s_j}, \tag{RegNet Eq. 3}
$$

and the quantized per-block width is obtained by rounding $s_j$ (the paper
writes $\lfloor s_j \rceil$ for round-to-nearest):

$$
w_j = w_0 \cdot w_m^{\,\lfloor s_j \rceil}. \tag{RegNet Eq. 4}
$$

**Our simplified variant.** [KB: RegNet] The paper reports (Fig. 9, middle)
that *"setting $w_0 = w_a$ … performs even better"*, collapsing Eq. 2 to
$u_j = w_a \cdot (j+1)$, but declines to impose it in order to keep model
diversity. **We deliberately impose it here** and additionally identify $w_0 =
w_a = $ `stem_width`. This is the simplification flagged explicitly:

> **Simplification (flagged).** We set $w_0 = w_a = $ `stem_width`. Then
> $u_j = w_0\,(j+1)$ and $s_j = \log(u_j/w_0)/\log w_m = \log(j+1)/\log w_m$,
> so the code computes, for each block $j \in \{0,\dots,2^{d}-1\}$,
> $$ s_j = \frac{\log(j+1)}{\log w_m}, \qquad w_j = w_0 \cdot w_m^{\,\lfloor s_j \rceil}. $$

Here $\lfloor \cdot \rceil$ is Python's `round`, i.e. **round-half-to-even
(banker's rounding)**, which matches NumPy's `np.round` used in the reference
implementation (`pycls`); this is noted because half-integer $s_j$ values are
resolved to even — a deterministic but non-obvious tie-break. **[CODE]**

**Two further departures from the paper's text, flagged:** [KB: RegNet]
1. The paper, when tabulating four-stage networks, *"ignore[s] the parameter
   combinations that give rise to a different number of stages."* We do the
   opposite: we **keep** whatever number of stages the quantization produces
   (see §6.2). The stage count $S$ is therefore emergent.
2. The historical `[:-2]` depth truncation from the group's *previous* code
   (an ad-hoc "actual depth = selected − 2 for math reasons") is **removed**;
   $D = 2^d$ blocks are allocated with no truncation.

`compute_block_widths(depth_exponent, stem_width, width_multiplier)` returns the
list of **floating** per-block widths $[w_0, w_1, \dots, w_{D-1}]$ (length
$D = 2^d$); rounding to integers and group snapping happen in §6.2. Because
$s_j$ is non-decreasing in $j$ and $w_m > 1$, the sequence $w_j$ is
non-decreasing. **[CODE]**

**Non-consecutive stage levels (a subtlety worth recording).** [REASONING /
KB: RegNet] For $w_m < 2$ the rounded exponent $\lfloor s_j \rceil$ can *skip*
an integer between consecutive blocks (e.g. at $w_m = 1.5$, block $0$ has level
$0$ but block $1$ already has level $\lfloor \log 2/\log 1.5 \rceil = 2$). The
code handles this correctly because it computes $w_j = w_0 w_m^{\lfloor s_j
\rceil}$ from the *actual* rounded exponent and then takes distinct
consecutive widths (§6.2); a skipped level simply produces a wider jump between
adjacent stage widths, exactly as `pycls` behaves. It is never a bug, but it is
why the stage widths are not always the tidy $w_0 \cdot 2^i$ ladder.

### 6.2 Stage formation and group-compatible width snapping — `allocate_stages`, `adjust_width_to_group`

`allocate_stages(depth_exponent, stem_width, width_multiplier, group_width)`
converts the per-block widths into the per-stage $(w^{(i)}, d^{(i)})$ format.
[KB: RegNet] The paper's rule is: *"each stage $i$ has block width $w_i = w_0
\cdot w_m^{i}$ and number of blocks $d_i = \sum_j \mathbb{1}[\lfloor s_j \rceil
= i]$"* — i.e. count consecutive blocks of equal width. The code does this by:

1. rounding each floating $w_j$ to the nearest integer;
2. **snapping** each to a group-compatible value via `adjust_width_to_group`;
3. grouping maximal runs of equal snapped widths into stages, so stage $i$ gets
   width $w^{(i)}$ and depth $d^{(i)} =$ (length of the run).

The number of stages $S = \lvert \{\text{distinct snapped widths}\} \rvert$ is
whatever falls out — the **emergent stage count**. **[CODE]**

**Width snapping.** `adjust_width_to_group(width, group_width)` sets
$g = \min(g_w, \text{width})$ and returns
$\max\!\big(g,\; \lfloor \text{width}/g \rceil \cdot g\big)$,
i.e. the nearest multiple of $g$ (floored at $g$). **[CODE]**

> **Why (flagged as an engineering choice).** [REASONING, standard in `pycls`]
> A grouped convolution requires the channel count to be divisible by the
> channels-per-group. Rather than reject a search point whose quantized width
> is not divisible by $g_w$ (which would punch holes in the search space
> mid-optimization), we snap the width to the nearest compatible multiple. The
> use of $g = \min(g_w, \text{width})$ is a defensive guard for the degenerate
> case width $< g_w$; for every width in our regime (all $\geq$ `stem_width`
> $= 16 = g_w$ by default) we have $g = g_w$, so snapping is exactly
> "round to a multiple of $g_w$." Because the snap is monotone in width, the
> snapped per-block sequence is still non-decreasing and the run-grouping in
> step 3 remains valid.

### 6.3 GroupNorm group-count selection — `pick_G`, `make_norm`

Every normalization in the network is a `GroupNorm` whose group count is chosen
per feature width by `pick_G(num_channels, target_cpg, g_max)`. Let $C$ be the
channel count, $c_g^{\star} = $ `norm_target_cpg` the target channels per group,
and $G_{\max} = $ `norm_g_max`. The routine computes a target group count
$G_{\text{tgt}} = \max(1, \lfloor C / c_g^{\star} \rfloor)$, caps it at
$\min(G_{\text{tgt}}, G_{\max})$, and returns the **largest** $G$ in
$\{1, \dots, \min(G_{\text{tgt}}, G_{\max})\}$ that divides $C$; if none does it
falls back to $G = 1$. **[CODE]**

`make_norm(num_channels, cfg)` simply wraps `pick_G` and returns
`nn.GroupNorm(num_groups=G, num_channels=C)`. **[CODE]**

Two boundary behaviours worth recording: **[CODE / REASONING]**
- $G = 1$ is GroupNorm degenerating to a per-sample **LayerNorm over all
  channels-and-length** (normalization statistics computed across the whole
  $(C, L)$ map for each sample). This is the fallback and also the natural
  outcome for narrow stages (e.g. $C = 16$, $c_g^{\star} = 16 \Rightarrow
  G_{\text{tgt}} = 1$).
- Divisibility is guaranteed by construction because stage widths are snapped
  to multiples of $g_w$ (§6.2), and by the search convention $g_w$ and
  $c_g^{\star}$ are chosen from compatible sets; but `pick_G` is robust even if
  they are not, because it searches downward for a divisor.

The salient property GroupNorm buys is **batch independence**: its statistics
are computed per-sample, so the network's output for a given input is identical
in `train()` and `eval()` and does not depend on the other rows of the batch
(verified in §10, §12). This is the reason BatchNorm was replaced.
[REASONING; GroupNorm's original reference, Wu & He 2018, is not in the project
KB and is not cited here beyond this textbook-level statement.]

### 6.4 Length-preserving / strided padding — `_pad_same_1d`, `conv_out_length`

`conv_out_length(l_in, kernel, stride, padding, dilation=1)` returns the
standard 1-D convolution output length

$$
L_{\text{out}} = \left\lfloor
\frac{L_{\text{in}} + 2p - \mathrm{dil}\cdot(k-1) - 1}{s} \right\rfloor + 1,
$$

with $L_{\text{in}}$ the input length, $k$ the kernel, $s$ the stride, $p$ the
symmetric padding, and $\mathrm{dil}$ the dilation. It is a pure helper used for
reasoning and testing. **[CODE]**

The network never relies on symmetric padding for the strided (downsampling)
convolutions; instead `_pad_same_1d(x, kernel, stride, dilation=1)` applies
**explicit asymmetric zero-padding computed at forward time** so that a strided
convolution lands exactly on

$$
L_{\text{out}} = \left\lceil \frac{L_{\text{in}}}{s} \right\rceil .
$$

Concretely, for $s > 1$ it computes the total padding

$$
p_{\text{tot}} = \max\!\Big(0,\; (L_{\text{out}} - 1)\,s +
\mathrm{dil}\cdot(k - 1) + 1 - L_{\text{in}}\Big),
$$

splits it as $p_{\text{left}} = \lfloor p_{\text{tot}}/2 \rfloor$,
$p_{\text{right}} = p_{\text{tot}} - p_{\text{left}}$, and calls
`F.pad(x, (p_left, p_right))`; the following convolution then uses `padding=0`.
For $s = 1$ it returns the input unchanged (length preservation is handled by
static `padding = (k-1)//2` on the odd stride-1 kernels). **[CODE]**

> **Why compute padding in `forward` (flagged as an engineering choice).**
> [REASONING] Because $p_{\text{tot}}$ depends on the *actual* incoming length,
> computing it at forward time makes the module correct for **any** $T$ without
> bookkeeping, and guarantees that the strided main path and its stride-$s$
> $1\times1$ projection shortcut produce identical lengths: a $1\times1$ stride-
> $s$ convolution with zero padding already yields $\lceil L_{\text{in}}/s
> \rceil$, so padding the main path to the same target aligns the two exactly.
> Each block additionally asserts equal lengths before the residual add and
> raises `RuntimeError` on mismatch — a defensive net beyond the empirical
> shape checks in §12.

---

## 7. Component modules

### 7.1 Stem — `Stem`

A single strided convolution that coarsens the raw trace before the residual
body: `Conv1d(1 -> stem_width, kernel=stem_kernel, stride=stem_stride,
bias=False)` followed by `GroupNorm` (via `make_norm`) and `ReLU`. The strided
convolution is preceded by `_pad_same_1d`, so the stem reduces length by a
factor `stem_stride` (default $4$). Bias is disabled because a normalization
layer follows (a standard convention: the norm's shift $\beta$ subsumes any
bias). **[CODE]**

### 7.2 ResNetBlock — `ResNetBlock`

A D2L-style two-convolution residual block. **[CODE]** With input channels
$C_{\text{in}}$, output channels $C_{\text{out}}$, kernel $k$, and stride $s$:

- **Main path:** `Conv1d(C_in -> C_out, k, stride=s)` $\to$ `GroupNorm`
  ($\to$ `ReLU`) $\to$ `Conv1d(C_out -> C_out, k, stride=1)` $\to$ `GroupNorm`.
  The first convolution carries the stride; for $s>1$ its `padding` is $0$ and
  `_pad_same_1d` supplies the asymmetric padding, while the second convolution
  uses static `padding=(k-1)//2` to preserve length.
- **Shortcut:** identity when $s = 1$ **and** $C_{\text{in}} = C_{\text{out}}$;
  otherwise a **projection** `Conv1d(C_in -> C_out, 1, stride=s)` $\to$
  `GroupNorm`. This is the type-B projection convention. [KB: ResNeXt — *"The
  shortcuts are identity connections except for those increasing dimensions
  which are projections (type B)"*, and *"Downsampling … is done by stride-2
  convolutions in the 3×3 layer of the first block in each stage"*.]
- **Merge:** $\mathrm{ReLU}(\text{main} + \text{shortcut})$, then `Dropout`
  (identity when `dropout == 0`). The post-addition ReLU placement follows the
  same convention. [KB: ResNeXt]

The block's final normalization (`norm2`) is tagged `last_branch_norm` for the
zero-γ initialization (§8). **[CODE]**

### 7.3 ResNeXtBlock — `ResNeXtBlock`

A grouped residual block with **bottleneck ratio $b = 1$ (no bottleneck
compression added)**. **[CODE]** Structure:

- **Main path:** `1x1 Conv(C_in -> C_out)` $\to$ `GN` $\to$ `ReLU` $\to$
  **grouped** `Conv(C_out -> C_out, k, stride=s, groups=g_{\text{eff}}^{-1}
  C_{\text{out}})`$\to$ `GN` $\to$ `ReLU` $\to$ `1x1 Conv(C_out -> C_out)`
  $\to$ `GN`, where the number of groups is $C_{\text{out}}/g_{\text{eff}}$ with
  $g_{\text{eff}} = \min(g_w, C_{\text{out}})$. The strided (middle) convolution
  is preceded by `_pad_same_1d`; the $1\times1$ convolutions preserve length.
- **Shortcut** and **merge:** identical convention to §7.2 (type-B projection
  when downsampling or changing width; post-add ReLU; optional dropout). The
  final normalization (`n3`) is the `last_branch_norm`.

**Cardinality.** The number of groups $C_{\text{out}}/g_{\text{eff}}$ is the
ResNeXt *cardinality* [KB: ResNeXt]. Because widths are snapped to multiples of
$g_w$ (§6.2), the divisibility `out_ch % g_eff == 0` holds by construction; the
constructor nonetheless raises `RuntimeError` if it is ever violated.

> **Deviation (flagged), important.** [KB: ResNeXt / RegNet] The *original*
> ResNeXt block (Xie et al. 2017) uses a **bottleneck** (reduce width with the
> first $1\times1$, do the grouped $3\times3$ at the reduced width, expand with
> the last $1\times1$). Our block sets the bottleneck ratio to $1$, so all
> three convolutions operate at the full stage width $C_{\text{out}}$. This
> matches the **RegNet** finding that *"the best models do not use either a
> bottleneck or inverted bottleneck"* [KB: RegNet], and matches the RegNet
> parameterization's $b = 1$ setting; it is a deliberate departure from the
> original ResNeXt design and should be described as *"a RegNet $b{=}1$ grouped
> block,"* not *"a ResNeXt bottleneck block."*

### 7.4 Head — `_pool`, `MultiScaleHead`

**Pooling.** `_pool(h, op)` maps a $(M, C, L)$ feature map to $(M, C)$ by one of
three length statistics: **[CODE]**

- `mean`: $\frac{1}{L}\sum_{\ell} h_{\cdot,\cdot,\ell}$ (`h.mean(dim=-1)`).
- `max`: $\max_{\ell} h_{\cdot,\cdot,\ell}$ (`h.amax(dim=-1)`).
- `std`: computed as $\sqrt{\mathrm{Var}_\ell(h) + \varepsilon_{\text{std}}}$
  with the **population** variance
  $\mathrm{Var}_\ell(h) = \frac{1}{L}\sum_\ell (h_{\cdot,\cdot,\ell} -
  \bar h)^2$, $\bar h = \mathrm{mean}_\ell(h)$, and
  $\varepsilon_{\text{std}} = 10^{-5}$ **inside** the square root.

> **Why $\varepsilon$ inside the root (flagged).** [REASONING] Placing
> $\varepsilon_{\text{std}}$ inside $\sqrt{\cdot}$ keeps the gradient finite
> when the variance is exactly zero (a constant channel), which can occur at
> initialization; $\partial \sqrt{v+\varepsilon}/\partial v = 1/(2\sqrt{v+
> \varepsilon})$ is bounded at $v=0$, whereas $\sqrt{v}$ has an infinite
> gradient there. The population (biased) variance is used rather than the
> unbiased $\frac{1}{L-1}$ estimator to avoid a division by $L-1$ when $L=1$
> (the deepest stages can be very short) and to sidestep the deprecated
> `unbiased=`/`correction=` API differences across PyTorch versions — both
> portability considerations for the cluster.

**`MultiScaleHead`.** Aggregates stage outputs into the embedding. **[CODE]**
Let the ordered pooling ops be $\mathcal{O} \subseteq \{\text{mean, max, std}\}$
(canonical order, size $\lvert\mathcal{O}\rvert$), and let the fused stages have
widths $\{C^{(i)}\}$.

- **Baseline (`head_fusion = False`).** Only the last stage is used. Its
  feature map is pooled by each op in $\mathcal{O}$; the resulting vectors are
  concatenated to width $\lvert\mathcal{O}\rvert\,C^{(S)}$, projected by a bare
  `Linear` to $E$, and optionally L2-normalized. No LayerNorm is applied
  (`head_prenorm` is inoperative without fusion).
- **Fusion (`head_fusion = True`, the "pseudo-dense" variant).** *Every* stage
  $i$ is pooled by each op, concatenated per stage to width
  $\lvert\mathcal{O}\rvert\,C^{(i)}$, passed through a **per-stage `LayerNorm`**
  (when `head_prenorm`), then all stages are concatenated to total width
  $\lvert\mathcal{O}\rvert \sum_i C^{(i)}$ and projected by a bare `Linear` to
  $E$, optionally L2-normalized.

The input width to the final `Linear` is exposed as `head.in_features`
$= \lvert\mathcal{O}\rvert \sum_{i \in \text{fused}} C^{(i)}$. **[CODE]**

> **Provenance of the fusion design (flagged).** [REASONING, loosely inspired
> by KB: DenseNet] The all-stage tap-and-concatenate is a *design extension*
> of this project, motivated by the multi-scale / feature-reuse idea associated
> with densely connected networks (the DenseNet PDF is in the project KB). It
> is **not** part of RegNet or ResNeXt. The per-stage `LayerNorm` is included
> so that the wide, late-stage features do not dominate the concatenation over
> the narrow early-stage features purely by scale at initialization; this
> rationale is my own and belongs to the theory document for fuller treatment.

### 7.5 Assembly and forward — `OneDCNNBackbone`, `build_backbone`

**Construction.** `OneDCNNBackbone.__init__` (i) calls `allocate_stages` to get
$\{(w^{(i)}, d^{(i)})\}$ and stores them on the module; (ii) builds the `Stem`;
(iii) selects the block class by `block_family` and builds each stage as an
`nn.Sequential` of blocks, where the **first** block of stage $i$ carries stride
`downsampling_rate` and the rest stride $1$, threading the running input width
$C_{\text{in}}$ from `stem_width` through the stages; (iv) builds the
`MultiScaleHead` over the stage widths; and (v) runs `_init_weights` (§8). The
stages are held in an `nn.ModuleList` (not a single `Sequential`) precisely so
`forward` can capture each stage's output for the fusion head. **[CODE]**

**Forward.** Insert the channel axis if the input is rank-2; validate rank and
channel count; run the stem; iterate stages, appending each stage's output to a
list; pass the list of stage outputs to the head; return the head's output.
**[CODE]**

**Factory.** `build_backbone(cfg)` type-checks and returns the initialized
module; it is the supported construction entry point. **[CODE]**

---

## 8. Parameter initialization — `_init_weights`

Two passes over `self.modules()`: **[CODE]**

1. **Generic.** Every `Conv1d` and `Linear` weight is initialized with
   `kaiming_normal_(mode="fan_out", nonlinearity="relu")` and its bias (if any)
   zeroed; every `GroupNorm`/`LayerNorm` gets $\gamma = 1$, $\beta = 0$.
2. **Residual-friendly zero-γ.** Every module tagged `last_branch_norm` (the
   final normalization of each residual branch — `norm2` in ResNet, `n3` in
   ResNeXt) has its $\gamma$ set to $0$.

**Effect of the zero-γ pass.** [REASONING; standard "bag of tricks" technique,
He et al. 2019 / Goyal et al. 2017 — not in project KB, stated at textbook
level] With the last branch normalization scaled by $\gamma = 0$, each residual
branch outputs $0$ at initialization, so every block computes
$\mathrm{ReLU}(0 + \text{shortcut}) = \mathrm{ReLU}(\text{shortcut})$ — i.e. the
network starts as a near-identity/shortcut-only mapping, which stabilizes the
early optimization of deep stacks. A direct consequence, documented so it is not
mistaken for a bug (§10, §13): at exactly step $0$ the *internal* convolution
weights of a branch receive **zero** gradient (their contribution is scaled by
$\gamma = 0$), while $\gamma$ itself receives a non-zero gradient; after the
first optimizer step $\gamma \neq 0$ and the branch trains normally.

**Note on the embedding `Linear`.** Using `fan_out`/`relu` Kaiming on the final
projection (which is not followed by a ReLU) slightly over-scales it, but this
is immaterial because the subsequent L2 normalization is scale-invariant.
**[REASONING]**

---

## 9. End-to-end forward data flow (worked shape trace)

Concrete example: `depth_exponent = 4`, `width_multiplier = 2.0`,
`stem_width = 16`, `T = 1024`. The allocation is $S = 5$ stages with widths
$[16, 32, 64, 128, 256]$ and depths $[1, 1, 3, 6, 5]$ (16 blocks total). The
tensor shape evolves as follows (each stage's first block halves the length via
`downsampling_rate = 2`; the stem reduces by `stem_stride = 4`): **[MEASURED /
CODE]**

| Point | Shape $(M, C, L)$ |
|---|---|
| Input | $(M, 1, 1024)$ |
| After Stem | $(M, 16, 256)$ |
| After Stage 1 ($w=16$, $d=1$) | $(M, 16, 128)$ |
| After Stage 2 ($w=32$, $d=1$) | $(M, 32, 64)$ |
| After Stage 3 ($w=64$, $d=3$) | $(M, 64, 32)$ |
| After Stage 4 ($w=128$, $d=6$) | $(M, 128, 16)$ |
| After Stage 5 ($w=256$, $d=5$) | $(M, 256, 8)$ |
| Head output | $(M, E)$ |

With the **baseline** head and `head_pool_ops = ("mean",)`, the head pools the
last stage $(M, 256, 8) \to (M, 256)$ and projects to $(M, E)$. With the
**fusion** head and `("mean","max","std")`, it pools every stage, giving a
concatenated width $3 \times (16 + 32 + 64 + 128 + 256) = 1488$ before the
`Linear`. **[CODE]**

---

## 10. Numerical and correctness properties

The following properties are asserted by the smoke test (§12) and hold by
construction; they are the behavioural contract of the module. **[MEASURED /
CODE]**

- **Shape correctness.** For every configuration in the search grid, both
  `(M, T)` and `(M, 1, T)` inputs produce `(M, E)`.
- **Residual alignment.** Every block's main and shortcut lengths match; a
  mismatch raises rather than silently broadcasting.
- **Determinism in `eval`.** With dropout off, repeated forwards on the same
  input are bit-identical.
- **Batch independence (train $\equiv$ eval).** A sample embedded alone equals
  the same sample embedded inside a batch (max abs difference measured at
  $5.96\times10^{-8}$, i.e. floating-point noise). This is the defining payoff
  of GroupNorm over BatchNorm and is what makes the module safe for the
  small/variable batch sizes of metric learning.
- **Unit-norm output.** With `l2_normalize = True`, each output row has
  $\lVert u_i \rVert_2 = 1$ within tolerance.
- **Residual-friendly init.** All last-branch $\gamma$ are exactly zero at
  initialization, and an interior (stride-1, $C_{\text{in}} = C_{\text{out}}$)
  block equals $\mathrm{ReLU}(x)$ at init, confirming the near-identity start.
- **Finite std-pooling.** Forward and backward are finite even when a pooled
  channel is exactly constant (the $\varepsilon$-inside-root guard, §7.4).

---

## 11. Deviations and simplifications vs. reference methods

Collected here for auditability; each is also flagged at its point of use.
Sources: [KB: RegNet], [KB: ResNeXt], and [REASONING] for project-specific
engineering choices. **This is the section the theory document will expand
with justification and literature.**

| # | Deviation / simplification | Relative to | Rationale (short) |
|---|---|---|---|
| 1 | 1-D convolutions | 2-D CV backbones | Domain: univariate time series, not images. [REASONING] |
| 2 | GroupNorm everywhere | RegNet/ResNeXt use BatchNorm | Batch independence: train $\equiv$ eval, robust to small/variable $M$. [REASONING] |
| 3 | Emergent stage count $S$ | RegNet fixes 4 stages | We keep whatever the quantization yields; $S$ grows with $d$. [KB: RegNet] |
| 4 | $w_0 = w_a = $ `stem_width` | RegNet leaves $w_0, w_a$ free | Paper reports $w_0{=}w_a$ "performs even better"; we impose it. [KB: RegNet] |
| 5 | `round` (banker's) quantization | matches `np.round` in `pycls` | Deterministic tie-break; consistent with reference impl. [CODE] |
| 6 | Width snapping to group multiples | `pycls` compatibility adjust | Keeps every search point buildable. [REASONING] |
| 7 | ResNeXt with $b = 1$ (no bottleneck) | original ResNeXt uses $b{>}1$ | Matches RegNet's "no bottleneck" finding. [KB: RegNet/ResNeXt] |
| 8 | Explicit asymmetric padding at forward time | static same-padding | Correct for any $T$; guarantees main/shortcut length match. [REASONING] |
| 9 | Multi-scale "pseudo-dense" fusion head | not in RegNet/ResNeXt | Project extension; multi-scale feature reuse. [REASONING; KB: DenseNet loosely] |
| 10 | L2-normalized embedding | task-specific | Cosine-distance metric learning geometry. [REASONING] |
| 11 | Zero-γ residual init | "bag of tricks" | Near-identity start for deep stacks. [REASONING; refs not in KB] |
| 12 | Removed `[:-2]` depth truncation | group's previous code | The old ad-hoc "depth − 2" is dropped; full $2^d$ blocks. [CODE] |
| 13 | `group_width` is a clean field (was hardcoded `groups=16`) | previous code | See §14 — corrects a search-space mislabelling. [CODE] |

---

## 12. Verification — the smoke test

`smoke_test_backbone.py` is a standalone harness (imports only `backbone`,
`torch`, and the standard library; requires no data). It embodies the project
directive to *always ship a smoke test with a new algorithm and to control each
script for correctness*. **[CODE]**

**What it checks (ten checks).** **[CODE]**

1. **Shapes + residual matching** across the search grid ($d \in \{3,4,5,6\}$,
   $w_m \in \{1.5, 2.0, 2.3, 3.0\}$, both block families, several $E$), for
   both input ranks.
2. (folded into 1) every block builds and its residual add aligns.
3. **Determinism** in `eval`.
4. **Unit-norm** output.
5. **GroupNorm batch-independence** (sample alone vs. in a batch).
6. **Residual-friendly init** (all last-branch $\gamma = 0$; interior block
   $\equiv \mathrm{ReLU}(x)$).
7. **Allocation report** and final-length report across the depth grid, with a
   sanity flag against the RegNet "$\sim$20-block" observation.
8. **Width-snapping invariant** (every stage width divisible by its effective
   group width) and that a ResNeXt model actually builds for each group width.
9. **Fusion width** equals $\lvert\mathcal{O}\rvert \sum_i C^{(i)}$.
10. **std-pooling finiteness** in forward and backward at init, including the
    exactly-constant-input path.

**How to run.** **[CODE]**

```bash
python3 smoke_test_backbone.py            # full sweep (~all grid configs)
python3 smoke_test_backbone.py --quick    # one config per check (fast)
```

Both exit `0` and print `ALL SMOKE TESTS PASSED` on success; any failure raises
(non-zero exit) with a diagnostic naming the offending configuration.
**[MEASURED]**

**Minimal in-REPL correctness snippet** (a fast independent check without the
full harness): **[CODE]**

```python
import torch
from backbone import BackboneConfig, build_backbone

cfg = BackboneConfig(depth_exponent=4, block_family=1,
                     head_fusion=True, head_pool_ops=("mean","max","std"))
m = build_backbone(cfg).eval()

x = torch.randn(8, 1024)
z = m(x)
assert z.shape == (8, cfg.embedding_size)
assert torch.allclose(z.norm(dim=1), torch.ones(8), atol=1e-5)   # unit norm
# batch independence:
assert torch.allclose(m(x)[3], m(x[3:4])[0], atol=1e-5)
print("ok:", m.stage_widths, m.stage_depths)
```

**"Controlled twice" (per the project directive).** The implementation was
verified two independent ways in this environment: (a) the full smoke test and
its `--quick` variant both pass (exit `0`); and (b) an out-of-harness probe on
the deepest configuration exercised parameter count, an AMP `bfloat16` forward,
gradient flow to the first stage, and a three-step optimizer loop (§13). The
allocation produced by the code was additionally cross-checked against a
by-hand application of RegNet Eqs. (2)–(4) and matched exactly for
$d \in \{3,4,5,6\}$. **[MEASURED]**

---

## 13. Empirical characteristics (measured)

All numbers below are **[MEASURED]** in this environment (PyTorch 2.12, CPU) and
are properties of this implementation, not literature values.

**Allocation across the depth grid** (at $T = 1024$, $w_m = 2.0$,
`stem_width = 16`, `group_width = 16`):

| $d$ | blocks $2^d$ | stages $S$ | stage widths $w^{(i)}$ | stage depths $d^{(i)}$ | final length $L$ |
|---|---|---|---|---|---|
| 3 | 8 | 4 | $[16, 32, 64, 128]$ | $[1, 1, 3, 3]$ | 16 |
| 4 | 16 | 5 | $[16, 32, 64, 128, 256]$ | $[1, 1, 3, 6, 5]$ | 8 |
| 5 | 32 | 6 | $[16, 32, 64, 128, 256, 512]$ | $[1, 1, 3, 6, 11, 10]$ | 4 |
| 6 | 64 | 7 | $[16, 32, 64, 128, 256, 512, 1024]$ | $[1, 1, 3, 6, 11, 23, 19]$ | 2 |

This makes the emergent-stage property concrete: $d = 3 \to 4$ stages,
$\dots$, $d = 6 \to 7$ stages.

**Deep-configuration probe** (`depth_exponent=6`, `width_multiplier=2.3`,
`block_family=1` (ResNeXt), `head_fusion=True`,
`head_pool_ops=("mean","max","std")`, `norm_target_cpg=8`, at $T = 1024$):

- Parameter count: $\approx 5.84\times10^{7}$ ($58{,}415{,}804$).
- Allocation: stages $[16, 32, 80, 192, 448, 1024]$, depths
  $[1, 2, 5, 10, 24, 22]$ — note the non-power-of-two widths ($80, 192, 448$)
  produced by $w_m = 2.3$ and group snapping.
- AMP `bfloat16` autocast forward: runs; returns a finite $(M, E)$ tensor.
- Gradient reaches the first stage's first convolution (finite), with L2 norm
  exactly $0$ **at step 0** — the expected zero-γ behaviour documented in §8,
  **not** a blocked-gradient defect.
- A three-step AdamW loop on a trivial group-separation objective moved the
  objective monotonically ($0.055 \to 0.096 \to 0.168$), confirming the network
  does train off the near-identity initialization.

---

## 14. Integration notes for the optimization stage (Topic 3)

These notes concern how `backbone.py` plugs into the search harness; they do not
affect the module itself. **[REASONING / CODE]**

**Corrected search-space labelling.** Inspection of the group's previous driver
(`1D_CNN_Optimization.py`, `1D_CNN_functions.py`) revealed that the historical
searched knobs were `d`, `wm`, `blk`, `ws`, `es`, where **`ws` was
`width_shrink`** — a *head* knob controlling the old width-shrink fully-connected
chain — and that the convolution/cardinality group divisor was **hardcoded**
(`groups=16`), i.e. *not* searched. The handoff's comment that labelled `ws` as
a "group width" knob was therefore a mislabelling. Two consequences for Topic 3:

- The new bare-`Linear` head eliminates the width-shrink chain entirely, so the
  `ws`/`width_shrink` dimension is **obsolete** and its slot is freed.
- The conv group divisor is now the clean `BackboneConfig.group_width` field
  (default $16$, channels-per-group semantics). It was *fixed* before; whether
  to **promote** it to a searched axis is a fresh choice.

**Proposed search space (for the theory/optimization documents to finalize).**
`depth_exponent` $\in \{3,4,5,6\}$; `width_multiplier` $\in [1.5, 3.0]$
(continuous `Real`); `block_family` $\in \{0,1\}$; `embedding_size` (integer
range); `norm_target_cpg` $\in \{4,8,16,24,32\}$; and optionally `group_width`.
The head knobs `head_fusion`, `head_pool_ops`, `head_prenorm` are held
fixed-per-run as ablation settings, not BO dimensions, initially.

**Device placement.** `forward` is device-agnostic; the harness is responsible
for `model.to(device)` and for the CPU-parallel / AMP policy.

---

## 15. Known limitations and open items

**[REASONING]**

- **Head fusion axes are not yet searched.** `head_fusion`, `head_pool_ops`,
  `head_prenorm` are exposed and tested but treated as fixed ablation settings;
  incorporating them into the search (as categorical dimensions) is deferred.
- **Very deep, short tails.** At $d = 6$, the final stages operate at length
  $L = 2$ (at $T = 1024$); `std` pooling over length $2$ is well defined and
  finite but statistically thin. Whether the deepest configurations are useful
  is an empirical question for the optimization stage, not a code defect
  (RegNet's own finding is that the best depths are $\sim$20 blocks
  [KB: RegNet], which the grid brackets).
- **Odd-kernel assumption for stride-1 "same".** Length preservation on stride-1
  convolutions assumes odd kernels (`stem_kernel=5`, `stage_kernel=3` satisfy
  this); an even stride-1 kernel would be off by one. The strided paths handle
  arbitrary kernels via `_pad_same_1d`.
- **Assertions vs. `-O`.** The residual-length checks use `RuntimeError` (always
  active), but any future `assert`-based checks would be stripped under
  `python -O`; the empirical shape checks in §12 are the primary guarantee.
- **GroupNorm reference not in KB.** The batch-independence rationale rests on
  GroupNorm; its original paper is not in the project knowledge base and is not
  cited here beyond a textbook-level statement. The theory document will source
  this properly (and consider PubMed only insofar as normalization choices have
  been studied in biomedical-signal contexts).

---

## 16. Sources and provenance

**Primary source.** `backbone.py` and `smoke_test_backbone.py` (the artifacts
being documented). All API, configuration, data-flow, and verification content
is `[CODE]`.

**Project knowledge base (full text read).**
- Radosavovic, Kosaraju, Girshick, He, Dollár, *Designing Network Design
  Spaces*, CVPR 2020 — the RegNet quantized-linear allocation (Eqs. 2–4), the
  $w_0 = w_a$ simplification, the $1.5 \le w_m \le 3$ range, the "$\sim$20-block"
  and "no bottleneck" findings. Cited as `[KB: RegNet]`.
- Xie, Girshick, Dollár, Tu, He, *Aggregated Residual Transformations for Deep
  Neural Networks* (ResNeXt), CVPR 2017 — grouped-convolution cardinality,
  type-B projection shortcuts, stride-2-in-first-block downsampling, post-add
  ReLU convention, and the (original) bottleneck design our $b{=}1$ block
  departs from. Cited as `[KB: ResNeXt]`.
- *Densely Connected Convolutional Networks* (DenseNet), in the project KB —
  loose inspiration for the multi-scale fusion head; cited as `[KB: DenseNet]`
  only as an analogy, no specific numeric claim drawn from it.

**Measured in this environment (PyTorch 2.12, CPU).** All numbers in §9, §10,
§12 (pass/exit), and §13 are `[MEASURED]` and are properties of this
implementation, not literature results.

**Engineering reasoning.** Design-rationale statements are labelled
`[REASONING]`.

**Not sourced here (flagged).** GroupNorm (Wu & He 2018), the zero-init-γ
"bag of tricks" (He et al. 2019 / Goyal et al. 2017), and the metric-learning /
cosine-embedding geometry are described at textbook level only; their full
texts were **not** retrieved and they are **not** in the project knowledge base.
Per the project rules, no numeric result from these is stated. PubMed was not
queried for this implementation report because the relevant references are
computer-vision / machine-learning papers that PubMed does not index; the
biomedical literature grounding (MEA recordings, instantaneous firing rate,
neuronal-dynamics embeddings) is deferred to the *Theoretical Documentation*,
where PubMed full-text retrieval will be used and any full-text-inaccessible
but important paper will be flagged for you to supply via library access.

*End of technical report.*
