# Tuning Reference II — The Knobs You Set

**Document 2 of 2.** Companion: *Tuning Reference I — The Eighteen Searched Axes*.
**Applies to:** `Main/hpc/Config/config_l3c_joint_search.json`,
`search_mode = "joint_conditions"`, branch `feat/composite-dsn-loss`.
**Date:** 3 August 2026 — **Revision 2**

> **Revision 2:** `sep_centre_means` is no longer a searched axis and the
> centred form of $\mathcal{L}_{\mathrm{sep}}$ has been removed from the code.
> Both `train.sep_centre_means` and `search.sep_centre_means_choices` are now
> **inert** and warn if set; the generator no longer emits either. The space is
> 17 axes / 21 columns.

## Abstract

Document I covers the seventeen dimensions the Gaussian process moves. This one
covers everything else: every knob you set by hand, what it means, what it is
set to, what happens if you change it, and — a category that deserves its own
section — the knobs that are present in the configuration file and **have no
effect at all** under this search mode. The scientific question is whether the
study you launch is the study you intended, and the failure mode this document
guards against is a setting that looks meaningful, is edited in good faith, and
silently does nothing.

**Covered:** search control; budget and schedule; the clamp constants; the
selection metric and its calibration; batch geometry and data; runtime; the
Stage 7 verification gate; and the inert knobs.

**Deliberately excluded:** the seventeen searched axes (Document I); the
derivations (`THEORY_joint_condition_search.md`); and the synthetic
data-generating process, which is inherited unchanged and is only summarised
where it constrains a knob.

**One item is flagged as a probable error to fix before launch:** §3.6,
`runtime.device`.

---

## 1. Notation and Symbols

| Symbol | Name / Meaning | Type & domain | Units | First used in § |
|---|---|---|---|---|
| $N_{\mathrm{calls}}$ | `n_calls_joint`, trial budget | $N_{\mathrm{calls}} = 300$ | trials | §3.1 |
| $N_{\mathrm{init}}$ | `n_initial_points_joint`, random design size | $N_{\mathrm{init}} = 100$ | trials | §3.1 |
| $N_{\mathrm{seeds}}$ | `n_seeds`, seeds per trial | $N_{\mathrm{seeds}} = 1$ | seeds | §3.2 |
| $E_{\max}$ | `max_epochs` | $E_{\max} = 100$ | epochs | §3.2 |
| $P$ | `patience` | $P = 40$ | epochs | §3.2 |
| $n_{\mathrm{b}}$ | `batches_per_epoch` | $n_{\mathrm{b}} = 100$ | steps/epoch | §3.2 |
| $T$ | planned steps, $T = E_{\max} n_{\mathrm{b}}$ | $T = 10\,000$ | steps | §3.2 |
| $\tau$ | `sep_warmup_frac` | $\tau = 0.3$ | dimensionless | §3.2 |
| $t$ | optimiser-step index, steps **completed** | $t \in \{0,\dots,T\}$ | steps | §3.2 |
| $\lambda_{\mathrm{sep}}$ | asymptotic separation weight (searched; base value is the clamp) | base $= 0.1$ | dimensionless | §3.3 |
| $m_{\cos}$ | margin (searched under `triplet`; base value is the clamp) | base $= 0.3$ | cosine distance | §3.3 |
| $u$ | primary selection metric, by role | $u \in [-1,1]$ | dimensionless | §3.4 |
| $S_{\mathrm{val}}$ | validation cosine silhouette | $S_{\mathrm{val}} \in [-1,1]$ | dimensionless | §3.4 |
| $S_{0}$ | label-shuffled silhouette floor | $S_0 \in [-1,1]$ | dimensionless | §3.4 |
| $\sigma_{0}$ | standard deviation of the shuffled-label null | $\sigma_0 \ge 0$ | dimensionless | §3.4 |
| $\kappa_{\delta}$ | `min_delta_sil_kappa` | $\kappa_{\delta} = 2.0$ | dimensionless | §3.4 |
| $\delta_{\min}$ | derived early-stopping improvement threshold | $\delta_{\min} \ge 0$ | dimensionless | §3.4 |
| $R$ | `sil_floor_permutations` | $R = 200$ | permutations | §3.4 |
| $\gamma_{\mathrm{tb}}$ | `tie_break_gamma` | $\gamma_{\mathrm{tb}} = 0$ | dimensionless | §3.4 |
| $C$ | condition classes | $C = 3$ | dimensionless | §3.5 |
| $U_c$ | `cultures_per_class_per_batch` | $U_c = 9$ | cultures | §3.5 |
| $q$ | `windows_per_culture_per_batch` | $q = 1$ | windows | §3.5 |
| $M$ | batch rows, $M = C\,U_c\,q$ | $M = 27$ | rows | §3.5 |
| $\varrho$ | `class_overlap` | $\varrho = 0.05$ | dimensionless | §3.5 |
| $W$ | requested walltime | $W$ hours | hours | §3.7 |
| $\beta$ | gate margin | $\beta = 0.15$ | dimensionless | §3.7 |
| $R^2_{\min}$ | minimum admissible cost-model fit | $R^2_{\min} = 0.70$ | dimensionless | §3.7 |
| $\bar{e}$ | mean epochs actually run per trial | $\bar{e} \approx 0.55 E_{\max}$ | epochs | §3.7 |
| $\widehat{H}$ | extrapolated wall clock | hours | hours | §3.7 |
| $\eta$ | size-mix multiplier | $\eta > 0$ | dimensionless | §3.7 |
| $F$ | failure penalty | $F = +1.0$ | dimensionless | §3.4 |

### 1.1 Conventions

- Values quoted are those in the shipped configuration. Where they differ from
  the dataclass default, both are stated.
- "Inert" means: present in the configuration file, parsed without error, and
  **not read** by any code path under `search_mode = "joint_conditions"`.
- "Clamp constant" means: a base-configuration value that becomes the fixed
  value of a searched axis on the trials where that axis is inactive.
- Steps are counted by $t$ = steps **completed**, so the first batch of a run
  is evaluated at $t = 0$.

---

## 2. Glossary

Ordered by first appearance.

**Random initial design.** The trials drawn quasi-randomly before the surrogate
is first fitted. Size $N_{\mathrm{init}}$.

**Acquisition function.** The rule the optimiser uses, once a surrogate exists,
to choose the next point — trading expected improvement against uncertainty.

**Clamp constant.** See §1.1. The base configuration supplies these, which is
why editing a base value that "isn't searched anyway" can change every trial.

**Warm-up ($\tau$).** The fraction of the planned step budget over which the
separation weight ramps linearly from zero to its asymptotic value.

**Planned horizon ($T$).** $E_{\max} n_{\mathrm{b}}$. Not the number of steps a
run takes, because of early stopping.

**Early stopping.** Halting when the selection metric has failed to improve by
at least $\delta_{\min}$ for $P$ consecutive epochs.

**Label-shuffled null / silhouette floor.** The distribution of the silhouette
obtained by randomly permuting the labels. It is the value a *structureless*
embedding attains, and is not zero in a finite sample.

**Floor-scaled improvement threshold.** Deriving $\delta_{\min}$ from the
shuffled-label null rather than fixing it by hand.

**Tie-break.** A secondary metric used to order configurations the primary
metric cannot separate.

**Culture.** One simulated recording; the unit of the whole-culture split, so
that no culture contributes windows to more than one split.

**Cross-culture positives.** Requiring anchor and positive to come from
*different* cultures, so the loss cannot succeed by recognising a recording.

**Size-mix multiplier ($\eta$).** Expected per-epoch cost over the sampled
architecture distribution, divided by the cost at the reference architecture.

**Submission gate.** The predicate that must hold before the job may be
submitted.

---

## 3. The knobs

### 3.1 Search control

| Knob | Value | Default | Effect |
|---|---|---|---|
| `search.search_mode` | `joint_conditions` | `staged` | selects the whole design |
| `search.n_calls_joint` | 300 | 0 | trial budget |
| `search.n_initial_points_joint` | 100 | 0 | random design size |
| `search.gp_random_state` | 0 | 0 | seeds the trial sequence |

**`search_mode`.** One knob, three legal values: `staged`, `joint`,
`joint_conditions`. It is a single enumerated field rather than `joint` plus a
separate boolean flag, deliberately, so that the meaningless combination
(staged search + searched conditions) **is not representable at all**. Changing
it to `staged` reverts to the phase pipeline and makes most of this document
inapplicable.

**`n_calls_joint` — the budget.** 300 trials × 1 seed = 300 training runs. The
value 0 means "match the staged total", which is the setting under which a
staged-versus-joint comparison would be about strategy rather than compute;
that is not what this study is doing, so an explicit 300 is set.

**`n_initial_points_joint` — the random design.** This is the knob with the
best-quantified justification in the configuration, and it is worth stating the
arithmetic because it is not obvious.

The categorical factors partition the space into 52 cells. Under the sampling
measure and the legality projection, 20 cells carry probability $1/36$ and 32
carry $1/72$ (the projection *doubles* the mass of the conditions it merges).
Expected unvisited cells after $N$ draws is therefore

$$
\mathbb{E}[U_N] = 20\left(1-\tfrac{1}{36}\right)^{N} + 32\left(1-\tfrac{1}{72}\right)^{N},
\tag{1}
$$

validated against the sampler over 30 seeds. Consequences:

| $N_{\mathrm{init}}$ | $\mathbb{E}[U_N]$ | cells the surrogate has never seen |
|---|---|---|
| 40 | 24.8 | roughly half the partition |
| **100** | **9.1** | 8–9 cells |
| 250 | 1.0 | expected full coverage |

Complete coverage by random sampling alone would cost 250 of the 300 trials, so
it was never affordable; the question is only how much partition the surrogate
extrapolates over. At 100, any *specific* thin cell is still unseen with
probability $(1-1/72)^{100} = 0.247$.

**The trade-off, stated honestly.** $N_{\mathrm{init}} = 100$ is a third of the
budget spent before the surrogate exists — pure random search. The defence is
that a Gaussian process fitted to 40 noisy observations in 22 columns proposes
near-arbitrary points anyway, and that single-seed noise ($\sigma_s = 0.073$)
makes early surrogate fits unreliable. The counter-argument is that 200
adaptive trials in 22 columns is itself thin. No experiment settles this;
it is a judgement, and it is reversible.

**`gp_random_state`.** Seeds both the random design and the optimiser's
internal randomness. Two studies differing only in this value are independent
replicates. Changing it invalidates any comparison with a previous run at fixed
axis order.

### 3.2 Budget and schedule — the knobs that decide whether the study fits

| Knob | Value | Default | Effect |
|---|---|---|---|
| `train.n_seeds` | 1 | 3 | seeds per trial |
| `train.max_epochs` | 100 | — | epoch cap, $E_{\max}$ |
| `train.patience` | 40 | — | early-stopping patience, $P$ |
| `train.batches_per_epoch` | 100 | 0 | steps per epoch, $n_{\mathrm{b}}$ |
| `train.sep_warmup_frac` | 0.3 | 0.0 | warm-up fraction, $\tau$ |

**`n_seeds = 1` — what one seed buys and costs.** With across-seed standard
deviation $\sigma_s$, the probability that a single seed each misranks two
configurations whose true difference is $d$ is

$$
\mathbb{P}[\text{wrong order}] = \Phi\!\left(-\frac{d}{\sigma_s\sqrt{2}}\right).
\tag{2}
$$

At the measured $\sigma_s = 0.073$ (between-cell $\sigma_b = 0.117$, so the
ratio is 0.62):

| $d$ | 0.02 | 0.05 | 0.10 | 0.15 | 0.20 |
|---|---|---|---|---|---|
| misrank probability | 42% | 31% | 17% | 7.2% | 2.6% |

**Coarse structure is recoverable; fine ranking is not.** The reported
argument-minimum is a top-$k$ candidate, not a winner, and the confirmatory
re-fit of the leading configurations at several seeds each is part of the
design rather than optional.

A technical consequence worth knowing: at $N_{\mathrm{seeds}} = 1$ the
per-trial spread is computed with the **population** estimator (divisor $n$),
giving exactly $0$. The sample estimator (divisor $n-1$) would give NaN, and a
NaN cannot be fitted by the surrogate — it would abort the study after hours of
completed training. This was verified rather than assumed.

**`max_epochs` and `patience` — the unresolved decision.** These are the design
document's **stated** values. It *recommends* 60 and 20 but explicitly records
that recommendation as not yet accepted, so the generator defaults to the stated
values and prints the arithmetic instead of choosing for you. At $E_{\max}=100$,
$n_{\mathrm{b}}=100$, the estimate is **157 h against a 144 h walltime** — it
does not fit at the reference rate, before the size mix is even counted.

If the gate fails, regenerate:

```bash
python3 hpc/make_joint_search_config.py \
    --base hpc/Config/config_l3c_h_multimean_005.json \
    --out  hpc/Config/config_l3c_joint_search.json \
    --max-epochs 60 --patience 20
```

At 60 epochs the planned budget is still $T = 6\,000$ steps, four times the
screening's maximum of 1 500 — so the step-budget increase that motivated the
whole design survives the reduction.

**`batches_per_epoch = 100` — the most important single change from the
screening.** The screening ran 25 batches per epoch; this is 6.7×. Since
$T = E_{\max} n_{\mathrm{b}}$, it multiplies the optimiser's step budget
directly and it multiplies the cost directly. It is the knob that makes the
neural-collapse-inspired term even arguably applicable, since the phenomenon it
derives from is documented after hundreds of epochs of training.

**`sep_warmup_frac = 0.3` ($\tau$) — fixed, never searched.** The separation
weight ramps as
$\lambda_{\mathrm{sep}}(t) = \lambda_{\mathrm{sep}}\min(1, t/(\tau T))$, so the
total dose integrates to $\lambda_{\mathrm{sep}} T (1-\tau/2)$ — $\tau$ and
$\lambda_{\mathrm{sep}}$ trade off almost multiplicatively, and searching both
would move along a ridge. Hence one is fixed.

Three practical consequences:

1. At $\tau = 0.3$, $T = 10\,000$: full weight is reached at step 3 000, i.e.
   epoch 30 of 100.
2. **A run reaches full weight only if $\tilde{e}/E_{\max} \ge \tau$**, where
   $\tilde{e}$ is the epochs it actually ran. Since patience bounds how early it
   can stop, the preflight warns when $P/E_{\max} < \tau$. Here
   $40/100 = 0.4 > 0.3$, so it passes — but at the recommended 60/20,
   $20/60 = 0.33$, only just above $\tau = 0.3$. **If you reduce to 60/20, that
   margin becomes thin**, and runs stopping at their patience floor will barely
   reach full weight.
3. The logged value lags the live value by one step: history at the end of
   epoch $k$ records $g(k n_{\mathrm{b}} - 1)$, so full weight first appears at
   epoch $\lceil (\tau T + 1)/n_{\mathrm{b}}\rceil$, not
   $\lceil \tau E_{\max}\rceil$. Read `sep_step` alongside
   `sep_warmup_scale`.

$\tau = 0$ is a legitimate control arm: full weight from step 0, reproducing the
pre-existing ungated behaviour exactly.

### 3.3 The clamp constants — base values that are *not* dead

This section exists because the natural assumption — "these are searched, so
the base value doesn't matter" — is **wrong**.

Four axes are inactive on some trials. On those trials the built configuration
keeps the **base configuration's** value. The base configuration therefore
supplies the clamp constants, and it is what makes two points differing only in
inactive coordinates build byte-identical configurations.

| Knob | Base value | Inactive when | Consequence |
|---|---|---|---|
| `train.lambda_sep` | **0.1** | $\ell \ne \texttt{joint\_sep}$ | must be exactly the default |
| `train.margin` | 0.3 | $\ell \ne \texttt{triplet}$ | the *fixed* margin of the composite losses |
| ~~`train.sep_centre_means`~~ | — | — | **removed as an axis; now inert, warns if set** |
| `train.strict_semihard` | `False` | $\ell = \texttt{triplet}$ | forced by $\Pi$ anyway |

**`lambda_sep = 0.1` is not a value that will ever be used** — it is a clamp,
and it must equal the `TrainConfig` default exactly. Any other value causes
every `triplet` and `joint` trial to emit an "INERT lambda_sep" warning, and a
warning that fires on two-thirds of trials trains the reader to ignore
warnings. The preflight checks this.

**`margin = 0.3` is genuinely used** under `joint` and `joint_sep`: the
composite loss reads it for its hinge, at $2m_{\cos}$ in squared-Euclidean
units. So the base margin is a real hyper-parameter of two of the three loss
types, held fixed while $\alpha$ is searched. Changing it changes those trials.

These fields are also written by the search on the trials where they *are*
active — `train.lr`, `train.beta1`, `train.beta2`, `train.weight_decay` are
always overwritten, so their base values (3e-4, 0.9, 0.999, 1e-4) are inert in
every trial and exist only so the configuration is valid standalone.

### 3.4 Selection, early stopping, and the objective

| Knob | Value | Effect |
|---|---|---|
| `train.selection_primary` | `silhouette` | which metric selects the epoch and scores the trial |
| `train.min_delta_sil_mode` | `floor_scale` | derive the improvement threshold from the null |
| `train.min_delta_sil_kappa` | 2.0 | $\kappa_\delta$, multiples of the null's sd |
| `train.sil_floor_permutations` | 200 | $R$, permutations estimating the null |
| `train.min_delta_sil` | 0.0 | overridden by `floor_scale` |
| `train.min_delta_ari` | 0.0 | inert (ARI is not primary) |
| `search.tie_break_gamma` | 0.0 | tie-break disabled |

**`selection_primary = silhouette`.** The cosine silhouette on held-out data
selects the epoch within a run and scores the trial. It is continuous and
label-free given the labels, and it measures the geometry the loss is actually
shaping — unlike ARI, which measures a clustering *after* an algorithm has been
applied.

**Choosing it has a consequence you should not miss.** The tie-break machinery
requires a *discrete* primary metric: its safety guarantee is that the secondary
metric can only reorder configurations inside an exact primary tie, which
presupposes that a smallest positive gap exists. ARI on a fixed label vector has
one; a continuous silhouette does not — exact ties have probability zero. So the
tie-break is **inapplicable in principle** here, not merely switched off.
`tie_break_gamma = 0.0` states that fact quietly instead of emitting a warning
on every phase. `tie_break_sil_lo` / `tie_break_sil_hi` are consequently inert.

**`min_delta_sil_mode = floor_scale`.** Early stopping needs a threshold below
which an improvement is not an improvement. Fixing it by hand assumes you know
the noise scale. Instead, the trainer permutes the labels $R = 200$ times,
estimates the null distribution $(S_0, \sigma_0)$ of the silhouette on
*structureless* data, and derives

$$
\delta_{\min} = \kappa_{\delta}\,\sigma_0 .
\tag{3}
$$

With $\kappa_\delta = 2$, an epoch must beat the previous best by two standard
deviations of the null before it counts. This makes patience meaningful: without
it, $P = 40$ epochs of noise-driven "improvements" would prevent stopping
entirely.

Raising $\kappa_\delta$ stops runs earlier and more aggressively; lowering it
towards 0 recovers the naive behaviour. $R = 200$ costs 200 silhouette
evaluations per calibration, which is cheap relative to an epoch.

**The failure policy.** A trial that cannot be built, or that crashes, scores a
large **finite** penalty $F = +1.0$ — strictly worse than any achievable value
of the negated metric, which is bounded by 1 in magnitude. It is never NaN,
because the surrogate cannot be fitted to NaN. A trial in which *some* seeds
complete is scored as failed rather than averaged over survivors: a
configuration that crashed on two of three seeds and was lucky on the third
would otherwise report a high mean with zero spread, which is exactly the
signature the acquisition function finds most attractive.

### 3.4.1 Three smaller knobs, for completeness

| Knob | Value | Effect |
|---|---|---|
| `train.swap` | `True` | anchor-positive **swap** in the triplet hinge |
| `train.log_every_epochs` | 1 | how often a history record is emitted |
| `runtime.experiment_name` | `l3c_joint_search` | names the output directory |

**`swap = True` is a real semantic knob, not bookkeeping.** With swap enabled
the hinge uses the *harder* of the two anchor-negative and positive-negative
distances, i.e. it substitutes $\min(D_{an}, D_{pn})$ for $D_{an}$. This makes
every triple at least as hard as it was, so the loss is an upper bound on the
un-swapped one. It interacts with mining: under `hard` mining the triples are
already the most violating ones, so swap changes little; under `easy_positive`
it partially counteracts the deliberate leniency of the miner. It is held fixed
across the study, so it does not confound the comparison, but a winner should be
reported as "with swap".

**`log_every_epochs = 1`** means a history record per epoch, which is what makes
the warm-up auditable (`sep_lambda_t`, `sep_warmup_scale`, `sep_step`) and what
the census reads. Raising it would coarsen the ramp record; there is no reason
to, as the cost is negligible.

**`experiment_name`** names the output directory under `runtime.out_dir`.
Change it for a replicate run, or the second study will write beside the first.

### 3.5 Batch geometry and data

| Knob | Value | Effect |
|---|---|---|
| `data.cultures_per_class_per_batch` | 9 | $U_c$, cultures per class per batch |
| `data.windows_per_culture_per_batch` | 1 | $q$, windows per culture |
| `data.positives_mode` | `cross_culture` | anchor and positive from different cultures |
| `data.exclude_same_culture_positives` | `True` | enforces the above |
| `data.max_group_size` | 16 | cap on same-class rows per batch |
| `data.split_mode` | `trace` | whole-culture split |
| `data.split_fractions` | [0.6, 0.2, 0.2] | train / val / test |
| `data.latent.class_overlap` | 0.05 | generator difficulty |
| `data.latent.label_axes` | [0, 1, 2] | three axes carry the labels |
| `data.latent.class_center_mode` | `simplex` | non-collinear class centres |
| `train.windows_per_condition` | 4 | **inert here** — see §3.8 |

**Batch composition.** $M = C\,U_c\,q = 3 \times 9 \times 1 = 27$ rows, nine per
class. This is the number that makes every per-batch statistic noisy: the
separation term estimates three class means from nine rows each, and the mean of
nine unit vectors is a high-variance estimate of a direction. It is also why the
gate that this design *removed* was prone to latching on a fluctuation.

**Cross-culture positives.** Requiring the positive to come from a different
culture than the anchor prevents the embedding from succeeding by recognising a
recording rather than a condition. Combined with `split_mode = trace`
(whole-culture splits, so no culture spans two splits) this is what makes the
held-out silhouette a generalisation measure rather than a memorisation one.

**`class_overlap = 0.05`.** The generator's difficulty knob. **It has never
been calibrated against the class-centre construction**, so every absolute
silhouette in this study sits on an uncalibrated difficulty scale. Comparisons
*within* the study are unaffected; statements of the form "this configuration
achieves silhouette 0.4" mean little in absolute terms. Inherited unchanged.

**`label_axes = [0,1,2]` with `class_center_mode = simplex`.** Three of the six
latent axes carry class information, and the class centres are placed
non-collinearly. This matters for the separation term: a simplex arrangement in
the *generative* latent space is what makes a simplex ETF in the *embedding*
space a coherent target at all.

### 3.6 Runtime — **check this before launching**

| Knob | Value | Comment |
|---|---|---|
| `runtime.device` | **`cpu`** | **almost certainly wrong for this study** |
| `runtime.torch_threads` | 48 | CPU threads |
| `runtime.num_workers` | 0 | dataloader workers |
| `runtime.pin_memory` | `False` | consistent with CPU |
| `train.use_amp` | `False` | no mixed precision |
| `runtime.deterministic` | `True` | reproducibility |
| `runtime.seed` | 0 | base seed for splits and init |
| `train.checkpoint_every_epochs` | 5 | checkpoint cadence |
| `runtime.cache_dir`, `out_dir` | cluster paths | verify they exist |

**`device = cpu` is the one item in this configuration I would treat as an
error to fix before launch.** It is inherited from the base screening config.
The size mix reaches 31.6 M parameters, the budget is 300 runs of up to 100
epochs at 100 batches, and the wall-clock estimate that the design document
computed was not derived on the assumption of CPU-only execution. Confirm what
the cluster allocation actually provides and set it accordingly; if the
allocation *is* CPU-only, the walltime arithmetic needs redoing from scratch
rather than adjusting.

**`use_amp = False`.** Mixed precision would cut memory and time substantially
on a GPU. It is off, which is the conservative choice for numerical
reproducibility, and it interacts with `deterministic = True`. If the wall-clock
gate fails narrowly, enabling AMP is a lever — but it changes numerics, so
enable it *before* the timing measurement, not after.

**`deterministic = True`** costs some speed and pins algorithm choices. Keep it:
with one seed per trial, reproducibility of individual runs is the only defence
you have against an ambiguous result.

**`checkpoint_every_epochs = 5`.** With a sequential optimiser and a hard
walltime, this bounds what a wall-kill loses. Note that the warm-up step counter
lives in a registered buffer, so a checkpoint carrying the loss module's state
resumes the ramp correctly; one that does not resets it to zero and re-runs the
warm-up.

### 3.7 The Stage 7 gate — the knobs that decide whether you may submit

These are command-line arguments to `search_dry_run.py`, not configuration
fields.

| Knob | Default | Effect |
|---|---|---|
| `--n` | 300 | points sampled for coverage; use the real budget |
| `--time-points` | 0 | how many points to time end-to-end |
| `--time-epochs` | 2 | epochs per timed point |
| `--walltime` | — | $W$, hours requested |
| `--margin` | 0.15 | $\beta$, required unused fraction |
| `--no-params` | off | skip parameter counting (skips the size mix) |

The gate authorises submission only if **all** of the following hold:

$$
\underbrace{U_N = 0}_{\text{coverage}} \ \wedge\
\underbrace{n_{\mathrm{fail}} = 0}_{\text{every point builds}} \ \wedge\
\underbrace{R^2 \ge 0.70}_{\text{the cost model fits}} \ \wedge\
\underbrace{\widehat{H} \le (1-\beta)W}_{\text{it fits}}
\tag{4}
$$

with

$$
\widehat{H} = \frac{N_{\mathrm{calls}} N_{\mathrm{seeds}}\, \bar{e}\; \mathbb{E}[r(\Theta)]}{3600},
\qquad
\eta = \frac{\mathbb{E}[r(\Theta)]}{r(\theta_{\mathrm{ref}})} .
\tag{5}
$$

**`--time-points`.** With 0, the gate **fails by construction**: an unmeasured
wall clock is not a passing state, because the design document's figure is the
reference-architecture rate and excludes the size mix entirely. Use 10 or more.

**`--time-epochs = 2`.** The measurement wanted is per-epoch cost, which two
epochs estimate as well as sixty at a thirtieth of the price. The first epoch is
discarded where more than one is run, because it carries one-off costs.

**`--margin = 0.15`.** Not decoration. The size mix is an estimate, $\bar{e}$
is a historical average from a *different* configuration, and the optimiser is
**sequential** — it cannot be split across lanes if it overruns, and a job
killed at the wall loses everything since the last checkpoint. A study that
fits with zero margin does not fit.

**$R^2_{\min} = 0.70$ (in code, not a flag).** The linear-in-parameters cost
model must actually describe the timed points. On small models the per-epoch
cost is dominated by fixed overhead and the parameter term explains almost
nothing — an $R^2$ of 0.29 was observed to produce a confident-looking and
meaningless hour count. If this fires, time more points spanning a **wider range
of depths**, or model cost on something other than parameter count. An
unreliable estimate used as a gate is worse than no estimate, because it carries
authority it has not earned.

**How to run it:**

```bash
python3 search_dry_run.py --config hpc/Config/config_l3c_joint_search.json \
    --n 300 --time-points 10 --walltime 144 --out dry_run_report.json
echo $?    # 0 = submit, 1 = do not
```

### 3.8 Inert knobs — present in the file, read by nothing

Editing any of these under `search_mode = "joint_conditions"` will change
nothing, silently. They are retained so that archived configurations still parse
and so that switching back to `staged` works.

| Knob | Value | Why inert |
|---|---|---|
| `search.n_calls_arch` | 60 | staged phase 1 only |
| `search.n_calls_train` | 60 | staged phase 2 only |
| `search.n_initial_points` | 20 | staged phases; the joint search uses `n_initial_points_joint` |
| `regularization.n_calls` | 20 | staged regularisation phase only |
| `search.do_refine`, `refine_top_fraction` | False, 0.1 | staged refinement only |
| `search.do_retune_arch` | False | staged retune only |
| **`search.weight_decay_range`** | [1e-4, 1e-2] | **the joint space reads `regularization.weight_decay_range`** |
| `search.tie_break_sil_lo/hi` | −1.0, 1.0 | tie-break disabled by $\gamma_{\mathrm{tb}}=0$ |
| `train.min_delta_ari` | 0.0 | ARI is not the primary metric |
| `train.windows_per_condition` | 4 | only used to *derive* $n_{\mathrm{b}}$ when `batches_per_epoch = 0`; it is 100 |
| `train.scheduler_type` | `cosine` | `use_scheduler = False` |
| `train.mining_strategy`, `loss_type` | `hard`, `triplet` | overwritten every trial by the search |
| `train.lr`, `beta1`, `beta2`, `weight_decay` | — | overwritten every trial |
| `train.sep_gate_threshold` / `_momentum` / `_min_batches` | — | **the gate was removed**; a non-`None` threshold warns |
| `train.sep_centre_means` | — | **the centred form was removed** (scale-invariant, blind to collapse); warns if set |
| `search.sep_centre_means_choices` | — | no longer a searched axis |

The two most likely to catch you are **`search.weight_decay_range`** (edit
`regularization.weight_decay_range` instead) and **`train.windows_per_condition`**
(inert because `batches_per_epoch` is explicit).

---

## 4. Summary of results

**K1.** `search_mode = "joint_conditions"` is one enumerated knob, chosen so the
combination staged + searched-conditions is unrepresentable. §3.1.

**K2.** $N_{\mathrm{init}} = 100$ leaves $\mathbb{E}[U_N] = 9.1$ cells
unobserved against 24.8 at 40; full random coverage would cost 250 of 300
trials. Eq. (1), §3.1.

**K3.** One seed resolves differences of 0.2 at 2.6% error and 0.05 at 31%.
The output is a top-$k$ candidate set. Eq. (2), §3.2.

**K4.** At one seed the spread uses the population estimator, giving 0 rather
than NaN; NaN would abort the study. §3.2.

**K5.** $E_{\max}=100$, $P=40$ extrapolate to 157 h against 144 h **before** the
size mix is counted. The 60/20 alternative is one regeneration away and keeps
$T = 6\,000$, still 4× the screening. §3.2.

**K6.** $\tau = 0.3$ is fixed because $\tau$ and $\lambda_{\mathrm{sep}}$ trade
off through the dose $\lambda_{\mathrm{sep}} T(1-\tau/2)$. A run reaches full
weight only if $\tilde e / E_{\max} \ge \tau$; at 60/20 that margin
($0.33$ vs $0.30$) becomes thin. §3.2.

**K7.** Base values of inactive axes are **clamp constants, not dead settings**.
`lambda_sep` must be exactly 0.1; `margin = 0.3` is genuinely used by both
composite losses. §3.3.

**K8.** The tie-break is inapplicable in principle under a continuous primary
metric, not merely disabled. §3.4.

**K9.** $\delta_{\min} = \kappa_\delta \sigma_0$ is derived from a
200-permutation label-shuffled null, which is what makes $P = 40$ meaningful.
Eq. (3), §3.4.

**K10.** Batches are 27 rows, nine per class — the source of per-batch
statistical noise. §3.5.

**K11.** `runtime.device = cpu` should be checked and almost certainly changed
before launch. §3.6.

**K12.** The gate fails by construction without `--time-points`, and fails on
$R^2 < 0.70$ however precise the resulting hour count looks. Eq. (4), §3.7.

**K13.** Thirteen knobs in the file are inert; `search.weight_decay_range` and
`train.windows_per_condition` are the two most likely to mislead. §3.8.

---

## 5. Open points, caveats and assumptions

1. **`device = cpu` is flagged as a probable error, not diagnosed.** I cannot
   see the allocation. If it is genuinely CPU-only, the entire wall-clock
   analysis needs redoing rather than adjusting.

2. **$E_{\max}$ and $P$ are unresolved.** The stated values do not fit; the
   recommended ones were explicitly not accepted. The gate is the arbiter, and
   it has not been run on the cluster.

3. **$\tau = 0.3$ is asserted, not tuned.** The dose integral justifies fixing
   *some* value, not this one. A one-dimensional sweep at the winning point is
   the cheap check.

4. **$N_{\mathrm{init}} = 100$ is a judgement.** The coverage arithmetic
   quantifies what it buys but not what the 60 forgone adaptive trials would
   have bought. No experiment settles it.

5. **$\sigma_s = 0.073$ comes from a different configuration** — the screening
   tier at 1 500 steps. Whether it transfers to 6 000–10 000 steps is an
   assumption, and the misranking table inherits that uncertainty.

6. **Eq. (2) assumes normality**, untested. The measured maximum within-cell
   spread was three times the median, which is not what a well-behaved normal
   sample looks like.

7. **$\kappa_\delta = 2.0$ and $R = 200$ are conventional choices,** not derived
   from a power calculation.

8. **`class_overlap = 0.05` is uncalibrated,** so absolute silhouette values
   have no external meaning. Within-study comparisons are unaffected.

9. **$R^2_{\min} = 0.70$ is a judgement** chosen to exclude an observed failure
   with margin; no analysis relates it to the resulting error in $\widehat{H}$.

10. **$\bar{e} \approx 0.55 E_{\max}$ is a historical average** from the
    screening. If the joint search's configurations stop earlier or later on
    average, Eq. (5) is biased by that ratio.

11. **The inert list is derived from reading the current source.** If the search
    mode's dispatch changes, an inert knob may silently become live; the list is
    a snapshot, not an invariant.

---

## 6. References

- **TUNING_1_searched_axes.md** — the seventeen axes the optimiser moves.
- **THEORY_joint_condition_search.md** — derivations for the coverage
  arithmetic (Eq. 1), the warm-up dose, the misranking probability (Eq. 2), the
  tie-break guarantee, the failure policy, and the cost model (Eq. 5).
- **`hpc/make_joint_search_config.py`** — the generator, which prints the budget
  and wall-clock arithmetic and is the source of truth for this configuration.
  Regenerate rather than hand-editing.
- **`hpc/preflight_config.py`** — checks the clamp constants, the warm-up
  horizon, the $P/E_{\max} < \tau$ condition, and reports the searched factors.
- **`search_dry_run.py`** — the gate of §3.7.
- **Repository source**, read directly: `config.py` (fields, defaults,
  validation), `train.py` (`derive_batches_per_epoch`, the floor calibration,
  the early-stopping rule), `batch_geometry.py` (batch composition),
  `search.py` (which ranges the joint space actually reads).
- **Measured in this work:** the coverage table of §3.1; the batch-size
  arithmetic of §3.5; the $R^2 = 0.29$ cost-model failure of §3.7; the inert
  list of §3.8.
- **From the predecessor handoffs, not re-measured:** $\sigma_s = 0.073$,
  $\sigma_b = 0.117$, and $\bar{e} \approx 0.55 E_{\max}$ (§3.2, §3.7); the
  157 h estimate (§3.2).
