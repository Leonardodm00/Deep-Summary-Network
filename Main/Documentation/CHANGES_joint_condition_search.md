# CHANGES: the joint condition search replaces the 52-cell factorial

**Branch:** `feat/composite-dsn-loss`
**Date:** 3 August 2026
**Design document:** `HANDOFF_joint_gp_search.md` (stages 1-8)
**Status:** stages 1-7 implemented and tested; **stage 7's gate has not been
run on the cluster**, and stage 8 (launch) is downstream of it.

## Abstract

The 52-cell factorial fixed four categorical factors per cell and ran a
separate Bayesian search inside each. This change collapses it into **one**
joint Bayesian search in which those factors are themselves searched
dimensions: an 18-axis space over 22 surrogate columns, with illegal
combinations removed by a deterministic legality projection rather than
enumerated away. It also replaces the latching silhouette gate on the
separation term with a deterministic warm-up.

**Covered here:** what each stage added, the decisions taken where the design
document was silent, the three places this implementation knowingly departs
from it, and what remains open.

**Not covered:** the screening findings (predecessor handoff), and any claim
about which objective wins -- that is what the search is for.

## 1. Notation and symbols

| Symbol | Meaning | Type / domain | First used |
|---|---|---|---|
| $\ell$ | the `loss_type` categorical value | $\ell \in \{\texttt{triplet}, \texttt{joint}, \texttt{joint\_sep}\}$ | S2 |
| $A(\ell)$ | activity mask: which loss HPs $\ell$ reads | subset of the 4-element superset | S3 |
| $\Pi$ | legality projection on the condition | idempotent map, 18 raw triples $\to$ 13 legal | S2 |
| $t$ | global optimiser-step index, persisting across epochs | $t \in \{0, 1, \dots, T\}$ | S5 |
| $T$ | planned step budget, $T = E_{\max} \cdot n_{\mathrm{batches}}$ | $T \in \mathbb{N}$ | S5 |
| $E_{\max}$ | `max_epochs` | $E_{\max} \in \mathbb{N}$ | S5 |
| $n_{\mathrm{batches}}$ | `batches_per_epoch` | $n_{\mathrm{batches}} \in \mathbb{N}$ | S5 |
| $\tau$ | warm-up fraction (`sep_warmup_frac`), **fixed, not searched** | $\tau \in [0, 1]$ | S5 |
| $\lambda_{\mathrm{sep}}$ | asymptotic weight on the separation term | $\lambda_{\mathrm{sep}} \in [0, \infty)$ | S5 |
| $\lambda_{\mathrm{sep}}(t)$ | the weight in force at step $t$ | $\in [0, \lambda_{\mathrm{sep}}]$ | S5 |
| $g(t)$ | dimensionless ramp factor, $\lambda_{\mathrm{sep}}(t) = \lambda_{\mathrm{sep}} \, g(t)$ | $g(t) \in [0, 1]$ | S5 |
| $P$ | `patience`, epochs without improvement before stopping | $P \in \mathbb{N}$ | S7 |
| $N_{\mathrm{calls}}$ | GP trial budget (`n_calls_joint`) | $N_{\mathrm{calls}} \in \mathbb{N}$ | S6 |
| $N_{\mathrm{seeds}}$ | seeds per trial (`n_seeds`) | $N_{\mathrm{seeds}} \in \mathbb{N}$ | S6 |
| $\mu_c, \mu_G$ | class mean and global mean of the class means | $\in \mathbb{R}^{E}$ | S5 |
| $R^2$ | goodness of fit of the cost model | $R^2 \le 1$ | S7 |

**Conventions.** "S$k$" means stage $k$. Steps are counted by $t$ and an epoch
is $n_{\mathrm{batches}}$ steps. "Column" means a surrogate-facing dimension
after skopt's one-hot expansion; "axis" means a declared dimension.

## 2. Glossary

Ordered by first appearance.

**Condition.** One combination of (mining strategy, loss type, strict
semi-hard filter), plus the 2x2 head geometry. 13 legal conditions x 4 heads
= the 52 historical cells.

**Legality projection ($\Pi$).** A deterministic idempotent map applied to a
sampled point *before* any config is built, collapsing illegal combinations
onto legal ones. A projection, not a penalty: it produces duplicate
observations, which a GP handles natively, rather than teaching the surrogate
that a factor is bad for reasons unrelated to that factor. Section 4.2.

**Provably empty cell.** `mining_strategy="hard"` with `strict_semihard=True`:
hard mining returns only triples with $D_{an} < D_{ap}$, the filter keeps only
$D_{ap} < D_{an}$, so the intersection is empty by construction. Measured:
16814 mined, 0 surviving. The run trains nothing and looks stable.

**Inactive coordinate.** An axis present in every point but not read for the
sampled $\ell$. Required because `gp_minimize` needs a fixed-length vector.

**Activity mask $A(\ell)$.** Which loss hyper-parameters $\ell$ actually reads:
$A(\texttt{triplet}) = \{m_{\cos}\}$, $A(\texttt{joint}) = \{\alpha\}$,
$A(\texttt{joint\_sep}) = \{\alpha, \lambda_{\mathrm{sep}}, \texttt{sep\_centre\_means}\}$.

**Warm-up.** The deterministic ramp replacing the latching silhouette gate.

**Size mix.** The distribution of model sizes the search actually draws, as
opposed to the corners of the architecture space. Section 4.7.

## 3. What changed, by file

**New:** `condition_space.py`, `search_dry_run.py`,
`hpc/make_joint_search_config.py`, `hpc/Config/config_l3c_joint_search.json`,
and four smoke tests in `Smoke_Tests/`.

**Modified:** `config.py`, `search.py`, `dsn_joint_loss.py`, `train.py`,
`run_optimization.py`, `hpc/preflight_config.py`, and two existing smoke tests
that reached the gate through `CompositeDSNLoss`.

## 4. Stage by stage

### 4.1 Stage 1 -- config surface

`TrainConfig` gains `sep_centre_means` (`Optional[bool]`, `None` = the
pre-existing automatic rule) and `sep_warmup_frac` ($\tau$, default `0.0` =
full weight from the first step = pre-existing ungated behaviour).
`SearchConfig` gains six `*_choices` level lists and `n_initial_points_joint`.
Every default reproduces current behaviour, so an archived config parses to
the object it always did.

### 4.2 Stage 2 -- the legality layer

`condition_space.py` is **pure**: standard library only, no torch, no skopt,
not even `config`. The search, the preflight, the factorial generator and the
tests all import it without a deep-learning stack.

$\Pi$ covers **two** clauses, where the design document states only the
second:

1. `strict_semihard <- False` whenever $\ell = \texttt{triplet}$ (the filter
   does not exist there; this is the D1 clamp expressed as part of $\Pi$);
2. `strict_semihard <- False` whenever mining is `hard` (the provably empty
   cell).

Folding both into one function means exactly one place decides what a
condition means. Verified: 18 raw triples $\to$ 13 legal, idempotent, and the
13 x 4 cell names reproduce the shipped filenames of **both** tiers exactly
(104 config JSONs checked).

### 4.3 Stage 3 -- the loss-HP triple

`loss_hp_names(train_cfg, superset=False)`. $A(\ell)$ is the single source of
truth for both branches; the staged branch removes `sep_centre_means`, which
that phase predates.

Decision D1 is implemented as **"leave inactive fields at the base config's
value"**. The base config is fixed for the whole study, so it *is* the clamp
constant, and two points differing only in inactive coordinates build
byte-identical configs. Consequence: the base config must set
`lambda_sep = 0.1` (the `TrainConfig` default) or every non-`joint_sep` trial
fires the "INERT lambda_sep" warning. `preflight_config.py` now checks this.

### 4.4 Stage 4 -- the 18-axis space

18 declared axes, 22 surrogate columns (11 numeric + 5 `Integer` binaries + 2
three-level `Categorical` one-hotting to 6), confirmed by two independent
counts. `config_from_joint_condition_point` projects first, then builds, then
rebuilds `cfg.train` through `dataclasses.replace` so
`TrainConfig.__post_init__` genuinely re-runs -- direct attribute assignment
skips validation and `cfg.validate()` only warns, so without this a bug in
$\Pi$ would yield a silently zero-loss run instead of a loud failure.

### 4.5 Stage 5 -- the warm-up replaces the gate

$$\lambda_{\mathrm{sep}}(t) = \lambda_{\mathrm{sep}} \cdot \min\left(1, \frac{t}{\tau T}\right), \qquad T = E_{\max} \cdot n_{\mathrm{batches}} \tag{1}$$

for all $t \in \{0, 1, \dots, T\}$, with $\tau$ **fixed**. `SepWarmup` is
deliberately the same shape as `SilhouetteGate` (step buffer, 0-dim tensor
return, `stats()`), so `CompositeDSNLoss.forward` changed by one line. The
gate class is **kept** for the archived analysis tooling but is no longer
constructed; `CompositeDSNLoss` now *rejects* `gate_threshold` with
`TypeError` rather than accepting it inertly, and `build_loss_and_miner` warns
when a config carries one (all 20 archived `joint_sep` cells do).

**Step convention.** $g$ is evaluated at $t$ = steps *completed*, so the first
batch sees $t = 0$ and weight exactly 0. The alternative would start a
"warm-up" at nonzero weight.

**A consequence, verified against a real run.** History logs
$g(k \cdot n_{\mathrm{batches}} - 1)$ at the end of epoch $k$, not
$g(k \cdot n_{\mathrm{batches}})$, because the scale is computed before the
counter advances. With $\tau T = 20$ and $n_{\mathrm{batches}} = 5$, epoch 4
logs $0.95$ and epoch 5 is the first to log $1.0$, though full weight arrives
at step 20 -- the first batch of epoch 5. Read `sep_step` alongside
`sep_warmup_scale` when the exact step matters.

$T$ is the **planned** budget, not what a run takes. A run reaching less than
$\tau$ of its cap never sees full $\lambda_{\mathrm{sep}}$; `sep_lambda_t` in
history makes that visible, and preflight warns when $P / E_{\max} < \tau$.

### 4.6 Stage 6 -- objective wiring

`search_mode` gains `"joint_conditions"` -- one knob rather than `"joint"`
plus a boolean, so the meaningless combination (staged + searched conditions)
is not representable. `_run_gp` gains an `annotate` hook; every trial record
carries `raw_condition`, `condition`, `projected`, `cell` and
`active_loss_hps`, on the failure path too.

**The two named acceptance tests, checked rather than assumed:**

- $N_{\mathrm{seeds}} = 1$ does **not** yield NaN. `evaluate_candidate` uses
  `arr.std()`, i.e. population std with `ddof = 0`, so a single seed gives
  `std = 0.0`. A `ddof = 1` std would give NaN, which `_run_gp` documents as
  unfittable.
- `tie_break_gamma = 0.0` genuinely disables the tie-break: $\epsilon =$
  `None` and the objective is exactly $-\mathrm{mean}(\text{primary})$
  whatever the secondary metric does. A nonzero $\gamma$ demonstrably differs,
  so the test cannot pass vacuously.

Note that `selection_primary = "silhouette"` is continuous and therefore
disables the tie-break regardless of $\gamma$ -- with a warning unless
$\gamma = 0$, which is why the generated config sets it to zero.

### 4.7 Stage 7 -- the dry run and the submission gate

`search_dry_run.py` samples the space, projects, builds every point **through
the same builder the real search uses**, and reports coverage. It trains
nothing and needs no data, so it runs on a login node.

**Coverage, measured.** All 13 conditions and 4 heads are covered by 40 draws;
the *joint* cells are what discriminate:

| draws | conditions | heads | cells | median draws/cell |
|---|---|---|---|---|
| 40 | 13/13 | 4/4 | 28/52 | 1.0 |
| 100 | 13/13 | 4/4 | 44/52 | 2.0 |
| 300 | 13/13 | 4/4 | 52/52 | 5.0 |

This is why `n_initial_points_joint = 100`: at 40 the surrogate would take over
having never sampled 24 of the 52 cells.

**Size mix, measured for the first time.** Over 300 draws at the configured
ranges: depths near-uniform across $\{2,3,4,5\}$; parameters min 0.01 M,
median 0.71 M, **mean 4.46 M**, max 31.6 M. Mean/median $= 6.26$. Wall clock
scales with the **mean**, so the typical sampled model badly understates cost.

**The gate** fails on incomplete coverage, on an unmeasured wall clock, on
overrun against walltime minus margin, and on a badly-fitting cost model
($R^2 < 0.70$). The last was added after observing $R^2 = 0.293$ on small
models, where fixed overhead swamps the parameter term and `polyfit` still
returns a confident-looking hour count. An unreliable estimate used as a
submission gate is worse than none, because it carries authority.

## 5. Where this departs from the design document

Three places, all deliberate:

1. **`loss_hp_names` is not unconditionally the superset.** Taken literally
   that widens the staged phase-2 space from 1 axis to 4 under `triplet` and
   from 2 to 4 under `joint_sep`, rewriting every archived staged run. It is a
   flag instead.
2. **The joint space is extended additively, not in place.** `joint_space()`
   and `config_from_joint_point()` are untouched at 10 axes; the condition
   search gets its own functions. Extending in place would have changed the
   existing `search_mode="joint"`.
3. **"Integer yields genuine Python ints" is false.** skopt's `Integer.rvs`
   returns `numpy.int64`. The *substance* of the BUG 2 warning survives --
   that warning is about `Real` yielding floats such as 0.37, and
   `Block_array[0.37]` raises `TypeError` while `Block_array[numpy.int64(1)]`
   does not -- so the encoding decision stands. But numpy scalars are not
   JSON-serialisable and the trial log is JSON, so points are normalised to
   native types at the projection boundary.

## 6. Verification

Every suite in the repository passes, on a **fresh extract of the branch with
these files applied**, not merely in the development copy: `dsn_joint_loss`
30/30, `loss_type_wiring` 14/14, `run_optimization` 21/21, plus `config`,
`search`, `train`, `objective_wiring` and the four new suites. All Python
files are pure ASCII (hpc-python-compat) and compile.

Beyond the suites: the staged pipeline was proven unchanged by dumping every
staged artefact from the original module and the modified one in separate
processes (2458 lines of JSON each, zero differences); $\Pi$ and the cell
names were checked against the 104 shipped config JSONs rather than the
generator that produced them; the warm-up was read out of a real 8-epoch
`train()` run; and a real 10-trial joint-condition search was run end to end
with `train()` in the loop.

## 7. Open points

1. **The Stage 7 gate has not been run on the cluster.** The 157 h estimate is
   the depth-4 rate and excludes the size mix. Run
   `search_dry_run.py --time-points 10 --walltime 144` on davinci and check
   the exit code before `qsub`.
2. **$E_{\max} = 100$, $P = 40$ are the design document's stated values.** It
   recommends 60/20 but records that as not yet accepted, so the generator
   defaults to the stated values and prints the arithmetic rather than
   choosing.
3. **This search does not answer the original scientific question.** The GP
   allocates trials adaptively, so `triplet` may receive very few. It returns
   a tuned configuration; it does not establish that the composite objective
   beats the triplet baseline at matched budget. That needs its own run.
4. **Single-seed ranking is unreliable among near-ties.** Measured
   within-cell seed sd $s = 0.073$ against between-cell 0.117. The Stage 8
   confirmatory re-fit of the top 5 at 5 seeds is part of the plan, not
   optional.
5. **$\tau = 0.3$ is asserted, not tuned.** The ridge argument justifies
   fixing *some* value, not this one.
6. **`sep_centre_means = False` is a different objective, not a corrected
   one.** It imposes $\mu_G \to 0$ in addition to the ETF condition. Any
   write-up must say which form ran; `sep_centred` in history records it.
7. **Inactive dimensions cost sample efficiency by an unquantified amount.**
   Up to 3 of the 22 columns are flat for any given trial.
8. **`class_overlap = 0.01` remains uncalibrated**, so every absolute
   silhouette sits on an uncalibrated difficulty scale. Unchanged by this work.

## 8. References

- Papyan, V., Han, X. Y., Donoho, D. L. (2020). *Prevalence of neural collapse
  during the terminal phase of deep learning training.* PNAS 117(40),
  24652-24663. DOI 10.1073/pnas.2015509117 (PMC7547234). **Full text read**;
  the source for the NC2 definition used by the separation term, and for NC
  being a long-training phenomenon.
- `HANDOFF_joint_gp_search.md` and `HANDOFF_factorial52_analysis.md` (project
  knowledge base) -- the design and its predecessor.
- Everything else in this document is measured in this work, from the
  repository source, or labelled as inference.
