# Wiring report: C1-C6 implemented into Deep-Summary-Network

**Date:** 25 July 2026
**Input:** `HANDOFF_wiring_changes.md` (25 July 2026), `dsn_pipeline`, `dsn_smoke_tests`,
`search_3class_hpc.zip` (archived baseline).
**Status:** all six changes wired. Rung 0 and Rung 1 pass here. **Rung 2, the stated
acceptance gate, has NOT been run** -- see section 4.

## Abstract

The handoff specified six changes (C1-C6) restoring discriminative power to the
model-selection benchmark, three of them already built but unwired, one specified but
unwritten, two config-only. This document records what was implemented, at which exact
integration point, what was verified and how, what was deliberately deviated from and
why, and what remains for the target machine. It covers the wiring only: it makes no
claim about the network's performance on the new benchmark, which nothing here has
measured.

---

## 1. Notation and symbols

| Symbol | Name / meaning | Type & domain | Units | First used in |
|---|---|---|---|---|
| $n$ | number of latent factors | $n \in \mathbb{N}$, $n \ge 1$ | dimensionless | 2.1 |
| $k$ | latent axis index | 1-based in prose, 0-based in code; stated at each use | dimensionless | 2.1 |
| $\phi$ | latent coordinate vector | $\phi \in [0,1]^n$ | dimensionless | 2.1 |
| $\phi_k$ | $k$-th latent coordinate | $\phi_k \in [0,1]$ | dimensionless | 2.1 |
| $C$ | number of phenotype classes | $C \in \mathbb{N}$, $C \ge 2$ here | dimensionless | 2.1 |
| $c$ | class label | $c \in \{0,\dots,C-1\}$ | dimensionless | 2.1 |
| $n_c$ | traces in class $c$ | $n_c \in \mathbb{N}$, $n_c \ge 1$ | count | 2.1 |
| $S$ | label-carrying axis subset | $S \subseteq \{1,\dots,n\}$, $S \ne \emptyset$ | dimensionless | 2.1 |
| $S^{\complement}$ | label-irrelevant ("free") axes | $S^{\complement}=\{1,\dots,n\}\setminus S$ | dimensionless | 2.1 |
| $\tau$ | class overlap (difficulty knob) | $\tau \in \mathbb{R}_{\ge 0}$ | normalized latent units | 2.1 |
| $T_{\mathrm{rec}}$ | trace duration | $T_{\mathrm{rec}} \in \mathbb{R}_{>0}$ | s | 2.1 |
| $f_s$ | IFR sampling rate | $f_s \in \mathbb{R}_{>0}$ | Hz | 2.1 |
| $\Delta t$ | IFR bin width, $\Delta t = 1/f_s$ | $\Delta t \in \mathbb{R}_{>0}$ | s | 2.1 |
| $y$ | true labels of the evaluation split | $y_i \in \{0,\dots,C-1\}$ | dimensionless | 2.3 |
| $N_{\mathrm{eval}}$ | number of evaluation windows | $N_{\mathrm{eval}} \in \mathbb{N}$, $\ge 2$ | count | 2.3 |
| $\Delta_{\min}(y)$ | ARI resolution of the evaluation set | $\in (0, 1.5]$ | dimensionless | 2.3 |
| $s_{\mathrm{lo}}, s_{\mathrm{hi}}$ | assumed bounds of the secondary metric | $s_{\mathrm{lo}} < s_{\mathrm{hi}}$ | dimensionless | 2.3 |
| $W_{\mathrm{sec}}$ | secondary range, $s_{\mathrm{hi}} - s_{\mathrm{lo}}$ | $\in \mathbb{R}_{>0}$ | dimensionless | 2.3 |
| $\gamma$ | safety factor | $\gamma \in [0,1]$; $0$ disables | dimensionless | 2.3 |
| $\varepsilon$ | tie-break weight | $\varepsilon \in \mathbb{R}_{>0}$ | dimensionless | 2.3 |
| $t$ | trial index | $t \in \{0,\dots,n_{\mathrm{calls}}-1\}$ | dimensionless | 2.3 |
| $\sigma$ | seed index | $\sigma \in \{1,\dots,S_{\mathrm{seeds}}\}$ | dimensionless | 2.3 |
| $e$ | epoch index (1-based) | $e \in \{1,\dots,E_{\max}\}$ | dimensionless | 2.3 |
| $e^\star(t,\sigma)$ | selected epoch | $e^\star \in \{0,1,\dots,E_{\max}\}$; $0$ = none | dimensionless | 2.3 |
| $u_e, v_e$ | primary / secondary metric at epoch $e$ | $\in \mathbb{R} \cup \{-\infty\}$ | dimensionless | 2.3 |
| $J_\varepsilon(t)$ | composite search objective | $J_\varepsilon: \text{trial} \to \mathbb{R}$ | dimensionless | 2.3 |
| $n_{\mathrm{calls}}$ | trials in a phase | $n_{\mathrm{calls}} \in \mathbb{N}$, $\ge 1$ | count | 2.2 |
| $n_{\mathrm{init}}$ | random initial-design trials | $n_{\mathrm{init}} \in \{1,\dots,n_{\mathrm{calls}}\}$ | count | 2.2 |
| $Z$ | held-out embedding matrix | $Z \in \mathbb{R}^{N_{\mathrm{eval}} \times E}$ | dimensionless | 2.5 |
| $E$ | embedding dimension | $E \in \mathbb{N}$ | dimensionless | 2.5 |
| $i$ | evaluation-window index | $i \in \{0,\dots,N_{\mathrm{eval}}-1\}$ | dimensionless | 2.5 |
| $g_i$ | group (trace index) of window $i$ | $g_i \in \{0,\dots,n_{\mathrm{traces}}-1\}$ | dimensionless | 2.5 |
| $\phi_k^{(i)}$ | true $k$-th coordinate of window $i$'s trace | $\in [0,1]$ | dimensionless | 2.5 |
| $\hat\phi_k^{(i)}(Z)$ | out-of-fold ridge prediction | $\in \mathbb{R}$ | dimensionless | 2.5 |
| $\bar\phi_k$ | mean of $\phi_k$ over evaluation windows | $\in [0,1]$ | dimensionless | 2.5 |
| $R^2_k$ | factor-retention score of axis $k$ | $\in (-\infty, 1]$ | dimensionless | 2.5 |

### 1.1 Conventions

- **Index bases.** Epochs $e$ and mathematical axis indices $k$ are 1-based; class
  labels $c$, trial indices $t$, group indices $g_i$, window indices $i$ and *all
  Python indices* are 0-based. Where a code identifier is quoted
  (e.g. `label_axes=[0,1]`), the indices are 0-based, i.e. $S=\{1,2\}$ in 1-based
  notation. This is restated at every occurrence rather than left to inference.
- **Conditioning carried in full.** $\Delta_{\min}(y)$ is always written with its
  argument, because its dependence on the evaluation set is the entire point of C2.
  Likewise $\hat\phi_k^{(i)}(Z)$ keeps its $Z$.
- **Objective sign.** `gp_minimize` minimizes, so every objective is a negated quality
  score: larger quality $\Rightarrow$ more negative $J_\varepsilon(t)$.
- **Verification status.** **[MEASURED-HERE]** = produced by executing code in this
  session. **[MEASURED-PRIOR]** = carried from the handoff and re-reproduced here.
  **[UNVERIFIED]** = implemented but not yet executed. No claim mixes the three.

---

## 2. Glossary

Ordered by first appearance, because the concepts build on one another.

**Latent factor.** An unobserved scalar controlling one physically meaningful property
of the generated signal. Generative sense (a cause of the data), not the
neural-network sense (a hidden activation). Operative in 2.1.

**Label-irrelevant (free) axis.** A latent factor that varies across traces but does
not determine the class label -- the analogue of real biological variation the
phenotype label does not name. Operative in 2.1 and 2.5.

**Cache fingerprint.** A hash of everything determining the cached traces. Because
`cache_traces` skips any trace whose `.npz` already exists, the fingerprint is the
only thing that notices when the trace a name refers to has changed. Operative in 2.1.

**Initial design.** The first $n_{\mathrm{init}}$ Bayesian-optimization trials, drawn
quasi-randomly *before* any surrogate is fitted. Operative in 2.2.

**Metric resolution $\Delta_{\min}(y)$.** The smallest strictly-positive gap the
primary metric can take on a given evaluation set -- the finest distinction that set
can register at all. Operative in 2.3.

**Lexicographic ordering.** Ranking by a primary key, using a secondary key only to
break exact ties. Approximated here by a scalarization with a provably bounded weight.
Operative in 2.3.

**Selected epoch $e^\star$.** The epoch whose weights `train()` restores and returns:
the lexicographic argmax over $(u_e, v_e)$ against the component-wise running maxima.
Operative in 2.3.

**Drift (of a mirrored rule).** The failure mode where two copies of one rule -- here
`train.py`'s and the search's recomputation -- silently diverge after one is edited.
Operative in 2.3 and 4.2.

**Grouped cross-validation.** Cross-validation whose folds never split a group across
the train/test boundary. Here the group is the *trace*. Operative in 2.5.

**Factor retention.** The degree to which label-irrelevant latent factors remain
linearly decodable from the embedding, quantified by $R^2_k$. Operative in 2.5.

---

## 3. What was implemented

### 2.1 C1 -- latent generator wired (was: built, not wired)

New data mode `"latent"`, additive rather than a mutation of `"synthetic"`, so every
existing test and `config_toy.json` keep meaning what they meant.

- `config.py`: new `LatentConfig` and `LatentAxisOverride` dataclasses, nested as
  `DataConfig.latent`; `data_mode` validated set extended to
  `("synthetic", "real", "numpy", "latent")`.
- `latent_burst_generator.py`: added `AXIS_REGISTRY` (name to axis), `resolve_axes`,
  `build_latent_spec`. Purely additive.
- `run_optimization.py`: `latent_spec_from_config` adapter, `save_latent_artifacts`,
  a `mode == "latent"` branch in `build_traces` reusing `make_synthetic_specs`
  unchanged, the latent block added to `_data_fingerprint`, and `latent` added to the
  CLI `--data-mode` choices.

The ground-truth table is written to `latent_ground_truth.json` **before** the
`--dry-run` early return, so a dry run produces the artefact C5 needs.

Class structure, unchanged from the handoff, for each fixed $(c,r)$ and independently
for each $k$:

$$
\phi_k=\begin{cases}
\mathrm{clip}(m_c+\tau\varepsilon_k,\,0,\,1), & k\in S,\ \varepsilon_k\sim\mathcal N(0,1)\ \text{i.i.d.},\\[2pt]
\mathrm{Uniform}(0,1)\ \text{i.i.d.}, & k\in S^{\complement},
\end{cases}
\qquad m_c=\frac{c}{C-1}.
\tag{1}
$$

### 2.2 C3 -- explicit `n_initial_points`

`SearchConfig.n_initial_points: int = 0`, where $0$ means the legacy rule

$$
n_{\mathrm{init}}=\min\!\big(10,\ \max(1,\lfloor n_{\mathrm{calls}}/2\rfloor)\big),
\tag{2}
$$

resolved by `objective_utils.resolve_n_initial_points` at the single call site in
`search.py::_run_gp`. Validated at config-construction time against **both**
`n_calls_arch` and `n_calls_train`, so the error fires before any trace is generated.

### 2.3 C2 -- adaptive tie-break

For each trial $t$, with both metrics read at the *same* selected epoch:

$$
J_\varepsilon(t)=-\frac{1}{S_{\mathrm{seeds}}}\sum_{\sigma=1}^{S_{\mathrm{seeds}}}
\Big[\mathrm{ARI}^{(t,\sigma)}_{e^\star(t,\sigma)}+\varepsilon\,\mathrm{Sil}^{(t,\sigma)}_{e^\star(t,\sigma)}\Big],
\tag{3}
$$

$$
\varepsilon=\gamma\,\frac{\Delta_{\min}(y)}{W_{\mathrm{sec}}}
\quad\Longrightarrow\quad
\varepsilon\,W_{\mathrm{sec}}=\gamma\,\Delta_{\min}(y)<\Delta_{\min}(y)
\quad\text{for all }\gamma\in(0,1).
\tag{4}
$$

$e^\star(t,\sigma)$ is recomputed from `history` by
`objective_utils.selected_epoch_index`, mirroring `train.py` exactly: for
$e = 1,\dots,E_{\max}$, with $u_{\mathrm{best}}=v_{\mathrm{best}}=-\infty$ initially,

$$
e^\star \leftarrow e \ \text{ iff } \ (u_e, v_e) > (u_{\mathrm{best}}, v_{\mathrm{best}}),
\qquad
u_{\mathrm{best}} \leftarrow \max(u_{\mathrm{best}}, u_e), \quad
v_{\mathrm{best}} \leftarrow \max(v_{\mathrm{best}}, v_e).
\tag{5}
$$

Two properties of (5) are load-bearing: the comparison is against the *component-wise*
running maxima, which need not be any single epoch's pair; and the $>$ is strict, so
the **first** epoch attaining a tied pair wins.

`evaluate_candidate` gains `epsilon=None` (None reproduces the pre-C2 objective).
`record["mean"]`, `["std"]`, `["scores"]` still report the **primary** metric, so they
stay comparable across runs with and without the tie-break; `record["objective"]` is
what `gp_minimize` minimizes. New keys: `sil_scores`, `sil_mean`, `selected_epochs`,
`epsilon`.

### 2.4 C4 + C6 -- config only

Two HPC configs derived programmatically from the archived `config_input.json`, so
nothing else drifts: `width_multiplier_range` $[1.5,2.5] \to [1.5,3.0]$;
`depth_exponent_range` **left at** $[3,5]$; `n_initial_points = 10`;
`tie_break_gamma = 0.5`. The pair differs in exactly one substantive field,
`mining_strategy` $\in$ {`hard`, `easy_positive`} **[MEASURED-HERE]** (field-by-field
diff, section 4.1).

### 2.5 C5 -- factor retention (new module)

`factor_retention.py`. For each axis $k$, with predictions pooled across folds and the
denominator taken about the **global** mean $\bar\phi_k$:

$$
R^2_k=1-\frac{\sum_i\big(\phi_k^{(i)}-\hat\phi_k^{(i)}(Z)\big)^2}
{\sum_i\big(\phi_k^{(i)}-\bar\phi_k\big)^2},
\qquad k \in \{1,\dots,n\},
\tag{6}
$$

both sums over all $i \in \{0,\dots,N_{\mathrm{eval}}-1\}$. This is *not* the average
of per-fold $R^2$ values, which would use a different mean in each denominator.

Cross-validation is `GroupKFold` with $g_i$ = the trace index, never the window index.
The ridge penalty is selected by `RidgeCV` **inside each training fold**. A constant
target yields `NaN`, not a spurious $1.0$, with a warning.

---

## 4. Verification

### 4.1 What passed here

| Check | Result | Status |
|---|---|---|
| Byte scan, 48 files | pure ASCII | **[MEASURED-HERE]** |
| `py_compile`, all modules + smoke tests | clean | **[MEASURED-HERE]** |
| `smoke_test_latent_and_objective.py` (handoff's own) | **21/21**, unchanged | **[MEASURED-PRIOR]** re-reproduced |
| naive / rich baseline ARI on the new benchmark | $0.0277$ / $0.3006$ | **[MEASURED-PRIOR]** re-reproduced |
| label-axis vs class correlation | $\lvert\rho\rvert = 0.956,\ 0.955$; free axes $\max = 0.084$ | **[MEASURED-PRIOR]** re-reproduced |
| $\Delta_{\min}(y)$ at $N_{\mathrm{eval}}=36$, $C=3$ balanced | $0.0846$ (max ARI below 1 $=0.9154$) | **[MEASURED-PRIOR]** re-reproduced |
| $\varepsilon$ at $\gamma=0.5$, $s\in[-1,1]$ | $0.02115$; $\varepsilon W_{\mathrm{sec}} = 0.04229 < 0.08459$ | **[MEASURED-HERE]** |
| $\Delta_{\min}(y)$ shrinks with $N_{\mathrm{eval}}$ | $0.0846 \to 0.0335 \to 0.0167$ at $36/90/180$ | **[MEASURED-HERE]** |
| $\Delta_{\min}(y)$ cost per phase | $43$ / $222$ / $821$ ms at $N_{\mathrm{eval}} = 36/180/600$ | **[MEASURED-HERE]** |
| legacy $n_{\mathrm{init}}$ reproduced | exact for $n_{\mathrm{calls}} \in \{1,2,3,4,15,20,50,100,753\}$ | **[MEASURED-HERE]** |
| `FAILED_OBJECTIVE` still worst | $1.0 > 0.52115$ = worst attainable $J_\varepsilon$ | **[MEASURED-HERE]** |
| C5 recovery / null | $R^2_k \approx 1.0000$ / all $< 0$ | **[MEASURED-HERE]** |
| **C5 grouping is load-bearing** | grouped $R^2 = -0.3023$ vs ungrouped $+1.0000$ | **[MEASURED-HERE]** |
| C1 fingerprint sensitivity | changes for $\tau$, $S$, axis set, ranges, `n_neurons`, `gaussian_window`, seed; stable otherwise | **[MEASURED-HERE]** |
| **C1 stale-cache refusal** | changing $\tau$ against the same cache dir raises | **[MEASURED-HERE]** |
| C6 configs differ in one field | `mining_strategy` only (plus `experiment_name`) | **[MEASURED-HERE]** |
| `smoke_test_config.py` (pre-existing) | PASS, unmodified | **[MEASURED-HERE]**, see caveat |
| `smoke_test_synthetic_config.py` | **21/21** PASS | **[MEASURED-HERE]**, see caveat |
| `smoke_test_search.py` | PASS through [J] end-to-end | **[MEASURED-HERE]**, see caveat |

**The caveat on the last three rows.** Torch could not be installed on the machine this
session ran on: the PyPI wheel needs the CUDA runtime libraries and the install failed
repeatedly under the available memory. Those three tests were therefore run against a
**verification-only import stub** for `torch` and `pytorch_metric_learning`. The stub
is not part of the deliverable and is not shipped. It is adequate for `config.py`
(pure dataclass logic; torch is never called) and for `build_traces` (numpy only), and
`smoke_test_search.py` patches `train` with a fake anyway. It is **not** adequate for
anything that performs real tensor arithmetic -- `smoke_test_data_splits.py` fails
inside the stub at `torch.as_tensor`, which is a stub limitation, not a regression.

### 4.2 What has NOT been run -- the acceptance gate

Rung 2, in the handoff's own terms, is the acceptance gate for this wiring, and it has
not been executed. Run `sh run_wiring_checks.sh` on the machine with torch. Not yet
run, all **[UNVERIFIED]**:

1. The pre-existing suite against **real** torch, in particular
   `smoke_test_data_splits.py`, `smoke_test_train.py`, `smoke_test_data_pipeline.py`,
   `smoke_test_run_optimization.py`, `smoke_test_end_to_end.py`.
2. `smoke_test_latent_wiring.py` against real torch (it passed under the stub, but the
   stub is the thing under suspicion).
3. **`smoke_test_selected_epoch.py`** -- the drift test. This is the one that matters
   most, because it is the only check that $e^\star$ recomputed by the search equals
   the `best_epoch` `train.py` itself recorded. It performs real toy training runs and
   therefore could not run here at all.
4. `run_optimization.py --dry-run` on `hpc/config_latent_3class_hard.json`.

Rung 3 (a real search, and the C6 two-miner ablation) is cluster-only and out of scope,
as stipulated.

---

## 5. Summary of results

- Eq. (1) latent construction: unchanged from the handoff, now reachable via
  `data_mode = "latent"` (section 2.1).
- Eq. (2) legacy $n_{\mathrm{init}}$: reproduced exactly by passing `0`/`None`
  (section 2.2), so no existing config changes behaviour.
- Eqs. (3)-(4) composite objective and its guarantee: wired, with $\gamma$ exposed and
  $\gamma = 0$ restoring pre-C2 behaviour (section 2.3).
- Eq. (5) selected-epoch rule: mirrored from `train.py`; agreement **[UNVERIFIED]**
  pending the drift test (section 4.2 item 3).
- Eq. (6) factor retention: implemented with grouping by trace, and the grouping shown
  to be load-bearing at $-0.3023$ vs $+1.0000$ (section 4.1).

---

## 6. Open points, caveats and assumptions

**Deviations from the handoff's literal text, all deliberate:**

1. **`n_latent` is a derived property, not a config field.** The handoff asked for
   `n_latent` alongside the axis selection; two fields that can disagree have no useful
   behaviour when they do, so $n = \lvert\texttt{axis\_names}\rvert$.
2. **The latent block does not re-declare $C$, $n_c$, $T_{\mathrm{rec}}$, $f_s$.** They
   are read from `synthetic_n_per_class`, `synthetic_duration_s`, `synthetic_fs`, which
   already existed and were already fingerprinted. Two sources of truth for a sampling
   rate is a bug waiting to happen.
3. **`selected_epoch_index`/`_scores` live in `objective_utils.py`, not `search.py`**
   as section 5.3 of the handoff specified. `search.py` imports torch, which would make
   the rule untestable on Rung 1. `search.py` re-exports them, so
   `from search import selected_epoch_index` works.
4. **`tie_break_gamma` defaults to $0.5$, i.e. C2 is ON by default.** This changes the
   objective for every run. Set it to $0.0$ for exact pre-C2 behaviour. Chosen because
   an inert fix is not a fix, and because $47$ of $50$ phase-1 trials returning exactly
   $1.0$ is the failure C2 exists to prevent.
5. **The regularization phase keeps the legacy $n_{\mathrm{init}}$ rule.** It has its
   own budget (`RegularizationConfig.n_calls`), and applying a `SearchConfig` field to
   a different phase's budget is cross-wiring. Two lines to change if you disagree; the
   call site says so.

**Assumed without verification here (inherited from the handoff, unchanged):**

6. The generator's numeric ranges $[a_k,b_k]$ are **not** calibrated to real
   recordings. They bracket the repository's own `CONTROL_PARAMS`/`PATHO_PARAMS`. The
   generator is biologically *motivated*, not *calibrated*, and must not support an
   effect-size claim until refitted. `LatentAxisOverride` exists so that refit needs no
   code edit.
7. Latent factors are drawn independently; real burst statistics are correlated.
8. Class centres are equally spaced, $m_c = c/(C-1)$.
9. The network has never been run on the new benchmark. $0.3006$ is a **baseline
   floor** only. If the network cannot beat it, lower $\tau$ before drawing any
   architecture conclusion.

**Specific to this session:**

10. With $n_c = 3$ traces per class, the free-axis coordinates take only
    $C \cdot n_c = 9$ distinct values, so the effective sample size of Eq. (6) is
    closer to $9$ than to $N_{\mathrm{eval}}$. `GroupKFold` removes the *bias*; it
    cannot manufacture *power*. Raising $n_c$ remains the cheapest structural fix, and
    should be done before reading much into any single $R^2_k$.
11. C5 is not yet called from anywhere in the pipeline. The module, its ground-truth
    export and its smoke test are complete, but wiring it into `run_final` /
    `evaluate.py` as a reported metric was not part of the six changes and was not
    done. It is currently an analysis you run on a finished run's embeddings.
12. $\Delta_{\min}(y)$ is recomputed once per phase rather than once per study
    (up to four times per run). At the archived $N_{\mathrm{eval}} = 36$ this is
    $43$ ms; the growth is quadratic.

---

## 7. References

**Handoff and prior session:** `HANDOFF_wiring_changes.md`, `latent_factor_benchmark_design.md`,
and the three modules built there. All quantities marked **[MEASURED-PRIOR]** are that
session's measurements, re-reproduced here by re-running its smoke test.

**Read directly this session:** the repository sources listed in section 2;
`scikit-optimize` 0.10.2 (installed and used, matching the version the handoff read
`create_result` from).

**Literature:** none retrieved this session. The biological grounding of the axis
choice and direction (Cotterill et al., *J Biomol Screen* 2016) and the record-wise vs
subject-wise argument behind the grouping constraint (Saeb et al., *GigaScience* 2017)
are carried from the handoff, which recorded both as full-text reads. **No new
literature claim is made here**, and no numeric value in this document comes from a
paper -- every number is either a measurement from this session, a re-reproduced
measurement from the prior session, or a definition.

**Stated from reasoning rather than a checked source:** the design deviations in
section 6 items 1-5, and the interpretation of $R^2_k$ in `factor_retention.py`'s
docstring.
