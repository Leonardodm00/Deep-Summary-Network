# Change 5 (whole-culture split) + growing patience

**Date:** 28 July 2026
**Target:** `Deep Summary Network/Deep_v2`, package root `Main/`
**Base:** the repository as shipped in `dsn_wiring_C1_C6.zip`

**Abstract.** Two things land here. First, Change 5 of the v3 handoff: a
whole-culture train/validation/test splitter, `make_trace_splits`, added
alongside the existing `make_time_segment_splits` rather than replacing it, plus
the `trace_of_window` provenance array that Changes 4 and 3 both depend on.
Second, the requested growing-patience early-stopping rule, in which the
patience budget expands while the primary metric sits on a plateau, with its
improvement threshold derived from a measured label-shuffled silhouette floor.
**Covered:** the new splitter and its assignment algebra, the patience state
machine and its termination guarantee, the floor measurement, the config surface
for both, and the two new smoke-test suites. **Deliberately excluded:** Changes
1, 2 and 4 of the handoff; any change to the backbone, the miner, the loss, or
the search algorithm; and the pre-existing `smoke_test_selected_epoch.py`
failure documented in section 6, which is diagnosed but not fixed.

---

## 1. Notation and symbols

| Symbol | Name / meaning | Type & domain | Units | First used in |
|---|---|---|---|---|
| $u$ | culture (trace) index, GLOBAL | $u \in \{0,\dots,n_{\mathrm{tr}}-1\}$, 0-based | dimensionless | 2 |
| $n_{\mathrm{tr}}$ | total cultures in the dataset | $n_{\mathrm{tr}} \in \mathbb{N}$ | count | 2 |
| $c$ | class (condition) label | $c \in \{0,\dots,C-1\}$, 0-based | dimensionless | 2 |
| $C$ | number of condition classes | $C \in \mathbb{N}$, $C \ge 2$ | count | 2 |
| $n_c$ | cultures of class $c$ | $n_c \in \mathbb{N}$ | count | 2 |
| $g_i$ | culture index of window $i$ (`trace_of_window[i]`) | $g_i \in \{0,\dots,n_{\mathrm{tr}}-1\}$ | dimensionless | 2 |
| $f_k$ | split fraction, $k \in \{\mathrm{tr},\mathrm{va},\mathrm{te}\}$ | $f_k \in (0,1)$, $\sum_k f_k = 1$ | dimensionless | 2 |
| $a_k$ | cultures apportioned to split $k$, per class | $a_k \in \mathbb{Z}_{\ge 0}$, $\sum_k a_k = n_c$ | count | 2 |
| $T$ | analysis-window length | $T \in \mathbb{N}$ samples | samples | 2 |
| $L_u$ | length of culture $u$'s trace | $L_u \in \mathbb{N}$ samples | samples | 2 |
| $w$ | early-stopping wait counter | $w \in \mathbb{Z}_{\ge 0}$ | count | 3 |
| $P_0$ | starting patience budget (`train.patience`) | $P_0 \in \mathbb{N}$ | count | 3 |
| $P$ | current patience budget | $P \in \mathbb{R}_{>0}$ | count | 3 |
| $P_{\max}$ | cap on $P$ (`train.max_patience`) | $P_{\max} \in \mathbb{N}$; 0 means uncapped | count | 3 |
| $g$ | patience growth rate (`train.patience_growth`) | $g \in \mathbb{R}_{\ge 0}$; $g < 1$ unless $P_{\max}$ set | count/epoch | 3 |
| $n^{*}$ | consecutive plateau epochs survivable | $n^{*} \in \mathbb{N} \cup \{\infty\}$ | count | 3 |
| $\bar{s}$ | mean silhouette over an evaluation split | $\bar{s} \in [-1,1]$ | dimensionless | 4 |
| $\hat{\mu}_{\mathrm{floor}}$ | mean of $\bar{s}$ under label permutation | $\in [-1,1]$ | dimensionless | 4 |
| $\hat{\sigma}_{\mathrm{floor}}$ | standard deviation of the same null | $\in \mathbb{R}_{\ge 0}$ | dimensionless | 4 |
| $R$ | permutations used to estimate the null | $R \in \mathbb{N}$, $R \ge 2$ | count | 4 |
| $\kappa$ | multiplier on the floor statistic | $\kappa \in \mathbb{R}_{>0}$; default 2 | dimensionless | 4 |
| $\delta_{\mathrm{sil}}$ | silhouette improvement threshold | $\delta_{\mathrm{sil}} \in \mathbb{R}_{\ge 0}$ | dimensionless | 4 |

### 1.1 Conventions

- All code and all JSON are 0-based; every index in this document is 0-based.
- $g$ is overloaded in the handoff's own notation ($g_i$ is a culture index,
  $g$ here is the patience growth rate). They never appear in the same
  expression; $g_i$ always carries its subscript and $g$ never does.
- "Culture" and "trace" are the same object, indexed by $u$.
- All Python source is pure ASCII (handoff section 10), verified by byte scan.

---

## 2. What changed in `data_splits.py`

### 2.1 New public functions

```
apportion(n, fractions, rule="largest_remainder") -> [a_tr, a_va, a_te]
assign_cultures(conditions, fractions, seed, mode, fold,
                min_train_cultures_per_class, alloc_rule) -> {"train": [...], ...}
make_trace_splits(traces, conditions, fs, data_cfg, base_seed=0,
                  mode="fractional", fold=None, split_seed=0,
                  min_train_cultures_per_class=2, fractions=None,
                  alloc_rule="largest_remainder") -> SplitBundle
```

`assign_cultures` is the pure assignment core: it takes only the label vector
and returns three index lists, so the whole assignment algebra is testable
without generating a single trace.

**Signature deviation from handoff section 8.2, flagged.** The handoff specifies
a flat signature carrying `window_s`, `train_stride_s`, `eval_stride_s` and `fs`
individually. Implemented instead with the first five parameters positionally
IDENTICAL to `make_time_segment_splits`, because the windowing parameters and the
augmentation config already live in `DataConfig` and `MEAWindowDataset` needs
`data_cfg.resolved_augmentation(fs)` regardless. The consequence is that the two
splitters are drop-in swappable at every existing call site. `fractions=` still
allows an explicit override of `data_cfg.split_fractions`.

### 2.2 `SplitBundle` gains four fields

All with defaults, so no existing construction breaks:

| field | meaning |
|---|---|
| `trace_of_window` | split name -> int array $g$, with $g_i$ the GLOBAL culture index of window $i$, in Dataset enumeration order |
| `cultures` | split name -> sorted int array of global culture indices in that split |
| `split_kind` | `"time_segment"` or `"trace"` |
| `fold` | leave-one-out fold index, or `None` |

`trace_of_window` is populated by BOTH splitters, per handoff section 8.3. For
the time-segment splitter this makes the leakage visible rather than hiding it:
all three arrays cover every culture.

It is read back out of `MEAWindowDataset.index` rather than re-derived from the
trace lengths, so it cannot drift from the windows the Dataset actually yields.
The smoke test derives it independently, which is what makes the comparison
meaningful.

### 2.3 Apportionment: a deviation from the handoff, with a reason

Handoff section 8.2 specifies $a_{\mathrm{tr}} = \lfloor f_{\mathrm{tr}} n_c
\rfloor$, $a_{\mathrm{va}} = \lfloor f_{\mathrm{va}} n_c \rfloor$, remainder to
test. Handoff section 9.2 assertion (c) requires that counts "differ by at most
one" from the request. **These two are inconsistent.** At $n_c = 18$ under
$(0.6, 0.2, 0.2)$:

| rule | $a_{\mathrm{tr}}, a_{\mathrm{va}}, a_{\mathrm{te}}$ | ideal | worst error |
|---|---|---|---|
| `floor` (handoff) | 10, 3, **5** | 10.8, 3.6, 3.6 | **1.40 cultures** |
| `largest_remainder` (default) | 11, 4, 3 | 10.8, 3.6, 3.6 | 0.60 cultures |

The floor rule gives test 28 percent of the cultures where 20 percent was asked
for, and fails assertion (c). Largest-remainder (Hamilton) apportionment gives
every split either $\lfloor f_k n_c \rfloor$ or $\lceil f_k n_c \rceil$, hence

$$\bigl| a_k - f_k n_c \bigr| < 1 \quad \text{for every } k, \qquad \sum_k a_k = n_c \text{ exactly}. \tag{1}$$

`largest_remainder` is therefore the default. `rule="floor"` is retained so a
pre-existing assignment can be reproduced bit-for-bit. Eq. (1) is asserted
directly in `smoke_test_trace_splits.py` [I] over $n \in [0, 60)$ and four
fraction triples, and the floor rule's violation at $n_c = 18$ is asserted in
[C] so that flipping the default back fails the suite.

### 2.4 Minimum-occupancy repair

After apportionment the counts are repaired so that each split holds at least
one culture of class $c$ and train holds at least
`min_train_cultures_per_class`. Items are taken from the split with the largest
surplus over its own minimum, ties broken by ascending split index, so the
perturbation away from the requested fractions is minimal and deterministic.
Feasibility is checked first: $n_c \ge$ `min_train_cultures_per_class` $+ 2$, else
a `ValueError` naming $n_c$ and the fractions.

**`min_train_cultures_per_class = 2` is not a preference.** Cross-culture
positives (Change 4) require at least two same-class training cultures, or an
anchor has no same-class partner from a different culture and the batch offers
no positive at all. 2 is the smallest value at which Change 4 functions.

### 2.5 Modes

- **`fractional`** -- per class, permute with `np.random.default_rng([split_seed,
  c])`, apportion, repair. The per-class stream is independent, so adding a
  class does not reshuffle the classes already present (asserted in [G]).
- **`leave_one_out`** -- $n_{\mathrm{folds}} = \min_c n_c$. In fold $f$, culture
  $f$ of each class is test and culture $(f+1) \bmod n_c$ is validation, taken in
  ASCENDING GLOBAL INDEX order and NOT permuted, so "fold 3" names the same
  held-out culture on every machine and in every log. Each culture is test in
  exactly one fold if and only if every class has the same $n_c$; unequal class
  sizes emit a `RuntimeWarning` saying so.

### 2.6 Determinism

Class order comes from `sorted()`, the per-class stream from
`np.random.default_rng([split_seed, c])`, and no set or dict iteration order is
consulted anywhere in the assignment. `split_seed` is deliberately SEPARATE from
`base_seed`: seed-averaging over training seeds must not reshuffle the split
underneath the average.

---

## 3. Growing patience -- `adaptive_patience.py` (new module)

### 3.1 The rule

With $w$ the wait counter and $P$ the budget, per epoch:

$$\text{improvement:} \quad w \leftarrow 0, \;\; P \leftarrow P_0 \text{ (if reset\_on\_improvement)}$$
$$\text{plateau:} \quad w \leftarrow w+1, \;\; P \leftarrow \min(P_0 + g\,\nu,\; P_{\max})$$
$$\text{stop} \iff w \ge P \tag{2}$$

where $\nu$ counts plateau epochs since the last reset.

### 3.2 Termination

On $n$ consecutive plateau epochs from a consistent state with $w = 0$,

$$n^{*} \;=\; \Bigl\lceil \frac{P_0}{1-g} \Bigr\rceil, \qquad \text{for all } g \in [0,1), \; P_{\max} \text{ unset}. \tag{3}$$

So $g$ multiplies the effective patience by $1/(1-g)$, which is the number to
choose $g$ from:

| $g$ | 0 | 0.5 | 0.75 | 0.8 | 0.9 |
|---|---|---|---|---|---|
| effective patience at $P_0 = 10$ | 10 | 20 | 40 | 50 | 100 |

**$g \ge 1$ with no cap never terminates** and is rejected at construction, in
both `AdaptivePatience.__init__` and `TrainConfig.__post_init__`, rather than
being discovered as a burnt cluster job.

Eq. (3) is verified against brute-force simulation over 494 combinations of
$(P_0, g, P_{\max}, w)$ in `smoke_test_adaptive_patience.py` [B], with an
independent reference computed in exact rational arithmetic (`fractions.Fraction`)
because $4/(1-0.8)$ evaluates to 20.000000000000004 in binary floating point and
a float reference would demand 21 for an answer that is exactly 20.

### 3.3 The budget is derived, not accumulated

$P$ is computed as $\min(P_0 + g\nu, P_{\max})$ from an integer counter $\nu$,
never by repeated addition. Two bugs this removes, both found during review:

1. **Drift.** $2.0 + 0.9$ twenty times is not $20.0$ in binary floating point,
   so an accumulated budget disagreed with Eq. (3) at $(P_0, g) = (2, 0.9)$.
2. **Unrepresentable states.** With an accumulated budget it was possible to
   hold $w$ and $P$ that disagreed about how many plateau epochs had happened.
   With a derived budget that state cannot be constructed.

### 3.4 Backward compatibility

`patience_growth = 0.0` (the default) reproduces the existing fixed-patience
rule step for step. Asserted in [A] over 200 random improvement sequences across
five patience values against an independently written reference implementation
of the current rule. Wiring this in therefore changes no archived result until
the growth is turned on.

---

## 4. The improvement threshold -- a flagged deviation

The request was: patience grows when $\bar{s}$ does not change beyond a
threshold of "2 times or more the evaluated floor", i.e.
$\delta_{\mathrm{sil}} = \kappa\,\hat{\mu}_{\mathrm{floor}}$ with $\kappa = 2$.

**Taken literally this disables early stopping.** Under a random labelling
$a(i)$ and $b(i)$ estimate the same underlying mean distance, so
$\mathbb{E}[\bar{s}]$ sits AT zero rather than at some positive level. Finite
samples push it below zero, because $b(i)$ takes a minimum over the $C-1$ other
classes and a minimum of noisy quantities is biased downward, whereas $a(i)$
involves no minimum. The bias grows with $C$ and is weakest at $C = 2$.

Measured on L2-normalised embeddings, $R$ = 120 permutations, cosine metric
(`smoke_test_adaptive_patience.py` [E], reproducible with fixed seeds):

| | $\hat{\mu}_{\mathrm{floor}}$ | $\hat{\sigma}_{\mathrm{floor}}$ | $2\hat{\mu}_{\mathrm{floor}}$ |
|---|---|---|---|
| $C = 2$ (the real experiment) | $-0.00002$ | $0.00841$ | $-0.00004$ |
| $C = 4$ (the benchmark) | $-0.04810$ | $0.00445$ | $-0.09620$ |

A negative $\delta_{\mathrm{sil}}$ makes a DECREASE in $\bar{s}$ count as an
improvement. That does not loosen early stopping, it inverts it, and silently.

**Resolution.** Three modes, defaulting to the second:

| `min_delta_sil_mode` | $\delta_{\mathrm{sil}}$ | note |
|---|---|---|
| `"absolute"` | `train.min_delta_sil` verbatim | current behaviour, the default |
| `"floor_scale"` | $\kappa\,\hat{\sigma}_{\mathrm{floor}}$ | RECOMMENDED for silhouette-primary |
| `"floor_location"` | $\kappa\,\hat{\mu}_{\mathrm{floor}}$ | the literal reading; RAISES when $\hat{\mu}_{\mathrm{floor}} \le 0$ |

`"floor_scale"` still honours the "2 times the floor" intent: it asks for a gain
twice the size of the run-to-run wobble the null can produce on this evaluation
set. `"floor_location"` refuses rather than returning $\delta_{\mathrm{sil}} \le
0$, and its error message names `"floor_scale"` and prints what that would give.

**This is a deviation from the instruction as given and is one config field
wide.** Set `min_delta_sil_mode = "floor_location"` to get the literal behaviour,
on evaluation sets where $\hat{\mu}_{\mathrm{floor}} > 0$.

The floor is measured ONCE, at the first epoch yielding a finite silhouette,
then frozen for the run. It must be frozen: a threshold that moved with the
embedding it is judging would make "improvement" mean something different every
epoch. It is persisted into the checkpoint so a resumed run keeps the same one.

---

## 5. Config surface

New `TrainConfig` fields. **Every default reproduces current behaviour exactly**,
so the block is inert until opted into.

| field | type | default | meaning |
|---|---|---|---|
| `patience_growth` | `float` $\ge 0$ | `0.0` | $g$; must be $< 1$ unless `max_patience` is set |
| `max_patience` | `int` $\ge 0$ | `0` | $P_{\max}$; 0 means uncapped |
| `patience_reset_on_improvement` | `bool` | `True` | `False` keeps the earned budget across improvements |
| `min_delta_sil_mode` | `str` | `"absolute"` | see section 4 |
| `min_delta_sil_kappa` | `float` $> 0$ | `2.0` | $\kappa$ |
| `sil_floor_permutations` | `int` $\ge 0$ | `200` | $R$ |

New method `TrainConfig.effective_patience()`, returning Eq. (3) capped by
$P_{\max}$. Both the `__post_init__` warning and the cross-field `validate()`
now compare `max_epochs` against THIS rather than against raw `patience`, which
understated the requirement by the factor $1/(1-g)$. The formula is not
duplicated: it is imported lazily from `adaptive_patience`, so the trainer and
the validator can never disagree. The import is lazy because
`adaptive_patience` pulls in sklearn and `config.py` must stay importable in a
parse-only environment (the PBS pre-flight does exactly that).

A suggested silhouette-primary configuration, for reference:

```json
"train": {
  "selection_primary": "silhouette",
  "patience": 10,
  "patience_growth": 0.5,
  "patience_reset_on_improvement": false,
  "min_delta_sil_mode": "floor_scale",
  "min_delta_sil_kappa": 2.0,
  "sil_floor_permutations": 200,
  "max_epochs": 60
}
```

Note `max_epochs` must exceed the EFFECTIVE patience, here
$\lceil 10/(1-0.5)\rceil = 20$, or early stopping cannot fire.

### 5.1 One decision left open

`patience_reset_on_improvement` defaults to `True`, the classic rule. `False`
keeps the budget earned so far, so a run that has already shown itself to be a
slow improver stays patient permanently. On a compressed, noisy metric at
$S_{\mathrm{seeds}} = 2$ the recommendation is `False`, but it is a real
behavioural choice and was not authorised, so the default is the conservative
one. It is irrelevant while `patience_growth = 0`.

---

## 6. `train.py` changes, and a PRE-EXISTING failure found

### 6.1 What was changed

- imports `AdaptivePatience`, `silhouette_floor`, `resolve_min_delta_sil`;
- `patience_counter` replaced by a `stopper` object;
- the floor is measured once and the stopper armed, before `_primary_secondary`;
- `min_delta_sil_eff` is passed to `_primary_secondary` in place of
  `tcfg.min_delta_sil` (a no-op under `mode = "absolute"`);
- per-epoch record gains `min_delta_sil_eff`, `patience_wait`, `patience_budget`;
- checkpoint extras gain `stopper_state` and `sil_floor`; the legacy
  `patience_counter` key is still written, and a checkpoint lacking
  `stopper_state` is still resumable.

Regression status after these changes: `smoke_test_train.py`,
`smoke_test_checkpoint.py`, `smoke_test_objective_wiring.py`,
`smoke_test_config.py`, `smoke_test_data_splits.py`,
`smoke_test_data_pipeline.py` and `smoke_test_metrics.py` all pass.

### 6.2 The pre-existing failure

`smoke_test_selected_epoch.py` FAILS, and it fails identically in the untouched
`dsn_wiring_C1_C6.zip` snapshot:

```
DRIFT: search recomputed e* = 1 but train.py recorded best_epoch = 4
(seed 0, selection_primary='silhouette').
```

Verified: the ARI-primary path passes; only the silhouette-primary path fails;
the comparison is epoch-number against epoch-number, not an index off-by-one;
and the failure reproduces byte-identically before any edit of mine. **It is not
caused by these changes.**

It matters because it sits directly on Change 3's critical path. Handoff section
7.1 states that the `selection_primary = "silhouette"` path is "already
implemented end-to-end" and that "no new code is required for the switch
itself". The wiring is indeed present in `config.py`, `train.py`,
`objective_utils.py` and `search.py` -- but the drift test that guards the
agreement between `train.py`'s selection rule and
`objective_utils.selected_epoch_index` does not hold under that setting. Until
it does, the search would score a different epoch from the one the trainer
selected. **Change 3 should not be considered done on the strength of the
section 7.1 verification alone.** Root cause not yet isolated.

---

## 7. Smoke tests

Both are registered in `run_all_smoke_tests.py` (18 suites now):
`smoke_test_trace_splits.py` after `smoke_test_data_splits.py`, and
`smoke_test_adaptive_patience.py` before `smoke_test_train.py`.

```bash
cd "Deep Summary Network/Deep_v2/Main"
python3 -m py_compile data_splits.py adaptive_patience.py train.py config.py

PYTHONPATH=. python3 Smoke_Tests/smoke_test_trace_splits.py
PYTHONPATH=. python3 Smoke_Tests/smoke_test_adaptive_patience.py
PYTHONPATH=. python3 Smoke_Tests/run_all_smoke_tests.py

# ASCII byte scan (handoff section 10)
python3 - data_splits.py adaptive_patience.py train.py config.py << 'EOF'
import sys
for path in sys.argv[1:]:
    data = open(path, 'rb').read()
    bad = [(i + 1, hex(b)) for i, b in enumerate(data) if b > 127]
    print(path, 'OK -- pure ASCII' if not bad else 'NON-ASCII: %s' % bad[:8])
    if bad:
        sys.exit(1)
EOF
```

`smoke_test_trace_splits.py`, 10 groups. [A] disjointness, [B] class coverage,
[C] requested counts including the floor-rule violation, [D] tiling against
`MEAWindowDataset` itself, [E] leave-one-out coverage, [F] `trace_of_window`,
[G] determinism, [H] guard rails, [I] apportionment invariants, [J] leakage
contrast against the old splitter.

[F] is worth singling out. Cultures are generated as constant-valued traces,
culture $u$ being the constant $u+1$, so `trace_of_window[i]` is checked against
the window's own SAMPLE VALUES, not only against a recomputation of the same
bookkeeping. A wrong local-to-global index map cannot pass it.

`smoke_test_adaptive_patience.py`, 7 groups. [A] backward compatibility,
[B] termination bound against simulation, [C] rejection of non-terminating
configurations, [D] budget accounting, [E] floor statistics, [F] threshold
resolution, [G] checkpoint round-trip.

---

## 8. Summary of results

| # | Statement | Established in |
|---|---|---|
| T1 | `make_trace_splits` assigns whole cultures, stratified by class, in `fractional` and `leave_one_out` modes, and exposes `trace_of_window`. | 2.1, 2.5 |
| T2 | Largest-remainder apportionment satisfies Eq. (1); the handoff's floor rule does not, at $n_c = 18$. | 2.3 |
| T3 | `min_train_cultures_per_class = 2` is forced by Change 4, not chosen. | 2.4 |
| T4 | Effective patience is $\lceil P_0/(1-g)\rceil$, Eq. (3), capped by $P_{\max}$. | 3.2 |
| T5 | $g \ge 1$ uncapped never terminates and is rejected at construction. | 3.2 |
| T6 | $g = 0$ reproduces the current rule exactly, over 200 random sequences. | 3.4 |
| T7 | $\hat{\mu}_{\mathrm{floor}} \approx 0$ and is typically negative, so $\delta_{\mathrm{sil}} = 2\hat{\mu}_{\mathrm{floor}}$ inverts early stopping; $\kappa\hat{\sigma}_{\mathrm{floor}}$ is used instead. | 4 |
| T8 | `smoke_test_selected_epoch.py` already fails under silhouette-primary, in the unmodified repository. | 6.2 |

---

## 9. Open points, caveats and assumptions

- **T8 is unresolved and blocks Change 3.** Root cause not isolated.
- **`patience_reset_on_improvement` is undecided** (section 5.1). Default is the
  conservative one; the recommendation is the other.
- **$\kappa = 2$ is inherited from the instruction, not measured.** With
  `floor_scale` it means "twice the null's standard deviation", which is a
  reasonable but arbitrary strictness. The floor now being measured and logged,
  it can be calibrated against real runs.
- **The floor is measured on the VALIDATION embedding at one epoch.** It
  characterises the null for that geometry. Whether it drifts materially as the
  embedding trains is unmeasured; `sil_floor` is persisted per run so this can
  be checked by re-measuring at the last epoch and comparing.
- **Trace-level versus window-level silhouette (handoff section 7.4) is
  untouched here.** `trace_of_window` is what remedy 1 needs, and it now exists,
  but no trace-level silhouette is computed.
- **Leave-one-out with unequal $n_c$** warns and uses $\min_c n_c$ folds;
  exact-once test coverage does not hold there.
- **No real MEA data was involved.** Every number in section 4 comes from
  synthetic embeddings; the real-data floor must be re-measured.

---

## 10. References

- **v3 handoff**, `HANDOFF_pipeline_changes_v3.md`, 28 July 2026. Sections 8
  (Change 5), 7 (Change 3), 9.2 (required assertions), 10 (ASCII), 12.3
  (silhouette reference levels unmeasured). The source for every specification
  followed or deviated from above; deviations are flagged in 2.1, 2.3 and 4.
- **The repository**, `dsn_wiring_C1_C6.zip`. Read directly. Source for every
  statement about current behaviour, including the baseline failure in 6.2,
  which was reproduced by running the suite against the unmodified snapshot.
- **Stated from this document's own reasoning, with no external source:** the
  apportionment inconsistency (2.3), the termination analysis (3.2), the
  derived-budget argument (3.3), the sign analysis of the permutation null (4),
  and the diagnosis in 6.2. The permutation-null bias argument is standard
  reasoning about order statistics applied to the silhouette's $b(i)$; it was
  not taken from a cited source, and the numbers in section 4 are measurements
  made here, not literature values.
- **Not searched:** PubMed and bioRxiv/medRxiv. No claim in this document rests
  on published empirical work; every quantitative statement is either an
  algebraic identity verified by test or a measurement made in the accompanying
  smoke tests.
