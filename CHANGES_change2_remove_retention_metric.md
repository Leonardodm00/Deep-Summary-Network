# Change 2 -- removal of the latent-factor retention metric (C5)

**Date:** 28 July 2026
**Target:** `Deep-Summary-Network`, branch `change5-trace-splits-growing-patience`,
package root `Main/`
**Specification:** `HANDOFF_changes_2_3_4_1.md` section 8 (the deletion list) and
section 8.1 (the required assertions); ordering from section 4, step 1 of
2 -> 3 -> 4 -> 1.
**Supersedes for this change:** `HANDOFF_pipeline_changes_v3.md` section 2.

**Abstract.** The C5 latent-factor retention metric, added during the C1-C6
wiring on 25 July 2026, is deleted. The scientific question it answered -- how
much LABEL-IRRELEVANT latent structure survives in the learned embedding -- is
not withdrawn, and the artefact it needed is still written on every latent run;
what is withdrawn is the claim that this repository computes it. The metric had
no production consumer three days after it was built, was called from no module,
and was reachable only by hand from a finished run's embeddings, so carrying it
further would have meant maintaining, porting and re-verifying a module that
nothing depended on. **Covered:** the exact deletion list and the two deviations
from the handoff's version of it; what was deliberately kept; the preserved
specification and measured record of the deleted metric, since this document is
the only place that record now lives; the new deletion guard and its negative
controls; and the verification actually performed versus the verification still
owed. **Deliberately excluded:** Changes 3, 4 and 1, which follow this one and
are untouched here; any change to the backbone, loss, optimiser, miner, sampler
or search algorithm; and any decision about whether a retention measurement
should be reimplemented later, which section 9 leaves open.

---

## 1. Notation and symbols

Every symbol below appears only inside section 5, which preserves the deleted
metric's definition; no equation elsewhere in this document uses notation.

| Symbol | Name / meaning | Type and domain | Units | First used in |
|---|---|---|---|---|
| $n$ | number of latent factors of the generator | $n \in \mathbb{N}$, $n \ge 1$ | count | 5.1 |
| $k$ | latent-axis index | $k \in \{1,\dots,n\}$ in mathematics, 0-based in code | dimensionless | 5.1 |
| $S$ | label-carrying axis subset | $S \subseteq \{1,\dots,n\}$, $S \ne \emptyset$ | -- | 5.1 |
| $S^{c}$ | label-IRRELEVANT ("free") axes | $S^{c} = \{1,\dots,n\} \setminus S$ | -- | 5.1 |
| $N_{\mathrm{eval}}$ | number of evaluation windows | $N_{\mathrm{eval}} \in \mathbb{N}$, $N_{\mathrm{eval}} \ge 2$ | count | 5.1 |
| $i$ | evaluation-window index | $i \in \{0,\dots,N_{\mathrm{eval}}-1\}$ | dimensionless | 5.1 |
| $E$ | embedding dimension | $E \in \mathbb{N}$ | count | 5.1 |
| $Z$ | held-out embedding matrix, rows L2-normalised | $Z \in \mathbb{R}^{N_{\mathrm{eval}} \times E}$ | dimensionless | 5.1 |
| $z_i$ | row $i$ of $Z$ | $z_i \in \mathbb{R}^{E}$ | dimensionless | 5.1 |
| $\phi_k^{(i)}$ | TRUE $k$-th latent coordinate of the trace window $i$ was cut from | $\phi_k^{(i)} \in [0,1]$ | dimensionless | 5.1 |
| $\hat{\phi}_k^{(i)}(Z)$ | out-of-fold ridge prediction of $\phi_k^{(i)}$ from $z_i$ | $\hat{\phi}_k^{(i)}(Z) \in \mathbb{R}$ | dimensionless | 5.1 |
| $\bar{\phi}_k$ | mean of $\phi_k$ over the evaluation windows, $\bar{\phi}_k = N_{\mathrm{eval}}^{-1}\sum_{i} \phi_k^{(i)}$ | $\bar{\phi}_k \in [0,1]$ | dimensionless | 5.1 |
| $R^{2}_{k}$ | retention score of axis $k$ | $R^{2}_{k} \in (-\infty, 1]$ | dimensionless | 5.1 |
| $g_i$ | GROUP of window $i$: index of the trace it was cut from | $g_i \in \{0,\dots,n_{\mathrm{traces}}-1\}$ | dimensionless | 5.2 |
| $n_{\mathrm{traces}}$ | number of traces (cultures) in the evaluation split | $n_{\mathrm{traces}} \in \mathbb{N}$ | count | 5.2 |
| $n_c$ | traces per class | $n_c \in \mathbb{N}$ | count | 5.3 |
| $C$ | number of condition classes | $C \in \mathbb{N}$, $C \ge 2$ | count | 5.3 |

### 1.1 Conventions

- Mathematical indices are 1-based where stated and 0-based in code; every
  occurrence above says which. $i$ is 0-based in both.
- Both sums in Eq. (1) run over the SAME index set,
  $i \in \{0,\dots,N_{\mathrm{eval}}-1\}$, and the denominator is taken about the
  GLOBAL mean $\bar{\phi}_k$, not about a per-fold mean. This is stated again at
  the equation because it is the one thing about Eq. (1) that is easy to get
  wrong.
- "C5" names the fifth change of the earlier C1-C6 wiring effort
  (`HANDOFF_wiring_changes.md`, 25 July 2026). "Change 2" names the second
  change of the later v3 pipeline plan. They are different numbering schemes for
  different documents; C5 is the thing deleted, Change 2 is the act of deleting
  it.
- File paths are relative to the repository root unless prefixed `Main/`.
- All Python written for this change is pure 7-bit ASCII, LF-terminated.

---

## 2. Glossary

Ordered alphabetically.

**Deletion guard.** A test whose subject is an absence: it asserts that a
removed file, symbol or reference has not come back. Operative in section 6.

**Free axis.** A latent axis $k \in S^{c}$ that carries no class information by
construction. The retention metric's whole point was to ask what happens to
these. Operative in section 5.

**Grouped cross-validation.** Cross-validation whose folds never split a group
across the train/test boundary. Here the group was the trace $g_i$, never the
window index $i$: two windows cut from the same trace share the same
$\phi_k$ exactly, so an ungrouped split lets the model memorise a value it will
then be tested on. Operative in section 5.2.

**Latent ground truth.** The table `latent_ground_truth.json`, written once per
latent run, recording the true latent coordinates $\phi_k$ of every generated
trace. KEPT by this change. Operative in section 4.

**Production consumer.** A caller reached by an ordinary run of the pipeline --
`run_optimization.py`, `train.py`, `evaluate.py`, `search.py` -- as opposed to a
module a human can import by hand. The metric had none, which is the entire
basis for deleting it. Operative in section 3.2.

**Retention (of a latent factor).** The degree to which a label-irrelevant
latent factor remains linearly decodable from the learned embedding, quantified
by $R^{2}_{k}$. Operative in section 5.

---

## 3. What was deleted, and on what evidence

**This section establishes** the deletion list actually applied and the check
that made it safe.

### 3.1 Files removed

| Path | Lines |
|---|---|
| `Main/factor_retention.py` | 415 |
| `Main/Smoke_Tests/smoke_test_factor_retention.py` | 302 |

717 lines total.

### 3.2 The safety check

Before deleting, the tree was scanned for importers. The module was imported
from exactly one place, its own smoke test, which is deleted with it. No
pipeline module, no configuration file and no PBS script referenced the symbol.
The handoff's section 8 states the same conclusion; it was re-verified here
rather than inherited, and it held.

### 3.3 References cleaned

Removing the module leaves prose and scripts pointing at a file that no longer
exists. Every such reference under `Main/` was resolved:

| Path | What it was | What was done |
|---|---|---|
| `Main/run_wiring_checks.sh` | Rung 1 EXECUTED the deleted suite, under `set -e` | invocation replaced by the new deletion guard; the C5 rung is explained in a header note; two downstream mentions ("the C5 ground truth", "the C5 $R^{2}_{k}$") reworded |
| `Main/DEPLOYMENT_PIPELINE.md` | told the reader to run the deleted module on a finished run (3.7), listed the deleted suite in Rung 1, and had a troubleshooting row for a metric that no longer exists | 3.7 rewritten to say what is and is not now available; Rung 1 entry replaced; troubleshooting row removed |
| `Main/WIRING_REPORT.md` | the dated record of building and measuring C5 | **annotated, not amended** -- see section 4.2 |
| `Main/run_optimization.py` | `save_latent_artifacts` justified the artefact by the deleted metric | re-justified: the artefact is the run's provenance record and is kept deliberately |
| `Main/train.py` | miner comment referred to "the C5 metric" | reworded; the scientific point (easy positives avoid collapse pressure on label-irrelevant factors) is unchanged |
| `Main/Smoke_Tests/smoke_test_inspect_latent.py` | four references to "the C5 ground truth" and to the analysis "the table will use" | reworded to name the artefact, `latent_ground_truth.json`, which still exists |
| `Main/hpc/dsn_4c_tracesplit_diag.pbs` | status header listed Change 2 as unimplemented | status line updated |

---

## 4. What was kept, and why

**This section establishes** that two things which look like C5 residue are not,
and must not be swept up by a later tidy.

### 4.1 `latent_ground_truth.json` is still written

Per the handoff's deletion list. `save_latent_artifacts` in
`Main/run_optimization.py` still writes the table on every latent run, including
`--dry-run`, and `Main/run_wiring_checks.sh` still checks that a dry run produced
it. The justification changed and is now recorded in the function's own
docstring: the table is the only record of what the synthetic benchmark actually
contained, so a run without it cannot be re-analysed against its own ground
truth after the fact. Its cost is one small JSON per run.

`Main/Smoke_Tests/smoke_test_inspect_latent.py` check [D], which asserts that the
inspected coordinates match this table exactly, is therefore still live and still
load-bearing.

### 4.2 `WIRING_REPORT.md` was annotated, not rewritten

That report is dated 25 July 2026 and marks its claims `[MEASURED-HERE]` or
`[MEASURED-PRIOR]`. Its C5 material is a record of measurements that were really
made. Editing them out would silently rewrite the provenance of a session that
happened, so instead the report carries a status banner in its header, a
`DELETED in Change 2` banner on section 2.5, and a dated annotation on the
section 6 item that had already observed C5 had no caller. Only the two literal
import paths were replaced, so that no reader is sent to a file that is gone.

**This is a judgement, and it is reversible.** The alternative -- excising the
C5 subsection and the two C5 rows of the verification table outright -- is
defensible too, on the grounds that a wiring report should describe the wiring
that exists. It was not chosen because the measured numbers are not recoverable
by re-running anything: the code that produced them is deleted.

---

## 5. Preserved record: what the deleted metric computed

**This section establishes** the specification of the deleted module, because
after this commit no other file in the repository contains it. Everything here is
transcribed from `Main/factor_retention.py` and `Main/WIRING_REPORT.md`
section 2.5 as they stood at commit time; nothing is reconstructed from memory.

### 5.1 The score

For each fixed axis $k \in \{1,\dots,n\}$, with predictions pooled across folds
and the denominator taken about the global mean $\bar{\phi}_k$,

$$
R^{2}_{k}
= 1 - \frac{\sum_{i} \bigl(\phi_k^{(i)} - \hat{\phi}_k^{(i)}(Z)\bigr)^{2}}
{\sum_{i} \bigl(\phi_k^{(i)} - \bar{\phi}_k\bigr)^{2}},
\qquad
i \in \{0,\dots,N_{\mathrm{eval}}-1\} \text{ in BOTH sums}.
\tag{1}
$$

Eq. (1) is *not* the average of per-fold $R^{2}$ values, which would use a
different mean in each denominator.

The question Eq. (1) makes decidable, for each fixed $k \in S^{c}$: is the
label-irrelevant coordinate $\phi_k$ still linearly decodable from the
embedding? Two miners can reach the same ARI on the labels while differing in
what they DESTROY, and under this benchmark's latent construction, pulling every
positive of a class to a single point is an instruction to erase $\phi_k$ for
every $k \in S^{c}$. This is also why $\mathrm{eff\_rank}(Z)$ was judged
insufficient: on a benchmark whose latent manifold is one-dimensional by
construction, $\mathrm{eff\_rank} \approx 1$ is simultaneously the correct answer
and the signature of collapse, so the two hypotheses predict the same number.

### 5.2 Estimation

- Cross-validation: `GroupKFold` with groups $g_i$ = the TRACE index, never the
  window index $i$.
- Ridge penalty: selected by `RidgeCV` INSIDE each training fold.
- A constant target yields `NaN`, with a warning, rather than a spurious $1.0$.

### 5.3 What had been measured, as of 25 July 2026

Recorded in `Main/WIRING_REPORT.md` section 4.1 and marked `[MEASURED-HERE]` in
that document. These are transcribed, not re-measured -- the code that produced
them no longer exists, so they cannot be reproduced from this repository:

| Quantity | Value |
|---|---|
| recovery case: $R^{2}_{k}$ against a decodable target | $\approx 1.0000$ |
| null case: $R^{2}_{k}$ against a shuffled target | all $< 0$ |
| grouping is load-bearing: grouped versus ungrouped $R^{2}$ | $-0.3023$ versus $+1.0000$ |

The third row is the one worth carrying forward: without grouping by trace the
metric reports near-perfect retention on data where the grouped estimate says
there is none.

**Power caveat, from the same report.** With $n_c = 3$ traces per class the free
-axis coordinates take only $C \cdot n_c = 9$ distinct values, so the effective
sample size of Eq. (1) is closer to $9$ than to $N_{\mathrm{eval}}$. `GroupKFold`
removes the BIAS; it cannot manufacture POWER. Any reimplementation should raise
$n_c$ before reading much into a single $R^{2}_{k}$.

---

## 6. The deletion guard

**This section establishes** what now prevents the deletion from being silently
undone.

New file: `Main/Smoke_Tests/smoke_test_removed_modules.py`, added as the FIRST
entry of `ORDER` in `Main/Smoke_Tests/run_all_smoke_tests.py` because it needs no
torch, no data and no import from the package, and runs in milliseconds. Three
assertion groups:

- **[A]** the module and its suite are absent from disk, no compiled `.pyc` of
  the module survives in `__pycache__`, and the module is not importable with
  `Main/` on `sys.path`;
- **[B]** `run_all_smoke_tests.py --list` exits 0, names neither forbidden token,
  and lists a non-empty suite set (so [B] cannot pass vacuously against a broken
  runner);
- **[C]** neither forbidden token occurs in any text file under `Main/`, nor in
  the member list of any `.zip` under `Main/`.

Group [C] covers the handoff's assertion [B]. The zip-membership scan is a
deliberate strengthening: `Main/hpc/dsn_pipeline.zip` and
`Main/Colab_zips/dsn_pipeline.zip` are the payloads copied to the cluster and to
Colab, a text grep cannot see inside them, and a stale archive would redeploy the
deleted file. Both archives were checked and are clean.

The suite must scan for two tokens without containing them, or it would fail
against itself; they are therefore assembled from fragments at run time. That is
documented in the file and must not be "tidied" into literals.

**Negative controls.** A guard that only ever passes proves nothing, so each
group was made to fail on purpose, in isolation, from a clean copy of the tree:

| Control | Injected fault | Groups that failed |
|---|---|---|
| NC1 | `factor` module restored from git | A, C |
| NC2 | source absent, stale `.pyc` only | A only |
| NC3 | deleted suite restored and re-listed in `ORDER` | A, B, C |
| NC4 | one dangling reference in one markdown file | C only, naming the file and line |
| NC5 | token present ONLY as a member of a deployment zip | C only |
| NC6 | an importable shadow copy elsewhere on `PYTHONPATH` | A only, reporting the shadowing origin |

Each failure named the injected fault and no other group failed spuriously.

---

## 7. Verification performed

**This section establishes** exactly which gates were run and which were not, so
that nothing is assumed green.

Run and passing, in the preparation environment:

- byte scan for non-ASCII over every changed `.py`: clean;
- `python3 -m py_compile` over every changed `.py`: clean;
- LF line endings on every changed file: verified;
- `run_all_smoke_tests.py --list`: exits 0, 19 suites listed, deleted suite
  absent;
- `smoke_test_removed_modules.py`: 3/3 groups PASS, plus the six negative
  controls of section 6.

**NOT run here, and owed before the commit lands:** the full suite,
`PYTHONPATH=. python3 Smoke_Tests/run_all_smoke_tests.py`, which is the handoff's
gate in its section 10.1. The preparation environment has no `torch`, no
`pytorch_metric_learning` and no `optuna`, so the sixteen suites that import them
could not be executed. Nothing in this change touches a code path any of them
exercises -- the only executable edits are one comment in `train.py`, one
docstring in `run_optimization.py`, four strings in
`smoke_test_inspect_latent.py`, and the runner's `ORDER` list -- but "should not
regress" is not the same statement as "did not regress", and only the full run
settles it.

---

## 8. Summary of results

| # | Statement | Established in |
|---|---|---|
| **C2-1** | The module was imported nowhere but its own suite; the deletion cannot break a caller. Re-verified, not inherited. | 3.2 |
| **C2-2** | `Main/run_wiring_checks.sh` EXECUTED the deleted suite under `set -e`. It is not in the handoff's deletion list; had it been missed, the wiring ladder would have aborted at Rung 1. | 3.3 |
| **C2-3** | `run_all_smoke_tests.py` contained no entry for the deleted suite, in `ORDER` or in the description map, so the handoff's third deletion item was already satisfied and no removal was needed there. | 9, R1 |
| **C2-4** | `latent_ground_truth.json` is still written, and its justification is now recorded in the writing function rather than in the deleted consumer. | 4.1 |
| **C2-5** | The specification and the measured record of the deleted metric survive in section 5 of this document, which is the carve-out the handoff's assertion [B] provides for. | 5 |
| **C2-6** | The deletion guard fails, in isolation, on each of six distinct ways of undoing the deletion, including two a plain text grep cannot detect. | 6 |
| **C2-7** | Every gate that does not require torch passes; the full-suite gate is owed. | 7 |

---

## 9. Open points, caveats and assumptions

**R1 -- the handoff's deletion list was wrong in both directions, mildly.** It
directed a removal from `run_all_smoke_tests.py` that had nothing to remove, and
omitted `run_wiring_checks.sh`, which really did call the deleted suite. Both are
recorded above. Neither is consequential on its own; together they are a reason
to treat the remaining three changes' file lists as starting points to verify
rather than as inventories to apply.

**R2 -- section 4.2 is a judgement call, and it is reversible.** Annotating
`WIRING_REPORT.md` rather than excising its C5 material preserves measurements
that cannot be reproduced, at the cost of a report that documents a module the
repository no longer has. If the opposite preference wins, the excision is a
small, mechanical follow-up commit; this document already carries the content
that would be lost.

**R3 -- the numbers in section 5.3 are transcribed, not verified here.** They
are `[MEASURED-HERE]` claims of `WIRING_REPORT.md` (25 July 2026), and the code
that produced them is now deleted, so they can no longer be reproduced from this
repository. They are recorded so that a reimplementation has something to check
itself against, and should be treated as a prior session's measurement, not as
this document's.

**R4 -- the scientific question is not settled, only unimplemented.** Whether
easy-positive mining preserves label-irrelevant latent structure better than hard
mining is exactly the comparison C5 existed to make, and after this change
nothing in the repository answers it. The C6 ablation can still be read through
`ari` and `eff_rank`, with the caveat recorded in section 5.1 that
$\mathrm{eff\_rank}$ cannot distinguish correct low rank from collapse on this
benchmark. If that comparison is wanted later, section 5 is the specification to
reimplement, ideally as a consumer-facing metric with a caller rather than as a
module to be run by hand.

**R5 -- Change 4 will move the batch content, and the interaction is untested.**
Cross-culture positives change which windows share a group; a reimplemented
retention metric would need $g_i$ from `trace_of_window`, which now exists but
was not what the deleted module was written against.

**R6 -- no real MEA data was involved.** Everything above concerns the
four-class and three-class synthetic benchmarks.

---

## 10. References and provenance

- **Specification followed.** `HANDOFF_changes_2_3_4_1.md`, 28 July 2026:
  section 4 (ordering: Change 2 first, zero dependencies), section 8 (the
  deletion list), section 8.1 (assertions [A] and [B]), section 10.1 (the commit
  gate). Deviations from its deletion list are flagged in 3.3 and R1.
- **Repository, read directly.** The branch as delivered. Source for every
  statement in sections 3, 4 and 7, including the importer scan, the line counts,
  the contents of `run_wiring_checks.sh`, and the absence of an entry in
  `run_all_smoke_tests.py`.
- **Transcribed from the repository's own prior documentation.**
  `Main/WIRING_REPORT.md`, 25 July 2026, sections 2.5, 4.1 and 6 item 10 --
  the source for all of section 5, including the three measured values in 5.3;
  and `Main/factor_retention.py`'s module docstring, read before deletion, for
  the motivation restated in 5.1.
- **Verified by execution here.** Every item listed as run in section 7, and the
  six negative controls in section 6.
- **Stated from reasoning, with no external source.** The judgement in 4.2 and
  its reversal cost; the argument in R4 that the question is unimplemented rather
  than settled; and the choice to scan zip members, which follows from the
  archives being the deployment path rather than from any measurement.
- **Not searched.** PubMed and bioRxiv/medRxiv. No claim in this document rests
  on published empirical work: every statement is either a fact about this
  repository's files, a transcription of its own prior measurement record, or an
  explicitly labelled judgement.
