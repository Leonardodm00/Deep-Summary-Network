# Deployment pipeline: applying the C1-C6 wiring changes to the davinci-1 clone

**Date:** 25 July 2026
**Scope covered.** Moving the six wired changes (C1-C6) from this session's sandbox onto
the `davinci-1` HPC clone of Deep-Summary-Network; running Rungs 0-2 of the verification
ladder there, since this session could not run Rung 2 (no working `torch` on the sandbox
machine); and submitting the Rung-3 C6 ablation as a PBS job.
**Scope excluded.** Interpreting the ablation's scientific result beyond pointing at the
tool that computes it (a follow-on task); troubleshooting specific to a PBS/module
configuration this document has not seen; and any further change to the six modules
themselves, which are already delivered and described in `WIRING_REPORT.md`.

This is a procedural document, not a mathematical one. Section 1 fixes the meaning of
every placeholder, path, script and flag used below in place of mathematical symbols.

---

## 1. Notation and symbols

| Symbol | Name / meaning | Type & domain | Units | First used in |
|---|---|---|---|---|
| `<REPO_ROOT>` | root of your local clone of the repository | local absolute path | n/a | 3.2 |
| `<HPC_REPO_ROOT>` | root of the same repository on davinci-1, e.g. `/davinci-1/home/ldellamea/Deep Summary Network` | remote absolute path | n/a | 3.2 |
| `Main/` | the working directory inside the repository holding every pipeline module | path, fixed, relative to the repo root | n/a | 3.1 |
| `ALL_CHANGES.diff` | unified diff of the five edited modules, delivered this session | local file, `git apply`-compatible | n/a | 3.2 |
| `dsn_wiring_C1_C6.zip` | archive containing a complete replacement `Main/` | local file | n/a | 3.1 |
| `WIRING_REPORT.md` | this session's own verification report (referenced, not reproduced here) | local file | n/a | 3.1 |
| `meacnn` | the conda environment name pinned in `environment.yml`, `install_env.sh`, `verify_env.py` | environment identifier | n/a | 3.3 |
| `brian_env` | the environment name used in one documented PBS example; DISAGREES with `meacnn` (5.2) | environment identifier | n/a | 3.3 |
| `PBS_JOBID` | shell variable set only inside a PBS job; the login-node guard's test | shell environment variable | n/a | 3.3 |
| `qsub` | the PBS/Torque job-submission command | shell command | n/a | 3.3, 3.5 |
| Rung 0 / 1 / 2 / 3 | the four stages of the verification ladder (byte-scan+compile / torch-free module checks / torch acceptance gate / cluster-only real search), as defined in `WIRING_REPORT.md` section 4 | ordinal label | n/a | 3.4 |
| `run_wiring_checks.sh` | the script that runs Rungs 0-2 in one invocation | path, relative to `Main/` | n/a | 3.4 |
| `hpc/config_latent_3class_hard.json`, `hpc/config_latent_3class_easypos.json` | the two C6 ablation configs; differ only in `train.mining_strategy` | path, relative to `Main/`; JSON config | n/a | 3.5 |
| `<PBS_ncpus>`, `<PBS_walltime>` | resource-request placeholders in the job script, to be set from a timed dry run on the real node, never guessed | PBS resource-string fields | n/a | 3.5 |

### 1.1 Conventions

- `<angle brackets>` mark a placeholder the reader fills in with a site- or
  account-specific value. Nothing written this way is a literal string to paste.
- All shell commands assume a POSIX shell and are given relative to `Main/` unless an
  absolute path is shown.
- "This session" means the conversation in which C1-C6 were implemented and verified at
  Rung 0/1 only. "The target machine" and "davinci-1" both mean the machine where Rung 2
  and Rung 3 must actually run.
- File names without a path are relative to `Main/`.

---

## 2. Glossary

Ordered by first appearance.

**Verification ladder.** The four-rung sequence -- Rung 0 (ASCII byte-scan + compile),
Rung 1 (module checks, no torch needed), Rung 2 (the acceptance gate, needs torch), Rung 3
(a real cluster search) -- carried from the handoff into `WIRING_REPORT.md` section 4.

**Acceptance gate.** Rung 2 specifically: the point below which a change is verified
enough to trust, and above which only a real cluster job can add further confidence.

**Drift test.** `Smoke_Tests/smoke_test_selected_epoch.py`, the one Rung-2 check that runs
real (tiny) training and confirms the search's recomputed selected epoch agrees with
`train.py`'s own arithmetic. The check this document's author could not run.

**Login-node guard.** The `PBS_JOBID` check inside `install_env.sh` that warns (does not
hard-stop) against running a heavy install outside a job.

**Ablation (here).** The C6 comparison of `mining_strategy in {hard, easy_positive}`, with
every other field held fixed.

**Stale-cache refusal.** The `ValueError` `build_traces` raises when the on-disk cache
fingerprint disagrees with the one the current config computes -- the mechanism that makes
rerunning C1's latent generator with a different `class_overlap` safe rather than silently
wrong.

---

## 3. Main body

### 3.1 What you already have, and what it replaces

*Establishes: the three deliverables from this session, and that the zip alone is
sufficient -- nothing from the original `dsn_pipeline.zip` / `dsn_smoke_tests.zip` needs to
travel to davinci-1 separately.*

| Deliverable | Contents |
|---|---|
| `dsn_wiring_C1_C6.zip` | a **complete, ready-to-use `Main/`**: every original pipeline and smoke-test file, plus the five edited modules, plus every new file (`factor_retention.py`, four new smoke tests, two new `hpc/*.json` configs, `run_wiring_checks.sh`, `WIRING_REPORT.md`) |
| `ALL_CHANGES.diff` | a unified diff of the five edited modules only, for applying against an existing tracked clone instead of copying files wholesale |
| `WIRING_REPORT.md` | what changed, at which exact integration point, and what was and was not verified (its section 4.2 is required reading before 3.4 below) |

Because the zip already contains the merged result of `dsn_pipeline` + `dsn_smoke_tests` +
the handoff's three modules + this session's edits, it **supersedes** those two original
archives. You do not need to reconcile three sources by hand.

### 3.2 Getting the changes onto davinci-1

*Establishes: two supported routes, and the one hazard that applies to both.*

**The hazard, both routes.** Every file here was verified byte-for-byte pure ASCII
specifically because transfer through Windows-side tooling can silently re-encode it.
This is not a precaution invented for this task -- it is already documented as the
project's own compliance rule: *"transfer through Windows tooling (MobaXterm,
copy-paste, `scp` from a Windows box) can re-encode non-ASCII bytes, producing a
`SyntaxError` that surfaces only at job-submission time on the cluster."* Use `scp`,
`rsync`, or `git`; never a copy-paste through a terminal emulator.

**Option A -- git (recommended when your clone is a git checkout).** This session cannot
push to your remote, so these commands are yours to run:

```sh
# on your machine, inside <REPO_ROOT>
git checkout -b apply-c1-c6
git apply /path/to/ALL_CHANGES.diff        # the five edited modules
# then add the wholly new files by hand (git apply only touches tracked files):
#   Main/factor_retention.py
#   Main/Smoke_Tests/smoke_test_objective_wiring.py
#   Main/Smoke_Tests/smoke_test_factor_retention.py
#   Main/Smoke_Tests/smoke_test_latent_wiring.py
#   Main/Smoke_Tests/smoke_test_selected_epoch.py
#   Main/hpc/config_latent_3class_hard.json
#   Main/hpc/config_latent_3class_easypos.json
#   Main/run_wiring_checks.sh
#   Main/WIRING_REPORT.md
git add -A
git commit -m "Wire C1-C6: latent benchmark, adaptive tie-break, factor retention"
git push origin apply-c1-c6
```

Then on davinci-1, inside `<HPC_REPO_ROOT>`:

```sh
git fetch origin
git checkout apply-c1-c6      # or merge into your working branch
```

`git` transfers bytes exactly; it does not pass through anything that re-encodes text, so
this route sidesteps the hazard above by construction.

**Option B -- direct transfer (if your clone is not git-tracked, or you'd rather not
branch).** From the extracted `dsn_wiring_C1_C6.zip`:

```sh
rsync -av Main/ <user>@davinci-1:"<HPC_REPO_ROOT>/Main/"
```

`rsync -a` (or `scp` with `-p`) copies bytes verbatim; it is safe by the same logic as
`git`. After the transfer, re-run the byte scan **on davinci-1** rather than trusting the
local one, since the local scan only proves the files were clean before the trip:

```sh
cd "<HPC_REPO_ROOT>/Main"
python3 -c "
import pathlib, sys
bad = [f for f in list(pathlib.Path('.').glob('*.py')) + list(pathlib.Path('Smoke_Tests').glob('*.py'))
       if any(b > 127 for b in f.read_bytes())]
print('OK, pure ASCII' if not bad else 'CORRUPTED IN TRANSIT: %r' % bad)
sys.exit(1 if bad else 0)
"
```

### 3.3 Environment

*Establishes: which environment to activate, and a discrepancy in the project's own
documentation that you must resolve before trusting either name.*

The project ships `install_env.sh`, `environment.yml`, and `verify_env.py` in
`Main/`, all three agreeing on the environment name **`meacnn`**. One documented PBS
example (3.5) instead runs `source activate brian_env`. That is an unresolved
discrepancy in the project's own documentation, not something this session can settle.
Check what actually exists before submitting a real job:

```sh
conda env list
```

If `meacnn` exists, use it. If only `brian_env` exists, the PBS example is the authority
and `meacnn` was never created on this site. If neither exists, build it:

```sh
module load cuda/12.1          # match your node; check with: module avail cuda
module load miniconda3         # or the site's actual module name
qsub -I -l select=1:ncpus=4:ngpus=1 -l walltime=01:00:00   # get a compute node; installs on the login node may be killed or against policy
bash install_env.sh            # creates meacnn from environment.yml, then installs the matching CUDA torch wheel, then pytorch-metric-learning + scikit-optimize + optuna
```

Then, in every subsequent shell (interactive or inside a PBS script):

```sh
conda activate meacnn          # or whichever name 3.3 confirmed
python verify_env.py           # must print "N/N checks passed" with 0 failures
```

Do not proceed past a `verify_env.py` failure -- it is ordered from the most fundamental
dependency to the most specific, so the first failure tells you exactly which layer is
broken.

### 3.4 Running the verification ladder on davinci-1

*Establishes: this is the acceptance gate this session could not execute, and nothing in
section 3.5 should be attempted before it passes clean.*

Run this from an interactive compute node, not the login node -- Rung 2 performs real
(if tiny) training:

```sh
qsub -I -l select=1:ncpus=4:ngpus=1 -l walltime=00:30:00
cd "<HPC_REPO_ROOT>/Main"
conda activate meacnn
sh run_wiring_checks.sh
```

`run_wiring_checks.sh` runs, in order, and stops at the first failure:

1. **Rung 0** -- byte scan of every `.py`/`.json` file, then `py_compile` over the whole
   import chain.
2. **Rung 1** -- `smoke_test_latent_and_objective.py` (the handoff's own 21 checks),
   `smoke_test_objective_wiring.py` (C2/C3 math), `smoke_test_factor_retention.py` (C5,
   including the grouping test).
3. **Rung 2** -- every pre-existing smoke test in `Main/Smoke_Tests/` against **real**
   torch (this is the regression check: nothing here may newly fail), then
   `smoke_test_latent_wiring.py` (C1, re-run against real torch instead of the sandbox
   stub this session used), then **`smoke_test_selected_epoch.py`**, then
   `run_optimization.py --dry-run` on `hpc/config_latent_3class_hard.json`.

The whole ladder runs in well under the half-hour requested above, since every Rung-2
check uses toy-sized data. If `smoke_test_selected_epoch.py` fails, stop: it means the
recomputed selected epoch has drifted from `train.py`'s own rule, which the search
objective (C2) silently depends on, and no result downstream of it can be trusted until
fixed.

### 3.5 Submitting the C6 ablation (Rung 3)

*Establishes: the actual PBS submission for the two ablation runs, adapted from the
project's own documented template, and the checks to run before submitting either.*

**Before you submit anything** (the project's own checklist, unchanged):

1. `--dry-run` first and read the reported `TOTAL_train_runs`.
2. Time one real trial by hand and multiply by that count; compare against your intended
   walltime.
3. Run `smoke_test_end_to_end.py` on the login node -- five minutes, and it exercises
   every code path the job will hit.
4. Compare the pre-flight's reported max model size / RAM against the node's actual RAM.
   A soft OOM is caught and scored as a failed trial; a hard Linux OOM-kill takes the
   whole study with it, silently, with no traceback.

**Two independent jobs**, one per miner, so they can run in parallel rather than
sequentially:

```bash
#!/bin/bash
#PBS -N dsn_latent_hard
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=<PBS_walltime>
#PBS -o dsn_latent_hard.out
#PBS -e dsn_latent_hard.err

cd $PBS_O_WORKDIR
conda activate meacnn

python3 run_optimization.py \
    --config hpc/config_latent_3class_hard.json \
    --verbose
```

```bash
#!/bin/bash
#PBS -N dsn_latent_easypos
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=<PBS_walltime>
#PBS -o dsn_latent_easypos.out
#PBS -e dsn_latent_easypos.err

cd $PBS_O_WORKDIR
conda activate meacnn

python3 run_optimization.py \
    --config hpc/config_latent_3class_easypos.json \
    --verbose
```

Two things to decide deliberately, not by accident:

- **Device.** Both delivered configs carry `runtime.device = "cpu"`, inherited unchanged
  from the archived baseline. The template above requests a GPU (`ngpus=1`) but never
  passes `--device`, so it will run on CPU regardless of what PBS allocates. If you want
  GPU training, either edit `runtime.device` to `"cuda"` in the JSON, or add
  `--device cuda` to the command line (it overrides the JSON).
- **`out_dir` / `cache_dir`.** Both configs already point at
  `<HPC_REPO_ROOT>/out` and `<HPC_REPO_ROOT>/cache_latent` respectively, with distinct
  `experiment_name`s (`latent_3class_hard`, `latent_3class_easypos`), so the two runs
  will not collide on disk even sharing one `out_dir`. `cache_latent` is deliberately
  separate from whatever `cache/` your archived synthetic-mode run used.

`--resume` only applies to the final training stage, not the search phases: a search
killed mid-way must restart, but the trace cache survives (it is fingerprinted, C1
section 2.1), so restarting does not repeat trace synthesis.

The archived comparable run (39 CPU-hours, 242 trials, on the old 1-D benchmark) is the
only timing reference available; the new benchmark's per-trial cost has not been
measured by anyone yet, so treat that number as a floor, not an estimate, and let step 2
of the pre-submission checklist set `<PBS_walltime>` and `<PBS_ncpus>` for real.

### 3.6 Reading the results

*Establishes: where to look once both jobs complete, as a pointer rather than a full
analysis protocol -- interpreting the outcome is the next task, not this one.*

Each run writes `out/<experiment_name>/results.json`. The block that matters:

```json
"test": {
  "ari":        {"mean": ..., "std": ..., "values": [...]},
  "eff_rank":   {"mean": ..., "std": ..., "values": [...]},
  "per_seed":   [{"seed": 0, "epochs_run": ..., "best_val_ari": ..., ...}]
}
```

Report `mean +/- std` over seeds; a difference between the two ablation runs smaller than
either's `std` is not a difference. Check `per_seed[*].epochs_run` against
`train.max_epochs` first -- if it equals the ceiling for most seeds, the runs never
converged and the comparison is measuring training speed, not representation quality.

The decisive comparison the handoff's C6 section asked for is not `ari` (both miners are
expected to tie there) but the **factor-retention** numbers, which are not written to
`results.json` automatically -- run `factor_retention.py` on each run's saved
`latent_ground_truth.json` and held-out embeddings, exactly as
`Smoke_Tests/smoke_test_factor_retention.py` section G demonstrates end to end.

### 3.7 If something goes wrong

*Establishes: a first-response table, combining the project's own documented failure
modes with the ones specific to C1-C6.*

| Symptom | Likely cause | Where to look |
|---|---|---|
| `Killed`, no traceback | Linux OOM-killer | narrow `depth_exponent_range`, per the pre-flight's reported model sizes |
| `SyntaxError` on a non-ASCII byte | re-encoded in transfer | re-transfer in binary mode (3.2); every file here was verified pure ASCII before it left this session |
| Hangs, no output | should be impossible; check `matplotlib` import order | `evaluate.py` forces the Agg backend before `pyplot` is imported anywhere |
| `ValueError: infeasible window / split geometry` | the pre-flight is working correctly | it names the fix directly in the message |
| `STALE TRACE CACHE` | a latent parameter changed but the cache dir did not | this is the C1 fingerprint refusing to silently reuse old traces (WIRING_REPORT.md 4.1); pass `--overwrite-cache` or point at a fresh `cache_dir` |
| results look impossibly unchanged after editing `class_overlap` or `label_axes` | you edited the JSON but pointed at an old `cache_dir` | confirm the fingerprint check actually fired; it should have raised, not stayed silent |
| the search seems to score a different epoch than `train.py`'s own log | C2's recomputed $e^\star$ has drifted from `train.py`'s rule | `smoke_test_selected_epoch.py` (3.4) exists to catch this before a real job does |
| unexpectedly low or high $R^2_k$ in a C5 result | check `n_c` (traces per class) before drawing a conclusion | with $n_c = 3$ the free-axis effective sample size is 9, not $N_{\mathrm{eval}}$ (WIRING_REPORT.md 6.10) |
| a range you set in `config_input.json` doesn't seem to take effect | `config.py` defaults vs the JSON file diverge | always check which one is actually in force before concluding a range is wrong |

---

## 4. Summary of results

The full command sequence, in the order sections 3.2-3.5 explain it:

```sh
# 3.2 -- transfer (pick ONE route)
git apply ALL_CHANGES.diff && git add -A && git commit -m "..." && git push   # Option A
# or
rsync -av Main/ <user>@davinci-1:"<HPC_REPO_ROOT>/Main/"                      # Option B

# 3.3 -- environment
ssh <user>@davinci-1
conda env list                                  # confirm meacnn vs brian_env
conda activate meacnn
python verify_env.py                            # must be 0 failures

# 3.4 -- acceptance gate (interactive compute node, NOT the login node)
qsub -I -l select=1:ncpus=4:ngpus=1 -l walltime=00:30:00
cd "<HPC_REPO_ROOT>/Main"
sh run_wiring_checks.sh                         # stop here if anything fails

# 3.5 -- Rung 3 (only after 3.4 is fully green)
python3 run_optimization.py --config hpc/config_latent_3class_hard.json --dry-run
qsub dsn_latent_hard.pbs
qsub dsn_latent_easypos.pbs

# 3.6 -- results
cat out/latent_3class_hard/results.json
cat out/latent_3class_easypos/results.json
```

---

## 5. Open points, caveats, and assumptions

**Assumed without verification here:**

1. This session has no push access to the group's git remote; section 3.2 Option A's
   commands are written for the reader to execute, not something performed here.
2. The environment-name discrepancy (`meacnn` vs `brian_env`) is unresolved in the
   project's own documentation. Section 3.3's `conda env list` check is the only
   safeguard; do not submit a job assuming one or the other without running it.
3. The delivered HPC configs default to `runtime.device = "cpu"`, unchanged from the
   archived baseline. Whether that is what you want on the requested GPU node is a
   decision this document raises but does not make.
4. `<PBS_walltime>` and `<PBS_ncpus>` are left as placeholders. Filling them from a
   guess rather than from step 2 of the pre-submission checklist (3.5) risks either a
   wasted allocation or a killed job with no partial result.

**Carried forward from `WIRING_REPORT.md`, unchanged:**

5. Rung 2 has not been executed by anyone against real torch as of this document. In
   particular, `smoke_test_selected_epoch.py` -- the check that matters most for trusting
   C2 -- has never run. Section 3.4 is where that finally happens.
6. Nothing in this document, or the session before it, has measured the network's actual
   performance on the new latent benchmark. $0.3006$ (the rich hand-crafted baseline) is
   a floor, not a target; section 3.6 points at where that measurement will eventually
   live, not at a result.

---

## 6. References

**This session's own deliverables (25 July 2026):** `WIRING_REPORT.md`,
`ALL_CHANGES.diff`, `dsn_wiring_C1_C6.zip`.

**Prior session:** `HANDOFF_wiring_changes.md` (25 July 2026).

**Project knowledge base, read in full this turn:**
`Patched/Augmentation/Installation/install_env.sh`,
`Patched/Augmentation/Installation/environment.yml`,
`Patched/Augmentation/Installation/verify_env.py`,
`Patched/Optimization/03_USAGE.md` (sections 5-7),
`Patched/Optimization/02_TECHNICAL.md` (sections 13-14).
Every command and path claim above that is not marked with a `<placeholder>` was checked
against one of these files rather than assumed; the one place they disagree with each
other (`meacnn` vs `brian_env`) is flagged rather than silently resolved (5.2).

**No literature consulted.** This document contains no empirical or biomedical claim, so
neither PubMed nor bioRxiv/medRxiv was queried for it.
