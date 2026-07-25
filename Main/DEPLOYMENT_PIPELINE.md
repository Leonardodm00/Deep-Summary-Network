# Deployment pipeline: applying the C1-C6 wiring changes to the davinci-1 clone

**Date:** 25 July 2026
**Scope covered.** A git-mediated pipeline on your own computer for merging the C1-C6
changes into your tracked clone and pushing them, followed by a pipeline for getting the
resulting files onto `davinci-1`. The two machines have different capabilities and are
treated as such throughout: git is assumed present and usable on your computer; **git is
assumed ABSENT on davinci-1**, so every davinci-1-side step is a download or a copy, never
a `git clone` / `git pull`. Also covers running Rungs 0-2 of the verification ladder on
davinci-1 (this session could not, having no working `torch`) and submitting the Rung-3
C6 ablation as a PBS job.
**Scope excluded.** Interpreting the ablation's scientific result beyond pointing at the
tool that computes it; troubleshooting specific to a PBS/module configuration this
document has not seen; any further change to the six modules themselves, already
delivered and described in `WIRING_REPORT.md`.

This is a procedural document, not a mathematical one. Section 1 fixes the meaning of
every placeholder, path, script and flag used below in place of mathematical symbols.

---

## 1. Notation and symbols

| Symbol | Name / meaning | Type & domain | Units | First used in |
|---|---|---|---|---|
| `<REPO_ROOT>` | root of your local, git-tracked clone | local absolute path | n/a | 3.1 |
| `<HPC_REPO_ROOT>` | root of the corresponding folder on davinci-1, e.g. `/davinci-1/home/ldellamea/Deep Summary Network` (NOT a git clone there) | remote absolute path | n/a | 3.1 |
| `Main/` | the working directory holding every pipeline module, on both machines | path, fixed, relative to the repo root | n/a | 3.1 |
| `dsn_wiring_C1_C6.zip` | archive containing a complete replacement `Main/`, delivered this session | local file | n/a | 3.1 |
| `ALL_CHANGES.diff` | unified diff of the five edited modules, delivered this session; optional, for review only (3.1) | local file | n/a | 3.1 |
| `WIRING_REPORT.md` | this session's own verification report (referenced, not reproduced here) | local file | n/a | 3.1 |
| `<GITHUB_URL>` | the repository's remote, `https://github.com/Leonardodm00/Deep-Summary-Network` per the prior session's handoff -- substitute your own fork/mirror if different | remote URL string | n/a | 3.1 |
| `<branch>` | the git branch you push the changes to | git ref | n/a | 3.1 |
| `<GITHUB_TOKEN>` | a GitHub personal access token; needed ONLY if `<GITHUB_URL>` is a private repository and you use the download-by-URL route (3.2, option B) | credential string; never logged, committed, or pasted into a job script | n/a | 3.2 |
| `meacnn` | the conda environment name pinned in `environment.yml`, `install_env.sh`, `verify_env.py` | environment identifier | n/a | 3.3 |
| `brian_env` | the environment name used in one documented PBS example; DISAGREES with `meacnn` (5.2) | environment identifier | n/a | 3.3 |
| `PBS_JOBID` | shell variable set only inside a PBS job; the login-node guard's test | shell environment variable | n/a | 3.3 |
| `qsub` | the PBS/Torque job-submission command | shell command | n/a | 3.3, 3.5 |
| Rung 0 / 1 / 2 / 3 | the four stages of the verification ladder, as defined in `WIRING_REPORT.md` section 4 | ordinal label | n/a | 3.4 |
| `run_wiring_checks.sh` | the script that runs Rungs 0-2 in one invocation | path, relative to `Main/` | n/a | 3.4 |
| `hpc/config_latent_3class_hard.json`, `hpc/config_latent_3class_easypos.json` | the two C6 ablation configs; differ only in `train.mining_strategy` | path, relative to `Main/`; JSON config | n/a | 3.5 |
| `<PBS_ncpus>`, `<PBS_walltime>` | resource-request placeholders in the job script, set from a timed dry run, never guessed | PBS resource-string fields | n/a | 3.5 |

### 1.1 Conventions

- `<angle brackets>` mark a placeholder the reader fills in with a site- or
  account-specific value. Nothing written this way is a literal string to paste.
- **Two machines, two capability sets, stated once here rather than re-qualified at every
  command:** your computer has `git`; davinci-1 does not. A command shown "on your
  computer" assumes git; a command shown "on davinci-1" never invokes it.
- All shell commands are given relative to `Main/` unless an absolute path is shown.
- "This session" means the conversation in which C1-C6 were implemented and verified at
  Rung 0/1 only. "The target machine" and "davinci-1" both mean the machine where Rung 2
  and Rung 3 must actually run.

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

**Archive snapshot.** A `.zip` or `.tar.gz` of a git branch's contents at push time,
servable over plain HTTPS by GitHub with no git client involved on the receiving end --
what makes a git-less davinci-1 able to obtain the code at all.

**Personal access token.** A GitHub credential that authenticates an HTTPS download when
`git`/a browser login is not available and the repository is private. Not needed for a
public repository.

**Login-node guard.** The `PBS_JOBID` check inside `install_env.sh` that warns (does not
hard-stop) against running a heavy install outside a job.

**Ablation (here).** The C6 comparison of `mining_strategy in {hard, easy_positive}`, with
every other field held fixed.

**Stale-cache refusal.** The `ValueError` `build_traces` raises when the on-disk cache
fingerprint disagrees with the one the current config computes.

---

## 3. Main body

### 3.1 What you already have, and what it replaces

*Establishes: the three deliverables from this session, and that the zip alone is
sufficient for either machine.*

| Deliverable | Contents |
|---|---|
| `dsn_wiring_C1_C6.zip` | a **complete, ready-to-use `Main/`**: every original pipeline and smoke-test file, plus the five edited modules, plus every new file |
| `ALL_CHANGES.diff` | a unified diff of the five edited modules only -- useful for `git diff`-style review before you commit, but not required for either pipeline below |
| `WIRING_REPORT.md` | what changed, at which exact integration point, and what was and was not verified |

Because the zip already contains the merged result, it supersedes needing the original
`dsn_pipeline.zip` / `dsn_smoke_tests.zip` at all, on either machine.

### 3.2 Pipeline A -- git, on your computer

*Establishes: the complete, ordered git sequence for merging this session's changes into
your tracked clone and pushing them. Everything in this subsection runs on your computer.
Nothing here touches davinci-1.*

```sh
# 1. start from a clean, up-to-date clone
cd <REPO_ROOT>
git status                              # confirm no uncommitted local changes are in the way
git checkout <branch>                   # e.g. main, or a feature branch you want to extend
git pull <GITHUB_URL> <branch>          # or just: git pull, if <branch>'s upstream is already set

# 2. bring the delivered Main/ in on top of your tracked one
unzip -o /path/to/dsn_wiring_C1_C6.zip -d /tmp/dsn_wiring
rsync -av /tmp/dsn_wiring/Main/ Main/   # overwrites the five edited files, ADDS every new one,
                                         # touches nothing else in the repo

# 3. review exactly what changed before staging anything
git status
git diff -- Main/config.py Main/search.py Main/objective_utils.py \
             Main/latent_burst_generator.py Main/run_optimization.py
# (ALL_CHANGES.diff is the same content pre-computed, if you'd rather read it as one file)

# 4. commit and push
git add -A
git commit -m "Wire C1-C6: latent benchmark, adaptive tie-break, factor retention, HPC ablation configs"
git push <GITHUB_URL> <branch>
```

At the end of Pipeline A, `<GITHUB_URL>`'s `<branch>` contains everything davinci-1 needs.
Your local `<REPO_ROOT>/Main/` is also now the exact folder to hand to Pipeline B.

### 3.3 Pipeline B -- getting the files onto davinci-1 without git

*Establishes: two independent ways to get the SAME bytes onto davinci-1, since it has no
git client. Pick one; they are not sequential steps.*

**Both options carry the same hazard.** Every file here was verified byte-for-byte pure
ASCII specifically because transfer through Windows-side tooling can silently re-encode
it -- documented in the project's own compliance rule: *"transfer through Windows tooling
(MobaXterm, copy-paste, `scp` from a Windows box) can re-encode non-ASCII bytes, producing
a `SyntaxError` that surfaces only at job-submission time on the cluster."* Both options
below are binary-safe by construction (they never pass through a re-encoding step), but
re-run the byte scan on davinci-1 regardless (given at the end of this subsection), since
that is what actually closes the loop rather than trusting the transport.

**Option B1 -- direct copy from your computer (recommended: no dependency on davinci-1's
outbound internet, no GitHub token to manage).** Run this from your computer, any time
after step 2 of Pipeline A (you do not even need to have committed or pushed yet -- this
copies your local `Main/` directly):

```sh
rsync -av Main/ <user>@davinci-1:"<HPC_REPO_ROOT>/Main/"
```

If `rsync` is not available on davinci-1's side, `scp` is binary-safe the same way:

```sh
scp -r Main/ <user>@davinci-1:"<HPC_REPO_ROOT>/"
```

This needs only an SSH client on your computer and `sshd` (plus, for the first form,
`rsync`) on davinci-1 -- both essentially universal on HPC login nodes. Git is not
involved on either side of this specific command.

**Option B2 -- download an archive snapshot directly on davinci-1 (useful if you'd rather
pull independently from the login node, without keeping your computer connected).**
Requires that Pipeline A's `git push` has already happened, and that the login node has
outbound HTTPS access to `github.com` -- many sites restrict this, so try it before
relying on it.

```sh
# on davinci-1
cd <HPC_REPO_ROOT_PARENT>          # wherever the folder should land
wget "<GITHUB_URL>/archive/refs/heads/<branch>.zip" -O dsn_latest.zip
# or, if wget is unavailable:
curl -L -o dsn_latest.zip "<GITHUB_URL>/archive/refs/heads/<branch>.zip"
unzip dsn_latest.zip
```

This downloads a **snapshot** of the branch at push time -- no `.git` history, just the
files, which is exactly what running the pipeline needs. GitHub serves this over plain
HTTPS with no git client involved on the receiving end. Two things to know before relying
on it:

- The unzipped folder is named `<repo-name>-<branch>` (slashes in `<branch>` become
  hyphens), e.g. `Deep-Summary-Network-apply-c1-c6/`, **not** the name your existing
  `<HPC_REPO_ROOT>` uses -- move or rename `Main/` out of it into place:
  ```sh
  rsync -av Deep-Summary-Network-<branch>/Main/ "<HPC_REPO_ROOT>/Main/"
  ```
- If `<GITHUB_URL>` is a **private** repository, an anonymous `wget`/`curl` gets a `404`,
  not a permission error that says so. You need a personal access token:
  ```sh
  curl -L -H "Authorization: token <GITHUB_TOKEN>" \
       -o dsn_latest.zip "<GITHUB_URL>/archive/refs/heads/<branch>.zip"
  ```
  This session does not know whether the repository is public or private -- confirm this
  before assuming Option B2 will work unauthenticated (5.1).

**After either option, on davinci-1:**

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

This does not need git either -- it is plain Python, already required by the pipeline
itself.

### 3.4 Environment

*Establishes: which environment to activate, and a discrepancy in the project's own
documentation you must resolve before trusting either name. No git involved.*

The project ships `install_env.sh`, `environment.yml`, and `verify_env.py` in `Main/`,
all three agreeing on the environment name **`meacnn`**. One documented PBS example
instead runs `source activate brian_env` (3.5). That is an unresolved discrepancy in the
project's own documentation. Check what actually exists before submitting a real job:

```sh
conda env list
```

If `meacnn` exists, use it. If only `brian_env` exists, that PBS example is the authority
and `meacnn` was never created on this site. If neither exists, build it:

```sh
module load cuda/12.1          # match your node; check with: module avail cuda
module load miniconda3         # or the site's actual module name
qsub -I -l select=1:ncpus=4:ngpus=1 -l walltime=01:00:00   # get a compute node
bash install_env.sh            # creates meacnn, installs the matching CUDA torch wheel,
                                # then pytorch-metric-learning + scikit-optimize + optuna
```

Then, in every subsequent shell (interactive or inside a PBS script):

```sh
conda activate meacnn          # or whichever name was confirmed above
python verify_env.py           # must print "N/N checks passed" with 0 failures
```

Do not proceed past a `verify_env.py` failure.

### 3.5 Running the verification ladder on davinci-1

*Establishes: this is the acceptance gate this session could not execute, and nothing in
3.6 should be attempted before it passes clean.*

Run from an interactive compute node, not the login node -- Rung 2 performs real (if
tiny) training:

```sh
qsub -I -l select=1:ncpus=4:ngpus=1 -l walltime=00:30:00
cd "<HPC_REPO_ROOT>/Main"
conda activate meacnn
sh run_wiring_checks.sh
```

`run_wiring_checks.sh` runs, in order, stopping at the first failure:

1. **Rung 0** -- byte scan of every `.py`/`.json` file, then `py_compile` over the whole
   import chain.
2. **Rung 1** -- `smoke_test_latent_and_objective.py` (the handoff's own 21 checks),
   `smoke_test_objective_wiring.py` (C2/C3), `smoke_test_factor_retention.py` (C5,
   including the grouping test).
3. **Rung 2** -- every pre-existing smoke test against **real** torch (nothing here may
   newly fail), then `smoke_test_latent_wiring.py` (C1, this time against real torch
   rather than this session's sandbox stub), then **`smoke_test_selected_epoch.py`**
   (the drift test), then `run_optimization.py --dry-run` on
   `hpc/config_latent_3class_hard.json`.

If `smoke_test_selected_epoch.py` fails, stop: the recomputed selected epoch has drifted
from `train.py`'s own rule, and C2's objective silently depends on it.

### 3.6 Submitting the C6 ablation (Rung 3)

*Establishes: the PBS submission for the two ablation runs, and the checks to run before
submitting either.*

**Before submitting:**

1. `--dry-run` first and read the reported `TOTAL_train_runs`.
2. Time one real trial by hand and multiply; compare against your intended walltime.
3. Run `smoke_test_end_to_end.py` on the login node -- five minutes, exercises every path
   the job will hit.
4. Compare the pre-flight's reported max model size / RAM against the node's RAM. A soft
   OOM is caught and scored as a failed trial; a hard Linux OOM-kill takes the whole
   study with it, silently, with no traceback.

**Two independent PBS jobs**, one per miner:

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

Two things to decide deliberately:

- **Device.** Both delivered configs carry `runtime.device = "cpu"`, inherited unchanged
  from the archived baseline. The template above requests a GPU but never passes
  `--device`, so it will run on CPU regardless of what PBS allocates. Edit
  `runtime.device` to `"cuda"` in the JSON, or add `--device cuda` to the command line, if
  you want GPU training.
- **`out_dir` / `cache_dir`.** Both configs already point at `<HPC_REPO_ROOT>/out` and
  `<HPC_REPO_ROOT>/cache_latent`, with distinct `experiment_name`s, so the two runs will
  not collide on disk even sharing one `out_dir`.

Submit with:

```sh
qsub dsn_latent_hard.pbs
qsub dsn_latent_easypos.pbs
```

`--resume` only applies to the final training stage, not the search phases: a search
killed mid-way must restart, but the trace cache survives (fingerprinted, section 2.1 of
`WIRING_REPORT.md`), so restarting does not repeat trace synthesis.

### 3.7 Reading the results

*Establishes: where to look once both jobs complete -- a pointer, not a full analysis.*

Each run writes `out/<experiment_name>/results.json`, with a `"test"` block reporting
`mean +/- std` over seeds for `ari` and `eff_rank`, plus `per_seed[*].epochs_run` (check
this against `train.max_epochs` first -- if it equals the ceiling, the runs never
converged). The decisive comparison the handoff's C6 section asked for is the
**factor-retention** numbers, not written to `results.json` automatically -- run
`factor_retention.py` on each run's saved `latent_ground_truth.json` and held-out
embeddings, as `Smoke_Tests/smoke_test_factor_retention.py` section G demonstrates
end to end.

### 3.8 If something goes wrong

| Symptom | Likely cause | Where to look |
|---|---|---|
| `Killed`, no traceback | Linux OOM-killer | narrow `depth_exponent_range`, per the pre-flight's reported model sizes |
| `SyntaxError` on a non-ASCII byte | re-encoded in transfer | re-transfer (3.3); every file was verified pure ASCII before leaving this session |
| `wget`/`curl` returns 404 downloading the archive | private repo needing a token, or wrong `<branch>` name | 3.3 Option B2's authenticated form |
| `wget`/`curl` hangs or times out on davinci-1 | login node has no outbound HTTPS | use Option B1 (direct copy from your computer) instead |
| Hangs, no output | should be impossible; check `matplotlib` import order | `evaluate.py` forces the Agg backend before `pyplot` is imported anywhere |
| `ValueError: infeasible window / split geometry` | the pre-flight is working correctly | it names the fix directly in the message |
| `STALE TRACE CACHE` | a latent parameter changed but the cache dir did not | pass `--overwrite-cache` or point at a fresh `cache_dir` |
| the search seems to score a different epoch than `train.py`'s own log | C2's recomputed e* has drifted | `smoke_test_selected_epoch.py` (3.5) exists to catch this before a real job does |
| unexpectedly low or high R^2_k in a C5 result | check `n_c` before drawing a conclusion | with `n_c = 3` the free-axis effective sample size is 9, not `N_eval` |
| a range set in `config_input.json` doesn't seem to take effect | `config.py` defaults vs the JSON file diverge | always check which one is actually in force |

---

## 4. Summary of results

```sh
# ===== on your computer (Pipeline A, needs git) =====
cd <REPO_ROOT>
git checkout <branch> && git pull <GITHUB_URL> <branch>
unzip -o dsn_wiring_C1_C6.zip -d /tmp/dsn_wiring
rsync -av /tmp/dsn_wiring/Main/ Main/
git add -A && git commit -m "Wire C1-C6" && git push <GITHUB_URL> <branch>

# ===== get the files onto davinci-1 (Pipeline B, NO git needed) =====
# EITHER, from your computer:
rsync -av Main/ <user>@davinci-1:"<HPC_REPO_ROOT>/Main/"
# OR, on davinci-1 itself, after the push above:
wget "<GITHUB_URL>/archive/refs/heads/<branch>.zip" -O dsn_latest.zip && unzip dsn_latest.zip

# ===== on davinci-1 =====
ssh <user>@davinci-1
conda env list                                  # confirm meacnn vs brian_env
conda activate meacnn
python verify_env.py                            # must be 0 failures

qsub -I -l select=1:ncpus=4:ngpus=1 -l walltime=00:30:00
cd "<HPC_REPO_ROOT>/Main"
sh run_wiring_checks.sh                         # stop here if anything fails

python3 run_optimization.py --config hpc/config_latent_3class_hard.json --dry-run
qsub dsn_latent_hard.pbs
qsub dsn_latent_easypos.pbs

cat out/latent_3class_hard/results.json
cat out/latent_3class_easypos/results.json
```

---

## 5. Open points, caveats, and assumptions

1. **This session does not know whether `<GITHUB_URL>` is public or private.** That
   determines whether Option B2's plain `wget`/`curl` works unauthenticated (5, 3.3).
2. **This session does not know whether davinci-1's login node permits outbound HTTPS to
   `github.com`.** Many HPC sites restrict this; if Option B2 hangs or times out, Option
   B1 (direct copy from your computer) has no such dependency.
3. The environment-name discrepancy (`meacnn` vs `brian_env`) is unresolved in the
   project's own documentation. Section 3.4's `conda env list` check is the safeguard;
   do not submit a job assuming one or the other without running it.
4. The delivered HPC configs default to `runtime.device = "cpu"`, unchanged from the
   archived baseline; whether that matches what you actually want on a requested GPU
   node is a decision this document raises but does not make.
5. `<PBS_walltime>` and `<PBS_ncpus>` are left as placeholders; fill them from the
   pre-submission checklist (3.6), not a guess.
6. Rung 2 has not been executed by anyone against real torch as of this document. In
   particular, `smoke_test_selected_epoch.py` has never run. Section 3.5 is where that
   finally happens.
7. Nothing in this document, or the session before it, has measured the network's actual
   performance on the new latent benchmark.

---

## 6. References

**This session's own deliverables (25 July 2026):** `WIRING_REPORT.md`,
`ALL_CHANGES.diff`, `dsn_wiring_C1_C6.zip`.

**Prior session:** `HANDOFF_wiring_changes.md` (25 July 2026), which states the
repository as `https://github.com/Leonardodm00/Deep-Summary-Network`, working area
`Main/` -- used directly as `<GITHUB_URL>`'s example above; substitute your own fork or
mirror if that is not the remote you actually push to.

**Project knowledge base, read in full this turn:**
`Patched/Augmentation/Installation/install_env.sh`,
`Patched/Augmentation/Installation/environment.yml`,
`Patched/Augmentation/Installation/verify_env.py`,
`Patched/Optimization/03_USAGE.md` (sections 5-7),
`Patched/Optimization/02_TECHNICAL.md` (sections 13-14).
Every command and path claim above that is not marked with a `<placeholder>` was checked
against one of these files rather than assumed.

**GitHub's archive-download-without-git mechanism** (Pipeline B, option B2) is stated
from general, stable knowledge of GitHub's HTTPS interface, not from a source checked
this session; it has not changed in years and is not the kind of claim this project's
citation rules require a literature source for, but it was not re-verified against
GitHub's current documentation in this turn.

**No literature consulted.** This document contains no empirical or biomedical claim, so
neither PubMed nor bioRxiv/medRxiv was queried for it.
