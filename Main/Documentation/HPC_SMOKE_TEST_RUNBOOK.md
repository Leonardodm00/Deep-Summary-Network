# Running the smoke tests on davinci-1

**Date:** 31 July 2026
**Applies to:** branch `multichannel` (30 pipeline suites + 7 extractor suites)

This is the operational runbook. The scientific content of the changes is in
`MULTICHANNEL_TECHNICAL_DOCUMENT.md`.

---

## 0. The one invocation mistake that costs an afternoon

Do **not** run this:

```bash
cd Main && PYTHONPATH=. python3 Smoke_Tests/run_all_smoke_tests.py    # WRONG
```

`run_all_smoke_tests.py` spawns every suite as a **separate subprocess with
`cwd` set to `Smoke_Tests/`**. A relative `PYTHONPATH="."` is resolved by each
subprocess against *its own* cwd, which is `Smoke_Tests/`, not `Main/`. Suites
that self-locate `Main/` via `__file__` pass; suites that don't raise
`ModuleNotFoundError: No module named 'config'`. The result is a confusing
**partial** pass/fail pattern that looks like broken code but is purely an
invocation error.

`PYTHONPATH` must be an **absolute** path to `Main/`. The wrapper
`Main/hpc/run_all_smoke_tests.sh` does exactly this and nothing else, so prefer
it over hand-rolling the command.

---

## 1. Environment

Once per cluster account, on an **interactive compute node** (never the login
node):

```bash
qsub -I -l select=1:ncpus=4 -l walltime=01:00:00
module load miniconda3
cd ~/Deep-Summary-Network/Main/hpc
bash setup_env_davinci.sh --env-name meacnn_cpu
```

This builds the env from `environment_hpc.yml`, installs the CPU torch wheel,
writes the `LD_LIBRARY_PATH` activation hook for the `GLIBCXX_3.4.26` fix, and
runs `verify_env_hpc.py`.

For GPU use a **separate** env so the CPU one stays intact:

```bash
module load cuda/12.1
module load miniconda3
bash setup_env_davinci.sh --env-name meacnn_gpu --gpu
```

Verify at any time:

```bash
conda activate meacnn_cpu
python3 Main/hpc/verify_env_hpc.py
```

---

## 2. Get the branch onto the cluster

```bash
cd ~
git clone https://github.com/Leonardodm00/Deep-Summary-Network.git
cd Deep-Summary-Network
git checkout multichannel
```

If your cluster has no outbound git access, clone on your workstation and
`scp -r` the tree across. Do **not** copy through a Windows tool that rewrites
line endings: `.gitattributes` declares `eol=lf` precisely because a trailing
`CR` makes bash fail with `$'\r': command not found` and `qsub` reject the job.

Sanity check after transfer:

```bash
cd ~/Deep-Summary-Network
file Main/hpc/*.sh | grep -i crlf && echo "CRLF PRESENT -- fix before running"
```

No output means you are clean.

---

## 3. Run the 30 pipeline suites

```bash
conda activate meacnn_cpu
cd ~/Deep-Summary-Network
bash Main/hpc/run_all_smoke_tests.sh
```

The wrapper prints the resolved `Main/` and the absolute `PYTHONPATH` before it
starts, so a misresolution is visible immediately rather than as a puzzling
suite failure.

Useful variants:

```bash
bash Main/hpc/run_all_smoke_tests.sh --list          # names only, runs nothing
bash Main/hpc/run_all_smoke_tests.sh --quick         # pass --quick where accepted
bash Main/hpc/run_all_smoke_tests.sh --only in_channels augmentation_mc
```

`--only` is a substring match, so `--only mc` catches all four `_mc` suites.

**Expected: 30/30.** The five multichannel suites are last in the run order:

```
smoke_test_in_channels.py         stem: Conv1d(C, ...) and (M, C, T) forward
smoke_test_augmentation_mc.py     shared warp field / shared shift across C
smoke_test_generate_mc.py         generate_multichannel_traces -> (C, K)
smoke_test_data_pipeline_mc.py    (C, W) windows and (M, C, W) collation
smoke_test_pipeline_mc.py         config wiring -> eval forward -> gradients
```

---

## 4. Run the 7 extractor suites

These are **not** in the runner: they belong to a different package, need only
numpy/scipy/matplotlib, and are independently runnable.

```bash
cd ~/Deep-Summary-Network/Main/hpc/MultiChannel
for t in smoke_test_channel_subsets_*.py smoke_test_channel_subset_viz.py; do
    printf '%-46s ' "$t"
    python3 "$t" >/dev/null 2>&1 && echo PASS || echo FAIL
done
```

**Expected: 7/7.** These run happily in `brian_env` too, since they never
import torch.

---

## 5. As a batch job

Interactive nodes time out. For the full sweep, submit it:

```bash
cat > run_smoke_all.pbs <<'EOF'
#!/bin/bash
#PBS -N dsn_smoke
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o dsn_smoke.log

set -euo pipefail
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate meacnn_cpu

cd "$PBS_O_WORKDIR"
echo "[job] host=$(hostname) started=$(date -Is)"

echo "=== 30 pipeline suites ==="
bash Main/hpc/run_all_smoke_tests.sh

echo "=== 7 extractor suites ==="
cd Main/hpc/MultiChannel
fail=0
for t in smoke_test_channel_subsets_*.py smoke_test_channel_subset_viz.py; do
    printf '%-46s ' "$t"
    if python3 "$t" >/dev/null 2>&1; then echo PASS; else echo FAIL; fail=1; fi
done
echo "[job] finished=$(date -Is)"
exit $fail
EOF

qsub run_smoke_all.pbs
qstat -u "$USER"
tail -f dsn_smoke.log
```

Set `#PBS -q <queue>` and any account/project flag your allocation requires;
`qstat -Q` lists the queues.

---

## 6. Reading failures

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'config'` in **some** suites | relative `PYTHONPATH` | use the wrapper (section 0) |
| `GLIBCXX_3.4.26 not found` | system `libstdc++` winning | re-`conda activate`; the hook from `setup_env_davinci.sh` sets `LD_LIBRARY_PATH` |
| `$'\r': command not found` | CRLF from a Windows transfer | `dos2unix`, or re-clone with `.gitattributes` honoured |
| `SyntaxError: Non-UTF-8 code starting with '\x97'` | non-ASCII byte introduced in transit | files ship pure ASCII; re-transfer |
| `smoke_test_inspect_latent.py` FAILS: *neither shipped hpc/ config was found* | **pre-existing on `main`**, path dependency, not multichannel | see K1 below |
| Suite killed with no traceback | OOM | raise `mem=` in the PBS select line |

### Known non-defects

- **K1.** `smoke_test_inspect_latent.py` fails identically on untouched `main`.
  It looks for a config under `hpc/` that is not staged flat with the pipeline.
  Pre-existing; not introduced by the multichannel work.
- **K2.** `smoke_test_removed_modules.py` passes both its checks then may be
  killed by a memory-constrained node. Give it more `mem` or run it alone.

---

## 7. Before trusting a green run

The merge was verified with **torch 2.13.0+cu130 (CPU), numpy 2.4.4,
scipy 1.17.1, pytorch_metric_learning 2.9.0, scikit-optimize 0.10.2**.
`environment.yml` pins **numpy 1.26**. The code proved version-agnostic across
that gap in testing, but the cluster environment is the one that counts, which
is the entire reason for running these suites there.

Record the actual versions alongside the results:

```bash
python3 - <<'EOF'
import importlib
for m in ("numpy","scipy","torch","sklearn","pytorch_metric_learning","skopt"):
    try:
        mod = importlib.import_module(m)
        print("%-26s %s" % (m, getattr(mod, "__version__", "?")))
    except Exception as e:
        print("%-26s MISSING (%s)" % (m, type(e).__name__))
EOF
```
