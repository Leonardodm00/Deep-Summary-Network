# K3 culture grouping -- apply and verify

**Date:** 6 August 2026
**Applies to:** `Leonardodm00/Deep-Summary-Network`, branch `main`
**Contents:** 6 modified files + 1 new smoke test, all under `Main/`

What this implements: the culture-grouping map

    gamma : {0, ..., U_tot - 1} -> {0, ..., U_cult - 1}

from trace record index to culture index. Before K3 the pipeline hardcoded
gamma = identity (one trace == one culture). That is wrong for the extractor's
`--mode per_region_single`, where one well yields C single-channel subregion
traces: C records, ONE culture.

`cultures=None` everywhere reproduces the pre-K3 behaviour index-for-index, so
every existing configuration is unaffected.

---

## 1. Transfer

Move `K3_culture_grouping.tar.gz` with a BINARY-safe tool -- `scp`, `rsync`, or
MobaXterm's file browser with "text/ASCII transfer mode" OFF. Do not paste the
source into a terminal: that is the path by which a UTF-8 byte collapses to
cp1252 and the file dies at `SyntaxError` on the cluster.

```bash
scp K3_culture_grouping.tar.gz ldellamea@davinci-1:~/
```

## 2. Apply

```bash
cd ~/Deep-Summary-Network
git status                      # confirm a clean tree before overwriting
git rev-parse --short HEAD      # record what you are patching
cp -r Main Main.bak.$(date +%Y%m%d)      # cheap insurance
tar -xzf ~/K3_culture_grouping.tar.gz    # unpacks into ./Main/
git diff --stat                 # expect exactly 6 modified files
```

Files overwritten:

| File | Change |
|---|---|
| `Main/preprocessing_cache.py` | `TraceSpec(..., culture=)`, culture in npz + manifest, new `load_cached_cultures()` |
| `Main/data_splits.py` | `make_trace_splits(..., cultures=)`; split at culture granularity; `g` emits culture indices |
| `Main/data_pipeline.py` | `MEAWindowDataset` carries a local-trace -> culture map |
| `Main/train.py` | composes the local trace index through that map (this was the silent bug) |
| `Main/run_optimization.py` | `culture` / `culture_id` spec keys, orphan-subregion guard, grouping in the fingerprint, `build_cultures()` |
| `Main/search_dry_run.py` | dry run uses the same grouping as the real run |

New: `Main/Smoke_Tests/smoke_test_culture_grouping.py`.

## 3. Verify BEFORE any qsub

Paste as one block on the login node. No allocation needed.

```bash
cd ~/Deep-Summary-Network/Main

# ---- 0. what interpreter is this, actually? ------------------------------
python3 --version; echo "conda env: ${CONDA_DEFAULT_ENV:-none}"

# ---- 1. LINE ENDINGS: anything with a shebang. FIRST, because a broken   --
#         shebang means the scheduler never starts python at all.         --
python3 - <<'EOF'
import os, sys
bad = []
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
    for f in files:
        if f.endswith((".sh", ".pbs", ".slurm")):
            p = os.path.join(root, f)
            n = open(p, "rb").read().count(b"\r")
            if n:
                bad.append((p, n))
for p, n in bad:
    print("CRLF", p, "(%d CR bytes)" % n)
print("FAIL: %d file(s)" % len(bad) if bad
      else "OK: every .sh/.pbs/.slurm file is LF-only")
sys.exit(1 if bad else 0)
EOF
# fix any FAIL with:  sed -i 's/\r$//' <file>   then re-run

# ---- 2. ENCODING: transfer corruption in Python source -------------------
python3 - <<'EOF'
import os, sys
bad = []
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
    for f in files:
        if f.endswith(".py"):
            p = os.path.join(root, f)
            b = [hex(c) for c in open(p, "rb").read() if c > 127]
            if b:
                bad.append((p, b[:6]))
for p, b in bad:
    print("NON-ASCII", p, b)
print("FAIL: %d file(s)" % len(bad) if bad else "OK: every .py file is pure ASCII")
sys.exit(1 if bad else 0)
EOF

# ---- 3. SYNTAX: the whole import chain -----------------------------------
python3 -m py_compile $(find . -name '*.py' -not -path '*/__pycache__/*') \
  && echo "OK: everything compiles"

# ---- 4. IMPORT: catches a missing module the syntax check cannot ---------
PYTHONPATH="$(pwd)" python3 -c "import run_optimization" && echo "OK: imports resolve"

# ---- 5. the K3 suite -----------------------------------------------------
PYTHONPATH="$(pwd)" python3 Smoke_Tests/smoke_test_culture_grouping.py

# ---- 6. regression: nothing else moved -----------------------------------
bash hpc/run_all_smoke_tests.sh
```

Step 6 note: `PYTHONPATH` must be an ABSOLUTE path to `Main/`. The wrapper does
this; hand-rolling `PYTHONPATH=.` produces a confusing partial pass/fail pattern
that looks like broken code but is purely an invocation error (see
`Documentation/HPC_SMOKE_TEST_RUNBOOK.md` sec. 0).

### Expected result of step 6

35 of 36 suites pass. `smoke_test_loss_type_wiring.py` fails 2 of its 15 checks
("gate momentum 0 = cumulative" and "census is JSON-serialisable"). **This
failure is PRE-EXISTING on `main` and unrelated to K3** -- it was reproduced on
the unpatched tree and the pass/fail pattern is byte-identical before and after.
It lives in the loss-gate / census area, which K3 does not touch.

---

## 4. Using it

A specs record may now carry a culture:

```json
[
  {"name": "wellA1_sub00", "condition": 0, "culture": "plate3__ptrain_A1",
   "path": "out_A1/trace_subregion_00.npz"},
  {"name": "wellA1_sub01", "condition": 0, "culture": "plate3__ptrain_A1",
   "path": "out_A1/trace_subregion_01.npz"}
]
```

Both `culture` and `culture_id` are accepted. Omit it and the record falls back
to `culture = name`, the identity grouping.

**The guard.** Pointing at a per-subregion archive (one carrying
`subregion_index` or `culture_id`, as the extractor writes) WITHOUT a culture
field now raises at spec-build time. That is deliberate: without it, a forgotten
field degrades silently to the identity grouping, siblings split across
train/test, the miner can pick a window's own sibling as its positive, and the
reported geometry looks better than it is.

**The fingerprint.** The grouping is hashed into `data_fingerprint.json`.
Regrouping the same traces into different cultures will make a stale `cache_dir`
refuse rather than silently reuse the old assignment. Pass `--overwrite-cache`
when that refusal is expected.

**Leave-one-out.** `trace_split_fold` now indexes CULTURES, so the held-out unit
is one whole recording per class. Under the identity grouping this is unchanged.

---

## 5. What K3 does NOT fix

Evaluation is still WINDOW-level: there is no per-culture aggregation in
`metrics.py` or `evaluate.py`. After K3 the test split contains no leaked wells,
but it does contain C correlated subregion traces per test well, so ARI is
computed over C-times-correlated items and its confidence interval is
optimistic. This is a separate issue, deliberately out of scope for this pass.
