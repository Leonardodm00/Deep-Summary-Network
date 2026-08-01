# Merge report -- multichannel onto main

Base: `Leonardodm00/Deep-Summary-Network`, branch `main`, `Main/` tree
(fetched via codeload; the GitHub REST API was rate-limited).
Common ancestor used for the 3-way merge: `Main/Colab_zips/dsn_pipeline.zip`
(confirmed: 4 of 5 non-`.py` assets byte-identical to BOTH sides).

Environment: torch 2.13.0+cu130 (CPU), pytorch_metric_learning 2.9.0,
scikit-optimize 0.10.2, numpy 2.4.4, scipy 1.17.1.

## STEP 1 -- pure additions (no merge)

- `augmentation_diagnostics.py`
- assets: `config_search_3class_9ch.json`, `run_search_9ch.pbs`, `run_aug_diag.pbs`
- tests: `smoke_test_in_channels.py`, `_augmentation_mc.py`, `_generate_mc.py`,
  `_data_pipeline_mc.py`, `_pipeline_mc.py`
- MultiChannel: `channel_subset_extraction.py` (**was absent from GitHub -- the
  folder could not run as committed**), `generate_burst_data.py`, 6 extractor
  suites. The 4 files already on GitHub were byte-identical; nothing overwritten.

## STEP 2 -- take-upload wholesale (repo == ancestor for all five)

`backbone.py`, `evaluate.py`, `generate_burst_data.py`, `inference.py`,
`run_data_pipeline.py`.

## STEP 3 -- clean auto-merge, zero conflicts

`data_pipeline.py`, `data_splits.py`, `train.py`.

## STEP 4 -- conflict resolution (6 hunks, 3 files)

- `config.py` (4) and `run_optimization.py` (1): add-only adjacency, resolved by
  union (repo's `LatentConfig` / `data_mode=="latent"` block vs. the multichannel
  `n_channels` validator).
- `augmentation.py` (1): the only genuine semantic collision. Repo restructured
  `build_triplet_instance` so `warp_bands` no longer retries (P_b = 0 is legal
  under cross-culture). Resolution keeps the repo control flow and ports the
  multichannel `(T,) | (C, T)` guard into it, replacing
  `window = _to_tensor(window).reshape(-1)`, which would have flattened the
  channel axis.
- `_generate_pool` empty path (repo-only new code) now returns `(0, T)` for 1-D
  and `(0, C, T)` for multichannel instead of computing `T` from a flattened
  `C*T` length.

## STEP 5 -- (C, K) axis fixes

| File | Fix |
|---|---|
| `run_optimization.py:1565` | `t.shape[0]` -> `t.shape[-1]` (trace_lengths) |
| `data_splits.py` | `tr.shape[0]` -> `shape[-1]`; `tr[s:e]` -> `tr[..., s:e]` |
| `data_splits.py` (`make_trace_splits`) | `arrays[u].shape[0]` -> `shape[-1]`, x3 -- NEW repo code the multichannel branch never saw |
| `preprocessing_cache.py` | `ndim != 1` -> `ndim not in (1,2)`; `shape[0]` -> `shape[-1]` |

## STEP 6 -- config wiring

Blanket `NotImplementedError` for `n_channels > 1` replaced by mode-specific
validation: `numpy` / `real` now proceed; `synthetic` and `latent` refuse C > 1
with a message naming the alternative. Latent x multichannel is refused
deliberately (agreed): it requires a generative decision (one shared phi per
recording with independent per-channel spike noise, vs. an independent phi per
subregion) that must not be guessed.

`NumpyTraceProvider` now validates metadata-first: `in_channels` is read from the
archive when present and never inferred from `ifr_trace.shape[0]`, because the
extractor emits `(9, K)` in BOTH `multichannel` (rows = channels) and
`per_region_single` (rows = independent samples) modes -- shape alone is
ambiguous and would silently mislabel the latter. Missing `ifr_trace` now raises
a message naming the fix instead of a bare `KeyError`.

## STEP 7 -- extractor

- `traces.npz` now carries `ifr_trace` as an alias of `X` (same array; `X` kept
  so the viz code and stage suites keep working).
- `mode="per_region_single"` additionally writes one `trace_subregion_NN.npz` per
  subregion, each `(K,)` with `in_channels=1` and a shared `culture_id` (the
  recording folder name). Verified: 9 sibling files written, correct shapes.

## STEP 8 -- runner

5 multichannel suites appended to `ORDER` in `run_all_smoke_tests.py`
(previously 25, now 30). `--list` verified clean.

## VERIFICATION

- 57 `.py` files: **0 non-ASCII bytes**, 0 compile failures (HPC transfer rule).
- Baseline regression, repo's own suites vs. the MERGED tree: **20/20 PASS**
  (config, backbone, augmentation, data_pipeline, data_splits, trace_splits,
  batch_geometry, cross_culture_batches, latent_wiring, objective_wiring,
  silhouette_floor, metrics, evaluate, inference, burst_pipeline, checkpoint,
  train, search, selected_epoch, synthetic_config).
- Multichannel suites: **5/5 PASS**.
- Extractor package standalone in a clean dir: **7/7 PASS**.
- End-to-end, real-format MEA data (6 recordings, 3 classes x 2 reps,
  90 electrodes, 120 s @ 10110.09 Hz, binary uint8 ptrain rasters):
  - multichannel C = 9: TEST ARI 1.0000, AMI 1.0000, silhouette 0.6839
  - single channel (whole_culture): TEST ARI -0.0606, AMI -0.0870, silhouette -0.2832
- Everything above run TWICE; identical results both runs.

## KNOWN ISSUES

K1. `smoke_test_inspect_latent.py` FAILS -- but it fails **identically on the
    untouched repo baseline** (`AssertionError: neither shipped hpc/ config was
    found`). It looks for a config in `hpc/`, which is not staged flat with the
    pipeline. Pre-existing path dependency, NOT caused by this merge.

K2. `smoke_test_removed_modules.py` passes both its checks then is SIGKILLed by
    this sandbox (memory). Sandbox artifact; re-run on the cluster.

K3. **NOT IMPLEMENTED -- sibling-subregion culture grouping.** Under
    `positives_mode="cross_culture"`, culture identity is currently the trace
    index, one-to-one (`data_splits.py`, `trace_of_window`). With
    `per_region_single` each subregion becomes its own trace, so siblings from
    one recording would get DIFFERENT culture ids and could be paired as
    positives -- which is exactly what must not happen. This needs a
    culture-of-trace grouping vector threaded through `assign_cultures` /
    `make_trace_splits` so that (a) all C siblings land in the same split and
    (b) `trace_of_window` emits the recording id. The extractor already writes
    `culture_id` into each per-subregion `.npz` in preparation; nothing consumes
    it yet. `npz_specs` records would need a `culture` field too.

K4. `generate_burst_data.py` ships in two packages (pipeline + MultiChannel).
    Verified byte-identical at merge time; must be kept in sync.

K5. `run_search_cpu.pbs` is referenced by shipped PBS scripts but is in no
    package and not on GitHub. `run_search_9ch.pbs` still contains
    `CONDA_ENV="REPLACE_WITH_YOUR_ENV_NAME"` and calls a smoke test from the
    tests package, so both packages must unpack into the same working directory.

K6. Real-data format assumptions unconfirmed against your actual files:
    `ptrain_<k>.mat` must hold a binary uint8 raster of shape (n, 1); `index_base`
    (0 vs 1); `fs_raw` (default 10110.09 Hz). One real folder resolves all three.

## PUSHING TO THE REPO

`push_multichannel_branch_v5.sh` clones `main`, creates branch `multichannel`,
unpacks the three archives into the repo's own folder layout, runs the gates,
and only then commits and pushes. Dry run by default; `--push` to go live. It also stages the three
documents in this set into `Main/Documentation/`.

Placement (follows existing repo convention):

| Source | Destination |
|---|---|
| pipeline modules, `*.yml`, `*.sh`, `COLAB_RUNBOOK.md`, `config_example/toy.json` | `Main/` |
| all suites (`*.py` only) | `Main/Smoke_Tests/` |
| `config_search_3class_9ch.json`, `run_search_9ch.pbs`, `run_aug_diag.pbs` | `Main/hpc/` |
| extractor package | `Main/hpc/MultiChannel/` |

`smoke_test_pipeline_mc.py` was updated to resolve its config from own-dir ->
`../hpc/` -> cwd, so the foldered repo layout AND the flat cluster runtime
layout both work from a SINGLE copy of the file. Both layouts verified passing.

Dry run against a live clone of `main` @ ada539a: 81 files staged, 68 `.py`
scanned (pure ASCII, all compile), no CRLF, no bytecode, 30 files changed,
6537 insertions. Suites re-run from the staged branch layout: PASS.

## PRE-EXISTING REPO ISSUE (not introduced here, not fixed here)

Five files under `Main/hpc/Config/` are committed with CRLF while
`.gitattributes` declares `*.json text eol=lf`:

    config_l3c_epsh_multiall_005.json
    config_l3c_epsh_multimean_005.json
    config_l3c_h_multiall_005.json
    config_l3c_h_multimean_005.json
    config_l3c_h_singlemean_005.json

Git normalises them on every fresh clone, so they show as modified before
anything is edited. A blanket `git add -A` would sweep that unrelated
normalisation into this commit, which is why the push script stages only a
recorded manifest of the paths it wrote. Worth fixing on its own branch
(`git add --renormalize .`).

## DEFECTS FOUND AFTER THE FIRST PACKAGING (both fixed in v3)

D1. RUNNER REGRESSION (mine, serious). When the archives were rebuilt after
    patching smoke_test_pipeline_mc.py, the runner was copied from a staging
    directory that had never received the Step-8 patch. The shipped
    dsn_smoke_tests_MERGED_v2.zip therefore contained the UNPATCHED 25-suite
    run_all_smoke_tests.py, so the five multichannel suites would never have
    run under the runner. Fixed; verified INSIDE the shipped archive:
    30 ORDER entries, 5 multichannel entries.

D2. WRAPPER SELF-LOCATION (pre-existing on main). Main/hpc/run_all_smoke_tests.sh
    resolves Main/ by testing "$HERE/Main" then "$HERE/config.py", so from its
    OWN committed location (Main/hpc/) both tests fail and it aborts with
    "could not locate Main/". A third branch was added for the Main/hpc/ case.
    Verified: resolves correctly and lists/runs all 30 suites from the repo
    layout.

D3. WINDOWS CASE-INSENSITIVITY (in the push script, not the code). On Windows
    the revision `main` collides with the directory `Main/` and git aborts with
    "ambiguous argument 'main': both revision and filename". Every command
    naming a revision now ends with the `--` separator. Invisible on
    Linux/macOS, which is why it was not caught in the first dry runs.

D4. ASSET PLACEMENT vs TEST REQUIREMENT. smoke_test_pipeline_mc.py check A6 hard
    -failed in the foldered repo layout: it looked for config_search_3class_9ch
    .json in its own directory only, while repo convention puts configs in
    Main/hpc/. Rather than duplicating the file, the suite now searches
    own-dir -> ../hpc/ -> cwd. Both the repo layout AND the flat cluster runtime
    layout verified passing from a SINGLE copy.
