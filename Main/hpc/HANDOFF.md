# Handoff -- Deep-Summary-Network HPC setup (continue in a fresh chat)

## Who / what
PhD student running a 1D-CNN contrastive-learning pipeline
(github.com/Leonardodm00/Deep-Summary-Network, the flat `Main/`-equivalent
layout) to cluster synthetic MEA "burst" traces into 3 phenotype classes via a
Bayesian-optimization hyperparameter search. Goal: run the FULL search to
CONVERGENCE on synthetic 3-class data, on davinci-1 (Leonardo S.p.A., PBS
scheduler), CPU for now (GPU later). Colab was abandoned because the run exceeds
its session wall-time.

## Project conventions (important)
- Pipeline modules import each other by BARE NAME and must sit in ONE FLAT
  directory (no Main/ subfolder). The user's two working zips are
  `dsn_pipeline.zip` (24 files) and `dsn_smoke_tests.zip` (17 files).
- ALL .py files must be pure ASCII (hpc-python-compat skill) -- davinci +
  MobaXterm/cp1252 transfer corrupts non-ASCII bytes into SyntaxErrors.
- User preferences: full mathematical notation (no dropped conditioning/
  subscripts); scientific code = established libraries, separation of concerns,
  ask before assuming; EVERY new algorithm/script ships with a smoke test,
  controlled twice for correctness.
- Two davinci-1 gotchas already solved: (1) libstdc++/GLIBCXX_3.4.26 -> env's
  libstdcxx-ng + LD_LIBRARY_PATH activation hook; (2) ASCII encoding.

## What was built THIS session (all validated, in /mnt/user-data/outputs)

### A. Synthetic-generator config + per-run saving (merged into the pipeline)
Delivered as updated `dsn_pipeline.zip` + `dsn_smoke_tests.zip` earlier:
- config.py: new nested `SyntheticConfig` + `SyntheticClassOverride` in
  DataConfig -> burst-generator params (rate_min/max, width_min/max,
  amp_jitter_min/max) now configurable, with optional per-class overrides.
  Backward-compatible: default runs are byte-identical to before.
- data_splits.py: MultiClassSyntheticProvider extended (amp jitter + per_class),
  rng draw order preserved.
- run_optimization.py: passes synthetic config through; NEW
  save_synthetic_artifacts() writes synthetic_generator_params.json +
  figures/synthetic_traces_overview.png into out_dir on every synthetic run.
- Smoke_Tests/smoke_test_synthetic_config.py (NEW, 21 checks) + registered in
  run_all_smoke_tests.py ORDER list.
Also: two standalone viz scripts (visualize_synthetic_classes.py +
smoke_test) that plot the 3 classes with burst detection -- separate download,
NOT in the zip.

### B. HPC run kit (7 files, this delivery)
- README_HPC.md          -- 8-step runbook (upload->setup->verify->time->size->
                            submit->read convergence->collect)
- setup_env_davinci.sh   -- one-shot conda env; CPU default; --gpu flag (cu121);
                            --rebuild; writes libstdc++ activation hook; runs
                            verify_env_hpc.py at end. Needs --env-name <name>.
- environment_hpc.yml    -- conda base (numpy1.26/scipy1.13/mpl3.9/sklearn1.5/
                            tqdm/libstdcxx-ng + pip: scikit-optimize,
                            pytorch-metric-learning). NO torch (installed by
                            setup script), NO optuna (pipeline doesn't use it).
- verify_env_hpc.py      -- 9 real-sub-API checks (TripletMarginLoss fwd+bwd,
                            gp_minimize, CubicSpline libstdc++ canary, etc).
                            RENAMED from verify_env.py to avoid collision with
                            the OLD verify_env.py that dsn_pipeline.zip ships
                            (that old one wants optuna and will report spurious
                            failures -- ignore it, use verify_env_hpc.py).
- estimate_walltime.py   -- times a short real run on the node, multiplies by
                            the full budget, prints suggested #PBS walltime.
- config_search_3class_hpc.json -- 3-class synthetic, CPU, budget 15/15/10
                            seeds=2 ep=50 = 82 train() runs (~28 h @ 25 s/epoch).
- run_search_cpu.pbs     -- PBS job. BUDGET block (5 shell vars at top) overrides
                            the config via --n-calls-*/--n-seeds/--max-epochs, so
                            15/15/10 <-> 30/30/20 is a 5-number edit, NO JSON
                            change. Self-preflights (verify + dry-run). GPU
                            switch = 3 marked edits.

## KEY NUMBERS / decisions
- Full budget 30/30/20 seeds=3 ep=60 = 243 runs ~= 100 h CPU (@25 s/epoch here).
  Chosen default = 82 runs ~28 h (needs a 48 h PBS queue).
- CONVERGENCE criteria (from an earlier project chat, confirmed): TEST ARI near
  1.0 on clean synthetic; mined triplets FALL across epochs; silhouette climbs;
  config_best.json margin settles ~0.3 (NOT at a boundary like 0.84, which
  signals under-explored search -> increase trial counts, NEVER cut epochs).
  6/6/4 trials was shown to FAIL (TEST ARI ~0.05); that's why trims must come
  from trial counts/seeds, not epochs.

## USER MUST DO before submitting (both flagged in-file)
1. run_search_cpu.pbs: set CONDA_ENV="<their env name>" (placeholder now).
2. Run estimate_walltime.py on the REAL davinci node; set #PBS -l walltime to
   the suggested value (the 28 h assumes this sandbox's 25 s/epoch, may differ).

## VERIFICATION already done
Simulated davinci workdir (pipeline unzipped + HPC files): verify_env_hpc.py
9/9; both shell scripts bash-syntax OK; PBS CLI dry-run -> 82 runs (and 243 on
the 30/30/20 swap); a real 2-epoch --skip-search train produced the exact
README Step-8 artifact tree (results.json, best_model.pt, config_best.json,
synthetic_generator_params.json, figures/{embedding_test_seed_0,
synthetic_traces_overview}.png).

## NEXT STEPS (candidate topics for the fresh chat)
- Run the actual search on davinci and interpret results.json / the embedding
  figures / convergence.
- Build the GPU variant (run_search_gpu.pbs + a meacnn_gpu env) once CPU works.
- Move from synthetic to real data (data_mode="numpy", .npz specs) -- the
  pipeline already supports it; needs the user's traces pre-computed to .npz
  (keys ifr_trace, fs_ifr).
- Technical docs / config.json reference doc (was requested in an earlier chat,
  never finished).
