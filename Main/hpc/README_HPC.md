# Running the full 3-class synthetic search on davinci-1 (PBS, CPU)

An ordered runbook: upload -> set up env -> verify -> measure real walltime ->
size the job -> submit -> read convergence -> collect results. Follow it top to
bottom the first time; after that only Steps 6-8 matter per run.

Everything here targets davinci-1 (Leonardo S.p.A., PBS scheduler), CPU by
default, with the GPU switch noted where relevant.

--------------------------------------------------------------------------------
## What you need on the cluster (one directory)

Put all of these in a single working directory on davinci (e.g.
`~/Deep_Summary_Network/`). They must sit together (the pipeline modules import
each other by bare name):

Pipeline + smoke tests (from your existing zips, already ASCII-clean):
  - everything in `dsn_pipeline.zip`   (config.py, data_splits.py,
    run_optimization.py, train.py, backbone.py, ... )
  - everything in `dsn_smoke_tests.zip` (optional on the cluster, but
    smoke_test_end_to_end is a good login-node check)

HPC setup + run files (this delivery):
  - environment_hpc.yml
  - setup_env_davinci.sh
  - verify_env_hpc.py
  - estimate_walltime.py
  - config_search_3class_hpc.json
  - run_search_cpu.pbs

Unzip both zips INTO this directory (flat, no subfolders):
    unzip -o dsn_pipeline.zip
    unzip -o dsn_smoke_tests.zip

NOTE on verify_env: dsn_pipeline.zip ships its OWN older `verify_env.py` (46
checks, including optuna, which the current pipeline does not actually import).
Use `verify_env_hpc.py` (this delivery) instead -- it checks exactly the
packages the pipeline uses and matches what environment_hpc.yml installs. You
can ignore or delete the repo's verify_env.py; if you run it after this setup it
will report spurious optuna failures because environment_hpc.yml intentionally
omits optuna.

--------------------------------------------------------------------------------
## Step 1 -- set up the conda environment (once)

Do this on an INTERACTIVE compute node, not the login node.

    # grab an interactive CPU node for ~1 h
    qsub -I -l select=1:ncpus=4 -l walltime=01:00:00

    # load the conda module (name may differ; check `module avail`)
    module load miniconda3

    # create the env and install everything (CPU wheel by default).
    # PICK YOUR OWN ENV NAME here and remember it -- you will reuse it.
    bash setup_env_davinci.sh --env-name meacnn_cpu

This creates the env, installs CPU PyTorch + skopt + pytorch-metric-learning,
writes the libstdc++ activation hook (the davinci GLIBCXX fix), and runs
verify_env_hpc.py at the end. You want to see "ALL CHECKS PASSED".

GPU variant (only if/when you move to GPU -- keep the CPU env too):
    module load cuda/12.1
    module load miniconda3
    bash setup_env_davinci.sh --env-name meacnn_gpu --gpu

--------------------------------------------------------------------------------
## Step 2 -- verify the env (every session, before any job)

    module load miniconda3
    conda activate meacnn_cpu     # your env name
    python verify_env_hpc.py

9 checks, a few seconds. If scipy/CubicSpline fails with a GLIBCXX error, the
libstdc++ hook did not take effect -- run `echo $LD_LIBRARY_PATH` (it should
start with your env's lib dir) and re-activate the env.

--------------------------------------------------------------------------------
## Step 3 -- measure the REAL per-epoch time on davinci

The ~28 h estimate baked into run_search_cpu.pbs assumes ~25 s/epoch. davinci's
CPU may be faster or slower. Measure it before committing walltime:

    conda activate meacnn_cpu
    python estimate_walltime.py --config config_search_3class_hpc.json --device cpu

This runs a short real training (3 epochs, 1 seed, search skipped), measures
seconds/epoch on THIS node, and multiplies by the full budget. It prints a line
like:

    suggested PBS walltime     : 36:12:00   (projection x 1.30 safety margin)
      -> in your .pbs script:  #PBS -l walltime=36:12:00

Note the number. The default 82-run budget (15/15/10, seeds=2, ep=50) is what
this config carries.

--------------------------------------------------------------------------------
## Step 4 -- size the job (walltime vs budget)

Open run_search_cpu.pbs. Two things to reconcile:

  (a) the BUDGET block (five numbers at the top): arch / train / reg trial
      counts, seeds, max epochs. Cost table is in the file header.
  (b) the `#PBS -l walltime=...` directive.

Set walltime to at least what Step 3 suggested. If that exceeds your queue's
walltime cap:

  - EITHER request a longer-walltime queue (add the appropriate `#PBS -q ...`),
  - OR shrink the BUDGET block. Cut trial counts (N_ARCH / N_TRAIN / N_REG) or
    N_SEEDS. Do NOT cut MAX_EPOCHS below ~50 -- too few epochs per trial is the
    documented non-convergence trap (individual trials never separate the
    classes, and the search ends up ranking configs by how fast they train
    rather than how well they cluster).

To go the OTHER way (fuller search) if you have the walltime: set
N_ARCH=30, N_TRAIN=30, N_REG=20, N_SEEDS=3, MAX_EPOCHS=60 (243 runs, ~100 h at
25 s/epoch), and raise the walltime to match. No JSON edit needed -- the block
overrides the config.

Also set, in run_search_cpu.pbs:
    CONDA_ENV="meacnn_cpu"      # your env name (currently a placeholder)

--------------------------------------------------------------------------------
## Step 5 -- optional login-node smoke check (5 min, catches path/env bugs)

    conda activate meacnn_cpu
    python smoke_test_end_to_end.py

Exercises the full pipeline on a tiny synthetic case. If it passes, every code
path the real job hits is known-good in this environment.

--------------------------------------------------------------------------------
## Step 6 -- submit

    qsub run_search_cpu.pbs

Track it:
    qstat -u $USER            # Q = queued, R = running, C/F = done
    tail -f search.out        # live log (stderr merged in via #PBS -j oe)

The script self-protects: it runs verify_env_hpc.py and a --dry-run pre-flight
BEFORE the real search, so a broken env or bad config fails in seconds, not
hours in.

--------------------------------------------------------------------------------
## Step 7 -- read convergence in the log

The full search runs phase 1 (arch) -> phase 2 (train HPs) -> regularization ->
final training + held-out TEST. In `search.out`, convergence looks like:

  - Final TEST line clears a real bar:
        [run] TEST  ARI 0.9xx ...   (near 1.0 on clean 3-class synthetic)
    ARI ~0.05 means it did NOT converge (random-level clustering).
  - During the final train, mined triplet counts FALL across epochs (the
    embedding is genuinely separating, not K-means luck).
  - Silhouette climbs alongside ARI (independent, K-means-free confirmation).
  - config_best.json shows `margin` settled to a sane value (near 0.3), NOT
    pinned at a boundary like 0.84 -- a boundary value is the tell-tale sign of
    an under-explored search (too few trials); bump the BUDGET and resubmit.

--------------------------------------------------------------------------------
## Step 8 -- collect results

Everything lands under:

    out/search_3class_hpc/
      config_input.json                  config as resolved (file + CLI overrides)
      config_best.json                   config after all search phases (the winner)
      results.json                       THE deliverable (test ARI/AMI/silhouette,
                                         per-seed, full budget, best config)
      synthetic_generator_params.json    exact per-class generator params used
      figures/
        synthetic_traces_overview.png    the traces this run trained on
        pdp_phase1_arch.png              partial-dependence of the arch search
        pdp_phase2_train.png             ... the train-HP search
        pdp_regularization.png           ... the regularization search
        embedding_test_seed_<n>.png      final embedding on the held-out test set
      checkpoints/
        best_model.pt                    deployable model, SELECTED ON VALIDATION
        final_seed_<n>.pt                per-seed best-epoch weights
        seed_<n>/{last,best}.pt          resumable checkpoints

The two numbers you care about are in results.json (`test.ari`) and the
`embedding_test_seed_*.png` figures.

--------------------------------------------------------------------------------
## Resuming a died final-training run

If the job dies DURING the final training stage (not the search phases), you can
resume that stage from its last checkpoint:

    # re-submit with --resume added to the RUN block in run_search_cpu.pbs,
    # or run interactively:
    python -u run_optimization.py --config config_search_3class_hpc.json \
        --device cpu --n-calls-arch 15 --n-calls-train 15 --n-calls-reg 10 \
        --n-seeds 2 --max-epochs 50 \
        --out-dir ./out --cache-dir ./cache \
        --experiment-name search_3class_hpc --resume --verbose

IMPORTANT: only the FINAL training stage is resumable. The SEARCH PHASES are
not -- if the job dies mid-search, it restarts from the beginning of the search.
This is why sizing walltime correctly (Steps 3-4) matters: give the whole search
room to finish inside one job.

--------------------------------------------------------------------------------
## Quick reference -- the whole thing, once set up

    module load miniconda3
    conda activate meacnn_cpu
    python verify_env_hpc.py
    python estimate_walltime.py --config config_search_3class_hpc.json --device cpu
    # (edit walltime + CONDA_ENV in run_search_cpu.pbs if not done)
    qsub run_search_cpu.pbs
    tail -f search.out
