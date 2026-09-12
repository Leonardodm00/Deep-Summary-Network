# Deep-Summary-Network: search persistence + resume

Applies on top of `main` HEAD. Six files, all under `Main/`.

## Files

| File | Status |
|---|---|
| `Main/search_persistence.py` | NEW - JSONL flush, warm-start reconstruction, resume budget |
| `Main/best_from_trials.py` | NEW - recover `config_best.json` + shortlist from a partial log |
| `Main/smoke_test_search_persistence.py` | NEW - 60 unit checks |
| `Main/smoke_test_search_resume_integration.py` | NEW - 25 end-to-end kill/resume checks |
| `Main/search.py` | MODIFIED - `_run_gp` persistence + warm start; `search_joint_conditions` wiring |
| `Main/run_optimization.py` | MODIFIED - `--resume-search`, `out_dir` threading |
| `Main/objective_utils.py` | MODIFIED - `resolve_n_initial_points` guard `>` -> `>=` |
| `Main/Smoke_Tests/smoke_test_objective_wiring.py` | MODIFIED - updated to the new guard contract |

## Transfer

Move the ARCHIVE with a binary-safe tool (scp / rsync / MobaXterm file browser).
Do NOT paste source into a terminal: a Windows/cp1252 boundary silently
collapses multi-byte sequences and produces a SyntaxError only at job time.
Every file here is pure ASCII, which makes that path harmless, but the archive
closes it anyway.

    scp dsn_search_resume.tar.gz you@davinci:~/
    cd ~/Deep-Summary-Network
    tar -xzf ~/dsn_search_resume.tar.gz --strip-components=1

`--strip-components=1` drops the archive's top-level `delivery/` folder so
`Main/...` lands on `Main/...`.

## Use

Cold start, with persistence on (persistence is automatic; `out_dir` is
threaded from `runtime.out_dir/experiment_name`):

    python3 Main/run_optimization.py --config hpc/Config/config_l3c_joint_search_e100.json

Killed at the walltime. Resume:

    python3 Main/run_optimization.py --config hpc/Config/config_l3c_joint_search_e100.json --resume-search

Or take the best found so far and stop searching:

    python3 Main/best_from_trials.py --run-dir out/<experiment_name> --top-k 5
    python3 Main/run_optimization.py --config out/<experiment_name>/config_best.json --skip-search

## What appears in the run directory

    <out_dir>/<experiment_name>/trials.jsonl        one JSON object per completed trial
    <out_dir>/<experiment_name>/search_state.json   running best, rewritten atomically

`trials.jsonl` is flushed and fsynced after every trial. `cat search_state.json`
while the job runs to see progress.

## Behaviour changes to be aware of

1. `resolve_n_initial_points` now REJECTS `n_initial_points == n_calls`.
   Previously accepted. Equality means every trial is part of the random
   initial design and the surrogate is never fitted. `n_calls == 1` is exempt.

2. A run that finds a non-empty `trials.jsonl` and was NOT given
   `--resume-search` REFUSES to start. Move the file aside or pass the flag.

3. `--resume` (final training) and `--resume-search` (the GP search) are
   different flags. `--resume` never resumed the search; its help text said so
   and now says so more explicitly.

4. Only `search_mode=joint_conditions` is resumable. The staged phases are
   unchanged and write nothing.
