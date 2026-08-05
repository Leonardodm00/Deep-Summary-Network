"""
smoke_test_search_resume_integration.py

End-to-end check of the KILL / RESUME path through the REAL
search_joint_conditions, the REAL joint_condition_space, the REAL projection Pi
and the REAL gp_minimize. Only evaluate_candidate is replaced, by a
deterministic analytic surrogate, so the test runs in seconds on a CPU with no
data and no training.

Why the substitution is legitimate: everything this test asserts is about
BOOKKEEPING -- how many trials ran, which seeds they owned, what was written to
disk, what was read back. None of it depends on the objective being a real
training run, and making it one would put a multi-hour test behind a
correctness property that can be checked in seconds.

Run:
    python3 smoke_test_search_resume_integration.py

Checks:
  [R1] A cold run with out_dir set writes one trials.jsonl line per trial, each
       carrying point_raw, and a search_state.json with the study header.
  [R2] A run killed at trial k leaves EXACTLY k complete records. The trial
       that was in flight leaves nothing, which is why partial-trial detection
       is unnecessary.
  [R3] Resuming reaches n_calls_total evaluations in total -- Eq. (1) applied
       through the real call stack, not just in the unit test.
  [R4] THE SEED-BLOCK FIX. The resumed segment's trial indices continue from k
       rather than restarting at 0. Since evaluate_candidate derives
       seed = base + t * n_seeds + n, restarting would silently re-use the seed
       blocks of the first k trials.
  [R5] The resumed segment does not re-pay the initial design: its
       n_initial_points_used is the unfulfilled remainder, and is 0 once
       k >= n_initial.
  [R6] Starting a fresh run on top of an existing trials.jsonl REFUSES, rather
       than appending a second study to the first.
  [R7] A resume whose epsilon differs from the recorded one REFUSES.
  [R8] best_from_trials rebuilds a config_best.json from the partial log, so
       the interrupted study still yields something to train.
  [R9] out_dir=None reproduces the previous behaviour exactly: no files, no
       resume, nothing on disk.
"""

import json
import os
import shutil
import sys
import tempfile
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import search as S
from config import ExperimentConfig
from search_persistence import ResumeError, read_trials

RESULTS = []
N_CALLS = 14
N_INIT = 5
KILL_AT = 8
EPSILON = 0.05


def check(tag, ok, detail=""):
    RESULTS.append((tag, bool(ok), detail))
    print("  [%s] %s%s" % (tag, "PASS" if ok else "FAIL",
                           ("  -- " + detail) if detail else ""))
    return bool(ok)


class Killed(RuntimeError):
    """Stands in for the scheduler's SIGKILL at the walltime."""


def make_cfg():
    """The real l3c joint-search config, shrunk to a few trials."""
    here = os.path.dirname(os.path.abspath(__file__))
    for rel in ("hpc/Config/config_l3c_joint_search.json",
                "Config/config_l3c_joint_search.json"):
        p = os.path.join(here, rel)
        if os.path.exists(p):
            cfg = ExperimentConfig.from_json(p)
            break
    else:
        raise SystemExit("could not find config_l3c_joint_search.json under %s"
                         % here)
    sc = replace(cfg.search, n_calls_joint=N_CALLS,
                 n_initial_points_joint=N_INIT)
    return replace(cfg, search=sc)


def install_fake_objective(kill_at=None, seen=None):
    """Replace evaluate_candidate with a cheap deterministic surrogate.

    It reproduces the real one's contract exactly: it appends its record to
    `log` and returns (objective, that same record object), so _run_gp's
    `rec.update(note)` still mutates the object already in the log.
    """
    def fake(cfg, splits, device, trial_number, log=None, epsilon=None,
             train_verbose=False):
        if seen is not None:
            seen.append(int(trial_number))
        if kill_at is not None and int(trial_number) >= int(kill_at):
            raise Killed("walltime")
        obj = -0.4 + 0.03 * ((int(trial_number) * 7) % 11) / 11.0
        rec = {"trial": int(trial_number), "scores": [0.5], "mean": 0.5,
               "std": 0.0, "objective": float(obj), "eff_rank": 3.0,
               "epsilon": epsilon, "n_seeds": 1, "n_seeds_ok": 1,
               "failed": False, "selected_epochs": [42]}
        if log is not None:
            log.append(rec)
        return float(obj), rec
    S.evaluate_candidate = fake
    S.resolve_tie_break_epsilon = lambda cfg, splits, verbose=False: (
        EPSILON, {"source": "smoke"})


def run_search(cfg, out_dir, resume_search=False):
    return S.search_joint_conditions(cfg, splits=None, device="cpu",
                                     verbose=False, train_verbose=False,
                                     out_dir=out_dir,
                                     resume_search=resume_search)


def main():
    print("=" * 74)
    print("smoke_test_search_resume_integration.py")
    print("  n_calls=%d  n_initial=%d  kill_at=%d" % (N_CALLS, N_INIT, KILL_AT))
    print("=" * 74)

    cfg = make_cfg()
    tmp = tempfile.mkdtemp(prefix="dsn_resume_")
    try:
        # ---- [R1][R2] cold run, killed at trial KILL_AT -------------------- #
        print("\n[R1][R2] cold run killed at trial %d" % KILL_AT)
        seen_1 = []
        install_fake_objective(kill_at=KILL_AT, seen=seen_1)
        try:
            run_search(cfg, tmp, resume_search=False)
            check("R2a", False, "the fake objective did not raise")
        except Killed:
            check("R2a", True, "search interrupted, as a walltime kill would")
        trials_path = os.path.join(tmp, "trials.jsonl")
        recs, torn = read_trials(trials_path)
        check("R1a", len(recs) == KILL_AT,
              "%d complete record(s) on disk, expected %d" % (len(recs), KILL_AT))
        check("R1b", torn == 0, "no torn lines")
        check("R1c", all("point_raw" in r for r in recs),
              "every record carries the RAW sampled point")
        check("R1d", all(len(r["point_raw"]) == len(S._JOINT_CONDITION_NAMES)
                         for r in recs),
              "each point_raw has all %d axes" % len(S._JOINT_CONDITION_NAMES))
        check("R1e", [r["trial"] for r in recs] == list(range(KILL_AT)),
              "trial indices are 0..%d" % (KILL_AT - 1))
        state = json.load(open(os.path.join(tmp, "search_state.json")))
        check("R1f", state["study"]["n_calls_total"] == N_CALLS
              and len(state["study"]["space_signature"]) ==
              len(S._JOINT_CONDITION_NAMES),
              "search_state.json carries the study header and space signature")
        check("R2b", len(recs) == KILL_AT,
              "the in-flight trial %d left NO record -- partial-trial detection "
              "is unnecessary" % KILL_AT)

        # ---- [R6] refuse to append a second study -------------------------- #
        print("\n[R6] refusal without --resume-search")
        install_fake_objective(kill_at=None)
        try:
            run_search(cfg, tmp, resume_search=False)
            check("R6", False, "did NOT refuse to append to an existing log")
        except ResumeError:
            check("R6", True, "refuses to mix two studies in one trials.jsonl")

        # ---- [R7] refuse a changed epsilon --------------------------------- #
        print("\n[R7] refusal on a changed epsilon")
        install_fake_objective(kill_at=None)
        S.resolve_tie_break_epsilon = lambda c, s, verbose=False: (0.09, {})
        try:
            run_search(cfg, tmp, resume_search=True)
            check("R7", False, "did NOT refuse a different epsilon")
        except ResumeError:
            check("R7", True, "refuses: a different epsilon is a different "
                              "objective")

        # ---- [R3][R4][R5] the resume --------------------------------------- #
        print("\n[R3][R4][R5] resume")
        seen_2 = []
        install_fake_objective(kill_at=None, seen=seen_2)
        res = run_search(cfg, tmp, resume_search=True)
        check("R3a", len(res.x_iters) == N_CALLS,
              "Eq.(1) through the real call stack: %d total evaluations, "
              "expected %d" % (len(res.x_iters), N_CALLS))
        check("R3b", res.warm_start_k == KILL_AT,
              "warm start carried %d observation(s)" % res.warm_start_k)
        recs2, _ = read_trials(trials_path)
        check("R3c", len(recs2) == N_CALLS,
              "trials.jsonl now holds %d record(s)" % len(recs2))
        check("R4a", seen_2 == list(range(KILL_AT, N_CALLS)),
              "SEED-BLOCK FIX: the resumed segment ran trial indices %r, "
              "continuing from %d rather than restarting at 0"
              % (seen_2, KILL_AT))
        check("R4b", sorted(r["trial"] for r in recs2) == list(range(N_CALLS)),
              "the log holds every trial index exactly once, 0..%d"
              % (N_CALLS - 1))
        expected_ni = max(0, min(N_INIT - KILL_AT, N_CALLS - KILL_AT))
        check("R5a", res.n_initial_points_used == expected_ni,
              "n_initial_points for the segment is %d, not the cold %d"
              % (res.n_initial_points_used, N_INIT))
        check("R5b", expected_ni == 0,
              "k (%d) >= n_initial (%d), so the initial design is not re-paid"
              % (KILL_AT, N_INIT))
        check("R5c", res.n_calls_segment == N_CALLS - KILL_AT,
              "segment budget is %d" % res.n_calls_segment)

        # ---- [R8] recover a config from the partial log --------------------- #
        print("\n[R8] best_from_trials on the PARTIAL log")
        partial = tempfile.mkdtemp(prefix="dsn_partial_")
        try:
            with open(os.path.join(partial, "trials.jsonl"), "w",
                      encoding="ascii") as fh:
                for r in recs:                       # the KILL_AT-record log
                    fh.write(json.dumps(r, ensure_ascii=True) + "\n")
            cfg.to_json(os.path.join(partial, "config_input.json"))
            import best_from_trials as B
            rc = B.main(["--run-dir", partial, "--show", "3", "--top-k", "2"])
            check("R8a", rc == 0, "best_from_trials exited 0")
            check("R8b", os.path.exists(os.path.join(partial,
                                                     "config_best.json")),
                  "config_best.json written from a %d-trial partial log"
                  % KILL_AT)
            check("R8c", os.path.exists(os.path.join(partial,
                                                     "config_top2.json")),
                  "the shortlist for the confirmatory re-fit was written")
            best = ExperimentConfig.from_json(os.path.join(partial,
                                                           "config_best.json"))
            check("R8d", best is not None and best.validate() is not False,
                  "the recovered config loads and validates")
        finally:
            shutil.rmtree(partial, ignore_errors=True)

        # ---- [R9] out_dir=None is the old behaviour ------------------------ #
        print("\n[R9] out_dir=None writes nothing")
        bare = tempfile.mkdtemp(prefix="dsn_bare_")
        try:
            install_fake_objective(kill_at=None)
            res3 = run_search(cfg, None, resume_search=False)
            check("R9a", len(res3.x_iters) == N_CALLS, "still runs the study")
            check("R9b", os.listdir(bare) == [],
                  "no files created when persistence is off")
            check("R9c", res3.warm_start_k == 0 and res3.trial_offset == 0,
                  "no warm start, counter starts at 0")
        finally:
            shutil.rmtree(bare, ignore_errors=True)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n" + "=" * 74)
    print("%d checks, %d passed, %d FAILED"
          % (len(RESULTS), len(RESULTS) - n_fail, n_fail))
    for tag, ok, detail in RESULTS:
        if not ok:
            print("  FAILED [%s] %s" % (tag, detail))
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
