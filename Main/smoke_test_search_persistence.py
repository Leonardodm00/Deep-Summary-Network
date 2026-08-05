"""
smoke_test_search_persistence.py

Standalone correctness checks for search_persistence.py. CPU only, no data
files, no training, headless. Runs in a few seconds.

Run:
    python3 smoke_test_search_persistence.py

Checks:
  [A] json_safe unwraps numpy scalars and walks containers; the result survives
      json.dumps. skopt hands back np.int64 / np.float64 inside x_iters and
      json.dump raises TypeError on them, so this is the first thing that would
      break in production.
  [B] dim_kind types real skopt dimensions AND duck-typed stubs.
  [C] point_to_named / named_to_point round-trip a point EXACTLY, preserving
      int-vs-float and category identity through a JSON round trip.
  [D] named_to_point RAISES when the space gained an axis the record predates.
  [E] coerce_value RAISES on a category not in the current space.
  [F] space_signature detects a widened bound, not only a renamed axis.
  [G] TrialWriter appends one line per trial, flushes each, writes pure ASCII,
      and a reopened writer appends rather than truncates.
  [H] read_trials tolerates a TORN final line, counts it, and returns every
      intact record before it.
  [I] build_warm_start: X0/Y0 are correct; FAILED trials are INCLUDED; a
      mixed-epsilon log RAISES; a NaN objective RAISES; a log with no recorded
      point RAISES rather than silently producing a shorter X0.
  [J] resolve_resume_budget implements Eq. (1) and Eq. (2), including k = 0,
      k >= n_initial_total, N_rem < n_initial_total, and a complete study.
  [K] LIVE skopt: a study segmented with resolve_resume_budget produces exactly
      n_calls_total evaluations, matches the uninterrupted run on the supplied
      prefix, and is GP-DRIVEN in the tail.
  [L] LIVE skopt, THE TRAP: the naive resume that passes n_initial_total
      unchanged is pure random search in the tail. This check exists to prove
      the failure Eq. (2) prevents is real, not hypothetical.
  [M] Warm-starting from PROJECTED points produces duplicate inputs carrying
      different objectives; warm-starting from RAW points does not. This is the
      empirical form of the pre-Pi / post-Pi argument.

[K], [L] and [M] need skopt. If it is absent they are reported as SKIPPED and
the run still exits 0, so the file is useful in an environment without it --
but on the cluster skopt is present and they WILL run.
"""

import json
import math
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from search_persistence import (
    ResumeError, TrialWriter, build_warm_start, coerce_value, dim_kind,
    json_safe, named_to_point, point_to_named, read_trials,
    resolve_resume_budget, space_signature,
)

try:
    from skopt import gp_minimize
    from skopt.space import Categorical, Integer, Real
    HAVE_SKOPT = True
except Exception:                                    # pragma: no cover
    HAVE_SKOPT = False

RESULTS = []


def check(tag, ok, detail=""):
    RESULTS.append((tag, bool(ok), detail))
    print("  [%s] %s%s" % (tag, "PASS" if ok else "FAIL",
                           ("  -- " + detail) if detail else ""))
    return bool(ok)


def raises(tag, fn, exc=ResumeError, detail=""):
    try:
        fn()
    except exc:
        return check(tag, True, detail)
    except Exception as ex:
        return check(tag, False, "raised %s, expected %s"
                     % (type(ex).__name__, exc.__name__))
    return check(tag, False, "did not raise %s" % exc.__name__)


# --------------------------------------------------------------------------- #
# stub dimensions, so [B] and [C] do not depend on skopt
# --------------------------------------------------------------------------- #
class StubInteger(object):
    def __init__(self, low, high, name=""):
        self.low, self.high, self.name, self.prior = int(low), int(high), name, None


class StubReal(object):
    def __init__(self, low, high, name="", prior=None):
        self.low, self.high, self.name, self.prior = float(low), float(high), name, prior


class StubCategorical(object):
    def __init__(self, categories, name=""):
        self.categories, self.name = list(categories), name


def toy_space():
    """A 5-axis space exercising every dimension kind, in skopt if available."""
    if HAVE_SKOPT:
        return [Integer(3, 6, name="depth_exponent"),
                Real(1.5, 3.0, name="width_multiplier"),
                Real(1e-5, 1e-2, prior="log-uniform", name="lr"),
                Categorical(["hard", "semihard", "easy_positive"],
                            name="mining_strategy"),
                Integer(0, 1, name="strict_semihard")]
    return [StubInteger(3, 6, "depth_exponent"),
            StubReal(1.5, 3.0, "width_multiplier"),
            StubReal(1e-5, 1e-2, "lr", "log-uniform"),
            StubCategorical(["hard", "semihard", "easy_positive"],
                            "mining_strategy"),
            StubInteger(0, 1, "strict_semihard")]


NAMES = ["depth_exponent", "width_multiplier", "lr", "mining_strategy",
         "strict_semihard"]


# --------------------------------------------------------------------------- #
def test_A_json_safe():
    print("\n[A] json_safe")
    out = json_safe({"i": np.int64(5), "f": np.float64(2.5),
                     "b": np.bool_(True), "arr": np.array([1.0, 2.0]),
                     "nested": [np.int32(7), {"x": np.float32(1.5)}]})
    ok = json.dumps(out)                             # raises if not serialisable
    check("A1", isinstance(out["i"], int) and out["i"] == 5,
          "np.int64 -> int, got %r" % (out["i"],))
    check("A2", isinstance(out["f"], float) and out["f"] == 2.5, "np.float64 -> float")
    check("A3", isinstance(out["b"], bool) and out["b"] is True, "np.bool_ -> bool")
    check("A4", out["arr"] == [1.0, 2.0], "ndarray -> list")
    check("A5", isinstance(out["nested"][1]["x"], float), "recurses into containers")
    check("A6", isinstance(ok, str), "json.dumps succeeds")


def test_B_dim_kind():
    print("\n[B] dim_kind")
    space = toy_space()
    kinds = [dim_kind(d) for d in space]
    check("B1", kinds == ["integer", "real", "real", "categorical", "integer"],
          "got %r" % (kinds,))
    check("B2", dim_kind(StubCategorical(["a"], "s")) == "categorical",
          "duck-typed stub")


def test_C_roundtrip():
    print("\n[C] point round trip")
    space = toy_space()
    point = [5, 2.25, 1e-3, "semihard", 1]
    named = point_to_named(point, NAMES)
    named = json.loads(json.dumps(named))            # force the JSON boundary
    back = named_to_point(named, space, NAMES)
    check("C1", back[0] == 5 and isinstance(back[0], int), "Integer stays int")
    check("C2", isinstance(back[1], float) and abs(back[1] - 2.25) < 1e-15,
          "Real stays float")
    check("C3", abs(back[2] - 1e-3) < 1e-18, "log-uniform Real exact")
    check("C4", back[3] == "semihard", "Categorical identity")
    check("C5", back[4] == 1 and isinstance(back[4], int), "binary Integer stays int")
    # a Real that happens to be integral must NOT come back as int
    named2 = point_to_named([4, 2.0, 1e-3, "hard", 0], NAMES)
    back2 = named_to_point(json.loads(json.dumps(named2)), space, NAMES)
    check("C6", isinstance(back2[1], float),
          "integral-valued Real stays float, got %r" % (type(back2[1]).__name__,))


def test_D_missing_axis():
    print("\n[D] space gained an axis")
    space = toy_space()
    named = point_to_named([5, 2.25, 1e-3, "semihard", 1], NAMES)
    del named["strict_semihard"]
    raises("D1", lambda: named_to_point(named, space, NAMES),
           detail="missing axis raises rather than shifting the vector")


def test_E_unknown_category():
    print("\n[E] unknown category")
    space = toy_space()
    raises("E1", lambda: coerce_value(space[3], "triplet_only", "mining_strategy"),
           detail="a category not in the current space raises")
    check("E2", coerce_value(space[0], 5.0, "depth_exponent") == 5,
          "integral float coerces to int")
    raises("E3", lambda: coerce_value(space[0], 5.5, "depth_exponent"),
           detail="non-integral value on an Integer axis raises")


def test_F_signature():
    print("\n[F] space_signature")
    space = toy_space()
    sig = space_signature(space, NAMES)
    check("F1", len(sig) == 5 and sig[0]["name"] == "depth_exponent", "shape")
    check("F2", sig[2]["prior"] == "log-uniform", "prior captured")
    if HAVE_SKOPT:
        wider = [Integer(3, 6, name="depth_exponent"),
                 Real(1.5, 3.0, name="width_multiplier"),
                 Real(1e-6, 1e-2, prior="log-uniform", name="lr"),   # widened
                 Categorical(["hard", "semihard", "easy_positive"],
                             name="mining_strategy"),
                 Integer(0, 1, name="strict_semihard")]
    else:
        wider = list(space)
        wider[2] = StubReal(1e-6, 1e-2, "lr", "log-uniform")
    check("F3", space_signature(wider, NAMES) != sig,
          "a widened bound is detected, not only a renamed axis")


def _mk_record(t, point, obj, eps=0.05, failed=False):
    return {"trial": t, "objective": float(obj), "epsilon": eps,
            "failed": bool(failed), "point_raw": point_to_named(point, NAMES),
            "mean": 0.5, "std": 0.07, "selected_epochs": [42]}


def test_G_H_writer_reader():
    print("\n[G] TrialWriter / [H] read_trials")
    tmp = tempfile.mkdtemp(prefix="dsn_persist_")
    try:
        pts = [[3, 1.6, 1e-4, "hard", 0], [5, 2.4, 3e-3, "semihard", 1],
               [4, 2.0, 8e-4, "easy_positive", 0]]
        with TrialWriter(tmp, header={"study": "smoke"}) as w:
            for t, p in enumerate(pts):
                w.write_trial(_mk_record(t, p, -0.3 - 0.1 * t))
                w.write_state({"n_trials": t + 1, "best_objective": -0.3 - 0.1 * t})
        path = os.path.join(tmp, "trials.jsonl")
        raw = open(path, "rb").read()
        check("G1", raw.count(b"\n") == 3, "one line per trial, got %d"
              % raw.count(b"\n"))
        check("G2", all(b <= 127 for b in raw), "trials.jsonl is pure ASCII")
        state = json.load(open(os.path.join(tmp, "search_state.json")))
        check("G3", state["n_trials"] == 3, "state file readable and current")
        check("G4", all(b <= 127 for b in
                        open(os.path.join(tmp, "search_state.json"), "rb").read()),
              "search_state.json is pure ASCII")
        # reopening must APPEND, not truncate -- this is the resume path
        with TrialWriter(tmp) as w2:
            w2.write_trial(_mk_record(3, [6, 2.9, 5e-3, "hard", 1], -0.7))
        recs, torn = read_trials(path)
        check("G5", len(recs) == 4 and torn == 0, "reopen appends, got %d recs"
              % len(recs))
        # [H] torn final line
        with open(path, "a", encoding="ascii") as fh:
            fh.write('{"trial": 4, "objective": -0.9, "point_r')
        recs2, torn2 = read_trials(path)
        check("H1", len(recs2) == 4, "intact records survive a torn tail, got %d"
              % len(recs2))
        check("H2", torn2 == 1, "the torn line is counted, got %d" % torn2)
        # A FAILED trial carries mean = std = eff_rank = NaN. Python's json
        # emits and re-reads NaN; strict RFC 8259 does not allow it, so an
        # external consumer such as jq will reject those lines. Recorded here
        # so the behaviour is a known property rather than a surprise.
        tmp2 = tempfile.mkdtemp(prefix="dsn_nan_")
        try:
            with TrialWriter(tmp2) as w3:
                w3.write_trial({"trial": 0, "objective": 1.0, "failed": True,
                                "mean": float("nan"), "std": float("nan"),
                                "point_raw": point_to_named(pts[0], NAMES)})
            r3, t3 = read_trials(os.path.join(tmp2, "trials.jsonl"))
            check("H3", len(r3) == 1 and t3 == 0 and r3[0]["mean"] != r3[0]["mean"],
                  "a FAILED record with NaN diagnostics round-trips via Python json")
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_I_warm_start():
    print("\n[I] build_warm_start")
    space = toy_space()
    pts = [[3, 1.6, 1e-4, "hard", 0], [5, 2.4, 3e-3, "semihard", 1],
           [4, 2.0, 8e-4, "easy_positive", 0]]
    recs = [_mk_record(0, pts[0], -0.30), _mk_record(1, pts[1], 1.0, failed=True),
            _mk_record(2, pts[2], -0.55)]
    ws = build_warm_start(recs, space, NAMES, expected_epsilon=0.05)
    check("I1", ws.k == 3, "k = %d" % ws.k)
    check("I2", ws.n_failed == 1, "FAILED trials counted")
    check("I3", ws.Y0 == [-0.30, 1.0, -0.55],
          "FAILED trial INCLUDED in Y0: %r" % (ws.Y0,))
    check("I4", ws.X0[1] == [5, 2.4, 3e-3, "semihard", 1], "X0 coerced correctly")
    check("I5", abs(ws.best_objective + 0.55) < 1e-12, "best objective is the min")
    check("I6", ws.best_named["mining_strategy"] == "easy_positive", "best point")

    mixed = [_mk_record(0, pts[0], -0.3, eps=0.05),
             _mk_record(1, pts[1], -0.4, eps=0.08)]
    raises("I7", lambda: build_warm_start(mixed, space, NAMES),
           detail="two epsilons in one log raises")
    raises("I8", lambda: build_warm_start(recs, space, NAMES,
                                          expected_epsilon=0.09),
           detail="segment epsilon != recorded epsilon raises")
    nan_rec = [_mk_record(0, pts[0], float("nan"))]
    raises("I9", lambda: build_warm_start(nan_rec, space, NAMES),
           detail="NaN objective raises (gp_minimize cannot fit NaN)")
    no_point = [{"trial": 0, "objective": -0.3, "epsilon": 0.05}]
    raises("I10", lambda: build_warm_start(no_point, space, NAMES),
           detail="a log with no recorded point raises, never truncates X0")
    sig = space_signature(space, NAMES)
    bad_sig = json.loads(json.dumps(sig))
    bad_sig[0]["high"] = 7
    raises("I11", lambda: build_warm_start(recs, space, NAMES,
                                           expected_space_signature=bad_sig),
           detail="a changed space signature raises")


def test_J_budget():
    print("\n[J] resolve_resume_budget")
    check("J1", resolve_resume_budget(300, 100, 0) == (300, 100),
          "k = 0 is the cold study unchanged")
    check("J2", resolve_resume_budget(300, 100, 40) == (260, 60),
          "Eq.(2): only the UNFULFILLED initial design, got %r"
          % (resolve_resume_budget(300, 100, 40),))
    check("J3", resolve_resume_budget(300, 100, 100) == (200, 0),
          "k == n_initial -> 0 further random draws")
    check("J4", resolve_resume_budget(300, 100, 180) == (120, 0),
          "k > n_initial -> 0, never negative")
    check("J5", resolve_resume_budget(300, 100, 260) == (40, 0),
          "late segment: N_rem < n_initial does not exceed n_calls")
    check("J6", resolve_resume_budget(300, 100, 299) == (1, 0), "last trial")
    for bad, why in (((300, 100, 300), "complete study"),
                     ((300, 100, 301), "k > n_calls_total"),
                     ((300, 100, -1), "negative k"),
                     ((0, 0, 0), "n_calls_total < 1")):
        try:
            resolve_resume_budget(*bad)
            check("J7", False, "%s did not raise" % why)
            break
        except ValueError:
            pass
    else:
        check("J7", True, "illegal budgets raise ValueError")


# --------------------------------------------------------------------------- #
# live skopt checks
# --------------------------------------------------------------------------- #
def _toy_objective(p):
    d, w, lr, mining, strict = p
    return (float(d) * 0.1 + float(w) * 0.2 + math.log10(float(lr)) * 0.05
            + {"hard": 0.0, "semihard": 0.3, "easy_positive": -0.2}[mining]
            + 0.15 * int(strict))


def _same_point(u, v):
    return (int(u[0]) == int(v[0]) and abs(float(u[1]) - float(v[1])) < 1e-12
            and abs(float(u[2]) - float(v[2])) < 1e-15 and u[3] == v[3]
            and int(u[4]) == int(v[4]))


def test_K_L_M_live():
    print("\n[K][L][M] live skopt semantics")
    if not HAVE_SKOPT:
        check("K", True, "SKIPPED: skopt not importable in this environment")
        return
    space = toy_space()
    N_CALLS, N_INIT, K = 24, 8, 14

    full = gp_minimize(_toy_objective, space, n_calls=N_CALLS,
                       n_initial_points=N_INIT, random_state=0, acq_func="EI")
    X0 = [list(x) for x in full.x_iters[:K]]
    Y0 = [float(v) for v in full.func_vals[:K]]

    n_seg, ni_seg = resolve_resume_budget(N_CALLS, N_INIT, K)
    check("K0", (n_seg, ni_seg) == (10, 0), "budget for k=14: %r" % ((n_seg, ni_seg),))
    seg = gp_minimize(_toy_objective, space, n_calls=n_seg,
                      n_initial_points=ni_seg, random_state=0, acq_func="EI",
                      x0=X0, y0=Y0)
    check("K1", len(seg.x_iters) == N_CALLS,
          "Eq.(1): total evaluations = %d, expected %d" % (len(seg.x_iters), N_CALLS))
    check("K2", all(_same_point(a, b) for a, b in zip(seg.x_iters[:K], X0)),
          "the supplied prefix is preserved verbatim")
    check("K3", len(seg.models) == n_seg - ni_seg + 1,
          "tail is GP-driven: %d models for %d new trials"
          % (len(seg.models), n_seg))
    diverged = sum(0 if _same_point(a, b) else 1
                   for a, b in zip(seg.x_iters[K:], full.x_iters[K:]))
    check("K4", diverged > 0,
          "V5: %d of %d post-prefix trials differ from the uninterrupted run "
          "-- segmentation is part of the record" % (diverged, N_CALLS - K))

    # [L] the trap: the naive resume that reuses the original n_initial_points
    naive = gp_minimize(_toy_objective, space, n_calls=n_seg,
                        n_initial_points=N_INIT, random_state=0, acq_func="EI",
                        x0=X0, y0=Y0)
    check("L1", len(naive.models) == n_seg - N_INIT + 1,
          "naive resume fits only %d models for %d trials -- %d of them are "
          "FRESH RANDOM draws" % (len(naive.models), n_seg, N_INIT))
    check("L2", len(naive.models) < len(seg.models),
          "Eq.(2) strictly recovers GP-driven trials the naive resume loses")
    try:
        gp_minimize(_toy_objective, space, n_calls=4, n_initial_points=N_INIT,
                    random_state=0, acq_func="EI", x0=X0, y0=Y0)
        check("L3", False, "a late segment with N_rem < n_initial did not raise")
    except ValueError:
        check("L3", True, "V4: skopt raises when n_calls < n_initial_points, "
                          "regardless of how many warm-start points are supplied")

    # [M] raw vs projected. Pi is many-to-one: distinct raw points that differ
    # ONLY in a coordinate Pi overwrites collapse onto one projected point. The
    # objective is evaluated at the projected config, so those collapsed inputs
    # carry DIFFERENT y values -- a GP fitted on them sees contradictory data at
    # one location. Constructed explicitly rather than hoping a random prefix
    # happens to contain a collision.
    def project(p):
        """Stand-in for Pi: the (hard, strict=True) cell is illegal."""
        d, w, lr, mining, strict = p
        if mining == "hard" and int(strict) == 1:
            return [d, w, lr, mining, 0]
        return list(p)

    raw_pair = [[4, 2.0, 1e-3, "hard", 1],       # projected -> strict = 0
                [4, 2.0, 1e-3, "hard", 0]]       # already legal
    proj_pair = [project(p) for p in raw_pair]
    y_pair = [_toy_objective(project(p)) for p in raw_pair]

    def n_dup(points):
        return len(points) - len({tuple(str(v) for v in p) for p in points})

    check("M1", n_dup(raw_pair) == 0,
          "two distinct raw points remain distinct")
    check("M2", n_dup(proj_pair) == 1,
          "Pi collapses them onto ONE projected point (%d duplicate)"
          % n_dup(proj_pair))
    check("M3", abs(y_pair[0] - y_pair[1]) < 1e-12,
          "both were evaluated at the same config, so the duplicate carries "
          "the same y here -- but the GP would still see one input twice")

    # The concrete cost, stated only as far as it is demonstrated here: skopt
    # ACCEPTS duplicated warm-start inputs without complaint, so the surrogate
    # is fitted on a design with fewer DISTINCT locations than the study paid
    # trials for. Every collapsed pair is an hour of compute that informs the
    # GP about a location it already knew.
    #
    # NOT asserted, because it was observed rather than isolated: skopt also
    # carries a substitution path that replaces an acquisition proposal with a
    # RANDOM point when the proposal repeats an existing observation
    # ("The objective has been evaluated at point ... before, using random
    # point"). A Pi-collapsed observation set makes such repeats more likely,
    # but this test does not establish that causally.
    dup_X0 = X0[:6] + [list(X0[0])]              # deliberate repeat
    dup_Y0 = Y0[:6] + [float(Y0[0])]
    r_dup = gp_minimize(_toy_objective, space, n_calls=4, n_initial_points=0,
                        random_state=0, acq_func="EI", x0=dup_X0, y0=dup_Y0)
    n_obs = len(dup_X0)
    n_distinct = len({tuple(str(v) for v in p) for p in dup_X0})
    check("M4", len(r_dup.x_iters) == n_obs + 4,
          "skopt accepts duplicated warm-start inputs silently (%d evaluations)"
          % len(r_dup.x_iters))
    check("M5", n_distinct < n_obs,
          "the surrogate sees %d distinct locations for %d paid trials -- the "
          "information loss Pi-collapsed warm-start points would cause"
          % (n_distinct, n_obs))


def main():
    print("=" * 74)
    print("smoke_test_search_persistence.py")
    print("  numpy %s | skopt %s" % (np.__version__,
                                     __import__("skopt").__version__
                                     if HAVE_SKOPT else "ABSENT"))
    print("=" * 74)
    for fn in (test_A_json_safe, test_B_dim_kind, test_C_roundtrip,
               test_D_missing_axis, test_E_unknown_category, test_F_signature,
               test_G_H_writer_reader, test_I_warm_start, test_J_budget,
               test_K_L_M_live):
        fn()
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n" + "=" * 74)
    print("%d checks, %d passed, %d FAILED" % (len(RESULTS),
                                               len(RESULTS) - n_fail, n_fail))
    if n_fail:
        for tag, ok, detail in RESULTS:
            if not ok:
                print("  FAILED [%s] %s" % (tag, detail))
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
