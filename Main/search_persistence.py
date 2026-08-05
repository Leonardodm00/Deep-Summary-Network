"""
search_persistence.py
=====================

Per-trial persistence and warm-start reconstruction for the joint condition
search. Two capabilities, in dependency order:

  A. INSPECTABLE PARTIAL RESULTS. Every completed trial is appended to
     <out_dir>/trials.jsonl and flushed immediately, so a job killed at the
     walltime leaves a machine-readable record of the k trials it completed
     instead of nothing. A running search_state.json carries the best point so
     far, so "how is it going" is a cat rather than a parse.

  B. RESUME. The k records are read back into the (X0, Y0) warm-start pair
     gp_minimize accepts, so a study can be run as a sequence of walltime-
     bounded segments.

Separation of concerns (directive 2): this module PERSISTS ONLY. It does not
search, does not train, does not build configs, does not touch skopt beyond
reading the type of a dimension object. search.py owns the search; this module
owns the bytes on disk.

WHY THE RAW POINT AND NOT THE PROJECTED ONE
-------------------------------------------
The legality projection Pi runs INSIDE config_from_joint_condition_point, i.e.
inside the objective. gp_minimize therefore never sees a projected point: its
x_iters, and the surrogate fitted on them, hold the RAW coordinates the
acquisition function proposed. A resume that supplies projected points would
fit a DIFFERENT surrogate than the uninterrupted run would have held, and would
additionally collapse the roughly one third of trials that Pi moves onto
duplicate inputs carrying different objective values -- degenerate for a GP.

So X0 is built from the RAW point. The projected point is recorded too, because
a trial whose coordinates said (hard, joint_sep, strict=True) but which trained
(hard, joint_sep, strict=False) must be readable as such, but it is recorded for
the reader, not for the surrogate.

VERIFIED skopt SEMANTICS (measured against skopt 0.10.2, not assumed)
---------------------------------------------------------------------
  V1. n_calls EXCLUDES the points supplied in x0/y0. Supplying k = 12 points
      with n_calls = 20 produced 32 evaluations. A resume must therefore pass
      n_calls = N_rem = n_calls_total - k, never n_calls_total.

  V2. n_initial_points is NOT satisfied by x0/y0. The number of surrogate fits
      is n_calls - n_initial_points + 1 whether or not warm-start points were
      supplied. A resume that passes the ORIGINAL n_initial_points therefore
      draws that many FRESH random points in every segment: a 300-trial study
      run as 3 segments at n_initial_points = 100 would be 300 random draws and
      zero Bayesian optimisation, silently. This is the single most dangerous
      property of the resume and resolve_resume_budget exists to prevent it.

  V3. n_initial_points = 0 is LEGAL and is what a segment with k >= n_initial
      wants: every trial in that segment is GP-driven. Measured: n_calls = 6
      with n_initial_points = 0 fitted 7 models.

  V4. skopt validates n_calls >= n_initial_points BEFORE considering x0, and
      raises ValueError otherwise. A late segment with N_rem < n_initial would
      CRASH if the original value were passed through.

  V5. A resumed study does not reproduce the trial sequence of an uninterrupted
      one at the same random_state: the supplied k points match by construction
      and everything after them diverges. This is expected -- the acquisition
      path depends on the surrogate's fitting history -- but it means the
      segmentation boundaries are part of the experimental record, and
      write_state records them.

WHAT IS DELIBERATELY NOT HANDLED HERE
--------------------------------------
Partial trials need no special handling: a record is written only after the
objective returns, so a job killed during trial k+1 simply leaves no line for
it. The one residual hazard is a torn final line, which read_trials tolerates
by design (that is the whole reason for JSON Lines over a JSON array) and
counts, so the caller can report it rather than discover it.
"""

from __future__ import annotations

import json
import os
import tempfile
import time

__all__ = [
    "SCHEMA_VERSION",
    "TRIALS_FILENAME",
    "STATE_FILENAME",
    "json_safe",
    "dim_kind",
    "coerce_value",
    "point_to_named",
    "named_to_point",
    "space_signature",
    "TrialWriter",
    "read_trials",
    "WarmStart",
    "build_warm_start",
    "resolve_resume_budget",
    "ResumeError",
]

SCHEMA_VERSION = 1
TRIALS_FILENAME = "trials.jsonl"
STATE_FILENAME = "search_state.json"


class ResumeError(RuntimeError):
    """A resume cannot be performed safely. Always raised, never warned.

    Every condition that reaches this class is one where continuing would
    optimise a different objective, or a differently-shaped space, than the
    completed trials were scored under. Degrading silently is exactly the
    failure mode the whole module exists to prevent.
    """


# --------------------------------------------------------------------------- #
# JSON safety
# --------------------------------------------------------------------------- #
def json_safe(value):
    """A JSON-serialisable echo of value, with numpy scalars unwrapped.

    skopt hands back numpy scalars (np.int64, np.float64, np.bool_) inside
    x_iters, and json.dump raises TypeError on all three. Unwrapping via
    .item() rather than int()/float() preserves the distinction between an
    integer axis and a real one, which named_to_point later depends on.

    Containers are walked recursively. Anything not recognised is returned
    unchanged, so json.dump still raises on a genuinely unserialisable object
    rather than this function silently stringifying it.
    """
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float, str)) or value is None:
        return value
    item = getattr(value, "item", None)
    if item is not None and getattr(value, "shape", None) == ():
        return json_safe(item())
    if isinstance(value, dict):
        return dict((str(k), json_safe(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    tolist = getattr(value, "tolist", None)
    if tolist is not None:
        return json_safe(tolist())
    return value


# --------------------------------------------------------------------------- #
# skopt dimension typing
# --------------------------------------------------------------------------- #
def dim_kind(dim):
    """One of "integer", "real", "categorical" for a skopt Dimension.

    Typed by class name first and by duck typing second, so this module never
    has to import skopt. That keeps it testable against a stub space, which the
    smoke test uses to check the coercion logic without a GP anywhere near it.

    The distinction is load-bearing, not cosmetic. JSON has one number type, so
    a depth_exponent written as 5 reads back as int 5 but a width_multiplier
    written as 2.0 also reads back as int 2 on some encoders. Feeding an int
    where skopt expects a Real, or a float where it expects an Integer, either
    raises inside the surrogate or silently rounds the coordinate.
    """
    name = type(dim).__name__
    if name in ("Integer", "Real", "Categorical"):
        return {"Integer": "integer", "Real": "real",
                "Categorical": "categorical"}[name]
    if hasattr(dim, "categories"):
        return "categorical"
    low = getattr(dim, "low", None)
    if low is not None and isinstance(low, int) and not isinstance(low, bool):
        return "integer"
    if low is not None:
        return "real"
    raise ResumeError("cannot type skopt dimension %r (class %s)" % (dim, name))


def coerce_value(dim, value, axis_name=""):
    """value cast to the type dimension dim requires, or ResumeError.

    For a categorical axis the value must be one of the declared categories.
    An unknown category means the space changed between segments -- e.g.
    loss_type_choices was edited -- and the completed trials are no longer
    points of the space being searched. That is not repairable by coercion and
    must not be papered over.
    """
    kind = dim_kind(dim)
    where = (" on axis %r" % axis_name) if axis_name else ""
    if kind == "categorical":
        cats = list(dim.categories)
        if value in cats:
            return cats[cats.index(value)]
        for c in cats:                       # tolerate int/str drift via JSON
            if str(c) == str(value):
                return c
        raise ResumeError(
            "value %r%s is not among the declared categories %r: the search "
            "space changed since the trial was recorded, so the completed "
            "trials are not points of the space now being searched."
            % (value, where, cats))
    if kind == "integer":
        try:
            out = int(value)
        except (TypeError, ValueError):
            raise ResumeError("cannot cast %r%s to int" % (value, where))
        if float(out) != float(value):
            raise ResumeError(
                "value %r%s is not an integer, but the axis is Integer"
                % (value, where))
        return out
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ResumeError("cannot cast %r%s to float" % (value, where))


# --------------------------------------------------------------------------- #
# point <-> named dict
# --------------------------------------------------------------------------- #
def point_to_named(point, names):
    """{axis_name: value} for a sampled point, JSON-safe.

    A dict rather than a bare list is deliberate. A list is positional, so a
    later edit that reorders the space would silently reinterpret every stored
    coordinate as a different axis -- the exact class of bug the repository's
    get_newspace notes call BUG 1. With names attached, a reordering is
    detected by named_to_point as a lookup, and an added or removed axis raises
    instead of shifting the vector.
    """
    point = list(point)
    names = list(names)
    if len(point) != len(names):
        raise ResumeError(
            "point has %d coordinates but %d axis names were supplied"
            % (len(point), len(names)))
    return dict((str(n), json_safe(v)) for n, v in zip(names, point))


def named_to_point(named, space, names):
    """The positional point list for space, rebuilt from a {name: value} dict.

    Raises ResumeError on a missing axis. An axis present in the record but
    absent from names is IGNORED, because dropping an axis from the space is a
    deliberate act whose consequence (the recorded trials are still valid
    points of the smaller space, projected) is the caller's to accept; adding
    one is not recoverable, because no recorded trial has a value for it.
    """
    names = list(names)
    space = list(space)
    if len(space) != len(names):
        raise ResumeError(
            "space has %d dimensions but %d axis names were supplied"
            % (len(space), len(names)))
    out = []
    for dim, name in zip(space, names):
        if name not in named:
            raise ResumeError(
                "trial record has no value for axis %r: the space gained an "
                "axis since the trial was recorded, so the completed trials "
                "cannot be lifted into it." % (name,))
        out.append(coerce_value(dim, named[name], axis_name=name))
    return out


# --------------------------------------------------------------------------- #
# space signature
# --------------------------------------------------------------------------- #
def space_signature(space, names):
    """A comparable, JSON-safe description of the search space.

    Written into search_state.json at the head of a study and checked on
    resume. Bounds are included, not only names and types: a resume into a
    space whose lr_range was widened is optimising a different problem, and the
    completed trials would be an unrepresentative sample of the new space
    rather than a warm start for it.
    """
    space = list(space)
    names = list(names)
    if len(space) != len(names):
        raise ResumeError(
            "space has %d dimensions but %d axis names were supplied"
            % (len(space), len(names)))
    sig = []
    for dim, name in zip(space, names):
        kind = dim_kind(dim)
        entry = {"name": str(name), "kind": kind}
        if kind == "categorical":
            entry["categories"] = [json_safe(c) for c in dim.categories]
        else:
            entry["low"] = json_safe(getattr(dim, "low", None))
            entry["high"] = json_safe(getattr(dim, "high", None))
            prior = getattr(dim, "prior", None)
            entry["prior"] = None if prior is None else str(prior)
        sig.append(entry)
    return sig


# --------------------------------------------------------------------------- #
# writer
# --------------------------------------------------------------------------- #
class TrialWriter(object):
    """Append-and-flush writer for trials.jsonl, plus an atomic state file.

    One JSON object per line. JSON Lines rather than a JSON array because a
    partially written array is not parseable at all, whereas a JSONL file torn
    mid-line loses exactly one record and every line before it still reads.

    Durability. Each write is followed by flush() and, when
    fsync_every_trial is true, os.fsync(). fsync costs a few milliseconds
    against a trial that takes about an hour, so it is on by default: the
    scenario this module exists for is a SIGKILL at the walltime, and a record
    sitting in the kernel page cache when the node goes down is a record that
    was never written. On a parallel filesystem the atomicity of a single small
    append is not guaranteed by POSIX, which is why read_trials tolerates a
    torn line rather than assuming there will not be one.

    ASCII. Records are written with ensure_ascii=True, matching
    run_optimization._write_json_ascii, so the artifact survives any locale.
    """

    def __init__(self, out_dir, header=None, fsync_every_trial=True):
        self.out_dir = str(out_dir)
        self.fsync_every_trial = bool(fsync_every_trial)
        os.makedirs(self.out_dir, exist_ok=True)
        self.trials_path = os.path.join(self.out_dir, TRIALS_FILENAME)
        self.state_path = os.path.join(self.out_dir, STATE_FILENAME)
        self._fh = open(self.trials_path, "a", encoding="ascii")
        self._t0 = time.time()
        self.n_written = 0
        self.header = dict(header or {})

    # -- trials ------------------------------------------------------------- #
    def write_trial(self, record):
        """Append one trial record and flush it. Returns the record written."""
        payload = json_safe(dict(record))
        payload.setdefault("schema_version", SCHEMA_VERSION)
        payload.setdefault("wall_elapsed_s", float(time.time() - self._t0))
        line = json.dumps(payload, ensure_ascii=True, sort_keys=True)
        self._fh.write(line + "\n")
        self._fh.flush()
        if self.fsync_every_trial:
            try:
                os.fsync(self._fh.fileno())
            except OSError:
                pass                 # a logging fault must never cost a trial
        self.n_written += 1
        return payload

    # -- state -------------------------------------------------------------- #
    def write_state(self, state):
        """Rewrite search_state.json atomically (temp file then os.replace).

        Atomic because this file is rewritten every trial and is the file a
        human will cat while the job runs; a torn read of it would be
        misleading in a way a torn JSONL line is not.
        """
        payload = json_safe(dict(state))
        payload.setdefault("schema_version", SCHEMA_VERSION)
        payload.setdefault("wall_elapsed_s", float(time.time() - self._t0))
        if self.header:
            payload.setdefault("study", self.header)
        text = json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2)
        fd, tmp = tempfile.mkstemp(dir=self.out_dir, prefix=".state-",
                                   suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="ascii") as fh:
                fh.write(text + "\n")
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
            os.replace(tmp, self.state_path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        return self.state_path

    def close(self):
        if self._fh is not None and not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# --------------------------------------------------------------------------- #
# reader
# --------------------------------------------------------------------------- #
def read_trials(path):
    """(records, n_torn) parsed from a JSONL trial log.

    A line that does not parse is COUNTED, not raised on, and is not returned.
    In practice there is at most one such line and it is the last, from a
    process killed mid-write. Counting it lets the caller state how many trials
    were recovered against how many were attempted instead of guessing.
    """
    records, n_torn = [], 0
    if not os.path.exists(path):
        return records, n_torn
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                n_torn += 1
                continue
            if isinstance(obj, dict):
                records.append(obj)
            else:
                n_torn += 1
    return records, n_torn


# --------------------------------------------------------------------------- #
# warm start
# --------------------------------------------------------------------------- #
class WarmStart(object):
    """The (X0, Y0) pair plus everything a caller must report about it."""

    def __init__(self, X0, Y0, k, n_torn, n_failed, epsilon, best_objective,
                 best_named, trial_indices):
        self.X0 = X0
        self.Y0 = Y0
        self.k = int(k)
        self.n_torn = int(n_torn)
        self.n_failed = int(n_failed)
        self.epsilon = epsilon
        self.best_objective = best_objective
        self.best_named = best_named
        self.trial_indices = list(trial_indices)

    def __repr__(self):
        return ("WarmStart(k=%d, n_failed=%d, n_torn=%d, best=%s)"
                % (self.k, self.n_failed, self.n_torn,
                   "None" if self.best_objective is None
                   else "%+.4f" % self.best_objective))


def build_warm_start(records, space, names, expected_epsilon=None,
                     expected_space_signature=None, point_key="point_raw",
                     n_torn=0):
    """Reconstruct (X0, Y0) from completed trial records.

    Parameters
    ----------
    records  : the list returned by read_trials.
    space    : the skopt dimension list currently being searched.
    names    : axis names, in the same order as space.
    expected_epsilon : the tie-break weight resolved for THIS segment, or None
        to skip the check. If given, every record carrying an epsilon must
        match it to within 1e-12 relative, or ResumeError. epsilon rescales the
        secondary term of J_eps = -(ARI + eps * sbar), so a segment that
        re-resolves it to a different value is minimising a different function
        and Y0 from earlier segments is not commensurable with new trials.
    expected_space_signature : the signature written at the head of the study,
        or None to skip. Compared field for field.
    point_key : which recorded point to use. Defaults to the RAW point, which
        is the only correct choice for a GP warm start -- see the module
        docstring. Overridable only so the smoke test can demonstrate that the
        projected point produces duplicate inputs.

    FAILED trials are INCLUDED, deliberately. A point that scored
    FAILED_OBJECTIVE = +1.0 is a real observation: the uninterrupted study's
    surrogate was fitted on it and learned to avoid that region. Dropping such
    trials on resume would hand the GP a rosier picture of the space than the
    trials actually support and invite it back into a region already known to
    be invalid.
    """
    if expected_space_signature is not None:
        current = space_signature(space, names)
        if current != expected_space_signature:
            raise ResumeError(
                "the search space does not match the one recorded at the head "
                "of this study. Resuming would warm-start a different problem "
                "with the old problem's observations.\n  recorded: %s\n  "
                "current:  %s" % (json.dumps(expected_space_signature,
                                             sort_keys=True),
                                  json.dumps(current, sort_keys=True)))

    X0, Y0, indices = [], [], []
    n_failed = 0
    seen_epsilon = None
    for rec in records:
        if point_key not in rec:
            raise ResumeError(
                "trial record %r carries no %r field. A log written before "
                "per-trial point recording was added cannot support a resume: "
                "the sampled coordinates were never stored."
                % (rec.get("trial"), point_key))
        y = rec.get("objective", None)
        if y is None:
            raise ResumeError("trial record %r carries no objective"
                              % (rec.get("trial"),))
        y = float(y)
        if y != y:                                   # NaN
            raise ResumeError(
                "trial record %r has a NaN objective. gp_minimize cannot fit "
                "NaN; the study that wrote this record was already broken."
                % (rec.get("trial"),))
        eps = rec.get("epsilon", None)
        if eps is not None:
            eps = float(eps)
            if seen_epsilon is None:
                seen_epsilon = eps
            elif abs(eps - seen_epsilon) > 1e-12 * max(1.0, abs(seen_epsilon)):
                raise ResumeError(
                    "the trial log contains two different tie-break weights "
                    "epsilon (%r and %r). The objective changed mid-study, so "
                    "the recorded objectives are not comparable."
                    % (seen_epsilon, eps))
        if bool(rec.get("failed", False)):
            n_failed += 1
        X0.append(named_to_point(rec[point_key], space, names))
        Y0.append(y)
        indices.append(rec.get("trial"))

    if expected_epsilon is not None and seen_epsilon is not None:
        if abs(float(expected_epsilon) - seen_epsilon) > 1e-12 * max(
                1.0, abs(seen_epsilon)):
            raise ResumeError(
                "epsilon resolved for this segment (%r) differs from the "
                "epsilon the completed trials were scored under (%r). Pin it "
                "or the objective changes mid-study."
                % (float(expected_epsilon), seen_epsilon))

    best_objective, best_named = None, None
    if Y0:
        i_best = min(range(len(Y0)), key=lambda i: Y0[i])
        best_objective = Y0[i_best]
        best_named = dict(records[i_best].get(point_key, {}))

    return WarmStart(X0=X0, Y0=Y0, k=len(X0), n_torn=int(n_torn),
                     n_failed=n_failed, epsilon=seen_epsilon,
                     best_objective=best_objective, best_named=best_named,
                     trial_indices=indices)


# --------------------------------------------------------------------------- #
# the budget arithmetic -- the part that silently destroys a study if wrong
# --------------------------------------------------------------------------- #
def resolve_resume_budget(n_calls_total, n_initial_total, k):
    """(n_calls_segment, n_initial_segment) for a segment resuming after k trials.

    Two corrections, both measured against skopt 0.10.2 rather than assumed:

      n_calls_segment = n_calls_total - k                             (Eq. 1)

        because n_calls EXCLUDES x0/y0. Passing n_calls_total would run a
        study of n_calls_total + k trials and overrun the very walltime the
        resume exists to respect.

      n_initial_segment = max(0, min(n_initial_total - k,
                                     n_calls_segment))                (Eq. 2)

        because n_initial_points is NOT satisfied by x0/y0: skopt draws that
        many fresh random points in EVERY segment. Passing n_initial_total
        unchanged turns a J-segment study into J * n_initial_total random
        draws. At n_initial_total = 100 and three segments of 100, the entire
        300-trial study would be random search with no error and no warning.
        Once k >= n_initial_total the initial design is already paid for and 0
        is correct: measured, n_initial_points = 0 is legal and yields a fully
        GP-driven segment.

        The min() against n_calls_segment additionally prevents the
        ValueError skopt raises when n_calls < n_initial_points, which a late
        segment (N_rem = 50 against n_initial_total = 100) would otherwise hit.

    Raises ValueError if k is not a legal trial count, or if the study is
    already complete (n_calls_segment < 1), which the caller should treat as
    "nothing to resume" rather than as an error to swallow.
    """
    n_calls_total = int(n_calls_total)
    n_initial_total = int(n_initial_total)
    k = int(k)
    if n_calls_total < 1:
        raise ValueError("n_calls_total must be >= 1; got %d" % n_calls_total)
    if k < 0:
        raise ValueError("k must be >= 0; got %d" % k)
    if k > n_calls_total:
        raise ValueError(
            "k (%d) exceeds n_calls_total (%d): the log holds more completed "
            "trials than the study budget allows." % (k, n_calls_total))
    n_calls_segment = n_calls_total - k
    if n_calls_segment < 1:
        raise ValueError(
            "the study is already complete: k = %d of n_calls_total = %d. "
            "There is nothing to resume." % (k, n_calls_total))
    n_initial_segment = max(0, min(n_initial_total - k, n_calls_segment))
    return n_calls_segment, n_initial_segment
