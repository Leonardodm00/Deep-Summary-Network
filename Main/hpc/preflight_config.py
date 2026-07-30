"""
preflight_config.py

Report the DERIVED quantities of an ExperimentConfig -- the numbers that are not
in the JSON but that decide whether a run is admissible and what it will cost --
WITHOUT generating a single trace. Trace generation for a 45 x 600 s latent
study takes minutes; this takes milliseconds, so it is the thing to run before
queueing a cluster job.

What it reports
---------------
  * windowing      : T (samples), windows per trace for the train and eval strides
  * split          : cultures per class per split, W_min, N_train
  * batch geometry : U_eff (Eq. 3), n_g (Eq. 2a), M (Eq. 2b), and every cap
  * cost proxy     : M * T, the sample-values one forward pass must embed
  * gates          : every condition that RAISES, and every one that only WARNS

What it does NOT do
-------------------
It does not build traces, splits, datasets, or a model, so it cannot catch a
failure that depends on the realised data (e.g. a class whose cultures all landed
in one split under an unlucky permutation). It assumes every generated trace has
the same duration, which is true for data_mode in {"synthetic", "latent"} and NOT
guaranteed for {"real", "numpy"} -- for those it reports the windowing per unit
duration and skips the split arithmetic, saying so.

Usage
-----
    cd Main && PYTHONPATH=. python3 hpc/preflight_config.py hpc/<config>.json

HPC note (hpc-python-compat): pure ASCII, LF endings.
"""

import math
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ExperimentConfig                                # noqa: E402
from batch_geometry import (                                       # noqa: E402
    EASY_POSITIVE_STRATEGIES, DEFAULT_Q_CAP_FRACTION,
)


def _windows_per_trace(duration_s, window_s, stride_s):
    """floor((T_rec - window_s) / stride) + 1, or 0 when the trace is too short."""
    if duration_s < window_s:
        return 0
    return int(math.floor((float(duration_s) - float(window_s))
                          / float(stride_s))) + 1


def _apportion_largest_remainder(n, fractions):
    """Split n items into len(fractions) parts by the largest-remainder rule."""
    raw = [n * f for f in fractions]
    base = [int(math.floor(r)) for r in raw]
    leftover = n - sum(base)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - base[i], reverse=True)
    for i in range(leftover):
        base[order[i % len(order)]] += 1
    return base


def preflight(path, verbose=True):
    problems = {"raise": [], "warn": []}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = ExperimentConfig.from_json(path)
        cfg.validate()
    for w in caught:
        problems["warn"].append(str(w.message))

    d, t = cfg.data, cfg.train
    C = len(d.synthetic_n_per_class)
    fs = float(d.synthetic_fs)
    T = int(round(float(d.window_s) * fs))
    generated = d.data_mode in ("synthetic", "latent")

    lines = []
    lines.append("CONFIG: %s" % path)
    lines.append("  data_mode=%s  positives_mode=%s  split_mode=%s  mining=%s"
                 % (d.data_mode, d.positives_mode, d.split_mode,
                    t.mining_strategy))
    lines.append("")
    lines.append("WINDOWING")
    lines.append("  C = %d classes, traces per class = %s"
                 % (C, list(d.synthetic_n_per_class)))
    lines.append("  window_s = %.4g s at fs = %.4g Hz  ->  T = %d samples"
                 % (d.window_s, fs, T))
    if generated:
        w_tr = _windows_per_trace(d.synthetic_duration_s, d.window_s,
                                  d.train_stride_s)
        w_ev = _windows_per_trace(d.synthetic_duration_s, d.window_s,
                                  d.eval_stride_s)
        lines.append("  trace duration = %.4g s  ->  %d train windows/trace "
                     "(stride %.4g s), %d eval windows/trace (stride %.4g s)"
                     % (d.synthetic_duration_s, w_tr, d.train_stride_s,
                        w_ev, d.eval_stride_s))
        if w_tr < 1:
            problems["raise"].append(
                "window_s (%.4g s) exceeds the trace duration (%.4g s): no "
                "training window can be cut." % (d.window_s,
                                                 d.synthetic_duration_s))
    else:
        w_tr = w_ev = None
        lines.append("  data_mode=%r: trace durations are external, so the "
                     "split and geometry arithmetic below is SKIPPED."
                     % (d.data_mode,))

    # ---- split arithmetic (trace mode only; time_segment cuts the time axis) --
    lines.append("")
    lines.append("SPLIT")
    n_train_windows = None
    u_available = None
    w_min = None
    if not generated:
        lines.append("  skipped (see above)")
    elif d.split_mode == "trace":
        if d.trace_split_mode != "fractional":
            lines.append("  trace_split_mode=%r: leave-one-out fold sizes are "
                         "not modelled here; run the real splitter."
                         % (d.trace_split_mode,))
        else:
            per_split = [_apportion_largest_remainder(int(n),
                                                      list(d.split_fractions))
                         for n in d.synthetic_n_per_class]
            tr = [p[0] for p in per_split]
            va = [p[1] for p in per_split]
            te = [p[2] for p in per_split]
            u_available = min(tr)
            w_min = w_tr                      # equal-duration traces
            n_train_windows = sum(tr) * w_tr
            lines.append("  whole-culture split, fractions %s (%s)"
                         % (list(d.split_fractions), d.trace_alloc_rule))
            lines.append("    train cultures/class = %s  (U_available = %d)"
                         % (tr, u_available))
            lines.append("    val   cultures/class = %s" % (va,))
            lines.append("    test  cultures/class = %s" % (te,))
            lines.append("    W_min = %d windows in the smallest train culture"
                         % (w_min,))
            lines.append("    N_train = %d windows" % (n_train_windows,))
            if u_available < int(d.min_train_cultures_per_class):
                problems["raise"].append(
                    "only %d training culture(s) in the smallest class, below "
                    "min_train_cultures_per_class = %d"
                    % (u_available, d.min_train_cultures_per_class))
    else:
        lines.append("  split_mode='time_segment': each trace is cut along its "
                     "OWN time axis, so every culture appears in all three "
                     "splits. Cross-culture positives are not available here.")
        n_train_windows = (sum(int(n) for n in d.synthetic_n_per_class)
                           * _windows_per_trace(
                               d.synthetic_duration_s * d.split_fractions[0],
                               d.window_s, d.train_stride_s))
        lines.append("  N_train ~= %d windows (approx: boundary windows may be "
                     "dropped)" % (n_train_windows,))

    # ---- batch geometry ------------------------------------------------------
    lines.append("")
    lines.append("BATCH GEOMETRY")
    if d.positives_mode == "cross_culture":
        if d.split_mode != "trace":
            problems["raise"].append(
                "positives_mode='cross_culture' needs split_mode='trace'; a "
                "culture cannot supply its own cross-culture positives.")
        if int(d.augmentation.n_positives) != 0:
            problems["raise"].append(
                "positives_mode='cross_culture' requires "
                "augmentation.n_positives == 0; got %d"
                % (d.augmentation.n_positives,))
        if (t.mining_strategy in EASY_POSITIVE_STRATEGIES
                and not bool(d.exclude_same_culture_positives)):
            problems["raise"].append(
                "mining_strategy=%r requires exclude_same_culture_positives=True"
                % (t.mining_strategy,))
        q = int(d.windows_per_culture_per_batch)
        n_s = int(d.augmentation.n_negatives)
        if u_available is not None:
            u_eff = min(int(d.cultures_per_class_per_batch), u_available)
            n_g = u_eff * q
            M = C * u_eff * q * (1 + n_s)
            lines.append("  U_c requested = %d, U_available = %d  ->  U_eff = %d%s"
                         % (d.cultures_per_class_per_batch, u_available, u_eff,
                            "  (CLAMPED by Eq. 3)"
                            if u_eff < d.cultures_per_class_per_batch else ""))
            lines.append("  q = %d, N_s = %d" % (q, n_s))
            lines.append("  n_g = U_eff * q = %d   (Eq. 2a)" % (n_g,))
            lines.append("  M   = C * U_eff * q * (1 + N_s) = %d rows  (Eq. 2b)"
                         % (M,))
            lines.append("  cross-culture positives per anchor = %d"
                         % (u_eff * q - 1,))
            if t.mining_strategy in EASY_POSITIVE_STRATEGIES:
                if n_g > int(d.max_group_size):
                    problems["raise"].append(
                        "n_g = %d exceeds max_group_size = %d under %r mining"
                        % (n_g, d.max_group_size, t.mining_strategy))
                else:
                    lines.append("  n_g cap  = %d (easy-positive ceiling) -> OK"
                                 % (d.max_group_size,))
                if w_min is not None:
                    q_cap = max(1, int(math.floor(DEFAULT_Q_CAP_FRACTION
                                                  * w_min)))
                    lines.append("  q cap    = max(1, floor(%.3g * W_min=%d)) "
                                 "= %d" % (DEFAULT_Q_CAP_FRACTION, w_min, q_cap))
                    if q > q_cap:
                        problems["raise"].append(
                            "q = %d exceeds the degeneracy cap %d" % (q, q_cap))
            else:
                lines.append("  n_g / q caps: NOT applied (mining_strategy=%r "
                             "is not an easy-positive strategy)"
                             % (t.mining_strategy,))
            if w_min is not None and q > w_min:
                problems["raise"].append(
                    "q = %d exceeds W_min = %d (windows would be drawn WITH "
                    "replacement)" % (q, w_min))
            cost = M * T
        else:
            lines.append("  (skipped: split arithmetic unavailable)")
            cost = None
    else:
        B_c = int(t.windows_per_condition)
        P = int(d.augmentation.n_positives)
        N = int(d.augmentation.n_negatives)
        rows_per_source = 1 + P + N
        M = C * B_c * rows_per_source
        lines.append("  augmentation mode: B_c = %d, P = %d, N = %d" % (B_c, P, N))
        lines.append("  rows per source window = 1 + P + N = %d" % (rows_per_source,))
        lines.append("  M = C * B_c * (1 + P + N) = %d rows" % (M,))
        cost = M * T

    # ---- cost + epoch length -------------------------------------------------
    lines.append("")
    lines.append("COST")
    if cost is not None:
        lines.append("  forward-pass size proxy = M * T = %s sample-values/batch"
                     % ("{:,}".format(cost),))
    if n_train_windows:
        if int(t.batches_per_epoch) > 0:
            nb = int(t.batches_per_epoch)
            lines.append("  batches_per_epoch = %d (explicit)" % (nb,))
        else:
            nb = int(math.ceil(n_train_windows
                               / float(C * int(t.windows_per_condition))))
            lines.append("  batches_per_epoch = ceil(N_train / (C * B_c)) = %d "
                         "(derived)" % (nb,))
        lines.append("  epochs <= %d, patience = %d, seeds = %d"
                     % (t.max_epochs, t.patience, t.n_seeds))
        total = cfg.search.n_calls_arch + cfg.search.n_calls_train \
            + cfg.regularization.n_calls
        lines.append("  search trials (staged total) = %d, each x %d seed(s)"
                     % (total, t.n_seeds))

    out = "\n".join(lines)
    if verbose:
        print(out)
        print("")
        if problems["raise"]:
            print("WOULD RAISE (%d):" % len(problems["raise"]))
            for p in problems["raise"]:
                print("  ERROR  %s" % p)
        if problems["warn"]:
            print("WARNINGS (%d):" % len(problems["warn"]))
            for p in problems["warn"]:
                print("  WARN   %s" % p)
        if not problems["raise"] and not problems["warn"]:
            print("No gate violations and no warnings.")
    return cfg, problems


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    _cfg, _p = preflight(sys.argv[1])
    sys.exit(1 if _p["raise"] else 0)
