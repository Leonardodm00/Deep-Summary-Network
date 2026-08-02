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

    # ---- objective feasibility -------------------------------------------- #
    # The margin demands that the closest class pair sit at least m_cos apart in
    # cosine distance. The largest achievable MINIMUM pairwise cosine distance
    # over C classes on the unit hypersphere is C/(C-1), so the margin implies
    #
    #     S >= m_cos * (C - 1) / C                                        (7)
    #
    # on the mean silhouette.
    #
    # NO PRACTICAL CEILING IS IMPOSED ON S. An earlier version of this guard
    # refused any margin implying S >= 0.8, on the grounds that the archived
    # cells only ever reached 0.424 to 0.556. That calibration is NOT a property
    # of the geometry: every one of those runs used an ANTI-COLLAPSE mining
    # strategy (easy positives, chosen precisely so that same-class windows are
    # not dragged to a single point), so 0.556 describes what is reachable while
    # actively resisting within-class collapse, not what is reachable at all.
    # Under a collapse-SEEKING configuration -- hard mining, small alpha, the
    # NC2 separation term -- a high S is the OBJECTIVE, not a red flag, and
    # S -> 1 is the intended terminal state (NC1 in Papyan et al.'s decomposition,
    # of which L_sep implements only NC2).
    #
    # What remains is the one bound that is arithmetic rather than judgement:
    # S <= 1 always, so a margin demanding more than that cannot be met by any
    # embedding whatsoever. That is the only case still refused.
    lines.append("")
    lines.append("OBJECTIVE")
    lines.append("  loss_type=%s  mining=%s  swap=%s  strict_semihard=%s"
                 % (t.loss_type, t.mining_strategy, t.swap, t.strict_semihard))
    s_req = float(t.margin) * (C - 1) / float(C)
    lines.append("  margin m_cos = %.4g at C = %d  ->  implies silhouette "
                 ">= %.4g" % (t.margin, C, s_req))
    if s_req > 1.0:
        problems["raise"].append(
            "margin %.3f implies silhouette >= %.3f at C = %d, and the "
            "silhouette is bounded above by 1: no embedding can satisfy this "
            "objective. Lower train.margin below %.3f."
            % (t.margin, s_req, C, float(C) / (C - 1)))
    if t.loss_type == "triplet":
        # the FIXED margin is only the starting point: under "triplet" the
        # margin is SEARCHED, so the binding number is the TOP of margin_range
        m_hi = float(cfg.search.margin_range[1])
        s_hi = m_hi * (C - 1) / float(C)
        lines.append("  margin_range high = %.4g (SEARCHED)  ->  worst-case "
                     "implied silhouette >= %.4g" % (m_hi, s_hi))
        if s_hi > 1.0:
            problems["raise"].append(
                "search.margin_range high %.3f lets phase 2 sample a margin "
                "implying silhouette >= %.3f at C = %d, which exceeds the "
                "silhouette's upper bound of 1. Lower the range high below "
                "%.3f." % (m_hi, s_hi, C, float(C) / (C - 1)))
    if t.loss_type in ("joint", "joint_sep"):
        # the angular hinge is exactly a silhouette floor: S >= 1 - 4 sin^2 alpha
        floor = 1.0 - 4.0 * math.sin(math.radians(float(t.angular_alpha_deg))) ** 2
        lines.append("  angular alpha = %.4g deg  ->  silhouette floor "
                     "S >= %.4g  (tolerated a/b <= %.4g)"
                     % (t.angular_alpha_deg, floor, 1.0 - floor))
        if floor <= 0.0:
            problems["warn"].append(
                "angular_alpha_deg = %.4g gives a silhouette floor of %.3g: the "
                "angular term is VACUOUS at this angle on L2-normalised "
                "embeddings and will contribute no gradient. Lower alpha; "
                "SMALL alpha is the collapse-forcing direction."
                % (t.angular_alpha_deg, floor))
        # alpha is the within-class collapse knob: the floor S >= 1 - 4 sin^2
        # alpha rises monotonically as alpha falls, and alpha -> 0 demands
        # a/b -> 0, i.e. exact within-class collapse (NC1).
        if t.mining_strategy in ("easy_positive", "easy_pos_semihard_neg"):
            problems["warn"].append(
                "mining_strategy=%r is an ANTI-COLLAPSE strategy: easy "
                "positives require only the CLOSEST same-class window to be "
                "near, which is what lets a class spread over a manifold. It "
                "works against the angular floor (S >= %.3g) and against the "
                "NC2 separation term. Use mining_strategy='hard' for a "
                "collapse-seeking run."
                % (t.mining_strategy, floor))
    if t.loss_type == "joint_sep":
        lines.append("  lambda_sep = %.4g  gate_threshold = %s  "
                     "(sep means: %s)"
                     % (t.lambda_sep,
                        "none (always on)" if t.sep_gate_threshold is None
                        else "%.4g" % t.sep_gate_threshold,
                        "raw" if C == 2 else "centred"))
        if C == 2 and t.lambda_sep > 0.05:
            problems["warn"].append(
                "at C = 2 the separation term is computed on RAW class means "
                "and its natural scale is roughly 30x larger than at C >= 3, so "
                "lambda_sep = %.3g is likely far too strong. Search "
                "search.lambda_sep_range rather than reusing a C = 3 value."
                % (t.lambda_sep,))

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
