"""
dsn_load.py -- loading layer ONLY for the l3c_joint_full 4-lane search handoff.

Separation of concerns: this module reads files and returns tidy pandas objects.
It computes no statistics and draws nothing. All numbers reported downstream must
originate here, from the files, and nowhere else.

Schema reference: HANDOFF_interpreting_joint_search_results.md, section 5.
Everything in this module was written against the ACTUAL records, not the document.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import pandas as pd

LANES = (0, 1, 2, 3)

# Fields promoted to top-level columns from each trial record.
_SCALAR_FIELDS = [
    "trial",
    "objective",
    "mean",
    "std",
    "epsilon",
    "failed",
    "projected",
    "cell",
    "loss_type",
    "mining_strategy",
    "strict_semihard",
    "head_fusion",
    "eff_rank",
    "ari_mean",
    "sil_mean",
    "n_seeds",
    "n_seeds_ok",
    "wall_elapsed_s",
    "selection_primary",
    "schema_version",
]


def _read_jsonl(path: str) -> List[dict]:
    """Return every non-blank line of a JSONL file as a dict, in file order."""
    out = []
    with open(path, "r") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_states(root: str) -> Dict[int, dict]:
    """Return {lane_index: parsed search_state.json}."""
    states = {}
    for j in LANES:
        p = os.path.join(root, f"state_lane{j}.json")
        if os.path.exists(p):
            with open(p, "r") as fh:
                states[j] = json.load(fh)
    return states


def load_config(root: str) -> dict:
    """Return the shared config_input.json (lane 0's copy)."""
    with open(os.path.join(root, "config_input.json"), "r") as fh:
        return json.load(fh)


def axis_bounds(states: Dict[int, dict]) -> Dict[str, dict]:
    """Return {axis_name: {kind, low, high, prior, log}} from the study header.

    Read from search_state.json's recorded `space_signature` rather than
    hard-coded, so the bounds drawn on a figure are always the bounds the
    study was actually run under. Lanes must agree; a disagreement is raised,
    not averaged, because it would mean the lanes are not poolable.

    NOTE on sep_warmup_frac: the runtime clips this axis at
    patience / max_epochs (0.4 for P=40, E_max=100). The configured upper
    bound recorded here is 0.5, so `high` overstates the reachable range for
    that axis alone. Callers that draw bounds should pass the clip in
    explicitly rather than trusting `high`.
    """
    out, source = {}, None
    for j, st in sorted(states.items()):
        sig = (st.get("study") or {}).get("space_signature")
        if not sig:
            continue
        cur = {}
        for e in sig:
            cur[e["name"]] = {
                "kind": e.get("kind"),
                "low": e.get("low"),
                "high": e.get("high"),
                "prior": e.get("prior"),
                "categories": e.get("categories"),
                "log": e.get("prior") == "log-uniform",
            }
        if source is None:
            out, source = cur, j
        elif cur != out:
            raise ValueError(
                "space_signature of lane %d differs from lane %d: the lanes "
                "did not search the same space and must not be pooled." % (j, source))
    return out


def load_results(root: str) -> Dict[int, dict]:
    """Return {lane: results.json}. ABSENCE IS INFORMATIVE: an empty dict means
    no lane completed its final training."""
    out = {}
    for j in LANES:
        p = os.path.join(root, f"results_lane{j}.json")
        if os.path.exists(p):
            with open(p, "r") as fh:
                out[j] = json.load(fh)
    return out


def load_trials(root: str) -> pd.DataFrame:
    """Return one tidy row per trial, pooled over lanes, with a `lane` column.

    Columns:
      - the scalar fields listed in _SCALAR_FIELDS
      - `raw_<axis>` for each of the 18 axes of point_raw (pre-projection)
      - `head_pool_ops_str`, `condition_str`, `raw_condition_str` (list -> str)
      - `sel_epoch` = selected_epochs[0]   (n_seeds == 1 in this study)
      - `active_loss_hps` kept as a Python list (needed for per-axis filtering)
    """
    rows = []
    for j in LANES:
        path = os.path.join(root, f"trials_lane{j}.jsonl")
        if not os.path.exists(path):
            continue
        for rec in _read_jsonl(path):
            row = {"lane": j}
            for f in _SCALAR_FIELDS:
                row[f] = rec.get(f)
            for axis, val in rec.get("point_raw", {}).items():
                row[f"raw_{axis}"] = val
            hpo = rec.get("head_pool_ops")
            row["head_pool_ops_str"] = "+".join(hpo) if isinstance(hpo, list) else str(hpo)
            # NB: the pre-Pi condition is deliberately NOT called raw_condition_str,
            # so that the `raw_<axis>` namespace contains exactly the 18 search axes.
            for key, out_name in (("condition", "condition_str"),
                                  ("raw_condition", "pre_pi_condition_str")):
                v = rec.get(key)
                row[out_name] = "|".join(map(str, v)) if isinstance(v, list) else str(v)
            se = rec.get("selected_epochs") or []
            row["sel_epoch"] = se[0] if se else None
            row["n_sel_epochs"] = len(se)
            row["active_loss_hps"] = rec.get("active_loss_hps") or []
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["lane", "trial"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "."
    df = load_trials(root)
    print(f"loaded {len(df)} trials, {df['lane'].nunique()} lanes, "
          f"{df.shape[1]} columns")
    print(df.groupby("lane").size())
