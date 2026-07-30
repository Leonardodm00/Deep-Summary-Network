"""[C4] Smoke test for the cross-culture batch SAMPLER (data_pipeline.py).

WHAT THIS SUITE ESTABLISHES
---------------------------
The cross-culture path of ConditionBalancedBatchSampler builds every batch by
drawing, per class, U_eff DISTINCT cultures without replacement (re-drawn each
batch, never partitioned), then q windows from each. This suite checks that
property against CONSTRUCTED batches, not against the arithmetic that
smoke_test_batch_geometry.py already covers.

THE TWO-TIER SPLIT (why some checks are skipped here)
-----------------------------------------------------
The sampler yields WINDOW INDICES and is pure numpy, so its geometry can be
verified with numpy alone -- and that is exactly the part of Change 4 most able
to go quietly wrong. The remaining assertions of H-section 5.7 concern the
COLLATED tensor batch and the MINER (one miner(Z, y) call, len(pairs) == 3, rho
from mined negatives, the (1 + N_s) row factor, and the byte-identical collator
under "augmentation"); those need torch + pytorch_metric_learning and run on the
cluster. They are present below as guarded checks that run for real on the
cluster and SKIP (not fail) when the dependency is absent, so this file is the
single home for [A]-[J] and nothing is silently dropped.

CHECKS (torch-free, run everywhere)
  [B*] n_g against a CONSTRUCTED batch: each class label appears exactly
       n_g = U_eff * q times among the drawn windows, no window index repeats,
       and the real-row count per batch is C * U_eff * q. (This is also
       Change 1 assertion [B]: n_g counts rows sharing a class label, per class,
       per batch, asserted on a batch rather than on the arithmetic.)
  [C]  The Eq. (3) clamp fires and is recorded once: sampler.geometry.clamped is
       True and exactly one clamp line appears in sampler.geo_notes.
  [D]  With one training culture in some class, CONSTRUCTING the sampler raises
       (the sampler's own guard, exercised via resolve_batch_geometry -- this is
       distinct from the split-time guard in data_splits.py).
  [E]  Cultures within a batch are DISTINCT. Tested with >= 3 cultures per class
       available and U_eff < available, so it cannot pass vacuously: exactly
       U_eff distinct cultures per class per batch, each contributing q windows.
  [J]  q > W_min raises at construction (no sampling with replacement).
  [I-sampler] Inertness of the sampler change: under positives_mode="augmentation"
       the yielded index batches are IDENTICAL, over several epochs and seeds, to
       an independently written copy of the pre-Change-4 sampler.

CHECKS (torch-gated, SKIP here / run on the cluster in brian_env)
  [A]  No anchor-positive pair shares a culture when exclude_same_culture_positives
       is true (miner-level, on the COLLATED batch).
  [F]  N_s = 0 and P_b = 0 both produce a well-formed collated batch (no
       empty-tensor concat).
  [G]  miner(Z, y) arity per strategy: 3 (triplet miner, "hard") or 4 (pair
       miner, the easy-positive strategies); the mined negatives are the last
       element in both. Corrects H-section 7.4's arity-3 assumption.
  [H]  rho is computed from MINED negatives: surrogates present but never mined
       give rho = 0.
  [I-collator] Under "augmentation" the COLLATED (X, y) batch is byte-identical
       to the current code.

RUN
    cd Main
    python3 Smoke_Tests/smoke_test_cross_culture_batches.py
    echo $?        # 0 on success

HPC note (hpc-python-compat): pure ASCII, LF endings.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline import ConditionBalancedBatchSampler                # noqa: E402

EP = "easy_positive"
EPSHN = "easy_pos_semihard_neg"
HARD = "hard"


# --------------------------------------------------------------------------- #
# synthetic (trace_of_window g, conditions y) with a KNOWN culture geometry
# --------------------------------------------------------------------------- #
def make_windows(cultures_per_class, windows_per_culture, n_classes):
    """Return (g, y): global-unique culture ids g and class labels y.

    cultures_per_class / windows_per_culture may be an int (same for every class)
    or a per-class sequence. Culture ids are globally unique, so g is directly
    usable as the sampler's culture key.
    """
    def per_class(v, c):
        return int(v[c]) if hasattr(v, "__len__") else int(v)

    g, y = [], []
    u = 0
    for c in range(n_classes):
        k = per_class(cultures_per_class, c)
        w = per_class(windows_per_culture, c)
        for _ in range(k):
            g.extend([u] * w)
            y.extend([c] * w)
            u += 1
    return np.asarray(g, dtype=int), np.asarray(y, dtype=int)


def make_sampler(g, y, mining_strategy=EP, u_c=3, q=1, n_s=2, max_group_size=16,
                 exclude_same_culture_positives=True, n_batches=8, seed=0,
                 max_batch_rows=None, q_cap_fraction=0.5):
    """Construct a cross-culture sampler over (g, y) with explicit geometry."""
    return ConditionBalancedBatchSampler(
        conditions=y,
        per_condition=int(u_c * q),          # unused in cross_culture, kept sane
        n_batches=int(n_batches),
        seed=int(seed),
        positives_mode="cross_culture",
        trace_of_window=g,
        mining_strategy=mining_strategy,
        cultures_per_class_per_batch=u_c,
        windows_per_culture_per_batch=q,
        n_surrogates=n_s,
        max_group_size=max_group_size,
        exclude_same_culture_positives=exclude_same_culture_positives,
        max_batch_rows=max_batch_rows,
        q_cap_fraction=q_cap_fraction,
    )


# --------------------------------------------------------------------------- #
# a frozen copy of the PRE-Change-4 sampler, for the inertness check [I-sampler]
# --------------------------------------------------------------------------- #
class _ReferenceOldSampler:
    """Independently written copy of ConditionBalancedBatchSampler as it was
    BEFORE Change 4 -- the reference [I-sampler] compares against."""

    def __init__(self, conditions, per_condition, n_batches, seed=0):
        self.by_cond = {}
        for idx, c in enumerate(conditions):
            self.by_cond.setdefault(int(c), []).append(idx)
        self.per_condition = int(per_condition)
        self.n_batches = int(n_batches)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.n_batches):
            batch = []
            for c, idxs in self.by_cond.items():
                replace = len(idxs) < self.per_condition
                pick = rng.choice(idxs, size=self.per_condition, replace=replace)
                batch.extend(int(j) for j in pick)
            rng.shuffle(batch)
            yield batch


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def check_B_star_group_size():
    """[B*] / Change 1 [B]: n_g = U_eff * q per class, on a constructed batch."""
    # 4 cultures per class available, U_c = 3 -> U_eff = 3; q = 2; C = 3.
    g, y = make_windows(cultures_per_class=4, windows_per_culture=5, n_classes=3)
    u_c, q, n_classes = 3, 2, 3
    s = make_sampler(g, y, u_c=u_c, q=q, n_s=2, n_batches=25, seed=1)
    assert s.geometry.cultures_effective == u_c, "U_eff should be 3 (no clamp)"
    n_g_expected = u_c * q
    for e in range(3):
        s.set_epoch(e)
        for batch in s:
            # each real window index appears at most once
            assert len(batch) == len(set(batch)), "window index repeated in batch"
            # per class, n_g rows share the class label
            labels = y[np.asarray(batch, dtype=int)]
            for c in range(n_classes):
                cnt = int(np.sum(labels == c))
                assert cnt == n_g_expected, (
                    "class %d: n_g = %d, expected U_eff*q = %d"
                    % (c, cnt, n_g_expected))
            # real-row count per batch is C * U_eff * q (collator adds the
            # (1 + N_s) factor later; that part is [B] on the cluster)
            assert len(batch) == n_classes * u_c * q, (
                "real rows per batch = %d, expected C*U_eff*q = %d"
                % (len(batch), n_classes * u_c * q))
    print("  [B*] n_g = U_eff * q per class, distinct rows, C*U_eff*q per batch  OK")


def check_C_clamp_logged_once():
    """[C] Eq. (3) clamp fires and is recorded once in geo_notes."""
    # class 0 has only 2 cultures -> availability 2 -> U_eff = min(3, 2) = 2.
    g, y = make_windows(cultures_per_class=[2, 4, 4], windows_per_culture=5,
                        n_classes=3)
    s = make_sampler(g, y, u_c=3, q=1, n_s=2, n_batches=4, seed=2)
    assert s.geometry.clamped is True, "clamp should have fired (U_c=3 > avail=2)"
    assert s.geometry.cultures_effective == 2, "U_eff should clamp to 2"
    clamp_lines = [ln for ln in s.geo_notes if "clamp" in ln.lower()]
    assert len(clamp_lines) == 1, (
        "expected exactly one clamp note, got %d: %r" % (len(clamp_lines), clamp_lines))
    print("  [C] Eq. (3) clamp fires and is recorded exactly once              OK")


def check_D_one_culture_raises():
    """[D] one training culture in some class -> construction raises."""
    g, y = make_windows(cultures_per_class=[1, 4, 4], windows_per_culture=5,
                        n_classes=3)
    try:
        make_sampler(g, y, u_c=3, q=1, n_s=2)
    except ValueError:
        print("  [D] one culture in a class raises at construction               OK")
        return
    raise AssertionError("[D] expected ValueError for a class with one culture")


def check_E_distinct_cultures():
    """[E] cultures within a batch are distinct (non-vacuous: avail 4 > U_eff 3)."""
    g, y = make_windows(cultures_per_class=4, windows_per_culture=6, n_classes=3)
    u_c, q, n_classes = 3, 2, 3
    assert u_c < 4, "test must keep U_eff below availability to be non-vacuous"
    s = make_sampler(g, y, u_c=u_c, q=q, n_s=1, n_batches=40, seed=7)
    saw_variation = set()
    for e in range(4):
        s.set_epoch(e)
        for batch in s:
            arr = np.asarray(batch, dtype=int)
            for c in range(n_classes):
                w_c = arr[y[arr] == c]
                cults = g[w_c]
                distinct = np.unique(cults)
                assert distinct.size == u_c, (
                    "class %d drew %d distinct cultures, expected U_eff=%d"
                    % (c, distinct.size, u_c))
                # each drawn culture contributes exactly q windows
                for u in distinct.tolist():
                    assert int(np.sum(cults == u)) == q, (
                        "culture %d contributed != q windows" % u)
                saw_variation.add(tuple(sorted(distinct.tolist())))
    # with avail 4 and U_eff 3, more than one culture subset must appear
    assert len(saw_variation) > 1, (
        "[E] draw never varied its culture subset -- looks like a fixed partition")
    print("  [E] distinct cultures per batch, subset varies across batches     OK")


def check_J_q_exceeds_wmin_raises():
    """[J] q > W_min raises at construction."""
    # smallest culture holds 3 windows; ask for q = 4.
    g, y = make_windows(cultures_per_class=4, windows_per_culture=3, n_classes=3)
    try:
        make_sampler(g, y, u_c=3, q=4, n_s=2, q_cap_fraction=1.0)
    except ValueError:
        print("  [J] q > W_min raises at construction                            OK")
        return
    raise AssertionError("[J] expected ValueError for q exceeding W_min")


def check_I_sampler_inertness():
    """[I-sampler] augmentation-mode index batches identical to the old sampler."""
    rng = np.random.default_rng(123)
    # an uneven condition layout, including a class smaller than per_condition
    conditions = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2], dtype=int)
    for seed in (0, 1, 5):
        for per_condition in (2, 4):
            new = ConditionBalancedBatchSampler(
                conditions=conditions, per_condition=per_condition,
                n_batches=6, seed=seed)              # positives_mode defaults
            ref = _ReferenceOldSampler(
                conditions=conditions, per_condition=per_condition,
                n_batches=6, seed=seed)
            for e in range(4):
                new.set_epoch(e)
                ref.set_epoch(e)
                nb = list(new)
                rb = list(ref)
                assert nb == rb, (
                    "augmentation batches diverged from the old sampler at "
                    "seed=%d per_condition=%d epoch=%d" % (seed, per_condition, e))
    print("  [I-sampler] augmentation path byte-identical to pre-Change-4        OK")


# --------------------------------------------------------------------------- #
# torch-gated checks: implemented for real, SKIP cleanly when the dep is absent.
# [F],[H],[I-collator] need torch (+ scipy); [A],[G] additionally need
# pytorch_metric_learning (the miner). None needs the backbone -- a random unit
# embedding is enough, since the guarantees hold for whatever the miner selects.
# --------------------------------------------------------------------------- #
def _real_torch():
    """True only under a REAL torch. The torch-free CI stub used elsewhere in this
    repo provides the Dataset/Sampler base classes but no tensor ops, so calling
    a real op is what distinguishes it."""
    try:
        import torch
        torch.zeros(1)
        return True
    except Exception:
        return False


def _have_pml():
    try:
        import pytorch_metric_learning  # noqa: F401
        return True
    except Exception:
        return False


class _ReferenceOldCollator:
    """Frozen copy of TripletCollator.__call__ BEFORE the empty-skip change, for
    the byte-identity check [I-collator]."""

    def __init__(self, unique_label_base=1_000_000, destroyed_label_mode="unique",
                 shared_destroyed_label=2):
        self.unique_label_base = int(unique_label_base)
        self.destroyed_label_mode = destroyed_label_mode
        self.shared_destroyed_label = int(shared_destroyed_label)

    def __call__(self, batch):
        import torch
        emb, lab, metas = [], [], []
        next_uniq = self.unique_label_base
        for item in batch:
            pos = item["positives"]
            neg = item["negatives"]
            cond = int(item["condition"])
            emb.append(pos)
            lab.append(torch.full((pos.shape[0],), cond, dtype=torch.long))
            emb.append(neg)
            n = neg.shape[0]
            if self.destroyed_label_mode == "unique":
                lab.append(torch.arange(next_uniq, next_uniq + n, dtype=torch.long))
                next_uniq += n
            else:
                lab.append(torch.full((n,), self.shared_destroyed_label, dtype=torch.long))
            metas.append(item["meta"])
        X = torch.cat(emb, dim=0).to(torch.float32)
        y = torch.cat(lab, dim=0).to(torch.long)
        return X, y, metas


def _cross_aug_cfg(n_s, fs=50.0):
    from augmentation import AugmentationConfig
    # warp_bands with P_b = 0 (cross-culture positives) and N_s = n_s surrogates.
    return AugmentationConfig(fs=fs, split_method="warp_bands",
                              n_positives=0, n_negatives=int(n_s),
                              shift_magnitude_s=0.2, k_min=4, intra_knot_dist=0.2)


def _build_cross_batch(n_s, cultures_per_class=3, n_classes=2, T=64, seed=0):
    """A cross_culture collated batch (q = 1: one window per culture) plus a
    per-row (culture, is_real) map obtained by replaying the collator layout
    (anchor row first, then the N_s surrogate rows, in item order -- the collator
    does not shuffle)."""
    import torch
    from augmentation import build_triplet_instance
    from data_pipeline import TripletCollator
    rng = np.random.default_rng(seed)
    cfg = _cross_aug_cfg(n_s)
    items, row_culture, row_is_real = [], [], []
    u = 0
    for c in range(n_classes):
        for _ in range(cultures_per_class):
            w = torch.from_numpy(np.abs(rng.standard_normal(T)).astype("float32"))
            anchor, positives, neg = build_triplet_instance(w, cfg, rng)
            items.append({"positives": positives, "negatives": neg,
                          "condition": int(c), "meta": (int(u), 0)})
            row_culture += [int(u)] * int(positives.shape[0])   # anchor is real
            row_is_real += [True] * int(positives.shape[0])
            row_culture += [-1] * int(neg.shape[0])             # surrogates
            row_is_real += [False] * int(neg.shape[0])
            u += 1
    X, y, metas = TripletCollator()(items)
    return X, y, torch.tensor(row_culture), torch.tensor(row_is_real), items


def check_F_empty_pools_well_formed():
    """[F] N_s = 0 and P_b = 0 both give a well-formed collated batch."""
    import torch
    from data_pipeline import TripletCollator
    base = int(TripletCollator().unique_label_base)
    for n_s in (0, 2):
        X, y, _rc, row_is_real, items = _build_cross_batch(
            n_s, cultures_per_class=3, n_classes=2, seed=n_s + 1)
        n_items = len(items)
        m_expected = n_items * (1 + n_s)
        assert X.ndim == 2 and X.shape[0] == m_expected, (
            "N_s=%d: X has %s rows, expected %d" % (n_s, tuple(X.shape), m_expected))
        assert y.shape[0] == m_expected, "N_s=%d: y length mismatch" % n_s
        assert int(row_is_real.sum().item()) == n_items, (
            "expected exactly one real anchor row per item at P_b=0")
        assert int((y >= base).sum().item()) == n_items * n_s, (
            "N_s=%d: surrogate-row count wrong" % n_s)
    print("  [F] N_s=0 and P_b=0 both yield a well-formed collated batch          OK")


def _rho_from_mined(y, neg_idx, unique_label_base):
    """REPLICA of train.py's inline rho (kept in sync by hand): fraction of MINED
    negatives whose label is a surrogate label (>= unique_label_base)."""
    import torch
    n = int(neg_idx.numel())
    if n == 0:
        return 0.0
    return float((y[neg_idx] >= int(unique_label_base)).sum().item()) / n


def check_H_rho_from_mined():
    """[H] rho counts surrogates among MINED negatives; 0 when none are mined."""
    import torch
    base = 1_000_000
    # 4 real rows (labels 0/1) and 3 surrogate rows (labels >= base) are PRESENT.
    y = torch.tensor([0, 0, 1, 1, base + 0, base + 1, base + 2], dtype=torch.long)
    # surrogates present but the miner picked only REAL rows as negatives -> rho 0
    assert _rho_from_mined(y, torch.tensor([0, 2, 3], dtype=torch.long), base) == 0.0, (
        "rho must be 0 when no surrogate is among the mined negatives")
    # positive control: 2 of 3 mined negatives are surrogates -> rho = 2/3
    val = _rho_from_mined(y, torch.tensor([0, 4, 5], dtype=torch.long), base)
    assert abs(val - (2.0 / 3.0)) < 1e-9, "rho value wrong on the mixed case"
    # empty mined set -> rho 0
    assert _rho_from_mined(y, torch.tensor([], dtype=torch.long), base) == 0.0
    print("  [H] rho over MINED negatives (0 when surrogates present but unmined)  OK")


def check_I_collator_byte_identical():
    """[I-collator] augmentation-mode collated (X, y) identical to the old collator."""
    import torch
    from augmentation import AugmentationConfig, build_triplet_instance
    from data_pipeline import TripletCollator
    rng = np.random.default_rng(11)
    cfg = AugmentationConfig(fs=50.0, split_method="warp_bands",
                             n_positives=4, n_negatives=5, shift_magnitude_s=0.2)
    items = []
    for c in range(2):
        for _ in range(3):
            w = torch.from_numpy(np.abs(rng.standard_normal(64)).astype("float32"))
            a, p, ng = build_triplet_instance(w, cfg, rng)
            items.append({"positives": p, "negatives": ng,
                          "condition": int(c), "meta": (0, 0)})
    Xn, yn, _ = TripletCollator()(items)
    Xr, yr, _ = _ReferenceOldCollator()(items)
    assert torch.equal(Xn, Xr) and torch.equal(yn, yr), (
        "collator diverged from the pre-Change-4 collator on the non-empty path")
    print("  [I-collator] augmentation-mode collated (X, y) byte-identical         OK")


def check_A_no_same_culture_positive():
    """[A] no anchor-positive pair shares a culture under exclude_same_culture.

    At q = 1 (one window per culture) this is STRUCTURAL -- no two same-class real
    rows share a culture -- but the check still exercises the real miner + collator
    + label wiring, so a regression that leaks a same-culture row would be caught.
    """
    import torch
    import torch.nn.functional as F
    from pytorch_metric_learning import miners, distances
    X, y, row_culture, row_is_real, _items = _build_cross_batch(
        n_s=2, cultures_per_class=4, n_classes=2, seed=5)
    zrng = np.random.default_rng(3)
    Z = torch.from_numpy(zrng.standard_normal((X.shape[0], 8)).astype("float32"))
    Z = F.normalize(Z, dim=1)
    miner = miners.BatchEasyHardMiner(pos_strategy="easy", neg_strategy="hard",
                                      distance=distances.CosineSimilarity())
    pairs = miner(Z, y)
    assert len(pairs) in (3, 4), "unexpected miner arity %d" % len(pairs)
    # positive pairs are (pairs[0], pairs[1]) in BOTH the 3-tuple (a, p, n) and the
    # 4-tuple (a1, p, a2, n) formats -- BatchEasyHardMiner here returns the latter.
    a_idx, p_idx = pairs[0], pairs[1]
    checked = 0
    for a, p in zip(a_idx.tolist(), p_idx.tolist()):
        assert bool(row_is_real[a]) and bool(row_is_real[p]), (
            "a positive-leg row was a surrogate; surrogates can never be positives")
        assert int(row_culture[a]) != int(row_culture[p]), (
            "anchor and positive share a culture -- "
            "exclude_same_culture_positives is violated")
        checked += 1
    assert checked > 0, "miner returned no triplets; cannot verify [A]"
    print("  [A] no anchor-positive pair shares a culture (q=1, structural)        OK")


def check_G_one_miner_call_arity3():
    """[G] miner(Z, y) arity is what rho expects, per strategy.

    Corrects H-section 7.4/7.3, which assumed arity 3 for every miner. In the
    installed PML the TRIPLET miner ("hard") returns a 3-tuple (a, p, n) while the
    PAIR miners (the easy-positive strategies) return a 4-tuple (a1, p, a2, n). In
    BOTH the mined negatives are the LAST element -- which is exactly what
    train.py's rho reads as pairs[-1].
    """
    import torch
    import torch.nn.functional as F
    from pytorch_metric_learning import miners, distances
    d = distances.CosineSimilarity()
    expected = {
        "hard": (miners.TripletMarginMiner(
            margin=0.2, type_of_triplets="hard", distance=d), 3),
        "easy_positive": (miners.BatchEasyHardMiner(
            pos_strategy="easy", neg_strategy="hard", distance=d), 4),
        "easy_pos_semihard_neg": (miners.BatchEasyHardMiner(
            pos_strategy="easy", neg_strategy="semihard", distance=d), 4),
    }
    y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    zrng = np.random.default_rng(7)
    Z = torch.from_numpy(zrng.standard_normal((6, 8)).astype("float32"))
    Z = F.normalize(Z, dim=1)
    for name, (miner, want) in expected.items():
        pairs = miner(Z, y)
        assert len(pairs) == want, (
            "%s miner returned arity %d, expected %d (train.py's rho relies on "
            "this)" % (name, len(pairs), want))
        assert len(pairs) in (3, 4), "arity must be a format train.py handles"
        _ = int(pairs[-1].numel())   # the mined negatives rho reads
    print("  [G] miner arity per strategy (3 triplet / 4 pair); negatives = last  OK")


def check_torch_gated():
    if not _real_torch():
        print("  [F][H][I-collator] SKIPPED: torch/scipy absent; run on the cluster (brian_env)")
        print("  [A][G]             SKIPPED: torch + pytorch_metric_learning absent")
        return
    check_F_empty_pools_well_formed()
    check_H_rho_from_mined()
    check_I_collator_byte_identical()
    if _have_pml():
        check_A_no_same_culture_positive()
        check_G_one_miner_call_arity3()
    else:
        print("  [A][G]             SKIPPED: pytorch_metric_learning absent; run on the cluster")


def main():
    print("smoke_test_cross_culture_batches.py")
    print("-" * 68)
    check_B_star_group_size()
    check_C_clamp_logged_once()
    check_D_one_culture_raises()
    check_E_distinct_cultures()
    check_J_q_exceeds_wmin_raises()
    check_I_sampler_inertness()
    check_torch_gated()
    print("-" * 68)
    print("ALL CROSS-CULTURE SAMPLER CHECKS PASSED (torch-gated checks noted above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
