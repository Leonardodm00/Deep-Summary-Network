"""
smoke_test_checkpoint.py

Standalone correctness checks for checkpoint.py (Stage 4). CPU only, no data.

Run:
    python3 smoke_test_checkpoint.py

Checks:
  [A] rebuild_model_from_checkpoint rebuilds from the EMBEDDED config alone and
      reproduces the original model's forward output BIT-FOR-BIT; the output
      dimension equals the config's embedding_size (zero remembered HPs).
  [B] Resume equivalence: after saving mid-training, a run resumed from the
      checkpoint (weights + optimizer + scheduler + RNG restored) produces
      parameters IDENTICAL to the uninterrupted continuation.
  [C] Atomicity: a save that fails mid-write leaves the previous checkpoint at
      the destination intact and leaves no temp file behind.
  [D] CheckpointManager: last.pt round-trips the epoch counter; periodic saves
      honour the cadence (save on multiples of periodic_every, skip otherwise).
"""

import os
import sys
import glob
import tempfile

import torch

import checkpoint as ckpt_mod
from checkpoint import (
    save_checkpoint, load_checkpoint, rebuild_model_from_checkpoint,
    capture_rng_state, restore_rng_state, CheckpointManager,
)
from config import ExperimentConfig, BackboneConfig
from backbone import build_backbone

_T = 1024          # input length: safe past stem stride 4 + per-stage downsampling
_B = 4             # batch


def _small_cfg(embedding_size=8):
    bb = BackboneConfig(depth_exponent=3, width_multiplier=1.5, stem_width=8,
                        embedding_size=embedding_size)
    return ExperimentConfig(backbone=bb)


def _clone_params(model):
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def _one_step(model, opt, sched):
    """A single deterministic training step whose only randomness is the input
    draw from the torch global RNG (so restoring RNG reproduces it exactly)."""
    x = torch.randn(_B, _T)
    opt.zero_grad()
    loss = model(x).pow(2).mean()
    loss.backward()
    opt.step()
    sched.step()


def check_rebuild_bitforbit():
    cfg = _small_cfg(embedding_size=8)
    torch.manual_seed(0)
    model = build_backbone(cfg.backbone).eval()
    x = torch.randn(_B, _T)
    out1 = model(x)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.pt")
        save_checkpoint(p, cfg, model)
        model2, ck = rebuild_model_from_checkpoint(p)
    model2.eval()
    out2 = model2(x)
    assert torch.equal(out1, out2), "forward not bit-for-bit after rebuild"
    assert out2.shape == (_B, 8), out2.shape        # embedding dim from embedded config
    print("  [A] rebuild reproduces forward bit-for-bit; embedding dim from config OK")


def check_resume_equivalence():
    cfg = _small_cfg()
    torch.manual_seed(123)
    model = build_backbone(cfg.backbone).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.9)

    for _ in range(3):                 # steps 1..3
        _one_step(model, opt, sched)

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "mid.pt")
        rng = capture_rng_state()      # snapshot RNG exactly at this point
        save_checkpoint(p, cfg, model, optimizer=opt, scheduler=sched,
                        epoch=3, rng_state=rng)

        # uninterrupted continuation: steps 4..5 on the live objects
        for _ in range(2):
            _one_step(model, opt, sched)
        params_live = _clone_params(model)

        # resumed continuation: rebuild + restore everything, replay steps 4..5
        model_r, ck = rebuild_model_from_checkpoint(p)
        model_r.train()
        opt_r = torch.optim.AdamW(model_r.parameters(), lr=1e-3)
        sched_r = torch.optim.lr_scheduler.StepLR(opt_r, step_size=1, gamma=0.9)
        opt_r.load_state_dict(ck["optimizer_state"])
        sched_r.load_state_dict(ck["scheduler_state"])
        restore_rng_state(ck["rng_state"])
        for _ in range(2):
            _one_step(model_r, opt_r, sched_r)
        params_res = _clone_params(model_r)

    assert params_live.keys() == params_res.keys()
    for name in params_live:
        assert torch.equal(params_live[name], params_res[name]), \
            "resume mismatch in parameter %s" % name
    print("  [B] resume equivalence: params identical after resumed vs "
          "uninterrupted steps OK")


def check_atomicity():
    cfg = _small_cfg()
    torch.manual_seed(1)
    model = build_backbone(cfg.backbone)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.pt")
        save_checkpoint(p, cfg, model, epoch=1)              # valid v1
        assert load_checkpoint(p)["epoch"] == 1

        # force the next write to fail mid-save
        orig_save = ckpt_mod.torch.save

        def _boom(*a, **k):
            raise RuntimeError("simulated interrupted write")

        ckpt_mod.torch.save = _boom
        raised = False
        try:
            save_checkpoint(p, cfg, model, epoch=999)        # must fail
        except RuntimeError:
            raised = True
        finally:
            ckpt_mod.torch.save = orig_save
        assert raised, "patched save did not raise"

        # destination still holds v1; no temp file left behind
        assert load_checkpoint(p)["epoch"] == 1, "previous checkpoint was clobbered"
        leftovers = glob.glob(os.path.join(d, "*.pt.tmp"))
        assert not leftovers, "temp files left behind: %r" % leftovers
    print("  [C] atomic write: failed save leaves prior checkpoint intact, no temp OK")


def check_manager_cadence():
    cfg = _small_cfg()
    torch.manual_seed(2)
    model = build_backbone(cfg.backbone)
    with tempfile.TemporaryDirectory() as d:
        mgr = CheckpointManager(d, periodic_every=3)
        mgr.save_last(config=cfg, model=model, epoch=5)
        ck = mgr.load_last()
        assert ck is not None and ck["epoch"] == 5, ck

        r3 = mgr.maybe_save_periodic(config=cfg, model=model, epoch=3)   # 3 % 3 == 0
        r4 = mgr.maybe_save_periodic(config=cfg, model=model, epoch=4)   # skip
        assert r3 is not None and r4 is None
        assert os.path.exists(mgr.periodic_path(3))
        assert not os.path.exists(mgr.periodic_path(4))
    print("  [D] CheckpointManager: epoch round-trip + periodic cadence OK")


def main():
    print("Running checkpoint smoke tests...")
    check_rebuild_bitforbit()
    check_resume_equivalence()
    check_atomicity()
    check_manager_cadence()
    print("ALL CHECKPOINT SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
