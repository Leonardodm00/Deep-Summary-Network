#!/usr/bin/env python3
"""
Smoke test for the in_channels backbone change.

Verifies, against the real backbone.py:
  1. multichannel (C=9) forward: (M, 9, T) -> (M, E), finite, unit-norm rows;
  2. backward compatibility (C=1): (M, T) and (M, 1, T) still work;
  3. the input-shape guard rejects a 2D input when C > 1;
  4. STRUCTURAL INVARIANCE: switching C=1 -> C=9 changes ONLY the stem conv's
     input dimension; the stage width schedule and every downstream parameter
     count are identical (the "subsequent enlargement stays as-is" guarantee).

Run:
    python3 smoke_test_in_channels.py
Exit code 0 = all passed.
"""
import sys
import torch
from backbone import BackboneConfig, build_backbone


def _check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print("[%s] %s%s" % (tag, name, ("  (%s)" % detail) if detail else ""))
    return cond


def stem_conv(model):
    return model.stem.conv


def total_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    ok = True
    torch.manual_seed(0)
    M, T, E = 4, 1024, 16
    common = dict(depth_exponent=3, stem_width=16, width_multiplier=2.0,
                  embedding_size=E, l2_normalize=True)

    # ---- 1. multichannel forward ----------------------------------------- #
    C = 9
    m9 = build_backbone(BackboneConfig(in_channels=C, **common))
    m9.eval()
    x9 = torch.randn(M, C, T)
    with torch.no_grad():
        z9 = m9(x9)
    ok &= _check("C=9 forward shape (M, E)", tuple(z9.shape) == (M, E),
                 "got %s" % (tuple(z9.shape),))
    ok &= _check("C=9 output finite", bool(torch.isfinite(z9).all()))
    norms = z9.norm(dim=1)
    ok &= _check("C=9 rows L2-normalized (unit norm)",
                 bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-4)),
                 "norm range [%.4f, %.4f]" % (norms.min(), norms.max()))
    ok &= _check("C=9 stem conv in_channels == 9",
                 stem_conv(m9).weight.shape[1] == 9,
                 "weight shape %s" % (tuple(stem_conv(m9).weight.shape),))

    # ---- 2. backward compatibility (C=1) --------------------------------- #
    m1 = build_backbone(BackboneConfig(in_channels=1, **common))
    m1.eval()
    with torch.no_grad():
        z_2d = m1(torch.randn(M, T))         # (M, T)
        z_3d = m1(torch.randn(M, 1, T))      # (M, 1, T)
    ok &= _check("C=1 accepts (M, T)", tuple(z_2d.shape) == (M, E),
                 "got %s" % (tuple(z_2d.shape),))
    ok &= _check("C=1 accepts (M, 1, T)", tuple(z_3d.shape) == (M, E),
                 "got %s" % (tuple(z_3d.shape),))
    ok &= _check("C=1 stem conv in_channels == 1",
                 stem_conv(m1).weight.shape[1] == 1)

    # ---- 3. shape guard rejects 2D when C > 1 ---------------------------- #
    raised = False
    try:
        m9(torch.randn(M, T))                # ambiguous for C=9
    except ValueError:
        raised = True
    ok &= _check("C=9 rejects 2D (M, T) input", raised)

    # ---- 4. structural invariance ---------------------------------------- #
    ok &= _check("stage widths identical for C=1 and C=9",
                 m1.stage_widths == m9.stage_widths,
                 "%s vs %s" % (m1.stage_widths, m9.stage_widths))
    ok &= _check("stage depths identical for C=1 and C=9",
                 m1.stage_depths == m9.stage_depths)
    # only the stem conv weight differs; delta = stem_width * (9 - 1) * kernel
    k = m1.cfg.stem_kernel
    expected_delta = common["stem_width"] * (9 - 1) * k
    actual_delta = total_params(m9) - total_params(m1)
    ok &= _check("param delta == stem_width*(9-1)*kernel (only stem input grew)",
                 actual_delta == expected_delta,
                 "delta=%d expected=%d" % (actual_delta, expected_delta))

    print("=" * 56)
    print("SMOKE RESULT: %s" % ("ALL PASSED" if ok else "FAILURES ABOVE"))
    print("=" * 56)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
