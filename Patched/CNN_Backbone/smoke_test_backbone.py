"""
smoke_test_backbone.py

Standalone correctness checks for backbone.py. Everything needed to debug and
verify the implementation; no external data required. Runs on CPU.

Run:
    python3 smoke_test_backbone.py            # full sweep
    python3 smoke_test_backbone.py --quick    # one config per check (fast)

Checks:
  1+2. Shape: (M,T) and (M,1,T) -> (M,E) across the grid; every block builds and
       its residual add aligns (a length mismatch raises inside the block).
  3.   Head determinism in eval mode (same input -> same output).
  4.   Unit-norm output when l2_normalize=True.
  5.   GroupNorm batch-independence (a sample alone == the same sample in a batch),
       i.e. train == eval.
  6.   Residual-friendly init (last-GN gamma == 0 => interior block ~ identity).
  7.   Allocation + final-length report across the depth grid vs ~20-block prior.
  8.   Width-snapping invariant (every stage width divisible by its group width)
       and a ResNeXt model actually builds for each group width.
  9.   Head fusion width == n_ops * sum(fused stage widths).
  10.  std-pooling finite forward + backward at init, including a constant input.

Self-contained: imports only backbone, torch, stdlib.
"""

import argparse
import itertools
import sys

import torch

import backbone as bb
from backbone import BackboneConfig, build_backbone, allocate_stages


TOL = 1e-5


def _grid(quick):
    depths = [4] if quick else [3, 4, 5, 6]
    wms = [2.0] if quick else [1.5, 2.0, 2.3, 3.0]
    families = [0] if quick else [0, 1]
    embeds = [16] if quick else [8, 12, 16]
    return depths, wms, families, embeds


def _base_cfg(**kw):
    params = dict(depth_exponent=4, width_multiplier=2.0, block_family=0,
                  embedding_size=16, norm_target_cpg=16)
    params.update(kw)
    return BackboneConfig(**params)


def check_shapes_and_residuals(quick, T=1024, M=6):
    depths, wms, families, embeds = _grid(quick)
    n = 0
    for d, wm, fam, e in itertools.product(depths, wms, families, embeds):
        cfg = _base_cfg(depth_exponent=d, width_multiplier=wm,
                        block_family=fam, embedding_size=e)
        model = build_backbone(cfg).eval()
        with torch.no_grad():
            y2 = model(torch.randn(M, T))        # (M, T) input
            y3 = model(torch.randn(M, 1, T))     # (M, 1, T) input
        assert y2.shape == (M, e), (cfg, y2.shape)
        assert y3.shape == (M, e), (cfg, y3.shape)
        n += 1
    print("  [1+2] shapes + residual length matching OK across %d configs" % n)


def check_determinism(quick, T=1024, M=5):
    cfg = _base_cfg(block_family=1, embedding_size=12)
    model = build_backbone(cfg).eval()
    x = torch.randn(M, T)
    with torch.no_grad():
        a = model(x)
        b = model(x)
    assert torch.equal(a, b), "eval forward is not deterministic"
    print("  [3] head determinism (eval) OK")


def check_unit_norm(quick, T=1024, M=7):
    for fam in ([0] if quick else [0, 1]):
        cfg = _base_cfg(block_family=fam, l2_normalize=True, head_fusion=True,
                        head_pool_ops=("mean", "max", "std"))
        model = build_backbone(cfg).eval()
        with torch.no_grad():
            y = model(torch.randn(M, T))
        norms = y.norm(p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=TOL), norms
    print("  [4] unit-norm output OK")


def check_batch_independence(quick, T=1024):
    cfg = _base_cfg(block_family=0, head_fusion=True,
                    head_pool_ops=("mean", "max", "std"))
    model = build_backbone(cfg).eval()
    batch = torch.randn(8, T)
    single = batch[3:4]
    with torch.no_grad():
        out_in_batch = model(batch)[3]
        out_alone = model(single)[0]
    diff = (out_in_batch - out_alone).abs().max().item()
    assert diff < TOL, "batch dependence detected (max abs diff %.2e)" % diff
    print("  [5] GroupNorm batch-independence OK (train == eval), max diff %.2e" % diff)


def check_residual_init(quick, T=1024, M=4):
    cfg = _base_cfg(depth_exponent=5, block_family=0)
    model = build_backbone(cfg)
    n_zeroed = 0
    for m in model.modules():
        if getattr(m, "last_branch_norm", None) is not None:
            assert torch.count_nonzero(m.last_branch_norm.weight) == 0
            n_zeroed += 1
    # interior block (stride 1, in == out): block = ReLU(F(x) + x) with F(x) == 0
    # at init (last GN gamma == 0), so the block must equal ReLU(x).
    found = False
    for stage in model.stages:
        for blk in stage:
            if not blk.needs_projection:
                C = blk.norm2.num_channels
                x = torch.randn(M, C, 64)
                with torch.no_grad():
                    out = blk.eval()(x)
                err = (out - torch.relu(x)).abs().max().item()
                assert err < TOL, "interior block not identity at init (err %.2e)" % err
                found = True
                break
        if found:
            break
    assert found, "no interior block available to test identity init"
    print("  [6] residual-friendly init OK (%d branch gammas zeroed; block ~ identity)"
          % n_zeroed)


def check_allocation_report(quick, T=1024):
    print("  [7] allocation + final-length report (T=%d, wm=2.0):" % T)
    header = "      %-3s %-7s %-7s %-32s %-26s %-8s %-5s" % (
        "d", "blocks", "stages", "stage_widths", "stage_depths", "L_final", "~20?")
    print(header)
    for d in [3, 4, 5, 6]:
        cfg = _base_cfg(depth_exponent=d)
        sw, sd = allocate_stages(d, cfg.stem_width, cfg.width_multiplier,
                                 cfg.group_width)
        total = sum(sd)
        L = T
        L = -(-L // cfg.stem_stride)
        for _ in sw:
            L = -(-L // cfg.downsampling_rate)
        near = "yes" if 15 <= total <= 25 else ""

        def _fit(s, width):
            return s if len(s) <= width else s[:width - 3] + "..."
        w_str = _fit(str([int(w) for w in sw]), 32)
        d_str = _fit(str([int(x) for x in sd]), 26)
        print("      %-3d %-7d %-7d %-32s %-26s %-8d %-5s"
              % (d, total, len(sw), w_str, d_str, L, near))


def check_width_snapping(quick, T=512, M=3):
    depths = [4] if quick else [3, 4, 5, 6]
    wms = [2.0] if quick else [1.5, 2.0, 2.3, 3.0]
    gws = [16] if quick else [2, 4, 8, 16]
    n = 0
    for d, wm, gw in itertools.product(depths, wms, gws):
        sw, sd = allocate_stages(d, 16, wm, gw)
        for w in sw:
            g = min(gw, w)
            assert w % g == 0, ("divisibility broken", d, wm, gw, w, g)
        n += 1
    # a ResNeXt model must actually build (and run) for each group width
    for gw in gws:
        cfg = _base_cfg(depth_exponent=4, block_family=1, group_width=gw)
        model = build_backbone(cfg).eval()
        with torch.no_grad():
            _ = model(torch.randn(M, T))
    print("  [8] width-snapping invariant OK across %d (d,wm,gw); ResNeXt builds for all gw"
          % n)


def check_fusion_width(quick, T=1024, M=4):
    op_sets = [("mean",)] if quick else [("mean",), ("mean", "max"),
                                         ("mean", "max", "std")]
    n = 0
    for fusion in (False, True):
        for ops in op_sets:
            cfg = _base_cfg(depth_exponent=4, block_family=0, head_fusion=fusion,
                            head_pool_ops=ops, head_prenorm=True)
            model = build_backbone(cfg)
            sw = model.stage_widths
            fused = sw if fusion else [sw[-1]]
            n_ops = len(tuple(o for o in bb._POOL_ORDER if o in ops))
            expected = n_ops * sum(fused)
            assert model.head.in_features == expected, \
                (fusion, ops, model.head.in_features, expected)
            with torch.no_grad():
                y = model(torch.randn(M, T))
            assert y.shape == (M, cfg.embedding_size)
            n += 1
    print("  [9] head fusion input width == n_ops * sum(fused widths) OK (%d cells)" % n)


def check_std_finite_init(quick, T=1024, M=4):
    # full model: finite forward + backward with mean,max,std at init
    cfg = _base_cfg(depth_exponent=4, block_family=1, head_fusion=True,
                    head_pool_ops=("mean", "max", "std"), head_prenorm=True)
    model = build_backbone(cfg).train()
    y = model(torch.randn(M, T))
    assert torch.isfinite(y).all(), "non-finite output at init"
    y.pow(2).sum().backward()
    bad = [name for name, p in model.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, "non-finite gradients at init in: %s" % bad[:5]
    # direct stress of the var -> 0 path: std pool of a constant channel
    h = torch.ones(2, 3, 5, requires_grad=True)
    s = bb._pool(h, "std")
    assert torch.isfinite(s).all(), "std of constant input not finite"
    s.sum().backward()
    assert torch.isfinite(h.grad).all(), "std gradient on constant input not finite"
    print("  [10] std-pooling finite forward+backward at init (incl. constant input) OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="one config per check")
    args = ap.parse_args()
    torch.manual_seed(0)
    print("Running backbone smoke tests (%s)..." % ("quick" if args.quick else "full"))
    check_shapes_and_residuals(args.quick)
    check_determinism(args.quick)
    check_unit_norm(args.quick)
    check_batch_independence(args.quick)
    check_residual_init(args.quick)
    check_allocation_report(args.quick)
    check_width_snapping(args.quick)
    check_fusion_width(args.quick)
    check_std_finite_init(args.quick)
    print("ALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
