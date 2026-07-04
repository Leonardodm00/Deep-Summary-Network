"""
backbone.py

Pure tensor-to-tensor 1D-CNN summary network (RegNet-inspired) for
cumulative-IFR MEA traces. Produces an embedding on the unit hypersphere for
triplet metric learning with cosine distance.

Public API (build to this exactly):
    OneDCNNBackbone, BackboneConfig, build_backbone

Contract:
    input : (M, T) or (M, 1, T) float32
    output: (M, E) float32, L2-normalized per row when cfg.l2_normalize is True

Design notes (see the project Topic-2 handoff for full rationale):
  - RegNet quantized-linear width allocation, w0 = wa = stem_width simplified
    variant: per-block width w_j = w0 * wm ** round(s_j) with
    s_j = log(j+1) / log(wm). No [:-2] truncation; round (not floor).
  - The number of stages is EMERGENT (number of distinct quantized widths),
    not fixed at 4. It grows with depth_exponent.
  - GroupNorm replaces BatchNorm everywhere, so train == eval (no batch stats).
  - Robust padding: stride-1 "same" via static conv padding; strided convs use
    explicit asymmetric padding computed at forward time, so main and shortcut
    paths always align. Lengths are never mutated in place.
  - ResNet and ResNeXt blocks both available. ResNeXt is a b=1 grouped block
    (NO bottleneck added), parameterized by group_width (channels per group).
  - Residual-friendly init: Kaiming plus zero-init of the last GroupNorm gamma in
    each residual branch, so every block starts near identity.
  - Multi-scale head (optional): per-stage length pooling (mean/max/std) ->
    optional per-stage LayerNorm -> channel concat -> bare Linear -> L2-norm.
  - Device-agnostic: no .to(device), no State, no augmentation inside forward.
    DDP / torch.compile / AMP ready. Device placement is the caller's job.

This file is intentionally pure ASCII for HPC (davinci-1) transfer safety.
"""

import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


# Canonical order for pooling statistics, so concatenated columns are
# deterministic regardless of the order given in the config.
_POOL_ORDER = ("mean", "max", "std")
_STD_EPS = 1e-5   # inside sqrt(var + eps): keeps step-0 gradients finite


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BackboneConfig:
    # --- body: RegNet allocation ---
    depth_exponent: int = 4          # d in {3,4,5,6}; total blocks = 2 ** d
    width_multiplier: float = 2.0    # wm in [1.5, 3.0], continuous
    stem_width: int = 16             # w0 = wa = stem_width (simplified variant)

    # --- block family ---
    block_family: int = 0            # 0 = ResNet, 1 = ResNeXt
    group_width: int = 16            # ResNeXt channels per group (was fixed groups=16)

    # --- head / embedding ---
    embedding_size: int = 16         # E
    l2_normalize: bool = True
    head_fusion: bool = False        # False: last stage only; True: fuse all stages
    head_pool_ops: Tuple[str, ...] = ("mean",)
    head_prenorm: bool = True        # per-stage LayerNorm before concat (fusion only)

    # --- normalization (GroupNorm) ---
    norm_target_cpg: int = 16        # target channels per group; in {4,8,16,24,32}
    norm_g_max: int = 32             # cap on number of groups

    # --- kernels / strides (fixed today; code handles arbitrary values) ---
    stem_kernel: int = 5
    stem_stride: int = 4
    stage_kernel: int = 3
    downsampling_rate: int = 2

    # --- regularization ---
    dropout: float = 0.0

    def __post_init__(self):
        if self.depth_exponent < 1:
            raise ValueError("depth_exponent must be >= 1")
        if self.width_multiplier <= 1.0:
            raise ValueError("width_multiplier must be > 1.0")
        if self.stem_width < 1:
            raise ValueError("stem_width must be >= 1")
        if self.block_family not in (0, 1):
            raise ValueError("block_family must be 0 (ResNet) or 1 (ResNeXt)")
        if self.group_width < 1:
            raise ValueError("group_width must be >= 1")
        if self.embedding_size < 1:
            raise ValueError("embedding_size must be >= 1")
        if self.norm_target_cpg < 1:
            raise ValueError("norm_target_cpg must be >= 1")
        if self.norm_g_max < 1:
            raise ValueError("norm_g_max must be >= 1")
        if len(self.head_pool_ops) == 0:
            raise ValueError("head_pool_ops must contain at least one op")
        bad = [op for op in self.head_pool_ops if op not in _POOL_ORDER]
        if bad:
            raise ValueError("unknown head_pool_ops %r; allowed %r" % (bad, _POOL_ORDER))
        for k, s in ((self.stem_kernel, self.stem_stride),
                     (self.stage_kernel, self.downsampling_rate)):
            if k < 1 or s < 1:
                raise ValueError("kernels and strides must be >= 1")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")


# ----------------------------------------------------------------------
# Pure helpers (no torch state)
# ----------------------------------------------------------------------
def conv_out_length(l_in, kernel, stride, padding, dilation=1):
    """Output length of a 1D convolution (floor formula)."""
    return (l_in + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def adjust_width_to_group(width, group_width):
    """Snap 'width' to the nearest multiple of g = min(group_width, width), so a
    grouped conv with channels-per-group g always divides evenly. This is the 1D
    analog of the pycls width/group compatibility adjustment; it keeps every
    config buildable instead of asserting and crashing mid-search."""
    g = min(group_width, width)
    if g <= 0:
        return width
    adjusted = int(round(width / g)) * g
    return max(g, adjusted)


def pick_G(num_channels, target_cpg, g_max):
    """Largest number of groups G <= g_max that divides num_channels, aiming for
    ~target_cpg channels per group. Falls back to G = 1 (LayerNorm-for-conv) when
    nothing in range divides num_channels."""
    g_target = max(1, num_channels // target_cpg)
    upper = min(g_target, g_max)
    for G in range(upper, 0, -1):
        if num_channels % G == 0:
            return G
    return 1


def compute_block_widths(depth_exponent, stem_width, width_multiplier):
    """RegNet per-block widths under the w0 = wa = stem_width simplification.
    u_j = w0 * (j + 1); s_j = log(j+1) / log(wm); w_j = w0 * wm ** round(s_j).
    Returns the list of floating per-block widths (length 2 ** depth_exponent)."""
    total_blocks = 2 ** depth_exponent
    w0 = float(stem_width)
    wm = float(width_multiplier)
    log_wm = math.log(wm)
    widths = []
    for j in range(total_blocks):
        s_j = math.log(j + 1) / log_wm
        k = int(round(s_j))
        widths.append(w0 * (wm ** k))
    return widths


def allocate_stages(depth_exponent, stem_width, width_multiplier, group_width):
    """Quantize per-block widths to integers, snap to group compatibility, then
    group consecutive equal widths into stages. The snapped per-block widths are
    non-decreasing, so consecutive-equal grouping yields clean stages.
    Returns (stage_widths, stage_depths), both lists of ints."""
    per_block = compute_block_widths(depth_exponent, stem_width, width_multiplier)
    snapped = [adjust_width_to_group(int(round(w)), group_width) for w in per_block]
    stage_widths = []
    stage_depths = []
    for w in snapped:
        if stage_widths and w == stage_widths[-1]:
            stage_depths[-1] += 1
        else:
            stage_widths.append(w)
            stage_depths.append(1)
    return stage_widths, stage_depths


def _pad_same_1d(x, kernel, stride, dilation=1):
    """Asymmetric zero-padding so a strided conv lands on ceil(L_in / stride).
    For stride == 1 returns x unchanged (handled by static conv padding)."""
    if stride == 1:
        return x
    l_in = x.shape[-1]
    l_out = -(-l_in // stride)  # ceil division
    pad_total = max(0, (l_out - 1) * stride + dilation * (kernel - 1) + 1 - l_in)
    if pad_total == 0:
        return x
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return F.pad(x, (pad_left, pad_right))


def make_norm(num_channels, cfg):
    """GroupNorm feature normalization with group count chosen by pick_G."""
    G = pick_G(num_channels, cfg.norm_target_cpg, cfg.norm_g_max)
    return nn.GroupNorm(num_groups=G, num_channels=num_channels)


# ----------------------------------------------------------------------
# Stem
# ----------------------------------------------------------------------
class Stem(nn.Module):
    """1 -> stem_width strided conv + GroupNorm + ReLU. Downsamples length.
    The single input channel encodes the univariate cumulative-IFR assumption."""

    def __init__(self, cfg):
        super().__init__()
        self.kernel = cfg.stem_kernel
        self.stride = cfg.stem_stride
        self.conv = nn.Conv1d(1, cfg.stem_width, self.kernel,
                              stride=self.stride, padding=0, bias=False)
        self.norm = make_norm(cfg.stem_width, cfg)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = _pad_same_1d(x, self.kernel, self.stride)
        return self.act(self.norm(self.conv(x)))


# ----------------------------------------------------------------------
# Residual blocks
# ----------------------------------------------------------------------
class ResNetBlock(nn.Module):
    """D2L-style ResNet block: two (Conv -> GN -> [ReLU]) layers, residual add.
    Order: ... -> add -> ReLU -> dropout. Projection shortcut (1x1 strided conv
    + GN) is present iff stride != 1 or in_ch != out_ch (standard option B)."""

    def __init__(self, in_ch, out_ch, kernel, stride, cfg):
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        pad1 = 0 if stride > 1 else ((kernel - 1) // 2)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                               padding=pad1, bias=False)
        self.norm1 = make_norm(out_ch, cfg)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, stride=1,
                               padding=(kernel - 1) // 2, bias=False)
        self.norm2 = make_norm(out_ch, cfg)
        self.act = nn.ReLU(inplace=True)

        self.needs_projection = (stride != 1) or (in_ch != out_ch)
        if self.needs_projection:
            self.proj = nn.Conv1d(in_ch, out_ch, 1, stride=stride,
                                  padding=0, bias=False)
            self.proj_norm = make_norm(out_ch, cfg)
        else:
            self.proj = None
            self.proj_norm = None

        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()
        self.last_branch_norm = self.norm2  # gamma zero-init target

    def forward(self, x):
        identity = x
        out = _pad_same_1d(x, self.kernel, self.stride)
        out = self.act(self.norm1(self.conv1(out)))
        out = self.norm2(self.conv2(out))
        if self.needs_projection:
            shortcut = self.proj_norm(self.proj(identity))
        else:
            shortcut = identity
        if out.shape[-1] != shortcut.shape[-1]:
            raise RuntimeError("ResNet residual length mismatch: %s vs %s"
                               % (tuple(out.shape), tuple(shortcut.shape)))
        out = self.act(out + shortcut)
        return self.drop(out)


class ResNeXtBlock(nn.Module):
    """b = 1 grouped residual block (NO bottleneck added):
    1x1 -> grouped(kxk) -> 1x1, residual add, optional 1x1 + GN shortcut.
    Parameterized by group_width (channels per group). out_ch is snapped upstream
    to be divisible by g = min(group_width, out_ch), so the grouped conv builds."""

    def __init__(self, in_ch, out_ch, kernel, stride, cfg):
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        g_eff = min(cfg.group_width, out_ch)
        if out_ch % g_eff != 0:
            raise RuntimeError("ResNeXt out_ch %d not divisible by group width %d"
                               % (out_ch, g_eff))
        num_groups = out_ch // g_eff

        self.c1 = nn.Conv1d(in_ch, out_ch, 1, stride=1, padding=0, bias=False)
        self.n1 = make_norm(out_ch, cfg)
        pad2 = 0 if stride > 1 else ((kernel - 1) // 2)
        self.c2 = nn.Conv1d(out_ch, out_ch, kernel, stride=stride,
                            padding=pad2, groups=num_groups, bias=False)
        self.n2 = make_norm(out_ch, cfg)
        self.c3 = nn.Conv1d(out_ch, out_ch, 1, stride=1, padding=0, bias=False)
        self.n3 = make_norm(out_ch, cfg)
        self.act = nn.ReLU(inplace=True)

        self.needs_projection = (stride != 1) or (in_ch != out_ch)
        if self.needs_projection:
            self.proj = nn.Conv1d(in_ch, out_ch, 1, stride=stride,
                                  padding=0, bias=False)
            self.proj_norm = make_norm(out_ch, cfg)
        else:
            self.proj = None
            self.proj_norm = None

        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()
        self.last_branch_norm = self.n3  # gamma zero-init target

    def forward(self, x):
        identity = x
        out = self.act(self.n1(self.c1(x)))
        out = _pad_same_1d(out, self.kernel, self.stride)
        out = self.act(self.n2(self.c2(out)))
        out = self.n3(self.c3(out))
        if self.needs_projection:
            shortcut = self.proj_norm(self.proj(identity))
        else:
            shortcut = identity
        if out.shape[-1] != shortcut.shape[-1]:
            raise RuntimeError("ResNeXt residual length mismatch: %s vs %s"
                               % (tuple(out.shape), tuple(shortcut.shape)))
        out = self.act(out + shortcut)
        return self.drop(out)


# ----------------------------------------------------------------------
# Head: multi-scale pooling -> optional per-stage LayerNorm -> Linear
# ----------------------------------------------------------------------
def _pool(h, op):
    """Pool a (M, C, L) feature map along length to (M, C)."""
    if op == "mean":
        return h.mean(dim=-1)
    if op == "max":
        return h.amax(dim=-1)
    if op == "std":
        mu = h.mean(dim=-1, keepdim=True)
        var = (h - mu).pow(2).mean(dim=-1)
        return torch.sqrt(var + _STD_EPS)
    raise ValueError("unknown pool op: %r" % (op,))


class MultiScaleHead(nn.Module):
    """Per-stage length pooling -> optional per-stage LayerNorm -> concat -> bare
    Linear -> optional L2-normalize.

    head_fusion=False uses only the last stage; True fuses all stage outputs.
    head_pool_ops selects which statistics per stage (ordered canonically).
    Per-stage LayerNorm (head_prenorm, fusion only) balances the scales so the
    fine early-stage features are not under-exposed relative to the wide late
    stages at initialization."""

    def __init__(self, stage_widths, cfg):
        super().__init__()
        self.fusion = bool(cfg.head_fusion)
        self.ops = tuple(op for op in _POOL_ORDER if op in cfg.head_pool_ops)
        self.n_ops = len(self.ops)
        self.l2 = bool(cfg.l2_normalize)

        fused_widths = list(stage_widths) if self.fusion else [stage_widths[-1]]
        self.prenorm = bool(cfg.head_prenorm and self.fusion)
        if self.prenorm:
            self.norms = nn.ModuleList(
                [nn.LayerNorm(self.n_ops * w) for w in fused_widths])
        else:
            self.norms = None

        self.in_features = self.n_ops * sum(fused_widths)
        self.proj = nn.Linear(self.in_features, cfg.embedding_size)

    def forward(self, stage_outputs):
        selected = stage_outputs if self.fusion else [stage_outputs[-1]]
        parts = []
        for idx, h in enumerate(selected):
            stats = [_pool(h, op) for op in self.ops]
            p = torch.cat(stats, dim=1) if len(stats) > 1 else stats[0]
            if self.prenorm:
                p = self.norms[idx](p)
            parts.append(p)
        z = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        z = self.proj(z)
        if self.l2:
            z = F.normalize(z, p=2, dim=1)
        return z


# ----------------------------------------------------------------------
# Assembled backbone
# ----------------------------------------------------------------------
class OneDCNNBackbone(nn.Module):
    """Pure tensor-to-tensor RegNet-style 1D-CNN summary network.
    forward: (M, T) or (M, 1, T) -> (M, E)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        stage_widths, stage_depths = allocate_stages(
            cfg.depth_exponent, cfg.stem_width, cfg.width_multiplier, cfg.group_width)
        self.stage_widths = list(stage_widths)
        self.stage_depths = list(stage_depths)

        self.stem = Stem(cfg)
        block_cls = ResNetBlock if cfg.block_family == 0 else ResNeXtBlock

        stages = []
        in_ch = cfg.stem_width
        for width, depth in zip(stage_widths, stage_depths):
            blocks = []
            for b in range(depth):
                stride = cfg.downsampling_rate if b == 0 else 1
                blocks.append(block_cls(in_ch, width, cfg.stage_kernel, stride, cfg))
                in_ch = width
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)

        self.head = MultiScaleHead(stage_widths, cfg)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # residual-friendly: zero the last GroupNorm gamma in every residual branch
        for m in self.modules():
            if getattr(m, "last_branch_norm", None) is not None:
                nn.init.zeros_(m.last_branch_norm.weight)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3 or x.shape[1] != 1:
            raise ValueError("expected (M, T) or (M, 1, T); got shape %s"
                             % (tuple(x.shape),))
        x = self.stem(x)
        stage_outputs = []
        for stage in self.stages:
            x = stage(x)
            stage_outputs.append(x)
        return self.head(stage_outputs)


def build_backbone(cfg):
    """Factory: validate config and return an initialized OneDCNNBackbone."""
    if not isinstance(cfg, BackboneConfig):
        raise TypeError("cfg must be a BackboneConfig")
    return OneDCNNBackbone(cfg)
