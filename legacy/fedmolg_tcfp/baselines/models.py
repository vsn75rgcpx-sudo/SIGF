import torch
import torch.nn as nn
import torch.nn.functional as F

from model_swin3d.layers import WindowAttention3D, compute_attn_mask, window_partition, window_reverse
from model_swin3d.swin3d_block import DropPath, LayerNormChannel


class ConvNormAct3D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct3D(in_ch, out_ch),
            ConvNormAct3D(out_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableConv3D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False),
            nn.InstanceNorm3d(in_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(in_ch, out_ch, 1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = ConvNormAct3D(in_ch, out_ch, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
        )
        self.act = nn.LeakyReLU(0.01, inplace=True)
        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return self.act(out + self.skip(x))


class ResidualDepthwiseBlock3D(nn.Module):
    """Parameter-efficient residual block for compact FedNAS-UNet variants."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        ) if in_ch != out_ch else nn.Identity()
        self.conv = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
        )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        x = self.proj(x)
        return self.act(x + self.conv(x))


class SkipProject3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        if in_ch == out_ch:
            self.block = nn.Identity()
        else:
            self.block = ConvNormAct3D(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.block(x)


class FedNASUNetBlock3D(nn.Module):
    """
    Searchable U-Net block with a small, U-Net-friendly operation set.

    Use nas_op="mixed" for a differentiable soft mixture, or choose one fixed
    op after search/ablation: conv3, conv5, depthwise3, residual_depthwise,
    residual, skip.
    """

    OPS = ("conv3", "conv5", "depthwise3", "residual_depthwise", "residual", "skip")

    def __init__(self, in_ch, out_ch, nas_op="mixed"):
        super().__init__()
        self.nas_op = nas_op.lower()
        ops = {
            "conv3": DoubleConv3D(in_ch, out_ch),
            "conv5": nn.Sequential(
                ConvNormAct3D(in_ch, out_ch, kernel_size=5),
                ConvNormAct3D(out_ch, out_ch, kernel_size=5),
            ),
            "depthwise3": nn.Sequential(
                DepthwiseSeparableConv3D(in_ch, out_ch, kernel_size=3),
                ConvNormAct3D(out_ch, out_ch, kernel_size=3),
            ),
            "residual_depthwise": ResidualDepthwiseBlock3D(in_ch, out_ch),
            "residual": ResidualBlock3D(in_ch, out_ch),
            "skip": SkipProject3D(in_ch, out_ch),
        }
        if self.nas_op == "mixed":
            self.op_names = list(self.OPS)
            self.ops = nn.ModuleDict({name: ops[name] for name in self.op_names})
            self.op_logits = nn.Parameter(torch.zeros(len(self.op_names)))
        else:
            if self.nas_op not in ops:
                raise ValueError(f"Unknown nas_op={nas_op}. Available: mixed, {', '.join(self.OPS)}")
            self.op_names = [self.nas_op]
            self.ops = nn.ModuleDict({self.nas_op: ops[self.nas_op]})
            self.register_parameter("op_logits", None)

    def forward(self, x):
        if self.op_logits is None:
            return self.ops[self.nas_op](x)

        weights = torch.softmax(self.op_logits, dim=0)
        out = None
        for weight, name in zip(weights, self.op_names):
            candidate = self.ops[name](x) * weight
            out = candidate if out is None else out + candidate
        return out


class UNet3D(nn.Module):
    """Strong plain 3D U-Net baseline. Outputs WT/TC/ET logits."""

    def __init__(self, in_channels=4, out_channels=3, base_channels=32):
        super().__init__()
        c = base_channels
        self.enc0 = DoubleConv3D(in_channels, c)
        self.enc1 = DoubleConv3D(c, c * 2)
        self.enc2 = DoubleConv3D(c * 2, c * 4)
        self.enc3 = DoubleConv3D(c * 4, c * 8)
        self.bottleneck = DoubleConv3D(c * 8, c * 16)

        self.pool = nn.MaxPool3d(2)
        self.up3 = nn.ConvTranspose3d(c * 16, c * 8, 2, stride=2)
        self.dec3 = DoubleConv3D(c * 16, c * 8)
        self.up2 = nn.ConvTranspose3d(c * 8, c * 4, 2, stride=2)
        self.dec2 = DoubleConv3D(c * 8, c * 4)
        self.up1 = nn.ConvTranspose3d(c * 4, c * 2, 2, stride=2)
        self.dec1 = DoubleConv3D(c * 4, c * 2)
        self.up0 = nn.ConvTranspose3d(c * 2, c, 2, stride=2)
        self.dec0 = DoubleConv3D(c * 2, c)
        self.head = nn.Conv3d(c, out_channels, 1)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        x = self.up3(b)
        x = self.dec3(torch.cat([_match_size(x, e3), e3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([_match_size(x, e2), e2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([_match_size(x, e1), e1], dim=1))
        x = self.up0(x)
        x = self.dec0(torch.cat([_match_size(x, e0), e0], dim=1))
        return self.head(x)


class FedNASUNet3D(nn.Module):
    """U-Net backbone with searchable/fixed local convolution blocks."""

    def __init__(self, in_channels=4, out_channels=3, base_channels=32, nas_op="mixed"):
        super().__init__()
        c = base_channels
        block = lambda in_ch, out_ch: FedNASUNetBlock3D(in_ch, out_ch, nas_op=nas_op)
        self.enc0 = block(in_channels, c)
        self.enc1 = block(c, c * 2)
        self.enc2 = block(c * 2, c * 4)
        self.enc3 = block(c * 4, c * 8)
        self.bottleneck = block(c * 8, c * 16)

        self.pool = nn.MaxPool3d(2)
        self.up3 = nn.ConvTranspose3d(c * 16, c * 8, 2, stride=2)
        self.dec3 = block(c * 16, c * 8)
        self.up2 = nn.ConvTranspose3d(c * 8, c * 4, 2, stride=2)
        self.dec2 = block(c * 8, c * 4)
        self.up1 = nn.ConvTranspose3d(c * 4, c * 2, 2, stride=2)
        self.dec1 = block(c * 4, c * 2)
        self.up0 = nn.ConvTranspose3d(c * 2, c, 2, stride=2)
        self.dec0 = block(c * 2, c)
        self.head = nn.Conv3d(c, out_channels, 1)

    def arch_parameters(self):
        return [m.op_logits for m in self.modules()
                if isinstance(m, FedNASUNetBlock3D) and m.op_logits is not None]

    def export_architecture(self):
        arch = {}
        for name, module in self.named_modules():
            if isinstance(module, FedNASUNetBlock3D):
                if module.op_logits is None:
                    arch[name] = module.nas_op
                else:
                    idx = int(torch.argmax(module.op_logits.detach()).item())
                    arch[name] = module.op_names[idx]
        return arch

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        x = self.up3(b)
        x = self.dec3(torch.cat([_match_size(x, e3), e3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([_match_size(x, e2), e2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([_match_size(x, e1), e1], dim=1))
        x = self.up0(x)
        x = self.dec0(torch.cat([_match_size(x, e0), e0], dim=1))
        return self.head(x)


class ModalityStemFusion3D(nn.Module):
    """Four shallow modality-specific stems followed by lightweight fusion."""

    def __init__(self, in_channels=4, out_channels=32, stem_channels=None, use_se=True):
        super().__init__()
        stem_channels = stem_channels or max(4, out_channels // 4)
        self.in_channels = in_channels
        self.stems = nn.ModuleList([
            nn.Sequential(
                ConvNormAct3D(1, stem_channels, kernel_size=3),
                ConvNormAct3D(stem_channels, stem_channels, kernel_size=3),
            )
            for _ in range(in_channels)
        ])
        fused_channels = in_channels * stem_channels
        self.use_se = use_se
        if use_se:
            hidden = max(4, fused_channels // 4)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(fused_channels, hidden, 1),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv3d(hidden, fused_channels, 1),
                nn.Sigmoid(),
            )
        else:
            self.se = None
        self.reduce = ConvNormAct3D(fused_channels, out_channels, kernel_size=1)

    def forward(self, x):
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} modalities, got {x.shape[1]}")
        feats = [stem(x[:, idx:idx + 1]) for idx, stem in enumerate(self.stems)]
        fused = torch.cat(feats, dim=1)
        if self.se is not None:
            fused = fused * self.se(fused)
        return self.reduce(fused)


class ModalityAttentionFusion3D(nn.Module):
    """Voxel-wise modality attention over shallow modality-specific CNN stems."""

    def __init__(self, in_channels=4, out_channels=32, stem_channels=None):
        super().__init__()
        stem_channels = stem_channels or max(8, out_channels // 2)
        self.in_channels = in_channels
        self.stems = nn.ModuleList([
            nn.Sequential(
                ConvNormAct3D(1, stem_channels, kernel_size=3),
                ConvNormAct3D(stem_channels, stem_channels, kernel_size=3),
            )
            for _ in range(in_channels)
        ])
        fused_channels = in_channels * stem_channels
        hidden = max(8, fused_channels // 4)
        self.attn = nn.Sequential(
            nn.Conv3d(fused_channels, hidden, 1, bias=False),
            nn.InstanceNorm3d(hidden, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(hidden, in_channels, 1),
        )
        self.reduce = ConvNormAct3D(stem_channels, out_channels, kernel_size=1)

    def forward(self, x):
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} modalities, got {x.shape[1]}")
        feats = [stem(x[:, idx:idx + 1]) for idx, stem in enumerate(self.stems)]
        stacked = torch.stack(feats, dim=1)  # B, M, C, D, H, W
        logits = self.attn(torch.cat(feats, dim=1)).unsqueeze(2)
        weights = torch.softmax(logits, dim=1)
        fused = (stacked * weights).sum(dim=1)
        return self.reduce(fused)


class DynamicSwinBottleneckBlock3D(nn.Module):
    """Swin-style shifted-window attention block that tolerates dynamic patch sizes."""

    def __init__(
            self,
            dim,
            num_heads=8,
            window_size=(3, 3, 3),
            shift_size=(0, 0, 0),
            mlp_ratio=2.0,
            drop_path=0.0,
    ):
        super().__init__()
        self.window_size = tuple(window_size)
        self.shift_size = tuple(shift_size)
        self.norm1 = LayerNormChannel(dim)
        self.attn = WindowAttention3D(dim, window_size=self.window_size, num_heads=num_heads)
        self.drop_path = DropPath(drop_path)
        self.norm2 = LayerNormChannel(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Conv3d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv3d(hidden, dim, 1),
        )

    def _pad_to_window(self, x):
        _, _, d, h, w = x.shape
        wd, wh, ww = self.window_size
        pad_d = (wd - d % wd) % wd
        pad_h = (wh - h % wh) % wh
        pad_w = (ww - w % ww) % ww
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        return x, (d, h, w)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x, original_size = self._pad_to_window(x)
        b, c, d, h, w = x.shape

        wd, wh, ww = self.window_size
        sd = min(self.shift_size[0], wd - 1)
        sh = min(self.shift_size[1], wh - 1)
        sw = min(self.shift_size[2], ww - 1)
        shift = (sd, sh, sw)
        if any(shift):
            shifted = torch.roll(x, shifts=(-sd, -sh, -sw), dims=(2, 3, 4))
            mask = compute_attn_mask(d, h, w, self.window_size, shift, x.device)
        else:
            shifted = x
            mask = None

        windows = window_partition(shifted, self.window_size)
        windows = windows.view(-1, wd * wh * ww, c)
        attn_windows = self.attn(windows, mask=mask)
        attn_windows = attn_windows.view(-1, wd, wh, ww, c)
        shifted = window_reverse(attn_windows, self.window_size, b, d, h, w)

        if any(shift):
            x = torch.roll(shifted, shifts=shift, dims=(2, 3, 4))
        else:
            x = shifted

        od, oh, ow = original_size
        x = x[:, :, :od, :oh, :ow]
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.ffn(self.norm2(x)))


class SwinBottleneck3D(nn.Module):
    def __init__(self, channels, depth=2, num_heads=8, window_size=(3, 3, 3), mlp_ratio=2.0):
        super().__init__()
        blocks = []
        for idx in range(depth):
            shift = (0, 0, 0) if idx % 2 == 0 else tuple(v // 2 for v in window_size)
            blocks.append(
                DynamicSwinBottleneckBlock3D(
                    channels,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift,
                    mlp_ratio=mlp_ratio,
                    drop_path=0.05 * idx / max(1, depth - 1),
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class ModalitySwinUNet3D(nn.Module):
    """
    Modality-specific shallow CNN stems + shared U-Net encoder/decoder +
    lightweight Swin bottleneck global module.
    """

    def __init__(
            self,
            in_channels=4,
            out_channels=3,
            base_channels=24,
            stem_channels=None,
            swin_depth=2,
            swin_heads=8,
            swin_window_size=(3, 3, 3),
    ):
        super().__init__()
        c = base_channels
        self.modal_stem = ModalityStemFusion3D(
            in_channels=in_channels,
            out_channels=c,
            stem_channels=stem_channels,
            use_se=True,
        )
        self.enc0 = DoubleConv3D(c, c)
        self.enc1 = DoubleConv3D(c, c * 2)
        self.enc2 = DoubleConv3D(c * 2, c * 4)
        self.enc3 = DoubleConv3D(c * 4, c * 8)
        self.bottleneck = DoubleConv3D(c * 8, c * 16)
        self.global_bottleneck = SwinBottleneck3D(
            c * 16,
            depth=swin_depth,
            num_heads=swin_heads,
            window_size=swin_window_size,
            mlp_ratio=2.0,
        )

        self.pool = nn.MaxPool3d(2)
        self.up3 = nn.ConvTranspose3d(c * 16, c * 8, 2, stride=2)
        self.dec3 = DoubleConv3D(c * 16, c * 8)
        self.up2 = nn.ConvTranspose3d(c * 8, c * 4, 2, stride=2)
        self.dec2 = DoubleConv3D(c * 8, c * 4)
        self.up1 = nn.ConvTranspose3d(c * 4, c * 2, 2, stride=2)
        self.dec1 = DoubleConv3D(c * 4, c * 2)
        self.up0 = nn.ConvTranspose3d(c * 2, c, 2, stride=2)
        self.dec0 = DoubleConv3D(c * 2, c)
        self.head = nn.Conv3d(c, out_channels, 1)

    def forward(self, x):
        x = self.modal_stem(x)
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        b = self.global_bottleneck(b)

        x = self.up3(b)
        x = self.dec3(torch.cat([_match_size(x, e3), e3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([_match_size(x, e2), e2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([_match_size(x, e1), e1], dim=1))
        x = self.up0(x)
        x = self.dec0(torch.cat([_match_size(x, e0), e0], dim=1))
        return self.head(x)


class ModalityAttentionSwinUNet3D(ModalitySwinUNet3D):
    """Scheme A with explicit modality attention instead of concat+SE fusion."""

    def __init__(
            self,
            in_channels=4,
            out_channels=3,
            base_channels=24,
            stem_channels=None,
            swin_depth=2,
            swin_heads=8,
            swin_window_size=(3, 3, 3),
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            stem_channels=stem_channels,
            swin_depth=swin_depth,
            swin_heads=swin_heads,
            swin_window_size=swin_window_size,
        )
        self.modal_stem = ModalityAttentionFusion3D(
            in_channels=in_channels,
            out_channels=base_channels,
            stem_channels=stem_channels,
        )


class ScaleFusion3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv3d(channels * 2, channels, 1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.refine = ConvNormAct3D(channels * 2, channels, kernel_size=1)

    def forward(self, local_feat, global_feat):
        global_feat = _match_size(global_feat, local_feat)
        merged = torch.cat([local_feat, global_feat], dim=1)
        gate = self.gate(merged)
        mixed = gate * local_feat + (1.0 - gate) * global_feat
        return mixed + self.refine(merged)


class ParallelSwinEncoder3D(nn.Module):
    """Lightweight parallel Swin pyramid branch for scheme B."""

    def __init__(self, channels, depth_per_stage=(0, 1, 1, 1, 2), window_size=(3, 3, 3)):
        super().__init__()
        c = channels
        self.stage0 = ConvNormAct3D(c, c, kernel_size=3)
        self.down1 = ConvNormAct3D(c, c * 2, kernel_size=3, stride=2)
        self.swin1 = SwinBottleneck3D(c * 2, depth=depth_per_stage[1], num_heads=4, window_size=window_size)
        self.down2 = ConvNormAct3D(c * 2, c * 4, kernel_size=3, stride=2)
        self.swin2 = SwinBottleneck3D(c * 4, depth=depth_per_stage[2], num_heads=4, window_size=window_size)
        self.down3 = ConvNormAct3D(c * 4, c * 8, kernel_size=3, stride=2)
        self.swin3 = SwinBottleneck3D(c * 8, depth=depth_per_stage[3], num_heads=8, window_size=window_size)
        self.down4 = ConvNormAct3D(c * 8, c * 16, kernel_size=3, stride=2)
        self.swin4 = SwinBottleneck3D(c * 16, depth=depth_per_stage[4], num_heads=8, window_size=window_size)

    def forward(self, x):
        g0 = self.stage0(x)
        g1 = self.swin1(self.down1(g0))
        g2 = self.swin2(self.down2(g1))
        g3 = self.swin3(self.down3(g2))
        g4 = self.swin4(self.down4(g3))
        return [g0, g1, g2, g3, g4]


class ParallelModalSwinUNet3D(nn.Module):
    """
    Scheme B: modality-attention fusion, parallel CNN encoder and Swin pyramid
    encoder, then scale-wise fusion in a U-Net decoder.
    """

    def __init__(
            self,
            in_channels=4,
            out_channels=3,
            base_channels=24,
            stem_channels=None,
            swin_window_size=(3, 3, 3),
    ):
        super().__init__()
        c = base_channels
        self.modal_stem = ModalityAttentionFusion3D(
            in_channels=in_channels,
            out_channels=c,
            stem_channels=stem_channels,
        )

        self.pool = nn.MaxPool3d(2)
        self.enc0 = DoubleConv3D(c, c)
        self.enc1 = DoubleConv3D(c, c * 2)
        self.enc2 = DoubleConv3D(c * 2, c * 4)
        self.enc3 = DoubleConv3D(c * 4, c * 8)
        self.bottleneck = DoubleConv3D(c * 8, c * 16)

        self.swin_encoder = ParallelSwinEncoder3D(c, window_size=swin_window_size)
        self.fuse0 = ScaleFusion3D(c)
        self.fuse1 = ScaleFusion3D(c * 2)
        self.fuse2 = ScaleFusion3D(c * 4)
        self.fuse3 = ScaleFusion3D(c * 8)
        self.fuse4 = ScaleFusion3D(c * 16)

        self.up3 = nn.ConvTranspose3d(c * 16, c * 8, 2, stride=2)
        self.dec3 = DoubleConv3D(c * 16, c * 8)
        self.up2 = nn.ConvTranspose3d(c * 8, c * 4, 2, stride=2)
        self.dec2 = DoubleConv3D(c * 8, c * 4)
        self.up1 = nn.ConvTranspose3d(c * 4, c * 2, 2, stride=2)
        self.dec1 = DoubleConv3D(c * 4, c * 2)
        self.up0 = nn.ConvTranspose3d(c * 2, c, 2, stride=2)
        self.dec0 = DoubleConv3D(c * 2, c)
        self.head = nn.Conv3d(c, out_channels, 1)

    def forward(self, x):
        x = self.modal_stem(x)
        g0, g1, g2, g3, g4 = self.swin_encoder(x)

        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        e0 = self.fuse0(e0, g0)
        e1 = self.fuse1(e1, g1)
        e2 = self.fuse2(e2, g2)
        e3 = self.fuse3(e3, g3)
        b = self.fuse4(b, g4)

        x = self.up3(b)
        x = self.dec3(torch.cat([_match_size(x, e3), e3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([_match_size(x, e2), e2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([_match_size(x, e1), e1], dim=1))
        x = self.up0(x)
        x = self.dec0(torch.cat([_match_size(x, e0), e0], dim=1))
        return self.head(x)


class ParallelModalRawSwinUNet3D(ParallelModalSwinUNet3D):
    """
    Dual-input ablation: keep the local CNN branch on modality-attention features,
    but feed the parallel Swin/global branch from a direct 4-channel patch embedding.
    """

    def __init__(
            self,
            in_channels=4,
            out_channels=3,
            base_channels=24,
            stem_channels=None,
            swin_window_size=(3, 3, 3),
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            stem_channels=stem_channels,
            swin_window_size=swin_window_size,
        )
        self.raw_swin_embed = ConvNormAct3D(in_channels, base_channels, kernel_size=3)

    def forward(self, x):
        local_x = self.modal_stem(x)
        global_x = self.raw_swin_embed(x)
        g0, g1, g2, g3, g4 = self.swin_encoder(global_x)

        e0 = self.enc0(local_x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        e0 = self.fuse0(e0, g0)
        e1 = self.fuse1(e1, g1)
        e2 = self.fuse2(e2, g2)
        e3 = self.fuse3(e3, g3)
        b = self.fuse4(b, g4)

        x = self.up3(b)
        x = self.dec3(torch.cat([_match_size(x, e3), e3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([_match_size(x, e2), e2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([_match_size(x, e1), e1], dim=1))
        x = self.up0(x)
        x = self.dec0(torch.cat([_match_size(x, e0), e0], dim=1))
        return self.head(x)


class ParallelModalUNetNoSwin3D(nn.Module):
    """
    Ablation of ParallelModalSwinUNet3D without the parallel Swin/global branch.
    Keeps modality-specific stems and the local U-Net encoder-decoder.
    """

    def __init__(
            self,
            in_channels=4,
            out_channels=3,
            base_channels=24,
            stem_channels=None,
    ):
        super().__init__()
        c = base_channels
        self.modal_stem = ModalityAttentionFusion3D(
            in_channels=in_channels,
            out_channels=c,
            stem_channels=stem_channels,
        )

        self.pool = nn.MaxPool3d(2)
        self.enc0 = DoubleConv3D(c, c)
        self.enc1 = DoubleConv3D(c, c * 2)
        self.enc2 = DoubleConv3D(c * 2, c * 4)
        self.enc3 = DoubleConv3D(c * 4, c * 8)
        self.bottleneck = DoubleConv3D(c * 8, c * 16)

        self.up3 = nn.ConvTranspose3d(c * 16, c * 8, 2, stride=2)
        self.dec3 = DoubleConv3D(c * 16, c * 8)
        self.up2 = nn.ConvTranspose3d(c * 8, c * 4, 2, stride=2)
        self.dec2 = DoubleConv3D(c * 8, c * 4)
        self.up1 = nn.ConvTranspose3d(c * 4, c * 2, 2, stride=2)
        self.dec1 = DoubleConv3D(c * 4, c * 2)
        self.up0 = nn.ConvTranspose3d(c * 2, c, 2, stride=2)
        self.dec0 = DoubleConv3D(c * 2, c)
        self.head = nn.Conv3d(c, out_channels, 1)

    def forward(self, x):
        x = self.modal_stem(x)
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        x = self.up3(b)
        x = self.dec3(torch.cat([_match_size(x, e3), e3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([_match_size(x, e2), e2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([_match_size(x, e1), e1], dim=1))
        x = self.up0(x)
        x = self.dec0(torch.cat([_match_size(x, e0), e0], dim=1))
        return self.head(x)


class ParallelSwinUNetNoModalityStem3D(ParallelModalSwinUNet3D):
    """
    Ablation of ParallelModalSwinUNet3D without modality-specific shallow stems.
    Uses a standard early-concat convolutional stem before the same local/global
    branches.
    """

    def __init__(
            self,
            in_channels=4,
            out_channels=3,
            base_channels=24,
            swin_window_size=(3, 3, 3),
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            stem_channels=None,
            swin_window_size=swin_window_size,
        )
        self.modal_stem = ConvNormAct3D(in_channels, base_channels, kernel_size=3)


class SegResNet3D(nn.Module):
    """Compact SegResNet-style encoder-decoder baseline."""

    def __init__(self, in_channels=4, out_channels=3, base_channels=32, blocks=(1, 2, 2, 4)):
        super().__init__()
        c = base_channels
        self.stem = ConvNormAct3D(in_channels, c)
        self.enc0 = _make_res_stage(c, c, blocks[0], stride=1)
        self.enc1 = _make_res_stage(c, c * 2, blocks[1], stride=2)
        self.enc2 = _make_res_stage(c * 2, c * 4, blocks[2], stride=2)
        self.enc3 = _make_res_stage(c * 4, c * 8, blocks[3], stride=2)
        self.bottleneck = _make_res_stage(c * 8, c * 16, 2, stride=2)

        self.up3 = _UpResBlock(c * 16, c * 8)
        self.up2 = _UpResBlock(c * 8, c * 4)
        self.up1 = _UpResBlock(c * 4, c * 2)
        self.up0 = _UpResBlock(c * 2, c)
        self.head = nn.Conv3d(c, out_channels, 1)

    def forward(self, x):
        x = self.stem(x)
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        x = self.bottleneck(e3)

        x = self.up3(x, e3)
        x = self.up2(x, e2)
        x = self.up1(x, e1)
        x = self.up0(x, e0)
        return self.head(x)


class FedMUNetAttention3D(nn.Module):
    """Channel-token cross-modality attention adapted from Fed-MUnet to 3D."""

    def __init__(self, channels, dim_head=16, heads=4):
        super().__init__()
        self.heads = max(1, heads)
        self.dim_head = dim_head
        hidden = self.heads * self.dim_head
        self.pos = nn.Conv3d(channels, channels, 3, padding=1, bias=False, groups=channels)
        self.to_q = nn.Linear(channels, hidden, bias=False)
        self.to_k = nn.Linear(channels, hidden, bias=False)
        self.to_v = nn.Linear(channels, hidden, bias=False)
        self.rescale = nn.Parameter(torch.ones(self.heads, 1, 1))
        self.proj = nn.Linear(hidden, channels)

    def forward(self, x):
        x = x + self.pos(x)
        b, c, d, h, w = x.shape
        n = d * h * w
        tokens = x.permute(0, 2, 3, 4, 1).reshape(b, n, c)

        q = self.to_q(tokens).view(b, n, self.heads, self.dim_head).permute(0, 2, 1, 3)
        k = self.to_k(tokens).view(b, n, self.heads, self.dim_head).permute(0, 2, 1, 3)
        v = self.to_v(tokens).view(b, n, self.heads, self.dim_head).permute(0, 2, 1, 3)

        q = F.normalize(q.transpose(-2, -1), dim=-1)
        k = F.normalize(k.transpose(-2, -1), dim=-1)
        v = v.transpose(-2, -1)
        attn = (k @ q.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.permute(0, 3, 1, 2).reshape(b, n, self.heads * self.dim_head)
        out = self.proj(out).view(b, d, h, w, c).permute(0, 4, 1, 2, 3)
        return out


class FedMUNetFusion3D(nn.Module):
    def __init__(self, in_ch, out_ch, dim_head=16):
        super().__init__()
        heads = max(1, out_ch // max(1, dim_head))
        self.attn = FedMUNetAttention3D(in_ch, dim_head=dim_head, heads=heads)
        self.ffn = nn.Sequential(
            nn.Conv3d(in_ch, in_ch * 2, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(in_ch * 2, in_ch * 2, 3, padding=1, bias=False, groups=in_ch * 2),
            nn.GELU(),
            nn.Conv3d(in_ch * 2, in_ch, 1, bias=False),
        )
        self.reduce = ConvNormAct3D(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ffn(x) + x
        return self.reduce(x)


class FedMUNet3D(nn.Module):
    """
    3D Fed-MUnet-style baseline.

    Four modality-specific encoders first extract T1/T1c/T2/FLAIR features, then
    each scale is fused through a cross-modality attention block before a U-Net
    decoder predicts WT/TC/ET logits.
    """

    def __init__(self, in_channels=4, out_channels=3, base_channels=16, stages=4, deep_supervision=True):
        super().__init__()
        self.in_channels = in_channels
        self.stages = stages
        self.deep_supervision = deep_supervision
        c = base_channels

        self.in_convs = nn.ModuleList([
            ConvNormAct3D(1, c, kernel_size=3, stride=1)
            for _ in range(in_channels)
        ])

        self.encoders = nn.ModuleList()
        for _ in range(in_channels):
            encoder = nn.ModuleList()
            stage_ch = c
            for _ in range(stages):
                encoder.append(ConvNormAct3D(stage_ch, stage_ch * 2, kernel_size=3, stride=2))
                stage_ch *= 2
            self.encoders.append(encoder)

        stage_channels = [c * (2 ** (level + 1)) for level in range(stages)]
        self.fusions = nn.ModuleList([
            FedMUNetFusion3D(in_channels * ch, ch, dim_head=c)
            for ch in stage_channels
        ])

        self.decoders = nn.ModuleList()
        self.heads = nn.ModuleList()
        stage_ch = stage_channels[-1]
        for level in range(stages):
            out_ch = stage_ch // 2
            skip_ch = stage_channels[-2 - level] if level < stages - 1 else 0
            merge_in = out_ch + skip_ch
            self.decoders.append(nn.ModuleDict({
                "up": nn.ConvTranspose3d(stage_ch, out_ch, 2, stride=2),
                "merge": ConvNormAct3D(merge_in, out_ch, kernel_size=3, stride=1),
            }))
            self.heads.append(nn.Conv3d(out_ch, out_channels, 1))
            stage_ch = out_ch

    def forward(self, x):
        if x.shape[1] != self.in_channels:
            raise ValueError(f"FedMUNet3D expects {self.in_channels} modalities, got {x.shape[1]}")

        modal_inputs = torch.chunk(x, self.in_channels, dim=1)
        modal_feats = []
        for idx, x_mod in enumerate(modal_inputs):
            feats = []
            feat = self.in_convs[idx](x_mod)
            for layer in self.encoders[idx]:
                feat = layer(feat)
                feats.append(feat)
            modal_feats.append(feats)

        fused_feats = []
        for level in range(self.stages):
            level_feats = [feats[level] for feats in modal_feats]
            fused_feats.append(self.fusions[level](torch.cat(level_feats, dim=1)))

        logits = []
        feat = fused_feats[-1]
        for level, decoder in enumerate(self.decoders):
            feat = decoder["up"](feat)
            if level < self.stages - 1:
                skip = fused_feats[-2 - level]
                feat = torch.cat([_match_size(feat, skip), skip], dim=1)
            feat = decoder["merge"](feat)
            logits.append(self.heads[level](feat))

        final_logits = logits[-1]
        if self.training and self.deep_supervision:
            aux_logits = list(reversed(logits[:-1]))[:2]
            return final_logits, aux_logits
        return final_logits


class MedNeXtBlock3D(nn.Module):
    """Lightweight 3D MedNeXt/ConvNeXt-style residual block."""

    def __init__(self, channels, expansion=4, kernel_size=7, layer_scale=1e-6):
        super().__init__()
        padding = kernel_size // 2
        hidden = channels * expansion
        self.dwconv = nn.Conv3d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.norm = nn.InstanceNorm3d(channels, affine=True)
        self.pwconv1 = nn.Conv3d(channels, hidden, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(hidden, channels, 1)
        self.gamma = nn.Parameter(layer_scale * torch.ones(channels)) if layer_scale > 0 else None

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x * self.gamma.view(1, -1, 1, 1, 1)
        return shortcut + x


class MedNeXtStage3D(nn.Module):
    def __init__(self, channels, blocks=2, expansion=4):
        super().__init__()
        self.blocks = nn.Sequential(*[
            MedNeXtBlock3D(channels, expansion=expansion)
            for _ in range(blocks)
        ])

    def forward(self, x):
        return self.blocks(x)


class MedNeXtDown3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 2, stride=2, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.GELU(),
        )

    def forward(self, x):
        return self.down(x)


class MedNeXtUp3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.merge = nn.Sequential(
            nn.Conv3d(out_ch * 2, out_ch, 1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.GELU(),
        )
        self.refine = MedNeXtStage3D(out_ch, blocks=1)

    def forward(self, x, skip):
        x = self.up(x)
        x = _match_size(x, skip)
        x = self.merge(torch.cat([x, skip], dim=1))
        return self.refine(x)


class MedNeXtLite3D(nn.Module):
    """Compact MedNeXt-style 3D encoder-decoder baseline for WT/TC/ET logits."""

    def __init__(self, in_channels=4, out_channels=3, base_channels=24, blocks=(2, 2, 2, 2, 2)):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c, 3, padding=1, bias=False),
            nn.InstanceNorm3d(c, affine=True),
            nn.GELU(),
        )

        self.enc0 = MedNeXtStage3D(c, blocks=blocks[0])
        self.down1 = MedNeXtDown3D(c, c * 2)
        self.enc1 = MedNeXtStage3D(c * 2, blocks=blocks[1])
        self.down2 = MedNeXtDown3D(c * 2, c * 4)
        self.enc2 = MedNeXtStage3D(c * 4, blocks=blocks[2])
        self.down3 = MedNeXtDown3D(c * 4, c * 8)
        self.enc3 = MedNeXtStage3D(c * 8, blocks=blocks[3])
        self.down4 = MedNeXtDown3D(c * 8, c * 16)
        self.bottleneck = MedNeXtStage3D(c * 16, blocks=blocks[4])

        self.up3 = MedNeXtUp3D(c * 16, c * 8)
        self.up2 = MedNeXtUp3D(c * 8, c * 4)
        self.up1 = MedNeXtUp3D(c * 4, c * 2)
        self.up0 = MedNeXtUp3D(c * 2, c)
        self.head = nn.Conv3d(c, out_channels, 1)

    def forward(self, x):
        x = self.stem(x)
        e0 = self.enc0(x)
        e1 = self.enc1(self.down1(e0))
        e2 = self.enc2(self.down2(e1))
        e3 = self.enc3(self.down3(e2))
        x = self.bottleneck(self.down4(e3))

        x = self.up3(x, e3)
        x = self.up2(x, e2)
        x = self.up1(x, e1)
        x = self.up0(x, e0)
        return self.head(x)


class _UpResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.refine = ResidualBlock3D(out_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = _match_size(x, skip)
        return self.refine(x + skip)


def _make_res_stage(in_ch, out_ch, num_blocks, stride):
    blocks = [ResidualBlock3D(in_ch, out_ch, stride=stride)]
    for _ in range(num_blocks - 1):
        blocks.append(ResidualBlock3D(out_ch, out_ch, stride=1))
    return nn.Sequential(*blocks)


def _match_size(x, ref):
    if x.shape[2:] == ref.shape[2:]:
        return x
    return F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)


def _build_benchmark_fets_dynunet(in_channels=4, out_channels=3):
    try:
        import monai.networks.blocks.dynunet_block as dynunet_block
        from monai.networks.blocks.convolutions import Convolution
        from monai.networks.blocks.dynunet_block import get_output_padding, get_padding
        from monai.networks.layers.factories import Act, Norm
        from monai.networks.nets import DynUNet
    except ImportError as exc:
        raise ImportError(
            "MONAI is required for benchmark_fets_dynunet. Install with: pip install monai"
        ) from exc

    def modified_get_conv_layer(
        spatial_dims,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout=None,
        bias=True,
        conv_only=True,
        is_transposed=False,
    ):
        padding = get_padding(kernel_size, stride)
        output_padding = None
        if is_transposed:
            output_padding = get_output_padding(kernel_size, stride, padding)
        return Convolution(
            spatial_dims,
            in_channels,
            out_channels,
            strides=stride,
            kernel_size=kernel_size,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            conv_only=conv_only,
            is_transposed=is_transposed,
            padding=padding,
            output_padding=output_padding,
        )

    dynunet_block.get_conv_layer = modified_get_conv_layer
    return DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=[3, 3, 3, 3, 3],
        filters=[32, 64, 128, 256, 512],
        strides=[1, 2, 2, 2, 2],
        upsample_kernel_size=[2, 2, 2, 2],
        norm_name=("INSTANCE", {"affine": False}),
        act_name=("LeakyReLu", {"negative_slope": 0.01}),
        trans_bias=True,
    )


def build_baseline_model(name, in_channels=4, out_channels=3, base_channels=32,
                         patch_size=(96, 96, 96), nas_op="mixed", search_arch=None):
    name = name.lower()
    if name == "unet3d":
        return UNet3D(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
    if name in {"fednas_unet3d", "fednas_unet"}:
        return FedNASUNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            nas_op=nas_op,
        )
    if name in {"modal_swin_unet3d", "mscs_unet3d"}:
        return ModalitySwinUNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
        )
    if name in {"modal_attn_swin_unet3d", "msca_swin_unet3d"}:
        return ModalityAttentionSwinUNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
        )
    if name in {"parallel_modal_swin_unet3d", "parallel_msca_swin_unet3d"}:
        return ParallelModalSwinUNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
        )
    if name in {"parallel_modal_raw_swin_unet3d", "parallel_raw_swin_unet3d"}:
        return ParallelModalRawSwinUNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
        )
    if name in {"parallel_modal_unet_no_swin3d", "parallel_no_swin_unet3d"}:
        return ParallelModalUNetNoSwin3D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
        )
    if name in {"parallel_swin_unet_no_modal_stem3d", "parallel_no_modal_stem_unet3d"}:
        return ParallelSwinUNetNoModalityStem3D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
        )
    if name in {"search_modal_swin_unet3d", "search_scheme_a_unet3d"}:
        from nnunetv2_custom.modal_swin_unet import SearchableModalSwinUNet
        return SearchableModalSwinUNet(
            input_channels=in_channels,
            num_classes=out_channels,
            base_channels=base_channels,
            deep_supervision=False,
            swin_depth=2,
            swin_window_size=(3, 3, 3),
        )
    if name in {"search_modal_attn_swin_unet3d", "search_recommended_unet3d"}:
        from nnunetv2_custom.modal_swin_unet import SearchableModalAttnSwinUNet
        return SearchableModalAttnSwinUNet(
            input_channels=in_channels,
            num_classes=out_channels,
            base_channels=base_channels,
            deep_supervision=False,
            swin_depth=2,
            swin_window_size=(3, 3, 3),
        )
    if name in {"search_parallel_modal_swin_unet3d", "search_scheme_b_unet3d"}:
        from nnunetv2_custom.modal_swin_unet import SearchableParallelModalSwinUNet
        return SearchableParallelModalSwinUNet(
            input_channels=in_channels,
            num_classes=out_channels,
            base_channels=base_channels,
            deep_supervision=False,
            swin_window_size=(3, 3, 3),
        )
    if name in {"searched_modal_swin_unet3d", "searched_scheme_a_unet3d"}:
        from nnunetv2_custom.modal_swin_unet import SearchedModalSwinUNet
        if not search_arch:
            raise ValueError("searched_modal_swin_unet3d requires --search_arch")
        return SearchedModalSwinUNet(
            input_channels=in_channels,
            num_classes=out_channels,
            base_channels=base_channels,
            deep_supervision=False,
            swin_depth=2,
            swin_window_size=(3, 3, 3),
            search_arch=search_arch,
        )
    if name in {"searched_modal_attn_swin_unet3d", "searched_recommended_unet3d"}:
        from nnunetv2_custom.modal_swin_unet import SearchedModalAttnSwinUNet
        if not search_arch:
            raise ValueError("searched_modal_attn_swin_unet3d requires --search_arch")
        return SearchedModalAttnSwinUNet(
            input_channels=in_channels,
            num_classes=out_channels,
            base_channels=base_channels,
            deep_supervision=False,
            swin_depth=2,
            swin_window_size=(3, 3, 3),
            search_arch=search_arch,
        )
    if name in {"searched_parallel_modal_swin_unet3d", "searched_scheme_b_unet3d"}:
        from nnunetv2_custom.modal_swin_unet import SearchedParallelModalSwinUNet
        if not search_arch:
            raise ValueError("searched_parallel_modal_swin_unet3d requires --search_arch")
        return SearchedParallelModalSwinUNet(
            input_channels=in_channels,
            num_classes=out_channels,
            base_channels=base_channels,
            deep_supervision=False,
            swin_window_size=(3, 3, 3),
            search_arch=search_arch,
        )
    if name == "segresnet":
        return SegResNet3D(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
    if name == "fedmunet3d":
        return FedMUNet3D(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
    if name in {"mednext_lite", "mednextlite"}:
        return MedNeXtLite3D(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
    if name in {"benchmark_fets_dynunet", "fets_dynunet"}:
        return _build_benchmark_fets_dynunet(in_channels=in_channels, out_channels=out_channels)
    if name == "monai_segresnet":
        try:
            from monai.networks.nets import SegResNet
        except ImportError as exc:
            raise ImportError("MONAI is required for monai_segresnet. Install with: pip install monai") from exc
        return SegResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=base_channels,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
        )
    if name == "monai_swinunetr":
        try:
            from monai.networks.nets import SwinUNETR
        except ImportError as exc:
            raise ImportError("MONAI is required for monai_swinunetr. Install with: pip install monai") from exc
        try:
            return SwinUNETR(
                in_channels=in_channels,
                out_channels=out_channels,
                feature_size=48,
                use_checkpoint=True,
                spatial_dims=3,
            )
        except TypeError:
            return SwinUNETR(
                img_size=patch_size,
                in_channels=in_channels,
                out_channels=out_channels,
                feature_size=48,
                use_checkpoint=True,
                spatial_dims=3,
            )
    raise ValueError(f"Unknown baseline model: {name}")
