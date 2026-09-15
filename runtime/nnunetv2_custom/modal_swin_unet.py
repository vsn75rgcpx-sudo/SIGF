import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ModalityStemFusion3D(nn.Module):
    def __init__(self, in_channels, out_channels, stem_channels=None):
        super().__init__()
        stem_channels = stem_channels or max(4, out_channels // 4)
        self.in_channels = in_channels
        self.stems = nn.ModuleList([
            nn.Sequential(
                ConvNormAct3D(1, stem_channels),
                ConvNormAct3D(stem_channels, stem_channels),
            )
            for _ in range(in_channels)
        ])
        fused_channels = in_channels * stem_channels
        hidden = max(4, fused_channels // 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(fused_channels, hidden, 1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(hidden, fused_channels, 1),
            nn.Sigmoid(),
        )
        self.reduce = ConvNormAct3D(fused_channels, out_channels, kernel_size=1)

    def forward(self, x):
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input modalities, got {x.shape[1]}")
        feats = [stem(x[:, idx:idx + 1]) for idx, stem in enumerate(self.stems)]
        fused = torch.cat(feats, dim=1)
        fused = fused * self.se(fused)
        return self.reduce(fused)


class ModalityAttentionFusion3D(nn.Module):
    def __init__(self, in_channels, out_channels, stem_channels=None):
        super().__init__()
        stem_channels = stem_channels or max(8, out_channels // 2)
        self.in_channels = in_channels
        self.stems = nn.ModuleList([
            nn.Sequential(
                ConvNormAct3D(1, stem_channels),
                ConvNormAct3D(stem_channels, stem_channels),
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
            raise ValueError(f"Expected {self.in_channels} input modalities, got {x.shape[1]}")
        feats = [stem(x[:, idx:idx + 1]) for idx, stem in enumerate(self.stems)]
        stacked = torch.stack(feats, dim=1)
        weights = torch.softmax(self.attn(torch.cat(feats, dim=1)).unsqueeze(2), dim=1)
        fused = (stacked * weights).sum(dim=1)
        return self.reduce(fused)


class LayerNormChannel(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm(x)
        return x.permute(0, 4, 1, 2, 3).contiguous()


def window_partition(x, window_size):
    b, c, d, h, w = x.shape
    wd, wh, ww = window_size
    x = x.permute(0, 2, 3, 4, 1).contiguous()
    x = x.view(b, d // wd, wd, h // wh, wh, w // ww, ww, c)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return windows.view(-1, wd * wh * ww, c)


def window_reverse(windows, window_size, b, d, h, w):
    wd, wh, ww = window_size
    c = windows.shape[-1]
    x = windows.view(b, d // wd, h // wh, w // ww, wd, wh, ww, c)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(b, d, h, w, c)
    return x.permute(0, 4, 1, 2, 3).contiguous()


def compute_attn_mask(d, h, w, window_size, shift_size, device):
    wd, wh, ww = window_size
    sd, sh, sw = shift_size
    if sd == 0 and sh == 0 and sw == 0:
        return None

    img_mask = torch.zeros((1, 1, d, h, w), device=device)
    d_slices = (slice(0, -wd), slice(-wd, -sd), slice(-sd, None))
    h_slices = (slice(0, -wh), slice(-wh, -sh), slice(-sh, None))
    w_slices = (slice(0, -ww), slice(-ww, -sw), slice(-sw, None))
    cnt = 0
    for ds in d_slices:
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, :, ds, hs, ws] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size).view(-1, wd * wh * ww)
    attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(1)
    return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)


class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size=(3, 3, 3), num_heads=8):
        super().__init__()
        self.dim = dim
        self.window_size = tuple(window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        wd, wh, ww = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * wd - 1) * (2 * wh - 1) * (2 * ww - 1), num_heads)
        )
        coords = torch.stack(torch.meshgrid(
            torch.arange(wd), torch.arange(wh), torch.arange(ww), indexing="ij"
        ))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += wd - 1
        relative_coords[:, :, 1] += wh - 1
        relative_coords[:, :, 2] += ww - 1
        relative_coords[:, :, 0] *= (2 * wh - 1) * (2 * ww - 1)
        relative_coords[:, :, 1] *= (2 * ww - 1)
        self.register_buffer("relative_position_index", relative_coords.sum(-1))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        b_windows, n_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(b_windows, n_tokens, 3, self.num_heads, channels // self.num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        attn = (q @ k) * self.scale

        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(n_tokens, n_tokens, -1).permute(2, 0, 1).unsqueeze(0)
        attn = attn + bias
        if mask is not None:
            n_windows = mask.shape[0]
            attn = attn.view(b_windows // n_windows, n_windows, self.num_heads, n_tokens, n_tokens)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n_tokens, n_tokens)

        attn = attn.softmax(dim=-1)
        v = v.permute(0, 2, 1, 3)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(b_windows, n_tokens, channels)
        return self.proj(out)


def valid_num_heads(channels, requested_heads):
    requested_heads = max(1, int(requested_heads))
    requested_heads = min(requested_heads, channels)
    while channels % requested_heads != 0 and requested_heads > 1:
        requested_heads -= 1
    return requested_heads


def build_searched_ffn(dim, choice, mlp_ratio=2.0):
    name = str(choice or "conv3").lower()
    if "dilated" in name or "dil3" in name:
        return nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.InstanceNorm3d(dim, affine=True),
            nn.GELU(),
        )
    if "conv5" in name:
        return nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=5, padding=2, bias=False),
            nn.InstanceNorm3d(dim, affine=True),
            nn.GELU(),
        )
    if "conv1" in name:
        return nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1, bias=False),
            nn.InstanceNorm3d(dim, affine=True),
            nn.GELU(),
        )
    if "identity" in name or name in {"skip", "none"}:
        return nn.Identity()
    if "dw" in name or "depthwise" in name or "separable" in name:
        return nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.Conv3d(dim, dim, kernel_size=1, bias=False),
            nn.InstanceNorm3d(dim, affine=True),
            nn.GELU(),
        )
    if "mlp" in name:
        hidden = int(dim * mlp_ratio)
        return nn.Sequential(
            nn.Conv3d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv3d(hidden, dim, 1),
        )
    return nn.Sequential(
        nn.Conv3d(dim, dim, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm3d(dim, affine=True),
        nn.GELU(),
    )


class DynamicSwinBottleneckBlock3D(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            window_size=(3, 3, 3),
            shift_size=(0, 0, 0),
            mlp_ratio=2.0,
            ffn_choice="mlp",
            residual_alpha=1.0,
    ):
        super().__init__()
        self.window_size = tuple(window_size)
        self.shift_size = tuple(shift_size)
        self.residual_alpha = float(residual_alpha)
        self.norm1 = LayerNormChannel(dim)
        self.attn = WindowAttention3D(dim, self.window_size, valid_num_heads(dim, num_heads))
        self.norm2 = LayerNormChannel(dim)
        self.ffn = build_searched_ffn(dim, ffn_choice, mlp_ratio=mlp_ratio)

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
        shift = (
            min(self.shift_size[0], wd - 1),
            min(self.shift_size[1], wh - 1),
            min(self.shift_size[2], ww - 1),
        )
        if any(shift):
            shifted = torch.roll(x, shifts=tuple(-s for s in shift), dims=(2, 3, 4))
            mask = compute_attn_mask(d, h, w, self.window_size, shift, x.device)
        else:
            shifted = x
            mask = None

        windows = window_partition(shifted, self.window_size)
        attn_windows = self.attn(windows, mask=mask)
        shifted = window_reverse(attn_windows, self.window_size, b, d, h, w)
        if any(shift):
            x = torch.roll(shifted, shifts=shift, dims=(2, 3, 4))
        else:
            x = shifted

        od, oh, ow = original_size
        global_feature = shortcut + x[:, :, :od, :oh, :ow]
        x = (1.0 - self.residual_alpha) * shortcut + self.residual_alpha * global_feature
        return x + self.ffn(self.norm2(x))


class SwinBottleneck3D(nn.Module):
    def __init__(self, channels, depth=2, window_size=(3, 3, 3)):
        super().__init__()
        heads = max(1, min(8, channels // 32))
        while channels % heads != 0 and heads > 1:
            heads -= 1
        blocks = []
        for idx in range(depth):
            shift = (0, 0, 0) if idx % 2 == 0 else tuple(v // 2 for v in window_size)
            blocks.append(DynamicSwinBottleneckBlock3D(channels, heads, window_size, shift))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


def load_search_architecture(arch_config):
    if arch_config is None or arch_config == "":
        return None
    if isinstance(arch_config, dict):
        return arch_config
    path = Path(str(arch_config)).expanduser()
    with path.open("r") as f:
        return json.load(f)


def select_arch_blocks(arch_config, stage_id=-1, depth=2):
    arch_config = load_search_architecture(arch_config)
    if not isinstance(arch_config, dict) or "stages" not in arch_config:
        return []
    stages = sorted(arch_config["stages"], key=lambda item: int(item.get("stage_id", 0)))
    if not stages:
        return []
    if stage_id < 0:
        stage = stages[stage_id]
    else:
        stage = next((item for item in stages if int(item.get("stage_id", -1)) == int(stage_id)), stages[-1])
    blocks = list(stage.get("blocks", []))
    if not blocks:
        return []
    while len(blocks) < depth:
        blocks.append(blocks[-1])
    return blocks[:depth]


class SearchedSwinBottleneck3D(nn.Module):
    """
    Fixed retraining module for a searched CardiacNAS-style genotype.

    It reads exported per-block gamma_global=(h,k,alpha) records and builds
    ordinary PyTorch blocks. The search itself happens before nnU-Net training;
    nnU-Net receives the selected architecture for stable retraining.
    """

    def __init__(
            self,
            channels,
            arch_config,
            depth=2,
            stage_id=-1,
            fallback_window_size=(3, 3, 3),
    ):
        super().__init__()
        blocks_cfg = select_arch_blocks(arch_config, stage_id=stage_id, depth=depth)
        blocks = []
        for idx in range(depth):
            cfg = blocks_cfg[idx] if idx < len(blocks_cfg) else {}
            attn_choice = cfg.get("attn_choice", {}) if isinstance(cfg, dict) else {}
            window_size = tuple(attn_choice.get("window_size", fallback_window_size))
            num_heads = valid_num_heads(channels, attn_choice.get("num_heads", max(1, min(8, channels // 32))))
            ffn_choice = cfg.get("ffn_choice", "conv3") if isinstance(cfg, dict) else "conv3"
            residual_alpha = float(cfg.get("residual_alpha", 1.0)) if isinstance(cfg, dict) else 1.0
            raw_shift = cfg.get("shift_size", None) if isinstance(cfg, dict) else None
            if raw_shift is None:
                shift = (0, 0, 0) if idx % 2 == 0 else tuple(v // 2 for v in window_size)
            else:
                shift = tuple(min(int(s), max(int(w) - 1, 0)) for s, w in zip(raw_shift, window_size))
            blocks.append(
                DynamicSwinBottleneckBlock3D(
                    channels,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift,
                    ffn_choice=ffn_choice,
                    residual_alpha=residual_alpha,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class MixedSearchedFFN3D(nn.Module):
    def __init__(self, dim, choices=("conv3", "conv5", "dilated3")):
        super().__init__()
        self.choices = tuple(choices)
        self.ops = nn.ModuleList([build_searched_ffn(dim, choice) for choice in self.choices])
        self.alpha = nn.Parameter(torch.zeros(len(self.choices)))

    def _weights(self, temp=1.0, hard=False):
        logits = self.alpha.unsqueeze(0)
        return F.gumbel_softmax(logits, tau=max(float(temp), 1e-6), hard=hard, dim=1).squeeze(0)

    def forward(self, x, temp=1.0, hard=False):
        weights = self._weights(temp=temp, hard=hard)
        out = 0
        for weight, op in zip(weights, self.ops):
            out = out + weight * op(x)
        return out

    def selected(self):
        probs = torch.softmax(self.alpha.detach().cpu(), dim=0)
        idx = int(torch.argmax(probs).item())
        return idx, self.choices[idx], [float(v) for v in probs.tolist()]


class MixedWindowAttentionChoice3D(nn.Module):
    def __init__(self, dim, window_size=(3, 3, 3), head_choices=(2, 4, 6)):
        super().__init__()
        self.window_size = tuple(window_size)
        self.head_choices = tuple(valid_num_heads(dim, heads) for heads in head_choices)
        self.atts = nn.ModuleList([
            WindowAttention3D(dim, self.window_size, num_heads=heads)
            for heads in self.head_choices
        ])
        self.alpha = nn.Parameter(torch.zeros(len(self.head_choices)))

    def _weights(self, temp=1.0, hard=False):
        logits = self.alpha.unsqueeze(0)
        return F.gumbel_softmax(logits, tau=max(float(temp), 1e-6), hard=hard, dim=1).squeeze(0)

    def forward(self, windows, mask=None, temp=1.0, hard=False):
        weights = self._weights(temp=temp, hard=hard)
        out = 0
        for weight, attn in zip(weights, self.atts):
            out = out + weight * attn(windows, mask=mask)
        return out

    def selected(self):
        probs = torch.softmax(self.alpha.detach().cpu(), dim=0)
        idx = int(torch.argmax(probs).item())
        return idx, self.head_choices[idx], [float(v) for v in probs.tolist()]


class SearchableSwinBottleneckBlock3D(nn.Module):
    def __init__(
            self,
            dim,
            window_size=(3, 3, 3),
            shift_size=(0, 0, 0),
            head_choices=(2, 4, 6),
            ffn_choices=("conv3", "conv5", "dilated3"),
            residual_alpha_choices=(0.25, 0.5, 0.75),
    ):
        super().__init__()
        self.window_size = tuple(window_size)
        self.shift_size = tuple(shift_size)
        self.residual_alpha_choices = tuple(float(v) for v in residual_alpha_choices)
        self.norm1 = LayerNormChannel(dim)
        self.attn = MixedWindowAttentionChoice3D(dim, self.window_size, head_choices=head_choices)
        self.norm2 = LayerNormChannel(dim)
        self.ffn = MixedSearchedFFN3D(dim, choices=ffn_choices)
        self.residual_alpha_logits = nn.Parameter(torch.zeros(len(self.residual_alpha_choices)))
        self.temp = 1.0
        self.hard = False

    def arch_parameters(self):
        return [self.attn.alpha, self.ffn.alpha, self.residual_alpha_logits]

    def set_search_mode(self, temp=1.0, hard=False):
        self.temp = float(temp)
        self.hard = bool(hard)

    def _alpha_value(self):
        candidates = torch.tensor(
            self.residual_alpha_choices,
            device=self.residual_alpha_logits.device,
            dtype=self.residual_alpha_logits.dtype,
        )
        weights = F.gumbel_softmax(
            self.residual_alpha_logits.unsqueeze(0),
            tau=max(self.temp, 1e-6),
            hard=self.hard,
            dim=1,
        ).squeeze(0)
        return torch.sum(weights * candidates)

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
        shift = (
            min(self.shift_size[0], wd - 1),
            min(self.shift_size[1], wh - 1),
            min(self.shift_size[2], ww - 1),
        )
        if any(shift):
            shifted = torch.roll(x, shifts=tuple(-s for s in shift), dims=(2, 3, 4))
            mask = compute_attn_mask(d, h, w, self.window_size, shift, x.device)
        else:
            shifted = x
            mask = None

        windows = window_partition(shifted, self.window_size)
        attn_windows = self.attn(windows, mask=mask, temp=self.temp, hard=self.hard)
        shifted = window_reverse(attn_windows, self.window_size, b, d, h, w)
        if any(shift):
            x = torch.roll(shifted, shifts=shift, dims=(2, 3, 4))
        else:
            x = shifted

        od, oh, ow = original_size
        alpha = self._alpha_value()
        global_feature = shortcut + x[:, :, :od, :oh, :ow]
        x = (1.0 - alpha) * shortcut + alpha * global_feature
        return x + self.ffn(self.norm2(x), temp=self.temp, hard=self.hard)

    def export_block(self, block_index=0):
        ffn_idx, ffn_choice, ffn_probs = self.ffn.selected()
        attn_idx, num_heads, attn_probs = self.attn.selected()
        alpha_probs_t = torch.softmax(self.residual_alpha_logits.detach().cpu(), dim=0)
        alpha_idx = int(torch.argmax(alpha_probs_t).item())
        residual_alpha = float(self.residual_alpha_choices[alpha_idx])
        alpha_probs = [float(v) for v in alpha_probs_t.tolist()]
        return {
            "block_index": int(block_index),
            "ffn_idx": int(ffn_idx),
            "ffn_choice": ffn_choice,
            "ffn_probs": ffn_probs,
            "attn_idx": int(attn_idx),
            "attn_choice": {
                "window_size": list(self.window_size),
                "num_heads": int(num_heads),
            },
            "attn_probs": attn_probs,
            "residual_alpha_idx": int(alpha_idx),
            "residual_alpha": residual_alpha,
            "residual_alpha_candidates": list(self.residual_alpha_choices),
            "residual_alpha_probs": alpha_probs,
            "gamma_global": {
                "h": int(num_heads),
                "k": ffn_choice,
                "alpha": residual_alpha,
            },
            "shift_size": list(self.shift_size),
        }


class SearchableSwinBottleneck3D(nn.Module):
    def __init__(
            self,
            channels,
            depth=2,
            window_size=(3, 3, 3),
            head_choices=(2, 4, 6),
            ffn_choices=("conv3", "conv5", "dilated3"),
            residual_alpha_choices=(0.25, 0.5, 0.75),
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for idx in range(depth):
            shift = (0, 0, 0) if idx % 2 == 0 else tuple(v // 2 for v in window_size)
            self.blocks.append(
                SearchableSwinBottleneckBlock3D(
                    channels,
                    window_size=window_size,
                    shift_size=shift,
                    head_choices=head_choices,
                    ffn_choices=ffn_choices,
                    residual_alpha_choices=residual_alpha_choices,
                )
            )

    def arch_parameters(self):
        params = []
        for block in self.blocks:
            params.extend(block.arch_parameters())
        return params

    def set_search_mode(self, temp=1.0, hard=False):
        for block in self.blocks:
            block.set_search_mode(temp=temp, hard=hard)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def export_blocks(self, start_index=0):
        return [
            block.export_block(block_index=start_index + idx)
            for idx, block in enumerate(self.blocks)
        ]


def repeat_blocks_for_stages(blocks, num_stages=4):
    stages = []
    for stage_id in range(num_stages):
        if stage_id == num_stages - 1:
            stage_blocks = blocks
        else:
            stage_blocks = [dict(blocks[0], block_index=stage_id)]
        stages.append({
            "stage_id": stage_id,
            "num_blocks": len(stage_blocks),
            "blocks": stage_blocks,
        })
    return stages


def match_size(x, ref):
    if x.shape[2:] == ref.shape[2:]:
        return x
    return F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)


class ModalSwinUNet(nn.Module):
    def __init__(
            self,
            input_channels,
            num_classes,
            base_channels=32,
            deep_supervision=True,
            swin_depth=2,
            swin_window_size=(3, 3, 3),
    ):
        super().__init__()
        c = int(base_channels)
        self.deep_supervision = deep_supervision
        self.modal_stem = ModalityStemFusion3D(input_channels, c)
        self.enc0 = DoubleConv3D(c, c)
        self.enc1 = DoubleConv3D(c, c * 2)
        self.enc2 = DoubleConv3D(c * 2, c * 4)
        self.enc3 = DoubleConv3D(c * 4, c * 8)
        self.bottleneck = DoubleConv3D(c * 8, c * 16)
        self.global_bottleneck = SwinBottleneck3D(c * 16, depth=swin_depth, window_size=swin_window_size)

        self.pool = nn.MaxPool3d(2)
        self.up3 = nn.ConvTranspose3d(c * 16, c * 8, 2, stride=2)
        self.dec3 = DoubleConv3D(c * 16, c * 8)
        self.up2 = nn.ConvTranspose3d(c * 8, c * 4, 2, stride=2)
        self.dec2 = DoubleConv3D(c * 8, c * 4)
        self.up1 = nn.ConvTranspose3d(c * 4, c * 2, 2, stride=2)
        self.dec1 = DoubleConv3D(c * 4, c * 2)
        self.up0 = nn.ConvTranspose3d(c * 2, c, 2, stride=2)
        self.dec0 = DoubleConv3D(c * 2, c)

        self.head = nn.Conv3d(c, num_classes, 1)
        self.aux1 = nn.Conv3d(c * 2, num_classes, 1)
        self.aux2 = nn.Conv3d(c * 4, num_classes, 1)
        self.aux3 = nn.Conv3d(c * 8, num_classes, 1)

    def forward(self, x):
        x = self.modal_stem(x)
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        b = self.global_bottleneck(b)

        x = self.up3(b)
        x = self.dec3(torch.cat([match_size(x, e3), e3], dim=1))
        out3 = self.aux3(x)
        x = self.up2(x)
        x = self.dec2(torch.cat([match_size(x, e2), e2], dim=1))
        out2 = self.aux2(x)
        x = self.up1(x)
        x = self.dec1(torch.cat([match_size(x, e1), e1], dim=1))
        out1 = self.aux1(x)
        x = self.up0(x)
        x = self.dec0(torch.cat([match_size(x, e0), e0], dim=1))
        out = self.head(x)

        if self.deep_supervision:
            return [out, out1, out2, out3]
        return out


class ModalAttnSwinUNet(ModalSwinUNet):
    def __init__(
            self,
            input_channels,
            num_classes,
            base_channels=32,
            deep_supervision=True,
            swin_depth=2,
            swin_window_size=(3, 3, 3),
    ):
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            deep_supervision=deep_supervision,
            swin_depth=swin_depth,
            swin_window_size=swin_window_size,
        )
        self.modal_stem = ModalityAttentionFusion3D(input_channels, int(base_channels))


class SearchedModalSwinUNet(ModalSwinUNet):
    def __init__(
            self,
            input_channels,
            num_classes,
            base_channels=32,
            deep_supervision=True,
            swin_depth=2,
            swin_window_size=(3, 3, 3),
            search_arch=None,
            search_stage_id=-1,
    ):
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            deep_supervision=deep_supervision,
            swin_depth=swin_depth,
            swin_window_size=swin_window_size,
        )
        c = int(base_channels)
        self.global_bottleneck = SearchedSwinBottleneck3D(
            c * 16,
            arch_config=search_arch,
            depth=swin_depth,
            stage_id=search_stage_id,
            fallback_window_size=swin_window_size,
        )


class SearchedModalAttnSwinUNet(SearchedModalSwinUNet):
    def __init__(
            self,
            input_channels,
            num_classes,
            base_channels=32,
            deep_supervision=True,
            swin_depth=2,
            swin_window_size=(3, 3, 3),
            search_arch=None,
            search_stage_id=-1,
    ):
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            deep_supervision=deep_supervision,
            swin_depth=swin_depth,
            swin_window_size=swin_window_size,
            search_arch=search_arch,
            search_stage_id=search_stage_id,
        )
        self.modal_stem = ModalityAttentionFusion3D(input_channels, int(base_channels))


class SearchableModalSwinUNet(ModalSwinUNet):
    def __init__(
            self,
            input_channels,
            num_classes,
            base_channels=32,
            deep_supervision=True,
            swin_depth=2,
            swin_window_size=(3, 3, 3),
    ):
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            deep_supervision=deep_supervision,
            swin_depth=swin_depth,
            swin_window_size=swin_window_size,
        )
        c = int(base_channels)
        self.global_bottleneck = SearchableSwinBottleneck3D(
            c * 16,
            depth=swin_depth,
            window_size=swin_window_size,
        )

    def arch_parameters(self):
        return self.global_bottleneck.arch_parameters()

    def weight_parameters(self):
        arch_ids = {id(param) for param in self.arch_parameters()}
        return [param for param in self.parameters() if id(param) not in arch_ids]

    def set_search_mode(self, temp=1.0, hard=False):
        self.global_bottleneck.set_search_mode(temp=temp, hard=hard)

    def export_search_architecture(self, output_path, stage_id=3):
        blocks = self.global_bottleneck.export_blocks()
        arch = {
            "model": {
                "in_channels": 4,
                "num_classes": self.head.out_channels,
                "genotype": "gamma_global=(h,k,alpha)",
                "searched_in": self.__class__.__name__,
            },
            "stages": repeat_blocks_for_stages(blocks, num_stages=stage_id + 1),
            "description": "Architecture searched on the modal nnU-Net global bottleneck.",
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(arch, indent=4))
        return arch


class SearchableModalAttnSwinUNet(SearchableModalSwinUNet):
    def __init__(
            self,
            input_channels,
            num_classes,
            base_channels=32,
            deep_supervision=True,
            swin_depth=2,
            swin_window_size=(3, 3, 3),
    ):
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            deep_supervision=deep_supervision,
            swin_depth=swin_depth,
            swin_window_size=swin_window_size,
        )
        self.modal_stem = ModalityAttentionFusion3D(input_channels, int(base_channels))


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
        global_feat = match_size(global_feat, local_feat)
        merged = torch.cat([local_feat, global_feat], dim=1)
        gate = self.gate(merged)
        mixed = gate * local_feat + (1.0 - gate) * global_feat
        return mixed + self.refine(merged)


class ConcatFusion3D(nn.Module):
    """Gate-free fusion used to isolate the contribution of adaptive mixing."""

    def __init__(self, channels):
        super().__init__()
        self.fuse = ConvNormAct3D(channels * 2, channels, kernel_size=1)

    def forward(self, local_feat, global_feat):
        global_feat = match_size(global_feat, local_feat)
        return self.fuse(torch.cat([local_feat, global_feat], dim=1))


class ParallelSwinEncoder3D(nn.Module):
    def __init__(self, channels, window_size=(3, 3, 3)):
        super().__init__()
        c = channels
        self.stage0 = ConvNormAct3D(c, c)
        self.down1 = ConvNormAct3D(c, c * 2, stride=2)
        self.swin1 = SwinBottleneck3D(c * 2, depth=1, window_size=window_size)
        self.down2 = ConvNormAct3D(c * 2, c * 4, stride=2)
        self.swin2 = SwinBottleneck3D(c * 4, depth=1, window_size=window_size)
        self.down3 = ConvNormAct3D(c * 4, c * 8, stride=2)
        self.swin3 = SwinBottleneck3D(c * 8, depth=1, window_size=window_size)
        self.down4 = ConvNormAct3D(c * 8, c * 16, stride=2)
        self.swin4 = SwinBottleneck3D(c * 16, depth=2, window_size=window_size)

    def forward(self, x):
        g0 = self.stage0(x)
        g1 = self.swin1(self.down1(g0))
        g2 = self.swin2(self.down2(g1))
        g3 = self.swin3(self.down3(g2))
        g4 = self.swin4(self.down4(g3))
        return [g0, g1, g2, g3, g4]


class SearchedParallelSwinEncoder3D(nn.Module):
    def __init__(self, channels, arch_config, window_size=(3, 3, 3)):
        super().__init__()
        c = channels
        self.stage0 = ConvNormAct3D(c, c)
        self.down1 = ConvNormAct3D(c, c * 2, stride=2)
        self.swin1 = SearchedSwinBottleneck3D(c * 2, arch_config, depth=1, stage_id=0, fallback_window_size=window_size)
        self.down2 = ConvNormAct3D(c * 2, c * 4, stride=2)
        self.swin2 = SearchedSwinBottleneck3D(c * 4, arch_config, depth=1, stage_id=1, fallback_window_size=window_size)
        self.down3 = ConvNormAct3D(c * 4, c * 8, stride=2)
        self.swin3 = SearchedSwinBottleneck3D(c * 8, arch_config, depth=1, stage_id=2, fallback_window_size=window_size)
        self.down4 = ConvNormAct3D(c * 8, c * 16, stride=2)
        self.swin4 = SearchedSwinBottleneck3D(c * 16, arch_config, depth=2, stage_id=-1, fallback_window_size=window_size)

    def forward(self, x):
        g0 = self.stage0(x)
        g1 = self.swin1(self.down1(g0))
        g2 = self.swin2(self.down2(g1))
        g3 = self.swin3(self.down3(g2))
        g4 = self.swin4(self.down4(g3))
        return [g0, g1, g2, g3, g4]


class SearchableParallelSwinEncoder3D(nn.Module):
    def __init__(self, channels, window_size=(3, 3, 3)):
        super().__init__()
        c = channels
        self.stage0 = ConvNormAct3D(c, c)
        self.down1 = ConvNormAct3D(c, c * 2, stride=2)
        self.swin1 = SearchableSwinBottleneck3D(c * 2, depth=1, window_size=window_size)
        self.down2 = ConvNormAct3D(c * 2, c * 4, stride=2)
        self.swin2 = SearchableSwinBottleneck3D(c * 4, depth=1, window_size=window_size)
        self.down3 = ConvNormAct3D(c * 4, c * 8, stride=2)
        self.swin3 = SearchableSwinBottleneck3D(c * 8, depth=1, window_size=window_size)
        self.down4 = ConvNormAct3D(c * 8, c * 16, stride=2)
        self.swin4 = SearchableSwinBottleneck3D(c * 16, depth=2, window_size=window_size)
        self.search_stages = [self.swin1, self.swin2, self.swin3, self.swin4]

    def arch_parameters(self):
        params = []
        for stage in self.search_stages:
            params.extend(stage.arch_parameters())
        return params

    def set_search_mode(self, temp=1.0, hard=False):
        for stage in self.search_stages:
            stage.set_search_mode(temp=temp, hard=hard)

    def export_stages(self):
        stages = []
        block_index = 0
        for stage_id, stage in enumerate(self.search_stages):
            blocks = stage.export_blocks(start_index=block_index)
            block_index += len(blocks)
            stages.append({
                "stage_id": stage_id,
                "num_blocks": len(blocks),
                "blocks": blocks,
            })
        return stages

    def forward(self, x):
        g0 = self.stage0(x)
        g1 = self.swin1(self.down1(g0))
        g2 = self.swin2(self.down2(g1))
        g3 = self.swin3(self.down3(g2))
        g4 = self.swin4(self.down4(g3))
        return [g0, g1, g2, g3, g4]


class ParallelModalSwinUNet(nn.Module):
    def __init__(
            self,
            input_channels,
            num_classes,
            base_channels=32,
            deep_supervision=True,
            swin_window_size=(3, 3, 3),
    ):
        super().__init__()
        c = int(base_channels)
        self.deep_supervision = deep_supervision
        self.modal_stem = ModalityAttentionFusion3D(input_channels, c)

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
        self.head = nn.Conv3d(c, num_classes, 1)
        self.aux1 = nn.Conv3d(c * 2, num_classes, 1)
        self.aux2 = nn.Conv3d(c * 4, num_classes, 1)
        self.aux3 = nn.Conv3d(c * 8, num_classes, 1)

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
        x = self.dec3(torch.cat([match_size(x, e3), e3], dim=1))
        out3 = self.aux3(x)
        x = self.up2(x)
        x = self.dec2(torch.cat([match_size(x, e2), e2], dim=1))
        out2 = self.aux2(x)
        x = self.up1(x)
        x = self.dec1(torch.cat([match_size(x, e1), e1], dim=1))
        out1 = self.aux1(x)
        x = self.up0(x)
        x = self.dec0(torch.cat([match_size(x, e0), e0], dim=1))
        out = self.head(x)

        if self.deep_supervision:
            return [out, out1, out2, out3]
        return out


class ParallelSwinUNetNoMAF(ParallelModalSwinUNet):
    """Ablation replacing modality-specific attention with early concatenation."""

    def __init__(
            self,
            input_channels,
            num_classes,
            base_channels=32,
            deep_supervision=True,
            swin_window_size=(3, 3, 3),
    ):
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            deep_supervision=deep_supervision,
            swin_window_size=swin_window_size,
        )
        self.modal_stem = ConvNormAct3D(input_channels, int(base_channels))


class ParallelModalUNetNoSwin(ParallelModalSwinUNet):
    """Ablation retaining MAF and the local U-Net while removing the global branch."""

    def __init__(
            self,
            input_channels,
            num_classes,
            base_channels=32,
            deep_supervision=True,
            swin_window_size=(3, 3, 3),
    ):
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            deep_supervision=deep_supervision,
            swin_window_size=swin_window_size,
        )
        self.swin_encoder = None
        self.fuse0 = None
        self.fuse1 = None
        self.fuse2 = None
        self.fuse3 = None
        self.fuse4 = None

    def forward(self, x):
        x = self.modal_stem(x)
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        x = self.up3(b)
        x = self.dec3(torch.cat([match_size(x, e3), e3], dim=1))
        out3 = self.aux3(x)
        x = self.up2(x)
        x = self.dec2(torch.cat([match_size(x, e2), e2], dim=1))
        out2 = self.aux2(x)
        x = self.up1(x)
        x = self.dec1(torch.cat([match_size(x, e1), e1], dim=1))
        out1 = self.aux1(x)
        x = self.up0(x)
        x = self.dec0(torch.cat([match_size(x, e0), e0], dim=1))
        out = self.head(x)

        if self.deep_supervision:
            return [out, out1, out2, out3]
        return out


class ParallelModalSwinUNetConcatFusion(ParallelModalSwinUNet):
    """Ablation replacing every gated fusion block with concat plus 1x1x1 conv."""

    def __init__(
            self,
            input_channels,
            num_classes,
            base_channels=32,
            deep_supervision=True,
            swin_window_size=(3, 3, 3),
    ):
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            deep_supervision=deep_supervision,
            swin_window_size=swin_window_size,
        )
        c = int(base_channels)
        self.fuse0 = ConcatFusion3D(c)
        self.fuse1 = ConcatFusion3D(c * 2)
        self.fuse2 = ConcatFusion3D(c * 4)
        self.fuse3 = ConcatFusion3D(c * 8)
        self.fuse4 = ConcatFusion3D(c * 16)


class SearchedParallelModalSwinUNet(ParallelModalSwinUNet):
    def __init__(
            self,
            input_channels,
            num_classes,
            base_channels=32,
            deep_supervision=True,
            swin_window_size=(3, 3, 3),
            search_arch=None,
    ):
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            deep_supervision=deep_supervision,
            swin_window_size=swin_window_size,
        )
        self.swin_encoder = SearchedParallelSwinEncoder3D(
            int(base_channels),
            arch_config=search_arch,
            window_size=swin_window_size,
        )


class SearchableParallelModalSwinUNet(ParallelModalSwinUNet):
    def __init__(
            self,
            input_channels,
            num_classes,
            base_channels=32,
            deep_supervision=True,
            swin_window_size=(3, 3, 3),
    ):
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            deep_supervision=deep_supervision,
            swin_window_size=swin_window_size,
        )
        self.swin_encoder = SearchableParallelSwinEncoder3D(
            int(base_channels),
            window_size=swin_window_size,
        )

    def arch_parameters(self):
        return self.swin_encoder.arch_parameters()

    def weight_parameters(self):
        arch_ids = {id(param) for param in self.arch_parameters()}
        return [param for param in self.parameters() if id(param) not in arch_ids]

    def set_search_mode(self, temp=1.0, hard=False):
        self.swin_encoder.set_search_mode(temp=temp, hard=hard)

    def export_search_architecture(self, output_path, stage_id=None):
        arch = {
            "model": {
                "in_channels": 4,
                "num_classes": self.head.out_channels,
                "genotype": "gamma_global=(h,k,alpha)",
                "searched_in": self.__class__.__name__,
            },
            "stages": self.swin_encoder.export_stages(),
            "description": "Architecture searched on the parallel modal-Swin nnU-Net global branch.",
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(arch, indent=4))
        return arch


def initialize_he(module):
    if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(module.weight, a=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.InstanceNorm3d, nn.BatchNorm3d)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
