import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


def window_partition(x: torch.Tensor, window_size: Tuple[int, int, int]):
    assert x.dim() == 5
    B, C, D, H, W = x.shape
    wd, wh, ww = window_size
    assert D % wd == 0 and H % wh == 0 and W % ww == 0
    x = x.permute(0, 2, 3, 4, 1).contiguous()
    x = x.view(B, D // wd, wd, H // wh, wh, W // ww, ww, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, wd, wh, ww, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: Tuple[int, int, int], B: int, D: int, H: int, W: int):
    wd, wh, ww = window_size
    C = windows.shape[-1]
    x = windows.view(B, D // wd, H // wh, W // ww, wd, wh, ww, C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, C)
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x


class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size=(2, 7, 7), num_heads=4, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        wd, wh, ww = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * wd - 1) * (2 * wh - 1) * (2 * ww - 1), num_heads)
        )

        coords_d = torch.arange(wd)
        coords_h = torch.arange(wh)
        coords_w = torch.arange(ww)
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += wd - 1
        relative_coords[:, :, 1] += wh - 1
        relative_coords[:, :, 2] += ww - 1
        relative_coords[:, :, 0] *= (2 * wh - 1) * (2 * ww - 1)
        relative_coords[:, :, 1] *= (2 * ww - 1)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask: torch.Tensor = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        attn = (q @ k) * self.scale

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.relative_position_index.shape[0], self.relative_position_index.shape[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)
        attn = attn + relative_position_bias

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = v.permute(0, 2, 1, 3)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


def compute_attn_mask(D: int, H: int, W: int, window_size: Tuple[int, int, int], shift_size: Tuple[int, int, int],
                      device):
    """
    [Fixed] Correct logic for SW-MSA mask generation.
    Checks intra-window connectivity.
    """
    wd, wh, ww = window_size
    sd, sh, sw = shift_size

    if sd == 0 and sh == 0 and sw == 0:
        return None

    # 1. Create a volume with slice indices
    # [Fix] Changed shape from (1, D, H, W, 1) to (1, 1, D, H, W)
    # to match window_partition expectation (B, C, D, H, W)
    img_mask = torch.zeros((1, 1, D, H, W), device=device)

    # 2. Slice based on shift/window configuration to assign regions
    d_slices = (slice(0, -wd), slice(-wd, -sd), slice(-sd, None))
    h_slices = (slice(0, -wh), slice(-wh, -sh), slice(-sh, None))
    w_slices = (slice(0, -ww), slice(-ww, -sw), slice(-sw, None))

    cnt = 0
    for d in d_slices:
        for h in h_slices:
            for w in w_slices:
                # [Fix] Updated indexing to match (B, C, D, H, W)
                img_mask[:, :, d, h, w] = cnt
                cnt += 1

    # 3. Partition into windows
    # shape: (1, 1, D, H, W) -> (num_windows, wd, wh, ww, 1)
    mask_windows = window_partition(img_mask, window_size)  # (num_windows, wd, wh, ww, 1)
    mask_windows = mask_windows.view(-1, wd * wh * ww)  # (num_windows, N)

    # 4. Generate attention mask
    # Broadcast subtraction: (nW, N, 1) - (nW, 1, N) -> (nW, N, N)
    attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(1)

    # Fill -100 where regions differ, 0 otherwise
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


class PatchMerging3D(nn.Module):
    def __init__(self, input_dim, out_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim or 2 * input_dim
        self.reduction = nn.Linear(8 * input_dim, self.out_dim, bias=False)
        self.norm = nn.LayerNorm(8 * input_dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        # Pad if needed
        pad_d = (2 - D % 2) % 2
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

        x0 = x[:, :, 0::2, 0::2, 0::2]
        x1 = x[:, :, 0::2, 0::2, 1::2]
        x2 = x[:, :, 0::2, 1::2, 0::2]
        x3 = x[:, :, 0::2, 1::2, 1::2]
        x4 = x[:, :, 1::2, 0::2, 0::2]
        x5 = x[:, :, 1::2, 0::2, 1::2]
        x6 = x[:, :, 1::2, 1::2, 0::2]
        x7 = x[:, :, 1::2, 1::2, 1::2]
        x_cat = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=1)
        B, C8, D2, H2, W2 = x_cat.shape
        x_flat = x_cat.permute(0, 2, 3, 4, 1).contiguous().view(-1, C8)
        x_norm = self.norm(x_flat)
        x_reduced = self.reduction(x_norm).view(B, D2, H2, W2, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x_reduced


class PatchExpanding3D(nn.Module):
    def __init__(self, input_dim, expand_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.expand_dim = expand_dim or input_dim // 2
        self.expand = nn.Linear(input_dim, 8 * self.expand_dim, bias=False)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_perm = x.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        x_norm = self.norm(x_perm)
        x_exp = self.expand(x_norm)
        x_exp = x_exp.view(B, D, H, W, 2, 2, 2, self.expand_dim).permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        out = x_exp.view(B, self.expand_dim, D * 2, H * 2, W * 2)
        return out


class MLP3D(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        if x.dim() == 3:
            B, N, C = x.shape
            x = x.view(-1, C)
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            x = x.view(B, N, -1)
            return x
        else:
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x