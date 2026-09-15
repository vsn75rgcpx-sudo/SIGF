"""
swin3d_unet_fixed.py
--------------------
Contains:
1. Standard SwinUNet3D_Fixed (for search/baseline)
2. HybridBraTSModel (The complex model requested by user)
3. PriorNet (Curvature extractor)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import (
    PatchMerging3D,
    PatchExpanding3D,
    WindowAttention3D,
    compute_attn_mask,
    window_partition,
    window_reverse,
)
from .swin3d_block import LayerNormChannel, DropPath
from .nas_ops import build_op_3d

    # --- Basics ---


class FixedFFN3D(nn.Module):
    def __init__(self, chosen_op):
        super().__init__()
        self.op = chosen_op

    def forward(self, x): return self.op(x)


class Swin3DBlock_Fixed(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size, shift_size, ffn_op, drop_path,
                 residual_alpha=1.0):
        super().__init__()
        self.dim = dim
        self.resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = LayerNormChannel(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads)
        self.drop_path = DropPath(drop_path)
        self.norm2 = LayerNormChannel(dim)
        self.ffn = FixedFFN3D(ffn_op)
        self.residual_alpha = float(residual_alpha)

    def forward(self, x):
        B, C, D, H, W = x.shape
        shortcut = x
        x = self.norm1(x)

        if any(self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                                   dims=(2, 3, 4))
            attn_mask = compute_attn_mask(D, H, W, self.window_size, self.shift_size, x.device)
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        N = self.window_size[0] * self.window_size[1] * self.window_size[2]
        x_windows = x_windows.view(-1, N, C)

        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)

        shifted_x = window_reverse(attn_windows, self.window_size, B, D, H, W)

        if any(self.shift_size):
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=(2, 3, 4))
        else:
            x = shifted_x

        global_feature = shortcut + self.drop_path(x)
        x = (1.0 - self.residual_alpha) * shortcut + self.residual_alpha * global_feature
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class SwinStage_Fixed(nn.Module):
    def __init__(self, dim, resolution, depth, num_heads, window_size, drop_path_list, ffn_ops,
                 downsample=True, block_attn_settings=None, residual_alphas=None):
        super().__init__()
        self.blocks = nn.ModuleList()
        residual_alphas = residual_alphas or [1.0] * depth
        for i in range(depth):
            if block_attn_settings is not None:
                attn_cfg = block_attn_settings[min(i, len(block_attn_settings) - 1)]
                block_window_size = tuple(attn_cfg["window_size"])
                block_num_heads = int(attn_cfg["num_heads"])
            else:
                block_window_size = window_size
                block_num_heads = num_heads

            shift = (0, 0, 0) if i % 2 == 0 else (
                block_window_size[0] // 2, block_window_size[1] // 2, block_window_size[2] // 2
            )
            self.blocks.append(
                Swin3DBlock_Fixed(
                    dim, resolution, block_num_heads, block_window_size, shift, ffn_ops[i], drop_path_list[i],
                    residual_alpha=residual_alphas[min(i, len(residual_alphas) - 1)]
                )
            )
        self.downsample = PatchMerging3D(dim) if downsample else None

    def forward(self, x):
        for blk in self.blocks: x = blk(x)
        if self.downsample: x = self.downsample(x)
        return x


class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=2):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x): return self.proj(x)


# --- Modular Encoder/Decoder ---
class Swin3DEncoder(nn.Module):
    def __init__(self, in_channels, dims, depths, arch_config, window_size=(2, 6, 6)):
        super().__init__()
        # [确认] 这里正确使用了传入的 in_channels
        self.patch_embed = PatchEmbed3D(in_channels, dims[0])

        # Parse Arch
        ops = [["conv3"] * d for d in depths]
        default_attn = []
        for stage_idx, dim in enumerate(dims):
            if stage_idx == len(dims) - 1:
                default_attn.append({"window_size": (6, 6, 6), "num_heads": 6})
            else:
                default_attn.append({"window_size": window_size, "num_heads": max(1, dim // 32)})
        attn_cfgs = [[default_attn[i].copy() for _ in range(depths[i])] for i in range(len(depths))]
        residual_alpha_cfgs = [[1.0 for _ in range(depths[i])] for i in range(len(depths))]
        parsed_from_arch = False

        if isinstance(arch_config, str):
            import json
            with open(arch_config, 'r') as f: arch_config = json.load(f)

        if isinstance(arch_config, dict) and "stages" in arch_config:
            parsed_from_arch = True
            sorted_stages = sorted(arch_config["stages"], key=lambda x: x["stage_id"])
            ops = []
            parsed_attn = []
            parsed_alphas = []
            for stage_idx, stage in enumerate(sorted_stages):
                blocks = stage.get("blocks", [])
                ops.append([blk.get("ffn_choice", "conv3") for blk in blocks])
                parsed_alphas.append([float(blk.get("residual_alpha", 1.0)) for blk in blocks])

                stage_attn = []
                for blk in blocks:
                    attn_choice = blk.get("attn_choice", {})
                    stage_attn.append({
                        "window_size": tuple(attn_choice.get("window_size", default_attn[stage_idx]["window_size"])),
                        "num_heads": int(attn_choice.get("num_heads", default_attn[stage_idx]["num_heads"]))
                    })
                parsed_attn.append(stage_attn or [default_attn[stage_idx].copy() for _ in range(depths[stage_idx])])

            while len(ops) < len(depths):
                stage_idx = len(ops)
                ops.append(["conv3"] * depths[stage_idx])
                parsed_attn.append([default_attn[stage_idx].copy() for _ in range(depths[stage_idx])])
                parsed_alphas.append([1.0 for _ in range(depths[stage_idx])])

            attn_cfgs = parsed_attn
            residual_alpha_cfgs = parsed_alphas

        # 没有搜索架构时，bottleneck 默认使用全局窗口；有搜索 JSON 时保留
        # gamma_global 中导出的 h/k/alpha。
        if not parsed_from_arch:
            attn_cfgs[-1] = [{"window_size": (6, 6, 6), "num_heads": 6} for _ in range(depths[-1])]

        self.stages = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]
        ptr = 0
        for i in range(4):
            stage_ops = [build_op_3d(name, dims[i], dims[i]) for name in ops[i]]
            self.stages.append(SwinStage_Fixed(
                dims[i], (1, 1, 1), depths[i], max(1, dims[i] // 32), window_size, dpr[ptr:ptr + depths[i]], stage_ops,
                downsample=(i < 3), block_attn_settings=attn_cfgs[i],
                residual_alphas=residual_alpha_cfgs[i]
            ))
            ptr += depths[i]

    def forward(self, x):
        x = self.patch_embed(x)
        feats = []
        for i, stage in enumerate(self.stages):
            for blk in stage.blocks: x = blk(x)
            if i < 3: feats.append(x)  # Pre-downsample
            if stage.downsample: x = stage.downsample(x)
        feats.append(x)
        return feats


class Swin3DDecoder(nn.Module):
    def __init__(self, dims, num_classes, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.up_blocks = nn.ModuleList()

        # 构建上采样模块
        # i=0: dims[3]->dims[2] (384->192)
        # i=1: dims[2]->dims[1] (192->96)
        # i=2: dims[1]->dims[0] (96->48)
        for i in range(3, 0, -1):
            self.up_blocks.append(nn.Sequential(
                PatchExpanding3D(dims[i], dims[i - 1]),
                nn.Conv3d(dims[i - 1], dims[i - 1], 3, padding=1),
                nn.GELU()
            ))

        self.head = nn.Conv3d(dims[0], num_classes, 1)

        # [新增] 深层监督的辅助头 (Auxiliary Heads)
        # 在 1/4 和 1/8 分辨率处输出
        if self.deep_supervision:
            self.aux_head_1 = nn.Conv3d(dims[1], num_classes, 1)  # 1/4 res (e.g. 96 ch)
            self.aux_head_2 = nn.Conv3d(dims[2], num_classes, 1)  # 1/8 res (e.g. 192 ch)

    def forward(self, feats, aux_feats=None):
        # feats: [f0(1/4), f1(1/8), f2(1/16), f3(1/32)]
        x = feats[-1]

        # Deep Supervision 的输出列表
        deep_outputs = []

        if aux_feats: x = x + aux_feats[-1]

        # i=0: 1/32 -> 1/16 (384 -> 192 ch)
        # i=1: 1/16 -> 1/8  (192 -> 96 ch)
        # i=2: 1/8  -> 1/4  (96 -> 48 ch)

        for i, up in enumerate(self.up_blocks):
            x = up(x)
            skip = feats[2 - i]

            if aux_feats:
                aux_skip = aux_feats[2 - i]
                if aux_skip.shape[2:] != skip.shape[2:]:
                    aux_skip = F.interpolate(aux_skip, size=skip.shape[2:], mode='nearest')
                skip = skip + aux_skip

            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear')
            x = x + skip

            # [修复] 收集深层监督输出
            if self.deep_supervision:
                if i == 0:  # current res is 1/8 (dims[2] = 192 ch)
                    # 此时 x 的通道数是 192，匹配 aux_head_2
                    deep_outputs.append(self.aux_head_2(x))
                elif i == 1:  # current res is 1/4 (dims[1] = 96 ch)
                    # 此时 x 的通道数是 96，匹配 aux_head_1
                    deep_outputs.append(self.aux_head_1(x))

        # 最终输出 (1/4 分辨率 -> Logits)
        final_logits = self.head(x)

        if self.deep_supervision and self.training:
            # 返回: 最终预测, [1/4预测, 1/8预测]
            # 上面的 append 顺序是先 aux_head_2(深), 后 aux_head_1(浅)
            # 列表为 [1/8 res output, 1/4 res output]
            # 我们反转一下，使其对应 [aux_head1(浅), aux_head2(深)] 的权重顺序 [0.5, 0.25]
            return final_logits, deep_outputs[::-1]

        return final_logits


class SpatialModalityAttention(nn.Module):
    """
    体素级空间多模态注意力机制 (Voxel-wise Spatial-Modality Attention)
    自动评估每一个 3D 空间位置上，不同模态特征的重要性。
    """

    def __init__(self, num_streams, channels):
        super().__init__()
        # 通过卷积层对拼接后的多模态特征进行降维和评估
        self.conv = nn.Sequential(
            nn.Conv3d(num_streams * channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(channels, num_streams, kernel_size=1),  # 输出每个模态对应的权重通道
            nn.Sigmoid()  # 压缩到 0~1 之间
        )

    def forward(self, streams_list):
        # streams_list: 包含各个特征流的列表，每个特征形状 [B, C, D, H, W]
        concat_features = torch.cat(streams_list, dim=1)  # [B, num_streams * C, D, H, W]

        # 生成体素级注意力权重图
        attention_maps = self.conv(concat_features)  # [B, num_streams, D, H, W]

        # 跨模态归一化 (确保在同一个体素上，所有模态的权重和为 1)
        attention_maps = attention_maps / (attention_maps.sum(dim=1, keepdim=True) + 1e-8)

        # 乘以权重并相加融合
        out = 0
        for i in range(len(streams_list)):
            weight = attention_maps[:, i:i + 1, :, :, :]  # 取出第 i 个模态的 3D 权重图
            out = out + streams_list[i] * weight

        return out


class SwinUNet3D_Fixed(nn.Module):
    def __init__(self, in_channels, num_classes, dims, depths, arch_json, window_size):
        super().__init__()
        self.encoder = Swin3DEncoder(in_channels, dims, depths, arch_json, window_size)
        self.decoder = Swin3DDecoder(dims, num_classes)

    def forward(self, x):
        feats = self.encoder(x)
        return self.decoder(feats)


# --- Components for Hybrid Model ---

class CurvatureExtractor(nn.Module):
    """
    Calculates 'True' Mean Curvature of the 3D MRI implicit surface.
    Formula: Divergence of the normalized gradient field.
    Includes:
    1. Gaussian Smoothing (to reduce high-freq noise in tumor/necrotic regions).
    2. Hessian & Gradient Calculation via Central Differences.
    3. Skull-stripping Edge Artifact Removal (Masking).
    4. Robust Instance Normalization.
    """

    def __init__(self, sigma=1.0):
        super().__init__()
        # 1. Gaussian Kernel for smoothing (Derivatives are sensitive to noise!)
        # Create a small 3D Gaussian kernel
        self.sigma = sigma
        kernel_size = int(2 * 4 * sigma + 1)
        if kernel_size % 2 == 0: kernel_size += 1
        self.padding = kernel_size // 2

        # Pre-compute Gaussian kernel
        k = self._get_gaussian_kernel3d(kernel_size, sigma)
        self.register_buffer('gaussian_kernel', k)

        # 2. Derivative Kernels (Central Difference)
        # dx: [-0.5, 0, 0.5]
        self.register_buffer('k_d', torch.tensor([[[0.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 0.0]],
                                                  [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                                                  [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]]]).unsqueeze(
            0).unsqueeze(0))
        # Note: Orientation depends on tensor dim (D,H,W), handled in forward

    def _get_gaussian_kernel3d(self, kernel_size, sigma):
        # Generate 1D kernel
        coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        k1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        k1d = k1d / k1d.sum()
        # Outer product to 3D
        k3d = k1d[:, None, None] * k1d[None, :, None] * k1d[None, None, :]
        return k3d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K, K)

    def get_derivatives(self, x):
        # Assumes x is (B, 1, D, H, W)
        # Use standard conv3d to compute Ix, Iy, Iz
        # Permute weights for different axes

        # kz: (1,1,3,1,1), ky: (1,1,1,3,1), kx: (1,1,1,1,3)
        # Central difference: [-0.5, 0, 0.5]
        k = torch.tensor([-0.5, 0.0, 0.5], device=x.device).view(1, 1, 3, 1, 1)

        Iz = F.conv3d(x, k, padding=(1, 0, 0))
        Iy = F.conv3d(x, k.transpose(2, 3), padding=(0, 1, 0))
        Ix = F.conv3d(x, k.transpose(2, 4), padding=(0, 0, 1))

        return Ix, Iy, Iz

    def forward(self, x):
        # x: (B, 1, D, H, W) - Input T1Gd
        EPS = 1e-6

        # --- [修复 Start] ---
        # 原代码: mask = (x != 0).float()
        # 问题: Z-score 后背景不是 0，而是负的最小值。

        # 计算每个样本的最小值 (Batch 维度)
        # x.view(B, -1) -> (B, N) -> min(dim=1) -> values
        min_vals = x.reshape(x.size(0), -1).min(dim=1)[0]
        # 恢复形状以便广播: (B, 1, 1, 1, 1)
        min_vals = min_vals.view(-1, 1, 1, 1, 1)

        # 设定阈值: 比最小值大一点点 (e.g. +0.1) 认为是前景
        mask = (x > (min_vals + 0.1)).float()
        # --- [修复 End] ---

        # --- Step 1: Smoothing ---
        x_smooth = F.conv3d(x, self.gaussian_kernel, padding=self.padding, groups=1)

        # --- Step 2: First Order Derivatives ---
        Ix, Iy, Iz = self.get_derivatives(x_smooth)

        # --- Step 3: Second Order Derivatives ---
        Ixx, Ixy, Ixz = self.get_derivatives(Ix)
        Iyx, Iyy, Iyz = self.get_derivatives(Iy)
        Izx, Izy, Izz = self.get_derivatives(Iz)

        # --- Step 4: Mean Curvature ---
        num = (Iy ** 2 + Iz ** 2) * Ixx + \
              (Ix ** 2 + Iz ** 2) * Iyy + \
              (Ix ** 2 + Iy ** 2) * Izz - \
              2 * (Ix * Iy * Ixy + Ix * Iz * Ixz + Iy * Iz * Iyz)

        denom = 2 * torch.pow(Ix ** 2 + Iy ** 2 + Iz ** 2, 1.5) + EPS
        H = num / denom

        # --- Step 5: Handle Skull-stripping Edge Effect ---
        H = H * mask

        # Erosion to remove boundary artifacts
        mask_eroded = (F.avg_pool3d(mask, kernel_size=3, stride=1, padding=1) > 0.99).float()
        H = H * mask_eroded

        # --- Step 6: Robust Normalization ---
        B = H.shape[0]
        H_norm_list = []
        for b in range(B):
            h_vol = H[b, 0]
            m_vol = mask_eroded[b, 0]

            valid_vals = h_vol[m_vol > 0]
            if valid_vals.numel() > 0:
                mean = valid_vals.mean()
                std = valid_vals.std() + 1e-8
                h_norm = (h_vol - mean) / std
                h_norm = torch.clamp(h_norm, -3.0, 3.0)
                h_norm = h_norm * m_vol
            else:
                h_norm = h_vol

            H_norm_list.append(h_norm.unsqueeze(0))

        H_final = torch.stack(H_norm_list, dim=0)
        return H_final


class PriorNet(nn.Module):
    """
    [修改] 多尺度 PriorNet
    同时提取多尺度曲率特征 (Multi-scale Curvature Features)，增强几何感知能力。
    """
    def __init__(self, base_filters=16, out_channels_list=(48, 96, 192, 384)):
        super().__init__()
        self.curv_s = CurvatureExtractor(sigma=0.5)
        self.curv_m = CurvatureExtractor(sigma=1.0)
        self.curv_l = CurvatureExtractor(sigma=2.0)

        self.stem = nn.Sequential(
            nn.Conv3d(3, base_filters, 3, padding=1),
            nn.InstanceNorm3d(base_filters),
            nn.LeakyReLU(inplace=True)
        )

        self.layers = nn.ModuleList()
        c_in = base_filters
        for c_out in out_channels_list:
            self.layers.append(nn.Sequential(
                nn.Conv3d(c_in, c_out, 3, stride=2, padding=1),
                nn.InstanceNorm3d(c_out),
                nn.LeakyReLU(inplace=True)
            ))
            c_in = c_out

    # [新增] 专门用于输出物理曲率图的接口，供 SHAP 切块使用
    def get_curvature_map(self, t1gd):
        c1 = self.curv_s(t1gd)
        c2 = self.curv_m(t1gd)
        c3 = self.curv_l(t1gd)
        return torch.cat([c1, c2, c3], dim=1) # 输出 shape: (B, 3, D, H, W)

    def forward(self, t1gd):
        x = self.get_curvature_map(t1gd)
        x = self.stem(x)
        feats = []
        for l in self.layers:
            x = l(x)
            feats.append(x)
        return feats


class ECA3D(nn.Module):
    """
    Efficient Channel Attention for 3D feature maps.
    输入: [B, C, D, H, W]
    输出: [B, C, D, H, W]
    """
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(
            1, 1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).squeeze(-1).transpose(1, 2)
        y = self.conv(y)
        y = self.sigmoid(y).transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        return x * y


class LightConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
        )
        self.shortcut = nn.Identity()
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_ch),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class LightModalityEncoder3D(nn.Module):
    """
    轻量模态细节编码器。
    输出前三个尺度，用于补充 Uni-Encoder 的 skip/detail。
    """
    def __init__(self, dims):
        super().__init__()
        self.stage0 = LightConvBlock3D(1, dims[0], stride=2)
        self.stage1 = LightConvBlock3D(dims[0], dims[1], stride=2)
        self.stage2 = LightConvBlock3D(dims[1], dims[2], stride=2)

    def forward(self, x):
        d0 = self.stage0(x)
        d1 = self.stage1(d0)
        d2 = self.stage2(d1)
        return [d0, d1, d2]


class DetailFusionBlock3D(nn.Module):
    """
    融合四个轻量模态细节特征，并输出与 Uni-Encoder 对齐的通道数。
    """
    def __init__(self, channels, num_modalities=4):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv3d(channels * num_modalities, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.GELU(),
        )
        self.eca = ECA3D(channels)

    def forward(self, feats):
        x = torch.cat(feats, dim=1)
        x = self.fuse(x)
        x = self.eca(x)
        return x


class GlobalDetailFusion3D(nn.Module):
    """
    把 Uni-Encoder 的 global feature 和轻量 detail feature 融合。
    用 residual gate，避免 detail 分支压过主语义。
    """
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.refine = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.GELU()
        )

    def forward(self, global_feat, detail_feat):
        if detail_feat.shape[2:] != global_feat.shape[2:]:
            detail_feat = F.interpolate(
                detail_feat,
                size=global_feat.shape[2:],
                mode='trilinear',
                align_corners=False
            )
        gate = self.gate(torch.cat([global_feat, detail_feat], dim=1))
        out = global_feat + gate * detail_feat
        return self.refine(out)


class HybridBraTSModel(nn.Module):
    def __init__(self, config_a, config_b, config_uni=None, dims=(48, 96, 192, 384), depths=(2, 2, 2, 2),
                 deep_supervision=True, use_curvature=True, encoder_type='multi', fusion_type='weighted'):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.use_curvature = use_curvature
        self.encoder_type = encoder_type
        self.fusion_type = fusion_type

        # 1. Encoders
        if self.encoder_type == 'single':
            self.enc_single = Swin3DEncoder(4, dims, depths, config_a)
        elif self.encoder_type == 'uni_light':
            # 一个共享 Uni-Encoder 看完整 4 模态，负责全局语义和跨模态关系。
            uni_config = config_uni if config_uni is not None else config_a
            self.enc_uni = Swin3DEncoder(
                in_channels=4,
                dims=dims,
                depths=depths,
                arch_config=uni_config,
                window_size=(2, 6, 6)
            )
            # 四个轻量 CNN 分支只补前三个尺度的局部细节。
            self.detail_encoders = nn.ModuleList([
                LightModalityEncoder3D(dims) for _ in range(4)
            ])
            self.detail_fusers = nn.ModuleList([
                DetailFusionBlock3D(dims[i], num_modalities=4)
                for i in range(3)
            ])
            self.global_detail_fusers = nn.ModuleList([
                GlobalDetailFusion3D(dims[i])
                for i in range(3)
            ])
        else:
            self.enc_a = Swin3DEncoder(1, dims, depths, config_a)  # T1c
            self.enc_b_t1 = Swin3DEncoder(1, dims, depths, config_b)  # T1n
            self.enc_b_t2w = Swin3DEncoder(1, dims, depths, config_b)  # T2w
            self.enc_b_t2f = Swin3DEncoder(1, dims, depths, config_b)  # Flair

        # 2. PriorNet
        if self.use_curvature:
            self.prior_net = PriorNet(out_channels_list=dims)
        else:
            self.prior_net = None

        if self.encoder_type in ['single', 'uni_light']:
            self.num_streams = 1 + int(self.use_curvature)
        else:
            self.num_streams = 4 + int(self.use_curvature)

        # 3. Decoders
        self.dec_wt = Swin3DDecoder(dims, 1, deep_supervision)
        self.dec_tc = Swin3DDecoder(dims, 1, deep_supervision)
        self.dec_et = Swin3DDecoder(dims, 1, deep_supervision)

        # 4. Fusion
        if self.fusion_type == 'weighted':
            self.stream_weights_wt = nn.Parameter(torch.ones(self.num_streams))
            self.stream_weights_tc = nn.Parameter(torch.ones(self.num_streams))
            self.stream_weights_et = nn.Parameter(torch.ones(self.num_streams))
        elif self.fusion_type == 'attention':
            # [新增] 针对 WT, TC, ET 分别注册体素级空间注意力模块
            self.attention_wt = nn.ModuleList([SpatialModalityAttention(self.num_streams, d) for d in dims])
            self.attention_tc = nn.ModuleList([SpatialModalityAttention(self.num_streams, d) for d in dims])
            self.attention_et = nn.ModuleList([SpatialModalityAttention(self.num_streams, d) for d in dims])
        elif self.fusion_type == 'concat':
            self.concat_convs_wt = nn.ModuleList([nn.Conv3d(d * self.num_streams, d, 1) for d in dims])
            self.concat_convs_tc = nn.ModuleList([nn.Conv3d(d * self.num_streams, d, 1) for d in dims])
            self.concat_convs_et = nn.ModuleList([nn.Conv3d(d * self.num_streams, d, 1) for d in dims])

        if self.encoder_type == 'multi':
            # 为 4 个阶段分别准备跨模态注意力
            self.cross_attns = nn.ModuleList([CrossModalityAttention(dims[i], 4) for i in range(4)])

    # [修改] 前向传播增加 return_curvature 参数
    def forward(self, x, return_curvature=False):
        if x.shape[1] != 4:
            raise ValueError(f"HybridBraTSModel expects 4-channel input (B, 4, D, H, W), got {x.shape}")

        x_t1, x_t1c, x_t2, x_flair = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]

        # 统一使用 streams 列表来收集所有分支的特征
        streams = []

        if self.encoder_type == 'single':
            streams.append(self.enc_single(x))
        elif self.encoder_type == 'uni_light':
            # 1. Uni-Encoder 看完整 4 模态，负责全局语义和跨模态关系。
            global_feats = self.enc_uni(x)

            # 2. 轻量细节分支分别看单模态，顺序保持原 multi 习惯: T1c, T1, T2, Flair。
            modal_inputs = [x_t1c, x_t1, x_t2, x_flair]
            modal_detail_feats = [
                enc(m) for enc, m in zip(self.detail_encoders, modal_inputs)
            ]

            # 3. 前三个尺度融合四模态细节，并以 residual gate 注入 Uni-Encoder。
            fused_feats = list(global_feats)
            for level in range(3):
                detail_level_feats = [
                    modal_detail_feats[m][level] for m in range(4)
                ]
                detail_fused = self.detail_fusers[level](detail_level_feats)
                fused_feats[level] = self.global_detail_fusers[level](
                    global_feats[level],
                    detail_fused
                )

            # 4. bottleneck 保持 Uni-Encoder 原样，不引入轻量分支。
            streams.append(fused_feats)
        else:
            # --- 核心重构：多编码器同步步进 ---
            fa = self.enc_a.patch_embed(x_t1c)
            fb1 = self.enc_b_t1.patch_embed(x_t1)
            fb2 = self.enc_b_t2w.patch_embed(x_t2)
            fb3 = self.enc_b_t2f.patch_embed(x_flair)

            streams_a, streams_b1, streams_b2, streams_b3 = [], [], [], []

            cur_feats = [fa, fb1, fb2, fb3]
            for i in range(4):
                # 1. 先只运行 Transformer Blocks (此时尚未下采样，通道数对齐)
                for blk in self.enc_a.stages[i].blocks: cur_feats[0] = blk(cur_feats[0])
                for blk in self.enc_b_t1.stages[i].blocks: cur_feats[1] = blk(cur_feats[1])
                for blk in self.enc_b_t2w.stages[i].blocks: cur_feats[2] = blk(cur_feats[2])
                for blk in self.enc_b_t2f.stages[i].blocks: cur_feats[3] = blk(cur_feats[3])

                # 2. 在下采样之前进行跨模态注意力融合 (与 cross_attns 的初始化维度完美契合)
                cur_feats = self.cross_attns[i](cur_feats)

                # 3. 收集 pre-downsample 特征，用于传给 Decoder 的 Skip Connection
                if i < 3:
                    streams_a.append(cur_feats[0])
                    streams_b1.append(cur_feats[1])
                    streams_b2.append(cur_feats[2])
                    streams_b3.append(cur_feats[3])

                # 4. 最后手动执行下采样 (进入下一个 Stage 的通道尺寸)
                if self.enc_a.stages[i].downsample:
                    cur_feats[0] = self.enc_a.stages[i].downsample(cur_feats[0])
                    cur_feats[1] = self.enc_b_t1.stages[i].downsample(cur_feats[1])
                    cur_feats[2] = self.enc_b_t2w.stages[i].downsample(cur_feats[2])
                    cur_feats[3] = self.enc_b_t2f.stages[i].downsample(cur_feats[3])

            # 收集最后一层特征
            streams_a.append(cur_feats[0])
            streams_b1.append(cur_feats[1])
            streams_b2.append(cur_feats[2])
            streams_b3.append(cur_feats[3])

            # 统合所有编码器流
            streams = [streams_a, streams_b1, streams_b2, streams_b3]

        # [修复]: 移除混乱的 all_streams_feats 变量，直接 append 到 streams 中
        raw_curvature_map = None
        if self.use_curvature:
            if return_curvature:
                raw_curvature_map = self.prior_net.get_curvature_map(x_t1c)
            streams.append(self.prior_net(x_t1c))

        num_scales = len(streams[0])

        def fuse_features(fusion_method, weights=None, convs=None, attn_modules=None):
            fused = []
            for scale_idx in range(num_scales):
                # 提前把当前尺度的多模态特征打包成 list
                current_streams = [streams[i][scale_idx] for i in range(self.num_streams)]

                if fusion_method == 'weighted':
                    w = torch.softmax(weights, dim=0)
                    scale_fused = sum(w[i] * current_streams[i] for i in range(self.num_streams))
                elif fusion_method == 'add':
                    scale_fused = sum(current_streams)
                elif fusion_method == 'attention':
                    # [新增] 调用我们写好的空间注意力机制
                    scale_fused = attn_modules[scale_idx](current_streams)
                elif fusion_method == 'concat':
                    concated = torch.cat(current_streams, dim=1)
                    scale_fused = convs[scale_idx](concated)
                fused.append(scale_fused)
            return fused

        # 生成解码器所需特征，传入对应的 attention_modules
        fused_wt = fuse_features(self.fusion_type,
                                 weights=self.stream_weights_wt if self.fusion_type == 'weighted' else None,
                                 convs=self.concat_convs_wt if self.fusion_type == 'concat' else None,
                                 attn_modules=self.attention_wt if self.fusion_type == 'attention' else None)

        fused_tc = fuse_features(self.fusion_type,
                                 weights=self.stream_weights_tc if self.fusion_type == 'weighted' else None,
                                 convs=self.concat_convs_tc if self.fusion_type == 'concat' else None,
                                 attn_modules=self.attention_tc if self.fusion_type == 'attention' else None)

        fused_et = fuse_features(self.fusion_type,
                                 weights=self.stream_weights_et if self.fusion_type == 'weighted' else None,
                                 convs=self.concat_convs_et if self.fusion_type == 'concat' else None,
                                 attn_modules=self.attention_et if self.fusion_type == 'attention' else None)

        def decode_wrapper(decoder, main_f):
            out = decoder(feats=main_f, aux_feats=None)
            if self.deep_supervision and self.training: return out[0], out[1]
            return out, []

        wt_final, wt_deep = decode_wrapper(self.dec_wt, fused_wt)
        tc_final, tc_deep = decode_wrapper(self.dec_tc, fused_tc)
        et_final, et_deep = decode_wrapper(self.dec_et, fused_et)

        target_size = x.shape[2:]

        def up(y):
            if y.shape[2:] != target_size: return F.interpolate(y, size=target_size, mode='trilinear')
            return y

        final_out = torch.cat([up(wt_final), up(tc_final), up(et_final)], dim=1)

        if self.deep_supervision and self.training:
            ds_level1 = torch.cat([wt_deep[0], tc_deep[0], et_deep[0]], dim=1)
            ds_level2 = torch.cat([wt_deep[1], tc_deep[1], et_deep[1]], dim=1)
            if return_curvature: return final_out, [ds_level1, ds_level2], raw_curvature_map
            return final_out, [ds_level1, ds_level2]

        if return_curvature:
            return final_out, raw_curvature_map
        return final_out


# 1. 新增跨模态注意力模块
class CrossModalityAttention(nn.Module):
    def __init__(self, dim, num_modalities=4):
        super().__init__()
        self.num_modalities = num_modalities
        # 模态权重预测：通过全局池化 + 2层全连接
        self.attention_net = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(dim * num_modalities, dim * num_modalities, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim * num_modalities, num_modalities, 1),
            nn.Sigmoid()
        )

    def forward(self, feats):
        # feats 是一个 list: [f1, f2, f3, f4]，每个 shape (B, C, D, H, W)
        concat_feats = torch.cat(feats, dim=1)
        # 计算每个模态的注意力系数 (B, 4, 1, 1, 1)
        weights = self.attention_net(concat_feats)

        out_feats = []
        for i in range(self.num_modalities):
            # 将注意力分配给每个分支
            w = weights[:, i:i + 1, ...]
            out_feats.append(feats[i] * w + feats[i])  # 残差连接
        return out_feats
