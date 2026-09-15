
"""
Swin3D UNet NAS (EXTENDED SEARCH SPACE)
- Supports search over:
    * FFN op choices (MixedOp3D)
    * Attention variants (MixedAttention3D: different window sizes / heads)
    * Residual scaling alpha: alpha * global_block + (1-alpha) * local_identity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import PatchMerging3D, PatchExpanding3D, window_partition, window_reverse
from .nas_ops import MixedOp3D, MixedAttention3D, AlphaManager
from .swin3d_block import LayerNormChannel, DropPath

class AttentionGate3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='trilinear', align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# -------------------------
# Helper wrapper for a single block (NAS-aware)
# -------------------------
class Swin3DBlock_NAS(nn.Module):
    def __init__(
            self,
            dim,
            input_resolution,
            attn_candidates,
            ffn_op_names,
            shift_size=(0, 0, 0),
            drop_path=0.0,
            residual_alpha_candidates=(0.25, 0.5, 0.75),
    ):
        super().__init__()
        self.dim = dim
        self.D, self.H, self.W = input_resolution
        self.shift_size = shift_size
        self.norm1 = LayerNormChannel(dim)
        self.norm2 = LayerNormChannel(dim)

        # attention: either MixedAttention or standard single
        self.attn = MixedAttention3D(dim, attn_candidates)
        self.drop_path = DropPath(drop_path)
        self.ffn = MixedOp3D(dim, dim, op_names=ffn_op_names)
        self.residual_alpha = None
        self.residual_alpha_candidates = tuple(float(v) for v in residual_alpha_candidates)

        # [修改] 使用字典来缓存不同窗口大小对应的 mask
        self.attn_masks = {}

    def residual_alpha_value(self, mode='soft', temp=1.0, hard=False, device=None, dtype=None):
        device = device or self.norm1.ln.weight.device
        dtype = dtype or self.norm1.ln.weight.dtype
        candidates = torch.tensor(self.residual_alpha_candidates, device=device, dtype=dtype)
        if self.residual_alpha is None:
            return candidates.mean(), torch.ones_like(candidates) / len(candidates)

        if self.residual_alpha.numel() == 1:
            # Legacy scalar gate support: keep it bounded away from 0/1.
            alpha = 0.2 + 0.6 * torch.sigmoid(self.residual_alpha.squeeze())
            return alpha, torch.ones(1, device=device, dtype=dtype)

        logits = self.residual_alpha.unsqueeze(0)
        if mode == 'gumbel':
            weights = F.gumbel_softmax(logits, tau=temp, hard=hard, dim=1).squeeze(0)
        else:
            weights = torch.softmax(self.residual_alpha / max(temp, 1e-6), dim=0)
        return torch.sum(weights * candidates), weights

    def calculate_mask(self, x, window_size):
        D, H, W = self.D, self.H, self.W
        # 使用分辨率和窗口大小作为缓存 key
        key = f"{window_size}_{D}_{H}_{W}"

        if hasattr(self, 'attn_masks') and key in self.attn_masks:
            # 确保设备一致
            if self.attn_masks[key] is not None and self.attn_masks[key].device != x.device:
                self.attn_masks[key] = self.attn_masks[key].to(x.device)
            return self.attn_masks[key]

        wd, wh, ww = window_size
        sd, sh, sw = self.shift_size
        from .layers import compute_attn_mask
        mask = compute_attn_mask(D, H, W, (wd, wh, ww), (sd, sh, sw), x.device)

        if not hasattr(self, 'attn_masks'):
            self.attn_masks = {}
        self.attn_masks[key] = mask
        return mask

    def forward(self, x, mode='soft', temp=1.0, hard=False):
        B, C, D, H, W = x.shape
        shortcut = x
        x_norm = self.norm1(x)

        sd, sh, sw = self.shift_size
        if sd or sh or sw:
            shifted = torch.roll(x_norm, shifts=(-sd, -sh, -sw), dims=(2, 3, 4))
        else:
            shifted = x_norm

        # [修改] 为所有的候选窗口大小分别获取对应的 mask，组成字典
        masks = {}
        for cand in self.attn.candidates:
            ws = cand["window_size"]
            masks[ws] = self.calculate_mask(x_norm, ws)

        # Attention forward (将字典传入)
        if mode == 'gumbel':
            x_attn, attn_w = self.attn.forward_gumbel(shifted, mask=masks, temp=temp, hard=hard)
        else:
            x_attn = self.attn.forward_soft(shifted, mask=masks)

        if sd or sh or sw:
            x_attn = torch.roll(x_attn, shifts=(sd, sh, sw), dims=(2, 3, 4))

        # CardiacNAS-style residual scaling:
        # alpha=0.25 relies more on the local identity path, alpha=0.75
        # relies more on the global attention branch.
        alpha_val, _ = self.residual_alpha_value(
            mode=mode,
            temp=temp,
            hard=hard,
            device=x_attn.device,
            dtype=x_attn.dtype,
        )
        global_feature = shortcut + self.drop_path(x_attn)
        x = (1.0 - alpha_val) * shortcut + alpha_val * global_feature

        # FFN
        shortcut2 = x
        x2 = self.norm2(x)

        # FFN forward
        if mode == 'gumbel':
            ffn_out, ffn_w = self.ffn.forward_gumbel(x2, temp=temp, hard=hard)
        else:  # mode == 'soft'
            ffn_out = self.ffn.forward_soft(x2)

        x = shortcut2 + self.drop_path(ffn_out)
        return x

# -------------------------
# Stage builder using NAS blocks
# -------------------------
class SwinStage_NAS(nn.Module):
    def __init__(
            self,
            dim,
            resolution,
            depth,
            attn_candidates,
            ffn_op_names,
            drop_path_list,
            downsample=True,
            residual_alpha_candidates=(0.25, 0.5, 0.75),
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift = (0,0,0) if (i%2==0) else (attn_candidates[0]["window_size"][0]//2,
                                              attn_candidates[0]["window_size"][1]//2,
                                              attn_candidates[0]["window_size"][2]//2)
            blk = Swin3DBlock_NAS(
                dim,
                resolution,
                attn_candidates,
                ffn_op_names,
                shift_size=shift,
                drop_path=drop_path_list[i],
                residual_alpha_candidates=residual_alpha_candidates,
            )
            self.blocks.append(blk)
        self.downsample = PatchMerging3D(dim) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

# -------------------------
# Main model (NAS-enabled)
# -------------------------
class SwinUNet3D_NAS(nn.Module):
    def __init__(self,
                 in_channels=1,
                 num_classes=4,
                 dims=(48,96,192,384),
                 depths=(2,2,2,2),
                 window_candidates=None,
                 stage_window_candidates=None,
                 ffn_op_names=None,
                 residual_alpha_candidates=(0.25, 0.5, 0.75),
                 drop_path_rate=0.1,
                 resolution=(96,96,96),
                 nas=True):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dims = dims
        self.depths = depths
        self.resolution = resolution
        self.nas = nas
        self.residual_alpha_candidates = tuple(float(v) for v in residual_alpha_candidates)

        if window_candidates is None:
            window_candidates = [
                {"window_size": (2,6,6), "num_heads": 2},
                {"window_size": (2,6,6), "num_heads": 4},
                {"window_size": (2,6,6), "num_heads": 6},
            ]
        if stage_window_candidates is None:
            stage_window_candidates = [window_candidates for _ in depths]
        if len(stage_window_candidates) != len(depths):
            raise ValueError("stage_window_candidates must match the number of stages")
        if ffn_op_names is None:
            ffn_op_names = ["conv3","conv1","dw","identity","conv5"]

        self.patch_embed = nn.Conv3d(in_channels, dims[0], kernel_size=2, stride=2)

        alpha_groups = []
        total_blocks = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        idx = 0

        self.stages = nn.ModuleList()
        for s in range(len(depths)):
            dim = dims[s]
            depth = depths[s]
            drop_list = dpr[idx: idx+depth]; idx += depth
            attn_candidates = stage_window_candidates[s]
            res = (
                resolution[0] // (2**s),
                resolution[1] // (2**s),
                resolution[2] // (2**s)
            )
            stage = SwinStage_NAS(
                dim,
                res,
                depth,
                attn_candidates,
                ffn_op_names,
                drop_list,
                downsample=(s<3),
                residual_alpha_candidates=self.residual_alpha_candidates,
            )
            self.stages.append(stage)

            for b in range(depth):
                alpha_groups.append(len(ffn_op_names))
                alpha_groups.append(len(attn_candidates))
                alpha_groups.append(len(self.residual_alpha_candidates))

        self.up_blocks = nn.ModuleList()
        self.ag_blocks = nn.ModuleList()
        for i in range(3,0,-1):
            self.up_blocks.append(nn.Sequential(
                PatchExpanding3D(dims[i], dims[i-1]),
                nn.Conv3d(dims[i-1], dims[i-1], kernel_size=3, padding=1),
                nn.GELU()
            ))
            # [新增] 实例化 AG：上采样后的 x 作为 g, 对应深度的 skip 作为 x
            self.ag_blocks.append(AttentionGate3D(F_g=dims[i - 1], F_l=dims[i - 1], F_int=dims[i - 1] // 2))

        self.head = nn.Conv3d(dims[0], num_classes, kernel_size=1)

        if self.nas:
            self.alpha_mgr = AlphaManager(alpha_groups)
        else:
            self.alpha_mgr = None

        self._collect_blocks_for_alpha()

    def _collect_blocks_for_alpha(self):
        self._nas_blocks = []
        for stg in self.stages:
            for blk in stg.blocks:
                self._nas_blocks.append(blk)

    def set_alpha(self):
        if not self.nas: return
        alphas = self.alpha_mgr()
        ptr = 0
        for blk in self._nas_blocks:
            ffn_alpha = alphas[ptr]; ptr += 1
            attn_alpha = alphas[ptr]; ptr += 1
            scale_alpha = alphas[ptr]; ptr += 1

            blk.ffn.alpha = ffn_alpha
            blk.attn.alpha = attn_alpha
            blk.residual_alpha = scale_alpha

    def arch_parameters(self):
        if not self.nas: return []
        return list(self.alpha_mgr.parameters())

    def forward(self, x):
        return self.forward_soft(x)

    def forward_soft(self, x):
        return self._forward_shared(x, mode='soft')

    def forward_gumbel(self, x, temp=1.0, hard=False):
        return self._forward_shared(x, mode='gumbel', temp=temp, hard=hard)

    def _forward_shared(self, x, mode='soft', temp=1.0, hard=False):
        # 1. 记录原始输入分辨率 (例如 96, 96, 96)
        B, C, D, H, W = x.shape
        input_resolution = (D, H, W)

        x = self.patch_embed(x)

        # 重新计算各级分辨率分配给 Block (因为 patch_embed 降低了分辨率)
        res = [(D // 2, H // 2, W // 2)]
        for i in range(1, 4):
            res.append((res[-1][0] // 2, res[-1][1] // 2, res[-1][2] // 2))

        for stg, r in zip(self.stages, res):
            for blk in stg.blocks:
                blk.D, blk.H, blk.W = r

        if self.nas: self.set_alpha()

        features = []
        out = x
        for stage_idx, stg in enumerate(self.stages):
            for blk in stg.blocks:
                out = blk(out, mode=mode, temp=temp, hard=hard)

            # Encoder skip connections (0, 1, 2)
            if stage_idx < 3:
                features.append(out)

            if stg.downsample is not None:
                out = stg.downsample(out)

        features.append(out)  # Bottleneck

        # Decoder
        x = features[-1]
        bottleneck_feat = x  # [新增] 留存瓶颈层特征，用于后续计算 SIGReg
        for i, up in enumerate(self.up_blocks):
            x = up(x)
            skip = features[2 - i]
            # 确保 Skip Connection 尺寸匹配
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            # [修改核心] 使用 Attention Gate 对 skip 特征进行过滤
            skip = self.ag_blocks[i](g=x, x=skip)
            x = x + skip

        logits = self.head(x)

        # [修复] 最终上采样：将 48x48x48 的 Logits 恢复到原始输入 96x96x96
        if logits.shape[2:] != input_resolution:
            logits = F.interpolate(
                logits,
                size=input_resolution,
                mode="trilinear",
                align_corners=False
            )

        # [新增] 如果在训练且是 NAS 搜索阶段，同时返回瓶颈特征以计算防坍塌 Loss
        if self.training and self.nas:
            return logits, bottleneck_feat

        return logits

    def expected_flops(self, temp=1.0):
        total_flops = 0.0
        for blk in self._nas_blocks:
            # 这里的 += 会保留计算图
            total_flops = total_flops + blk.ffn.expected_flops(temp=temp)
            total_flops = total_flops + blk.attn.expected_flops(temp=temp)

        # [修改] 直接返回 total_flops 即可，它现在是一个带有梯度的 Tensor
        return total_flops

    def architecture_collapse_loss(
            self,
            temp=1.0,
            max_cheap_mass=0.45,
            min_spatial_mass=0.35,
            min_gate_value=0.60,
    ):
        """
        WT 搜索专用的架构约束：
        防止 FFN 搜索快速塌缩到 identity / 1x1 这种过轻配置。
        """
        if not self.nas:
            return torch.tensor(0.0, device=self.patch_embed.weight.device)

        stage_weights = [1.0, 1.0, 0.85, 0.65]
        penalty = None
        block_count = 0

        for stage_idx, stage in enumerate(self.stages):
            stage_weight = stage_weights[min(stage_idx, len(stage_weights) - 1)]
            for blk in stage.blocks:
                if blk.ffn.alpha is None:
                    continue

                probs = torch.softmax(blk.ffn.alpha / max(temp, 1e-6), dim=0)
                cheap_mass = probs.new_tensor(0.0)
                spatial_mass = probs.new_tensor(0.0)

                for prob, op_name in zip(probs, blk.ffn.op_names):
                    op_name = op_name.lower()
                    if "identity" in op_name:
                        cheap_mass = cheap_mass + prob
                    elif "conv1" in op_name:
                        cheap_mass = cheap_mass + 0.5 * prob

                    if ("conv3" in op_name) or ("conv5" in op_name) or ("dw" in op_name) or ("separable" in op_name):
                        spatial_mass = spatial_mass + prob

                if getattr(blk, "residual_alpha", None) is not None:
                    alpha_val, _ = blk.residual_alpha_value(temp=temp, device=probs.device, dtype=probs.dtype)
                else:
                    alpha_val = probs.new_tensor(0.5)
                block_penalty = (
                    F.relu(cheap_mass - max_cheap_mass) +
                    F.relu(min_spatial_mass - spatial_mass)
                )
                penalty = block_penalty * stage_weight if penalty is None else penalty + block_penalty * stage_weight
                block_count += 1

        if penalty is None or block_count == 0:
            return torch.tensor(0.0, device=self.patch_embed.weight.device)
        return penalty / block_count

    def forward_encoder(self, x):
        """
        仅运行编码器部分，返回 Bottleneck 特征。
        用于 Barlow Twins 预训练。
        """
        B, C, D, H, W = x.shape
        x = self.patch_embed(x)

        # 1. 重新计算每层的分辨率 (复制 _forward_shared 的逻辑)
        res = [(D // 2, H // 2, W // 2)]
        for i in range(1, 4):
            res.append((res[-1][0] // 2, res[-1][1] // 2, res[-1][2] // 2))

        # 2. 将分辨率信息赋予每个 block (处理变长输入)
        for stg, r in zip(self.stages, res):
            for blk in stg.blocks:
                blk.D, blk.H, blk.W = r

        if self.nas:
            self.set_alpha()

        # 3. 运行 Stages (编码器)
        out = x
        for stg in self.stages:
            # 直接调用 stage 的 forward (它会依次调用 blocks 和 downsample)
            # 注意：Barlow 预训练通常不涉及 Gumbel Softmax，所以默认 soft/standard forward 即可
            out = stg(out)

        # 返回最深层的特征 (Bottleneck)，不经过 Decoder
        return out
