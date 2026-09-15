import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt as distance


# --- Basic Components ---
def compute_edt_map(mask):
    """
    计算欧几里得距离变换图 (Euclidean Distance Transform)
    用于高级边界 Loss (Surface Loss)，预留接口。
    """
    if mask.sum() == 0: return np.ones_like(mask).astype(np.float32)
    if mask.sum() == mask.size: return -np.ones_like(mask).astype(np.float32)
    dist_out = distance(1 - mask)
    dist_in = distance(mask)
    return (dist_out - dist_in)


class CurvatureTopologyLoss(nn.Module):
    """
    曲率拓扑感知 Loss：
    利用 3D 拉普拉斯算子提取真实标签掩膜 (Mask) 上的高曲率角点和细小分支。
    在这些容易发生拓扑断裂的区域施加额外的 Dice 惩罚。
    """

    def __init__(self, smooth=1e-5, curvature_weight=2.0):
        super().__init__()
        self.smooth = smooth
        self.curvature_weight = curvature_weight

        # 3D Laplacian kernel 用于检测 mask 的急剧变化（高曲率边缘）
        kernel = torch.tensor([
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ], dtype=torch.float32).view(1, 1, 3, 3, 3)
        self.register_buffer('laplacian_kernel', kernel)

    def get_mask_curvature_weight(self, target_mask):
        """计算目标区域的空间曲率权重图"""
        # 计算二阶导数的绝对值近似局部曲率
        curv_map = F.conv3d(target_mask, self.laplacian_kernel, padding=1)
        curv_map = torch.abs(curv_map)

        # 归一化并放大权重
        if curv_map.max() > 0:
            curv_map = curv_map / curv_map.max()

        # 基础权重为 1，高曲率区域权重最高可达 1 + curvature_weight
        spatial_weight = 1.0 + self.curvature_weight * curv_map
        return spatial_weight

    def forward(self, pred, target):
        # pred: (B, 1, D, H, W) logits
        # target: (B, 1, D, H, W) binary mask
        probs = torch.sigmoid(pred)

        # 计算空间曲率权重
        with torch.no_grad():
            weight_map = self.get_mask_curvature_weight(target)

        probs_flat = probs.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        weight_flat = weight_map.contiguous().view(-1)

        # 加权 Dice
        intersection = (probs_flat * target_flat * weight_flat).sum()
        denominator = (probs_flat * weight_flat).sum() + (target_flat * weight_flat).sum()

        dice_loss = 1.0 - (2. * intersection + self.smooth) / (denominator + self.smooth)

        # 结合标准的 BCE 稳定训练
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, weight=weight_map)

        return dice_loss + 0.5 * bce_loss


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: logits, targets: 0/1 tensor same shape
        probs = torch.sigmoid(inputs)
        TP = (probs * targets).sum()
        FP = (probs * (1 - targets)).sum()
        FN = ((1 - probs) * targets).sum()
        score = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1.0 - score


class BoundaryLoss(nn.Module):
    """
    预留的边界 Loss 占位符。
    如果实现了基于 Distance Map 的训练，可在此扩展。
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        return torch.tensor(0.0, device=preds.device)


# [新增] 边界感知混合 Loss
class BoundaryAwareLoss(nn.Module):
    """
    Boundary/curvature weighted Tversky + BCE for one binary BraTS region.
    """

    def __init__(
            self,
            alpha=0.5,
            beta=0.5,
            bce_weight=0.5,
            boundary_weight=0.0,
            curvature_weight=0.0,
            smooth=1e-5,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_weight = bce_weight
        self.boundary_weight = boundary_weight
        self.curvature_weight = curvature_weight
        self.smooth = smooth

        kernel = torch.tensor([
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ], dtype=torch.float32).view(1, 1, 3, 3, 3)
        self.register_buffer("laplacian_kernel", kernel)

    def _target_weight_map(self, gt):
        if self.boundary_weight <= 0.0 and self.curvature_weight <= 0.0:
            return torch.ones_like(gt)

        with torch.no_grad():
            gt_bin = (gt > 0.5).float()
            weight = torch.ones_like(gt_bin)

            if self.boundary_weight > 0.0:
                dilated = F.max_pool3d(gt_bin, kernel_size=3, stride=1, padding=1)
                eroded = -F.max_pool3d(-gt_bin, kernel_size=3, stride=1, padding=1)
                boundary = (dilated - eroded).clamp_(0.0, 1.0)
                weight = weight + self.boundary_weight * boundary

            if self.curvature_weight > 0.0:
                curv = torch.abs(F.conv3d(gt_bin, self.laplacian_kernel, padding=1))
                reduce_dims = tuple(range(1, curv.dim()))
                max_per_sample = curv.amax(dim=reduce_dims, keepdim=True).clamp_min_(1e-6)
                curv = curv / max_per_sample
                weight = weight + self.curvature_weight * curv

        return weight

    def forward(self, pred, gt):
        probs = torch.sigmoid(pred)
        weight = self._target_weight_map(gt)

        tp = (probs * gt * weight).sum()
        fp = (probs * (1 - gt) * weight).sum()
        fn = ((1 - probs) * gt * weight).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        region_loss = 1.0 - tversky

        bce = F.binary_cross_entropy_with_logits(pred, gt, weight=weight)
        return region_loss + self.bce_weight * bce


class DistanceAwareTCFalsePositiveLoss(nn.Module):
    """
    Penalize tumor-core predictions that are far away from the ground-truth TC support.

    The loss is intentionally TC-specific because the paired analysis showed that
    remote TC false positives are a major source of HD95 outliers.
    """

    def __init__(self, weight=0.0, max_distance=32.0, power=1.0, min_distance=0.0, eps=1e-6):
        super().__init__()
        self.weight = float(weight)
        self.max_distance = float(max_distance)
        self.power = float(power)
        self.min_distance = float(min_distance)
        self.eps = eps

    def _distance_weights(self, gt_tc):
        maps = []
        gt_np = (gt_tc.detach().cpu().numpy() > 0.5).astype(np.uint8)
        for mask in gt_np[:, 0]:
            if mask.any():
                dist = distance(1 - mask).astype(np.float32)
            else:
                dist = np.full(mask.shape, self.max_distance, dtype=np.float32)
            if self.min_distance > 0.0:
                denom = max(self.max_distance - self.min_distance, self.eps)
                dist = (dist - self.min_distance) / denom
            else:
                dist = dist / max(self.max_distance, self.eps)
            dist = np.clip(dist, 0.0, 1.0)
            if self.power != 1.0:
                dist = np.power(dist, self.power)
            maps.append(dist)
        weight = torch.from_numpy(np.stack(maps, axis=0)).unsqueeze(1)
        return weight.to(device=gt_tc.device, dtype=gt_tc.dtype)

    def forward(self, tc_logits, gt_tc):
        if self.weight <= 0.0:
            return tc_logits.sum() * 0.0

        probs = torch.sigmoid(tc_logits)
        with torch.no_grad():
            dist_weight = self._distance_weights(gt_tc)

        # Weighted mean keeps the scale stable across patches with different TC sizes.
        loss = (probs * dist_weight).sum() / dist_weight.sum().clamp_min(self.eps)
        return self.weight * loss


# ==========================================
# [新增] 防坍塌正则化 Loss
# (放在 SearchLoss 类的前面)
# ==========================================
class SIGRegLoss3D(nn.Module):
    """
    基于方差与去相关的防坍塌正则化器 (借鉴 LeWorldModel 的 SIGReg 思想)。
    强制特征空间维持较高的方差和独立的维度特征，打破 Identity 收敛陷阱。
    """

    def __init__(self, lambda_reg=0.1):
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(self, z):
        # z: 瓶颈层特征 (B, C, D, H, W)
        B, C, D, H, W = z.shape
        z_flat = z.view(B, C, -1).permute(0, 2, 1).reshape(-1, C)

        # 中心化
        z_flat = z_flat - z_flat.mean(dim=0)

        # 1. Variance Loss (防止特征退化为单一的低频映射)
        std = torch.sqrt(z_flat.var(dim=0) + 1e-4)
        loss_var = torch.mean(F.relu(1.0 - std))

        # 2. Covariance Loss (防止不同通道提取相同的特征，迫使 NAS 使用更复杂的卷积组合)
        cov = (z_flat.T @ z_flat) / (z_flat.shape[0] - 1)
        cov_diag = torch.diagonal(cov)
        loss_cov = (cov.pow(2).sum() - cov_diag.pow(2).sum()) / C

        return loss_var + self.lambda_reg * loss_cov


# ==========================================
# 替换原本的 SearchLoss 类
# ==========================================
class SearchLoss(nn.Module):
    def __init__(self, target_region="ET", lambda_sigreg=0.1):
        super().__init__()
        self.target_region = target_region
        # 使用 Tversky Loss 处理类别不平衡
        self.dice = TverskyLoss(alpha=0.5, beta=0.5)
        self.bce = nn.BCEWithLogitsLoss()

        # [新增] 实例化防坍塌正则化器
        self.sigreg = SIGRegLoss3D(lambda_reg=0.1)
        self.lambda_sigreg = lambda_sigreg

    def forward(self, preds, target_mask):
        """
        preds: 可能是 (B, 4, D, H, W) 的 logits，
               也可能是元组: (logits, bottleneck_feat)
        target_mask: (B, D, H, W) - 0,1,2,3 labels
        """
        # [新增] 解析元组，提取 logits 和瓶颈层特征
        bottleneck_feat = None
        if isinstance(preds, tuple):
            preds, bottleneck_feat = preds  # preds 重新赋值为纯 logits，不影响后续原版代码

        # --- 以下是完全保留你原本计算主线 Loss 的逻辑 ---
        if self.target_region == "ET":
            gt_binary = (target_mask == 3).float().unsqueeze(1)
            pred_binary = preds[:, 3:4]
            main_loss = self.dice(pred_binary, gt_binary) + self.bce(pred_binary, gt_binary)

        elif self.target_region == "WT":
            gt_binary = (target_mask > 0).float().unsqueeze(1)

            # 1. 获得概率
            probs = torch.softmax(preds, dim=1)
            pred_prob_wt = torch.sum(probs[:, 1:], dim=1, keepdim=True)

            # 直接使用 pred_prob_wt 计算，杜绝 float16 精度溢出
            TP = (pred_prob_wt * gt_binary).sum()
            FP = (pred_prob_wt * (1 - gt_binary)).sum()
            FN = ((1 - pred_prob_wt) * gt_binary).sum()

            score = (TP + self.dice.smooth) / (TP + self.dice.alpha * FP + self.dice.beta * FN + self.dice.smooth)

            main_loss = 1.0 - score

        else:
            main_loss = F.cross_entropy(preds, target_mask)

        # --- [新增] 计算防坍塌正则化 Loss 并与主线 Loss 相加 ---
        if bottleneck_feat is not None:
            collapse_loss = self.sigreg(bottleneck_feat)
            main_loss = main_loss + self.lambda_sigreg * collapse_loss

        return main_loss


# --- Stage 3: Hybrid Loss (Base) ---
class HybridTaskLoss(nn.Module):
    def __init__(
            self,
            weight_boundary=0.0,
            curvature_weight=2.0,
            et_weight=1.5,
            bce_weight=0.5,
            tc_fp_weight=0.0,
            tc_fp_distance=32.0,
            tc_fp_power=1.0,
            tc_fp_min_distance=0.0,
    ):
        super().__init__()
        self.loss_main = BoundaryAwareLoss(
            alpha=0.3,
            beta=0.7,
            bce_weight=bce_weight,
            boundary_weight=weight_boundary,
            curvature_weight=curvature_weight,
        )
        self.loss_et = BoundaryAwareLoss(
            alpha=0.6,
            beta=0.4,
            bce_weight=bce_weight,
            boundary_weight=weight_boundary,
            curvature_weight=curvature_weight,
        )
        self.et_weight = et_weight
        self.tc_outlier_loss = DistanceAwareTCFalsePositiveLoss(
            weight=tc_fp_weight,
            max_distance=tc_fp_distance,
            power=tc_fp_power,
            min_distance=tc_fp_min_distance,
        )

    def forward(self, preds, target_mask, include_tc_fp=True):
        """
        preds: [B, 3, D, H, W]. Channels: 0->WT, 1->TC, 2->ET (Logits)
        target_mask: [B, D, H, W] labels 0-3
        """
        # 【修复重点：根据 target_mask 生成 gt_wt, gt_tc, gt_et】
        gt_wt = (target_mask > 0).float().unsqueeze(1)
        gt_tc = ((target_mask == 1) | (target_mask == 3)).float().unsqueeze(1)
        gt_et = (target_mask == 3).float().unsqueeze(1)

        # WT Loss
        l_wt = self.loss_main(preds[:, 0:1], gt_wt)

        # TC Loss
        l_tc = self.loss_main(preds[:, 1:2], gt_tc)

        # ET Loss (使用专门的保守 Loss 压制假阳性)
        l_et = self.loss_et(preds[:, 2:3], gt_et)

        l_tc_fp = self.tc_outlier_loss(preds[:, 1:2], gt_tc) if include_tc_fp else 0.0
        return l_wt + l_tc + self.et_weight * l_et + l_tc_fp

# --- [新增] Stage 3: Deep Supervision Wrapper ---
class HybridDeepSupervisionLoss(nn.Module):
    def __init__(
            self,
            weight_boundary=0.0,
            curvature_weight=2.0,
            et_weight=1.5,
            bce_weight=0.5,
            tc_fp_weight=0.0,
            tc_fp_distance=32.0,
            tc_fp_power=1.0,
            tc_fp_min_distance=0.0,
    ):
        super().__init__()
        self.base_loss = HybridTaskLoss(
            weight_boundary=weight_boundary,
            curvature_weight=curvature_weight,
            et_weight=et_weight,
            bce_weight=bce_weight,
            tc_fp_weight=tc_fp_weight,
            tc_fp_distance=tc_fp_distance,
            tc_fp_power=tc_fp_power,
            tc_fp_min_distance=tc_fp_min_distance,
        )
        self.ds_weights = [0.5, 0.25]

    def forward(self, preds, target):
        """
        preds: 可能是 Tensor (验证/推理时) 或 Tuple (训练时)
               Tuple 结构: (final_logits, [aux_logits_1, aux_logits_2, ...])
        target: (B, D, H, W) 原始标签
        """
        if isinstance(preds, tuple):
            final_pred, ds_preds = preds

            # 1. 计算主输出 Loss (1.0 权重)
            total_loss = self.base_loss(final_pred, target)

            # 2. 计算辅助输出 Loss
            # ds_preds 列表中的顺序应与 self.ds_weights 对应
            for i, ds_pred in enumerate(ds_preds):
                if i >= len(self.ds_weights): break

                weight = self.ds_weights[i]

                # 对 Target 进行下采样以匹配辅助输出的分辨率
                # 使用 nearest 插值保持标签的整数性质 (0,1,2,3)
                target_down = F.interpolate(
                    target.float().unsqueeze(1),
                    size=ds_pred.shape[2:],
                    mode='nearest'
                ).squeeze(1).long()

                # 计算辅助 Loss
                aux_loss = self.base_loss(ds_pred, target_down, include_tc_fp=False)
                total_loss += weight * aux_loss

            return total_loss
        else:
            # 仅有最终输出（如验证阶段或关闭 DS 时）
            return self.base_loss(preds, target)


# Legacy wrapper
class NewCombinedLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.region_loss = HybridDeepSupervisionLoss(weight_boundary=0.0, curvature_weight=0.0)

    def forward(self, preds, targets):
        logits = preds[0] if isinstance(preds, tuple) else preds
        if logits.dim() == 5 and logits.shape[1] == 3:
            return self.region_loss(preds, targets)
        return F.cross_entropy(logits, targets)
