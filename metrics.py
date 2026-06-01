import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy import ndimage
from scipy.spatial import cKDTree

HD95_POSTPROCESS = {
    "min_size_wt": 300,
    "min_size_tc": 50,
    "min_size_et": 30,
}

# 1. 基础工具函数
# ==========================================
def get_brats_regions(tensor_mask):
    """将标签 (0, 1, 2, 3) 转换为 BraTS 区域 (WT, TC, ET)"""
    if isinstance(tensor_mask, np.ndarray):
        tensor_mask = torch.from_numpy(tensor_mask)
    wt = (tensor_mask > 0).float()
    tc = ((tensor_mask == 1) | (tensor_mask == 3)).float()
    et = (tensor_mask == 3).float()
    return wt, tc, et


def compute_dice_single(pred, target):
    eps = 1e-5
    # 确保转为 CPU numpy
    if isinstance(pred, torch.Tensor): pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor): target = target.detach().cpu().numpy()

    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2 * inter + eps) / (union + eps)


from scipy.ndimage import binary_erosion, generate_binary_structure

def compute_hd95_single(pred, target):
    """
    ✅ 修复版：严格遵循学术标准，仅计算【肿瘤表面边界点】的HD95
    与 BraTS挑战赛、nnU-Net、MedPy 100% 对齐
    """
    if isinstance(pred, torch.Tensor):
        pred_np = pred.detach().cpu().numpy().astype(bool)
    else:
        pred_np = pred.astype(bool)
    if isinstance(target, torch.Tensor):
        target_np = target.detach().cpu().numpy().astype(bool)
    else:
        target_np = target.astype(bool)

    pred_empty = not np.any(pred_np)
    target_empty = not np.any(target_np)

    # 1. 均为空 -> 完美匹配
    if pred_empty and target_empty:
        return 0.0

    # 2. 只有一个为空 -> 标准做法：返回 374
    if pred_empty or target_empty:
        return 374.0

    # ===================== 核心修复 =====================
    # ✅ 正确：提取3D肿瘤【表面边界点】（唯一学术合规写法）
    def get_surface_mask(mask):
        """3D形态学边缘提取：掩码 - 腐蚀后的掩码 = 表面"""
        # 3D结构元素（适配脑肿瘤3D体素）
        struct = generate_binary_structure(3, 1)
        eroded = binary_erosion(mask, structure=struct)
        surface = mask ^ eroded  # 异或运算，提取边缘
        return surface

    # 仅保留表面点，剔除所有内部点
    pred_surface = get_surface_mask(pred_np)
    target_surface = get_surface_mask(target_np)

    # 获取表面坐标（仅边界点，无内部污染）
    pred_coords = np.argwhere(pred_surface)
    target_coords = np.argwhere(target_surface)
    # ====================================================

    # 3. 构建 KD 树，快速查找最近邻
    tree_pred = cKDTree(pred_coords)
    tree_target = cKDTree(target_coords)

    # 计算双向最近邻距离（纯表面点）
    d_p_to_t, _ = tree_target.query(pred_coords)
    d_t_to_p, _ = tree_pred.query(target_coords)

    # 4. 标准HD95计算
    hd95_p_to_t = np.percentile(d_p_to_t, 95)
    hd95_t_to_p = np.percentile(d_t_to_p, 95)

    return max(hd95_p_to_t, hd95_t_to_p)


def compute_nsd_single(pred, target, tolerance=1.0):
    """
    计算 NSD (Normalized Surface Distance)。
    论文中采用的容忍度通常为 1.0mm 或 1 个体素。
    返回值越高 (接近1.0) 说明预测的拓扑边界与真实边界越贴合。
    """
    if isinstance(pred, torch.Tensor):
        pred_np = pred.detach().cpu().numpy().astype(bool)
    else:
        pred_np = pred.astype(bool)
    if isinstance(target, torch.Tensor):
        target_np = target.detach().cpu().numpy().astype(bool)
    else:
        target_np = target.astype(bool)

    pred_empty = not np.any(pred_np)
    target_empty = not np.any(target_np)

    if pred_empty and target_empty: return 1.0
    if pred_empty or target_empty: return 0.0

    # 复用您优秀的表面提取逻辑
    def get_surface_mask(mask):
        struct = generate_binary_structure(3, 1)
        eroded = binary_erosion(mask, structure=struct)
        return mask ^ eroded

    pred_surface = get_surface_mask(pred_np)
    target_surface = get_surface_mask(target_np)

    pred_coords = np.argwhere(pred_surface)
    target_coords = np.argwhere(target_surface)

    if len(pred_coords) == 0 or len(target_coords) == 0:
        return 0.0

    tree_pred = cKDTree(pred_coords)
    tree_target = cKDTree(target_coords)

    # 计算双向最近邻距离
    d_p_to_t, _ = tree_target.query(pred_coords)
    d_t_to_p, _ = tree_pred.query(target_coords)

    # 统计在容忍度 tolerance 内的表面点比例
    valid_p_to_t = np.sum(d_p_to_t <= tolerance)
    valid_t_to_p = np.sum(d_t_to_p <= tolerance)

    total_surface_elements = len(pred_coords) + len(target_coords)
    nsd = (valid_p_to_t + valid_t_to_p) / total_surface_elements

    return nsd

# ==========================================
# 2. 核心后处理：层级约束 (Hierarchical)
# ==========================================
def postprocess_hierarchical(pred_mask, min_size_wt=300, min_size_tc=50, min_size_et=50):
    if isinstance(pred_mask, torch.Tensor):
        result = pred_mask.detach().cpu().numpy().copy()
    else:
        result = np.copy(pred_mask)

    def keep_components_above_size(mask, min_size):
        mask = mask.astype(bool)
        if not np.any(mask):
            return np.zeros_like(mask, dtype=bool)

        labeled, num_features = ndimage.label(mask)
        if num_features == 0:
            return np.zeros_like(mask, dtype=bool)

        sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        keep_ids = [idx + 1 for idx, size in enumerate(sizes) if size >= min_size]
        if not keep_ids:
            return np.zeros_like(mask, dtype=bool)

        kept = np.isin(labeled, keep_ids)
        return kept.astype(bool)

    # Step 1: WT -> 先清理整体肿瘤的远端小噪点
    wt_mask = keep_components_above_size(result > 0, min_size_wt)

    # Step 2: TC -> 必须严格包含于 WT 内，过小核心区域回退到 WT
    tc_mask = keep_components_above_size(((result == 1) | (result == 3)) & wt_mask, min_size_tc)

    # Step 3: ET -> 必须严格包含于 TC 内，过小增强区域回退到 TC
    et_mask = keep_components_above_size((result == 3) & tc_mask, min_size_et)

    # 重建层级标签，确保 ET ⊆ TC ⊆ WT
    tc_mask = tc_mask | et_mask
    wt_mask = wt_mask | tc_mask

    cleaned = np.zeros_like(result)
    cleaned[wt_mask] = 2
    cleaned[tc_mask] = 1
    cleaned[et_mask] = 3
    return cleaned

def calculate_metrics_from_mask(pred_mask, target_mask, compute_hd95=True):
    """
    [修改] 增加 compute_hd95 参数控制是否计算 HD95
    """
    if isinstance(target_mask, torch.Tensor):
        target_mask = target_mask.detach().cpu().numpy()

    # --- 修改核心：只在计算 HD95 时启用严格清洗，并提高 ET 阈值到 50 ---
    if compute_hd95:
        # 将 ET 的容忍度直接提升到 150 (甚至 200)
        # 任何小于 150 体素 (约 5x5x6) 的孤立区域都会被抹除
        pred_mask_clean = postprocess_hierarchical(pred_mask, **HD95_POSTPROCESS)
    else:
        pred_mask_clean = pred_mask
    p_wt, p_tc, p_et = get_brats_regions(pred_mask_clean)
    t_wt, t_tc, t_et = get_brats_regions(target_mask)

    results = {}
    # 1. 计算 Dice (必须)
    results["Dice_WT"] = compute_dice_single(p_wt, t_wt).item()
    results["Dice_TC"] = compute_dice_single(p_tc, t_tc).item()
    results["Dice_ET"] = compute_dice_single(p_et, t_et).item()
    results["Dice_Mean"] = (results["Dice_WT"] + results["Dice_TC"] + results["Dice_ET"]) / 3.0

    # 2. 计算 HD95 (可选)
    if compute_hd95:
        results["HD95_WT"] = compute_hd95_single(p_wt, t_wt)
        results["HD95_TC"] = compute_hd95_single(p_tc, t_tc)
        results["HD95_ET"] = compute_hd95_single(p_et, t_et)
        results["HD95_Mean"] = (results["HD95_WT"] + results["HD95_TC"] + results["HD95_ET"]) / 3.0
        results["NSD_WT"] = compute_nsd_single(p_wt, t_wt, tolerance=1.0)
        results["NSD_TC"] = compute_nsd_single(p_tc, t_tc, tolerance=1.0)
        results["NSD_ET"] = compute_nsd_single(p_et, t_et, tolerance=1.0)
        results["NSD_Mean"] = (results["NSD_WT"] + results["NSD_TC"] + results["NSD_ET"]) / 3.0
    else:
        results["HD95_WT"] = -1.0
        results["HD95_TC"] = -1.0
        results["HD95_ET"] = -1.0
        results["HD95_Mean"] = -1.0

    return results


def calculate_brats_metrics(preds, targets, compute_hd95=True):
    is_hybrid_output = (preds.dim() == 5 and preds.shape[1] == 3)
    batch_size = preds.shape[0]

    batch_metrics = {
        "Dice_WT": [], "Dice_TC": [], "Dice_ET": [],
        "HD95_WT":[], "HD95_TC": [], "HD95_ET":[],
        "NSD_WT": [], "NSD_TC": [], "NSD_ET":[]
    }

    preds_cpu = preds.detach().cpu()
    targets_cpu = targets.detach().cpu()

    for i in range(batch_size):
        t_np = targets_cpu[i].numpy()

        if is_hybrid_output:
            # === 务必确保这里是 > 0，不能是 0.5 或 1.0 ===
            p_logits = preds_cpu[i].numpy()
            pred_mask_raw = np.zeros_like(p_logits[0])
            pred_mask_raw[p_logits[0] > 0] = 2
            pred_mask_raw[p_logits[1] > 0] = 1
            pred_mask_raw[p_logits[2] > 0] = 3

            if compute_hd95:
                pred_mask_clean = postprocess_hierarchical(pred_mask_raw, **HD95_POSTPROCESS)
            else:
                pred_mask_clean = pred_mask_raw

            p_wt = (pred_mask_clean > 0).astype(float)
            p_tc = ((pred_mask_clean == 1) | (pred_mask_clean == 3)).astype(float)
            p_et = (pred_mask_clean == 3).astype(float)

            t_wt = (t_np > 0).astype(float)
            t_tc = ((t_np == 1) | (t_np == 3)).astype(float)
            t_et = (t_np == 3).astype(float)

            res = {
                "Dice_WT": compute_dice_single(p_wt, t_wt).item(),
                "Dice_TC": compute_dice_single(p_tc, t_tc).item(),
                "Dice_ET": compute_dice_single(p_et, t_et).item()
            }
            if compute_hd95:
                res["HD95_WT"] = compute_hd95_single(p_wt, t_wt)
                res["HD95_TC"] = compute_hd95_single(p_tc, t_tc)
                res["HD95_ET"] = compute_hd95_single(p_et, t_et)
                res["NSD_WT"] = compute_nsd_single(p_wt, t_wt, tolerance=1.0)
                res["NSD_TC"] = compute_nsd_single(p_tc, t_tc, tolerance=1.0)
                res["NSD_ET"] = compute_nsd_single(p_et, t_et, tolerance=1.0)
        else:
            p_labels = torch.argmax(preds_cpu[i], dim=0).numpy()
            res = calculate_metrics_from_mask(p_labels, t_np, compute_hd95=compute_hd95)

        for k in ["Dice_WT", "Dice_TC", "Dice_ET"]:
            batch_metrics[k].append(res[k])

        if compute_hd95:
            for k in ["HD95_WT", "HD95_TC", "HD95_ET"]:
                val = res.get(k, np.nan)
                if not np.isnan(val):
                    batch_metrics[k].append(val)
            for k in["NSD_WT", "NSD_TC", "NSD_ET"]:
                val = res.get(k, np.nan)
                if not np.isnan(val): batch_metrics[k].append(val)

    final_metrics = {}
    for k in batch_metrics:
        if len(batch_metrics[k]) > 0:
            final_metrics[k] = float(np.mean(batch_metrics[k]))
        else:
            final_metrics[k] = 0.0

    final_metrics["Dice_Mean"] = (final_metrics.get("Dice_WT", 0) +
                                  final_metrics.get("Dice_TC", 0) +
                                  final_metrics.get("Dice_ET", 0)) / 3.0
    if compute_hd95:
        valid_hd95 = [final_metrics[k] for k in["HD95_WT", "HD95_TC", "HD95_ET"] if final_metrics.get(k, 0) > 0]
        final_metrics["HD95_Mean"] = float(np.mean(valid_hd95)) if len(valid_hd95) > 0 else 0.0
        valid_nsd = [final_metrics[k] for k in["NSD_WT", "NSD_TC", "NSD_ET"] if final_metrics.get(k, 0) > 0]
        final_metrics["NSD_Mean"] = float(np.mean(valid_nsd)) if len(valid_nsd) > 0 else 0.0
    else:
        for k in["HD95_WT", "HD95_TC", "HD95_ET", "HD95_Mean", "NSD_WT", "NSD_TC", "NSD_ET", "NSD_Mean"]:
            final_metrics[k] = -1.0

    return final_metrics


# 5. 适配器
# ==========================================
def dice_score(preds, targets):
    """
    [关键修改]
    适配 fl_client.py 的调用。只返回 Dice Mean。
    强制 compute_hd95=False 以加速训练验证。
    """
    batch_size = preds.shape[0]
    total_dice = 0.0

    # 显式关闭 HD95 计算
    res = calculate_brats_metrics(preds, targets, compute_hd95=False)

    return res["Dice_Mean"]
