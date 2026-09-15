import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# [修改] 引入新的 Loss 类
from losses import (
    HybridTaskLoss,
    SearchLoss,
    NewCombinedLoss,
    HybridDeepSupervisionLoss  # 确保在 losses.py 中已定义
)
import metrics

# 尝试导入 AMP
try:
    from torch.amp import autocast as _autocast
    from torch.amp import GradScaler as _GradScaler
    AMP_AVAILABLE = True

    def autocast(device_type='cuda', enabled=True):
        return _autocast(device_type, enabled=enabled)

    def make_grad_scaler(enabled=True):
        return _GradScaler('cuda', enabled=enabled)
except ImportError:
    try:
        from torch.cuda.amp import autocast as _autocast
        from torch.cuda.amp import GradScaler as _GradScaler
        AMP_AVAILABLE = True

        def autocast(device_type='cuda', enabled=True):
            return _autocast(enabled=enabled)

        def make_grad_scaler(enabled=True):
            return _GradScaler(enabled=enabled)
    except ImportError:
        AMP_AVAILABLE = False

        def autocast(device_type='cuda', enabled=True):
            from contextlib import nullcontext
            return nullcontext()

        def make_grad_scaler(enabled=True):
            return None


class FederatedClient:
    def __init__(self, cid, model_fn, dataset, device="cuda",
                 batch_size=8, epochs=1, lr=1e-3, weight_decay=1e-4,
                 lr_alpha=0.0, val_split_ratio=0.1, num_workers=0,
                 grad_clip=1.0, use_amp=None, **kwargs):
        self.cid = cid
        self.device = device
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_alpha = lr_alpha
        self.num_workers = num_workers
        self.grad_clip = grad_clip

        # 初始化模型
        self.model = model_fn().to(device)

        # [新增] 检查模型是否开启了 Deep Supervision 属性，便于日志记录
        self.deep_supervision = getattr(self.model, 'deep_supervision', False)

        self.use_amp = (device == "cuda" and AMP_AVAILABLE) if use_amp is None else bool(use_amp)
        self.scaler = make_grad_scaler(enabled=self.use_amp) if self.use_amp else None

        # Split Dataset
        total_len = len(dataset)
        val_len = int(total_len * val_split_ratio)
        train_len = total_len - val_len

        if hasattr(dataset, 'test_mode') and dataset.test_mode:
            self.train_ds = None
            self.val_ds = dataset
            self.val_loader = DataLoader(self.val_ds, batch_size=1, shuffle=False, num_workers=self.num_workers)
            self.train_loader = None
        else:
            if val_len < 1:
                self.train_ds, self.val_ds = dataset, None
                self.val_loader = None
            else:
                self.train_ds, self.val_ds = torch.utils.data.random_split(
                    dataset, [train_len, val_len],
                    generator=torch.Generator().manual_seed(42 + cid)
                )
                self.val_loader = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                                             num_workers=self.num_workers)

            self.train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_workers, pin_memory=(self.device == "cuda"))

    # 通道切片辅助函数
    def _get_input_slice(self, batch_data, config):
        if config is None or "target_region" not in config:
            return batch_data  # 全通道 (Stage 3 / Test)

        target = config["target_region"]
        if target == "GLOBAL":
            return batch_data
        if target == "ET":
            return batch_data[:, 1:2, ...]  # T1ce
        elif target == "WT":
            return batch_data[:, 3:4, ...]  # Flair
        elif target == "TC":
            return batch_data[:, 0:1, ...]  # T1
        return batch_data

    def _apply_modality_dropout(self, input_data, config):
        if config is None:
            return input_data
        drop_prob = float(config.get("modality_dropout", 0.0) or 0.0)
        if drop_prob <= 0.0 or input_data.shape[1] <= 1:
            return input_data

        drop_prob = min(drop_prob, 0.95)
        bsz, channels = input_data.shape[:2]
        keep_mask = (torch.rand(bsz, channels, 1, 1, 1, device=input_data.device) > drop_prob).float()

        empty_samples = keep_mask.flatten(1).sum(dim=1) == 0
        if empty_samples.any():
            for sample_idx in torch.nonzero(empty_samples, as_tuple=False).flatten():
                channel_idx = int(torch.randint(channels, (1,), device=input_data.device).item())
                keep_mask[int(sample_idx.item()), channel_idx] = 1.0

        return input_data * keep_mask

    # 滑动窗口推理
    def _sliding_window_inference(self, full_volume, patch_size=(96, 96, 96), overlap=0.5, loss_config=None):
        b, c, d, h, w = full_volume.shape
        pd, ph, pw = patch_size
        pad_d = max(0, pd - d)
        pad_h = max(0, ph - h)
        pad_w = max(0, pw - w)
        if pad_d or pad_h or pad_w:
            full_volume = F.pad(full_volume, (0, pad_w, 0, pad_h, 0, pad_d))

        sd = int(pd * (1 - overlap))
        sh = int(ph * (1 - overlap))
        sw = int(pw * (1 - overlap))
        _, _, dd, hh, ww = full_volume.shape

        # 1. 确定输出通道数
        self.model.eval()  # 确保是评估模式
        with torch.no_grad():
            dummy_patch = full_volume[:, :, :pd, :ph, :pw]
            dummy_in = self._get_input_slice(dummy_patch, loss_config)
            with autocast('cuda', enabled=self.use_amp):
                dummy_out = self.model(dummy_in)
                # [安全检查] 滑窗只应该在 eval 模式运行，所以应该返回 Tensor
                if isinstance(dummy_out, tuple):
                    dummy_out = dummy_out[0]
            num_classes = dummy_out.shape[1]

        # 2. 初始化全图容器
        prob_map = torch.zeros((b, num_classes, dd, hh, ww), device=self.device)
        count_map = torch.zeros((b, num_classes, dd, hh, ww), device=self.device)

        # 3. 生成滑窗坐标
        def get_coords(dim_len, patch_len, stride_len):
            coords = list(range(0, dim_len - patch_len + 1, stride_len))
            if coords[-1] != dim_len - patch_len:
                coords.append(dim_len - patch_len)
            return coords

        dz_list = get_coords(dd, pd, sd)
        dy_list = get_coords(hh, ph, sh)
        dx_list = get_coords(ww, pw, sw)

        # 4. 滑动推理
        for z in dz_list:
            for y in dy_list:
                for x in dx_list:
                    patch = full_volume[:, :, z:z + pd, y:y + ph, x:x + pw]
                    patch_in = self._get_input_slice(patch, loss_config)

                    with autocast('cuda', enabled=self.use_amp):
                        pred = self.model(patch_in)
                        # [安全检查]
                        if isinstance(pred, tuple): pred = pred[0]

                    prob_map[:, :, z:z + pd, y:y + ph, x:x + pw] += pred.float()
                    count_map[:, :, z:z + pd, y:y + ph, x:x + pw] += 1.0

        # 5. 平均
        final_pred = prob_map / count_map
        return final_pred[:, :, :d, :h, :w]

    def train(self, global_weights, global_alphas=[], current_lr=None, loss_config=None, current_temp=1.0,is_warmup=False):
        # 1. 加载权重
        arch_param_names = set()
        if hasattr(self.model, "arch_parameters"):
            arch_ids = {id(p) for p in self.model.arch_parameters()}
            arch_param_names = {
                name
                for name, param in self.model.named_parameters()
                if id(param) in arch_ids
            }
        clean_state = {
            k: v
            for k, v in global_weights.items()
            if k not in arch_param_names and not k.endswith('.alpha') and not k.endswith('.gate')
        }
        self.model.load_state_dict(clean_state, strict=False)

        if hasattr(self.model, "arch_parameters") and len(global_alphas) > 0:
            with torch.no_grad():
                for p, g in zip(self.model.arch_parameters(), global_alphas):
                    p.copy_(g.to(device=p.device, dtype=p.dtype))

        if hasattr(self.model, 'alpha_mgr') and self.model.alpha_mgr and len(global_alphas) > 0:
            with torch.no_grad():
                for p, g in zip(self.model.alpha_mgr.parameters(), global_alphas): p.copy_(g)
            if hasattr(self.model, 'set_alpha'): self.model.set_alpha()

        # 2. 优化器设置
        actual_lr = current_lr if current_lr is not None else self.lr

        base_params = []
        stream_params = []

        arch_param_ids = set()
        if hasattr(self.model, "arch_parameters"):
            arch_param_ids = {id(p) for p in self.model.arch_parameters()}

        # 将参数分离：区分普通网络权重和特征流融合权重
        for n, p in self.model.named_parameters():
            if id(p) in arch_param_ids or not p.requires_grad:
                continue
            if 'stream_weights' in n:
                stream_params.append(p)
            else:
                base_params.append(p)

        # 构建参数组
        optimizer_groups = [{'params': base_params, 'lr': actual_lr}]

        # 如果存在特征流权重，给它们单独分配较大的学习率（例如 1e-2），且不使用权重衰减
        if len(stream_params) > 0:
            optimizer_groups.append({
                'params': stream_params,
                'lr': 1e-3,
                'weight_decay': 0.0
            })

        self.optimizer = optim.AdamW(optimizer_groups, weight_decay=self.weight_decay)

        self.alpha_optimizer = None
        if self.lr_alpha > 0 and hasattr(self.model, 'arch_parameters')and not is_warmup:
            alpha_params = self.model.arch_parameters()
            if len(alpha_params) > 0:
                self.alpha_optimizer = optim.Adam(alpha_params, lr=self.lr_alpha, weight_decay=1e-3)

        # 3. Loss 设置：支持 region / boundary / hybrid / search
        loss_type = loss_config.get("loss_type", "ce") if loss_config else "ce"
        if loss_type in {"hybrid", "boundary", "region"}:
            use_boundary = loss_type in {"hybrid", "boundary"}
            try:
                criterion = HybridDeepSupervisionLoss(
                    weight_boundary=loss_config.get("weight_boundary", 0.0) if use_boundary else 0.0,
                    curvature_weight=loss_config.get("curvature_weight", 2.0) if use_boundary else 0.0,
                    et_weight=loss_config.get("et_weight", 1.5),
                    bce_weight=loss_config.get("bce_weight", 0.5),
                    tc_fp_weight=loss_config.get("tc_fp_weight", 0.0),
                    tc_fp_distance=loss_config.get("tc_fp_distance", 32.0),
                    tc_fp_power=loss_config.get("tc_fp_power", 1.0),
                    tc_fp_min_distance=loss_config.get("tc_fp_min_distance", 0.0),
                )
            except NameError:
                print("[Warning] HybridDeepSupervisionLoss not found, falling back to HybridTaskLoss")
                criterion = HybridTaskLoss(
                    weight_boundary=loss_config.get("weight_boundary", 0.0) if use_boundary else 0.0,
                    curvature_weight=loss_config.get("curvature_weight", 2.0) if use_boundary else 0.0,
                    et_weight=loss_config.get("et_weight", 1.5),
                    bce_weight=loss_config.get("bce_weight", 0.5),
                    tc_fp_weight=loss_config.get("tc_fp_weight", 0.0),
                    tc_fp_distance=loss_config.get("tc_fp_distance", 32.0),
                    tc_fp_power=loss_config.get("tc_fp_power", 1.0),
                    tc_fp_min_distance=loss_config.get("tc_fp_min_distance", 0.0),
                )
        elif loss_config and loss_config.get("loss_type") == "search":
            criterion = SearchLoss(target_region=loss_config.get("target_region", "WT"))
        else:
            criterion = NewCombinedLoss()

        criterion = criterion.to(self.device)
        search_target = loss_config.get("target_region") if loss_config else None
        fedprox_mu = float(loss_config.get("fedprox_mu", 0.0) or 0.0) if loss_config else 0.0
        fedprox_reference = None
        if fedprox_mu > 0.0:
            fedprox_reference = {
                k: v.detach().to(self.device)
                for k, v in clean_state.items()
                if isinstance(v, torch.Tensor) and v.is_floating_point()
            }
        flops_weight = 0.0
        collapse_weight = 0.0
        if loss_config and loss_config.get("loss_type") == "search":
            flops_weight = 1e-8 if search_target != "WT" else 0.0

            # 让 ET 和 WT 都受到防坍塌机制的保护！
            # ET 甚至可以给稍微高一点点的惩罚，因为它的特征最难抓取
            if search_target == "ET":
                collapse_weight = 0.1
            elif search_target == "WT":
                collapse_weight = 0.08
            else:
                collapse_weight = 0.0

        # 4. 训练循环
        self.model.train()
        epoch_losses = []

        if self.train_loader:
            for ep in range(self.epochs):
                batch_losses = []
                # ====== [关键修复 2] 使用 enumerate 获取 batch_idx 以实现交替优化 ======
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(self.device), target.to(self.device)

                    # 确定当前 Batch 是更新 Alpha 还是更新 Weight
                    is_alpha_step = (self.alpha_optimizer is not None) and (batch_idx % 2 == 1)
                    if hasattr(self.model, "set_search_mode"):
                        self.model.set_search_mode(temp=current_temp, hard=not is_alpha_step)

                    # 1. 每次前向传播前，必须将【两个】优化器都清零，防止跨步残留的未 unscale 梯度污染全局计算图
                    self.optimizer.zero_grad()
                    if self.alpha_optimizer is not None:
                        self.alpha_optimizer.zero_grad()

                    with autocast('cuda', enabled=self.use_amp):
                        input_data = self._get_input_slice(data, loss_config)
                        if loss_config and loss_config.get("loss_type") == "search":
                            input_data = self._apply_modality_dropout(input_data, loss_config)

                        # 保留你的 hard=True，维持特征尺度的稳定，防止 NaN
                        if loss_config and loss_config.get("loss_type") == "search":
                            output = self.model.forward_gumbel(input_data, temp=current_temp, hard=True)
                        else:
                            output = self.model(input_data)

                        loss = criterion(output, target)
                        if fedprox_reference is not None:
                            prox_loss = None
                            for name, param in self.model.named_parameters():
                                if not param.requires_grad or name not in fedprox_reference:
                                    continue
                                ref = fedprox_reference[name].to(dtype=param.dtype)
                                term = torch.sum((param - ref) ** 2)
                                prox_loss = term if prox_loss is None else prox_loss + term
                            if prox_loss is not None:
                                loss = loss + 0.5 * fedprox_mu * prox_loss
                        if is_alpha_step and flops_weight > 0 and hasattr(self.model, 'expected_flops'):
                            loss = loss + flops_weight * self.model.expected_flops(temp=current_temp).squeeze()
                        if is_alpha_step and collapse_weight > 0 and hasattr(self.model, 'architecture_collapse_loss'):
                            loss = loss + collapse_weight * self.model.architecture_collapse_loss(
                                temp=current_temp).squeeze()

                    if self.scaler:
                        self.scaler.scale(loss).backward()

                        # 2. 严格隔离 unscale 和裁剪目标
                        if is_alpha_step:
                            self.scaler.unscale_(self.alpha_optimizer)
                            # 只计算和裁剪 Alpha 参数的 norm
                            if hasattr(self.model, 'arch_parameters'):
                                torch.nn.utils.clip_grad_norm_(self.model.arch_parameters(), max_norm=1.0)
                            self.scaler.step(self.alpha_optimizer)
                        else:
                            self.scaler.unscale_(self.optimizer)
                            # 只计算和裁剪 Weight 参数的 norm
                            opt_params = [p for group in self.optimizer.param_groups for p in group['params']]
                            torch.nn.utils.clip_grad_norm_(opt_params, max_norm=1.0)
                            self.scaler.step(self.optimizer)

                        self.scaler.update()
                    else:
                        # 非 AMP 模式同样需要隔离裁剪
                        loss.backward()
                        if is_alpha_step:
                            if hasattr(self.model, 'arch_parameters'):
                                torch.nn.utils.clip_grad_norm_(self.model.arch_parameters(), max_norm=1.0)
                            self.alpha_optimizer.step()
                        else:
                            opt_params = [p for group in self.optimizer.param_groups for p in group['params']]
                            torch.nn.utils.clip_grad_norm_(opt_params, max_norm=1.0)
                            self.optimizer.step()

                    batch_losses.append(loss.item())
                epoch_losses.append(np.mean(batch_losses))

        # 5. 验证循环 (Validation)
        val_dice = 0.0
        if self.val_loader:
            self.model.eval()  # [注意] 切换到 eval 模式，Deep Supervision 应该自动关闭
            if hasattr(self.model, "set_search_mode"):
                self.model.set_search_mode(temp=current_temp, hard=True)
            dices = []

            is_search_mode = (loss_config and "target_region" in loss_config)
            search_target = loss_config["target_region"] if is_search_mode else None

            with torch.no_grad():
                for data, target in self.val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    val_input = self._get_input_slice(data, loss_config)

                    with autocast('cuda', enabled=self.use_amp):
                        out = self.model(val_input)

                    # [关键修改] 验证时的安全检查
                    # 即使 model.eval() 关闭了 DS 返回，防止任何意外情况
                    if isinstance(out, tuple):
                        out = out[0]

                    if self.use_amp: out = out.float()

                    if is_search_mode:
                        # === Search Mode (Binary) Validation ===
                        if search_target == "ET":
                            pred_logits = out[:, 3:4]
                            gt = (target == 3).float().unsqueeze(1)
                            pred_mask = (pred_logits > 0.0).float()

                        elif search_target == "WT":
                            probs = torch.softmax(out, dim=1)
                            pred_prob_wt = torch.sum(probs[:, 1:], dim=1, keepdim=True)
                            gt = (target > 0).float().unsqueeze(1)
                            pred_mask = (pred_prob_wt > 0.5).float()
                        elif search_target == "GLOBAL":
                            res = metrics.calculate_brats_metrics(out, target, compute_hd95=False)
                            dices.append(res["Dice_Mean"])
                            continue

                        else:
                            pred_mask = torch.zeros_like(target).unsqueeze(1)
                            gt = target.unsqueeze(1)

                        dice_val = metrics.compute_dice_single(pred_mask, gt)
                        dices.append(dice_val)

                    else:
                        # === Normal/Retrain Mode (Multiclass) Validation ===
                        target_metric_key = "Dice_Mean"
                        # 验证时不需要 HD95，加速
                        res = metrics.calculate_brats_metrics(out, target, compute_hd95=False)
                        dices.append(res[target_metric_key])

            val_dice = np.mean(dices) if dices else 0.0
            self.model.train()  # 恢复训练模式

        # 收集最新的 alpha
        updated_alphas = []
        if hasattr(self.model, "arch_parameters"):
            updated_alphas = [p.detach().cpu() for p in self.model.arch_parameters()]
        elif hasattr(self.model, 'alpha_mgr') and self.model.alpha_mgr:
            updated_alphas = [p.detach().cpu() for p in self.model.alpha_mgr.parameters()]

        return {
            "weights": self.model.state_dict(),
            "loss": np.mean(epoch_losses) if epoch_losses else 0.0,
            "size": len(self.train_ds) if self.train_ds else 0,
            "val_dice": val_dice,
            "alpha": updated_alphas
        }

    def evaluate(self, global_weights, loss_config=None):
        if not self.val_loader: return {}

        arch_param_names = set()
        if hasattr(self.model, "arch_parameters"):
            arch_ids = {id(p) for p in self.model.arch_parameters()}
            arch_param_names = {
                name
                for name, param in self.model.named_parameters()
                if id(param) in arch_ids
            }
        clean_state = {
            k: v
            for k, v in global_weights.items()
            if k not in arch_param_names and not k.endswith('.alpha') and not k.endswith('.gate')
        }
        self.model.load_state_dict(clean_state, strict=False)
        self.model.eval()
        if hasattr(self.model, "set_search_mode"):
            self.model.set_search_mode(temp=0.2, hard=True)

        batch_metrics_list = []
        patch_size = (96, 96, 96)

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                b, c, d, h, w = data.shape

                # 判断是否全图 -> 启用滑动窗口
                if d > patch_size[0] or h > patch_size[1] or w > patch_size[2]:
                    out = self._sliding_window_inference(data, patch_size=patch_size, overlap=0.5,
                                                         loss_config=loss_config)
                else:
                    in_data = self._get_input_slice(data, loss_config)
                    with autocast('cuda', enabled=self.use_amp):
                        out = self.model(in_data)

                # [关键修改] 评估时的安全检查
                if isinstance(out, tuple):
                    out = out[0]

                if self.use_amp: out = out.float()

                # 计算指标 (开启 HD95)
                res = metrics.calculate_brats_metrics(out, target, compute_hd95=True)
                batch_metrics_list.append(res)

        if not batch_metrics_list: return {}
        avg_metrics = {}
        for k in batch_metrics_list[0].keys():
            vals = [m[k] for m in batch_metrics_list if m[k] != -1]
            avg_metrics[k] = float(np.mean(vals)) if vals else 0.0
        return avg_metrics
