

import os
import numpy as np
from torch.utils.data import Dataset
import torch
import random
import scipy.ndimage as ndi
from glob import glob


class BraTSMultiModalDataset(Dataset):
    """
    Loads 4-modality MRI volumes for BraTS.
    """

    def __init__(self,
                 case_dirs,
                 target_size=(96, 96, 96),
                 normalize="zscore",
                 augment=False,
                 label_mapping=None,
                 test_mode=False,
                 use_curvature=False):  # [修改] 增加标签映射参数

        self.case_dirs = case_dirs
        self.target_size = target_size
        self.normalize = normalize
        self.augment = augment
        self.test_mode = test_mode
        self.use_curvature = use_curvature

        # [修改] 默认映射: 将标签 4 (ET) 映射为 3 (ET)，适配 BraTS 2021 标准
        # 如果是旧数据或自定义数据，可传入 {} 或其他映射
        if label_mapping is None:
            self.label_mapping = {4: 3}
        else:
            self.label_mapping = label_mapping

        self.valid_cases = []
        for c in case_dirs:
            if os.path.isdir(c):
                self.valid_cases.append(c)
            else:
                print(f"[Dataset] Warning: Case directory missing: {c}")

    # ---------------- Utils -------------------
    # (保持 load_nii, zscore, resize_3d, random_crop_or_pad 不变)
    def load_nii(self, path):
        import nibabel as nib

        return nib.load(path).get_fdata().astype(np.float32)

    def zscore(self, x):
        mask = x > 0
        if mask.sum() > 0:
            mean = x[mask].mean()
            std = x[mask].std()
            res = (x - mean) / (std + 1e-8)

            # [修改] 方案A：背景取当前数据的最小值 (推荐)
            # 这样背景永远是数值最小的，比如 -2.5
            min_val = res[mask].min()
            res[~mask] = min_val

            # 或者 方案B：不要手动处理背景，直接保留 Z-score 结果
            # 如果计算均值包含了背景，背景自然会变成负数
            # 如果计算均值只用前景，通常需要在 ~mask 处填入一个比 min_val 更小的值

            return res
        return x

    def random_crop_or_pad(self, img, lbl, target_shape):
        D, H, W = img.shape[1:]
        tD, tH, tW = target_shape

        # 1. Padding 逻辑
        pad_d, pad_h, pad_w = max(0, tD - D), max(0, tH - H), max(0, tW - W)
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            lbl = np.pad(lbl, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            D, H, W = img.shape[1:]

        # 2. Crop 坐标计算
        d_start, h_start, w_start = 0, 0, 0

        # [修复重点] 将变量提取放在最外层，确保任何分支都能访问到
        foreground_indices = np.argwhere(lbl > 0)
        et_indices = np.argwhere((lbl == 3) | (lbl == 4))

        rand_val = random.random()

        if self.augment and len(et_indices) > 0 and rand_val < 0.2:
            # 20% 概率：以 ET 为中心
            center_idx = random.randint(0, len(et_indices) - 1)
            cd, ch, cw = et_indices[center_idx]
            d_start = max(0, min(cd - tD // 2, D - tD))
            h_start = max(0, min(ch - tH // 2, H - tH))
            w_start = max(0, min(cw - tW // 2, W - tW))

        elif self.augment and len(foreground_indices) > 0 and rand_val < 0.6:
            # 40% 概率：以整个肿瘤 (WT) 为中心
            center_idx = random.randint(0, len(foreground_indices) - 1)
            cd, ch, cw = foreground_indices[center_idx]
            d_start = max(0, min(cd - tD // 2, D - tD))
            h_start = max(0, min(ch - tH // 2, H - tH))
            w_start = max(0, min(cw - tW // 2, W - tW))

        else:
            # 40% 概率：完全随机裁剪 (包含大量背景)
            d_start = random.randint(0, max(0, D - tD))
            h_start = random.randint(0, max(0, H - tH))
            w_start = random.randint(0, max(0, W - tW))

        img_crop = img[:, d_start:d_start + tD, h_start:h_start + tH, w_start:w_start + tW]
        lbl_crop = lbl[d_start:d_start + tD, h_start:h_start + tH, w_start:w_start + tW]
        return img_crop, lbl_crop

    # ---------------- Main -------------------
    def __len__(self):
        return len(self.valid_cases)

    def __getitem__(self, idx):
        case = self.valid_cases[idx]

        def find_file(primary_key, candidates):
            search_list = candidates if isinstance(candidates, list) else [candidates]
            for s in search_list:
                f = glob(os.path.join(case, f"*{s}.nii.gz"))
                if not f: f = glob(os.path.join(case, f"*{s}*.nii*"))
                if f: return f[0]
            return None

        p_t1 = find_file("t1", ["t1n", "t1"])
        p_t1ce = find_file("t1c", ["t1c", "t1ce"])
        p_t2 = find_file("t2", ["t2w", "t2"])
        p_flair = find_file("flair", ["t2f", "flair"])
        p_seg = find_file("seg", ["seg"])

        # [新增] 动态加载曲率模态
        p_curv = None
        if self.use_curvature:
            p_curv = find_file("curv", ["curvature", "curv"])
            # if p_curv is None:
            #     print(f"[Dataset Warning] Curvature file not found for {case}, ignoring curvature.")

        if None in [p_t1, p_t1ce, p_t2, p_flair, p_seg]:
            raise FileNotFoundError(f"Missing base modality files in {case}")

        t1 = self.load_nii(p_t1)
        t1ce = self.load_nii(p_t1ce)
        t2 = self.load_nii(p_t2)
        flair = self.load_nii(p_flair)
        seg = self.load_nii(p_seg).astype(np.int64)

        for src, dst in self.label_mapping.items():
            seg[seg == src] = dst

        vols = [t1, t1ce, t2, flair]

        # [新增] 如果使用了曲率且存在该文件，将其加入通道中
        if self.use_curvature and p_curv is not None:
            curv_data = self.load_nii(p_curv)
            vols.append(curv_data)

        if self.normalize == "zscore":
            vols = [self.zscore(v) for v in vols]

        img = np.stack(vols, axis=0)  # (4 或 5, D, H, W)

        if self.test_mode:
            return torch.tensor(img).float(), torch.tensor(seg).long()

        img, seg = self.random_crop_or_pad(img, seg, self.target_size)

        if self.augment and random.random() < 0.5:
            axis = random.choice([1, 2, 3])
            img = np.flip(img, axis=axis).copy()
            seg = np.flip(seg, axis=axis - 1).copy()

        return torch.tensor(img).float(), torch.tensor(seg).long()
