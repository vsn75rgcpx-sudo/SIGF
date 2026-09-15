import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.models import build_baseline_model
from losses import HybridDeepSupervisionLoss, NewCombinedLoss
import metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Strong BraTS baseline trainer")
    parser.add_argument("--model", type=str, default="unet3d",
                        choices=["unet3d", "fednas_unet3d", "modal_swin_unet3d",
                                 "modal_attn_swin_unet3d", "parallel_modal_swin_unet3d",
                                 "parallel_modal_unet_no_swin3d",
                                 "parallel_swin_unet_no_modal_stem3d",
                                 "segresnet", "fedmunet3d", "mednext_lite",
                                 "benchmark_fets_dynunet", "fets_dynunet",
                                 "monai_segresnet", "monai_swinunetr"])
    parser.add_argument("--exp_name", type=str, default="baseline_unet3d")
    parser.add_argument("--data_root", type=str, default="dataset/train")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--nas_op", type=str, default="mixed",
                        choices=["mixed", "conv3", "conv5", "depthwise3",
                                 "residual_depthwise", "residual", "skip"],
                        help="Block operation for fednas_unet3d")
    parser.add_argument("--patch_size", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--loss_type", type=str, default="hybrid",
                        choices=["ce", "region", "boundary", "hybrid"],
                        help="region disables boundary/curvature weights; boundary/hybrid enable them")
    parser.add_argument("--boundary_weight", type=float, default=0.5)
    parser.add_argument("--curvature_weight", type=float, default=2.0)
    parser.add_argument("--et_loss_weight", type=float, default=1.5)
    parser.add_argument("--bce_weight", type=float, default=0.5)
    parser.add_argument("--tc_fp_weight", type=float, default=0.0,
                        help="Weight for distance-aware TC false-positive suppression loss")
    parser.add_argument("--tc_fp_distance", type=float, default=32.0,
                        help="Distance cap in voxels for TC false-positive suppression")
    parser.add_argument("--tc_fp_power", type=float, default=1.0,
                        help="Power applied to normalized TC distance weights")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_top_k", type=int, default=3)
    parser.add_argument("--checkpoint_score", type=str, default="dice",
                        choices=["dice", "dice_hd95"],
                        help="Checkpoint selection score: validation Dice or Dice - hd95_penalty * HD95")
    parser.add_argument("--hd95_penalty", type=float, default=0.01,
                        help="HD95 penalty used when checkpoint_score=dice_hd95")
    parser.add_argument("--checkpoint_score_candidates", type=int, default=10,
                        help="Number of Dice-preselected candidates to rescore with HD95")
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_cases(data_root):
    def has_case_files(path):
        def find_file(candidates):
            for suffix in candidates:
                if list(path.glob(f"*{suffix}.nii.gz")) or list(path.glob(f"*{suffix}*.nii*")):
                    return True
            return False

        return (
            find_file(["t1n", "t1"])
            and find_file(["t1c", "t1ce"])
            and find_file(["t2w", "t2"])
            and find_file(["t2f", "flair"])
            and find_file(["seg"])
        )

    root = Path(data_root)
    if not root.exists() and data_root == "dataset/train":
        root = Path("dataset")
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")

    candidate_roots = [root]
    for nested in (root / "train", root / "dataset", root / "dataset" / "train"):
        if nested.exists():
            candidate_roots.append(nested)

    cases = []
    seen = set()
    for candidate_root in candidate_roots:
        for path in sorted(candidate_root.iterdir()):
            if not path.is_dir():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            if has_case_files(path):
                cases.append(path)
                seen.add(resolved)
    if not cases:
        raise FileNotFoundError(
            f"No valid BraTS case directories found under {root}. "
            "Please pass the directory whose children are case folders, for example --data_root dataset/train."
        )
    return [str(p) for p in cases]


def split_cases(cases, test_ratio, seed):
    rng = random.Random(seed)
    cases = list(cases)
    rng.shuffle(cases)
    test_count = int(len(cases) * test_ratio)
    test_cases = cases[:test_count]
    train_cases = cases[test_count:]
    return train_cases, test_cases


def make_loaders(args):
    from datasets.brats_multimodal_dataset import BraTSMultiModalDataset

    cases = list_cases(args.data_root)
    train_cases, test_cases = split_cases(cases, args.test_ratio, args.seed)

    rng = random.Random(args.seed + 17)
    train_cases = list(train_cases)
    rng.shuffle(train_cases)
    val_count = max(1, int(len(train_cases) * args.val_ratio))
    val_cases = train_cases[:val_count]
    fit_cases = train_cases[val_count:]

    train_ds = BraTSMultiModalDataset(fit_cases, target_size=tuple(args.patch_size), augment=True)
    val_ds = BraTSMultiModalDataset(val_cases, target_size=tuple(args.patch_size), augment=False)
    test_ds = BraTSMultiModalDataset(test_cases, augment=False, test_mode=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader, test_cases


def build_model(args):
    model = build_baseline_model(
        args.model,
        in_channels=4,
        out_channels=3,
        base_channels=args.base_channels,
        patch_size=tuple(args.patch_size),
        nas_op=args.nas_op,
    )
    return model.to(args.device)


def build_criterion(args):
    if args.loss_type == "ce":
        return NewCombinedLoss().to(args.device)

    use_boundary = args.loss_type in {"boundary", "hybrid"}
    return HybridDeepSupervisionLoss(
        weight_boundary=args.boundary_weight if use_boundary else 0.0,
        curvature_weight=args.curvature_weight if use_boundary else 0.0,
        et_weight=args.et_loss_weight,
        bce_weight=args.bce_weight,
        tc_fp_weight=args.tc_fp_weight,
        tc_fp_distance=args.tc_fp_distance,
        tc_fp_power=args.tc_fp_power,
    ).to(args.device)


def train_one_epoch(model, loader, criterion, optimizer, scaler, args):
    model.train()
    losses = []
    for data, target in loader:
        data = data.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=args.amp and args.device == "cuda"):
            pred = model(data)
            loss = criterion(pred, target)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def validate(model, loader, args):
    model.eval()
    vals = []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(args.device)
            target = target.to(args.device)
            with torch.cuda.amp.autocast(enabled=args.amp and args.device == "cuda"):
                pred = model(data)
            pred = pred.float()
            res = metrics.calculate_brats_metrics(pred, target, compute_hd95=False)
            vals.append(res["Dice_Mean"])
    return float(np.mean(vals)) if vals else 0.0


def sliding_window_predict(model, volume, patch_size, device, overlap=0.5, amp=False):
    model.eval()
    _, _, d, h, w = volume.shape
    pd, ph, pw = patch_size
    pad_d = max(0, pd - d)
    pad_h = max(0, ph - h)
    pad_w = max(0, pw - w)
    if pad_d or pad_h or pad_w:
        volume = F.pad(volume, (0, pad_w, 0, pad_h, 0, pad_d))

    _, _, dd, hh, ww = volume.shape
    sd = max(1, int(pd * (1 - overlap)))
    sh = max(1, int(ph * (1 - overlap)))
    sw = max(1, int(pw * (1 - overlap)))

    def coords(dim, patch, stride):
        values = list(range(0, max(1, dim - patch + 1), stride))
        last = dim - patch
        if values[-1] != last:
            values.append(last)
        return values

    out = torch.zeros((volume.shape[0], 3, dd, hh, ww), device=device)
    count = torch.zeros_like(out)
    for z in coords(dd, pd, sd):
        for y in coords(hh, ph, sh):
            for x in coords(ww, pw, sw):
                patch = volume[:, :, z:z + pd, y:y + ph, x:x + pw]
                with torch.cuda.amp.autocast(enabled=amp and device == "cuda"):
                    pred = model(patch)
                out[:, :, z:z + pd, y:y + ph, x:x + pw] += pred.float()
                count[:, :, z:z + pd, y:y + ph, x:x + pw] += 1
    out = out / count.clamp_min(1)
    return out[:, :, :d, :h, :w]


def evaluate_cases(model, loader, case_paths, args, compute_hd95=True):
    model.eval()
    case_metrics = []
    patch_size = tuple(args.patch_size)
    with torch.no_grad():
        for idx, (data, target) in enumerate(loader):
            data = data.to(args.device)
            target = target.to(args.device)
            pred = sliding_window_predict(
                model,
                data,
                patch_size=patch_size,
                device=args.device,
                amp=args.amp,
            )
            res = metrics.calculate_brats_metrics(pred, target, compute_hd95=compute_hd95)
            case_name = Path(case_paths[idx]).name if idx < len(case_paths) else str(idx)
            res["case"] = case_name
            case_metrics.append(res)
            print(f"[Eval] {case_name}: Dice_Mean={res['Dice_Mean']:.4f}")
    return case_metrics


def summarize_case_metrics(case_metrics):
    keys = [k for k in case_metrics[0].keys() if k != "case"]
    summary = {}
    for key in keys:
        vals = np.array([m[key] for m in case_metrics if m[key] != -1], dtype=np.float64)
        if vals.size == 0:
            continue
        summary[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        }
    return summary


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_checkpoint(path, model, optimizer, epoch, val_dice):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "val_dice": val_dice,
        },
        path,
    )


def prune_checkpoints(saved, keep):
    while len(saved) > keep:
        _, path = saved.pop(-1)
        if os.path.exists(path):
            os.remove(path)


def checkpoint_pool_limit(args):
    if args.checkpoint_score == "dice_hd95":
        return max(args.save_top_k, args.checkpoint_score_candidates)
    return args.save_top_k


def validate_metrics(model, loader, args, compute_hd95=True):
    model.eval()
    vals = []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(args.device)
            target = target.to(args.device)
            with torch.cuda.amp.autocast(enabled=args.amp and args.device == "cuda"):
                pred = model(data)
            pred = pred.float()
            vals.append(metrics.calculate_brats_metrics(pred, target, compute_hd95=compute_hd95))

    if not vals:
        return {}
    keys = vals[0].keys()
    return {
        key: float(np.mean([item[key] for item in vals if item.get(key, -1) != -1]))
        for key in keys
        if any(item.get(key, -1) != -1 for item in vals)
    }


def select_checkpoint(saved, model, val_loader, args):
    if not saved:
        raise RuntimeError("No checkpoints were saved.")

    if args.checkpoint_score == "dice":
        return sorted(saved, key=lambda x: x[0], reverse=True)[0][1]

    preselect_count = max(args.save_top_k, args.checkpoint_score_candidates)
    preselected = sorted(saved, key=lambda x: x[0], reverse=True)[:preselect_count]
    scored = []
    current_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(
        f"[Checkpoint Selection] Scoring top-{len(preselected)} candidates with "
        f"Dice - {args.hd95_penalty:g} * HD95"
    )
    for dice, path in preselected:
        state = torch.load(path, map_location=args.device)
        model.load_state_dict(state["model"] if "model" in state else state)
        metric_res = validate_metrics(model, val_loader, args, compute_hd95=True)
        dice_mean = metric_res.get("Dice_Mean", dice)
        hd95_mean = metric_res.get("HD95_Mean", 0.0)
        score = dice_mean - args.hd95_penalty * hd95_mean
        scored.append({
            "score": score,
            "dice": dice_mean,
            "hd95": hd95_mean,
            "path": path,
        })
        print(f"  - Score: {score:.4f} | Dice: {dice_mean:.4f} | HD95: {hd95_mean:.4f} | Path: {path}")

    model.load_state_dict(current_state)
    scored.sort(key=lambda item: item["score"], reverse=True)
    print(f"[Selected] {scored[0]['path']}")
    return scored[0]["path"]


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    exp_dir = Path("results") / args.exp_name
    ckpt_dir = Path("checkpoints") / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"========== BASELINE: {args.exp_name} ({args.model}) ==========")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    train_loader, val_loader, test_loader, test_cases = make_loaders(args)
    model = build_model(args)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Params: {total_params / 1e6:.2f}M")

    criterion = build_criterion(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and args.device == "cuda")

    ckpt_path = args.checkpoint
    if ckpt_path:
        state = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(state["model"] if "model" in state else state)
        print(f"[Load] {ckpt_path}")

    saved = []
    if not args.eval_only:
        for epoch in range(1, args.epochs + 1):
            progress = (epoch - 1) / max(1, args.epochs)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = lr

            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, args)
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                val_dice = validate(model, val_loader, args)
                print(f"[Epoch {epoch:03d}] loss={train_loss:.4f} val_dice={val_dice:.4f} lr={lr:.2e}")
                path = ckpt_dir / f"epoch_{epoch:03d}_dice_{val_dice:.4f}.pth"
                save_checkpoint(path, model, optimizer, epoch, val_dice)
                saved.append((val_dice, str(path)))
                saved.sort(key=lambda x: x[0], reverse=True)
                prune_checkpoints(saved, checkpoint_pool_limit(args))

        best_ckpt = select_checkpoint(saved, model, val_loader, args)
        print(f"[Best] {best_ckpt}")
        state = torch.load(best_ckpt, map_location=args.device)
        model.load_state_dict(state["model"])

    case_metrics = evaluate_cases(model, test_loader, test_cases, args, compute_hd95=True)
    summary = summarize_case_metrics(case_metrics)
    summary["Params_M"] = total_params / 1e6

    save_json(exp_dir / "case_metrics.json", case_metrics)
    save_json(exp_dir / "final_metrics.json", summary)

    print("========== FINAL CASE-LEVEL SUMMARY ==========")
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"{key}: {value['mean']:.4f} ± {value['std']:.4f}")
        else:
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
