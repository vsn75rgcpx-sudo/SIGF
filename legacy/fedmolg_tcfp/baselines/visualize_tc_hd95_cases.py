import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize TC HD95-improved paired cases for fixed FL baseline models."
    )
    parser.add_argument("--paired_csv", required=True,
                        help="CSV from baselines/paired_case_comparison.py")
    parser.add_argument("--data_root", default="dataset/train")
    parser.add_argument("--out_dir", default="results/visual_tc_hd95_cases")
    parser.add_argument("--num_cases", type=int, default=5)
    parser.add_argument("--case", action="append", default=[],
                        help="Specific case name to visualize. Can be repeated.")
    parser.add_argument("--min_hd95_tc_improvement", type=float, default=0.0)
    parser.add_argument("--min_dice_tc_delta", type=float, default=-0.02,
                        help="Prefer cases where Dice_TC(method-baseline) is not lower than this.")
    parser.add_argument("--min_dice_tc_method", type=float, default=0.5,
                        help="Avoid visually uninformative TC cases where the method TC Dice is too low.")
    parser.add_argument("--max_hd95_tc_baseline", type=float, default=200.0,
                        help="Avoid sentinel/extreme baseline HD95_TC cases unless explicitly relaxed.")
    parser.add_argument("--max_hd95_tc_method", type=float, default=200.0,
                        help="Avoid sentinel/extreme method HD95_TC cases unless explicitly relaxed.")
    parser.add_argument("--allow_fallback", action="store_true",
                        help="If strict Dice filter selects no cases, fall back to top HD95_TC cases.")

    parser.add_argument("--method_model", default="parallel_modal_swin_unet3d")
    parser.add_argument("--method_checkpoint", required=True)
    parser.add_argument("--method_base_channels", type=int, default=24)
    parser.add_argument("--method_name", default="Ours")

    parser.add_argument("--baseline_model", default="unet3d")
    parser.add_argument("--baseline_checkpoint", required=True)
    parser.add_argument("--baseline_base_channels", type=int, default=32)
    parser.add_argument("--baseline_name", default="FedAvg U-Net")

    parser.add_argument("--patch_size", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--modality", type=int, default=3,
                        help="MRI channel to show: 0=T1, 1=T1ce, 2=T2, 3=FLAIR")
    parser.add_argument("--slice_axis", type=int, default=0,
                        help="0=axial/depth, 1=coronal/height, 2=sagittal/width")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_postprocess", action="store_true")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only rank/select cases and write selected_tc_hd95_cases.csv.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_ratio", type=float, default=0.3)
    return parser.parse_args()


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def float_value(row, key, default=np.nan):
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def select_cases(args):
    rows = read_rows(args.paired_csv)
    by_case = {row["case"]: row for row in rows}
    if args.case:
        missing = [case for case in args.case if case not in by_case]
        if missing:
            raise RuntimeError(f"Requested cases are not in paired CSV: {missing}")
        return args.case, by_case

    strict = []
    relaxed = []
    for row in rows:
        hd95_imp = float_value(row, "HD95_TC_improvement", -np.inf)
        dice_tc_delta = float_value(row, "Dice_TC_raw_delta", -np.inf)
        dice_tc_method = float_value(row, "Dice_TC_method", -np.inf)
        hd95_tc_method = float_value(row, "HD95_TC_method", np.inf)
        hd95_tc_baseline = float_value(row, "HD95_TC_baseline", np.inf)
        if not np.isfinite(hd95_imp):
            continue
        item = (hd95_imp, dice_tc_delta, row["case"])
        passes_visual_quality = (
            dice_tc_method >= args.min_dice_tc_method
            and hd95_tc_method <= args.max_hd95_tc_method
            and hd95_tc_baseline <= args.max_hd95_tc_baseline
        )
        if hd95_imp >= args.min_hd95_tc_improvement and passes_visual_quality:
            relaxed.append(item)
            if dice_tc_delta >= args.min_dice_tc_delta:
                strict.append(item)

    selected = strict
    if not selected and args.allow_fallback:
        selected = relaxed
    if not selected:
        raise RuntimeError(
            "No cases selected. Lower --min_hd95_tc_improvement/--min_dice_tc_delta "
            "or add --allow_fallback."
        )
    selected = sorted(selected, reverse=True)[:args.num_cases]
    return [case for _, _, case in selected], by_case


def case_map(data_root, seed, test_ratio):
    from baselines.train_baseline import list_cases, split_cases

    cases = list_cases(data_root)
    _, test_cases = split_cases(cases, test_ratio, seed)
    return {Path(path).name: path for path in test_cases}


def all_case_map(data_root):
    from baselines.train_baseline import list_cases

    return {Path(path).name: path for path in list_cases(data_root)}


def build_model(model_name, checkpoint, base_channels, args):
    from baselines.models import build_baseline_model
    from baselines.train_federated_baseline import load_checkpoint_to_model

    model = build_baseline_model(
        model_name,
        in_channels=4,
        out_channels=3,
        base_channels=base_channels,
        patch_size=tuple(args.patch_size),
    ).to(args.device)
    load_checkpoint_to_model(model, checkpoint, args.device)
    model.eval()
    return model


def infer(model, volume, args):
    import torch
    from baselines.train_baseline import sliding_window_predict

    with torch.no_grad():
        pred = sliding_window_predict(
            model,
            volume.to(args.device),
            patch_size=tuple(args.patch_size),
            device=args.device,
            amp=args.amp and args.device == "cuda",
        )
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    return pred.float()


def logits_to_label_mask(logits, postprocess=True):
    from metrics import postprocess_hierarchical

    logits = logits.detach().cpu().numpy()[0]
    pred = np.zeros_like(logits[0], dtype=np.int16)
    pred[logits[0] > 0] = 2
    pred[logits[1] > 0] = 1
    pred[logits[2] > 0] = 3
    if postprocess:
        pred = postprocess_hierarchical(pred)
    return pred


def tc_region(mask):
    return (mask == 1) | (mask == 3)


def take_slice(arr, axis, idx):
    if axis == 0:
        return arr[idx, :, :]
    if axis == 1:
        return arr[:, idx, :]
    return arr[:, :, idx]


def tumor_slice_index(gt_tc, baseline_tc, method_tc, axis):
    tumor = gt_tc | baseline_tc | method_tc
    axes = tuple(i for i in range(3) if i != axis)
    areas = tumor.sum(axis=axes)
    if areas.max() == 0:
        return tumor.shape[axis] // 2
    return int(np.argmax(areas))


def normalize_image(x):
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def overlay_region(image, region, color, alpha=0.42):
    gray = np.stack([normalize_image(image)] * 3, axis=-1)
    rgb = gray.copy()
    rgb[region] = (1 - alpha) * gray[region] + alpha * np.asarray(color, dtype=np.float32)
    return rgb


def boundary2d(region):
    region = region.astype(bool)
    if not np.any(region):
        return np.zeros_like(region, dtype=bool)
    eroded = ndimage.binary_erosion(region, structure=np.ones((3, 3)), border_value=0)
    return region ^ eroded


def boundary_overlay(image, gt_tc, baseline_tc, method_tc):
    rgb = np.stack([normalize_image(image)] * 3, axis=-1)
    rgb[boundary2d(gt_tc)] = np.array([1.0, 1.0, 1.0])
    rgb[boundary2d(baseline_tc)] = np.array([0.15, 0.45, 1.0])
    rgb[boundary2d(method_tc)] = np.array([1.0, 0.20, 0.20])
    return rgb


def error_map(pred_tc, gt_tc):
    fp = pred_tc & ~gt_tc
    fn = ~pred_tc & gt_tc
    overlap = pred_tc & gt_tc
    out = np.zeros(pred_tc.shape + (3,), dtype=np.float32)
    out[overlap] = np.array([0.25, 0.85, 0.30])
    out[fp] = np.array([1.0, 0.25, 0.25])
    out[fn] = np.array([0.25, 0.45, 1.0])
    return out


def save_case_figure(case_name, image, gt_mask, baseline_mask, method_mask, row, args):
    gt_tc = tc_region(gt_mask)
    baseline_tc = tc_region(baseline_mask)
    method_tc = tc_region(method_mask)
    idx = tumor_slice_index(gt_tc, baseline_tc, method_tc, args.slice_axis)

    image_sl = take_slice(image, args.slice_axis, idx)
    gt_sl = take_slice(gt_tc, args.slice_axis, idx)
    base_sl = take_slice(baseline_tc, args.slice_axis, idx)
    method_sl = take_slice(method_tc, args.slice_axis, idx)

    panels = [
        ("MRI", np.stack([normalize_image(image_sl)] * 3, axis=-1)),
        ("GT TC", overlay_region(image_sl, gt_sl, [1.0, 0.78, 0.18])),
        (f"{args.baseline_name} TC", overlay_region(image_sl, base_sl, [0.15, 0.45, 1.0])),
        (f"{args.method_name} TC", overlay_region(image_sl, method_sl, [1.0, 0.20, 0.20])),
        (f"{args.baseline_name} Error", error_map(base_sl, gt_sl)),
        (f"{args.method_name} Error", error_map(method_sl, gt_sl)),
        ("Boundary", boundary_overlay(image_sl, gt_sl, base_sl, method_sl)),
    ]

    fig, axes = plt.subplots(
        1, len(panels), figsize=(3.05 * len(panels), 3.4), constrained_layout=True
    )
    for ax, (title, panel) in zip(axes, panels):
        ax.imshow(np.rot90(panel))
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    dice_method = float_value(row, "Dice_TC_method")
    dice_base = float_value(row, "Dice_TC_baseline")
    hd_method = float_value(row, "HD95_TC_method")
    hd_base = float_value(row, "HD95_TC_baseline")
    hd_imp = float_value(row, "HD95_TC_improvement")
    fig.suptitle(
        f"{case_name} | TC Dice {args.method_name}/{args.baseline_name}: "
        f"{dice_method:.3f}/{dice_base:.3f} | TC HD95: {hd_method:.2f}/{hd_base:.2f} "
        f"(imp {hd_imp:.2f})",
        fontsize=11,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case_name}_tc_axis{args.slice_axis}_slice{idx}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[Saved] {out_path}")
    return str(out_path)


def write_selected_csv(out_dir, selected, by_case):
    path = Path(out_dir) / "selected_tc_hd95_cases.csv"
    keys = [
        "case", "HD95_TC_method", "HD95_TC_baseline", "HD95_TC_improvement",
        "Dice_TC_method", "Dice_TC_baseline", "Dice_TC_raw_delta",
        "Dice_Mean_method", "Dice_Mean_baseline", "Dice_Mean_raw_delta",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for case in selected:
            row = by_case[case]
            writer.writerow({key: row.get(key, "") for key in keys})
    print(f"[Saved] {path}")


def main():
    args = parse_args()
    selected, paired_by_case = select_cases(args)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("[Selected cases]")
    for case in selected:
        row = paired_by_case[case]
        print(
            f"  - {case}: HD95_TC imp={float_value(row, 'HD95_TC_improvement'):.4f}, "
            f"Dice_TC delta={float_value(row, 'Dice_TC_raw_delta'):.4f}"
        )
    write_selected_csv(args.out_dir, selected, paired_by_case)
    if args.dry_run:
        return

    test_case_map = case_map(args.data_root, args.seed, args.test_ratio)
    missing = [case for case in selected if case not in test_case_map]
    if missing:
        fallback_map = all_case_map(args.data_root)
        recovered = [case for case in missing if case in fallback_map]
        if recovered:
            print(
                "[Warning] Some selected cases were not found in the reconstructed held-out split; "
                "falling back to lookup by case name under data_root: "
                + ", ".join(recovered)
            )
            for case in recovered:
                test_case_map[case] = fallback_map[case]
        missing = [case for case in selected if case not in test_case_map]
    if missing:
        raise RuntimeError(
            f"Selected cases not found under data_root={args.data_root}: {missing}"
        )

    baseline = build_model(
        args.baseline_model, args.baseline_checkpoint, args.baseline_base_channels, args
    )
    method = build_model(
        args.method_model, args.method_checkpoint, args.method_base_channels, args
    )

    from datasets.brats_multimodal_dataset import BraTSMultiModalDataset

    saved = []
    for case in selected:
        ds = BraTSMultiModalDataset([test_case_map[case]], augment=False, test_mode=True)
        volume, target = ds[0]
        volume = volume.unsqueeze(0)
        target_np = target.numpy()

        baseline_logits = infer(baseline, volume, args)
        method_logits = infer(method, volume, args)
        baseline_mask = logits_to_label_mask(baseline_logits, postprocess=not args.no_postprocess)
        method_mask = logits_to_label_mask(method_logits, postprocess=not args.no_postprocess)
        image_np = volume[0, args.modality].numpy()
        saved.append(
            save_case_figure(
                case,
                image_np,
                target_np,
                baseline_mask,
                method_mask,
                paired_by_case[case],
                args,
            )
        )

    manifest = {
        "selected_cases": selected,
        "figures": saved,
        "paired_csv": args.paired_csv,
        "method_model": args.method_model,
        "method_checkpoint": args.method_checkpoint,
        "baseline_model": args.baseline_model,
        "baseline_checkpoint": args.baseline_checkpoint,
    }
    manifest_path = Path(args.out_dir) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[Saved] {manifest_path}")


if __name__ == "__main__":
    main()
