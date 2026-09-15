import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial import cKDTree

try:
    import nibabel as nib
except ImportError:
    nib = None

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HD95_POSTPROCESS = {
    "min_size_wt": 300,
    "min_size_tc": 50,
    "min_size_et": 30,
}


def load_label(path, map_label4_to3=True):
    if sitk is not None:
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.int16)
    elif nib is not None:
        arr = nib.load(str(path)).get_fdata().astype(np.int16)
    else:
        raise ImportError("Need either SimpleITK or nibabel to read NIfTI labels.")
    if map_label4_to3:
        arr[arr == 4] = 3
    return arr


def get_brats_regions(mask):
    wt = (mask > 0).astype(np.float32)
    tc = ((mask == 1) | (mask == 3)).astype(np.float32)
    et = (mask == 3).astype(np.float32)
    return wt, tc, et


def dice(pred, target):
    eps = 1e-5
    return float((2 * (pred * target).sum() + eps) / (pred.sum() + target.sum() + eps))


def surface(mask):
    struct = generate_binary_structure(3, 1)
    eroded = binary_erosion(mask.astype(bool), structure=struct)
    return mask.astype(bool) ^ eroded


def hd95(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return 374.0
    pred_coords = np.argwhere(surface(pred))
    target_coords = np.argwhere(surface(target))
    if len(pred_coords) == 0 or len(target_coords) == 0:
        return 374.0
    pred_tree = cKDTree(pred_coords)
    target_tree = cKDTree(target_coords)
    d_p_to_t, _ = target_tree.query(pred_coords)
    d_t_to_p, _ = pred_tree.query(target_coords)
    return float(max(np.percentile(d_p_to_t, 95), np.percentile(d_t_to_p, 95)))


def nsd(pred, target, tolerance=1.0):
    pred = pred.astype(bool)
    target = target.astype(bool)
    if not pred.any() and not target.any():
        return 1.0
    if not pred.any() or not target.any():
        return 0.0
    pred_coords = np.argwhere(surface(pred))
    target_coords = np.argwhere(surface(target))
    if len(pred_coords) == 0 or len(target_coords) == 0:
        return 0.0
    pred_tree = cKDTree(pred_coords)
    target_tree = cKDTree(target_coords)
    d_p_to_t, _ = target_tree.query(pred_coords)
    d_t_to_p, _ = pred_tree.query(target_coords)
    return float(((d_p_to_t <= tolerance).sum() + (d_t_to_p <= tolerance).sum()) / (len(d_p_to_t) + len(d_t_to_p)))


def keep_components_above_size(mask, min_size):
    mask = mask.astype(bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    keep_ids = [idx + 1 for idx, size in enumerate(sizes) if size >= min_size]
    if not keep_ids:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labeled, keep_ids)


def postprocess_hierarchical(pred_mask):
    wt_mask = keep_components_above_size(pred_mask > 0, HD95_POSTPROCESS["min_size_wt"])
    tc_mask = keep_components_above_size(((pred_mask == 1) | (pred_mask == 3)) & wt_mask, HD95_POSTPROCESS["min_size_tc"])
    et_mask = keep_components_above_size((pred_mask == 3) & tc_mask, HD95_POSTPROCESS["min_size_et"])
    tc_mask = tc_mask | et_mask
    wt_mask = wt_mask | tc_mask
    cleaned = np.zeros_like(pred_mask, dtype=np.int16)
    cleaned[wt_mask] = 2
    cleaned[tc_mask] = 1
    cleaned[et_mask] = 3
    return cleaned


def calculate_metrics_from_mask(pred_mask, target_mask, compute_hd95=True, apply_postprocess=True):
    if apply_postprocess:
        pred_mask = postprocess_hierarchical(pred_mask)
    p_wt, p_tc, p_et = get_brats_regions(pred_mask)
    t_wt, t_tc, t_et = get_brats_regions(target_mask)
    out = {
        "Dice_WT": dice(p_wt, t_wt),
        "Dice_TC": dice(p_tc, t_tc),
        "Dice_ET": dice(p_et, t_et),
        "NSD_WT": nsd(p_wt, t_wt),
        "NSD_TC": nsd(p_tc, t_tc),
        "NSD_ET": nsd(p_et, t_et),
    }
    if compute_hd95:
        out.update({
            "HD95_WT": hd95(p_wt, t_wt),
            "HD95_TC": hd95(p_tc, t_tc),
            "HD95_ET": hd95(p_et, t_et),
        })
        out["HD95_Mean"] = (out["HD95_WT"] + out["HD95_TC"] + out["HD95_ET"]) / 3.0
    out["Dice_Mean"] = (out["Dice_WT"] + out["Dice_TC"] + out["Dice_ET"]) / 3.0
    out["NSD_Mean"] = (out["NSD_WT"] + out["NSD_TC"] + out["NSD_ET"]) / 3.0
    return out


def summarize_case_metrics(case_metrics):
    keys = [k for k in case_metrics[0].keys() if k != "case"]
    summary = {}
    for key in keys:
        vals = np.array([m[key] for m in case_metrics], dtype=np.float64)
        summary[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction NIfTI folder with the paper BraTS metrics.")
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv_out", default="")
    parser.add_argument("--keep_label4", action="store_true")
    parser.add_argument(
        "--disable_postprocess",
        action="store_true",
        help="Evaluate raw predictions without hierarchical connected-component filtering.",
    )
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    label_dir = Path(args.label_dir)
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory does not exist: {pred_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory does not exist: {label_dir}")

    case_metrics = []
    labels = sorted(label_dir.glob("*.nii.gz"))
    if not labels:
        raise FileNotFoundError(f"No .nii.gz labels found in {label_dir}")

    for label_path in labels:
        case = label_path.name[:-7]
        pred_path = pred_dir / label_path.name
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction for {case}: {pred_path}")

        pred = load_label(pred_path, map_label4_to3=not args.keep_label4)
        target = load_label(label_path, map_label4_to3=not args.keep_label4)
        metrics = calculate_metrics_from_mask(
            pred,
            target,
            compute_hd95=True,
            apply_postprocess=not args.disable_postprocess,
        )
        metrics["case"] = case
        case_metrics.append(metrics)
        print(f"[Eval] {case}: Dice_Mean={metrics['Dice_Mean']:.4f}, HD95_Mean={metrics['HD95_Mean']:.4f}")

    summary = summarize_case_metrics(case_metrics)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "postprocess": not args.disable_postprocess,
            "postprocess_config": None if args.disable_postprocess else HD95_POSTPROCESS,
            "summary": summary,
            "cases": case_metrics,
        }, indent=2),
        encoding="utf-8",
    )

    if args.csv_out:
        csv_out = Path(args.csv_out)
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        keys = ["case"] + [k for k in case_metrics[0].keys() if k != "case"]
        with csv_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(case_metrics)

    print("========== PREDICTION FOLDER SUMMARY ==========")
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"{key}: {value['mean']:.4f} ± {value['std']:.4f}")
        else:
            print(f"{key}: {value}")
    print(f"[Saved] {out}")


if __name__ == "__main__":
    main()
