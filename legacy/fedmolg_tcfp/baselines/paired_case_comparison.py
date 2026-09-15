import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_METRICS = [
    "Dice_Mean", "Dice_WT", "Dice_TC", "Dice_ET",
    "HD95_Mean", "HD95_WT", "HD95_TC", "HD95_ET",
    "NSD_Mean", "NSD_WT", "NSD_TC", "NSD_ET",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paired case-level comparison between two BraTS experiments"
    )
    parser.add_argument("--method", required=True, help="case_metrics.json for the proposed method")
    parser.add_argument("--baseline", required=True, help="case_metrics.json for the baseline")
    parser.add_argument("--method_name", default="method")
    parser.add_argument("--baseline_name", default="baseline")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--tie_tol", type=float, default=1e-8)
    parser.add_argument("--out", default="", help="Optional JSON output path")
    parser.add_argument("--csv_out", default="", help="Optional per-case CSV output path")
    return parser.parse_args()


def load_by_case(path):
    path = Path(path)
    if not path.exists():
        available = sorted(Path("results").glob("*/case_metrics.json")) if Path("results").exists() else []
        hint = ""
        if available:
            shown = "\n".join(f"  - {p}" for p in available[:30])
            more = f"\n  ... and {len(available) - 30} more" if len(available) > 30 else ""
            hint = f"\nAvailable case_metrics.json files:\n{shown}{more}"
        raise FileNotFoundError(f"Missing case metrics file: {path}{hint}")

    data = json.loads(path.read_text())
    by_case = {}
    for item in data:
        case = item.get("case")
        if case is None:
            raise ValueError(f"Missing 'case' field in {path}")
        by_case[case] = item
    return by_case


def higher_is_better(metric):
    name = metric.lower()
    if name.startswith("hd95") or "hausdorff" in name:
        return False
    return True


def finite_values(rows, metric):
    values = []
    for method_item, baseline_item in rows:
        m_val = method_item.get(metric, -1)
        b_val = baseline_item.get(metric, -1)
        if m_val == -1 or b_val == -1:
            continue
        if not (np.isfinite(m_val) and np.isfinite(b_val)):
            continue
        values.append((float(m_val), float(b_val)))
    return values


def percentile_summary(vals):
    vals = np.asarray(vals, dtype=np.float64)
    q1, median, q3 = np.percentile(vals, [25, 50, 75])
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
    }


def wilcoxon_pvalue(improvement):
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None

    improvement = np.asarray(improvement, dtype=np.float64)
    nonzero = improvement[np.abs(improvement) > 0]
    if nonzero.size == 0:
        return None
    try:
        return float(wilcoxon(nonzero, alternative="greater", zero_method="wilcox").pvalue)
    except ValueError:
        return None


def summarize_metric(metric, rows, tie_tol):
    pairs = finite_values(rows, metric)
    if not pairs:
        return None, []

    method_vals = np.array([p[0] for p in pairs], dtype=np.float64)
    baseline_vals = np.array([p[1] for p in pairs], dtype=np.float64)
    raw_delta = method_vals - baseline_vals
    if higher_is_better(metric):
        improvement = raw_delta
    else:
        improvement = baseline_vals - method_vals

    wins = int(np.sum(improvement > tie_tol))
    ties = int(np.sum(np.abs(improvement) <= tie_tol))
    losses = int(np.sum(improvement < -tie_tol))
    baseline_nonzero = np.where(np.abs(baseline_vals) > 1e-12, np.abs(baseline_vals), np.nan)
    relative = improvement / baseline_nonzero * 100.0
    relative = relative[np.isfinite(relative)]

    summary = {
        "n": int(len(pairs)),
        "direction": "higher_better" if higher_is_better(metric) else "lower_better",
        "method": percentile_summary(method_vals),
        "baseline": percentile_summary(baseline_vals),
        "raw_delta_method_minus_baseline": percentile_summary(raw_delta),
        "improvement_positive_means_method_better": percentile_summary(improvement),
        "relative_improvement_percent": percentile_summary(relative) if relative.size > 0 else None,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": float(wins / len(pairs)),
        "wilcoxon_one_sided_p_improvement_gt_0": wilcoxon_pvalue(improvement),
    }
    return summary, list(zip(method_vals, baseline_vals, raw_delta, improvement))


def write_case_csv(path, cases, common_rows, metrics, paired_values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case"]
    for metric in metrics:
        if metric in paired_values:
            fieldnames.extend([
                f"{metric}_method",
                f"{metric}_baseline",
                f"{metric}_raw_delta",
                f"{metric}_improvement",
            ])

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case, (method_item, baseline_item) in zip(cases, common_rows):
            row = {"case": case}
            for metric in metrics:
                if metric not in paired_values:
                    continue
                m_val = method_item.get(metric, -1)
                b_val = baseline_item.get(metric, -1)
                if m_val == -1 or b_val == -1:
                    continue
                raw = float(m_val) - float(b_val)
                improvement = raw if higher_is_better(metric) else -raw
                row[f"{metric}_method"] = float(m_val)
                row[f"{metric}_baseline"] = float(b_val)
                row[f"{metric}_raw_delta"] = raw
                row[f"{metric}_improvement"] = improvement
            writer.writerow(row)


def main():
    args = parse_args()
    method = load_by_case(args.method)
    baseline = load_by_case(args.baseline)
    common_cases = sorted(set(method) & set(baseline))
    if not common_cases:
        raise ValueError("No overlapping case names found between the two case_metrics files.")

    missing_method = sorted(set(baseline) - set(method))
    missing_baseline = sorted(set(method) - set(baseline))
    common_rows = [(method[case], baseline[case]) for case in common_cases]

    output = {
        "method_name": args.method_name,
        "baseline_name": args.baseline_name,
        "method_path": args.method,
        "baseline_path": args.baseline,
        "n_common_cases": len(common_cases),
        "missing_in_method": missing_method,
        "missing_in_baseline": missing_baseline,
        "metrics": {},
    }

    paired_values = {}
    print(f"Common cases: {len(common_cases)}")
    if missing_method:
        print(f"Missing in method: {len(missing_method)}")
    if missing_baseline:
        print(f"Missing in baseline: {len(missing_baseline)}")

    for metric in args.metrics:
        summary, values = summarize_metric(metric, common_rows, args.tie_tol)
        if summary is None:
            continue
        output["metrics"][metric] = summary
        paired_values[metric] = values
        method_mean = summary["method"]["mean"]
        baseline_mean = summary["baseline"]["mean"]
        improvement_mean = summary["improvement_positive_means_method_better"]["mean"]
        improvement_median = summary["improvement_positive_means_method_better"]["median"]
        p_value = summary["wilcoxon_one_sided_p_improvement_gt_0"]
        p_text = "NA" if p_value is None else f"{p_value:.4g}"
        print(
            f"{metric}: {args.method_name}={method_mean:.4f}, "
            f"{args.baseline_name}={baseline_mean:.4f}, "
            f"improvement mean={improvement_mean:.4f}, median={improvement_median:.4f}, "
            f"W/T/L={summary['wins']}/{summary['ties']}/{summary['losses']}, "
            f"Wilcoxon p={p_text}"
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(output, indent=2))
    if args.csv_out:
        write_case_csv(args.csv_out, common_cases, common_rows, args.metrics, paired_values)


if __name__ == "__main__":
    main()
