import argparse
import csv
import json
from pathlib import Path


DEFAULT_METRICS = ("Dice_Mean", "Dice_TC", "HD95_Mean", "HD95_TC", "NSD_Mean")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export paired comparison JSON files as compact paper-ready tables."
    )
    parser.add_argument("paired_json", nargs="+", help="JSON outputs from paired_case_comparison.py")
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--csv_out", default="results/paired_summary_table.csv")
    parser.add_argument("--md_out", default="results/paired_summary_table.md")
    parser.add_argument("--digits", type=int, default=4)
    return parser.parse_args()


def fmt(value, digits=4):
    if value is None:
        return "NA"
    value = float(value)
    if abs(value) < 10 ** (-(digits + 1)):
        return f"{value:.1e}"
    return f"{value:.{digits}f}"


def fmt_p(value):
    if value is None:
        return "NA"
    value = float(value)
    if value < 1e-4:
        return "<1e-4"
    if value < 0.001:
        return f"{value:.1e}"
    return f"{value:.4f}"


def read_rows(path, metrics, digits):
    data = json.loads(Path(path).read_text())
    comparison = f"{data.get('method_name', 'method')} vs {data.get('baseline_name', 'baseline')}"
    rows = []
    for metric in metrics:
        item = data.get("metrics", {}).get(metric)
        if not item:
            continue
        imp = item["improvement_positive_means_method_better"]
        rows.append({
            "comparison": comparison,
            "n": data.get("n_common_cases", item.get("n", "")),
            "metric": metric,
            "method_mean": fmt(item["method"]["mean"], digits),
            "baseline_mean": fmt(item["baseline"]["mean"], digits),
            "improvement_mean": fmt(imp["mean"], digits),
            "improvement_median": fmt(imp["median"], digits),
            "iqr": f"{fmt(imp['q1'], digits)} to {fmt(imp['q3'], digits)}",
            "w_t_l": f"{item['wins']}/{item['ties']}/{item['losses']}",
            "p_value": fmt_p(item.get("wilcoxon_one_sided_p_improvement_gt_0")),
        })
    return rows


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "comparison", "n", "metric", "method_mean", "baseline_mean",
        "improvement_mean", "improvement_median", "iqr", "w_t_l", "p_value",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Comparison", "n", "Metric", "Method", "Baseline",
        "Mean Imp.", "Median Imp.", "IQR", "W/T/L", "p",
    ]
    keys = [
        "comparison", "n", "metric", "method_mean", "baseline_mean",
        "improvement_mean", "improvement_median", "iqr", "w_t_l", "p_value",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]) for key in keys) + " |")
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    rows = []
    for path in args.paired_json:
        rows.extend(read_rows(path, args.metrics, args.digits))

    write_csv(args.csv_out, rows)
    write_markdown(args.md_out, rows)
    print(f"[Saved] {args.csv_out}")
    print(f"[Saved] {args.md_out}")


if __name__ == "__main__":
    main()
