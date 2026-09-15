import argparse
import json
from pathlib import Path


KEY_METRICS = ("Dice_Mean", "HD95_Mean", "NSD_Mean")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize paired comparison JSON files")
    parser.add_argument("paired_json", nargs="+", help="paired_case_comparison.py JSON outputs")
    return parser.parse_args()


def fmt(value, ndigits=4):
    if value is None:
        return "NA"
    return f"{float(value):.{ndigits}g}"


def summarize(path):
    data = json.loads(Path(path).read_text())
    method = data.get("method_name", "method")
    baseline = data.get("baseline_name", "baseline")
    n_cases = data.get("n_common_cases", 0)

    print(f"\n== {Path(path).name} ==")
    print(f"{method} vs {baseline} | n={n_cases}")
    for metric in KEY_METRICS:
        item = data.get("metrics", {}).get(metric)
        if not item:
            continue
        method_mean = item["method"]["mean"]
        baseline_mean = item["baseline"]["mean"]
        imp = item["improvement_positive_means_method_better"]
        p_value = item.get("wilcoxon_one_sided_p_improvement_gt_0")
        print(
            f"{metric}: method={method_mean:.4f}, baseline={baseline_mean:.4f}, "
            f"imp_mean={imp['mean']:.4f}, imp_median={imp['median']:.4f}, "
            f"W/T/L={item['wins']}/{item['ties']}/{item['losses']}, "
            f"p={fmt(p_value)}"
        )


def main():
    args = parse_args()
    for path in args.paired_json:
        summarize(path)


if __name__ == "__main__":
    main()
