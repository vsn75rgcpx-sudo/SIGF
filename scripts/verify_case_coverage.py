#!/usr/bin/env python3
"""Check exact filename coverage for one authorized evaluation group."""

from __future__ import annotations

import argparse
from pathlib import Path


def cases(directory: Path) -> set[str]:
    return {path.name[:-7] for path in directory.glob("*.nii.gz")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--expected-count", type=int, choices=(100, 375), required=True)
    args = parser.parse_args()
    predictions = cases(Path(args.prediction_dir))
    labels = cases(Path(args.label_dir))
    if len(labels) != args.expected_count:
        raise RuntimeError(f"label count {len(labels)} != expected {args.expected_count}")
    if predictions != labels:
        raise RuntimeError(
            f"case coverage mismatch: missing_predictions={sorted(labels - predictions)[:10]} "
            f"unexpected_predictions={sorted(predictions - labels)[:10]}"
        )
    print(f"PASS case_count={len(labels)}")


if __name__ == "__main__":
    main()
