#!/usr/bin/env python3
"""Run only frozen-plan preprocessing on an authorized Linux host.

The planner is intentionally absent from this entry point. The input
fingerprint, dataset metadata and plans are copied only when absent or byte
identical, then nnU-Net's ``preprocess_dataset`` consumes the frozen
``nnUNetPlans`` identifier. No new split or planning decision is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path


EXPECTED_PLANS = "3521cc8e38c287b6cdc7b5e8af708bf01aecafc25c875fa8ee64a4534a0c3ec1"
EXPECTED_DATASET = "3f9dbc578af7f0cce580c97f3f02101c7355634dd4659a1d59ac94aaa8b7fa7c"
EXPECTED_FINGERPRINT = "b3969080df7acba4afd610c4c3d0da604260b0b34015818689e01552809b562a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_copy(source: Path, target: Path, expected_sha: str) -> None:
    if sha256_file(source) != expected_sha:
        raise RuntimeError(f"frozen source hash mismatch: {source}")
    if target.exists() or target.is_symlink():
        if not target.is_file() or sha256_file(target) != expected_sha:
            raise RuntimeError(f"refusing to overwrite different preprocessed metadata: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--persistent-root", default="")
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = Path(args.package_root).resolve()
    data_root = Path(args.data_root).resolve()
    persistent_root = Path(args.persistent_root).resolve() if args.persistent_root else None
    raw_dataset = data_root / "nnUNet_raw" / "Dataset901_BraTS2023_TCRegion"
    preprocessed_dataset = data_root / "nnUNet_preprocessed" / "Dataset901_BraTS2023_TCRegion"
    frozen = root / "frozen"
    source_paths = {
        "plans": (frozen / "nnUNetPlans.json", EXPECTED_PLANS),
        "dataset": (frozen / "dataset.json", EXPECTED_DATASET),
        "fingerprint": (frozen / "dataset_fingerprint.json", EXPECTED_FINGERPRINT),
    }
    if args.processes < 1:
        raise RuntimeError("--processes must be positive")
    if args.processes > 2:
        raise RuntimeError("preprocessing is capped at two processes for bounded CPU use")
    if args.execute and persistent_root is None:
        raise RuntimeError("--persistent-root is required for executable preprocessing")
    if persistent_root is not None and (persistent_root == data_root or
                                        str(persistent_root).startswith(str(data_root) + os.sep) or
                                        str(data_root).startswith(str(persistent_root) + os.sep)):
        raise RuntimeError("data-root and persistent-root must be separate, non-nested directories")
    for path, expected in source_paths.values():
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"missing or altered frozen input: {path}")
    required_raw = [raw_dataset / "imagesTr", raw_dataset / "labelsTr", raw_dataset / "imagesTs", raw_dataset / "labelsTs"]
    if not args.execute:
        print(json.dumps({
            "status": "INSPECT_ONLY",
            "linux_required": True,
            "data_root": str(data_root),
            "persistent_root": str(persistent_root) if persistent_root else None,
            "raw_dataset": str(raw_dataset),
            "preprocessed_dataset": str(preprocessed_dataset),
            "processes": args.processes,
            "would_call": "nnunetv2.experiment_planning.plan_and_preprocess_api.preprocess_dataset",
            "would_use": {"dataset_id": 901, "plans_identifier": "nnUNetPlans", "configuration": "3d_fullres"},
            "planner": "not called",
            "source_hashes": {key: value[1] for key, value in source_paths.items()},
        }, indent=2, sort_keys=True))
        return 0
    if platform.system() != "Linux":
        raise RuntimeError("frozen preprocessing requires Linux; refusing a non-Linux execution")
    if not all(path.is_dir() for path in required_raw):
        raise RuntimeError(f"raw nnU-Net layout is incomplete under {raw_dataset}")
    allowed_metadata = {"dataset.json", "nnUNetPlans.json", "dataset_fingerprint.json"}
    existing_payloads = [path.name for path in preprocessed_dataset.iterdir()
                         if path.name not in allowed_metadata and not path.name.endswith(".partial")]
    if existing_payloads:
        raise RuntimeError(f"preprocessed payloads already exist; refusing overwrite: {existing_payloads[:10]}")
    protected_copy(source_paths["dataset"][0], preprocessed_dataset / "dataset.json", EXPECTED_DATASET)
    protected_copy(source_paths["plans"][0], preprocessed_dataset / "nnUNetPlans.json", EXPECTED_PLANS)
    protected_copy(source_paths["fingerprint"][0], preprocessed_dataset / "dataset_fingerprint.json", EXPECTED_FINGERPRINT)
    os.environ["nnUNet_raw"] = str(data_root / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(data_root / "nnUNet_preprocessed")
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    started = time.time()
    from nnunetv2.experiment_planning.plan_and_preprocess_api import preprocess_dataset

    preprocess_dataset(
        901,
        plans_identifier="nnUNetPlans",
        configurations=("3d_fullres",),
        num_processes=(args.processes,),
        verbose=True,
        show_progress_bar=False,
    )
    output = {
        "status": "PREPROCESS_COMPLETE",
        "dataset_id": 901,
        "configuration": "3d_fullres",
        "plans_identifier": "nnUNetPlans",
        "processes": args.processes,
        "elapsed_seconds": time.time() - started,
        "raw_dataset": str(raw_dataset),
        "preprocessed_dataset": str(preprocessed_dataset),
        "persistent_root": str(persistent_root),
        "frozen_hashes": {key: value[1] for key, value in source_paths.items()},
        "planner_called": False,
        "created_by": sys.executable,
    }
    status_path = persistent_root / "preprocess_runs" / f"preprocess_{int(time.time())}_{os.getpid()}.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    output["persistent_record"] = str(status_path)
    record = preprocessed_dataset / "frozen_preprocess_provenance.json"
    record_text = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if record.exists() and record.read_text(encoding="utf-8") != record_text:
        raise RuntimeError(f"refusing to overwrite different preprocessing provenance: {record}")
    if not record.exists():
        record.write_text(record_text, encoding="utf-8")
    status_path.write_text(record_text, encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
