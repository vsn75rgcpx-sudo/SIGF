#!/usr/bin/env python3
"""Bind one run to the current public code, metadata and visible GPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path


TRAINERS = {
    "full_molg": "nnUNetTrainerParallelModalSwinUNetB24",
    "no_maf": "nnUNetTrainerParallelModalSwinUNetB24NoMAF",
    "no_swin": "nnUNetTrainerParallelModalSwinUNetB24NoSwin",
    "concat_fusion": "nnUNetTrainerParallelModalSwinUNetB24ConcatFusion",
    "proposed_loss_ds_matched": "nnUNetTrainerParallelModalSwinUNetB24ProposedLossDSMatchedSeed42",
}
MODELS = {
    "full_molg": "ParallelModalSwinUNet",
    "no_maf": "ParallelSwinUNetNoMAF",
    "no_swin": "ParallelModalUNetNoSwin",
    "concat_fusion": "ParallelModalSwinUNetConcatFusion",
    "proposed_loss_ds_matched": "ParallelModalSwinUNet",
}
PROFILE_VERSIONS = {
    "blackwell5090-cu128": {"torch": "2.7.1+cu128", "torchvision": "0.22.1+cu128", "triton": "3.3.1"},
    "ada4090-cu118": {"torch": "2.7.1+cu118", "torchvision": "0.22.1+cu118", "triton": "3.3.1"},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def absolute(path: str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(path)))


def assert_separate(a: Path, b: Path) -> None:
    if a == b or a in b.parents or b in a.parents:
        raise RuntimeError("data-root and persistent-root must be separate and non-nested")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=sorted(TRAINERS), required=True)
    parser.add_argument("--profile", choices=sorted(PROFILE_VERSIONS), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--persistent-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--augmentation-workers", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data_root = absolute(args.data_root)
    persistent_root = absolute(args.persistent_root)
    assert_separate(data_root, persistent_root)
    if args.augmentation_workers < 1:
        raise RuntimeError("augmentation-workers must be positive")
    for path in (data_root, persistent_root):
        if not path.is_dir():
            raise RuntimeError(f"missing root: {path}")
    config_path = root / "config/experiments.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = {}
    for name in ("dataset.json", "nnUNetPlans.json", "dataset_fingerprint.json"):
        path = root / "frozen" / name
        if not path.is_file():
            raise RuntimeError(f"missing frozen metadata: {path}")
        metadata[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    status = "INSPECT_ONLY"
    gpu = None
    imported = {}
    if args.execute:
        if platform.system() != "Linux":
            raise RuntimeError("GPU preflight requires Linux")
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("preflight requires exactly one visible CUDA device")
        package_versions = {
            "torch": torch.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
            "triton": importlib.metadata.version("triton"),
            "nnunetv2": importlib.metadata.version("nnunetv2"),
            "dynamic-network-architectures": importlib.metadata.version("dynamic-network-architectures"),
            "timm": importlib.metadata.version("timm"),
        }
        for name, expected in PROFILE_VERSIONS[args.profile].items():
            if package_versions[name] != expected:
                raise RuntimeError(f"{name}={package_versions[name]!r}, expected {expected!r}")
        for name, expected in {"nnunetv2": "2.8.0", "dynamic-network-architectures": "0.4.4", "timm": "1.0.22"}.items():
            if package_versions[name] != expected:
                raise RuntimeError(f"{name}={package_versions[name]!r}, expected {expected!r}")
        gpu = {"name": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0))}
        trainer_module = importlib.import_module("nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerModalSwinUNet")
        model_module = importlib.import_module("nnunetv2.training.nnUNetTrainer.variants.network_architecture.modal_swin_unet")
        trainer = getattr(trainer_module, TRAINERS[args.experiment])
        model = getattr(model_module, MODELS[args.experiment])
        trainer_path = Path(trainer_module.__file__)
        model_path = Path(model_module.__file__)
        source_trainer = root / "runtime/nnunetv2_custom/nnUNetTrainerModalSwinUNet.py"
        source_model = root / "runtime/nnunetv2_custom/modal_swin_unet.py"
        if sha256(trainer_path) != sha256(source_trainer) or sha256(model_path) != sha256(source_model):
            raise RuntimeError("the imported trainer/model files differ from this release")
        imported = {
            "trainer": f"{trainer.__module__}.{trainer.__name__}",
            "model": f"{model.__module__}.{model.__name__}",
            "trainer_file": str(trainer_path),
            "trainer_sha256": sha256(trainer_path),
            "model_file": str(model_path),
            "model_sha256": sha256(model_path),
            "package_versions": package_versions,
        }
        status = "PASS_BINDING_GPU_VISIBLE"
    report = {
        "status": status,
        "experiment": args.experiment,
        "trainer": TRAINERS[args.experiment],
        "profile": args.profile,
        "dataset": "Dataset901_BraTS2023_TCRegion",
        "configuration": "3d_fullres",
        "plans": "nnUNetPlans",
        "fold": "all",
        "scientific_contract": config["scientific_contract"],
        "augmentation_workers": args.augmentation_workers,
        "validation_workers_derived": max(1, args.augmentation_workers // 2),
        "library_threads": 1,
        "data_root": str(data_root),
        "persistent_root": str(persistent_root),
        "metadata": metadata,
        "config_sha256": sha256(config_path),
        "source_code_sha256": {
            "trainer": sha256(root / "runtime/nnunetv2_custom/nnUNetTrainerModalSwinUNet.py"),
            "model": sha256(root / "runtime/nnunetv2_custom/modal_swin_unet.py"),
        },
        "imported": imported,
        "gpu": gpu,
        "note": "Binding record; a real training run must still validate its data loaders and finite forward/backward step.",
    }
    out = absolute(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
