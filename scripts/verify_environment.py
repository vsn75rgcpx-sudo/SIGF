#!/usr/bin/env python3
"""Verify packages and the files of the trainer actually imported by Python."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


EXPECTED = {
    "blackwell5090-cu128": {"torch": "2.7.1+cu128", "torchvision": "0.22.1+cu128", "triton": "3.3.1"},
    "ada4090-cu118": {"torch": "2.7.1+cu118", "torchvision": "0.22.1+cu118", "triton": "3.3.1"},
}
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(EXPECTED), default="")
    parser.add_argument("--experiment", choices=sorted(TRAINERS), action="append")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    selected = args.experiment or sorted(TRAINERS)

    import torch
    import torchvision
    import triton

    versions = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "sys_base_prefix": sys.base_prefix,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "triton": triton.__version__,
        "nnunetv2": importlib.metadata.version("nnunetv2"),
        "dynamic-network-architectures": importlib.metadata.version("dynamic-network-architectures"),
        "timm": importlib.metadata.version("timm"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
    }
    for name, expected in {"nnunetv2": "2.8.0", "dynamic-network-architectures": "0.4.4", "timm": "1.0.22"}.items():
        if versions[name] != expected:
            raise RuntimeError(f"{name}={versions[name]!r}, expected {expected!r}")
    if args.profile:
        for name, expected in EXPECTED[args.profile].items():
            if versions[name] != expected:
                raise RuntimeError(f"{name}={versions[name]!r}, expected {expected!r}")

    trainer_name = "nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerModalSwinUNet"
    model_name = "nnunetv2.training.nnUNetTrainer.variants.network_architecture.modal_swin_unet"
    trainer_module = importlib.import_module(trainer_name)
    model_module = importlib.import_module(model_name)
    source_trainer = root / "runtime/nnunetv2_custom/nnUNetTrainerModalSwinUNet.py"
    source_model = root / "runtime/nnunetv2_custom/modal_swin_unet.py"
    trainer_path = Path(trainer_module.__file__)
    model_path = Path(model_module.__file__)
    imported = {
        "trainer_module": {"path": str(trainer_path), "sha256": sha256(trainer_path), "source_sha256": sha256(source_trainer)},
        "model_module": {"path": str(model_path), "sha256": sha256(model_path), "source_sha256": sha256(source_model)},
        "trainers": {},
        "models": {},
    }
    if imported["trainer_module"]["sha256"] != imported["trainer_module"]["source_sha256"]:
        raise RuntimeError("the imported trainer differs from this release")
    if imported["model_module"]["sha256"] != imported["model_module"]["source_sha256"]:
        raise RuntimeError("the imported model differs from this release")
    for experiment in selected:
        trainer = getattr(trainer_module, TRAINERS[experiment], None)
        model = getattr(model_module, MODELS[experiment], None)
        if trainer is None or model is None:
            raise RuntimeError(f"missing mapped class for {experiment}")
        imported["trainers"][experiment] = f"{trainer.__module__}.{trainer.__name__}"
        imported["models"][experiment] = f"{model.__module__}.{model.__name__}"
    report = {"status": "PASS", "versions": versions, "imported": imported}
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
