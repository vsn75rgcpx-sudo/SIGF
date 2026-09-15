#!/usr/bin/env python3
"""Install the frozen custom trainer into an nnU-Net v2 environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import shutil
from pathlib import Path


FILES = ("modal_swin_unet.py", "nnUNetTrainerModalSwinUNet.py")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_nnunet_root() -> Path:
    module = importlib.import_module("nnunetv2")
    return Path(module.__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnunetv2-root", default="")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "runtime" / "nnunetv2_custom"
    package_root = Path(args.nnunetv2_root) if args.nnunetv2_root else detect_nnunet_root()
    target = package_root / "training" / "nnUNetTrainer" / "variants" / "network_architecture"
    print(f"source={source}")
    print(f"target={target}")
    for name in FILES:
        src = source / name
        dst = target / name
        if not src.is_file():
            raise FileNotFoundError(src)
        print(f"{name} source_sha256={digest(src)} target_present={dst.exists()}")
        if dst.exists() and digest(dst) != digest(src) and not args.overwrite:
            raise RuntimeError(f"target differs; pass --overwrite only after reviewing it: {dst}")
    if not args.apply:
        print("INSPECT_ONLY: no installed package was changed.")
        return
    target.mkdir(parents=True, exist_ok=True)
    init_file = target / "__init__.py"
    if not init_file.exists():
        init_file.write_text("", encoding="utf-8")
    for name in FILES:
        shutil.copy2(source / name, target / name)
        print(f"installed={target / name}")
    print("CUSTOM_TRAINER_INSTALLED")


if __name__ == "__main__":
    main()
