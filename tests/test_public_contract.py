from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_config_contains_current_variants_and_scientific_contract():
    config = json.loads((ROOT / "config/experiments.json").read_text())
    assert config["project"]["name"] == "MoLG-UNet"
    assert config["scientific_contract"]["seed"] == 42
    assert config["scientific_contract"]["total_epochs"] == 1000
    assert config["scientific_contract"]["patch_size"] == [128, 160, 112]
    weights = config["formal_experiments"]["proposed_loss_ds_matched"]["deep_supervision_weights"]
    expected = [8 / 15, 4 / 15, 2 / 15, 1 / 15, 0.0]
    assert all(abs(actual - wanted) < 1e-15 for actual, wanted in zip(weights, expected))
    assert set(config["formal_experiments"]) == {"no_maf", "no_swin", "concat_fusion", "proposed_loss_ds_matched"}


def test_current_runtime_and_recipes_exist():
    expected = [
        "runtime/nnunetv2_custom/modal_swin_unet.py",
        "runtime/nnunetv2_custom/nnUNetTrainerModalSwinUNet.py",
        "runtime/model_swin3d/nas_ops.py",
        "runtime/baselines/prepare_data_lossless.py",
        "runtime/baselines/evaluate_prediction_folder.py",
        "requirements/constraints-common-runtime.txt",
        "requirements/constraints-blackwell5090-cu128.txt",
        "scripts/preprocess_frozen.py",
        "scripts/preprocess_frozen.sh",
    ]
    for relative in expected:
        assert (ROOT / relative).is_file(), relative
    common = (ROOT / "requirements/constraints-common-runtime.txt").read_text()
    assert "dynamic-network-architectures==0.4.4" in common
    assert "timm==1.0.22" in common
    assert "1.0.27" not in common


def test_public_tree_has_no_case_lists_or_scientific_artifacts():
    forbidden_suffixes = (".nii", ".nii.gz", ".npz", ".npy", ".pth", ".pt", ".zip")
    forbidden_names = {"membership_manifest.json", "case_metrics.csv"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        assert path.name not in forbidden_names, path
        assert not path.name.endswith(forbidden_suffixes), path


def test_public_text_has_no_private_runtime_identifiers():
    needles = (
        "file:///",
        "/Users/",
        "/home/",
        "/share/",
        "github" + "_pat_",
        "gh" + "p_",
        "BEGIN " + "OPENSSH PRIVATE KEY",
    )
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or ".git" in path.parts
            or path == Path(__file__)
            or path.suffix in {".png", ".jpg", ".pyc"}
        ):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for needle in needles:
            assert needle not in text, f"{needle} in {path}"
