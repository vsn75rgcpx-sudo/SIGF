from __future__ import annotations

import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_config_contains_current_variants_and_scientific_contract():
    config = json.loads((ROOT / "config/experiments.json").read_text())
    assert config["project"]["name"] == "SIGF"
    assert config["project"]["legacy_name"] == "MoLG-UNet"
    assert config["formal_experiments"]["full_molg"]["seed"] is None
    assert config["scientific_contract"]["total_epochs"] == 1000
    assert config["scientific_contract"]["patch_size"] == [128, 160, 112]
    weights = config["additional_experiments"]["proposed_loss_ds_matched"]["deep_supervision_weights"]
    expected = [8 / 15, 4 / 15, 2 / 15, 1 / 15, 0.0]
    assert all(abs(actual - wanted) < 1e-15 for actual, wanted in zip(weights, expected))
    assert set(config["formal_experiments"]) == {"full_molg", "no_maf", "no_swin", "concat_fusion"}
    assert set(config["additional_experiments"]) == {"proposed_loss_ds_matched"}


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


def test_citation_metadata_is_for_the_public_release():
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    assert 'version: "1.0.0"' in citation
    assert zenodo["version"] == "1.0.0"
    assert zenodo["upload_type"] == "software"
    assert zenodo["license"] == "MIT"
    assert "case lists" in zenodo["description"]


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
            or path == Path(__file__).resolve()
            or path.suffix in {".png", ".jpg", ".pyc"}
        ):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for needle in needles:
            assert needle not in text, f"{needle} in {path}"


def test_published_source_manifest_files_exist_and_match_hashes():
    for line in (ROOT / "provenance/source_code_sha256.txt").read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative
        assert path.is_file(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
