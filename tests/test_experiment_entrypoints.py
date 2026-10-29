"""Exercise public launch bindings without scientific packages or a GPU."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/train_one_experiment.sh"


def _config_experiments():
    config = json.loads((ROOT / "config/experiments.json").read_text())
    return config, {**config["formal_experiments"], **config["additional_experiments"]}


def _trainer_uses_seed_mixin(trainer):
    tree = ast.parse((ROOT / "runtime/nnunetv2_custom/nnUNetTrainerModalSwinUNet.py").read_text())
    bases = {
        node.name: [base.id for base in node.bases if isinstance(base, ast.Name)]
        for node in tree.body if isinstance(node, ast.ClassDef)
    }

    def inherits(name):
        return name == "SeededTrainerMixin" or any(inherits(base) for base in bases.get(name, []))

    return inherits(trainer)


def test_experiment_seed_policy_matches_real_trainer_inheritance():
    _, experiments = _config_experiments()
    for name, experiment in experiments.items():
        seeded = _trainer_uses_seed_mixin(experiment["trainer"])
        assert experiment["seed"] == (42 if seeded else None), name
        assert experiment["seed_policy"] == (
            "fixed_by_seeded_trainer" if seeded else "not_fixed_by_entrypoint"
        ), name


def test_inspect_launch_bindings_match_configuration_without_creating_runs():
    _, experiments = _config_experiments()
    with tempfile.TemporaryDirectory() as directory:
        parent = Path(directory)
        for name, experiment in experiments.items():
            run_root = parent / name
            result = subprocess.run([
                "bash", str(LAUNCHER), "--experiment", name,
                "--profile", "blackwell5090-cu128", "--env-prefix", str(parent / "env"),
                "--data-root", str(parent / "data"), "--persistent-root", str(parent / "persistent"),
                "--run-root", str(run_root), "--augmentation-workers", "3",
            ], capture_output=True, text=True, check=True)
            expected_seed = "unfixed" if experiment["seed"] is None else "42"
            assert f'trainer={experiment["trainer"]}' in result.stdout
            assert f'seed={expected_seed} seed_policy={experiment["seed_policy"]}' in result.stdout
            assert "validation_workers=1 " in result.stdout
            assert not run_root.exists()


def test_preflight_inspection_reports_per_experiment_seed_and_scope():
    config, experiments = _config_experiments()
    with tempfile.TemporaryDirectory() as directory:
        parent = Path(directory)
        (parent / "data").mkdir()
        (parent / "persistent").mkdir()
        for name, experiment in experiments.items():
            output = parent / f"{name}.json"
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/preflight.py"), "--experiment", name,
                "--profile", "blackwell5090-cu128", "--data-root", str(parent / "data"),
                "--persistent-root", str(parent / "persistent"), "--out", str(output),
            ], capture_output=True, text=True, check=True)
            report = json.loads(result.stdout)
            assert report == json.loads(output.read_text())
            assert report["status"] == "INSPECT_ONLY"
            assert report["trainer"] == experiment["trainer"]
            assert report["seed"] == experiment["seed"]
            assert report["seed_policy"] == experiment["seed_policy"]
            assert report["scientific_contract"]["seed"] == experiment["seed"]
            assert report["experiment_scope"] == (
                "formal_experiments" if name in config["formal_experiments"] else "additional_experiments"
            )
            assert report["gpu"] is None


def _write_executable(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(0o755)


def _mock_run(directory, experiment, legacy_identity=None):
    """Execute the real launcher with only GPU preflight/checkpoint/train mocked.

    The shell's actual metadata, command construction and resume checks run;
    no scientific trainer or checkpoint loader is invoked.
    """
    parent = Path(directory).resolve()
    env_prefix = parent / "env"
    fake_bin = parent / "bin"
    data_root = parent / "data"
    persistent_root = parent / "persistent"
    run_root = persistent_root / "run"
    data_root.mkdir()
    persistent_root.mkdir()
    fake_bin.mkdir()
    _write_executable(fake_bin / "uname", "#!/bin/sh\nprintf 'Linux\\n'\n")
    python_shim = f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
script = Path(sys.argv[1]).name if len(sys.argv) > 1 else ""
if script in {"preflight.py", "checkpoint_contract.py"}:
    output = Path(sys.argv[sys.argv.index("--out") + 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"status": "PASS_BINDING_GPU_VISIBLE", "mock": True}))
else:
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
'''
    _write_executable(env_prefix / "bin/python", python_shim)
    train_stub = f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
Path(os.environ["SIGF_TEST_TRAIN_CALL"]).write_text(json.dumps({
    "args": sys.argv[1:], "seed_environment": os.environ.get("NNUNET_RUN_SEED")
}))
'''
    _write_executable(env_prefix / "bin/nnUNetv2_train", train_stub)
    _, experiments = _config_experiments()
    trainer = experiments[experiment]["trainer"]
    identity_before = None
    if legacy_identity is not None:
        model_dir = run_root / "nnUNet_results" / experiment / "Dataset901_BraTS2023_TCRegion" / f"{trainer}__nnUNetPlans__3d_fullres" / "fold_all"
        model_dir.mkdir(parents=True)
        (model_dir / "checkpoint_latest.pth").write_bytes(b"mock checkpoint; never loaded")
        identity_before = json.dumps(legacy_identity, indent=2) + "\n"
        (run_root / "run_identity.json").write_text(identity_before)
    preflight = parent / "preflight.json"
    preflight.write_text(json.dumps({
        "status": "PASS_BINDING_GPU_VISIBLE", "experiment": experiment,
        "profile": "blackwell5090-cu128",
    }))
    train_call = parent / "train_call.json"
    environment = dict(os.environ, PATH=f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
                       CUDA_VISIBLE_DEVICES="0", NNUNET_RUN_SEED="999", SIGF_TEST_TRAIN_CALL=str(train_call))
    command = [
        "bash", str(LAUNCHER), "--experiment", experiment, "--profile", "blackwell5090-cu128",
        "--env-prefix", str(env_prefix), "--data-root", str(data_root),
        "--persistent-root", str(persistent_root), "--run-root", str(run_root),
        "--preflight", str(preflight), "--execute-training",
    ]
    if legacy_identity is not None:
        command.append("--resume")
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    return result, run_root, train_call, identity_before


def test_new_run_metadata_and_seed_environment_match_actual_launch_binding():
    for experiment, seed in (("full_molg", None), ("no_maf", 42)):
        with tempfile.TemporaryDirectory() as directory:
            result, run_root, train_call, _ = _mock_run(directory, experiment)
            assert result.returncode == 0, result.stderr
            identity = json.loads((run_root / "run_identity.json").read_text())
            call = json.loads(train_call.read_text())
            assert identity["seed"] == seed
            assert call["seed_environment"] == (None if seed is None else "42")
            assert call["args"][:3] == ["901", "3d_fullres", "all"]
            assert call["args"][3:] == ["-tr", identity["trainer"], "-p", "nnUNetPlans"]
            assert identity["seed_policy"] == (
                "not_fixed_by_entrypoint" if seed is None else "fixed_by_seeded_trainer"
            )


def test_legacy_full_model_resume_preserves_identity_and_marks_seed_unverified():
    identity = {
        "experiment": "full_molg", "trainer": "nnUNetTrainerParallelModalSwinUNetB24",
        "total_epochs": 1000, "seed": 42,
    }
    with tempfile.TemporaryDirectory() as directory:
        result, run_root, train_call, before = _mock_run(directory, "full_molg", identity)
        assert result.returncode == 0, result.stderr
        assert (run_root / "run_identity.json").read_text() == before
        check = json.loads((run_root / "resume_identity_check.json").read_text())
        assert check["legacy_seed_metadata_unverified"] is True
        assert check["entrypoint_seed"] is None
        assert check["recorded_identity_seed"] == 42
        call = json.loads(train_call.read_text())
        assert call["args"][-1] == "--c"
        assert call["seed_environment"] is None
        assert "metadata is unverified" in result.stderr


def test_resume_rejects_different_trainer_and_wrong_seed_without_launching():
    for experiment, trainer, seed in (
        ("full_molg", "another_trainer", 42),
        ("no_maf", "nnUNetTrainerParallelModalSwinUNetB24NoMAF", 17),
        ("full_molg", "nnUNetTrainerParallelModalSwinUNetB24", 17),
    ):
        identity = {"experiment": experiment, "trainer": trainer, "total_epochs": 1000, "seed": seed}
        with tempfile.TemporaryDirectory() as directory:
            result, run_root, train_call, before = _mock_run(directory, experiment, identity)
            assert result.returncode != 0
            assert not train_call.exists()
            assert (run_root / "run_identity.json").read_text() == before
            assert "same-run identity mismatch" in result.stderr
