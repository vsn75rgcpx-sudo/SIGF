import argparse
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Install the ModalSwinUNet custom trainer into an nnU-Net v2 package"
    )
    parser.add_argument(
        "--nnunetv2_root",
        type=str,
        default="",
        help="Path to the installed nnunetv2 package root. If omitted, import nnunetv2 and auto-detect.",
    )
    return parser.parse_args()


def detect_nnunetv2_root():
    import nnunetv2

    return Path(nnunetv2.__file__).resolve().parent


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    source_dir = repo_root / "nnunetv2_custom"

    nnunetv2_root = Path(args.nnunetv2_root).resolve() if args.nnunetv2_root else detect_nnunetv2_root()
    target_dir = nnunetv2_root / "training" / "nnUNetTrainer" / "variants" / "network_architecture"
    target_dir.mkdir(parents=True, exist_ok=True)
    init_file = target_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("")

    files = {
        source_dir / "modal_swin_unet.py": target_dir / "modal_swin_unet.py",
        source_dir / "nnUNetTrainerModalSwinUNet.py": target_dir / "nnUNetTrainerModalSwinUNet.py",
    }
    for src, dst in files.items():
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, dst)
        print(f"Installed {src} -> {dst}")

    print()
    print("Trainer installed. Use:")
    print("  nnUNetv2_train <DATASET_ID> 3d_fullres all -tr nnUNetTrainerProposedLoss")
    print("  nnUNetv2_train <DATASET_ID> 3d_fullres 0 -tr nnUNetTrainerModalSwinUNet")
    print("  nnUNetv2_train <DATASET_ID> 3d_fullres 0 -tr nnUNetTrainerModalAttnSwinUNet")
    print("  nnUNetv2_train <DATASET_ID> 3d_fullres 0 -tr nnUNetTrainerParallelModalSwinUNet")
    print("  nnUNetv2_train <DATASET_ID> 3d_fullres all -tr nnUNetTrainerParallelModalSwinUNetB24")
    print("  nnUNetv2_train <DATASET_ID> 3d_fullres all -tr nnUNetTrainerParallelModalSwinUNetB24ProposedLoss")
    print("BSPC repeats and main-pipeline ablations:")
    print("  NNUNET_RUN_SEED=42 nnUNetv2_train <DATASET_ID> 3d_fullres all -tr nnUNetTrainerSeeded")
    print("  NNUNET_RUN_SEED=42 nnUNetv2_train <DATASET_ID> 3d_fullres all -tr nnUNetTrainerParallelModalSwinUNetB24Seeded")
    print("  NNUNET_RUN_SEED=42 nnUNetv2_train <DATASET_ID> 3d_fullres all -tr nnUNetTrainerParallelModalSwinUNetB24NoMAF")
    print("  NNUNET_RUN_SEED=42 nnUNetv2_train <DATASET_ID> 3d_fullres all -tr nnUNetTrainerParallelModalSwinUNetB24NoSwin")
    print("  NNUNET_RUN_SEED=42 nnUNetv2_train <DATASET_ID> 3d_fullres all -tr nnUNetTrainerParallelModalSwinUNetB24ConcatFusion")
    print("Fast 100-epoch screening variant:")
    print("  nnUNetv2_train <DATASET_ID> 3d_fullres all -tr nnUNetTrainerProposedLoss_100ep")
    print("  nnUNetv2_train <DATASET_ID> 3d_fullres 0 -tr nnUNetTrainerModalSwinUNet100ep")
    print("  nnUNetv2_train <DATASET_ID> 3d_fullres 0 -tr nnUNetTrainerModalAttnSwinUNet100ep")
    print("  nnUNetv2_train <DATASET_ID> 3d_fullres 0 -tr nnUNetTrainerParallelModalSwinUNet100ep")
    print("  nnUNetv2_train <DATASET_ID> 3d_fullres all -tr nnUNetTrainerParallelModalSwinUNetB24_100ep")
    print("  nnUNetv2_train <DATASET_ID> 3d_fullres all -tr nnUNetTrainerParallelModalSwinUNetB24ProposedLoss_100ep")
    print("Searched global encoder variants; set NNUNET_SEARCH_ARCH first:")
    print("  export NNUNET_SEARCH_ARCH=/path/to/best_arch_GLOBAL.json")
    print("  nnUNetv2_train <DATASET_ID> 3d_fullres 0 -tr nnUNetTrainerSearchedModalSwinUNet100ep")
    print("  nnUNetv2_train <DATASET_ID> 3d_fullres 0 -tr nnUNetTrainerSearchedModalAttnSwinUNet100ep")
    print("  nnUNetv2_train <DATASET_ID> 3d_fullres 0 -tr nnUNetTrainerSearchedParallelModalSwinUNet100ep")


if __name__ == "__main__":
    main()
