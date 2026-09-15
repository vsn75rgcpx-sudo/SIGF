# Reproduction workflow

This is a public code release, not a data or weight release. Use a legally
obtained copy of the data and the frozen metadata in frozen/. The commands
below use placeholders and do not contain private machine paths.

## Environment

Use Python 3.10 on Linux. Select one GPU profile. The common recipe fixes
nnunetv2 2.8.0, dynamic-network-architectures 0.4.4 and timm 1.0.22. The
currently verified Blackwell profile fixes Torch 2.7.1+cu128, torchvision
0.22.1+cu128 and Triton 3.3.1. The Ada profile is an alternate cu118 recipe
and requires its own Linux installation and GPU preflight. The host driver and
GPU compatibility still require a real preflight.

    bash scripts/install_environment.sh --profile blackwell5090-cu128 --prefix /persistent/molg/envs/molg-cu128
    bash scripts/install_environment.sh --profile blackwell5090-cu128 --prefix /persistent/molg/envs/molg-cu128 --apply
    /persistent/molg/envs/molg-cu128/bin/python scripts/install_custom_trainer.py --apply
    /persistent/molg/envs/molg-cu128/bin/python scripts/verify_environment.py --profile blackwell5090-cu128 --out /persistent/molg/reports/environment.json

The first install command is inspect-only. The second is the explicit
installation command. The public release does not include an offline
wheelhouse and does not install anything on the Mac authoring machine.

## Data

Use the authorized membership procedure in docs/data.md, then materialize
nnU-Net raw data and run scripts/preprocess_frozen.sh. Pass data-root and
persistent-root separately; never run a new planner to replace frozen
nnUNetPlans.

## Training

The public launcher accepts full_molg, no_maf, no_swin, concat_fusion and
proposed_loss_ds_matched. It defaults to inspect-only. A real run requires
an explicit preflight report, a Linux GPU, an existing prepared data tree and
an empty run root:

    /persistent/molg/envs/molg-cu128/bin/python scripts/preflight.py --experiment proposed_loss_ds_matched --profile blackwell5090-cu128 --data-root /data/molg --persistent-root /persistent/molg --out /persistent/molg/preflight/proposed_loss_ds_matched.json --execute
    CUDA_VISIBLE_DEVICES=0 bash scripts/train_one_experiment.sh --experiment proposed_loss_ds_matched --profile blackwell5090-cu128 --env-prefix /persistent/molg/envs/molg-cu128 --data-root /data/molg --persistent-root /persistent/molg --run-root /persistent/molg/runs/proposed_loss_ds_matched-YYYYMMDD --preflight /persistent/molg/preflight/proposed_loss_ds_matched.json --execute-training

The launcher fixes seed 42, 1000 epochs, the frozen patch and batch, the
250/50 iteration schedule, AMP, compile, and library thread pools at one.
Augmentation workers are an explicit parameter, defaulting to four. Each run
gets a separate short /tmp/molg-* directory. A resume uses only the same run
root's checkpoint_latest.pth and restores the complete nnU-Net state; it
never silently falls back to a new run.

## Evaluation

Use the corresponding checkpoint_final.pth and the authorized group-specific
labels. For a complete inference-plus-scoring run:

    bash scripts/evaluate_one_experiment.sh --experiment proposed_loss_ds_matched --group internal375 --env-prefix /persistent/molg/envs/molg-cu128 --data-root /data/molg --persistent-root /persistent/molg --run-root /persistent/molg/runs/proposed_loss_ds_matched-YYYYMMDD --images-dir /persistent/molg/eval/internal375/imagesTs --label-dir /persistent/molg/eval/internal375/labelsTs --output-dir /persistent/molg/eval/results/proposed_loss_ds_matched-internal375 --execute-evaluation

The wrapper invokes nnUNetv2 prediction with the fixed inference options,
checks exact case coverage, then writes the case-level CSV and JSON summary.
The evaluator can also score an already generated prediction directory as
shown in docs/metrics.md. No checkpoint, prediction, case list, raw data or
result file is included in this repository.

## Validation status

Static checks on this release cover Python parsing, shell parsing, JSON
contracts, forbidden-material scans and small dependency-free synthetic
metric/contract tests. A fresh Linux install, driver compatibility, GPU
preflight, real data staging/preprocessing, training, resume execution and
full internal/external evaluation remain to be performed by the user on an
authorized environment.
