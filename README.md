# MoLG-UNet

Modality-aware local-global U-Net for multimodal brain tumor segmentation.

This repository contains the current nnU-Net v2 implementation used for the
MoLG model family and its four formal ablations. The runtime source includes
the modality-aware input path, local 3D U-Net branch, global Swin branch,
scale-wise fusion, matched deep supervision, the reviewed composite loss, and
the independent BraTS-region evaluation utility.

The release is code-first. It does not include clinical data, case lists,
weights, predictions, full logs, or an environment image.

## Current implementation

The public experiment contract is in config/experiments.json. The four formal
experiments are:

| Experiment | Trainer | Defined change |
| --- | --- | --- |
| no_maf | nnUNetTrainerParallelModalSwinUNetB24NoMAF | Early-fused ConvNormAct3D(4,24) replaces the complete modality-aware input/stem |
| no_swin | nnUNetTrainerParallelModalSwinUNetB24NoSwin | Removes the global Swin branch and corresponding fusion path |
| concat_fusion | nnUNetTrainerParallelModalSwinUNetB24ConcatFusion | Replaces all five gated fusions with concatenation plus 1x1 ConvNormAct3D |
| proposed_loss_ds_matched | nnUNetTrainerParallelModalSwinUNetB24ProposedLossDSMatchedSeed42 | Full MoLG network with the reviewed composite loss and matched deep supervision |

The reference implementation is
nnUNetTrainerParallelModalSwinUNetB24. The proposed-loss deep-supervision
weights are [8/15, 4/15, 2/15, 1/15, 0], with the TC false-positive term
applied only at the highest-resolution output. The standard variants use the
standard nnU-Net region Dice+BCE loss and its standard deep-supervision path.
NoSwin and proposed-loss/DS-matched are retained; they are not removed because
their observed result is different.

The earlier FedMoLG-TCFP federated/100-round prototype is retained under
legacy/fedmolg_tcfp. That historical entry, including its Dice-HD95 selection
utilities, is not the default current reproduction route.

## Frozen scientific contract

- Dataset901_BraTS2023_TCRegion, configuration 3d_fullres, plans nnUNetPlans,
  fold all.
- Four input channels in order T1, T1ce, T2, FLAIR.
- Seed 42, 1000 epochs, 250 training iterations and 50 validation iterations
  per epoch.
- Patch [128, 160, 112], batch size 2, AMP and torch.compile enabled.
- SGD with initial learning rate 0.01, momentum 0.99, Nesterov enabled and
  weight decay 3e-5; nnU-Net polynomial learning-rate schedule.
- Data augmentation and normalization remain the frozen nnU-Net pipeline.
- 876 training cases, 375 internal evaluation cases and 100 external novel
  cases. Internal and external cases are never training cases.

The 876-case membership was recovered from the complement of the frozen
internal set within the 2023 archive and checked against matching historical
case-metrics CSV files. It was not recovered from an original
splits_final.json; the public repository therefore publishes the method and
format, not the case identifiers.

## Environment

The current verified runtime uses Python 3.10.x, nnunetv2 2.8.0,
dynamic-network-architectures 0.4.4 and timm 1.0.22. The verified GPU profile
is:

- Blackwell/RTX 5090: Torch 2.7.1+cu128, torchvision 0.22.1+cu128, Triton
  3.3.1.
An Ada/RTX 4090 cu118 recipe is retained as an explicitly separate
alternative, but it requires its own Linux installation and GPU preflight;
it is not claimed as validated by this release.

The exact common recipe is requirements/constraints-common-runtime.txt and one
profile file. The old timm 1.0.27 file is not part of the default recipe.
provenance/environment_freeze_public.txt records the actual verified package
versions without private wheel URLs.

Install and verify on an authorized Linux host:

    bash scripts/install_environment.sh --profile blackwell5090-cu128 --prefix /persistent/molg/envs/molg-cu128
    bash scripts/install_environment.sh --profile blackwell5090-cu128 --prefix /persistent/molg/envs/molg-cu128 --apply
    /persistent/molg/envs/molg-cu128/bin/python scripts/install_custom_trainer.py --apply
    /persistent/molg/envs/molg-cu128/bin/python scripts/verify_environment.py --profile blackwell5090-cu128 --out /persistent/molg/reports/environment.json

The first command is inspect-only; --apply is required for any installation.
The installer requires Linux, Python 3.10 and a matching GPU profile. It does
not install drivers, upload code, rent instances or alter a global Python.

## Data preparation

Use only legally obtained data and the authorized membership files. The
lossless utility in runtime/baselines/prepare_data_lossless.py:

- streams nested 2023 ZIP NIfTI members without loading and re-serializing
  voxel arrays;
- copies existing GoAT .nii.gz payloads byte-for-byte;
- rejects path traversal, non-regular archive members, duplicate modalities,
  incomplete markers and unmarked collisions;
- does not run an automatic planner or invent a split.

The full membership, lossless staging and frozen preprocessing commands are
documented in docs/data.md. Keep raw/staged/preprocessed data on a data volume
and code, logs, checkpoints and recovery records on a separate persistent
volume.

## Training

The public launcher is inspect-only unless --execute-training is supplied.
It accepts full_molg, no_maf, no_swin, concat_fusion and
proposed_loss_ds_matched. A real run requires a current preflight report and
an empty persistent run root:

    /persistent/molg/envs/molg-cu128/bin/python scripts/preflight.py --experiment proposed_loss_ds_matched --profile blackwell5090-cu128 --data-root /data/molg --persistent-root /persistent/molg --out /persistent/molg/preflight/proposed_loss_ds_matched.json --execute
    CUDA_VISIBLE_DEVICES=0 bash scripts/train_one_experiment.sh --experiment proposed_loss_ds_matched --profile blackwell5090-cu128 --env-prefix /persistent/molg/envs/molg-cu128 --data-root /data/molg --persistent-root /persistent/molg --run-root /persistent/molg/runs/proposed_loss_ds_matched-YYYYMMDD --preflight /persistent/molg/preflight/proposed_loss_ds_matched.json --execute-training

The launcher records the actual trainer, run paths, command, worker setting,
thread settings and exit status. It creates an independent short
/tmp/molg-* directory per run. A resume may use only that run's
checkpoint_latest.pth with nnU-Net's full-state --c path; it does not rename a
best checkpoint or silently fall back to a new run. The checkpoint contract
checker is CPU-only and reports field completeness; actual restoration is
proven only by loading the checkpoint with the installed trainer.

## Evaluation

The evaluator in runtime/baselines/evaluate_prediction_folder.py computes
WT, TC and ET case-level Dice, HD95 and NSD. It maps source label 4 to 3,
uses hierarchical connected-component thresholds WT 300, TC 50 and ET 30,
and applies the fixed HD95 rule: both masks empty is 0, exactly one empty or
an empty surface is 374. Summary standard deviations use ddof=1.

Training pseudo Dice is not an internal375 or external novel100 test result.
Formal evaluation uses the corresponding checkpoint_final.pth, the authorized
case group, and the fixed inference/post-processing contract. See
docs/metrics.md for the command and details.

## Validation status and scope

This release has passed local Python parsing, shell parsing, JSON contract
checks, public-material scans, source-hash comparison to the reviewed
recipe-v2 snapshot, and dependency-free synthetic checks. It has not been
cold-start installed or GPU-validated from this public clone on the Mac
authoring environment. Real data staging, frozen preprocessing, GPU
preflight, training, resume execution and full internal/external reproduction
remain host- and data-dependent.

No claim is made that the model is universally superior, that results are
selected by this repository, or that a paper has been accepted. See
docs/reproduction.md, docs/data.md, docs/metrics.md,
docs/release_notes.md and docs/security.md.

## Citation

For a published version, cite the versioned GitHub Release and its Zenodo DOI.
The machine-readable citation metadata is kept in CITATION.cff and
.zenodo.json. The release does not include private data, weights or
case-level results.

## License

Authored repository material is provided under the MIT License in LICENSE.
Dependencies remain under their own licenses; see
THIRD_PARTY_NOTICES.md.
