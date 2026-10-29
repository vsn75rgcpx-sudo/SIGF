# Reproducing the SIGF experiments

The SIGF manuscript reports the `full_molg` model and three standard-loss
structural ablations: `no_maf`, `no_swin`, and `concat_fusion`. The existing
MoLG experiment and class names are preserved for checkpoint compatibility.
All commands below target the main model unless stated otherwise.

This release contains code and frozen metadata. It does not contain the
study-specific case lists, trained weights, or case-level results. Obtain the
original data under its applicable terms and the membership information
specified in [data.md](data.md) before attempting the exact study protocol.
The example paths are placeholders.

## Environment

Use Python 3.10 on Linux. The common recipe fixes nnunetv2 2.8.0,
dynamic-network-architectures 0.4.4, and timm 1.0.22. The recorded Blackwell
profile fixes Torch 2.7.1+cu128, torchvision 0.22.1+cu128, and Triton 3.3.1.
The Ada profile is a separate cu118 recipe that requires its own validation.
Driver and GPU compatibility must be checked on the target host.

```bash
bash scripts/install_environment.sh --profile blackwell5090-cu128 --prefix /persistent/molg/envs/molg-cu128
bash scripts/install_environment.sh --profile blackwell5090-cu128 --prefix /persistent/molg/envs/molg-cu128 --apply
/persistent/molg/envs/molg-cu128/bin/python scripts/install_custom_trainer.py --apply
/persistent/molg/envs/molg-cu128/bin/python scripts/verify_environment.py --profile blackwell5090-cu128 --out /persistent/molg/reports/environment.json
```

The first command only inspects the installation plan; the second performs
installation. The release does not include an offline wheelhouse. Full
installation and execution require the Linux runtime rather than the Mac
used to prepare these repository updates.

## Data and preprocessing

Follow [data.md](data.md) to obtain the membership information and materialize
the nnU-Net raw data. Then run the frozen preprocessing entry point:

```bash
bash scripts/preprocess_frozen.sh --python /persistent/molg/envs/molg-cu128/bin/python --data-root /data/molg --persistent-root /persistent/molg --processes 2 --execute
```

Keep `data-root` and `persistent-root` separate. Use the supplied
`nnUNetPlans`; running a new planner may produce a different patch size,
batch size, or preprocessing configuration. The supplied plans specify
1 mm isotropic spacing, foreground-mask Z-score normalization, patch size
`[128, 160, 112]`, and batch size 2.

## Main-model training

`full_molg` selects `nnUNetTrainerParallelModalSwinUNetB24`, the SIGF network
with standard region Dice–BCE and nnU-Net deep supervision. The launcher uses
`fold=all` for 1,000 epochs on the prepared training set. This is a single
full-data model; it does not launch five folds.

A run requires a prepared data tree, a passing preflight report, one visible
GPU, and a new persistent run directory:

```bash
CUDA_VISIBLE_DEVICES=0 /persistent/molg/envs/molg-cu128/bin/python scripts/preflight.py --experiment full_molg --profile blackwell5090-cu128 --data-root /data/molg --persistent-root /persistent/molg --out /persistent/molg/preflight/full_molg.json --execute
CUDA_VISIBLE_DEVICES=0 bash scripts/train_one_experiment.sh --experiment full_molg --profile blackwell5090-cu128 --env-prefix /persistent/molg/envs/molg-cu128 --data-root /data/molg --persistent-root /persistent/molg --run-root /persistent/molg/runs/full_molg-YYYYMMDD --preflight /persistent/molg/preflight/full_molg.json --execute-training
```

Without `--execute-training`, the training launcher prints its binding and
does not start a run. Preflight checks package/source bindings and GPU
visibility; a passing preflight is not a completed training test.

The remaining training settings follow the fixed nnU-Net protocol: 250
training iterations and 50 monitoring iterations per epoch, SGD with initial
learning rate 0.01 and polynomial decay, AMP, and compilation. Augmentation
workers are configurable and default to four. Library thread pools use one
thread, and each run receives its own short temporary directory.

### Seed behavior and resume

`full_molg` preserves the main B24 trainer, which does not inherit
`SeededTrainerMixin` and does not apply `NNUNET_RUN_SEED` to model RNGs. The
three structural ablations and `proposed_loss_ds_matched` do inherit that
mixin and apply seed 42. The launcher records the selected experiment's seed
behavior; it cannot establish the seed used by an earlier training run.

Older launcher records may state `seed: 42` for `full_molg` even though that
entry did not apply it. Preserve original records and checkpoints as evidence
of the original run rather than treating that field as verified RNG state.
For the legacy main-model record with seed 42 and no seed_policy field, the
launcher preserves run_identity.json and writes a separate
resume_identity_check.json marking that seed metadata as unverified. It then
continues through the same-run checkpoint restoration path. Other trainer or
seed identity mismatches are rejected before launching the trainer.

A resume uses only the same run's `checkpoint_latest.pth` with nnU-Net's
full-state continuation path. It is not a new initialization or a renamed
best checkpoint. The checkpoint checker verifies field presence on the CPU;
actual state restoration requires loading it with the installed trainer.

## Structural ablations

Use the same preparation and commands with a distinct experiment key,
preflight filename, and run directory for each variant:

| Key | Trainer | Change from SIGF |
| --- | --- | --- |
| `no_maf` | `nnUNetTrainerParallelModalSwinUNetB24NoMAF` | Replaces independent modality stems and attention aggregation with shared convolution |
| `no_swin` | `nnUNetTrainerParallelModalSwinUNetB24NoSwin` | Removes Swin and the associated fusion units; decodes CNN features |
| `concat_fusion` | `nnUNetTrainerParallelModalSwinUNetB24ConcatFusion` | Replaces each gated fusion with concatenation and channel projection |

All three use standard region Dice–BCE with the standard deep-supervision
path. NoMAF evaluates the complete shallow modality front end, rather than
isolating attention from feature extraction.

## Final-checkpoint evaluation

Use `checkpoint_final.pth` from the corresponding experiment and the
cohort-specific input images and reference labels. For the 375-case BraTS
2023 test list:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_one_experiment.sh --experiment full_molg --group internal375 --env-prefix /persistent/molg/envs/molg-cu128 --data-root /data/molg --persistent-root /persistent/molg --run-root /persistent/molg/runs/full_molg-YYYYMMDD --images-dir /persistent/molg/eval/internal375/imagesTs --label-dir /persistent/molg/eval/internal375/labelsTs --output-dir /persistent/molg/eval/results/full_molg-internal375 --execute-evaluation
```

For the supplemental BraTS-GoAT 2024 cohort, use `--group external_novel100`
and the corresponding 100-case images, labels, and a separate output
directory. `external_novel100` is the retained CLI identifier for the
supplemental cohort, not a claim of independent clinical external validation.

The wrapper uses the final checkpoint, `fold=all`, sliding-window step 0.5,
Gaussian blending, and mirroring. It checks the prediction/label case set and
expected count, then writes case-level CSV and JSON summaries. The metric
script applies hierarchical connected-component processing before scoring.
See [metrics.md](metrics.md) for NSD tolerance, empty-mask handling, and
component thresholds. Evaluation can also score an existing prediction folder.

## Additional experiments

`proposed_loss_ds_matched` remains available for a separate composite-loss
experiment. It uses the full network with boundary/curvature weighting and a
TC false-positive term applied at the highest-resolution output. Its
configured deep-supervision weights are `[8/15, 4/15, 2/15, 1/15, 0]`.
It is not the standard-loss SIGF model reported in the current manuscript.

The files under `legacy/fedmolg_tcfp` retain the earlier federated prototype.
They are not the reproduction route for the current paper.

## Validation boundary

Static checks cover parsing, configuration, source hashes, public-material
scans, and small dependency-free contract/data tests. They do not reproduce
trained weights, predictions, or manuscript tables. See
[validation_status.md](validation_status.md) for the checks performed and the
runtime execution that remains host- and data-dependent.
