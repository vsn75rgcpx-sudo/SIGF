# SIGF

**Sequence Integration and Gated Fusion for Brain Tumor Segmentation**

This repository provides the nnU-Net v2 implementation of SIGF for supervised
segmentation of whole tumor (WT), tumor core (TC), and enhancing tumor (ET)
from four MRI sequences: T1, T1ce, T2, and FLAIR.

SIGF is the name used in the manuscript. The repository URL, experiment key
`full_molg`, and Python class names retain the earlier MoLG-UNet naming so
existing commands and checkpoints remain compatible.

## Architecture

SIGF first extracts shallow features separately from each MRI sequence and
combines them through voxel-wise modality attention. Parallel CNN and
hierarchical Swin encoders then process the shared representation. Gated
residual fusion supplies features at five scales to the decoder.

- Each modality stem contains two 3D convolution blocks and produces 12
  channels. Attention combines these features before projection to 24 channels.
- The two encoding paths use 24, 48, 96, 192, and 384 channels. Each path
  passes its own features to the next scale; fused features feed the decoder.
- Swin attention uses 3×3×3 windows, with shifted attention in the second
  bottleneck block.
- Training uses region-based Dice–BCE and multiscale deep supervision: one
  full-resolution output and three auxiliary outputs. Inference returns the
  full-resolution output.

The registered model has **19,144,082 parameters (19.14M)**, including its
auxiliary segmentation heads. The main implementation is
[`ParallelModalSwinUNet`](runtime/nnunetv2_custom/modal_swin_unet.py), selected
by `nnUNetTrainerParallelModalSwinUNetB24`.

## Experiments reported in the manuscript

The main model and the three structural ablations all use standard nnU-Net
region Dice–BCE supervision. Their definitions are recorded in
[`config/experiments.json`](config/experiments.json).

| Experiment key | Model | Structural change |
| --- | --- | --- |
| `full_molg` | SIGF | Full modality front end, CNN/Swin encoders, and gated residual fusion |
| `no_maf` | NoMAF | Replaces the complete modality-specific stems and attention aggregation with shared `ConvNormAct3D(4,24)` |
| `no_swin` | NoSwin | Retains the modality front end and CNN; removes the Swin encoder and its five fusion units |
| `concat_fusion` | ConcatFusion | Retains both encoders; replaces all five gated fusion units with concatenation and 1×1×1 `ConvNormAct3D` |

NoMAF evaluates the modality front end as a whole. The NoSwin result is retained
as reported; the full network is not assumed to be best on every cohort.

`proposed_loss_ds_matched` is an **additional experiment**, not the SIGF main
model reported in the manuscript. It uses boundary/curvature terms and a
TC false-positive penalty. Its code remains available separately from the
paper's standard-loss experiments. The earlier federated/100-round prototype
is retained in [`legacy/fedmolg_tcfp`](legacy/fedmolg_tcfp).

## Training and evaluation settings

- `Dataset901_BraTS2023_TCRegion`, `3d_fullres`, supplied `nnUNetPlans`,
  and `fold=all`.
- Patch size `[128, 160, 112]`, batch size 2, and 1,000 epochs.
- 1 mm isotropic spacing and foreground-mask Z-score normalization for each
  sequence, as specified in the frozen plans.
- Nesterov SGD, initial learning rate 0.01, momentum 0.99, weight decay
  `3e-5`, and the nnU-Net polynomial learning-rate schedule.
- Standard nnU-Net augmentation, 250 training iterations and 50 training-time
  monitoring iterations per epoch, AMP, and `torch.compile`.
- Formal testing uses `checkpoint_final.pth`, sliding-window inference with
  mirroring, and fixed hierarchical connected-component postprocessing.

The `full_molg` trainer does not set the model RNG seed through
`NNUNET_RUN_SEED`. The three structural ablation trainers set seed 42 through
`SeededTrainerMixin`. A seed value written in an older launcher record is not
proof that the main trainer applied it; see
[reproduction details](docs/reproduction.md#seed-behavior-and-resume).

The manuscript uses 876 training cases, a 375-case BraTS 2023 test list, and a
100-case image-deduplicated BraTS-GoAT 2024 supplemental cohort. The existing
CLI group name `external_novel100` refers to that supplemental cohort; the
name does not establish clinical external validation.

## Reproduction

Start with [the reproduction guide](docs/reproduction.md). It covers the
Linux environment, data preparation, frozen preprocessing, training, and
final-checkpoint evaluation. The main-model experiment key is **`full_molg`**.

The public release contains source code and frozen metadata. It does not
include MRI scans, case lists, weights, predictions, complete logs, or
case-level results. Recreating the manuscript's exact cohorts requires the
membership information described in [the data guide](docs/data.md); the
public data sources alone do not supply this study-specific split.

The recorded environment uses Python 3.10.x, nnunetv2 2.8.0,
dynamic-network-architectures 0.4.4, and timm 1.0.22. The Blackwell/RTX 5090
recipe uses Torch 2.7.1+cu128; a separate Ada/RTX 4090 cu118 recipe is also
provided. Each target host needs its own installation and GPU preflight.

[Metric definitions](docs/metrics.md) specify the NSD tolerance, HD95 empty-mask
rule, and component thresholds. Training pseudo Dice is a monitoring signal;
formal results use the designated test cohort and final checkpoint.

## Validation and release history

[Validation status](docs/validation_status.md) distinguishes static checks
from execution with the scientific runtime. The current public clone has not
been used here to complete GPU training or reproduce the manuscript tables.

See [release notes](docs/release_notes.md),
[source and packaging changes](docs/source_diff.md), and
[security information](docs/security.md).

## Citation and license

Software citation metadata is provided in [`CITATION.cff`](CITATION.cff) and
[`.zenodo.json`](.zenodo.json). Cite the actual version or commit used; use a
release DOI only when that release has one.

Authored repository material is provided under the [MIT License](LICENSE).
Dependencies retain their own licenses; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
