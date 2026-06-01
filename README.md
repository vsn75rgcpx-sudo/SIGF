# FedMoLG-TCFP

Federated modality-aware local-global 3D U-Net for multimodal BraTS tumor segmentation, with a tumor-core distance-aware false-positive suppression loss.

This repository contains the code for the final model used in our experiments:

```text
Modality-specific shallow CNN stems
+ modality fusion
+ local 3D U-Net branch
+ Swin global branch
+ multi-scale local-global fusion
+ boundary-aware objective
+ TC distance-aware false-positive suppression
+ Dice-HD95 checkpoint selection
```

The model outputs three BraTS region logits:

```text
channel 0: WT
channel 1: TC
channel 2: ET
```

## Highlights

- `parallel_modal_swin_unet3d`: final local-global model.
- `DistanceAwareTCFalsePositiveLoss`: penalizes tumor-core predictions far from ground-truth TC support.
- `train_federated_baseline.py`: supports multi-client FedAvg and single-client centralized-like FL training.
- `paired_case_comparison.py`: case-level paired comparison with W/T/L and Wilcoxon p-values.
- Unified hierarchical post-processing for WT/TC/ET evaluation.

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

This code was developed with PyTorch and CUDA GPUs. Install a PyTorch build matching your CUDA version from the official PyTorch instructions.

## Data Layout

Put BraTS cases under `dataset/train` by default:

```text
dataset/train/
  BraTS-GLI-00000-000/
    *t1n*.nii.gz   or *t1*.nii.gz
    *t1c*.nii.gz   or *t1ce*.nii.gz
    *t2w*.nii.gz   or *t2*.nii.gz
    *t2f*.nii.gz   or *flair*.nii.gz
    *seg*.nii.gz
```

The loader maps label `4` to `3` for ET.

## Main Single-Client Run

The strongest single-client FL-pipeline setting is:

```bash
bash scripts/run_single_client_full_tcfp.sh
```

Equivalent command:

```bash
CUDA_VISIBLE_DEVICES=0 python -u baselines/train_federated_baseline.py \
  --exp_name centralized_parallel_modal_swin_unet3d_b24_boundary_dicehd95_tcfp010_r100 \
  --model parallel_modal_swin_unet3d \
  --num_clients 1 \
  --base_channels 24 \
  --rounds 100 \
  --warmup_rounds 20 \
  --local_epochs 1 \
  --batch_size 2 \
  --loss_type boundary \
  --tc_fp_weight 0.10 \
  --tc_fp_distance 32 \
  --tc_fp_power 1.0 \
  --checkpoint_score dice_hd95 \
  --checkpoint_score_candidates 8 \
  --amp
```

Outputs are saved to:

```text
results/<exp_name>/case_metrics.json
results/<exp_name>/final_metrics.json
checkpoints/<exp_name>/fedavg_final_avg_model_dice_hd95.pth
```

## Ablation Runs

Run the single-client ablations:

```bash
bash scripts/run_single_client_ablations.sh
```

The ablations cover:

- w/o Swin global branch
- w/o modality-specific stems
- w/o TC-FP loss
- w/o boundary objective
- w/o Dice-HD95 checkpoint selection
- U-Net baselines

## Paired Comparison

Example:

```bash
python baselines/paired_case_comparison.py \
  --method results/centralized_parallel_modal_swin_unet3d_b24_boundary_dicehd95_tcfp010_r100/case_metrics.json \
  --baseline results/central1_full_b24_boundary_dicehd95_r100/case_metrics.json \
  --method_name Full-TCFP \
  --baseline_name w/o-TCFP \
  --out results/paired_full_tcfp_vs_no_tcfp_central1.json \
  --csv_out results/paired_full_tcfp_vs_no_tcfp_central1.csv
```

Summarize multiple paired JSON files:

```bash
python baselines/summarize_paired_results.py results/paired_*.json
```

## Notes

- Do not upload BraTS data, checkpoints, logs, or generated results to GitHub.
- Keep `case_metrics.json` for paired statistical analysis.
- If you use this code in a paper, report whether results are from multi-client FL, single-client FL-pipeline, or ordinary centralized training.

