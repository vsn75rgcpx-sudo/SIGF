#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -u baselines/train_federated_baseline.py \
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
