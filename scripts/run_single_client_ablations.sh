#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

CUDA_VISIBLE_DEVICES="${GPU_NO_SWIN:-0}" nohup python -u baselines/train_federated_baseline.py \
  --exp_name central1_no_swin_b24_boundary_tcfp010_dicehd95_r100 \
  --model parallel_modal_unet_no_swin3d \
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
  --amp \
  > logs/central1_no_swin_b24_boundary_tcfp010_dicehd95_r100.log 2>&1 &

CUDA_VISIBLE_DEVICES="${GPU_NO_MODAL:-1}" nohup python -u baselines/train_federated_baseline.py \
  --exp_name central1_no_modal_stem_b24_boundary_tcfp010_dicehd95_r100 \
  --model parallel_swin_unet_no_modal_stem3d \
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
  --amp \
  > logs/central1_no_modal_stem_b24_boundary_tcfp010_dicehd95_r100.log 2>&1 &

CUDA_VISIBLE_DEVICES="${GPU_NO_TCFP:-2}" nohup python -u baselines/train_federated_baseline.py \
  --exp_name central1_full_b24_boundary_dicehd95_r100 \
  --model parallel_modal_swin_unet3d \
  --num_clients 1 \
  --base_channels 24 \
  --rounds 100 \
  --warmup_rounds 20 \
  --local_epochs 1 \
  --batch_size 2 \
  --loss_type boundary \
  --checkpoint_score dice_hd95 \
  --checkpoint_score_candidates 8 \
  --amp \
  > logs/central1_full_b24_boundary_dicehd95_r100.log 2>&1 &

CUDA_VISIBLE_DEVICES="${GPU_REGION:-3}" nohup python -u baselines/train_federated_baseline.py \
  --exp_name central1_full_b24_region_tcfp010_dicehd95_r100 \
  --model parallel_modal_swin_unet3d \
  --num_clients 1 \
  --base_channels 24 \
  --rounds 100 \
  --warmup_rounds 20 \
  --local_epochs 1 \
  --batch_size 2 \
  --loss_type region \
  --tc_fp_weight 0.10 \
  --tc_fp_distance 32 \
  --tc_fp_power 1.0 \
  --checkpoint_score dice_hd95 \
  --checkpoint_score_candidates 8 \
  --amp \
  > logs/central1_full_b24_region_tcfp010_dicehd95_r100.log 2>&1 &

CUDA_VISIBLE_DEVICES="${GPU_DICE_ONLY:-4}" nohup python -u baselines/train_federated_baseline.py \
  --exp_name central1_full_b24_boundary_tcfp010_diceonly_r100 \
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
  --checkpoint_score dice \
  --amp \
  > logs/central1_full_b24_boundary_tcfp010_diceonly_r100.log 2>&1 &

CUDA_VISIBLE_DEVICES="${GPU_UNET_FINAL:-5}" nohup python -u baselines/train_federated_baseline.py \
  --exp_name central1_unet3d_boundary_tcfp010_dicehd95_r100 \
  --model unet3d \
  --num_clients 1 \
  --base_channels 32 \
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
  --amp \
  > logs/central1_unet3d_boundary_tcfp010_dicehd95_r100.log 2>&1 &

CUDA_VISIBLE_DEVICES="${GPU_UNET_REGION:-6}" nohup python -u baselines/train_federated_baseline.py \
  --exp_name central1_unet3d_region_dicehd95_r100 \
  --model unet3d \
  --num_clients 1 \
  --base_channels 32 \
  --rounds 100 \
  --warmup_rounds 20 \
  --local_epochs 1 \
  --batch_size 2 \
  --loss_type region \
  --checkpoint_score dice_hd95 \
  --checkpoint_score_candidates 8 \
  --amp \
  > logs/central1_unet3d_region_dicehd95_r100.log 2>&1 &

echo "Ablation jobs launched. Check logs/*.log for progress."
