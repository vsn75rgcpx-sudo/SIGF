#!/usr/bin/env bash
set -euo pipefail

python baselines/paired_case_comparison.py \
  --method results/centralized_parallel_modal_swin_unet3d_b24_boundary_dicehd95_tcfp010_r100/case_metrics.json \
  --baseline results/central1_full_b24_boundary_dicehd95_r100/case_metrics.json \
  --method_name Full-TCFP \
  --baseline_name w/o-TCFP \
  --out results/paired_full_tcfp_vs_no_tcfp_central1.json \
  --csv_out results/paired_full_tcfp_vs_no_tcfp_central1.csv

python baselines/paired_case_comparison.py \
  --method results/centralized_parallel_modal_swin_unet3d_b24_boundary_dicehd95_tcfp010_r100/case_metrics.json \
  --baseline results/central1_no_swin_b24_boundary_tcfp010_dicehd95_r100/case_metrics.json \
  --method_name Full-TCFP \
  --baseline_name w/o-Swin \
  --out results/paired_full_tcfp_vs_no_swin_central1.json \
  --csv_out results/paired_full_tcfp_vs_no_swin_central1.csv

python baselines/summarize_paired_results.py results/paired_full_tcfp_vs_*_central1.json
