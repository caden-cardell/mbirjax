#!/usr/bin/env bash
# Shared anisotropic sweep definition, sourced by every sbatch script here.
#
# Baseline 512^3, then increase exactly ONE sinogram dimension by 256 up to 2048 while holding
# the other two at 512. The three sinogram axes are (num_views, num_det_rows, num_det_channels)
# and play different roles in recon -- rows drive the sharded recon-slice axis -- so sweeping
# one at a time reveals whether any single dimension scales non-linearly.
#
# 19 configs total => SLURM arrays below use --array=0-18.
CONFIGS=(
  "512 512 512"     # 0  baseline (shared by all three axes)
  # --- views sweep (rows=channels=512) ---
  "768 512 512"     # 1
  "1024 512 512"    # 2
  "1280 512 512"    # 3
  "1536 512 512"    # 4
  "1792 512 512"    # 5
  "2048 512 512"    # 6
  # --- rows sweep (views=channels=512) ---
  "512 768 512"     # 7
  "512 1024 512"    # 8
  "512 1280 512"    # 9
  "512 1536 512"    # 10
  "512 1792 512"    # 11
  "512 2048 512"    # 12
  # --- channels sweep (views=rows=512) ---
  "512 512 768"     # 13
  "512 512 1024"    # 14
  "512 512 1280"    # 15
  "512 512 1536"    # 16
  "512 512 1792"    # 17
  "512 512 2048"    # 18
)
