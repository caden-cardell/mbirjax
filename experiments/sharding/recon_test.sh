#!/bin/bash

LOG_OUTPUT_FILEPATH="logs/recon.log"

echo "Saving log to $LOG_OUTPUT_FILEPATH"

for num_gpus in 4; do
  case $num_gpus in
    1) export CUDA_VISIBLE_DEVICES=0 ;;
    2) export CUDA_VISIBLE_DEVICES=0,1 ;;
    4) export CUDA_VISIBLE_DEVICES=0,1,2,3 ;;
  esac

  DATA_OUTPUT_FILEPATH="recon_time_${num_gpus}gpu.txt"
  echo "Running with num_gpus=$num_gpus, output=$DATA_OUTPUT_FILEPATH"

  for size in 128 256 512 1024 1280 1536 1792 2048; do
    echo "  Running with size=$size"
    python3 recon.py "$size" "$size" "$size" "$DATA_OUTPUT_FILEPATH" "$num_gpus"
  done
done