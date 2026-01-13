#!/bin/bash

DATA_OUTPUT_FILEPATH="recon_time.txt"
LOG_OUTPUT_FILEPATH="../output/recon.log"

echo "Saving log to $LOG_OUTPUT_FILEPATH"

for size in 256 512 1024; do
  echo "Running with size=$size"
  python3 recon_on_gpu.py "$size" "1024" "1024" "$DATA_OUTPUT_FILEPATH"
done