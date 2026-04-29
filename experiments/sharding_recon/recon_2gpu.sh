#!/usr/bin/env bash

if [ -z "$1" ]; then
    echo "Usage: $0 <size>"
    exit 1
fi

SIZE=$1
NUM_GPUS=2

SCRIPT_DIR="/home/ncardel/repos/mbirjax/experiments/sharding"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

DATA_OUTPUT_FILEPATH="${SCRIPT_DIR}/recon_time_${SIZE}x${SIZE}x${SIZE}_${NUM_GPUS}gpu.txt"

exec > >(tee "${LOG_DIR}/recon_${SIZE}x${SIZE}x${SIZE}_${NUM_GPUS}gpu.out") \
     2> >(tee "${LOG_DIR}/recon_${SIZE}x${SIZE}x${SIZE}_${NUM_GPUS}gpu.err" >&2)

module purge
module load proxy
module load modtree/gpu
module load cuda/12.9.0
module use /depot/bouman/apps/modules
module load cudnn/9.11.0
module load conda

conda activate mbirjax

echo "Running: size=$SIZE, num_gpus=$NUM_GPUS, output=$DATA_OUTPUT_FILEPATH"
python3 "${SCRIPT_DIR}/recon.py" "$SIZE" "$SIZE" "$SIZE" "$DATA_OUTPUT_FILEPATH"
