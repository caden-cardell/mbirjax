#!/usr/bin/env bash

#SBATCH --job-name=recon_1gpu
#SBATCH -A bouman -p ai -q normal
#SBATCH -t 04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=14 --gpus-per-node=1
#SBATCH --array=0-1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ncardel@purdue.edu

NUM_GPUS=$SLURM_GPUS_PER_NODE

SIZES=(128 256 512 1024 1280 1536 1792 2048)
SIZE=${SIZES[$SLURM_ARRAY_TASK_ID]}

SCRIPT_DIR="/home/ncardel/repos/mbirjax/experiments/sharding"
LOG_DIR="${SCRIPT_DIR}/logs"
exec > "${LOG_DIR}/recon_${SIZE}x${SIZE}x${SIZE}_${NUM_GPUS}gpu.out" \
     2>"${LOG_DIR}/recon_${SIZE}x${SIZE}x${SIZE}_${NUM_GPUS}gpu.err"
DATA_OUTPUT_FILEPATH="${SCRIPT_DIR}/recon_time_${SIZE}x${SIZE}x${SIZE}_${NUM_GPUS}gpu.txt"

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
