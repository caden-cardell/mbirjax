#!/usr/bin/env bash

# ---- GPU configuration: uncomment ONE #SBATCH block and its matching body block ----

#SBATCH  --ntasks=14 --gpus-per-node=1   # 1 GPU   <-- ACTIVE
##SBATCH --ntasks=14 --gpus-per-node=2   # 2 GPUs
##SBATCH --ntasks=28 --gpus-per-node=4   # 4 GPUs
##SBATCH --ntasks=42 --gpus-per-node=5   # 5 GPUs
##SBATCH --ntasks=56 --gpus-per-node=7   # 7 GPUs
##SBATCH --ntasks=56 --gpus-per-node=8   # 8 GPUs

#SBATCH --job-name=recon
#SBATCH -A bouman -p ai -q normal
#SBATCH -t 04:00:00
#SBATCH --nodes=1
#SBATCH --array=0-1
#SBATCH --output="/home/ncardel/repos/mbirjax/experiments/sharding/logs/slurm-%A_%a.out"
#SBATCH --error="/home/ncardel/repos/mbirjax/experiments/sharding/logs/slurm-%A_%a.err"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ncardel@purdue.edu

# ---- GPU body config: uncomment the ONE block matching your #SBATCH above ----

NUM_GPUS=1; export CUDA_VISIBLE_DEVICES=0                 # <-- ACTIVE
#NUM_GPUS=2; export CUDA_VISIBLE_DEVICES=0,1
#NUM_GPUS=4; export CUDA_VISIBLE_DEVICES=0,1,2,3
#NUM_GPUS=5; export CUDA_VISIBLE_DEVICES=0,1,2,3,4
#NUM_GPUS=7; export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
#NUM_GPUS=8; export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ---- Size array: each SLURM array task runs one size ----
SIZES=(128 256 512 1024 1280 1536 1792 2048)
SIZE=${SIZES[$SLURM_ARRAY_TASK_ID]}

SCRIPT_DIR="/home/ncardel/repos/mbirjax/experiments/sharding"
DATA_OUTPUT_FILEPATH="${SCRIPT_DIR}/recon_time_${NUM_GPUS}gpu.txt"

# ---- Environment setup ----
module purge
module load proxy
module load modtree/gpu
module load cuda/12.9.0
module use /depot/bouman/apps/modules
module load cudnn/9.11.0
module load conda

conda activate mbirjax

# ---- Run ----
echo "Running: size=$SIZE, num_gpus=$NUM_GPUS, output=$DATA_OUTPUT_FILEPATH"
python3 "${SCRIPT_DIR}/recon.py" "$SIZE" "$SIZE" "$SIZE" "$DATA_OUTPUT_FILEPATH" "$NUM_GPUS"
