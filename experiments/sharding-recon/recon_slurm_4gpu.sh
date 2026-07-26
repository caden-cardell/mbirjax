#!/usr/bin/env bash

#SBATCH --job-name=recon_4gpu
#SBATCH -A bouman -p ai -q normal
#SBATCH -t 03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=56 --gpus-per-node=4
#SBATCH --array=0-18
#SBATCH --output="/home/ncardel/repos/mbirjax/experiments/sharding-recon/logs/slurm-%A_%a.out"
#SBATCH --error="/home/ncardel/repos/mbirjax/experiments/sharding-recon/logs/slurm-%A_%a.err"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ncardel@purdue.edu

SCRIPT_DIR="/home/ncardel/repos/mbirjax/experiments/sharding-recon"
NUM_GPUS=$SLURM_GPUS_PER_NODE

source "${SCRIPT_DIR}/configs.sh"
read -r VIEWS ROWS CHANNELS <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

LOG_DIR="${SCRIPT_DIR}/logs"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
exec  > "${LOG_DIR}/recon_${VIEWS}x${ROWS}x${CHANNELS}_${NUM_GPUS}gpu.out" \
     2> "${LOG_DIR}/recon_${VIEWS}x${ROWS}x${CHANNELS}_${NUM_GPUS}gpu.err"
# One CSV per (gpu-count, shape): concurrent array tasks never touch the same file, and
# re-submissions on later nights append repeats. plot_results.py globs results/*.csv.
DATA_OUTPUT_FILEPATH="${RESULTS_DIR}/recon_${NUM_GPUS}gpu_${VIEWS}x${ROWS}x${CHANNELS}.csv"

module purge
module load proxy
module load modtree/gpu
module load cuda/12.9.0
module use /depot/bouman/apps/modules
module load cudnn/9.11.0
module load conda

conda activate mbirjax

export GIT_COMMIT="$(git -C "${SCRIPT_DIR}" rev-parse --short HEAD 2>/dev/null)"
echo "Running: shape=${VIEWS}x${ROWS}x${CHANNELS}, num_gpus=${NUM_GPUS}, commit=${GIT_COMMIT}, output=${DATA_OUTPUT_FILEPATH}"
python3 "${SCRIPT_DIR}/recon.py" "$VIEWS" "$ROWS" "$CHANNELS" "$DATA_OUTPUT_FILEPATH"
