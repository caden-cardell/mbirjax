#!/usr/bin/env bash

# Pre-generate (once) the cached phantom + sinogram + params for every config in the sweep, so
# the timing nights never pay for data generation inside the timed region. Forward-projecting the
# larger phantoms is slow, hence the long walltime. This calls create_recon_data() only -- it does
# NOT time a recon -- by importing recon.py and invoking the generator directly.
#
# Run this ONCE (Night 0) before submitting any recon_slurm_*gpu.sh sweep.

#SBATCH --job-name=recon_prep
#SBATCH -A bouman -p ai -q normal
#SBATCH -t 08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=14 --gpus-per-node=1
#SBATCH --array=0-18
#SBATCH --output="/home/ncardel/repos/mbirjax/experiments/sharding-recon/logs/prep-%A_%a.out"
#SBATCH --error="/home/ncardel/repos/mbirjax/experiments/sharding-recon/logs/prep-%A_%a.err"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ncardel@purdue.edu

SCRIPT_DIR="/home/ncardel/repos/mbirjax/experiments/sharding-recon"

source "${SCRIPT_DIR}/configs.sh"
read -r VIEWS ROWS CHANNELS <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

module purge
module load proxy
module load modtree/gpu
module load cuda/12.9.0
module use /depot/bouman/apps/modules
module load cudnn/9.11.0
module load conda

conda activate mbirjax

echo "Generating data for shape=${VIEWS}x${ROWS}x${CHANNELS}"
python3 -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); import recon; recon.create_recon_data(${VIEWS}, ${ROWS}, ${CHANNELS})"
