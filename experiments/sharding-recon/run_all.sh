#!/usr/bin/env bash
#
# One-shot launcher: generate all cached data, then run the full 1/2/4-GPU timing sweep.
#
# It submits prepare_data.sh (the data-generation array) and then submits the three timing
# sweeps with a SLURM dependency so each timing task starts only once its data exists. Nothing
# blocks your shell -- everything is queued and runs unattended; watch progress with `squeue -u
# $USER` and results land in results/.
#
# Usage:
#   ./run_all.sh                # prep + 1,2,4-GPU sweeps (default)
#   ./run_all.sh --no-prep      # skip data gen (data already cached); just run the sweeps
#   ./run_all.sh --gpus "1 4"   # only run the listed GPU counts (default: "1 2 4")
#   ./run_all.sh --no-prep --gpus 4
#
# After the runs finish (or any time), build the figures + report:
#   python plot_results.py
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DO_PREP=1
GPU_COUNTS="1 2 4"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-prep) DO_PREP=0; shift ;;
    --gpus)    GPU_COUNTS="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${SCRIPT_DIR}/logs" "${SCRIPT_DIR}/results"

# --- 1. Data generation (once) ------------------------------------------------------------
DEP_ARG=()
if [[ "${DO_PREP}" -eq 1 ]]; then
  PREP_JOBID="$(sbatch --parsable "${SCRIPT_DIR}/prepare_data.sh")"
  echo "Submitted data generation: job ${PREP_JOBID} (prepare_data.sh, array 0-18)"
  # aftercorr: timing task i starts as soon as prep task i (its own shape) finishes, so small
  # shapes begin timing without waiting for the largest data to finish generating.
  DEP_ARG=(--dependency="aftercorr:${PREP_JOBID}")
else
  echo "Skipping data generation (--no-prep); assuming caches already exist."
fi

# --- 2. Timing sweeps (1 / 2 / 4 GPU) -----------------------------------------------------
for N in ${GPU_COUNTS}; do
  SCRIPT="${SCRIPT_DIR}/recon_slurm_${N}gpu.sh"
  if [[ ! -f "${SCRIPT}" ]]; then
    echo "WARNING: ${SCRIPT} not found; skipping ${N}-GPU sweep." >&2
    continue
  fi
  JOBID="$(sbatch --parsable ${DEP_ARG[@]+"${DEP_ARG[@]}"} "${SCRIPT}")"
  echo "Submitted ${N}-GPU sweep: job ${JOBID} (recon_slurm_${N}gpu.sh, array 0-18)"
done

echo
echo "All jobs queued. Monitor with:  squeue -u \$USER"
echo "When finished, build the report:  python ${SCRIPT_DIR}/plot_results.py"
