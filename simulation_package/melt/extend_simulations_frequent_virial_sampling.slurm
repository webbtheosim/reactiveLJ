#!/bin/bash
#SBATCH --job-name=reactiveLJ_virial_ext
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --array=0-58%11
#SBATCH --open-mode=append
#SBATCH --output=logs/extend_virial_%A_%a.out

# Dense short extension for stress-relaxation analysis.
# The default array range matches the current 59 clean output directories.
# If outputs_clean changes, update --array or let out-of-range tasks exit.

module purge
source ~/.bashrc
module load anaconda3/2025.12
conda activate '/home/mc7345/micromamba/envs/reactiveLJ_paper'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
if [[ ! -f "${WORK_DIR}/data_generation/extend_virial_trajectory.py" ]]; then
    WORK_DIR="${SCRIPT_DIR}"
fi
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
cd "${WORK_DIR}"
mkdir -p logs

INPUT_ROOT=${INPUT_ROOT:-${WORK_DIR}/data_generation/outputs_clean}
OUTPUT_ROOT=${OUTPUT_ROOT:-${WORK_DIR}/data_generation/outputs_virial_extended}
EXTENSION_TAU_R0=${EXTENSION_TAU_R0:-10}
VIRIAL_LOG_STEPS=${VIRIAL_LOG_STEPS:-100}

python "${WORK_DIR}/data_generation/extend_virial_trajectory.py" \
    --input-root "${INPUT_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --task-index "${SLURM_ARRAY_TASK_ID}" \
    --extension-runtime-tau-r0 "${EXTENSION_TAU_R0}" \
    --virial-log-steps "${VIRIAL_LOG_STEPS}"
