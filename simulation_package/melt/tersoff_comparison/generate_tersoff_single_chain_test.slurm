#!/bin/bash
#SBATCH --job-name=tersoff_single_chain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:02:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=400M
#SBATCH --array=0-1
#SBATCH --output=logs/generate_tersoff_single_chain_%A_%a.out

module purge
source ~/.bashrc
module load anaconda3/2025.12
conda activate '/home/mc7345/micromamba/envs/reactiveLJ_paper'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERSOFF_DIR="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
if [[ ! -f "${TERSOFF_DIR}/run_tersoff_single_chain_test.py" ]]; then
    TERSOFF_DIR="${SCRIPT_DIR}"
fi
TERSOFF_DIR="$(cd "${TERSOFF_DIR}" && pwd)"
WORK_DIR="$(cd "${TERSOFF_DIR}/.." && pwd)"
cd "${WORK_DIR}"
mkdir -p "${TERSOFF_DIR}/logs"

EPSILONS=(9 15)
TASK_ID=${SLURM_ARRAY_TASK_ID}
if [[ ${TASK_ID} -ge ${#EPSILONS[@]} ]]; then
    echo "Task ${TASK_ID} exceeds epsilon list length; exiting."
    exit 0
fi

EPS=${EPSILONS[$TASK_ID]}
FRAME_STEPS=${FRAME_STEPS:-10000}
RUN_STEPS=${RUN_STEPS:-1000000}
TERSOFF_PARAMS_CSV=${TERSOFF_PARAMS_CSV:-"${TERSOFF_DIR}/outputs/tersoff_fitted_params.csv"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${TERSOFF_DIR}/outputs/TEST_SINGLE_CHAIN"}

python "${TERSOFF_DIR}/run_tersoff_single_chain_test.py" \
    --epsilon "${EPS}" \
    --replicate 1 \
    --tersoff-params-csv "${TERSOFF_PARAMS_CSV}" \
    --output-root "${OUTPUT_ROOT}" \
    --frame-steps "${FRAME_STEPS}" \
    --run-steps "${RUN_STEPS}" \
    --device gpu
