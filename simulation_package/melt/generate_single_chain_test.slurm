#!/bin/bash
#SBATCH --job-name=reactiveLJ_single_chain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:02:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=400M
#SBATCH --array=0-1%1
#SBATCH --output=logs/generate_single_chain_%A_%a.out

module purge
source ~/.bashrc
module load anaconda3/2025.12
conda activate '/home/mc7345/micromamba/envs/reactiveLJ_paper'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}"
cd "${WORK_DIR}"
mkdir -p logs

EPSILONS=(10)
WEAKENING_EXPONENTS=(0 4)

TASK_ID=${SLURM_ARRAY_TASK_ID}
if [[ ${TASK_ID} -ge ${#WEAKENING_EXPONENTS[@]} ]]; then
    echo "Task ${TASK_ID} exceeds weakening exponent list length; exiting."
    exit 0
fi

EPS=${EPSILONS[0]}
WEXP=${WEAKENING_EXPONENTS[$TASK_ID]}
FRAME_STEPS=${FRAME_STEPS:-10000}
RUN_STEPS=${RUN_STEPS:-1000000}
OUTPUT_ROOT="${WORK_DIR}/data_generation/outputs/TEST/wexp_${WEXP}"

python "${WORK_DIR}/data_generation/run_reactive_lj_single_chain_test.py" \
    --epsilon "${EPS}" \
    --weakening-exponent "${WEXP}" \
    --replicate 1 \
    --output-root "${OUTPUT_ROOT}" \
    --frame-steps "${FRAME_STEPS}" \
    --run-steps "${RUN_STEPS}"
