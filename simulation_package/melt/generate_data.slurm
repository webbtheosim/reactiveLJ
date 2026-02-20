#!/bin/bash
#SBATCH --job-name=reactiveLJ_gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:02:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --array=0-59%24
#SBATCH --output=logs/generate_data_%A_%a.out

# NOTE:
# - This script assumes 6 epsilon values and 10 replicates each (60 total tasks).
# - If you change N_REP below, update the --array range accordingly.
# Set 0-59 to run all replicates.

module purge
source ~/.bashrc
module load anaconda3/2025.12
conda activate '/home/mc7345/micromamba/envs/reactiveLJ_paper'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
if [[ ! -f "${WORK_DIR}/data_generation/run_reactive_lj.py" ]]; then
    WORK_DIR="${SCRIPT_DIR}"
fi
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
cd "${WORK_DIR}"
mkdir -p logs

EPSILONS=(3 6 9 12 15 18)
N_REP=${N_REP:-10}
TOTAL=$(( ${#EPSILONS[@]} * N_REP ))

# Optional overrides (set as environment variables before submission).
FRAME_STEPS=${FRAME_STEPS:-10000}
UNSTICKY_EQ_STEPS=${UNSTICKY_EQ_STEPS:-100000}
REACTIVE_EQ_STEPS=${REACTIVE_EQ_STEPS:-1000000}
PROD_STEPS=${PROD_STEPS:-1000000}
WEAKENING_EXPONENT=${WEAKENING_EXPONENT:-4}

TASK_ID=${SLURM_ARRAY_TASK_ID}
if [[ ${TASK_ID} -ge ${TOTAL} ]]; then
    echo "Task ${TASK_ID} exceeds TOTAL=${TOTAL}; exiting."
    exit 0
fi

EPS_INDEX=$(( TASK_ID / N_REP ))
REP_INDEX=$(( TASK_ID % N_REP ))
EPS=${EPSILONS[$EPS_INDEX]}
REP=$(( REP_INDEX + 1 ))

python "${WORK_DIR}/data_generation/run_reactive_lj.py" \
    --epsilon "${EPS}" \
    --replicate "${REP}" \
    --output-root "${WORK_DIR}/data_generation/outputs" \
    --weakening-exponent "${WEAKENING_EXPONENT}" \
    --frame-steps "${FRAME_STEPS}" \
    --unsticky-equil-steps "${UNSTICKY_EQ_STEPS}" \
    --reactive-equil-steps "${REACTIVE_EQ_STEPS}" \
    --production-steps "${PROD_STEPS}"
