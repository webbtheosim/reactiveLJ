#!/bin/bash
#SBATCH --job-name=reactiveLJ_gen_eps0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --array=0-9%5
#SBATCH --output=logs/generate_data_eps0_%A_%a.out

# NOTE:
# - This script runs epsilon=0 only.
# - Default replicate count is N_REP=10 (rep_001 .. rep_010).
# - If you change N_REP, update --array accordingly.

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

EPS=0
N_REP=${N_REP:-10}

# Optional overrides (set as environment variables before submission).
FRAME_STEPS=${FRAME_STEPS:-10000}
UNSTICKY_EQ_STEPS=${UNSTICKY_EQ_STEPS:-100000}
REACTIVE_EQ_STEPS=${REACTIVE_EQ_STEPS:-1000000}
PRODUCTION_TAU_R0=${PRODUCTION_TAU_R0:-250}
VIRIAL_LOG_STEPS=${VIRIAL_LOG_STEPS:-1000}
WEAKENING_EXPONENT=${WEAKENING_EXPONENT:-4}

TASK_ID=${SLURM_ARRAY_TASK_ID}
if [[ ${TASK_ID} -ge ${N_REP} ]]; then
    echo "Task ${TASK_ID} exceeds N_REP=${N_REP}; exiting."
    exit 0
fi

REP=$(( TASK_ID + 1 ))

python "${WORK_DIR}/data_generation/run_reactive_lj.py" \
    --epsilon "${EPS}" \
    --replicate "${REP}" \
    --output-root "${WORK_DIR}/data_generation/outputs" \
    --weakening-exponent "${WEAKENING_EXPONENT}" \
    --frame-steps "${FRAME_STEPS}" \
    --virial-log-steps "${VIRIAL_LOG_STEPS}" \
    --unsticky-equil-steps "${UNSTICKY_EQ_STEPS}" \
    --reactive-equil-steps "${REACTIVE_EQ_STEPS}" \
    --production-runtime-tau-r0 "${PRODUCTION_TAU_R0}"
