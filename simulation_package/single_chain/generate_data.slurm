#!/bin/bash
#SBATCH --job-name=reactiveLJ_single_chain_gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-49%24
#SBATCH --output=logs/generate_data_%A_%a.out

# NOTE:
# - This script assumes 5 epsilon values and 10 replicates each (50 total tasks).
# - If you change N_REP below, update the --array range accordingly.

module purge
source ~/.bashrc
module load anaconda3/2025.12
conda activate '/home/mc7345/micromamba/envs/reactiveLJ_paper'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
if [[ ! -f "${WORK_DIR}/data_generation/run_reactive_lj.py" ]]; then
    ALT_WORK_DIR="${SLURM_SUBMIT_DIR:-}/simulation_package/single_chain"
    if [[ -f "${ALT_WORK_DIR}/data_generation/run_reactive_lj.py" ]]; then
        WORK_DIR="${ALT_WORK_DIR}"
    else
        WORK_DIR="${SCRIPT_DIR}"
    fi
fi
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
if [[ ! -f "${WORK_DIR}/data_generation/run_reactive_lj.py" ]]; then
    echo "ERROR: Could not resolve single_chain work directory."
    echo "Submit from simulation_package/single_chain or repo root."
    exit 1
fi
cd "${WORK_DIR}"
mkdir -p logs

EPSILONS=(0 6 12 15 18)
N_REP=${N_REP:-10}
TOTAL=$(( ${#EPSILONS[@]} * N_REP ))

# Optional overrides (set as environment variables before submission).
FRAME_STEPS=${FRAME_STEPS:-10000}
UNSTICKY_EQ_STEPS=${UNSTICKY_EQ_STEPS:-100000}
REACTIVE_EQ_STEPS=${REACTIVE_EQ_STEPS:-1000000}
PROD_STEPS=${PROD_STEPS:-10000000}
# Use a substantial no-output pre-production equilibration, but shorter than the
# final production window since the single-chain workflow is not targeting melt
# stress relaxation.
PRE_PROD_EQ_STEPS=${PRE_PROD_EQ_STEPS:-5000000}
WEAKENING_EXPONENT=${WEAKENING_EXPONENT:-4}
CHAIN_LENGTH=${CHAIN_LENGTH:-400}
STICKERS_PER_CHAIN=${STICKERS_PER_CHAIN:-40}
BOX_LENGTH=${BOX_LENGTH:-100}

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
    --chain-length "${CHAIN_LENGTH}" \
    --stickers-per-chain "${STICKERS_PER_CHAIN}" \
    --box-length "${BOX_LENGTH}" \
    --weakening-exponent "${WEAKENING_EXPONENT}" \
    --frame-steps "${FRAME_STEPS}" \
    --unsticky-equil-steps "${UNSTICKY_EQ_STEPS}" \
    --reactive-equil-steps "${REACTIVE_EQ_STEPS}" \
    --pre-production-equil-steps "${PRE_PROD_EQ_STEPS}" \
    --production-steps "${PROD_STEPS}"
