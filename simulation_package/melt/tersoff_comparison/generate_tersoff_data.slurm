#!/bin/bash
#SBATCH --job-name=tersoff_gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:02:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --array=0-39%5
#SBATCH --output=/scratch/gpfs/WEBB/chertok/REACTIVELJ_PAPER/simulation_package/melt/tersoff_comparison/logs/generate_tersoff_data_%A_%a.out

# NOTE:
# - This script assumes 4 epsilon values (6, 12, 15, 18) and 10 replicates each (40 total tasks).
# - If you change EPSILONS or N_REP below, update the --array range accordingly.
# Set 0-39 to run all replicates.

module purge
source ~/.bashrc
module load anaconda3/2025.12
conda activate '/scratch/gpfs/WEBB/chertok/conda_envs/reactiveLJ_oconnor_tersoff_comparison'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERSOFF_DIR="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
if [[ ! -f "${TERSOFF_DIR}/generate_tersoff_data.py" ]]; then
    TERSOFF_DIR="${SCRIPT_DIR}"
fi
TERSOFF_DIR="$(cd "${TERSOFF_DIR}" && pwd)"
WORK_DIR="$(cd "${TERSOFF_DIR}/.." && pwd)"
cd "${WORK_DIR}"
mkdir -p "${TERSOFF_DIR}/logs"
source "${TERSOFF_DIR}/slurm_resource_logging.sh"

EPSILONS=(6 12 15 18)
N_REP=${N_REP:-10}
TOTAL=$(( ${#EPSILONS[@]} * N_REP ))

# Optional overrides (set as environment variables before submission).
FRAME_STEPS=${FRAME_STEPS:-10000}
UNSTICKY_EQ_STEPS=${UNSTICKY_EQ_STEPS:-100000}
TERSOFF_EQ_STEPS=${TERSOFF_EQ_STEPS:-1000000}
PROD_STEPS=${PROD_STEPS:-1000000}
TERSOFF_PARAMS_CSV=${TERSOFF_PARAMS_CSV:-"${TERSOFF_DIR}/outputs/tersoff_fitted_params.csv"}

TASK_ID=${SLURM_ARRAY_TASK_ID}
if [[ ${TASK_ID} -ge ${TOTAL} ]]; then
    echo "Task ${TASK_ID} exceeds TOTAL=${TOTAL}; exiting."
    exit 0
fi

EPS_INDEX=$(( TASK_ID / N_REP ))
REP_INDEX=$(( TASK_ID % N_REP ))
EPS=${EPSILONS[$EPS_INDEX]}
REP=$(( REP_INDEX + 1 ))

resource_logging_init "${TERSOFF_DIR}/logs" "tersoff" "${EPS}" "${REP}"

resource_run_command python "${TERSOFF_DIR}/generate_tersoff_data.py" \
    --epsilon "${EPS}" \
    --replicate "${REP}" \
    --tersoff-params-csv "${TERSOFF_PARAMS_CSV}" \
    --output-root "${TERSOFF_DIR}/outputs" \
    --frame-steps "${FRAME_STEPS}" \
    --unsticky-equil-steps "${UNSTICKY_EQ_STEPS}" \
    --tersoff-equil-steps "${TERSOFF_EQ_STEPS}" \
    --production-steps "${PROD_STEPS}"
