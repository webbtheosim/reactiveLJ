#!/bin/bash
#SBATCH --job-name=reactiveLJ_mpcd_gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --array=0-49%11
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/generate_data_%A_%a.out

# NOTE:
# - This script assumes 5 epsilon values and 10 replicates each (50 total tasks).
# - If you change N_REP below, update the --array range accordingly.
# Set 0-49 to run all replicates.

module purge
source ~/.bashrc
module load anaconda3/2025.12
conda activate '/home/mc7345/anaconda3/envs/reactiveLJ_paper'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
if [[ ! -f "${WORK_DIR}/data_generation/run_reactive_lj.py" ]]; then
    WORK_DIR="${SCRIPT_DIR}"
fi
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
cd "${WORK_DIR}"
mkdir -p logs

EPSILONS=(0 6 12 15 18)
N_REP=${N_REP:-10}
TOTAL=$(( ${#EPSILONS[@]} * N_REP ))

# Optional overrides (set as environment variables before submission).
FRAME_STEPS=${FRAME_STEPS:-200000}
UNSTICKY_EQ_STEPS=${UNSTICKY_EQ_STEPS:-100000}
REACTIVE_EQ_STEPS=${REACTIVE_EQ_STEPS:-1000000}
STICKY_BURNIN_STEPS=${STICKY_BURNIN_STEPS:-10000000}
PRODUCTION_TAU_R0=${PRODUCTION_TAU_R0:-3000}
VIRIAL_LOG_STEPS=${VIRIAL_LOG_STEPS:-5000}
WEAKENING_EXPONENT=${WEAKENING_EXPONENT:-4}
POLYMER_WEIGHT_FRACTION=${POLYMER_WEIGHT_FRACTION:-0.02}
MPCD_NUMBER_DENSITY=${MPCD_NUMBER_DENSITY:-5.0}
COLLISION_PERIOD=${COLLISION_PERIOD:-20}
WALLTIME_CAP_SECONDS=${WALLTIME_CAP_SECONDS:-86340}
WALLTIME_SAFETY_BUFFER_SECONDS=${WALLTIME_SAFETY_BUFFER_SECONDS:-60}
PRODUCTION_CHUNK_STEPS=${PRODUCTION_CHUNK_STEPS:-1000000}

TASK_ID=${SLURM_ARRAY_TASK_ID}
if [[ ${TASK_ID} -ge ${TOTAL} ]]; then
    echo "Task ${TASK_ID} exceeds TOTAL=${TOTAL}; exiting."
    exit 0
fi

EPS_INDEX=$(( TASK_ID / N_REP ))
REP_INDEX=$(( TASK_ID % N_REP ))
EPS=${EPSILONS[$EPS_INDEX]}
REP=$(( REP_INDEX + 1 ))

set +e
python "${WORK_DIR}/data_generation/run_reactive_lj.py" \
    --epsilon "${EPS}" \
    --replicate "${REP}" \
    --output-root "${WORK_DIR}/data_generation/outputs" \
    --weakening-exponent "${WEAKENING_EXPONENT}" \
    --polymer-weight-fraction "${POLYMER_WEIGHT_FRACTION}" \
    --mpcd-number-density "${MPCD_NUMBER_DENSITY}" \
    --collision-period "${COLLISION_PERIOD}" \
    --frame-steps "${FRAME_STEPS}" \
    --virial-log-steps "${VIRIAL_LOG_STEPS}" \
    --unsticky-equil-steps "${UNSTICKY_EQ_STEPS}" \
    --reactive-equil-steps "${REACTIVE_EQ_STEPS}" \
    --sticky-burnin-steps "${STICKY_BURNIN_STEPS}" \
    --production-runtime-tau-r0 "${PRODUCTION_TAU_R0}" \
    --production-chunk-steps "${PRODUCTION_CHUNK_STEPS}" \
    --walltime-limit-seconds "${WALLTIME_CAP_SECONDS}" \
    --walltime-safety-buffer-seconds "${WALLTIME_SAFETY_BUFFER_SECONDS}" \
    --resume
status=$?
set -e

if [[ ${status} -eq 3 ]]; then
    REQUEUE_TARGET="${SLURM_JOB_ID}"
    if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        REQUEUE_TARGET="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    fi
    echo "Checkpointed before walltime cap; requeueing ${REQUEUE_TARGET}."
    scontrol requeue "${REQUEUE_TARGET}"
    exit 0
fi

exit "${status}"
