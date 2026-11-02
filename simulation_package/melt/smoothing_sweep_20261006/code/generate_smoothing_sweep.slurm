#!/bin/bash
#SBATCH --job-name=RLJ_smoothing
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --array=0-251%11
#SBATCH --requeue

# The submitted campaign manifest and source snapshot fix all physical settings.
# Activate the environment containing the pinned HOOMD build before submission.
set -euo pipefail

: "${SWEEP_MANIFEST:?Set SWEEP_MANIFEST to the campaign manifest}"
SWEEP_CODE_DIR="$(dirname "${SWEEP_MANIFEST}")/code"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

set +e
"${PYTHON:-python}" -u "${SWEEP_CODE_DIR}/data_generation/run_smoothing_sweep.py" \
    --manifest "${SWEEP_MANIFEST}"
RUN_STATUS=$?
set -e

if [[ ${RUN_STATUS} -eq 3 ]]; then
    if [[ ${SLURM_RESTART_COUNT:-0} -ge 3 ]]; then
        echo "Checkpoint saved, but this task reached its automatic retry limit." >&2
        exit 3
    fi
    echo "Checkpoint saved before walltime; requeueing task ${SLURM_JOB_ID}."
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
fi
exit "${RUN_STATUS}"
