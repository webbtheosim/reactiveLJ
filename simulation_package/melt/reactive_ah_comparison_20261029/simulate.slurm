#!/bin/bash
#SBATCH --job-name=AH_LJ_matched
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-23%24
#SBATCH --requeue
set -euo pipefail
: "${AH_COMPARISON_ROOT:?Set the add-on directory}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd "$AH_COMPARISON_ROOT"
set +e
"${PYTHON:-python}" -u code/run_comparison.py
result=$?
set -e
if [[ "$result" == 3 && "${SLURM_RESTART_COUNT:-0}" -lt 3 ]]; then
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    exit 0
fi
exit "$result"
