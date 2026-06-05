#!/bin/bash
#SBATCH --job-name=eps0FrameJumps
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=2G
#SBATCH --nodes=1
#SBATCH --output=logs/check_eps0_frame_jumps_%j.out

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

JOB_START_TS=$(date +%s)

log "Step 1/8: Purging modules"
module purge
log "Step 2/8: Sourcing ~/.bashrc"
source ~/.bashrc
log "Step 3/8: Loading anaconda module"
module load anaconda3/2025.12
log "Step 4/8: Activating conda environment"
conda activate '/home/mc7345/micromamba/envs/reactiveLJ_paper'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
if [[ ! -f "${WORK_DIR}/tests/check_eps0_frame_jumps.py" ]]; then
    WORK_DIR="${SCRIPT_DIR}"
fi
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
REPO_ROOT="$(cd "${WORK_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Keep any numerical libraries from oversubscribing the Slurm allocation.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

log "Step 5/8: Changing directory to ${WORK_DIR}"
cd "${WORK_DIR}"
log "Step 6/8: Ensuring logs directory exists"
mkdir -p logs

WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
EPSILON="${EPSILON:-0}"
EPSILON_DIR="${EPSILON_DIR:-eps_${EPSILON}}"
FAIL_ON="${FAIL_ON:-nonuniform}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

log "Step 7/8: Running ${EPSILON_DIR} frame-jump check"
log "workers=${WORKERS} fail_on=${FAIL_ON} extra_args=${EXTRA_ARGS:-<none>}"
CHECK_START_TS=$(date +%s)
set +e
python -u "${WORK_DIR}/tests/check_eps0_frame_jumps.py" \
    --roots "${WORK_DIR}/data_generation/outputs" "${WORK_DIR}/data_generation/outputs_clean" \
    --epsilon-dir "${EPSILON_DIR}" \
    --workers "${WORKERS}" \
    --fail-on "${FAIL_ON}" \
    ${EXTRA_ARGS}
CHECK_STATUS=$?
set -e
CHECK_END_TS=$(date +%s)
log "Step 7/8 complete: checker exited with status ${CHECK_STATUS} after $((CHECK_END_TS - CHECK_START_TS)) seconds"

JOB_END_TS=$(date +%s)
log "Step 8/8: Job complete in $((JOB_END_TS - JOB_START_TS)) seconds total"
exit "${CHECK_STATUS}"
