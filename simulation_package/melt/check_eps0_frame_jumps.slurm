#!/bin/bash
#SBATCH --job-name=eps0FrameJumps
#SBATCH --time=02:00:00
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
conda activate '/home/mc7345/anaconda3/envs/reactiveLJ_paper'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
if [[ ! -f "${WORK_DIR}/tests/check_eps0_frame_jumps.py" ]]; then
    WORK_DIR="${SCRIPT_DIR}"
fi
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
REPO_ROOT="$(cd "${WORK_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

log "Step 5/8: Changing directory to ${WORK_DIR}"
cd "${WORK_DIR}"
log "Step 6/8: Ensuring logs and test output directories exist"
mkdir -p logs tests

WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
CSV_OUT="${CSV_OUT:-${WORK_DIR}/tests/frame_jump_report_eps0_${SLURM_JOB_ID:-manual}.csv}"
FAIL_ON="${FAIL_ON:-nonuniform}"

log "Step 7/8: Running eps=0 frame-step continuity check"
log "workers=${WORKERS} fail_on=${FAIL_ON} csv_out=${CSV_OUT}"
CHECK_START_TS=$(date +%s)
python -u "${WORK_DIR}/tests/check_eps0_frame_jumps.py" \
    --roots "${WORK_DIR}/data_generation/outputs" "${WORK_DIR}/data_generation/outputs_clean" \
    --epsilon-dir eps_0 \
    --files trajectory.gsd msd_trajectory.gsd virial_tensor_log.gsd \
    --workers "${WORKERS}" \
    --csv-out "${CSV_OUT}" \
    --fail-on "${FAIL_ON}"
CHECK_END_TS=$(date +%s)
log "Step 7/8 complete: Checker finished in $((CHECK_END_TS - CHECK_START_TS)) seconds"

JOB_END_TS=$(date +%s)
log "Step 8/8: Job complete in $((JOB_END_TS - JOB_START_TS)) seconds total"
