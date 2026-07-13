#!/bin/bash
#SBATCH --job-name=cleanMPCD
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=1G
#SBATCH --nodes=1
#SBATCH --output=logs/clean_requeued_outputs_%j.out

# Defaults to writing cleaned active data to data_generation/outputs_clean.
# Set APPLY=0 for a dry run. SWAP is intentionally off by default so the raw
# data_generation/outputs tree remains untouched.

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
if [[ ! -f "${WORK_DIR}/data_generation/clean_requeued_outputs.py" ]]; then
    WORK_DIR="${SCRIPT_DIR}"
fi
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
REPO_ROOT="$(cd "${WORK_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

log "Step 5/8: Changing directory to ${WORK_DIR}"
cd "${WORK_DIR}"
log "Step 6/8: Ensuring logs directory exists"
mkdir -p logs

SOURCE_ROOT="${SOURCE_ROOT:-${WORK_DIR}/data_generation/outputs}"
CLEAN_ROOT="${CLEAN_ROOT:-${WORK_DIR}/data_generation/outputs_clean}"
RAW_ROOT="${RAW_ROOT:-${WORK_DIR}/data_generation/outputs_raw_bad_overlaps}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
APPLY="${APPLY:-1}"
SWAP="${SWAP:-0}"
FORWARD_GAP_POLICY="${FORWARD_GAP_POLICY:-segment}"
TARGET_FILES="${TARGET_FILES:-trajectory.gsd virial_tensor_log.gsd}"
read -r -a TARGET_FILES_ARRAY <<< "${TARGET_FILES}"

clean_args=(
    --source-root "${SOURCE_ROOT}"
    --clean-root "${CLEAN_ROOT}"
    --raw-root "${RAW_ROOT}"
    --forward-gap-policy "${FORWARD_GAP_POLICY}"
    --workers "${WORKERS}"
    --target-files "${TARGET_FILES_ARRAY[@]}"
)

if [[ "${APPLY}" == "1" ]]; then
    clean_args+=(--apply)
fi
if [[ "${SWAP}" == "1" ]]; then
    clean_args+=(--swap)
fi

log "Step 7/8: Running MPCD cleaner"
log "Policy=${FORWARD_GAP_POLICY} target=${TARGET_FILES} workers=${WORKERS} apply=${APPLY} swap=${SWAP}"
CLEAN_START_TS=$(date +%s)
python -u "${WORK_DIR}/data_generation/clean_requeued_outputs.py" "${clean_args[@]}"
CLEAN_END_TS=$(date +%s)
log "Step 7/8 complete: Cleaner finished in $((CLEAN_END_TS - CLEAN_START_TS)) seconds"

JOB_END_TS=$(date +%s)
log "Step 8/8: Job complete in $((JOB_END_TS - JOB_START_TS)) seconds total"
