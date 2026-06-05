#!/bin/bash
#SBATCH --job-name=cleanReactiveLJ
#SBATCH --time=01:01:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=500M
#SBATCH --nodes=1
#SBATCH --output=logs/clean_requeued_outputs_%j.out

# Defaults to writing cleaned data to data_generation/outputs_clean.
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
conda activate '/home/mc7345/micromamba/envs/reactiveLJ_paper'

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
TARGET_P_ROOTS=(p_2 p_8)

log "Step 7/8: Running cleaner"
CLEAN_START_TS=$(date +%s)
for target_p_root in "${TARGET_P_ROOTS[@]}"; do
    target_source_root="${SOURCE_ROOT}/${target_p_root}"
    target_clean_root="${CLEAN_ROOT}/${target_p_root}"
    target_raw_root="${RAW_ROOT}/${target_p_root}"

    clean_args=(
        --source-root "${target_source_root}"
        --clean-root "${target_clean_root}"
        --raw-root "${target_raw_root}"
        --forward-gap-policy segment
        --workers "${WORKERS}"
    )

    if [[ "${APPLY}" == "1" ]]; then
        clean_args+=(--apply)
    fi
    if [[ "${SWAP}" == "1" ]]; then
        clean_args+=(--swap)
    fi

    log "Policy=segment target=virial_tensor_log.gsd source=${target_source_root} clean=${target_clean_root} workers=${WORKERS} apply=${APPLY} swap=${SWAP}"
    python -u "${WORK_DIR}/data_generation/clean_requeued_outputs.py" "${clean_args[@]}"
done
CLEAN_END_TS=$(date +%s)
log "Step 7/8 complete: Cleaner finished in $((CLEAN_END_TS - CLEAN_START_TS)) seconds"

JOB_END_TS=$(date +%s)
log "Step 8/8: Job complete in $((JOB_END_TS - JOB_START_TS)) seconds total"
