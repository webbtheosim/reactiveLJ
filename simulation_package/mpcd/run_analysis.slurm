#!/bin/bash
#SBATCH --job-name=reactiveLJ_mpcd_analysis
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=11
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --output=logs/run_analysis_%j.out

# NOTE: This job does not request a GPU. If your cluster has a CPU-only
# partition, update --partition accordingly.

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
if [[ ! -f "${WORK_DIR}/analysis/analyze_trajectories.py" ]]; then
    WORK_DIR="${SCRIPT_DIR}"
fi
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
log "Step 5/8: Changing directory to ${WORK_DIR}"
cd "${WORK_DIR}"
log "Step 6/8: Ensuring logs directory exists"
mkdir -p logs

INPUT_ROOT="${INPUT_ROOT:-${WORK_DIR}/data_generation/outputs_clean}"

log "Step 7/8: Starting trajectory analysis"
log "Input root: ${INPUT_ROOT}"
ANALYSIS_START_TS=$(date +%s)
python -u "${WORK_DIR}/analysis/analyze_trajectories.py" \
    --input-root "${INPUT_ROOT}" \
    --output-dir "${WORK_DIR}/analysis/results" \
    --max-lag-frames 0 \
    --msd-max-lag-frames 0
ANALYSIS_END_TS=$(date +%s)
log "Step 7/8 complete: Analysis finished in $((ANALYSIS_END_TS - ANALYSIS_START_TS)) seconds"

JOB_END_TS=$(date +%s)
log "Step 8/8: Job complete in $((JOB_END_TS - JOB_START_TS)) seconds total"
