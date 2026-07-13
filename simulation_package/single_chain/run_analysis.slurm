#!/bin/bash
#SBATCH --job-name=reactiveLJ_single_chain_analysis
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
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
conda activate '/home/mc7345/anaconda3/envs/reactiveLJ_paper'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
if [[ ! -f "${WORK_DIR}/analysis/analyze_trajectories.py" ]]; then
    ALT_WORK_DIR="${SLURM_SUBMIT_DIR:-}/simulation_package/single_chain"
    if [[ -f "${ALT_WORK_DIR}/analysis/analyze_trajectories.py" ]]; then
        WORK_DIR="${ALT_WORK_DIR}"
    else
        WORK_DIR="${SCRIPT_DIR}"
    fi
fi
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
if [[ ! -f "${WORK_DIR}/analysis/analyze_trajectories.py" ]]; then
    echo "ERROR: Could not resolve single_chain work directory."
    echo "Submit from simulation_package/single_chain or repo root."
    exit 1
fi
log "Step 5/8: Changing directory to ${WORK_DIR}"
cd "${WORK_DIR}"
log "Step 6/8: Ensuring logs directory exists"
mkdir -p logs

MAX_LAG_FRAMES=${MAX_LAG_FRAMES:-500}

log "Step 7/8: Starting single-chain trajectory analysis"
ANALYSIS_START_TS=$(date +%s)
python -u "${WORK_DIR}/analysis/analyze_trajectories.py" \
    --input-root "${WORK_DIR}/data_generation/outputs" \
    --output-dir "${WORK_DIR}/analysis/results" \
    --max-lag-frames "${MAX_LAG_FRAMES}"
ANALYSIS_END_TS=$(date +%s)
log "Step 7/8 complete: Analysis finished in $((ANALYSIS_END_TS - ANALYSIS_START_TS)) seconds"

JOB_END_TS=$(date +%s)
log "Step 8/8: Job complete in $((JOB_END_TS - JOB_START_TS)) seconds total"
