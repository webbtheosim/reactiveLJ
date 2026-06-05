#!/bin/bash
#SBATCH --job-name=tersoff_analysis
#SBATCH --time=01:02:00
#SBATCH --cpus-per-task=16
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
conda activate '/scratch/gpfs/WEBB/chertok/conda_envs/reactiveLJ_oconnor_tersoff_comparison'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERSOFF_DIR="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
if [[ ! -f "${TERSOFF_DIR}/analyze_trajectories.py" ]]; then
    TERSOFF_DIR="${SCRIPT_DIR}"
fi
TERSOFF_DIR="$(cd "${TERSOFF_DIR}" && pwd)"
WORK_DIR="$(cd "${TERSOFF_DIR}/.." && pwd)"

log "Step 5/8: Changing directory to ${WORK_DIR}"
cd "${WORK_DIR}"
log "Step 6/8: Ensuring Tersoff analysis output/log directories exist"
mkdir -p "${TERSOFF_DIR}/logs" "${TERSOFF_DIR}/analysis/results"

log "Step 7/8: Starting Tersoff trajectory analysis"
ANALYSIS_START_TS=$(date +%s)
python -u "${TERSOFF_DIR}/analyze_trajectories.py" \
    --input-root "${TERSOFF_DIR}/outputs" \
    --output-dir "${TERSOFF_DIR}/analysis/results"
ANALYSIS_END_TS=$(date +%s)
log "Step 7/8 complete: Analysis finished in $((ANALYSIS_END_TS - ANALYSIS_START_TS)) seconds"

JOB_END_TS=$(date +%s)
log "Step 8/8: Job complete in $((JOB_END_TS - JOB_START_TS)) seconds total"
