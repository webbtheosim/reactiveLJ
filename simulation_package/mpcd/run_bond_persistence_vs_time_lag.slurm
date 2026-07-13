#!/bin/bash
#SBATCH --job-name=mpcdBondLag
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --output=logs/run_bond_persistence_vs_time_lag_%j.out

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
if [[ ! -f "${WORK_DIR}/analysis/make_bond_persistence_vs_time_lag.py" ]]; then
    WORK_DIR="${SCRIPT_DIR}"
fi
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
log "Step 5/8: Changing directory to ${WORK_DIR}"
cd "${WORK_DIR}"
log "Step 6/8: Ensuring logs and results directories exist"
mkdir -p logs analysis/results

log "Step 7/8: Generating MPCD bond autocorrelation lag plot"
PLOT_START_TS=$(date +%s)
python -u "${WORK_DIR}/analysis/make_bond_persistence_vs_time_lag.py" \
    --input-root "${WORK_DIR}/analysis/results" \
    --output "${WORK_DIR}/analysis/results/bond_persistence_vs_time_lag.svg"
PLOT_END_TS=$(date +%s)
log "Step 7/8 complete: Plot finished in $((PLOT_END_TS - PLOT_START_TS)) seconds"

JOB_END_TS=$(date +%s)
log "Step 8/8: Job complete in $((JOB_END_TS - JOB_START_TS)) seconds total"
