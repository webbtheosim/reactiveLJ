#!/bin/bash
#SBATCH --job-name=clump_p_analysis
#SBATCH --time=01:02:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --output=logs/run_analysis_%j.out

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
if [[ ! -f "${WORK_DIR}/make_plot.py" ]]; then
    WORK_DIR="${SCRIPT_DIR}"
fi
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
cd "${WORK_DIR}"
log "Step 5/8: Ensuring logs directory exists"
mkdir -p logs

log "Step 6/8: Starting clump-fraction analysis"
ANALYSIS_START_TS=$(date +%s)
python -u "${WORK_DIR}/make_plot.py" \
    --input-root "${WORK_DIR}/outputs" \
    --clump-output "${WORK_DIR}/results/clump_fraction_violin.svg" \
    --paired-output "${WORK_DIR}/results/paired_fraction_violin.svg" \
    --excess-coordination-output "${WORK_DIR}/results/excess_coordination_violin.svg" \
    --n-jobs "${SLURM_CPUS_PER_TASK}"

ANALYSIS_END_TS=$(date +%s)
log "Step 7/8 complete: Analysis finished in $((ANALYSIS_END_TS - ANALYSIS_START_TS)) seconds"

JOB_END_TS=$(date +%s)
log "Step 8/8: Job complete in $((JOB_END_TS - JOB_START_TS)) seconds total"
