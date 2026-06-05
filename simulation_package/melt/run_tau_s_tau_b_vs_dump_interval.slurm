#!/bin/bash
#SBATCH --job-name=tauDumpSweep
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --output=logs/run_tau_s_tau_b_vs_dump_interval_%j.out

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
if [[ ! -f "${WORK_DIR}/analysis/make_tau_s_tau_b_vs_dump_interval.py" ]]; then
    WORK_DIR="${SCRIPT_DIR}"
fi
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
log "Step 5/8: Changing directory to ${WORK_DIR}"
cd "${WORK_DIR}"
log "Step 6/8: Ensuring logs directory exists"
mkdir -p logs

log "Step 7/8: Measuring tau_s and tau_b versus dump interval"
PLOT_START_TS=$(date +%s)
python -u "${WORK_DIR}/analysis/make_tau_s_tau_b_vs_dump_interval.py" \
    --input-root "${WORK_DIR}/data_generation/outputs_clean" \
    --output "${WORK_DIR}/analysis/tau_s_tau_b_vs_dump_interval.svg" \
    --csv-output "${WORK_DIR}/analysis/tau_s_tau_b_vs_dump_interval.csv" \
    --epsilons 6 12 15 18 \
    --n-jobs "${SLURM_CPUS_PER_TASK}"
PLOT_END_TS=$(date +%s)
log "Step 7/8 complete: tau sweep finished in $((PLOT_END_TS - PLOT_START_TS)) seconds"

JOB_END_TS=$(date +%s)
log "Step 8/8: Job complete in $((JOB_END_TS - JOB_START_TS)) seconds total"
