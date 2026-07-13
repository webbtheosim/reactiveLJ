#!/bin/bash
#SBATCH --job-name=brachiationP
#SBATCH --time=01:01:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --output=logs/run_brachiation_plot_%j.out

# NOTE: This job does not request a GPU. If your cluster has a CPU-only
# partition, update --partition accordingly.

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

JOB_START_TS=$(date +%s)

log "Step 1/9: Purging modules"
module purge
log "Step 2/9: Sourcing ~/.bashrc"
source ~/.bashrc
log "Step 3/9: Loading anaconda module"
module load anaconda3/2025.12
log "Step 4/9: Activating conda environment"
conda activate '/home/mc7345/anaconda3/envs/reactiveLJ_paper'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
if [[ ! -f "${WORK_DIR}/analysis/make_brachiation_plot" ]]; then
    WORK_DIR="${SCRIPT_DIR}"
fi
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
REPO_ROOT="$(cd "${WORK_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Parallelism is across replicate-level open-sticker correlation jobs. Keep
# threaded numerical libraries from oversubscribing the Slurm allocation.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-brachiation_plot}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/fontconfig-${SLURM_JOB_ID:-brachiation_plot}"

log "Step 5/9: Changing directory to ${WORK_DIR}"
cd "${WORK_DIR}"
log "Step 6/9: Ensuring logs directory exists"
mkdir -p logs
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

OUTPUT_SVG="${WORK_DIR}/analysis/results/brachiation_tau_b_by_p.svg"
OUTPUT_CSV="${WORK_DIR}/analysis/results/brachiation_tau_b_by_p.csv"

log "Step 7/9: Starting brachiation p-sweep analysis"
ANALYSIS_START_TS=$(date +%s)
python -u "${WORK_DIR}/analysis/make_brachiation_plot" \
    --input-root "${WORK_DIR}/data_generation/outputs_clean" \
    --output "${OUTPUT_SVG}" \
    --csv-output "${OUTPUT_CSV}" \
    --n-jobs "${SLURM_CPUS_PER_TASK:-1}" \
    --analysis-stride 1 \
    --conditions eps=18,p=2 eps=18,p=4 eps=18,p=8

ANALYSIS_END_TS=$(date +%s)
log "Step 8/9: Brachiation p-sweep analysis finished in $((ANALYSIS_END_TS - ANALYSIS_START_TS)) seconds"

JOB_END_TS=$(date +%s)
log "Step 9/9: Job complete in $((JOB_END_TS - JOB_START_TS)) seconds total"
