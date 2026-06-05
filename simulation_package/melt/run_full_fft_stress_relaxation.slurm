#!/bin/bash
#SBATCH --job-name=fullFFTStress
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=2G
#SBATCH --nodes=1
#SBATCH --output=logs/run_full_fft_stress_relaxation_%j.out

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
conda activate '/home/mc7345/micromamba/envs/reactiveLJ_paper'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
if [[ ! -f "${WORK_DIR}/analysis/full_fft_stress_relaxation.py" ]]; then
    WORK_DIR="${SCRIPT_DIR}"
fi
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
REPO_ROOT="$(cd "${WORK_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Parallelism is across replicate-level FFT jobs. Keep threaded numerical
# libraries from oversubscribing the Slurm allocation.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

log "Step 5/9: Changing directory to ${WORK_DIR}"
cd "${WORK_DIR}"
log "Step 6/9: Ensuring logs directory exists"
mkdir -p logs

OUTPUT_DIR="${WORK_DIR}/analysis/results/full_fft_stress_relaxation_eps18_p_sweep"
log "Step 7/9: Starting full-FFT stress relaxation analysis"
ANALYSIS_START_TS=$(date +%s)
python -u "${WORK_DIR}/analysis/full_fft_stress_relaxation.py" \
    --input-root "${WORK_DIR}/data_generation/outputs_clean" \
    --output-dir "${OUTPUT_DIR}" \
    --n-jobs "${SLURM_CPUS_PER_TASK:-1}" \
    --conditions 18:2 18:4 18:8

ANALYSIS_END_TS=$(date +%s)
log "Step 8/9: Analysis finished in $((ANALYSIS_END_TS - ANALYSIS_START_TS)) seconds"

JOB_END_TS=$(date +%s)
log "Step 9/9: Job complete in $((JOB_END_TS - JOB_START_TS)) seconds total"
