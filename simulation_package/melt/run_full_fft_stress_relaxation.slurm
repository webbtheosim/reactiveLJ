#!/bin/bash
#SBATCH --job-name=fullFFTStress
#SBATCH --time=01:01:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --array=0-1
#SBATCH --output=logs/run_full_fft_stress_relaxation_%A_%a.out

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

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
case "${TASK_ID}" in
    0)
        SWEEP_NAME="p=4 epsilon sweep"
        OUTPUT_DIR="${WORK_DIR}/analysis/results/full_fft_stress_relaxation_p4_epsilon_sweep"
        PLOT_NAME="stress_modulus_vs_time_lag_by_epsilon.svg"
        SELECTION_ARGS=(--weakening-exponents 4)
        ;;
    1)
        SWEEP_NAME="eps=18 weakening-exponent sweep"
        OUTPUT_DIR="${WORK_DIR}/analysis/results/full_fft_stress_relaxation_eps18_p_sweep"
        PLOT_NAME="stress_modulus_vs_time_lag_by_weakening_exponent.svg"
        SELECTION_ARGS=(--conditions eps=18,p=2 eps=18,p=4 eps=18,p=8)
        ;;
    *)
        echo "Unsupported SLURM_ARRAY_TASK_ID=${TASK_ID}" >&2
        exit 2
        ;;
esac

# Avoid concurrent array tasks writing the same Matplotlib/font caches.
export MPLCONFIGDIR="/tmp/reactive_lj_mpl_${SLURM_ARRAY_JOB_ID:-manual}_${TASK_ID}"
export XDG_CACHE_HOME="/tmp/reactive_lj_xdg_${SLURM_ARRAY_JOB_ID:-manual}_${TASK_ID}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

log "Step 7/9: Starting full-FFT stress relaxation analysis for ${SWEEP_NAME}"
ANALYSIS_START_TS=$(date +%s)
python -u "${WORK_DIR}/analysis/full_fft_stress_relaxation.py" \
    --input-root "${WORK_DIR}/data_generation/outputs_clean" \
    --extension-input-root "${WORK_DIR}/data_generation/outputs_virial_extended" \
    --output-dir "${OUTPUT_DIR}" \
    --n-jobs "${SLURM_CPUS_PER_TASK:-1}" \
    --stress-max-runtime-fraction 0.71 \
    --extension-blend-start-time 60615 \
    --extension-blend-end-time 141435 \
    --min-plot-g 1e-3 \
    --replicate-sem-points 20 \
    --plot-name "${PLOT_NAME}" \
    "${SELECTION_ARGS[@]}"

ANALYSIS_END_TS=$(date +%s)
log "Step 8/9: Analysis finished in $((ANALYSIS_END_TS - ANALYSIS_START_TS)) seconds"

JOB_END_TS=$(date +%s)
log "Step 9/9: Job complete in $((JOB_END_TS - JOB_START_TS)) seconds total"
