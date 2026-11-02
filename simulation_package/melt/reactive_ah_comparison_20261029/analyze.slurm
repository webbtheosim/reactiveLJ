#!/bin/bash
#SBATCH --job-name=AH_LJ_analysis
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
set -euo pipefail
: "${AH_COMPARISON_ROOT:?Set the add-on directory}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd "$AH_COMPARISON_ROOT"
"${PYTHON:-python}" -u code/analyze_comparison.py --workers "${SLURM_CPUS_PER_TASK}"
