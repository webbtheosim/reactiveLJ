#!/bin/bash
#SBATCH --job-name=AH_force_check
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
set -euo pipefail
: "${AH_COMPARISON_ROOT:?Set the add-on directory}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd "$AH_COMPARISON_ROOT"
"${PYTHON:-python}" -u code/validate_force.py --gpu
