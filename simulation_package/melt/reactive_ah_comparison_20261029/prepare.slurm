#!/bin/bash
#SBATCH --job-name=AH_shared_melts
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-11%12
set -euo pipefail
: "${AH_COMPARISON_ROOT:?Set the add-on directory}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd "$AH_COMPARISON_ROOT"
"${PYTHON:-python}" -u code/prepare_initial.py
