#!/bin/bash
#SBATCH --job-name=RLJ_smoothing_report
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --nodes=1

# Activate the environment containing the pinned HOOMD build before submission.
set -euo pipefail

: "${SWEEP_MANIFEST:?Set SWEEP_MANIFEST to the campaign manifest}"
SWEEP_CODE_DIR="$(dirname "${SWEEP_MANIFEST}")/code"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
"${PYTHON:-python}" -u "${SWEEP_CODE_DIR}/analysis/analyze_smoothing_sweep.py" \
    --manifest "${SWEEP_MANIFEST}" --combine
