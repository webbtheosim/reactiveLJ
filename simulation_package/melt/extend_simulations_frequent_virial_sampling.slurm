#!/bin/bash
#SBATCH --job-name=reactiveLJ_virial_ext
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --array=0-58%30
#SBATCH --constraint=gpu80
#SBATCH --open-mode=append
#SBATCH --output=logs/extend_virial_%A_%a.out

# One continuous 50-tau_R^0 extension per available clean trajectory.
# This includes 49 p=4 trajectories plus five each for eps=18 at p=2 and p=8.

module purge
source ~/.bashrc
module load anaconda3/2025.12
conda activate '/home/mc7345/anaconda3/envs/reactiveLJ_paper'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
if [[ ! -f "${WORK_DIR}/data_generation/extend_virial_trajectory.py" ]]; then
    WORK_DIR="${SCRIPT_DIR}"
fi
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
cd "${WORK_DIR}"
mkdir -p logs

INPUT_ROOT=${INPUT_ROOT:-${WORK_DIR}/data_generation/outputs_clean}
OUTPUT_ROOT=${OUTPUT_ROOT:-${WORK_DIR}/data_generation/outputs_virial_extended}
EXTENSION_TAU_R0=${EXTENSION_TAU_R0:-50}
VIRIAL_LOG_STEPS=${VIRIAL_LOG_STEPS:-100}
export REACTIVELJ_SUPPORT_SITE_PACKAGES=${REACTIVELJ_SUPPORT_SITE_PACKAGES:-/home/mc7345/anaconda3/envs/hoomd-latest/lib/python3.14/site-packages}

python - <<'PY'
import os
import sys
from pathlib import Path

import hoomd

support_site = Path(
    os.environ["REACTIVELJ_SUPPORT_SITE_PACKAGES"]
)
if support_site.is_dir() and str(support_site) not in sys.path:
    sys.path.append(str(support_site))

import cupy
import numba

if not hasattr(hoomd.md.many_body, "ReactiveLJ"):
    raise RuntimeError("The active HOOMD build does not provide ReactiveLJ.")
print(
    f"Environment check: HOOMD={hoomd.version.version} "
    f"CuPy={cupy.__version__} Numba={numba.__version__}",
    flush=True,
)
PY

python "${WORK_DIR}/data_generation/extend_virial_trajectory.py" \
    --input-root "${INPUT_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --task-index "${SLURM_ARRAY_TASK_ID}" \
    --restart-incomplete \
    --extension-runtime-tau-r0 "${EXTENSION_TAU_R0}" \
    --virial-log-steps "${VIRIAL_LOG_STEPS}"
