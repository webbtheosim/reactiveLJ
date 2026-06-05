#!/bin/bash
#SBATCH --job-name=clump_p_gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:02:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --array=0-9
#SBATCH --open-mode=append
#SBATCH --output=logs/generate_data_%A_%a.out

module purge
source ~/.bashrc
module load anaconda3/2025.12
conda activate '/home/mc7345/micromamba/envs/reactiveLJ_paper'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
if [[ ! -f "${WORK_DIR}/generate_clump_test_data.py" ]]; then
    WORK_DIR="${SCRIPT_DIR}"
fi
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
cd "${WORK_DIR}"
mkdir -p logs

python -u "${WORK_DIR}/generate_clump_test_data.py" \
    --p "${SLURM_ARRAY_TASK_ID}" \
    --replicate 1 \
    --device gpu \
    --output-root "${WORK_DIR}/outputs"
