#!/bin/bash
#SBATCH --job-name=reactiveLJ_single_chain_cpu
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --nodes=1
#SBATCH --array=0-1%1
#SBATCH --output=logs/generate_single_chain_cpu_reactivelj_%A_%a.out

module purge
source ~/.bashrc

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}"
LOCAL_CUDA_LIBDIR="${WORK_DIR}/third_party/cuda/lib"
REQUIRED_CUDA_LIBS=(
    libcudart.so.12
    libcufft.so.11
    libcublas.so.12
    libcublasLt.so.12
    libcusolver.so.11
    libcusparse.so.12
    libnvJitLink.so.12
)

# Prefer a vendored CUDA runtime copy in the project to avoid module/path
# differences between login and batch nodes.
CUDA_LIBDIR=""
LOCAL_CUDA_COMPLETE=1
for lib in "${REQUIRED_CUDA_LIBS[@]}"; do
    if [[ ! -e "${LOCAL_CUDA_LIBDIR}/${lib}" ]]; then
        LOCAL_CUDA_COMPLETE=0
        break
    fi
done

if [[ "${LOCAL_CUDA_COMPLETE}" -eq 1 ]]; then
    CUDA_LIBDIR="${LOCAL_CUDA_LIBDIR}"
    echo "Using vendored CUDA runtime libdir: ${CUDA_LIBDIR}"
else
    # Load CUDA runtime first. On this cluster, loading anaconda can hide the
    # cudatoolkit modulefiles from `module load`.
    CUDA_MOD=""
    for CANDIDATE in cudatoolkit/12.9 cudatoolkit/12.8 cudatoolkit/12.6; do
        if module avail "${CANDIDATE}" 2>&1 | grep -q "${CANDIDATE}"; then
            module load "${CANDIDATE}"
            CUDA_MOD="${CANDIDATE}"
            break
        fi
    done

    # Fallback: direct CUDA runtime library paths.
    for CUDA_DIR in \
        /usr/local/cuda-12.9/targets/x86_64-linux/lib \
        /usr/local/cuda-12.8/targets/x86_64-linux/lib \
        /usr/local/cuda-12.6/targets/x86_64-linux/lib \
        /usr/local/cuda/lib64
    do
        if [[ -e "${CUDA_DIR}/libcudart.so.12" ]]; then
            CUDA_LIBDIR="${CUDA_DIR}"
            break
        fi
    done

    if [[ -n "${CUDA_MOD}" ]]; then
        echo "Loaded CUDA module: ${CUDA_MOD}"
    fi
    if [[ -n "${CUDA_LIBDIR}" ]]; then
        echo "Using CUDA runtime libdir: ${CUDA_LIBDIR}"
    fi
    if [[ -z "${CUDA_MOD}" && -z "${CUDA_LIBDIR}" ]]; then
        echo "ERROR: Could not locate CUDA runtime (need libcudart.so.12)."
        module avail cudatoolkit 2>&1 || true
        exit 1
    fi
fi

module load anaconda3/2025.12
conda activate '/home/mc7345/micromamba/envs/reactiveLJ_paper'
# Keep conda libs visible to the dynamic loader.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CUDA_LIBDIR}:${LD_LIBRARY_PATH:-}"

set -euo pipefail

cd "${WORK_DIR}"
mkdir -p logs

EPSILONS=(10)
WEAKENING_EXPONENTS=(0 4)

TASK_ID=${SLURM_ARRAY_TASK_ID}
if [[ ${TASK_ID} -ge ${#WEAKENING_EXPONENTS[@]} ]]; then
    echo "Task ${TASK_ID} exceeds weakening exponent list length; exiting."
    exit 0
fi

EPS=${EPSILONS[0]}
WEXP=${WEAKENING_EXPONENTS[$TASK_ID]}
FRAME_STEPS=${FRAME_STEPS:-10000}
RUN_STEPS=${RUN_STEPS:-1000000}
OUTPUT_ROOT="${WORK_DIR}/data_generation/outputs/TEST_CPU_REACTIVELJ/wexp_${WEXP}"

python "${WORK_DIR}/data_generation/run_reactivelj_singlechain_test_cpu.py" \
    --epsilon "${EPS}" \
    --weakening-exponent "${WEXP}" \
    --replicate 1 \
    --output-root "${OUTPUT_ROOT}" \
    --frame-steps "${FRAME_STEPS}" \
    --run-steps "${RUN_STEPS}"
