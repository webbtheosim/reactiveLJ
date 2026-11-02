#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python}"
PREFIX="$($PYTHON -c 'import sys; print(sys.prefix)')"
cmake -S "$ROOT/src" -B "$ROOT/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PREFIX" \
    -DPython_EXECUTABLE="$(command -v "$PYTHON")" "$@"
cmake --build "$ROOT/build" --parallel "${SLURM_CPUS_PER_TASK:-2}"
