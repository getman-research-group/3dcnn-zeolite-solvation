#!/usr/bin/env bash

# Portable launcher for test_pytorch.py.
#
# Local usage:
#   bash python_scripts/test_pytorch.sh
#   bash python_scripts/test_pytorch.sh --device cpu --size 2048
#
# Example SLURM usage on the Ohio Supercomputer Center:
#   sbatch --account=PAS2536 --cluster=ascend \
#       python_scripts/test_pytorch.sh --device cuda --size 4096
#
# The SLURM account, cluster, partition, and Conda environment are intentionally
# not hard-coded so that this script can be used outside the authors' system.
# To activate a Conda environment before running the test, export CONDA_ENV:
#   CONDA_ENV=torch bash python_scripts/test_pytorch.sh
#   sbatch --export=ALL,CONDA_ENV=torch python_scripts/test_pytorch.sh

# --- Optional SLURM resource request ---
# These lines are ignored when the script is run directly with Bash.
#SBATCH --job-name=pytorch_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}

echo "========================================================================"
echo "PyTorch environment test launcher"
echo "========================================================================"
echo "Date:              $(date)"
echo "Hostname:          $(hostname)"
echo "SLURM job ID:      ${SLURM_JOB_ID:-N/A}"
echo "Script directory:  ${SCRIPT_DIR}"

if [[ -n "${CONDA_ENV:-}" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "Error: CONDA_ENV is set, but the conda command is unavailable." >&2
        exit 1
    fi

    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
    echo "Conda environment: ${CONDA_DEFAULT_ENV}"
else
    echo "Conda environment: ${CONDA_DEFAULT_ENV:-current environment}"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Error: Python executable '${PYTHON_BIN}' was not found." >&2
    exit 1
fi

echo "Python executable: $(command -v "${PYTHON_BIN}")"
echo "========================================================================"

"${PYTHON_BIN}" "${SCRIPT_DIR}/test_pytorch.py" "$@"
