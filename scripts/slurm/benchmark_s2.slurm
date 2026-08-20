#!/usr/bin/env bash
#SBATCH --job-name=s2-benchmark
#SBATCH --clusters=arc
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --constraint=cpu_gen:Cascade_Lake

set -euo pipefail

module purge
module --ignore_cache load Python/3.11.3-GCCcore-12.3.0
module --ignore_cache load Boost/1.82.0-GCC-12.3.0
module --ignore_cache load Eigen/3.4.0-GCCcore-12.3.0

repo=${SLURM_SUBMIT_DIR:?Submit from the repository root}
cd "$repo"
source "$repo/.venv/bin/activate"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python scripts/benchmark_s2.py "$@"
