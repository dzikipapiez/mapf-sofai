#!/usr/bin/env bash
#SBATCH --job-name=mapf-dataset
#SBATCH --clusters=htc
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --constraint=cpu_gen:Cascade_Lake
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

module purge
module --ignore_cache load Python/3.11.3-GCCcore-12.3.0
module --ignore_cache load Boost/1.82.0-GCC-12.3.0
module --ignore_cache load Eigen/3.4.0-GCCcore-12.3.0

repo=${SLURM_SUBMIT_DIR:?Submit this job from the repository root}
stage=$1
run_name=$2
shift 2

cd "$repo"
source "$repo/.venv/bin/activate"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python scripts/generate_dataset.py "$stage" --run-name "$run_name" "$@"
