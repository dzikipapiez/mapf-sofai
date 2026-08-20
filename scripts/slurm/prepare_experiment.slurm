#!/usr/bin/env bash
#SBATCH --job-name=prepare-metacognitive
#SBATCH --clusters=arc
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --constraint=cpu_gen:Cascade_Lake
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 CONFIG OUTPUT_DIRECTORY" >&2
    exit 2
fi

module purge
module --ignore_cache load Python/3.11.3-GCCcore-12.3.0
module --ignore_cache load Boost/1.82.0-GCC-12.3.0
module --ignore_cache load Eigen/3.4.0-GCCcore-12.3.0

repo=${SLURM_SUBMIT_DIR:?Submit this job from the repository root}
cd "$repo"
source "$repo/.venv/bin/activate"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python scripts/run_experiment.py prepare "$1" "$2"
