#!/usr/bin/env bash
#SBATCH --job-name=mapf-policy
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --constraint=cpu_gen:Cascade_Lake
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

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

: "${TMPDIR:?ARC did not provide its per-job node-local TMPDIR}"
python scripts/run_experiment.py run "$1" \
  --task "$SLURM_ARRAY_TASK_ID" \
  --local-input-dir "$TMPDIR/mapf-anytime-inputs"
