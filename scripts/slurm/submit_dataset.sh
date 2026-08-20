#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "usage: $0 INSTANCE_LIST RUN_NAME [WORKERS] [generator options...]" >&2
  exit 2
fi

instances=$1
run_name=$2
workers=500
partition=${DATASET_PARTITION:-medium}
time_limit=${DATASET_TIME_LIMIT:-$([[ $partition == short ]] && echo 12:00:00 || echo 2-00:00:00)}
lacam_cpus=${LACAM_CPUS_PER_TASK:-10}
shift 2
if [[ ${1:-} =~ ^[0-9]+$ ]]; then
  workers=$1
  shift
fi
if (( workers < 1 )); then
  echo "WORKERS must be positive" >&2
  exit 2
fi

here=$(cd "$(dirname "$0")" && pwd)
arguments=(--instances "$instances" "$@")

submit_array() {
  local dependency=$1 memory=$2 cpus=$3
  shift 3
  sbatch --parsable --dependency="$dependency" \
    --partition="$partition" --time="$time_limit" --mem="$memory" \
    --cpus-per-task="$cpus" \
    --array="0-$((workers - 1))" "$here/dataset.slurm" "$@" \
    "$run_name" "${arguments[@]}"
}

job_id() {
  printf '%s' "${1%%;*}"
}

lacam=$(job_id "$(sbatch --parsable \
  --partition="$partition" --time="$time_limit" --mem=8G \
  --cpus-per-task="$lacam_cpus" \
  --array="0-$((workers - 1))" "$here/dataset.slurm" \
  lacam-worker "$run_name" "${arguments[@]}")")
lacam_retry=$(job_id "$(submit_array "afterany:$lacam" 20G "$lacam_cpus" lacam-worker)")
lacam_large=$(job_id "$(submit_array "afterany:$lacam_retry" 48G "$lacam_cpus" lacam-worker)")
lacam_final=$(job_id "$(submit_array "afterany:$lacam_large" 96G "$lacam_cpus" lacam-worker)")

lns=$(job_id "$(submit_array "afterok:$lacam_final" 8G 1 lns-worker)")
lns_retry=$(job_id "$(submit_array "afterany:$lns" 20G 1 lns-worker)")
lns_large=$(job_id "$(submit_array "afterany:$lns_retry" 48G 1 lns-worker)")
lns_final=$(job_id "$(submit_array "afterany:$lns_large" 96G 1 lns-worker)")

aggregate=$(job_id "$(sbatch --parsable --dependency="afterok:$lns_final" \
  --partition=short --time=04:00:00 --mem=16G \
  "$here/dataset.slurm" aggregate "$run_name" "${arguments[@]}")")

echo "lacam=$lacam retry=$lacam_retry large=$lacam_large final=$lacam_final"
echo "lns=$lns retry=$lns_retry large=$lns_large final=$lns_final"
echo "aggregate=$aggregate"
