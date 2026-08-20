#!/usr/bin/env python3
"""Build a LaCAM-initialized, repeated-LNS dataset from an instance list."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from mapf_anytime.datasets.generation import (
    DatasetConfig,
    aggregate,
    extend_neighborhoods,
    initialize,
    read_instances,
    run_lacam_stage,
    run_lns_stage,
    status,
)


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "run", "status", "aggregate", "extend-lns",
            "lacam-worker", "lns-worker",
        ),
    )
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--lacam-timeout", type=float, default=40)
    parser.add_argument("--lacam-seed", type=int, default=0)
    parser.add_argument("--lacam-pibt-num", type=int, default=1)
    parser.add_argument("--lns-timeout", type=float, default=100)
    parser.add_argument("--neighborhood-sizes", default="2,4,8,16,32")
    parser.add_argument("--lns-repetitions", type=int, default=4)
    parser.add_argument("--replan-time-limit", type=float, default=10)
    parser.add_argument("--probe-iterations", type=int, default=5)
    parser.add_argument(
        "--cells-mode", choices=("single-path", "exact"), default="single-path"
    )
    parser.add_argument("--feature-threads", type=int, default=1)
    parser.add_argument("--lns-max-iterations", type=int, default=1_000_000)
    parser.add_argument("--aggregate-workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    instance_list = args.instances.expanduser().resolve()
    instances = read_instances(instance_list, args.limit)
    run_dir = (args.run_dir or ROOT / "datasets" / "runs" / args.run_name).resolve()
    config = DatasetConfig(
        run_dir=run_dir,
        lacam_timeout=args.lacam_timeout,
        lacam_seed=args.lacam_seed,
        lacam_pibt_num=args.lacam_pibt_num,
        lns_timeout=args.lns_timeout,
        neighborhood_sizes=_sizes(args.neighborhood_sizes),
        lns_repetitions=args.lns_repetitions,
        replan_time_limit=args.replan_time_limit,
        probe_iterations=args.probe_iterations,
        cells_mode=args.cells_mode,
        feature_threads=args.feature_threads,
        lns_max_iterations=args.lns_max_iterations,
    )
    if args.command == "extend-lns":
        extend_neighborhoods(config, instance_list, instances)
        return 0
    initialize(config, instance_list, instances)

    if args.command == "run":
        code = run_lacam_stage(config, instances)
        if code == 0:
            code = run_lns_stage(config, instances)
        if code == 0:
            _print_outputs(aggregate(config, instances, args.aggregate_workers))
        return code
    worker = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    workers = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    if args.command == "lacam-worker":
        return run_lacam_stage(config, instances, worker, workers)
    if args.command == "lns-worker":
        return run_lns_stage(config, instances, worker, workers)
    if args.command == "aggregate":
        _print_outputs(aggregate(config, instances, args.aggregate_workers))
        return 0

    for name, (done, total) in status(config, instances).items():
        print(f"{name}: {done:,}/{total:,}")
    return 0


def _sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item) for item in value.split(",") if item.strip())
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("Neighbourhood sizes must be positive")
    return sizes


def _print_outputs(paths: tuple[Path, Path]) -> None:
    print(f"raw={paths[0]}\nmodel={paths[1]}")


if __name__ == "__main__":
    raise SystemExit(main())
