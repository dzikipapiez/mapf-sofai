#!/usr/bin/env python3
"""Compare sequential k=10 PIBT candidates with k=1 LaCAM."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import random
from pathlib import Path

from mapf_anytime.datasets.generation import read_instances
from mapf_anytime.problem import MapfProblem
from mapf_anytime.solvers import LaCAMConfig, run_lacam


def solve_pair(arguments):
    index, manifest, timeout, seed = arguments
    problem = MapfProblem.from_manifest(manifest)
    variants = (("k10_sequential", 10), ("k1", 1))
    rows = []
    for name, k in (variants if index % 2 else variants[::-1]):
        result = run_lacam(
            problem,
            LaCAMConfig(
                timeout,
                anytime=False,
                seed=seed,
                pibt_num=k,
                parallel_pibt=False,
            ),
        )
        rows.append({
            "instance": str(manifest),
            "variant": name,
            "status": result.status,
            "wall_seconds": result.wall_seconds,
            "soc": result.solution.soc() if result.solution else None,
            "error": result.error,
        })
    return problem.name, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notebooks/benchmarks/lacam_k1_vs_k10.csv"),
    )
    args = parser.parse_args()

    manifests = read_instances(args.instances)
    if args.count > len(manifests):
        parser.error(f"requested {args.count} instances from a list of {len(manifests)}")
    manifests = random.Random(args.sample_seed).sample(manifests, args.count)

    variants = (("k10_sequential", 10), ("k1", 1))
    jobs = ((index, manifest, args.timeout, args.seed) for index, manifest in enumerate(manifests, 1))
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for completed, (name, pair) in enumerate(executor.map(solve_pair, jobs), 1):
            rows.extend(pair)
            print(f"{completed}/{len(manifests)} {name}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    for name, _ in variants:
        selected = [row for row in rows if row["variant"] == name]
        solved = [row for row in selected if row["status"] == "success"]
        mean_runtime = sum(row["wall_seconds"] for row in selected) / len(selected)
        mean_soc = sum(row["soc"] for row in solved) / len(solved) if solved else float("nan")
        print(f"{name}: solved={len(solved)}/{len(selected)}, mean_time={mean_runtime:.3f}s, mean_soc={mean_soc:.1f}")
    paired = {}
    for row in rows:
        paired.setdefault(row["instance"], {})[row["variant"]] = row
    both = [pair for pair in paired.values() if all(pair[name]["soc"] is not None for name, _ in variants)]
    print(
        f"both_solved={len(both)}, "
        f"same_soc={sum(pair['k10_sequential']['soc'] == pair['k1']['soc'] for pair in both)}, "
        f"k10_better={sum(pair['k10_sequential']['soc'] < pair['k1']['soc'] for pair in both)}, "
        f"k1_better={sum(pair['k1']['soc'] < pair['k10_sequential']['soc'] for pair in both)}"
    )
    print(f"raw={args.output}")


if __name__ == "__main__":
    main()
