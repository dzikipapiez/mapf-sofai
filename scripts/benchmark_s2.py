#!/usr/bin/env python3
"""Run the anytime-LaCAM versus MAPF-LNS benchmark."""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys

import pandas as pd

from mapf_anytime.features import analyze
from mapf_anytime.problem import MapfProblem
from mapf_anytime.solution import MapfSolution
from mapf_anytime.solvers import LaCAMConfig, LNSConfig, run_lacam, run_lns


ROOT = Path(__file__).resolve().parents[1]
LACAM_PIBT_NUM = 1
LACAM_PARALLEL = False
LACAM_WATCHDOG_MULTIPLIER = 3.0


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("run_dir", type=Path)
    prepare.add_argument(
        "--dataset", type=Path, default=ROOT / "datasets/movingai_final.csv"
    )
    prepare.add_argument("--instances", type=Path, default=ROOT / "datasets/movingai/instances")
    prepare.add_argument("--time-limit", type=float, default=100)
    prepare.add_argument("--replan-time-limit", type=float, default=10)
    neighborhoods = prepare.add_mutually_exclusive_group()
    neighborhoods.add_argument("--neighborhood-size", type=int)
    neighborhoods.add_argument("--neighborhood-sizes", type=int, nargs="+")
    prepare.add_argument("--sample-size", type=int, default=500)
    prepare.add_argument("--sample-seed", type=int, default=42)
    prepare.add_argument("--lacam-seed", type=int, default=0)
    prepare.add_argument("--lns-seed", type=int, default=1)

    submit = commands.add_parser("submit")
    submit.add_argument("run_dir", type=Path)
    submit.add_argument("--workers", type=int, default=500)
    submit.add_argument("--retry-failed", action="store_true")

    run = commands.add_parser("run")
    run.add_argument("run_dir", type=Path)
    run.add_argument("--worker-index", type=int)
    run.add_argument("--workers", type=int)

    lacam_stage = commands.add_parser("lacam_stage")
    lacam_stage.add_argument("manifest", type=Path)
    lacam_stage.add_argument("output", type=Path)
    lacam_stage.add_argument("solution", type=Path)
    lacam_stage.add_argument("--timeout", type=float, required=True)
    lacam_stage.add_argument("--seed", type=int, required=True)
    lacam_stage.add_argument("--pibt-num", type=int, default=LACAM_PIBT_NUM)

    for name in ("aggregate", "status"):
        command = commands.add_parser(name)
        command.add_argument("run_dir", type=Path)
    return parser.parse_args()


def prepare(args: argparse.Namespace) -> None:
    sizes = args.neighborhood_sizes or [args.neighborhood_size or 4]
    sizes = list(dict.fromkeys(sizes))
    if min(args.time_limit, args.replan_time_limit, args.sample_size, *sizes) <= 0:
        raise ValueError("time limits and sample size must be positive")
    data = pd.read_csv(args.dataset)
    eligible = data[
        data.lns_neighborhood_size.eq(sizes[0])
        & data.lacam_seed.eq(args.lacam_seed)
        & data.lacam_solved.eq(1)
        & data.lacam_runtime_seconds.lt(0.8 * args.time_limit)
        & data.num_agents.between(100, 2000)
    ].drop_duplicates("instance_id")
    available = eligible.instance_id.map(
        lambda name: (args.instances / f"{name}.json").is_file()
    )
    eligible = eligible[available]
    if args.sample_size > len(eligible):
        raise ValueError(
            f"sample size {args.sample_size} exceeds {len(eligible)} eligible instances"
        )
    sampled = eligible.sample(args.sample_size, random_state=args.sample_seed)
    manifests = [args.instances / f"{name}.json" for name in sampled.instance_id]

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "instances": [str(path.resolve().relative_to(ROOT)) for path in manifests],
        "time_limit": args.time_limit,
        "replan_time_limit": args.replan_time_limit,
        "neighborhood_sizes": sizes,
        "sample_seed": args.sample_seed,
        "lacam_seed": args.lacam_seed,
        "lacam_pibt_num": LACAM_PIBT_NUM,
        "lacam_parallel_pibt": LACAM_PARALLEL,
        "lacam_parallel_refiners": LACAM_PARALLEL,
        "lns_seed": args.lns_seed,
    }
    path = run_dir / "config.json"
    if path.exists() and read_json(path) != config:
        raise RuntimeError(f"{path} belongs to another benchmark configuration")
    atomic_json(path, config)
    print(f"Prepared {len(manifests)} instances × {len(sizes)} neighborhoods in {run_dir}")


def run(args: argparse.Namespace) -> None:
    run_dir, config = load(args.run_dir)
    worker = args.worker_index
    if worker is None:
        worker = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    workers = args.workers
    if workers is None:
        workers = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    manifests = [ROOT / path for path in config["instances"]]
    if workers < 1 or worker not in range(workers):
        raise ValueError("worker must satisfy 0 <= worker < workers")
    for task in range(worker, len(manifests), workers):
        checkpoint = run_dir / "checkpoints" / f"{task:04d}.json"
        if checkpoint.exists() and read_json(checkpoint).get("status") in {
            "success", "complete"
        }:
            continue
        rows = benchmark(run_dir, manifests[task], config, task, worker, checkpoint)
        status = "success" if all(row["status"] == "success" for row in rows) else "error"
        atomic_json(checkpoint, {"status": status, "rows": rows})


def benchmark(
    run_dir: Path,
    manifest: Path,
    config: dict,
    task: int,
    worker: int,
    checkpoint: Path,
) -> list[dict]:
    sizes = neighborhood_sizes(config)
    try:
        stage(task, "problem/features start")
        problem = MapfProblem.from_manifest(manifest)
        lower_bound = analyze(problem).static["lower_bound_soc"]
        stage(task, "problem/features done; LaCAM start")
        anytime, initial = isolated_lacam(run_dir, manifest, config, task)
        stage(task, "LaCAM done")
        initial_seconds = anytime["trace"][0][0]
        if anytime["trace"][0][1] != initial.soc():
            raise RuntimeError("initial trace SOC does not match its solution")
        lns_limit = config["time_limit"] - initial_seconds
        if lns_limit <= 0:
            raise RuntimeError("no time remains for LNS")
        shared = {
            "task": task,
            "instance": problem.name,
            "manifest": str(manifest.relative_to(ROOT)),
            "agents": problem.agents,
            "lower_bound_soc": lower_bound,
            "initial_seconds": initial_seconds,
            "initial_soc": initial.soc(),
            "anytime_wall_seconds": anytime["wall_seconds"],
            "anytime_best_soc": anytime["trace"][-1][1],
            "anytime_trace": anytime["trace"],
            "benchmark_neighborhood_sizes": sizes,
            "host": socket.gethostname(),
            "worker": worker,
        }
        previous = read_json(checkpoint).get("rows", []) if checkpoint.exists() else []
        rows = {
            row["lns_neighborhood_size"]: row
            for row in previous
            if row.get("status") == "success"
        }
        for size in sizes:
            if size in rows:
                continue
            try:
                stage(task, f"LNS n={size} start")
                lns = run_lns(
                    problem,
                    initial,
                    LNSConfig(
                        lns_limit,
                        seed=config["lns_seed"],
                        neighborhood_size=size,
                        replan_time_limit=config["replan_time_limit"],
                        replan_algorithm="PP",
                    ),
                )
                stage(task, f"LNS n={size} done")
                rows[size] = {
                    **shared,
                    "status": (
                        "error" if lns.status == "failed_with_incumbent" else "success"
                    ),
                    "lns_neighborhood_size": size,
                    "lns_requested_seconds": lns_limit,
                    "lns_replan_time_limit": config["replan_time_limit"],
                    "lns_wall_seconds": lns.wall_seconds,
                    "lns_status": lns.status,
                    "lns_final_soc": lns.solution.soc(),
                    "lns_trace": [[point.seconds, point.soc] for point in lns.trace],
                    "lns_error": lns.error,
                    "lns_stdout_tail": lns.stdout_tail,
                    "lns_stderr_tail": lns.stderr_tail,
                }
            except Exception as error:
                rows[size] = error_row(
                    manifest, task, worker, size, sizes,
                    config["replan_time_limit"], error,
                )
            atomic_json(checkpoint, {"status": "partial", "rows": list(rows.values())})
        return [rows[size] for size in sizes]
    except Exception as error:
        return [
            error_row(
                manifest, task, worker, size, sizes,
                config["replan_time_limit"], error,
            )
            for size in sizes
        ]


def lacam_stage(args: argparse.Namespace) -> None:
    problem = MapfProblem.from_manifest(args.manifest)
    result = run_lacam(
        problem,
        LaCAMConfig(
            args.timeout,
            anytime=True,
            seed=args.seed,
            pibt_num=args.pibt_num,
            # LaCAM uses this shared multithreading flag for both parallel
            # PIBT candidates and its background refiners.
            parallel_pibt=LACAM_PARALLEL,
        ),
    )
    trace = [
        point for point in result.trace
        if point.seconds <= args.timeout
    ]
    if result.initial_solution is None or not trace:
        atomic_json(args.output, {
            "status": "error",
            "error": result.error or (
                f"no initial solution within the {args.timeout:g}s evaluation horizon"
            ),
        })
        return
    result.initial_solution.write(args.solution)
    atomic_json(args.output, {
        "status": "success",
        "wall_seconds": result.wall_seconds,
        "evaluation_horizon_seconds": args.timeout,
        "trace": [[point.seconds, point.soc] for point in trace],
    })


def isolated_lacam(
    run_dir: Path, manifest: Path, config: dict, task: int
) -> tuple[dict, MapfSolution]:
    directory = run_dir / "lacam"
    output = directory / f"{task:04d}.json"
    solution = directory / f"{task:04d}.json.gz"
    if (
        not output.exists()
        or not solution.exists()
        or read_json(output).get("status") != "success"
    ):
        command = [
            sys.executable, str(Path(__file__).resolve()), "lacam_stage",
            str(manifest), str(output), str(solution),
            "--timeout", str(config["time_limit"]),
            "--seed", str(config["lacam_seed"]),
            "--pibt-num", str(config.get("lacam_pibt_num", LACAM_PIBT_NUM)),
        ]
        try:
            process = subprocess.run(
                command,
                check=False,
                timeout=max(
                    config["time_limit"] + 10,
                    config["time_limit"] * LACAM_WATCHDOG_MULTIPLIER,
                ),
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("LaCAM subprocess exceeded its watchdog timeout") from error
        if process.returncode:
            raise RuntimeError(f"LaCAM subprocess terminated with {_exit_name(process.returncode)}")
    result = read_json(output)
    if result.get("status") != "success" or not solution.exists():
        raise RuntimeError(f"LaCAM failed: {result.get('error', 'missing solution')}")
    return result, MapfSolution.read(solution)


def _exit_name(returncode: int) -> str:
    if returncode < 0:
        try:
            return f"{signal.Signals(-returncode).name} ({returncode})"
        except ValueError:
            pass
    return f"exit code {returncode}"


def error_row(manifest, task, worker, size, sizes, replan_time_limit, error):
    return {
        "task": task,
        "status": "error",
        "instance": manifest.stem,
        "manifest": str(manifest.relative_to(ROOT)),
        "lns_neighborhood_size": size,
        "benchmark_neighborhood_sizes": sizes,
        "lns_replan_time_limit": replan_time_limit,
        "host": socket.gethostname(),
        "worker": worker,
        "error": f"{type(error).__name__}: {error}",
    }


def stage(task: int, message: str) -> None:
    print(f"task {task}: {message}", flush=True)


def aggregate(args: argparse.Namespace) -> None:
    run_dir, config = load(args.run_dir)
    output = run_dir / "results.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for path in sorted((run_dir / "checkpoints").glob("*.json")):
            checkpoint = read_json(path)
            rows = checkpoint.get("rows", [checkpoint])
            rows.sort(key=lambda row: row.get("lns_neighborhood_size", 4))
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                count += 1
    temporary.replace(output)
    expected = len(config["instances"]) * len(neighborhood_sizes(config))
    print(f"Wrote {count}/{expected} rows to {output}")


def status(args: argparse.Namespace) -> None:
    run_dir, config = load(args.run_dir)
    checkpoints = list((run_dir / "checkpoints").glob("*.json"))
    success = 0
    for path in checkpoints:
        value = read_json(path)
        success += sum(
            row.get("status") == "success" for row in value.get("rows", [value])
        )
    expected = len(config["instances"]) * len(neighborhood_sizes(config))
    print(f"{len(checkpoints)}/{len(config['instances'])} instances; {success}/{expected} successful")


def submit(args: argparse.Namespace) -> None:
    run_dir, config = load(args.run_dir)
    instances = len(config["instances"])
    workers = min(args.workers, instances)
    if workers < 1:
        raise ValueError("workers must be positive")
    if args.retry_failed:
        tasks = [
            task for task in range(instances)
            if not task_complete(run_dir / "checkpoints" / f"{task:04d}.json")
        ]
        if not tasks:
            print("No failed or missing tasks to retry")
            return
        array = ",".join(map(str, tasks))
        run_arguments = ["run", str(run_dir), "--workers", str(instances)]
    else:
        array = f"0-{workers - 1}"
        run_arguments = ["run", str(run_dir)]
    logs = run_dir / "logs"
    logs.mkdir(exist_ok=True)
    script = ROOT / "scripts/slurm/benchmark_s2.slurm"
    job = job_id(
        subprocess.check_output(
            [
                "sbatch", "--parsable", f"--array={array}",
                f"--output={logs}/%x-%A_%a.out", f"--error={logs}/%x-%A_%a.err",
                str(script), *run_arguments,
            ],
            text=True,
        ).strip()
    )
    aggregate_job = job_id(
        subprocess.check_output(
            [
                "sbatch", "--parsable", f"--dependency=afterany:{job}",
                f"--output={logs}/%x-%j.out", f"--error={logs}/%x-%j.err",
                str(script), "aggregate", str(run_dir),
            ],
            text=True,
        ).strip()
    )
    submitted = len(tasks) if args.retry_failed else workers
    print(f"tasks={submitted} workers={job} aggregate={aggregate_job}")


def task_complete(path: Path) -> bool:
    return path.exists() and read_json(path).get("status") in {"success", "complete"}


def load(path: Path) -> tuple[Path, dict]:
    run_dir = path.resolve()
    return run_dir, read_json(run_dir / "config.json")


def neighborhood_sizes(config: dict) -> list[int]:
    return config.get("neighborhood_sizes") or [config.get("neighborhood_size", 4)]


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def job_id(output: str) -> str:
    return output.split(";", 1)[0]


def main() -> None:
    faulthandler.enable()
    args = arguments()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
