#!/usr/bin/env python3
"""Prepare, run and aggregate sequential MAPF policy experiments."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from mapf_anytime.anytime import run_sequence
from mapf_anytime.policies.layered_v1 import (
    LayeredPolicyV1,
    TrainingConfig as V1TrainingConfig,
    prepare as prepare_layered_v1,
)
from mapf_anytime.policies.layered_variable_neighbourhood_v1 import (
    LayeredVariableNeighbourhoodPolicyV1,
    TrainingConfig as VariableNeighbourhoodV1TrainingConfig,
    prepare as prepare_layered_variable_neighbourhood_v1,
)
from mapf_anytime.policies.lacam_lns import LaCAMLNSPolicy
from mapf_anytime.policies.lacam_naive import LaCAMNaivePolicy
from mapf_anytime.policies.naive_budget import NaiveBudgetPolicy
from mapf_anytime.problem import MapfProblem


######## PREPARE


LEARNED_POLICIES = {"layered_v1", "layered_variable_neighbourhood_v1"}


def stage_problems(
    problems: list[MapfProblem], destination: Path | None
) -> list[MapfProblem]:
    """Copy timed solver inputs to an explicitly selected local directory."""
    if destination is None:
        return problems

    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    staged: dict[Path, Path] = {}

    def stage(path: Path) -> Path:
        source = path.resolve()
        if source not in staged:
            digest = hashlib.sha256(str(source).encode()).hexdigest()[:16]
            target = destination / f"{digest}-{source.name}"
            shutil.copy2(source, target)
            staged[source] = target
        return staged[source]

    return [
        replace(
            problem,
            map_path=stage(problem.map_path),
            scenario_path=stage(problem.scenario_path),
        )
        for problem in problems
    ]


def default_early_stop(name: str) -> float | None:
    return None if name in LEARNED_POLICIES else 10.0


def configure_learned_policy(policy, parameters: dict) -> None:
    percentile = float(
        parameters.get(
            "survival_limit_percentile", policy.survival_limit_percentile
        )
    )
    multiplier = float(parameters.get("survival_runtime_multiplier", 1.0))
    if not 0.5 <= percentile <= 1.0:
        raise ValueError("survival_limit_percentile must be between 0.5 and 1")
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("survival_runtime_multiplier must be positive and finite")
    policy.survival_limit_percentile = percentile
    policy.survival_runtime_multiplier = multiplier


def read_manifest_list(path: Path) -> list[Path]:
    path = path.resolve()
    manifests = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        manifest = Path(value).expanduser()
        manifest = (
            manifest if manifest.is_absolute() else path.parent / manifest
        ).resolve()
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        manifests.append(manifest)
    if not manifests:
        raise ValueError(f"No manifests listed in {path}")
    return manifests


def require_excluded_instances(excluded: set[str], split_path: Path) -> None:
    included = set(pd.read_csv(split_path).instance_id.astype(str))
    leaked = sorted(excluded & included)
    if leaked:
        raise RuntimeError(
            f"{len(leaked)} experiment instances leaked into model preparation: "
            + ", ".join(leaked[:10])
        )


def prepare(config_path: Path, experiment: Path) -> None:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.parent
    experiment.mkdir(parents=True, exist_ok=True)

    manifests = read_manifest_list(root / config["instance_list"])
    sequence_size = int(config.get("sequence_size", len(manifests)))
    repetitions = int(config.get("repetitions", 1))
    seed = int(config.get("seed", 1))
    if sequence_size <= 0 or repetitions <= 0:
        raise ValueError("sequence_size and repetitions must be positive")
    if sequence_size > len(manifests):
        raise ValueError(
            f"Requested {sequence_size} instances from a pool of {len(manifests)}"
        )

    sequences = {}
    names = [str(path.resolve()) for path in manifests]
    listed_ids = [MapfProblem.from_manifest(path).name for path in manifests]
    if len(listed_ids) != len(set(listed_ids)):
        raise ValueError("The instance list contains duplicate instance IDs")
    id_by_path = dict(zip(names, listed_ids))
    for repetition in range(1, repetitions + 1):
        order = np.random.default_rng(seed + repetition - 1).permutation(names)
        sequences[str(repetition)] = order[:sequence_size].tolist()
    evaluation_ids = {
        id_by_path[path]
        for sequence in sequences.values()
        for path in sequence
    }

    policies = config["policies"]
    if len({item["name"] for item in policies}) != len(policies):
        raise ValueError("Each policy name must occur only once")

    tasks = []
    for specification in policies:
        name = specification["name"]
        options = specification.get("prepare", {})
        model_dir = experiment / "models" / name
        if name in {"layered_v1", "layered_variable_neighbourhood_v1"}:
            dataset_value = options.get("dataset", config.get("dataset"))
            if dataset_value is None:
                raise ValueError(f"The {name} policy requires a dataset")
            dataset = (root / dataset_value).resolve()
            if name == "layered_v1":
                training = V1TrainingConfig(
                    split_seed=int(options.get("split_seed", 42)),
                    neighborhood_size=int(options.get("neighborhood_size", 4)),
                    jobs=int(options.get("jobs", 1)),
                    survival_limit_percentile=float(
                        options.get("survival_limit_percentile", 0.8)
                    ),
                )
                prepare_layered_v1(
                    dataset, model_dir, training,
                    exclude_instances=evaluation_ids,
                )
            else:
                training = VariableNeighbourhoodV1TrainingConfig(
                    split_seed=int(options.get("split_seed", 42)),
                    jobs=int(options.get("jobs", 1)),
                    survival_limit_percentile=float(
                        options.get("survival_limit_percentile", 0.8)
                    ),
                )
                prepare_layered_variable_neighbourhood_v1(
                    dataset, model_dir, training,
                    exclude_instances=evaluation_ids,
                )
            require_excluded_instances(
                evaluation_ids, model_dir / "instance_split.csv"
            )
        elif name in {
            "naive_budget",
            "lacam_naive",
            "lacam_lns",
        }:
            pass
        else:
            raise ValueError(f"Unknown policy: {name}")

        runs = specification.get("runs", [{}])
        if not runs:
            raise ValueError(f"{name} has no runs")
        for parameters in runs:
            if not isinstance(parameters, dict):
                raise TypeError(f"{name} run parameters must be JSON objects")
            early_stop = parameters.get(
                "early_stop_seconds", default_early_stop(name)
            )
            if early_stop is not None and float(early_stop) <= 0:
                raise ValueError(
                    "early_stop_seconds must be positive or null"
                )
            for repetition in range(1, repetitions + 1):
                tasks.append(
                    {
                        "task": len(tasks),
                        "policy": name,
                        "parameters": parameters,
                        "repetition": repetition,
                    }
                )

    _write_json(
        experiment / "plan.json",
        {
            "source_config": str(config_path),
            "config": config,
            "sequences": sequences,
            "tasks": tasks,
        },
    )
    print(
        f"Prepared {len(tasks)} tasks in {experiment} "
        f"(Slurm array: 0-{len(tasks) - 1})"
    )


######## RUN


def run(
    experiment: Path,
    task_number: int | None,
    local_input_dir: Path | None = None,
) -> None:
    plan = json.loads((experiment / "plan.json").read_text(encoding="utf-8"))
    tasks = plan["tasks"]
    if task_number is not None and not 0 <= task_number < len(tasks):
        raise ValueError(f"Task must be between 0 and {len(tasks) - 1}")
    selected = tasks if task_number is None else [tasks[task_number]]
    results = experiment / "results"
    results.mkdir(exist_ok=True)

    for task in selected:
        name = task["policy"]
        parameters = task["parameters"]
        repetition = int(task["repetition"])

        policy_types = {
            "layered_v1": LayeredPolicyV1,
            "layered_variable_neighbourhood_v1": (
                LayeredVariableNeighbourhoodPolicyV1
            ),
            "lacam_naive": LaCAMNaivePolicy,
            "lacam_lns": LaCAMLNSPolicy,
            "naive_budget": NaiveBudgetPolicy,
        }
        if name in policy_types:
            if "budget" not in parameters:
                raise ValueError(f"A {name} run requires a budget")
            policy = policy_types[name].load(
                experiment / "models" / name, seed=repetition
            )
            if name in LEARNED_POLICIES:
                configure_learned_policy(policy, parameters)
            early_stop = parameters.get(
                "early_stop_seconds", default_early_stop(name)
            )
            policy.early_stop_seconds = (
                None if early_stop is None else float(early_stop)
            )
            policy_budget = float(parameters["budget"])
            hard_limit = max(
                0.0, float(parameters.get("hard_limit", policy_budget))
            )
        else:
            raise ValueError(f"Unknown policy: {name}")

        problems = [
            MapfProblem.from_manifest(path)
            for path in plan["sequences"][str(repetition)]
        ]
        problems = stage_problems(problems, local_input_dir)
        started = time.perf_counter()
        rows = run_sequence(policy, problems, policy_budget, hard_limit)
        wall_seconds = time.perf_counter() - started
        result = {
            **task,
            "wall_seconds": wall_seconds,
            "rows": rows,
        }
        path = results / f"task_{task['task']:06d}.json"
        _write_json(path, result)
        print(f"Completed task {task['task']}: {name} {parameters} in {wall_seconds:.2f}s")


######## AGGREGATE


def aggregate(experiment: Path) -> None:
    plan = json.loads((experiment / "plan.json").read_text(encoding="utf-8"))
    records = []
    missing = []
    for task in plan["tasks"]:
        path = experiment / "results" / f"task_{task['task']:06d}.json"
        if not path.exists():
            missing.append(task["task"])
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        parameters = {
            f"parameter_{name}": value
            for name, value in result["parameters"].items()
        }
        for row in result["rows"]:
            record = {
                "task": result["task"],
                "policy": result["policy"],
                "repetition": result["repetition"],
                "task_wall_seconds": result["wall_seconds"],
                **parameters,
                **row,
            }
            for name, value in record.items():
                if isinstance(value, (dict, list)):
                    record[name] = json.dumps(value, separators=(",", ":"))
            records.append(record)

    if not records:
        raise ValueError("No completed tasks to aggregate")
    destination = experiment / "results.csv"
    pd.DataFrame(records).to_csv(destination, index=False)
    print(f"Wrote {len(records)} rows to {destination}")
    if missing:
        print(f"Missing tasks: {', '.join(map(str, missing))}")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("config", type=Path)
    prepare_parser.add_argument("experiment", type=Path)

    run_parser = commands.add_parser("run")
    run_parser.add_argument("experiment", type=Path)
    run_parser.add_argument("--task", type=int)
    run_parser.add_argument("--local-input-dir", type=Path)

    aggregate_parser = commands.add_parser("aggregate")
    aggregate_parser.add_argument("experiment", type=Path)

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.config, args.experiment)
    elif args.command == "run":
        run(args.experiment, args.task, args.local_input_dir)
    else:
        aggregate(args.experiment)


if __name__ == "__main__":
    main()
