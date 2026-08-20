from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
from typing import Any, Iterable

from ..features import FeatureSet, LACAM_FEATURES, STATIC_FEATURES, analyze
from ..problem import MapfProblem
from ..solution import MapfSolution
from ..solvers import LaCAMConfig, LNSConfig, run_lacam, run_lns


@dataclass(frozen=True)
class DatasetConfig:
    run_dir: Path
    lacam_timeout: float = 40.0
    lacam_seed: int = 0
    lacam_pibt_num: int = 1
    lns_timeout: float = 100.0
    neighborhood_sizes: tuple[int, ...] = (2, 4, 8, 16, 32)
    lns_repetitions: int = 4
    replan_time_limit: float = 10.0
    probe_iterations: int = 5
    cells_mode: str = "single-path"
    feature_threads: int = 1
    lns_max_iterations: int = 1_000_000

    def validate(self) -> None:
        positive = (
            self.lacam_timeout,
            self.lacam_pibt_num,
            self.lns_timeout,
            self.lns_repetitions,
            self.replan_time_limit,
            self.probe_iterations,
            self.feature_threads,
            self.lns_max_iterations,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError(
                "Timeouts, repetitions, probes and threads must be positive"
            )
        if self.lacam_seed < 0:
            raise ValueError("LaCAM seed must be non-negative")
        if not self.neighborhood_sizes or any(
            size <= 0 for size in self.neighborhood_sizes
        ):
            raise ValueError("Neighbourhood sizes must be positive")

    @property
    def lns_horizon(self) -> int:
        return math.ceil(self.lns_timeout)


def read_instances(path: str | Path, limit: int | None = None) -> list[Path]:
    path = Path(path).expanduser().resolve()
    instances = [
        (path.parent / line.strip()).expanduser().resolve()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return instances[:limit] if limit is not None else instances


def initialize(
    config: DatasetConfig, instance_list: Path, instances: list[Path]
) -> None:
    config.validate()
    if not instances:
        raise ValueError(f"No instances found in {instance_list}")
    expected = _config_record(config, instance_list, instances)
    path = config.run_dir / "config.json"
    config.run_dir.mkdir(parents=True, exist_ok=True)
    if path.exists() and read_json(path) != expected:
        raise RuntimeError(f"{path} belongs to another dataset configuration")
    if not path.exists():
        atomic_json(path, expected)


def extend_neighborhoods(
    config: DatasetConfig, instance_list: Path, instances: list[Path]
) -> None:
    config.validate()
    path = config.run_dir / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"No existing dataset run at {config.run_dir}")
    current = read_json(path)
    expected = _config_record(config, instance_list, instances)
    old_sizes = set(current.pop("neighborhood_sizes", ()))
    new_sizes = set(expected.pop("neighborhood_sizes"))
    if current != expected:
        raise RuntimeError("Only neighborhood_sizes may change when extending LNS")
    if not old_sizes <= new_sizes:
        raise RuntimeError("New neighborhood_sizes must retain every existing size")
    expected["neighborhood_sizes"] = sorted(new_sizes)
    atomic_json(path, expected)


def _config_record(
    config: DatasetConfig, instance_list: Path, instances: list[Path]
) -> dict[str, Any]:
    settings = asdict(config)
    settings.pop("run_dir")
    return {
        "schema_version": 3,
        **_jsonable(settings),
        "instance_list": str(instance_list),
        "instance_count": len(instances),
        "instance_list_digest": hashlib.sha256(instance_list.read_bytes()).hexdigest(),
    }


def run_lacam_stage(
    config: DatasetConfig,
    instances: list[Path],
    worker: int = 0,
    workers: int = 1,
) -> int:
    _validate_worker(worker, workers)
    failures = 0
    total = len(instances)
    for task in range(worker, total, workers):
        index = task
        checkpoint = lacam_checkpoint(config, index)
        if complete(checkpoint) and complete(static_checkpoint(config, index)):
            continue
        process = multiprocessing.get_context("spawn").Process(
            target=_run_lacam, args=(config, instances[index], index)
        )
        process.start()
        process.join()
        if complete(checkpoint):
            continue
        failures += 1
        if not checkpoint.exists() or process.exitcode != 0:
            atomic_json(
                checkpoint,
                {
                    "status": "error",
                    "manifest": str(instances[index]),
                    "lacam_seed": config.lacam_seed,
                    "error": f"worker exited with code {process.exitcode}",
                },
            )
        print(f"LaCAM task {task} failed: {read_json(checkpoint)['error']}", flush=True)
    return int(bool(failures))


def run_lns_stage(
    config: DatasetConfig,
    instances: list[Path],
    worker: int = 0,
    workers: int = 1,
) -> int:
    _validate_worker(worker, workers)
    failures = 0
    per_instance = len(config.neighborhood_sizes) * config.lns_repetitions
    for task in range(worker, len(instances) * per_instance, workers):
        index, remainder = divmod(task, per_instance)
        size_index, repetition = divmod(remainder, config.lns_repetitions)
        size = config.neighborhood_sizes[size_index]
        checkpoint = lns_checkpoint(config, index, size, repetition)
        if complete(checkpoint):
            continue
        try:
            payload = _run_lns(
                config, instances[index], index, size, repetition
            )
        except Exception as error:
            failures += 1
            payload = {"status": "error", "error": f"{type(error).__name__}: {error}"}
        atomic_json(
            checkpoint,
            {
                **payload,
                "manifest": str(instances[index]),
                "neighborhood_size": size,
                "replan_time_limit": config.replan_time_limit,
                "lns_repetitions": config.lns_repetitions,
                "repetition": repetition + 1,
                "lns_seed": repetition + 1,
            },
        )
        if payload["status"] == "error":
            print(f"LNS task {task} failed: {payload['error']}", flush=True)
    return int(bool(failures))


def status(
    config: DatasetConfig, instances: list[Path]
) -> dict[str, tuple[int, int]]:
    count = len(instances)
    lns_total = count * len(config.neighborhood_sizes) * config.lns_repetitions
    return {
        "static": (
            sum(complete(static_checkpoint(config, i)) for i in range(count)),
            count,
        ),
        "lacam": (
            sum(complete(lacam_checkpoint(config, i)) for i in range(count)),
            count,
        ),
        "lns": (
            sum(
                complete(lns_checkpoint(config, i, size, repetition))
                for i in range(count)
                for size in config.neighborhood_sizes
                for repetition in range(config.lns_repetitions)
            ),
            lns_total,
        ),
    }


def aggregate(
    config: DatasetConfig, instances: list[Path], workers: int = 8
) -> tuple[Path, Path]:
    if workers < 1:
        raise ValueError("Aggregation workers must be positive")

    probe = config.probe_iterations
    raw = config.run_dir / "dataset_raw.csv"
    model = config.run_dir / "dataset.csv"
    raw_tmp, model_tmp = raw.with_suffix(".csv.tmp"), model.with_suffix(".csv.tmp")
    with ThreadPoolExecutor(max_workers=workers) as executor, raw_tmp.open(
        "w", newline="", encoding="utf-8"
    ) as raw_handle, model_tmp.open("w", newline="", encoding="utf-8") as model_handle:
        raw_writer = csv.DictWriter(
            raw_handle,
            fieldnames=_raw_columns(probe),
        )
        model_writer = csv.DictWriter(
            model_handle,
            fieldnames=_model_columns(probe),
        )
        raw_writer.writeheader()
        model_writer.writeheader()
        for index, manifest in enumerate(instances):
            paths = [
                static_checkpoint(config, index),
                lacam_checkpoint(config, index),
                *(
                    lns_checkpoint(config, index, size, repetition)
                    for size in config.neighborhood_sizes
                    for repetition in range(config.lns_repetitions)
                ),
            ]
            static, lacam, *lns_runs = executor.map(_read_complete, paths)
            base = _base_row(config, manifest, static, lacam)
            for size_index, size in enumerate(config.neighborhood_sizes):
                start = size_index * config.lns_repetitions
                runs = lns_runs[start : start + config.lns_repetitions]
                raw_writer.writerows(_raw_row(base, size, run, probe) for run in runs)
                model_writer.writerow(_model_row(base, size, runs, probe))
    raw_tmp.replace(raw)
    model_tmp.replace(model)
    return raw, model


def _run_lacam(config: DatasetConfig, manifest: Path, index: int) -> None:
    checkpoint = lacam_checkpoint(config, index)
    try:
        problem = MapfProblem.from_manifest(manifest)
        if not complete(static_checkpoint(config, index)):
            features = analyze(problem, config.cells_mode, config.feature_threads)
            atomic_json(
                static_checkpoint(config, index),
                {
                    "status": "success",
                    "manifest": str(manifest),
                    "features": features.static,
                    "distances": features.distances,
                },
            )
        result = run_lacam(
            problem,
            LaCAMConfig(
                timeout=config.lacam_timeout,
                anytime=False,
                seed=config.lacam_seed,
                pibt_num=config.lacam_pibt_num,
            ),
        )
        solved = result.solution is not None
        initial_path = solution_path(config, index)
        if solved:
            result.solution.write(initial_path)
            runtime = result.trace[0].seconds if result.trace else result.wall_seconds
            initial_soc = result.solution.soc()
        else:
            runtime = initial_soc = None
        payload = {
            "status": "success" if solved else result.status,
            "manifest": str(manifest),
            "instance_id": problem.name,
            "lacam_seed": config.lacam_seed,
            "lacam_pibt_num": config.lacam_pibt_num,
            "timeout_seconds": config.lacam_timeout,
            "wall_seconds": result.wall_seconds,
            "runtime_seconds": runtime,
            "initial_soc": initial_soc,
            "initial_solution": str(initial_path) if solved else None,
            "error": result.error,
        }
    except Exception as error:
        payload = {
            "status": "error",
            "manifest": str(manifest),
            "lacam_seed": config.lacam_seed,
            "error": f"{type(error).__name__}: {error}",
        }
    atomic_json(checkpoint, payload)


def _run_lns(
    config: DatasetConfig,
    manifest: Path,
    index: int,
    size: int,
    repetition: int,
) -> dict[str, Any]:
    lacam = read_json(lacam_checkpoint(config, index))
    if lacam["status"] == "no_solution":
        return {"status": "skipped", "reason": "lacam_unsolved"}
    if lacam["status"] != "success":
        raise RuntimeError(f"LaCAM ended with {lacam['status']}")
    initial_soc = int(lacam["initial_soc"])
    lower_bound = float(
        read_json(static_checkpoint(config, index))["features"]["lower_bound_soc"]
    )
    if initial_soc <= lower_bound:
        return {
            "status": "skipped",
            "reason": "lacam_zero_sod",
            "wall_seconds": 0.0,
            "final_soc": initial_soc,
            "soc_curve": [initial_soc] * (config.lns_horizon + 1),
            "trace": [],
            "probe_reduction": 0.0,
            "probe_runtime": 0.0,
        }

    result = run_lns(
        MapfProblem.from_manifest(manifest),
        MapfSolution.read(lacam["initial_solution"]),
        LNSConfig(
            timeout=config.lns_timeout,
            seed=repetition + 1,
            neighborhood_size=size,
            max_iterations=config.lns_max_iterations,
            replan_time_limit=config.replan_time_limit,
            early_stop_seconds=None,
        ),
    )
    if result.status not in {"success", "timeout_with_incumbent"}:
        raise RuntimeError(result.error or f"LNS ended with {result.status}")
    curve = _best_by_second(
        initial_soc,
        [(point.seconds, point.soc) for point in result.trace],
        config.lns_horizon,
    )
    probe = (
        result.trace[config.probe_iterations]
        if len(result.trace) > config.probe_iterations
        else None
    )
    return {
        "status": "success",
        "solver_status": result.status,
        "wall_seconds": result.wall_seconds,
        "final_soc": result.solution.soc(),
        "soc_curve": curve,
        "trace": [asdict(point) for point in result.trace],
        "probe_reduction": max(0, initial_soc - probe.soc) if probe else None,
        "probe_runtime": probe.seconds if probe else None,
        "error": result.error,
    }


def _base_row(
    config: DatasetConfig,
    manifest: Path,
    static: dict[str, Any],
    lacam: dict[str, Any],
) -> dict[str, Any]:
    problem = MapfProblem.from_manifest(manifest)
    metadata = read_json(manifest)
    solved = lacam["status"] == "success"
    features = {}
    if solved:
        features = FeatureSet(
            static["features"], tuple(static["distances"])
        ).with_lacam(MapfSolution.read(lacam["initial_solution"])).lacam
    return {
        "source": _source(metadata, problem.name),
        "instance_id": problem.name,
        "map_name": Path(metadata.get("source_map", problem.map_path)).name,
        "manifest": str(manifest),
        "instance_metadata": json.dumps(metadata, separators=(",", ":")),
        **static["features"],
        "lacam_seed": config.lacam_seed,
        "lacam_solved": int(solved),
        "lacam_runtime_seconds": (
            lacam.get("runtime_seconds") if solved else config.lacam_timeout
        ),
        "lacam_wall_seconds": lacam.get("wall_seconds"),
        "lacam_initial_soc": lacam.get("initial_soc"),
        **{name: features.get(name) for name in LACAM_FEATURES},
    }


def _raw_row(
    base: dict[str, Any], size: int, run: dict[str, Any], probe: int
) -> dict[str, Any]:
    lower_bound = float(base["lower_bound_soc"])
    curve = run.get("soc_curve", [])
    return {
        **base,
        "lns_neighborhood_size": size,
        "lns_replan_time_limit": run["replan_time_limit"],
        "lns_repetitions": run["lns_repetitions"],
        "repetition": run["repetition"],
        "lns_seed": run["lns_seed"],
        "lns_status": run["status"],
        "lns_solver_status": run.get("solver_status"),
        "lns_error": run.get("error") or run.get("reason"),
        "lns_wall_seconds": run.get("wall_seconds"),
        "lns_best_soc_by_second": _curve(curve),
        "lns_best_sod_by_second": _curve(
            [max(0.0, value - lower_bound) for value in curve]
        ),
        f"lns_sod_reduction_after_{probe}_iterations": run.get("probe_reduction"),
        f"lns_internal_runtime_after_{probe}_iterations": run.get("probe_runtime"),
    }


def _model_row(
    base: dict[str, Any],
    size: int,
    runs: list[dict[str, Any]],
    probe: int,
) -> dict[str, Any]:
    lower_bound = float(base["lower_bound_soc"])
    soc_curve = _mean_curve([run.get("soc_curve", []) for run in runs])
    sod_curve = [max(0.0, value - lower_bound) for value in soc_curve]
    initial_sod = base.get("lacam_initial_sod")
    final_sod = sod_curve[-1] if sod_curve else None
    reduction = (
        max(0.0, initial_sod - final_sod)
        if initial_sod is not None and final_sod is not None
        else None
    )
    probe_reduction = _mean(run.get("probe_reduction") for run in runs)
    probe_runtime = _mean(run.get("probe_runtime") for run in runs)
    return {
        **base,
        "lns_neighborhood_size": size,
        "lns_replan_time_limit": runs[0]["replan_time_limit"],
        "lns_repetitions": len(runs),
        "lns_successful_repetitions": sum(run["status"] == "success" for run in runs),
        "lns_wall_seconds": _mean(run.get("wall_seconds") for run in runs),
        "lns_best_soc_by_second": _curve(soc_curve),
        "lns_best_sod_by_second": _curve(sod_curve),
        "final_lns_soc": soc_curve[-1] if soc_curve else None,
        "final_lns_sod": final_sod,
        "lns_curve_seconds": max(0, len(soc_curve) - 1),
        "target_sod_improvement": reduction,
        "target_sodn_improvement": (
            reduction / base["num_agents"] if reduction is not None else None
        ),
        "target_sodlb_improvement": (
            reduction / lower_bound if reduction is not None else None
        ),
        "target_sodfraction_improvement": (
            reduction / initial_sod
            if reduction is not None and initial_sod > 0
            else 0.0
        ),
        "target_elbow_time_seconds": _elbow(initial_sod, sod_curve),
        f"lns_sod_reduction_after_{probe}_iterations": (
            probe_reduction if probe_reduction is not None else -1.0
        ),
        f"lns_internal_runtime_after_{probe}_iterations": (
            probe_runtime if probe_runtime is not None else -1.0
        ),
        f"lns_{probe}iter_probe_incomplete": int(
            probe_reduction is None or probe_runtime is None
        ),
        f"lns_sod_reduction_per_agent_after_{probe}_iterations": (
            probe_reduction / base["num_agents"]
            if probe_reduction is not None
            else -1.0
        ),
    }


def _base_columns() -> list[str]:
    return [
        "source",
        "instance_id",
        "map_name",
        "manifest",
        "instance_metadata",
        *STATIC_FEATURES,
        "lacam_seed",
        "lacam_solved",
        "lacam_runtime_seconds",
        "lacam_wall_seconds",
        "lacam_initial_soc",
        *LACAM_FEATURES,
    ]


def _raw_columns(probe: int) -> list[str]:
    return [
        *_base_columns(),
        "lns_neighborhood_size",
        "lns_replan_time_limit",
        "lns_repetitions",
        "repetition",
        "lns_seed",
        "lns_status",
        "lns_solver_status",
        "lns_error",
        "lns_wall_seconds",
        "lns_best_soc_by_second",
        "lns_best_sod_by_second",
        f"lns_sod_reduction_after_{probe}_iterations",
        f"lns_internal_runtime_after_{probe}_iterations",
    ]


def _model_columns(probe: int) -> list[str]:
    return [
        *_base_columns(),
        "lns_neighborhood_size",
        "lns_replan_time_limit",
        "lns_repetitions",
        "lns_successful_repetitions",
        "lns_wall_seconds",
        "lns_best_soc_by_second",
        "lns_best_sod_by_second",
        "final_lns_soc",
        "final_lns_sod",
        "lns_curve_seconds",
        "target_sod_improvement",
        "target_sodn_improvement",
        "target_sodlb_improvement",
        "target_sodfraction_improvement",
        "target_elbow_time_seconds",
        f"lns_sod_reduction_after_{probe}_iterations",
        f"lns_internal_runtime_after_{probe}_iterations",
        f"lns_{probe}iter_probe_incomplete",
        f"lns_sod_reduction_per_agent_after_{probe}_iterations",
    ]


def static_checkpoint(config: DatasetConfig, index: int) -> Path:
    return config.run_dir / "checkpoints" / "static" / f"{index:06d}.json"


def lacam_checkpoint(config: DatasetConfig, index: int) -> Path:
    return config.run_dir / "checkpoints" / "lacam" / f"{index:06d}.json"


def lns_checkpoint(
    config: DatasetConfig, index: int, size: int, repetition: int
) -> Path:
    name = f"{index:06d}-n{size}-r{repetition + 1}.json"
    return config.run_dir / "checkpoints" / "lns" / name


def solution_path(config: DatasetConfig, index: int) -> Path:
    return config.run_dir / "solutions" / f"{index:06d}-initial.json.gz"


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_complete(path: Path) -> dict[str, Any]:
    try:
        checkpoint = read_json(path)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Incomplete checkpoint: {path}") from error
    if checkpoint.get("status") not in {"success", "no_solution", "skipped"}:
        raise RuntimeError(
            f"Incomplete checkpoint: {path} ({checkpoint.get('status', 'no status')})"
        )
    return checkpoint


def complete(path: Path) -> bool:
    try:
        return read_json(path).get("status") in {"success", "no_solution", "skipped"}
    except (OSError, ValueError):
        return False


def _validate_worker(worker: int, workers: int) -> None:
    if workers < 1 or worker not in range(workers):
        raise ValueError("worker must satisfy 0 <= worker < workers")


def _best_by_second(
    initial: float, trace: list[tuple[float, int]], horizon: int
) -> list[float]:
    curve, best, index = [], float(initial), 0
    trace.sort()
    for second in range(horizon + 1):
        while index < len(trace) and trace[index][0] <= second:
            best = min(best, trace[index][1])
            index += 1
        curve.append(best)
    return curve


def _mean(values: Iterable[float | None]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _mean_curve(curves: list[list[float]]) -> list[float]:
    curves = [curve for curve in curves if curve]
    return [sum(values) / len(values) for values in zip(*curves)] if curves else []


def _curve(values: list[float]) -> str:
    return json.dumps(values, separators=(",", ":"))


def _elbow(initial: float | None, curve: list[float]) -> float | None:
    if initial is None or not curve:
        return None
    total = max(0.0, initial - curve[-1])
    if total == 0:
        return 0.0
    return float(
        next(
            (
                second
                for second, value in enumerate(curve)
                if max(0.0, value - curve[-1]) <= 0.05 * total
            ),
            len(curve) - 1,
        )
    )


def _source(metadata: dict[str, Any], name: str) -> str:
    marker = f"{name} {metadata.get('generator', '')}".lower()
    return "pogema" if "pogema" in marker else "movingai"


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return value
