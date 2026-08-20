from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import math
from pathlib import Path
import sys
import time

from ..problem import PASSABLE, MapfProblem
from ..solution import MapfSolution
from .trace import TracePoint


NATIVE = Path(__file__).parent / "native" / "lacam"


@dataclass(frozen=True)
class LaCAMConfig:
    timeout: float
    anytime: bool = False
    seed: int = 0
    post_solution_seconds: float | None = None
    pibt_num: int = 1
    parallel_pibt: bool = True


@dataclass
class LaCAMResult:
    status: str
    wall_seconds: float
    solution: MapfSolution | None = None
    error: str | None = None
    trace: list[TracePoint] = field(default_factory=list)
    initial_solution: MapfSolution | None = None


def run_lacam(problem: MapfProblem, config: LaCAMConfig) -> LaCAMResult:
    if config.pibt_num < 1:
        raise ValueError("pibt_num must be positive")
    if config.post_solution_seconds is not None:
        if (
            not config.anytime
            or not math.isfinite(config.post_solution_seconds)
            or config.post_solution_seconds <= 0
        ):
            raise ValueError(
                "post_solution_seconds requires anytime=True and must be positive"
            )
    started = time.perf_counter()
    try:
        data = problem.map()
        starts, goals = problem.starts_goals()
        grid = [[0 if tile in PASSABLE else 1 for tile in row] for row in data.grid]
        raw, trace, raw_initial = _extension().solve_with_trace(
            grid,
            [(y, x) for x, y in starts],
            [(y, x) for x, y in goals],
            max(0.001, float(config.timeout)),
            config.anytime,
            config.seed,
            config.post_solution_seconds or -1,
            config.pibt_num,
            config.parallel_pibt,
        )
        if not raw:
            return LaCAMResult(
                "no_solution",
                time.perf_counter() - started,
                trace=[TracePoint(ms / 1000, soc) for ms, soc in trace],
            )
        solution = MapfSolution.from_raw(
            [[(column, row) for row, column in path] for path in raw]
        )
        initial_solution = MapfSolution.from_raw(
            [[(column, row) for row, column in path] for path in raw_initial]
        )
        return LaCAMResult(
            "success",
            time.perf_counter() - started,
            solution,
            trace=[TracePoint(ms / 1000, soc) for ms, soc in trace],
            initial_solution=initial_solution,
        )
    except Exception as error:
        return LaCAMResult(
            "error",
            time.perf_counter() - started,
            error=f"{type(error).__name__}: {error}",
        )


def _extension():
    binding = (
        NATIVE
        / f"build-pybind-{sys.version_info.major}{sys.version_info.minor}"
        / "bindings"
    )
    if str(binding) not in sys.path:
        sys.path.insert(0, str(binding))
    try:
        return importlib.import_module("lacam")
    except ImportError as error:
        raise ImportError("Build LaCAM with scripts/build_native.sh") from error
