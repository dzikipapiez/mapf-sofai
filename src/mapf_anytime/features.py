from __future__ import annotations

from dataclasses import dataclass, field, replace
import importlib
import math
from pathlib import Path
import sys

from .problem import MapfProblem
from .solution import MapfSolution


STATIC_FEATURES = (
    "num_agents",
    "num_obstacles",
    "agent_density",
    "obstacle_density",
    "avg_shortest_path_distance",
    "min_shortest_path_distance",
    "max_shortest_path_distance",
    "cells_at_sp_ratio",
    "num_total_cells",
    "num_free_cells",
    "lower_bound_soc",
    "sp_collision_count",
    "sp_vertex_collision_count",
    "sp_edge_collision_count",
)
LACAM_FEATURES = (
    "lacam_initial_sod",
    "lacam_initial_sod_over_lb",
    "lacam_fraction_delayed_agents",
    "lacam_avg_delay",
    "lacam_delay_90th_percentile",
)
NATIVE = Path(__file__).parent / "solvers" / "native" / "shortest_paths"


@dataclass(frozen=True)
class FeatureSet:
    static: dict[str, float]
    distances: tuple[int, ...]
    lacam: dict[str, float] = field(default_factory=dict)

    def with_lacam(self, solution: MapfSolution) -> "FeatureSet":
        delays = [
            max(0, len(path) - 1 - distance)
            for path, distance in zip(solution.paths, self.distances)
        ]
        soc = solution.soc()
        initial_sod = float(max(0, soc - sum(self.distances)))
        lower_bound_soc = float(self.static["lower_bound_soc"])
        return replace(
            self,
            lacam={
                "lacam_initial_soc": float(soc),
                "lacam_initial_sod": initial_sod,
                "lacam_initial_sod_over_lb": (
                    initial_sod / lower_bound_soc if lower_bound_soc else 0.0
                ),
                "lacam_fraction_delayed_agents": (
                    sum(delay > 0 for delay in delays) / len(delays) if delays else 0.0
                ),
                "lacam_avg_delay": sum(delays) / len(delays) if delays else 0.0,
                "lacam_delay_90th_percentile": _percentile(delays, 90),
            },
        )


def analyze(
    problem: MapfProblem, cells_mode: str = "single-path", threads: int = 1
) -> FeatureSet:
    binding = NATIVE / f"build-pybind-{sys.version_info.major}{sys.version_info.minor}"
    if str(binding) not in sys.path:
        sys.path.insert(0, str(binding))
    try:
        module = importlib.import_module("mapf_shortest_paths")
    except ImportError as error:
        raise ImportError(
            "Build the native feature module with scripts/build_native.sh"
        ) from error

    values = dict(
        module.analyze_instance(
            str(problem.map_path),
            str(problem.scenario_path),
            problem.agents,
            cells_mode,
            int(threads),
        )
    )
    distances = tuple(map(int, values.pop("shortest_path_distances")))
    missing = set(STATIC_FEATURES) - values.keys()
    if missing:
        raise RuntimeError(f"Native feature result is missing {sorted(missing)}")
    return FeatureSet(
        {name: values[name] for name in STATIC_FEATURES},
        distances,
    )


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
