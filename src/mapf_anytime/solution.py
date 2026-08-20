from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
from pathlib import Path
from typing import Iterable

from .problem import BLOCKED, MapfProblem


Location = tuple[int, int]


@dataclass
class MapfSolution:
    paths: list[list[Location]]

    @classmethod
    def from_raw(cls, paths: Iterable[Iterable[Iterable[int]]]) -> "MapfSolution":
        normalized = []
        for path in paths:
            locations = [(int(point[0]), int(point[1])) for point in path]
            while len(locations) > 1 and locations[-1] == locations[-2]:
                locations.pop()
            normalized.append(locations)
        return cls(normalized)

    @classmethod
    def read(cls, path: str | Path) -> "MapfSolution":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return cls.from_raw(json.load(handle)["paths"])

    @classmethod
    def read_lns(cls, path: str | Path, width: int) -> "MapfSolution":
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError(f"Empty LNS path file: {path}")
        expected = int(lines[0])
        paths = [
            [
                (location % width, location // width)
                for location in map(int, filter(None, line.split(",")))
            ]
            for line in lines[1:]
        ]
        if len(paths) != expected:
            raise ValueError(f"{path} contains {len(paths)}/{expected} paths")
        return cls(paths)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            json.dump({"paths": self.paths, "soc": self.soc()}, handle)
        temporary.replace(path)
        return path

    def write_lns_initial(self, path: str | Path, width: int) -> Path:
        path = Path(path)
        with path.open("w", encoding="utf-8") as handle:
            handle.write(f"{len(self.paths)}\n")
            for agent, route in enumerate(self.paths):
                encoded = [str(y * width + x) for x, y in route]
                handle.write(f"{agent} {len(route)} {' '.join(encoded)}\n")
        return path

    def soc(self) -> int:
        return sum(max(0, len(path) - 1) for path in self.paths)

    def makespan(self) -> int:
        return max((len(path) - 1 for path in self.paths), default=0)

    def errors(self, problem: MapfProblem) -> list[str]:
        if len(self.paths) != problem.agents:
            return [f"expected {problem.agents} paths; got {len(self.paths)}"]
        data = problem.map()
        starts, goals = problem.starts_goals()
        errors: list[str] = []
        for agent, path in enumerate(self.paths):
            if not path:
                errors.append(f"agent {agent} has an empty path")
                continue
            if path[0] != starts[agent] or path[-1] != goals[agent]:
                errors.append(f"agent {agent} has the wrong endpoint")
            for time, (x, y) in enumerate(path):
                if not (0 <= x < data.width and 0 <= y < data.height):
                    errors.append(f"agent {agent} leaves the map at t={time}")
                    break
                if data.grid[y][x] in BLOCKED:
                    errors.append(f"agent {agent} hits an obstacle at t={time}")
                    break
            for time, (left, right) in enumerate(zip(path, path[1:]), 1):
                if abs(left[0] - right[0]) + abs(left[1] - right[1]) > 1:
                    errors.append(f"agent {agent} jumps at t={time}")
                    break
        if any(not path for path in self.paths):
            return errors
        for time in range(self.makespan() + 1):
            positions = [_at(path, time) for path in self.paths]
            if len(positions) != len(set(positions)):
                errors.append(f"vertex conflict at t={time}")
                break
            if time:
                edges = [(_at(path, time - 1), _at(path, time)) for path in self.paths]
                if _has_edge_conflict(edges):
                    errors.append(f"edge conflict at t={time}")
                    break
        return errors


def _at(path: list[Location], time: int) -> Location:
    return path[min(time, len(path) - 1)]


def _has_edge_conflict(edges: list[tuple[Location, Location]]) -> bool:
    seen = set()
    for start, end in edges:
        if start != end and (end, start) in seen:
            return True
        seen.add((start, end))
    return False
