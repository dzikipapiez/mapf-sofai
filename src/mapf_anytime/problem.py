from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


PASSABLE = frozenset(".GSW")
BLOCKED = frozenset("@OT")


@dataclass(frozen=True)
class MapData:
    width: int
    height: int
    grid: tuple[str, ...]


@dataclass(frozen=True)
class MapfProblem:
    name: str
    map_path: Path
    scenario_path: Path
    agents: int

    @classmethod
    def from_manifest(cls, path: str | Path) -> "MapfProblem":
        manifest_path = Path(path).expanduser().resolve()
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        base = manifest_path.parent

        def resolve(value: str) -> Path:
            candidate = Path(value).expanduser()
            return (
                candidate if candidate.is_absolute() else base / candidate
            ).resolve()

        agents = int(data.get("agents", data.get("num_agents", 0)))
        if agents <= 0:
            raise ValueError(f"{manifest_path} has no positive agent count")
        return cls(
            name=str(data.get("name", manifest_path.stem)),
            map_path=resolve(data["map"]),
            scenario_path=resolve(data["scenario"]),
            agents=agents,
        )

    def map(self) -> MapData:
        lines = self.map_path.read_text(encoding="utf-8").splitlines()
        header: dict[str, str] = {}
        start = None
        for index, line in enumerate(lines):
            if line.strip().lower() == "map":
                start = index + 1
                break
            key, separator, value = line.partition(" ")
            if separator:
                header[key.lower()] = value.strip()
        if start is None:
            raise ValueError(f"{self.map_path} has no map section")
        width, height = int(header["width"]), int(header["height"])
        grid = tuple(lines[start : start + height])
        if len(grid) != height or any(len(row) != width for row in grid):
            raise ValueError(f"{self.map_path} dimensions do not match its header")
        return MapData(width, height, grid)

    def starts_goals(
        self,
    ) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
        starts, goals = [], []
        with self.scenario_path.open(encoding="utf-8") as handle:
            if not handle.readline().strip().startswith("version"):
                raise ValueError(f"{self.scenario_path} has no version header")
            for line in handle:
                fields = line.split()
                if not fields:
                    continue
                if len(fields) < 9:
                    raise ValueError(f"Malformed scenario row: {line.rstrip()}")
                starts.append((int(fields[4]), int(fields[5])))
                goals.append((int(fields[6]), int(fields[7])))
                if len(starts) == self.agents:
                    break
        if len(starts) != self.agents:
            raise ValueError(
                f"{self.scenario_path} supplies {len(starts)}/{self.agents} agents"
            )
        return tuple(starts), tuple(goals)
