from __future__ import annotations

from collections import deque
import hashlib
import json
import os
from pathlib import Path
import random

from ..problem import PASSABLE


DENSITY_BUCKETS = (
    (1.0, 6.0),
    (6.0, 11.0),
    (12.0, 17.0),
    (18.0, 23.0),
    (23.0, 28.0),
    (28.0, 33.0),
)


def prepare_instances(
    source: Path,
    output: Path,
    instance_list: Path,
    scenario_kind: str = "random",
    seed: int = 0,
    time_limit: int = 60,
) -> list[Path]:
    """Build the existing six-density-bucket MovingAI instance set."""
    maps = sorted((source / "maps").glob("*.map"))
    if not maps:
        raise FileNotFoundError(f"No MovingAI maps found under {source / 'maps'}")

    manifests: list[Path] = []
    for source_map in maps:
        map_data = _largest_component(source_map)
        map_path = output / "maps" / f"{source_map.stem}-processed.map"
        _write_map(map_path, map_data)
        scenarios = sorted(
            (source / "scen").glob(f"{source_map.stem}-{scenario_kind}-*.scen"),
            key=lambda path: _scenario_number(path, scenario_kind),
        )
        if len(scenarios) != 25:
            raise ValueError(
                f"Expected 25 {scenario_kind} scenarios for {source_map.stem}; "
                f"found {len(scenarios)}"
            )

        for scenario in scenarios:
            number = _scenario_number(scenario, scenario_kind)
            rng = random.Random(_seed(seed, source_map.stem, scenario_kind, number))
            samples = [
                (
                    max(1, round(len(map_data["free"]) * rng.uniform(low, high) / 100)),
                    (low, high),
                )
                for low, high in DENSITY_BUCKETS
            ]
            scenario_path = (
                output
                / "scenarios"
                / f"{source_map.stem}-{scenario_kind}-{number}-density.scen"
            )
            _write_scenario(
                scenario_path,
                map_path,
                map_data,
                max(count for count, _ in samples),
                rng,
            )
            for bucket, (agents, bounds) in enumerate(samples, 1):
                name = (
                    f"{source_map.stem}-{scenario_kind}-{number}"
                    f"-density-b{bucket}-n{agents}"
                )
                manifest = output / "instances" / f"{name}.json"
                _write_json(
                    manifest,
                    {
                        "name": name,
                        "map": os.path.relpath(map_path, manifest.parent),
                        "scenario": os.path.relpath(scenario_path, manifest.parent),
                        "agents": agents,
                        "scenario_kind": scenario_kind,
                        "agent_density_bucket": list(bounds),
                        "free_cells": len(map_data["free"]),
                        "source_map": str(source_map.resolve()),
                        "time_limit_sec": time_limit,
                    },
                )
                manifests.append(manifest.resolve())

    _write_list(instance_list, manifests)
    return manifests


def _largest_component(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = {
        line.partition(" ")[0].lower(): line.partition(" ")[2]
        for line in lines
        if " " in line and line.lower() != "map"
    }
    start = next(index for index, line in enumerate(lines) if line.lower() == "map") + 1
    width, height = int(header["width"]), int(header["height"])
    grid = lines[start : start + height]
    seen: set[tuple[int, int]] = set()
    largest: list[tuple[int, int]] = []
    for y in range(height):
        for x in range(width):
            if (x, y) in seen or grid[y][x] not in PASSABLE:
                continue
            component, queue = [], deque([(x, y)])
            seen.add((x, y))
            while queue:
                point = queue.popleft()
                component.append(point)
                px, py = point
                for neighbor in (
                    (px - 1, py),
                    (px + 1, py),
                    (px, py - 1),
                    (px, py + 1),
                ):
                    nx, ny = neighbor
                    if (
                        0 <= nx < width
                        and 0 <= ny < height
                        and neighbor not in seen
                        and grid[ny][nx] in PASSABLE
                    ):
                        seen.add(neighbor)
                        queue.append(neighbor)
            if len(component) > len(largest):
                largest = component
    if not largest:
        raise ValueError(f"{path} has no free component")

    min_x, max_x = min(x for x, _ in largest), max(x for x, _ in largest)
    min_y, max_y = min(y for _, y in largest), max(y for _, y in largest)
    cells = {(x - min_x, y - min_y) for x, y in largest}
    return {
        "width": max_x - min_x + 1,
        "height": max_y - min_y + 1,
        "free": sorted(cells, key=lambda point: (point[1], point[0])),
    }


def _write_map(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    free = set(data["free"])
    rows = [
        "".join("." if (x, y) in free else "@" for x in range(data["width"]))
        for y in range(data["height"])
    ]
    path.write_text(
        "type octile\n"
        f"height {data['height']}\nwidth {data['width']}\nmap\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )


def _write_scenario(
    path: Path, map_path: Path, data: dict, agents: int, rng: random.Random
) -> None:
    starts = rng.sample(data["free"], agents)
    goals = rng.sample(data["free"], agents)
    while any(start == goal for start, goal in zip(starts, goals)):
        goals = rng.sample(data["free"], agents)
    lines = ["version 1"]
    for start, goal in zip(starts, goals):
        distance = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        lines.append(
            f"0\t{map_path.name}\t{data['width']}\t{data['height']}\t"
            f"{start[0]}\t{start[1]}\t{goal[0]}\t{goal[1]}\t{distance}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scenario_number(path: Path, kind: str) -> int:
    return int(path.stem.rsplit(f"-{kind}-", 1)[1])


def _seed(seed: int, map_name: str, kind: str, number: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{map_name}:{kind}:{number}".encode()).digest()[:8],
        "big",
    )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _write_list(path: Path, manifests: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{item}\n" for item in manifests), encoding="utf-8")
