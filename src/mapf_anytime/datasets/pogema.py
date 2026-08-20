from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import numpy as np


def prepare_instances(
    output: Path,
    instance_list: Path,
    count: int = 5000,
    seed: int = 0,
    map_size: tuple[int, int] = (8, 500),
    obstacle_density: tuple[float, float] = (0.0, 0.4),
    agent_density: tuple[float, float] = (0.0, 0.3),
    minimum_component_fraction: float = 0.9,
    attempts: int = 20,
    time_limit: int = 60,
) -> list[Path]:
    """Generate the current broad, component-filtered POGEMA instance pool."""
    rng = random.Random(seed)
    manifests: list[Path] = []
    sampled = 0
    while len(manifests) < count:
        sampled += 1
        if sampled > count * attempts:
            raise RuntimeError(
                f"Generated only {len(manifests)}/{count} POGEMA instances"
            )
        size = rng.randint(*map_size)
        requested_obstacles = rng.uniform(*obstacle_density)
        free_estimate = max(1, round(size * size * (1.0 - requested_obstacles)))
        requested_agent_density = rng.uniform(*agent_density)
        requested_agents = max(1, round(free_estimate * requested_agent_density))
        realized = _realize(
            size,
            requested_obstacles,
            requested_agents,
            rng.randrange(1, 2**31 - 1),
            minimum_component_fraction,
            attempts,
        )
        if realized is None:
            continue
        obstacles, starts, goals, realized_seed, component_count, fractions = realized
        manifest = _write_instance(
            output,
            len(manifests),
            obstacles,
            starts,
            goals,
            {
                "generation_run_id": f"pogema-{seed}",
                "generation_index": len(manifests),
                "requested_map_size": size,
                "requested_obstacle_density": requested_obstacles,
                "requested_agent_density": requested_agent_density,
                "requested_num_agents": requested_agents,
                "realized_map_width": int(obstacles.shape[1]),
                "realized_map_height": int(obstacles.shape[0]),
                "realized_obstacle_density": float(obstacles.mean()),
                "realized_free_component_count": component_count,
                "dominant_agent_component_fraction": fractions[0],
                "reachable_agent_pair_fraction": fractions[1],
                "pre_filter_num_agents": requested_agents,
                "component_filter_dropped_agents": requested_agents - len(starts),
                "realized_seed": realized_seed,
                "time_limit_sec": time_limit,
            },
        )
        manifests.append(manifest)
    instance_list.parent.mkdir(parents=True, exist_ok=True)
    instance_list.write_text(
        "".join(f"{path.resolve()}\n" for path in manifests), encoding="utf-8"
    )
    return manifests


def _realize(
    size: int,
    density: float,
    agents: int,
    seed: int,
    minimum_fraction: float,
    attempts: int,
):
    from pogema import GridConfig, pogema_v0

    for attempt in range(attempts):
        realized_seed = seed + attempt * 1_000_003
        try:
            env = pogema_v0(
                grid_config=GridConfig(
                    num_agents=agents,
                    width=size,
                    height=size,
                    density=density,
                    seed=realized_seed,
                    obs_radius=2,
                    max_episode_steps=max(64, size * 4),
                    observation_type="MAPF",
                )
            )
            env.reset()
            obstacles = np.asarray(
                env.grid.get_obstacles(ignore_borders=True), dtype=np.int8
            )
            starts = [
                tuple(map(int, point))
                for point in env.grid.get_agents_xy(ignore_borders=True)
            ]
            goals = [
                tuple(map(int, point))
                for point in env.grid.get_targets_xy(ignore_borders=True)
            ]
            labels, sizes = _components(obstacles)
            largest = int(np.argmax(sizes))
            reachable = [
                labels[start] >= 0 and labels[start] == labels[goal]
                for start, goal in zip(starts, goals)
            ]
            keep = [
                labels[start] == largest and labels[goal] == largest
                for start, goal in zip(starts, goals)
            ]
            fraction = sum(keep) / len(keep)
            if fraction < minimum_fraction:
                continue
            kept_starts = [point for point, use in zip(starts, keep) if use]
            kept_goals = [point for point, use in zip(goals, keep) if use]
            cropped, kept_starts, kept_goals = _crop(obstacles, kept_starts, kept_goals)
            return (
                cropped,
                kept_starts,
                kept_goals,
                realized_seed,
                len(sizes),
                (fraction, sum(reachable) / len(reachable)),
            )
        except Exception:
            density *= 0.9
            agents = max(1, round(agents * 0.9))
    return None


def _components(obstacles: np.ndarray) -> tuple[np.ndarray, list[int]]:
    labels = np.full(obstacles.shape, -1, dtype=np.int32)
    sizes: list[int] = []
    height, width = obstacles.shape
    for row in range(height):
        for col in range(width):
            if obstacles[row, col] or labels[row, col] >= 0:
                continue
            label, stack, count = len(sizes), [(row, col)], 0
            labels[row, col] = label
            while stack:
                y, x = stack.pop()
                count += 1
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and not obstacles[ny, nx]
                        and labels[ny, nx] < 0
                    ):
                        labels[ny, nx] = label
                        stack.append((ny, nx))
            sizes.append(count)
    return labels, sizes


def _crop(
    obstacles: np.ndarray,
    starts: list[tuple[int, int]],
    goals: list[tuple[int, int]],
):
    free = obstacles == 0
    rows, cols = np.where(free)
    top, bottom, left, right = rows.min(), rows.max(), cols.min(), cols.max()

    def shift(points):
        return [(row - top, col - left) for row, col in points]

    return obstacles[top : bottom + 1, left : right + 1], shift(starts), shift(goals)


def _write_instance(
    output: Path,
    index: int,
    obstacles: np.ndarray,
    starts: list[tuple[int, int]],
    goals: list[tuple[int, int]],
    metadata: dict[str, Any],
) -> Path:
    seed = metadata["realized_seed"]
    name = f"pogema-{index:06d}-s{obstacles.shape[0]}-a{len(starts)}-seed{seed}"
    map_path, scenario_path = (
        output / "maps" / f"{name}.map",
        output / "scen" / f"{name}.scen",
    )
    manifest_path = output / "instances" / f"{name}.json"
    for directory in (map_path.parent, scenario_path.parent, manifest_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    height, width = obstacles.shape
    map_path.write_text(
        f"type octile\nheight {height}\nwidth {width}\nmap\n"
        + "\n".join(
            "".join("@" if obstacles[row, col] else "." for col in range(width))
            for row in range(height)
        )
        + "\n",
        encoding="utf-8",
    )
    scenario_lines = ["version 1"]
    for agent, (start, goal) in enumerate(zip(starts, goals)):
        distance = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        scenario_lines.append(
            f"{agent}\t{map_path.name}\t{width}\t{height}\t"
            f"{start[1]}\t{start[0]}\t{goal[1]}\t{goal[0]}\t{float(distance):.8f}"
        )
    scenario_path.write_text("\n".join(scenario_lines) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "name": name,
                "map": f"../maps/{map_path.name}",
                "scenario": f"../scen/{scenario_path.name}",
                "agents": len(starts),
                "source_map": f"../maps/{map_path.name}",
                **metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path.resolve()
