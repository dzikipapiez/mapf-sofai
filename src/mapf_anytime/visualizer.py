from __future__ import annotations

from pathlib import Path

from .problem import PASSABLE, MapfProblem
from .solution import MapfSolution


__all__ = ["visualize"]


def visualize(
    problem: MapfProblem,
    solution: MapfSolution,
    output: str | Path = "solution.svg",
) -> Path:
    """Replay a MAPF solution in POGEMA and save its SVG animation."""
    try:
        from pogema import AnimationConfig, AnimationMonitor, GridConfig, pogema_v0
    except ImportError as error:
        raise ImportError("Install mapf-anytime[pogema] to visualize solutions") from error

    grid = problem.map()
    starts, goals = problem.starts_goals()
    config = GridConfig(
        num_agents=problem.agents,
        size=max(grid.width, grid.height),
        map=[[0 if cell in PASSABLE else 1 for cell in row] for row in grid.grid],
        agents_xy=[[y, x] for x, y in starts],
        targets_xy=[[y, x] for x, y in goals],
        observation_type="MAPF",
        on_target="nothing",
        collision_system="soft",
        max_episode_steps=solution.makespan() + 1,
        obs_radius=2,
    )
    actions = {tuple(move): index for index, move in enumerate(config.MOVES)}
    environment = AnimationMonitor(
        pogema_v0(grid_config=config),
        AnimationConfig(save_every_idx_episode=None),
    )
    environment.reset()

    for time in range(solution.makespan()):
        step = []
        for path in solution.paths:
            current = path[min(time, len(path) - 1)]
            following = path[min(time + 1, len(path) - 1)]
            step.append(
                actions[
                    (
                        following[1] - current[1],
                        following[0] - current[0],
                    )
                ]
            )
        environment.step(step)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    environment.save_animation(str(output))
    environment.close()
    return output
