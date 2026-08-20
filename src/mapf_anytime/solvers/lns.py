from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import subprocess
import tempfile
import time

from ..problem import MapfProblem
from ..solution import MapfSolution
from .trace import TracePoint


BINARY = Path(__file__).parent / "native" / "mapf_lns" / "build" / "lns"


@dataclass(frozen=True)
class LNSConfig:
    timeout: float
    seed: int = 1
    neighborhood_size: int = 4
    max_iterations: int = 1_000_000
    replan_time_limit: float = 10.0
    init_algorithm: str = "EECBS"
    replan_algorithm: str = "PP"
    destroy_strategy: str = "Adaptive"
    trace: bool = True
    grace_seconds: float = 2.0
    early_stop_seconds: float | None = None


@dataclass
class LNSResult:
    status: str
    wall_seconds: float
    solution: MapfSolution
    trace: list[TracePoint]
    error: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""


def run_lns(
    problem: MapfProblem,
    incumbent: MapfSolution,
    config: LNSConfig,
) -> LNSResult:
    if not math.isfinite(config.replan_time_limit) or config.replan_time_limit <= 0:
        raise ValueError("replan_time_limit must be positive and finite")
    if config.early_stop_seconds is not None:
        if config.early_stop_seconds <= 0:
            raise ValueError("early_stop_seconds must be positive")
    started = time.perf_counter()
    if not BINARY.is_file():
        raise FileNotFoundError("Build MAPF-LNS with scripts/build_native.sh")
    data = problem.map()
    with tempfile.TemporaryDirectory(prefix="mapf-lns-") as temporary:
        directory = Path(temporary)
        initial = incumbent.write_lns_initial(directory / "initial.paths", data.width)
        output = directory / "best.paths"
        stdout, stderr = directory / "stdout.log", directory / "stderr.log"
        command = [
            str(BINARY),
            "-m",
            str(problem.map_path),
            "-a",
            str(problem.scenario_path),
            "-k",
            str(problem.agents),
            "-t",
            str(max(0.001, config.timeout)),
            "--solver",
            "LNS",
            "--seed",
            str(config.seed),
            "--screen",
            "0",
            "--neighborSize",
            str(config.neighborhood_size),
            "--maxIterations",
            str(config.max_iterations),
            "--replanTimeLimit",
            str(min(config.replan_time_limit, config.timeout)),
            "--initAlgo",
            config.init_algorithm,
            "--replanAlgo",
            config.replan_algorithm,
            "--destoryStrategy",
            config.destroy_strategy,
            "--initPaths",
            str(initial),
            "--paths",
            str(output),
        ]
        if config.early_stop_seconds is not None:
            command.extend(
                ["--earlyStopSeconds", str(config.early_stop_seconds)]
            )
        if config.trace:
            command.append("--trace")
        timed_out, returncode = False, None
        try:
            with stdout.open("w") as out, stderr.open("w") as err:
                process = subprocess.run(
                    command,
                    cwd=BINARY.parent.parent,
                    stdout=out,
                    stderr=err,
                    timeout=config.timeout + config.grace_seconds,
                    check=False,
                )
                returncode = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True

        stdout_text, stderr_text = _text(stdout), _text(stderr)
        trace = _trace(stdout_text)
        solution = incumbent
        output_error = None
        if output.exists() and (returncode == 0 or timed_out):
            try:
                # Parse the atomically written incumbent without Python-side
                # solution verification.
                solution = MapfSolution.read_lns(output, data.width)
            except (OSError, ValueError) as error:
                output_error = f"Could not read LNS output: {error}"
        elif returncode == 0:
            output_error = "LNS produced no output paths"
        status = (
            "timeout_with_incumbent"
            if timed_out
            else "success"
            if returncode == 0 and output_error is None
            else "failed_with_incumbent"
        )
        error = output_error or (
            None if returncode in (0, None) else f"LNS exited with {returncode}"
        )
        stdout_tail, stderr_tail = stdout_text[-4000:], stderr_text[-4000:]
    return LNSResult(
        status,
        time.perf_counter() - started,
        solution,
        trace,
        error,
        stdout_tail,
        stderr_tail,
    )


def _trace(text: str) -> list[TracePoint]:
    points = []
    for line in text.splitlines():
        if not line.startswith("LNS_TRACE "):
            continue
        fields = dict(token.split("=", 1) for token in line.split()[1:] if "=" in token)
        try:
            points.append(TracePoint(float(fields["runtime"]), int(fields["soc"])))
        except (KeyError, ValueError):
            continue
    return points


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
