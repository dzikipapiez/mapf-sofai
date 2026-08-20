from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..features import FeatureSet
from ..problem import MapfProblem
from ..solution import MapfSolution


@dataclass(frozen=True)
class SequenceContext:
    instances_left: int
    position: int


@dataclass(frozen=True)
class S1Request:
    timeout: float
    anytime: bool = False
    info: dict = field(default_factory=dict)


@dataclass(frozen=True)
class S2Request:
    timeout: float
    neighborhood_size: int = 4
    seed: int = 1
    max_iterations: int = 1_000_000
    info: dict = field(default_factory=dict)
    early_stop_seconds: float | None = 10.0
    replan_time_limit: float = 10.0


class Policy(Protocol):
    charge_feature_time: bool
    remaining_seconds: float

    def start_sequence(self, budget: float, instances: int) -> None: ...

    def observe(self, stage: str, seconds: float) -> None: ...

    def choose_s1(
        self,
        context: SequenceContext,
        problem: MapfProblem,
        features: FeatureSet,
    ) -> S1Request: ...

    def choose_s2(
        self,
        context: SequenceContext,
        problem: MapfProblem,
        features: FeatureSet,
        solution: MapfSolution,
    ) -> S2Request | None: ...
