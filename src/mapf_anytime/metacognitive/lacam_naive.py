from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..features import FeatureSet
from ..problem import MapfProblem
from ..solution import MapfSolution
from .base import S1Request, S2Request, SequenceContext


@dataclass
class LaCAMNaiveMetacognitiveModule:
    charge_feature_time = False
    remaining_seconds: float = 0.0

    @classmethod
    def load(cls, directory: Path, seed: int = 1) -> LaCAMNaiveMetacognitiveModule:
        del directory, seed
        return cls()

    def start_sequence(self, budget: float, instances: int) -> None:
        del instances
        self.remaining_seconds = max(0.0, float(budget))

    def observe(self, stage: str, seconds: float) -> None:
        del stage
        self.remaining_seconds = max(0.0, self.remaining_seconds - seconds)

    def choose_s1(
        self,
        context: SequenceContext,
        problem: MapfProblem,
        features: FeatureSet,
    ) -> S1Request:
        del context, problem, features
        return S1Request(min(40.0, self.remaining_seconds))

    def choose_s2(
        self,
        context: SequenceContext,
        problem: MapfProblem,
        features: FeatureSet,
        solution: MapfSolution,
    ) -> S2Request | None:
        del context, problem, features, solution
        return None
