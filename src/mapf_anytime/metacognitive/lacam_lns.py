from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..features import FeatureSet
from ..problem import MapfProblem
from ..solution import MapfSolution
from .base import S1Request, S2Request, SequenceContext


@dataclass
class LaCAMLNSMetacognitiveModule:
    charge_feature_time = False
    early_stop_seconds: float | None = 10.0
    seed: int = 1
    remaining_seconds: float = 0.0

    @classmethod
    def load(
        cls,
        directory: Path,
        seed: int = 1,
        early_stop_seconds: float | None = 10.0,
    ) -> LaCAMLNSMetacognitiveModule:
        del directory
        if early_stop_seconds is not None and early_stop_seconds <= 0:
            raise ValueError("early_stop_seconds must be positive or null")
        return cls(early_stop_seconds, seed)

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
        timeout = min(100.0, self.remaining_seconds)
        if timeout <= 0:
            return None
        return S2Request(
            timeout,
            neighborhood_size=4,
            seed=self.seed,
            early_stop_seconds=self.early_stop_seconds,
        )
