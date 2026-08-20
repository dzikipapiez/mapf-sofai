from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..features import FeatureSet
from ..problem import MapfProblem
from ..solution import MapfSolution
from .base import S1Request, S2Request, SequenceContext


LACAM_LIMIT = 40.0


@dataclass
class NaiveBudgetPolicy:
    """Reserve 40 seconds per future LaCAM run and share the rest equally."""

    charge_feature_time = False
    early_stop_seconds = 10.0
    seed: int = 1
    remaining_seconds: float = 0.0

    @classmethod
    def load(cls, directory: str | Path, seed: int = 1) -> "NaiveBudgetPolicy":
        return cls(seed)

    def start_sequence(self, budget: float, instances: int) -> None:
        self.remaining_seconds = max(0.0, float(budget))

    def observe(self, stage: str, seconds: float) -> None:
        self.remaining_seconds = max(0.0, self.remaining_seconds - seconds)

    def choose_s1(
        self,
        context: SequenceContext,
        problem: MapfProblem,
        features: FeatureSet,
    ) -> S1Request:
        return S1Request(min(LACAM_LIMIT, self.remaining_seconds))

    def choose_s2(
        self,
        context: SequenceContext,
        problem: MapfProblem,
        features: FeatureSet,
        solution: MapfSolution,
    ) -> S2Request | None:
        future = context.instances_left - 1
        limit = (
            (self.remaining_seconds - future * LACAM_LIMIT) / future
            if future
            else self.remaining_seconds
        )
        if limit <= 0:
            return None
        return S2Request(
            limit,
            neighborhood_size=4,
            seed=self.seed,
            early_stop_seconds=self.early_stop_seconds,
        )
