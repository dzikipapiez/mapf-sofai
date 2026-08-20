from dataclasses import dataclass


@dataclass(frozen=True)
class TracePoint:
    seconds: float
    soc: int
