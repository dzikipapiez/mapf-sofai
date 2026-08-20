from __future__ import annotations

import time
from typing import Iterable

from .features import LACAM_FEATURES, analyze
from .policies.base import Policy, SequenceContext
from .problem import MapfProblem
from .solvers import LaCAMConfig, LNSConfig, run_lacam, run_lns
from .solvers.lacam import LaCAMResult


def run_sequence(
    policy: Policy,
    problems: Iterable[MapfProblem],
    policy_budget: float,
    hard_limit: float | None = None,
) -> list[dict]:
    """Run one ordered sequence, reporting measured work back to its policy."""
    ordered = list(problems)
    rows = []
    hard_remaining = float("inf") if hard_limit is None else float(hard_limit)
    policy.start_sequence(policy_budget, len(ordered))
    sequence_started = time.perf_counter()
    for position, problem in enumerate(ordered):
        row, hard_remaining = _run_instance(
            policy,
            problem,
            position,
            len(ordered) - position,
            hard_remaining,
            sequence_started,
        )
        rows.append(row)
    return rows


def _run_instance(
    policy: Policy,
    problem: MapfProblem,
    position: int,
    instances_left: int,
    hard_remaining: float,
    sequence_started: float,
) -> tuple[dict, float]:
    instance_started = time.perf_counter()
    before = policy.remaining_seconds

    started = time.perf_counter()
    features = analyze(problem)
    static_feature_seconds = time.perf_counter() - started
    if policy.charge_feature_time:
        hard_remaining = max(0.0, hard_remaining - static_feature_seconds)
        policy.observe("static_features", static_feature_seconds)

    context = SequenceContext(instances_left, position)
    started = time.perf_counter()
    s1_request = policy.choose_s1(context, problem, features)
    s1_decision_seconds = time.perf_counter() - started
    hard_remaining = max(0.0, hard_remaining - s1_decision_seconds)
    policy.observe("s1_decision", s1_decision_seconds)
    s1_limit = min(max(0.0, s1_request.timeout), hard_remaining)

    if s1_limit > 0:
        s1 = run_lacam(problem, LaCAMConfig(s1_limit, anytime=s1_request.anytime))
        hard_remaining = max(0.0, hard_remaining - s1.wall_seconds)
        policy.observe("s1", s1.wall_seconds)
    else:
        s1 = LaCAMResult("skipped", 0.0)

    lacam_feature_seconds = 0.0
    if s1.solution is not None:
        started = time.perf_counter()
        features = features.with_lacam(s1.solution)
        lacam_feature_seconds = time.perf_counter() - started
        if policy.charge_feature_time:
            hard_remaining = max(0.0, hard_remaining - lacam_feature_seconds)
            policy.observe("lacam_features", lacam_feature_seconds)

    s2 = None
    s2_request = None
    s2_limit = 0.0
    s2_decision_seconds = 0.0
    if s1.solution is not None:
        context = SequenceContext(instances_left, position)
        started = time.perf_counter()
        s2_request = policy.choose_s2(context, problem, features, s1.solution)
        s2_decision_seconds = time.perf_counter() - started
        hard_remaining = max(0.0, hard_remaining - s2_decision_seconds)
        policy.observe("s2_decision", s2_decision_seconds)
        if s2_request is not None:
            s2_limit = min(max(0.0, s2_request.timeout), hard_remaining)
            if s2_limit > 0:
                s2 = run_lns(
                    problem,
                    s1.solution,
                    LNSConfig(
                        timeout=s2_limit,
                        seed=s2_request.seed,
                        neighborhood_size=s2_request.neighborhood_size,
                        max_iterations=s2_request.max_iterations,
                        early_stop_seconds=s2_request.early_stop_seconds,
                        replan_time_limit=min(
                            s2_request.replan_time_limit, s2_limit
                        ),
                    ),
                )
                hard_remaining = max(0.0, hard_remaining - s2.wall_seconds)
                policy.observe("s2", s2.wall_seconds)

    metrics_started = time.perf_counter()
    final = s2.solution if s2 else s1.solution
    lower_bound = features.static["lower_bound_soc"]
    agents = problem.agents
    initial_soc = s1.solution.soc() if s1.solution else None
    final_soc = final.soc() if final else None
    initial_sod = max(0.0, initial_soc - lower_bound) if initial_soc is not None else None
    final_sod = max(0.0, final_soc - lower_bound) if final_soc is not None else None
    improvement = (
        max(0.0, initial_sod - final_sod)
        if initial_sod is not None and final_sod is not None
        else None
    )
    fraction_improvement = None
    if improvement is not None:
        fraction_improvement = improvement / initial_sod if initial_sod else 0.0
    row = {
        **features.static,
        **{name: features.lacam.get(name) for name in LACAM_FEATURES},
        "position": position,
        "instance_id": problem.name,
        "instances_left": instances_left,
        "budget_before": before,
        "feature_time_charged": policy.charge_feature_time,
        "static_feature_seconds": static_feature_seconds,
        "lacam_feature_seconds": lacam_feature_seconds,
        "s1_decision_seconds": s1_decision_seconds,
        "s1_requested_seconds": s1_request.timeout,
        "s1_allocated_seconds": s1_limit,
        "s1_wall_seconds": s1.wall_seconds,
        "s1_status": s1.status,
        "s2_decision_seconds": s2_decision_seconds,
        "s2_requested_seconds": s2_request.timeout if s2_request else 0.0,
        "s2_allocated_seconds": s2_limit,
        "s2_neighborhood_size": (
            s2_request.neighborhood_size if s2_request else None
        ),
        "s2_early_stop_seconds": (
            s2_request.early_stop_seconds if s2_request else None
        ),
        "s2_replan_time_limit": (
            min(s2_request.replan_time_limit, s2_limit) if s2_request else None
        ),
        "s2_wall_seconds": s2.wall_seconds if s2 else 0.0,
        "s2_status": s2.status if s2 else "skipped",
        "solved": final is not None,
        "initial_soc": initial_soc,
        "final_soc": final_soc,
        "initial_sod": initial_sod,
        "final_sod": final_sod,
        "initial_sodn": initial_sod / agents if initial_sod is not None else None,
        "final_sodn": final_sod / agents if final_sod is not None else None,
        "sodn_improvement": improvement / agents if improvement is not None else None,
        "initial_sodlb": (
            initial_sod / lower_bound
            if initial_sod is not None and lower_bound
            else None
        ),
        "final_sodlb": (
            final_sod / lower_bound
            if final_sod is not None and lower_bound
            else None
        ),
        "sodlb_improvement": (
            improvement / lower_bound
            if improvement is not None and lower_bound
            else None
        ),
        "sod_fraction_improvement": fraction_improvement,
        "lns_trace": (
            [{"seconds": point.seconds, "soc": point.soc} for point in s2.trace]
            if s2
            else []
        ),
        "error": s2.error if s2 and s2.error else s1.error,
        "s1_info": s1_request.info,
        "s2_info": s2_request.info if s2_request else {},
    }
    metrics_seconds = time.perf_counter() - metrics_started
    hard_remaining = max(0.0, hard_remaining - metrics_seconds)
    policy.observe("metrics", metrics_seconds)
    row.update(
        {
            "metrics_seconds": metrics_seconds,
            "budget_after": policy.remaining_seconds,
            "instance_wall_seconds": time.perf_counter() - instance_started,
            "sequence_elapsed_seconds": time.perf_counter() - sequence_started,
        }
    )
    return row, hard_remaining
