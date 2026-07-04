from __future__ import annotations

from typing import List, Sequence

from sequential.engines import SequentialEngine
from sequential.rules import StoppingRule
from sequential.samplers import Sampler, SamplingPlan
from sequential.types import DatasetStreams, LookResult, SequentialOutcome, StopReason

def run_sequential_test(
    data_map: DatasetStreams,
    sampler: Sampler,
    sampling_plan: SamplingPlan,
    engine: SequentialEngine,
    rules: Sequence[StoppingRule],
) -> SequentialOutcome:
    """
    Execute the sequential testing pipeline over grouped dataset streams.

    This function is the orchestration layer for the sequential workflow. It asks the
    sampler to allocate samples into planned look batches, runs the sequential engine
    over the cumulative interim looks, and evaluates stopping rules after each look.

    Args:
        data_map:
            Per-dataset score streams. Every emitted score must have a `dataset_id`
            matching the dataset bucket it came from.
        sampler:
            Sampling policy that decides how per-dataset streams are allocated to look batches.
        sampling_plan:
            Declared dataset sizes and look cadence for this run.
        engine:
            Sequential engine that computes estimates, confidence intervals, and any
            optional boundary or conditional-power outputs.
        rules:
            Stopping rules evaluated after each interim look.
        Futility is binding: when a configured native futility rule fires, the run stops.

    Returns:
        A `SequentialOutcome` containing the stop reason, stop look, samples used,
        and the full look history observed before stopping.
    """

    looks: List[LookResult] = []
    total_available = sampling_plan.total_available
    total_planned_looks = len(sampling_plan.cumulative_sample_sizes())
    active_rules: List[StoppingRule] = []
    for rule in rules:
        if rule.requires.issubset(engine.capabilities):
            active_rules.append(rule)
            continue
        missing = rule.requires - engine.capabilities
        raise ValueError(
            f"StoppingRule for reason {rule.reason.value!r} requires unsupported capabilities: "
            f"{sorted(missing)}. Engine/runner supports: {sorted(engine.capabilities)}."
        )

    for look in engine.iter_looks(sampler=sampler):
        looks.append(look)
        look_history = tuple(looks)
        for rule in active_rules:
            decision = rule.check(look_history)
            if decision:
                return SequentialOutcome(
                    stop_reason=decision,
                    stop_look=look.look_index,
                    samples_used=look.n_seen,
                    total_available=total_available,
                    total_planned_looks=total_planned_looks,
                    looks=tuple(looks),
                )

    if not looks:
        raise ValueError("Sequential engine produced no looks.")

    last = looks[-1]
    return SequentialOutcome(
        stop_reason=StopReason.MAX_SAMPLES,
        stop_look=last.look_index,
        samples_used=last.n_seen,
        total_available=total_available,
        total_planned_looks=total_planned_looks,
        looks=tuple(looks),
    )
