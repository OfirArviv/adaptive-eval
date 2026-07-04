from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np

from sequential.types import (
    AggregationMode,
    DatasetId,
    DatasetSizes,
    DatasetStreams,
    ScoreLike,
)
from sequential.utils import (
    validate_dataset_source,
    validate_emitted_score,
)

@dataclass(frozen=True)
class SamplingPlan:
    """
    Sampling input for one sequential run.

    `dataset_sizes` declares the full benchmark shape. The first batch takes
    `initial_size_per_dataset` from each dataset, then later batches add `batch_size`
    total samples until the full benchmark is exhausted.
    """

    dataset_sizes: DatasetSizes
    initial_size_per_dataset: int
    batch_size: int

    def __post_init__(self) -> None:
        if not self.dataset_sizes:
            raise ValueError("dataset_sizes must not be empty")
        invalid_sizes = sorted(dataset_id for dataset_id, size in self.dataset_sizes.items() if size <= 0)
        if invalid_sizes:
            raise ValueError(f"dataset_sizes must be positive for all datasets: {invalid_sizes}")

        undersized_datasets = sorted(
            dataset_id
            for dataset_id, size in self.dataset_sizes.items()
            if size < self.initial_size_per_dataset
        )
        if undersized_datasets:
            raise ValueError(
                "initial_size_per_dataset must be less than or equal to every dataset size: "
                f"initial_size_per_dataset={self.initial_size_per_dataset}, "
                f"undersized_datasets={undersized_datasets}."
            )
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    @property
    def total_available(self) -> int:
        return sum(self.dataset_sizes.values())

    def cumulative_sample_sizes(self) -> Tuple[int, ...]:
        initial = self.initial_size_per_dataset * len(self.dataset_sizes)
        sizes = [initial]
        while sizes[-1] < self.total_available:
            sizes.append(min(self.total_available, sizes[-1] + self.batch_size))
        return tuple(sizes)


class Sampler(ABC):
    """Allocates scored examples into the planned interim-look batches."""

    @property
    def allocation_weights(self) -> Optional[Mapping[DatasetId, float]]:
        return None

    @cached_property
    @abstractmethod
    def batch_allocations(self) -> tuple[dict[DatasetId, int], ...]:
        pass

    @abstractmethod
    def sample_batches(self) -> Iterator[Tuple[ScoreLike, ...]]:
        pass

class NeymanSampler(Sampler):
    """
    Use a balanced pilot look, then allocate later batches with Neyman-style proportions.

    The first look estimates per-dataset variances from the balanced pilot samples. After that,
    each adaptive batch is split across active datasets by weights
    ``weight_i ∝ N_i * sqrt(variance_i)`` under the equal-cost Neyman allocation
    assumption, with a minimum allocation of 30 samples per active dataset in
    each adaptive batch when capacity allows.

    This yields the closest integer per-batch allocation to the Neyman target
    under capacity constraints (dataset exhaustion). After a dataset reaches its
    declared size, weights are renormalized over the remaining active datasets.
    Exact ties are resolved by lexicographic dataset id order.

    Assumption:
    All datasets have the same per-sample cost. Under unequal per-sample costs,
    Neyman allocation would also include a cost term in the weights.
    """

    MIN_INITIAL_SAMPLES_PER_DATASET = 30

    def __init__(self,
                 datasets_streams: DatasetStreams,
                 sampling_plan: SamplingPlan,
                 aggregation_mode: AggregationMode):
        validate_dataset_source(datasets_streams, sampling_plan.dataset_sizes)

        if sampling_plan.initial_size_per_dataset < NeymanSampler.MIN_INITIAL_SAMPLES_PER_DATASET:
            raise ValueError(
                "NeymanSampler requires initial_size_per_dataset to be at least "
                f"{NeymanSampler.MIN_INITIAL_SAMPLES_PER_DATASET} for a stable pilot variance estimate."
            )

        if sampling_plan.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if sampling_plan.batch_size < len(datasets_streams):
                raise ValueError("batch_size must be larer thqan num of datassets.")

        self.dataset_ids = list(sampling_plan.dataset_sizes)
        self.dataset_sizes = sampling_plan.dataset_sizes
        self.streams = {
            dataset_id: iter(datasets_streams[dataset_id])
            for dataset_id in self.dataset_ids
        }
        self.sampling_plan = sampling_plan
        self.aggregation_mode = aggregation_mode

        self.cached_scores: dict[DatasetId, list[ScoreLike]] = {
            dataset_id: []
            for dataset_id in self.dataset_ids
        }
        self.consumed_counts: dict[DatasetId, int] = {
            dataset_id: 0
            for dataset_id in self.dataset_ids
        }

        super().__init__()

    @cached_property
    def allocation_weights(self) -> Mapping[DatasetId, float]:
        pilot_scores = self._get_pilot()
        return self._estimate_weights(pilot_scores)

    @cached_property
    def batch_allocations(self) -> tuple[dict[DatasetId, int], ...]:
        return self._plan_batch_allocations(self.allocation_weights)

    def sample_batches(self) -> Iterator[tuple[ScoreLike, ...]]:
        allocations = self.batch_allocations
        cached_scores = {dataset_id: list(scores) for dataset_id, scores in self.cached_scores.items()}
        for allocation in allocations:
            batch: list[ScoreLike] = []

            for dataset_id in self.dataset_ids:
                cached = cached_scores[dataset_id]
                missing = allocation[dataset_id] - len(cached)
                for _ in range(max(0, missing)):
                    score = self._next_score(dataset_id)
                    cached.append(score)
                    self.cached_scores[dataset_id].append(score)

                batch.extend(cached[:allocation[dataset_id]])
                del cached[:allocation[dataset_id]]

            yield tuple(batch)
        _raise_if_any_stream_has_extra_sample(self.streams, self.dataset_sizes)

    def _get_pilot(self) -> dict[DatasetId, list[ScoreLike]]:
        pilot_size = self.sampling_plan.initial_size_per_dataset

        pilot_scores: dict[DatasetId, list[ScoreLike]] = {
            dataset_id: self.cached_scores[dataset_id][:pilot_size]
            for dataset_id in self.dataset_ids
        }

        for dataset_id in self.dataset_ids:
            while len(pilot_scores[dataset_id]) < pilot_size:
                score = self._next_score(dataset_id)
                pilot_scores[dataset_id].append(score)
                self.cached_scores[dataset_id].append(score)

        return pilot_scores

    def _estimate_weights(self, pilot_scores: Mapping[DatasetId, Sequence[ScoreLike]],
    ) -> dict[DatasetId, float]:
        weights: dict[DatasetId, float] = {}

        for dataset_id in self.dataset_ids:
            values = [float(score.score) for score in pilot_scores[dataset_id]]
            variance = float(np.var(values, ddof=1))
            std = math.sqrt(max(0.0, variance))

            if self.aggregation_mode == "mean_of_means":
                weights[dataset_id] = std
            elif self.aggregation_mode == "pooled_mean":
                weights[dataset_id] = self.dataset_sizes[dataset_id] * std
            else:
                raise ValueError(f"Unsupported aggregation_mode: {self.aggregation_mode!r}")

        if sum(weights.values()) == 0.0:
            return {dataset_id: 1.0 for dataset_id in self.dataset_ids}

        return weights

    def _plan_batch_allocations(
        self,
        weights: Mapping[DatasetId, float],
    ) -> tuple[dict[DatasetId, int], ...]:
        initial_allocation = {
            dataset_id: self.sampling_plan.initial_size_per_dataset
            for dataset_id in self.dataset_ids
        }

        allocations = [initial_allocation]
        planned_counts = dict(initial_allocation)

        while sum(planned_counts.values()) < sum(self.dataset_sizes.values()):
            remaining = {
                dataset_id: self.dataset_sizes[dataset_id] - planned_counts[dataset_id]
                for dataset_id in self.dataset_ids
            }
            batch_size = min(self.sampling_plan.batch_size, sum(remaining.values()))

            allocation = self._allocate_batch(
                weights=weights,
                remaining=remaining,
                batch_size=batch_size,
            )
            allocations.append(allocation)

            for dataset_id, count in allocation.items():
                planned_counts[dataset_id] += count

        return tuple(allocations)

    def _allocate_batch(
        self,
        *,
        weights: Mapping[DatasetId, float],
        remaining: Mapping[DatasetId, int],
        batch_size: int,
    ) -> dict[DatasetId, int]:
        active = [
            dataset_id
            for dataset_id in self.dataset_ids
            if remaining[dataset_id] > 0
        ]

        allocation = {
            dataset_id: 1 if dataset_id in active else 0
            for dataset_id in self.dataset_ids
        }

        if not active:
            return allocation

        active_weight_sum = sum(weights[dataset_id] for dataset_id in active)
        if active_weight_sum == 0.0:
            ideal = {
                dataset_id: batch_size / len(active)
                for dataset_id in active
            }
        else:
            ideal = {
                dataset_id: batch_size * weights[dataset_id] / active_weight_sum
                for dataset_id in active
            }

        for dataset_id in active:
            allocation[dataset_id] = min(
                int(math.floor(ideal[dataset_id])),
                remaining[dataset_id],
            )

        # Avoid an all-zero allocation when floors round every active target down.
        if sum(allocation.values()) == 0:
            selected = max(active, key=lambda dataset_id: ideal[dataset_id])
            allocation[selected] = 1

        left = batch_size - sum(allocation.values())

        while left > 0:
            candidates = [
                dataset_id
                for dataset_id in active
                if allocation[dataset_id] < remaining[dataset_id]
            ]
            if not candidates:
                break

            selected = max(
                candidates,
                key=lambda dataset_id: (
                    ideal[dataset_id] - allocation[dataset_id],
                    weights[dataset_id],
                ),
            )
            allocation[selected] += 1
            left -= 1

        return allocation


    def _next_score(self, dataset_id: DatasetId) -> ScoreLike:
        if self.consumed_counts[dataset_id] >= self.dataset_sizes[dataset_id]:
            raise ValueError(
                f"Dataset {dataset_id!r} exhausted before reaching its declared size: "
                f"declared={self.dataset_sizes[dataset_id]}."
            )

        try:
            score = next(self.streams[dataset_id])
        except StopIteration:
            raise ValueError(
                f"Dataset {dataset_id!r} exhausted before reaching its declared size: "
                f"emitted={self.consumed_counts[dataset_id]}, "
                f"declared={self.dataset_sizes[dataset_id]}."
            ) from None

        validate_emitted_score(dataset_id, score)
        self.consumed_counts[dataset_id] += 1
        return score


class RoundRobinSampler(Sampler):
    """Use a balanced pilot look, then fill later look batches in round-robin order."""

    def __init__(
        self,
        datasets_streams: DatasetStreams,
        sampling_plan: SamplingPlan,
    ) -> None:
        validate_dataset_source(datasets_streams, sampling_plan.dataset_sizes)
        self.dataset_ids = list(sampling_plan.dataset_sizes)
        self.dataset_sizes = sampling_plan.dataset_sizes
        self.streams = {
            dataset_id: iter(datasets_streams[dataset_id])
            for dataset_id in self.dataset_ids
        }
        self.sampling_plan = sampling_plan
        self.cached_scores: dict[DatasetId, list[ScoreLike]] = {
            dataset_id: []
            for dataset_id in self.dataset_ids
        }
        self.consumed_counts: dict[DatasetId, int] = {
            dataset_id: 0
            for dataset_id in self.dataset_ids
        }
        super().__init__()

    @property
    def allocation_weights(self) -> Mapping[DatasetId, float]:
        return {dataset_id: 1.0 for dataset_id in self.dataset_ids}

    @cached_property
    def batch_allocations(self) -> tuple[dict[DatasetId, int], ...]:
        initial_allocation = {
            dataset_id: self.sampling_plan.initial_size_per_dataset
            for dataset_id in self.dataset_ids
        }
        allocations = [initial_allocation]
        planned_counts = dict(initial_allocation)

        while sum(planned_counts.values()) < self.sampling_plan.total_available:
            batch_size = min(
                self.sampling_plan.batch_size,
                self.sampling_plan.total_available - sum(planned_counts.values()),
            )
            allocation = {dataset_id: 0 for dataset_id in self.dataset_ids}

            while sum(allocation.values()) < batch_size:
                allocated_this_pass = False
                for dataset_id in self.dataset_ids:
                    if sum(allocation.values()) >= batch_size:
                        break
                    if planned_counts[dataset_id] + allocation[dataset_id] >= self.dataset_sizes[dataset_id]:
                        continue
                    allocation[dataset_id] += 1
                    allocated_this_pass = True
                if not allocated_this_pass:
                    break

            allocations.append(allocation)
            for dataset_id, count in allocation.items():
                planned_counts[dataset_id] += count

        return tuple(allocations)

    def sample_batches(self) -> Iterator[Tuple[ScoreLike, ...]]:
        cached_scores = {
            dataset_id: list(scores)
            for dataset_id, scores in self.cached_scores.items()
        }
        for allocation in self.batch_allocations:
            batch: list[ScoreLike] = []
            for dataset_id in self.dataset_ids:
                cached = cached_scores[dataset_id]
                missing = allocation[dataset_id] - len(cached)
                for _ in range(max(0, missing)):
                    score = self._next_score(dataset_id)
                    cached.append(score)
                    self.cached_scores[dataset_id].append(score)

                batch.extend(cached[:allocation[dataset_id]])
                del cached[:allocation[dataset_id]]

            yield tuple(batch)

        _raise_if_any_stream_has_extra_sample(self.streams, self.dataset_sizes)

    def _next_score(self, dataset_id: DatasetId) -> ScoreLike:
        if self.consumed_counts[dataset_id] >= self.dataset_sizes[dataset_id]:
            raise ValueError(
                f"Dataset {dataset_id!r} exhausted before reaching its declared size: "
                f"declared={self.dataset_sizes[dataset_id]}."
            )

        try:
            score = next(self.streams[dataset_id])
        except StopIteration:
            raise ValueError(
                f"Dataset {dataset_id!r} exhausted before reaching its declared size: "
                f"emitted={self.consumed_counts[dataset_id]}, "
                f"declared={self.dataset_sizes[dataset_id]}."
            ) from None

        validate_emitted_score(dataset_id, score)
        self.consumed_counts[dataset_id] += 1
        return score



def _raise_if_any_stream_has_extra_sample(
    iterators: Mapping[DatasetId, Iterator[ScoreLike]],
    dataset_sizes: DatasetSizes,
) -> None:
    for dataset_id, iterator in iterators.items():
        try:
            extra_score = next(iterator)
        except StopIteration:
            continue
        validate_emitted_score(dataset_id, extra_score)
        raise ValueError(
            f"Dataset {dataset_id!r} produced more samples than declared in dataset_sizes: "
            f"declared={dataset_sizes[dataset_id]}."
        )
