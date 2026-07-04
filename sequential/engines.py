from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterator, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm

from sequential.aggregators import AggregatedEstimate, ScoreAggregator
from sequential.samplers import Sampler
from sequential.types import (
    AggregationMode,
    DatasetId,
    DatasetSizes,
    EngineCapability,
    FutilityConfig,
    LookResult,
    ScoreLike,
    SequentialTestType,
    SpendingFunction,
)
from sequential.utils import full_benchmark_standard_error_from_max_information

try:
    import rpy2.robjects as ro
    from rpy2.robjects import NULL
    from rpy2.robjects.packages import importr
except Exception:  # pragma: no cover
    ro = None
    NULL = None
    importr = None


TimingMode = Literal["observed_information", "planned_information"]


class SequentialEngine(ABC):
    capabilities: FrozenSet[EngineCapability]
    supported_aggregations: FrozenSet[AggregationMode]

    def __init__(
        self,
        aggregator: ScoreAggregator,
        dataset_weights: Optional[Dict[DatasetId, float]] = None,
    ):
        self.aggregator = aggregator
        if self.aggregator.aggregation_mode not in self.supported_aggregations:
            raise ValueError(
                f"{type(self).__name__} does not support aggregator "
                f"{self.aggregator.aggregation_mode!r}. Supported aggregations: "
                f"{sorted(self.supported_aggregations)!r}."
            )
        self.weights = dataset_weights or {}

    @abstractmethod
    def iter_looks(
        self,
        *,
        sampler: Sampler,
    ) -> Iterator[LookResult]:
        """Yield one LookResult per planned look from sampler-produced batches."""


def _append_batch_to_cumulative_scores(
    batch: Sequence[ScoreLike],
    scores_seen_so_far: Sequence[ScoreLike],
    expected_n_seen: int,
) -> List[ScoreLike]:
    next_prefix = list(scores_seen_so_far)
    next_prefix.extend(batch)
    if len(next_prefix) != expected_n_seen:
        raise ValueError(
            "Sample batch size must match the planned cumulative look size: "
            f"expected_n_seen={expected_n_seen}, actual={len(next_prefix)}."
        )
    return next_prefix


class BootstrapReferenceEngine(SequentialEngine):
    """
    Bootstrap-based reference CI engine evaluated at interim looks.

    This engine is intended as an empirical baseline for CI-width trajectories
    across orderings. It estimates the best reference CI available from repeated
    bootstrap resampling at each look, but it is not a real sequential design and
    does not implement alpha-spending boundaries.
    """

    # This capability supports CI-width reference comparisons, not controlled repeated-look stopping.
    capabilities = frozenset({"ci"})
    supported_aggregations = frozenset({"mean_of_means", "pooled_mean"})

    def __init__(
        self,
        alpha: float,
        aggregator: ScoreAggregator,
        n_boot: int = 1000,
        bootstrap_seed: int = 0,
        dataset_weights: Optional[Dict[DatasetId, float]] = None,
    ):
        if n_boot <= 0:
            raise ValueError("n_boot must be positive.")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in the open interval (0, 1).")
        super().__init__(aggregator=aggregator, dataset_weights=dataset_weights)
        self.n_boot = n_boot
        self.alpha = alpha
        self.bootstrap_seed = bootstrap_seed

    def _bootstrap_ci(
        self,
        scores: Sequence[ScoreLike],
        dataset_sizes: DatasetSizes,
        look_index: int,
    ) -> Tuple[float, float, float]:
        from fixed_sample import bootstrap_ci

        streams: Dict[DatasetId, List[ScoreLike]] = {}
        for score in scores:
            streams.setdefault(score.dataset_id, []).append(score)

        result = bootstrap_ci(
            streams=streams,
            dataset_sizes=dataset_sizes,
            aggregator=self.aggregator,
            alpha=self.alpha,
            n_boot=self.n_boot,
            seed=self.bootstrap_seed + look_index,
            dataset_weights=self.weights,
        )
        return float(result.estimate), float(result.ci_lower), float(result.ci_upper)

    def iter_looks(
        self,
        *,
        sampler: Sampler,
    ) -> Iterator[LookResult]:
        sampling_plan = sampler.sampling_plan
        sample_batches = sampler.sample_batches()
        dataset_sizes = sampling_plan.dataset_sizes
        total_available = sampling_plan.total_available
        planned_sample_sizes = list(sampling_plan.cumulative_sample_sizes())
        batch_iter = iter(sample_batches)
        current_scores: List[ScoreLike] = []
        looks_emitted = 0

        for idx, target_sample_size in enumerate(planned_sample_sizes):
            try:
                batch = next(batch_iter)
            except StopIteration as exc:
                raise ValueError(
                    f"Sample batch stream ended before planned look {idx} at {target_sample_size} samples."
                ) from exc
            current_scores = _append_batch_to_cumulative_scores(
                batch,
                current_scores,
                target_sample_size,
            )
            estimate, ci_lower, ci_upper = self._bootstrap_ci(current_scores, dataset_sizes, look_index=idx)
            looks_emitted += 1
            yield LookResult(
                look_index=idx,
                n_seen=target_sample_size,
                total_examples=total_available,
                estimate=estimate,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                samples=tuple(current_scores),
            )
        if looks_emitted != len(planned_sample_sizes):
            raise ValueError(
                "BootstrapReferenceEngine did not emit one look per planned sample size: "
                f"planned={len(planned_sample_sizes)}, emitted={looks_emitted}."
            )
        _raise_if_extra_batch(batch_iter)

def _raise_if_extra_batch(batch_iter: Iterator[Sequence[ScoreLike]]) -> None:
    try:
        next(batch_iter)
    except StopIteration:
        return
    raise ValueError("Sample batch stream produced an unexpected extra look.")



class SimpleMeanOfMeansGsDesignEngine(SequentialEngine):
    """
    Minimal R gsDesign wrapper for the first stable path:
    mean-of-means, two-sided efficacy, planned information timing, optional futility.
    """

    capabilities = frozenset({"ci", "efficacy"})
    supported_aggregations = frozenset({"mean_of_means"})

    def __init__(
        self,
        alpha: float,
        aggregator: ScoreAggregator,
        spending_function: SpendingFunction = "pocock",
        min_information_increment: float = 0.01,
        enable_futility: bool = False,
        futility_mdes: float = 0.005,
        beta: float = 0.1,
        futility_test_type: int = 4,
        futility_astar: float = 0.5,
        futility_spending_function: SpendingFunction = "pocock",
        dataset_weights: Optional[Dict[DatasetId, float]] = None,
    ):
        if ro is None or importr is None:
            raise ImportError("rpy2 and gsDesign are required for SimpleMeanOfMeansGsDesignEngine")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in the open interval (0, 1).")
        if spending_function != "pocock":
            raise ValueError("SimpleMeanOfMeansGsDesignEngine currently supports only spending_function='pocock'.")
        if futility_spending_function != "pocock":
            raise ValueError(
                "SimpleMeanOfMeansGsDesignEngine currently supports only futility_spending_function='pocock'."
            )
        if not (0.0 < min_information_increment <= 1.0):
            raise ValueError("min_information_increment must be in the interval (0, 1].")
        if futility_mdes <= 0.0 or not math.isfinite(futility_mdes):
            raise ValueError("futility_mdes must be positive and finite.")
        if not (0.0 < beta < 1.0):
            raise ValueError("beta must be in the open interval (0, 1).")
        if futility_test_type not in {3, 4, 5}:
            raise ValueError("futility_test_type must be 3, 4, or 5.")
        if not (0.0 < futility_astar < 1.0):
            raise ValueError("futility_astar must be in the open interval (0, 1).")
        super().__init__(aggregator=aggregator, dataset_weights=dataset_weights)
        if enable_futility:
            self.capabilities = frozenset({"ci", "efficacy", "futility"})
        self.alpha = alpha
        self.spending_function = spending_function
        self.min_information_increment = min_information_increment
        self.enable_futility = enable_futility
        self.futility_mdes = futility_mdes
        self.beta = beta
        self.futility_test_type = futility_test_type
        self.futility_astar = futility_astar
        self.futility_spending_function = futility_spending_function
        self._gsdesign = importr("gsDesign")

    def _planned_timing(
        self,
        *,
        batch_allocations: Sequence[Mapping[DatasetId, int]],
        dataset_sizes: Mapping[DatasetId, int],
        pilot_stddevs: Mapping[DatasetId, float],
    ) -> Tuple[Tuple[float, ...], Tuple[int, ...], float]:
        if not batch_allocations:
            raise ValueError("batch_allocations must not be empty.")
        expected_dataset_ids = set(dataset_sizes)
        if set(pilot_stddevs) != expected_dataset_ids:
            raise ValueError(
                "sampler allocation_weights must exactly match dataset_sizes for planned timing: "
                f"missing={sorted(expected_dataset_ids - set(pilot_stddevs))}, "
                f"unexpected={sorted(set(pilot_stddevs) - expected_dataset_ids)}."
            )

        if self.weights:
            if set(self.weights) != expected_dataset_ids:
                raise ValueError(
                    "dataset_weights must exactly match dataset_sizes: "
                    f"missing={sorted(expected_dataset_ids - set(self.weights))}, "
                    f"unexpected={sorted(set(self.weights) - expected_dataset_ids)}."
                )
            weight_sum = sum(float(weight) for weight in self.weights.values())
            if weight_sum <= 0.0 or not math.isfinite(weight_sum):
                raise ValueError("dataset_weights must sum to a positive finite value.")
            mean_weights = {
                dataset_id: float(weight) / weight_sum
                for dataset_id, weight in self.weights.items()
            }
        else:
            mean_weights = {
                dataset_id: 1.0 / len(dataset_sizes)
                for dataset_id in dataset_sizes
            }

        variance_terms = {}
        for dataset_id in dataset_sizes:
            stddev = float(pilot_stddevs[dataset_id])
            mean_weight = float(mean_weights[dataset_id])
            if not math.isfinite(stddev) or stddev < 0.0:
                raise ValueError(f"Pilot stddev must be finite and non-negative for dataset {dataset_id!r}.")
            if not math.isfinite(mean_weight) or mean_weight <= 0.0:
                raise ValueError(f"Mean-of-means weight must be finite and positive for dataset {dataset_id!r}.")
            variance_terms[dataset_id] = (mean_weight**2) * (stddev**2)

        if sum(variance_terms.values()) <= 0.0:
            raise ValueError("Planned timing requires positive pilot variance for at least one dataset.")

        full_variance = sum(
            variance_terms[dataset_id] / dataset_sizes[dataset_id]
            for dataset_id in dataset_sizes
        )
        if not math.isfinite(full_variance) or full_variance <= 0.0:
            raise ValueError("Planned full-benchmark variance must be positive and finite.")
        full_benchmark_se = math.sqrt(full_variance)

        cumulative_counts = {dataset_id: 0 for dataset_id in dataset_sizes}
        selected_timing: List[float] = []
        selected_look_indices: List[int] = []
        last_selected_timing: Optional[float] = None
        final_look_index: Optional[int] = None
        for look_index, allocation in enumerate(batch_allocations):
            if set(allocation) != expected_dataset_ids:
                raise ValueError(
                    f"batch allocation {look_index} must contain exactly the sampling plan datasets: "
                    f"missing={sorted(expected_dataset_ids - set(allocation))}, "
                    f"unexpected={sorted(set(allocation) - expected_dataset_ids)}."
                )
            for dataset_id, count in allocation.items():
                cumulative_counts[dataset_id] += int(count)
                if cumulative_counts[dataset_id] > dataset_sizes[dataset_id]:
                    raise ValueError(
                        f"Planned allocation for dataset {dataset_id!r} exceeds its declared size: "
                        f"planned={cumulative_counts[dataset_id]}, declared={dataset_sizes[dataset_id]}."
                    )

            current_variance = 0.0
            for dataset_id in dataset_sizes:
                n_seen = cumulative_counts[dataset_id]
                if n_seen <= 0:
                    raise ValueError(
                        "Planned mean-of-means timing requires every dataset to have samples at every look. "
                        f"Missing dataset={dataset_id!r}."
                    )
                current_variance += variance_terms[dataset_id] / n_seen
            if not math.isfinite(current_variance) or current_variance <= 0.0:
                raise ValueError("Planned current-look variance must be positive and finite.")
            information_fraction = full_variance / current_variance
            if information_fraction > 1.0 + 1e-12:
                raise ValueError(f"Timing values must be in (0, 1]. Got {information_fraction!r}.")
            if information_fraction >= 1.0 - 1e-12:
                information_fraction = 1.0
                final_look_index = look_index

            if (
                last_selected_timing is None
                or information_fraction == 1.0
                or information_fraction - last_selected_timing >= self.min_information_increment
            ):
                selected_timing.append(float(information_fraction))
                selected_look_indices.append(look_index)
                last_selected_timing = float(information_fraction)

            if information_fraction == 1.0:
                break

        if final_look_index is None:
            final_look_index = len(batch_allocations) - 1
            if selected_look_indices[-1] != final_look_index:
                selected_timing.append(1.0)
                selected_look_indices.append(final_look_index)
            else:
                selected_timing[-1] = 1.0

        if not all(math.isfinite(value) for value in selected_timing):
            raise ValueError("Timing values must be finite.")
        if any(value <= 0.0 or value > 1.0 + 1e-12 for value in selected_timing):
            raise ValueError(f"Timing values must be in (0, 1]. Got {selected_timing!r}.")
        if any(later <= earlier for earlier, later in zip(selected_timing, selected_timing[1:])):
            raise ValueError(f"Timing values must be strictly increasing. Got {selected_timing!r}.")

        return (
            tuple(float(value) for value in selected_timing),
            tuple(int(index) for index in selected_look_indices),
            float(full_benchmark_se),
        )

    def _boundaries(
        self,
        timing: Sequence[float],
        full_benchmark_se: float,
    ) -> Tuple[Tuple[float, ...], Tuple[Optional[float], ...]]:
        planned_information = [float(value) * len(timing) for value in timing]
        design_kwargs = {
            "k": len(timing),
            "alpha": self.alpha / 2.0,
            "test_type": self.futility_test_type if self.enable_futility else 2,
            "sfu": ro.r("gsDesign::sfHSD"),
            "sfupar": ro.FloatVector([1.0]),
            "n.I": ro.FloatVector(planned_information),
            "maxn.IPlan": float(len(timing)),
        }
        if self.enable_futility:
            design_kwargs["sfl"] = ro.r("gsDesign::sfHSD")
            design_kwargs["sflpar"] = ro.FloatVector([1.0])
            # Types 3 and 4 are MDES-based beta spending under the alternative.
            # Type 5 is null-calibrated lower-bound spending; futility_mdes is intentionally ignored.
            if self.futility_test_type in {3, 4}:
                design_kwargs["beta"] = self.beta
                design_kwargs["delta"] = self.futility_mdes / full_benchmark_se
            else:
                design_kwargs["astar"] = self.futility_astar
                design_kwargs["delta"] = 0.0
        else:
            design_kwargs["delta"] = 0.0
        design = self._gsdesign.gsDesign(**design_kwargs)
        upper_bounds_r = ro.r("function(design) if(!is.null(design$upper$bound)) design$upper$bound else NULL")(design)
        if upper_bounds_r is NULL or len(upper_bounds_r) == 0:
            raise ValueError("Failed to extract upper bounds from gsDesign object.")
        upper_bounds = tuple(float(value) for value in upper_bounds_r)
        if len(upper_bounds) != len(timing):
            binding_design_was_shortened = (
                self.enable_futility
                and self.futility_test_type == 3
                and len(upper_bounds) < len(timing)
            )
            if not binding_design_was_shortened:
                raise ValueError(
                    "gsDesign returned an unexpected number of upper boundaries: "
                    f"expected={len(timing)}, actual={len(upper_bounds)}."
                )
        lower_bounds_r = ro.r("function(design) if(!is.null(design$lower$bound)) design$lower$bound else NULL")(design)
        if lower_bounds_r is NULL or len(lower_bounds_r) == 0:
            lower_bounds: Tuple[Optional[float], ...] = tuple(None for _ in upper_bounds)
        else:
            lower_values = tuple(float(value) for value in lower_bounds_r)
            if len(lower_values) == len(upper_bounds) - 1:
                lower_bounds = lower_values + (None,)
            elif len(lower_values) != len(upper_bounds):
                raise ValueError(
                    "gsDesign returned an unexpected number of lower boundaries: "
                    f"expected={len(upper_bounds)}, actual={len(lower_values)}."
                )
            else:
                lower_bounds = lower_values
        return upper_bounds, lower_bounds

    def iter_looks(
        self,
        *,
        sampler: Sampler,
    ) -> Iterator[LookResult]:
        sampling_plan = sampler.sampling_plan
        batch_allocations = sampler.batch_allocations
        pilot_stddevs = sampler.allocation_weights
        if batch_allocations is None or pilot_stddevs is None:
            raise ValueError(
                "SimpleMeanOfMeansGsDesignEngine requires sampler.batch_allocations and "
                "sampler.allocation_weights to build planned information timing."
            )

        design_timing, design_look_indices, full_benchmark_se = self._planned_timing(
            batch_allocations=batch_allocations,
            dataset_sizes=sampling_plan.dataset_sizes,
            pilot_stddevs=pilot_stddevs,
        )
        upper_bounds, lower_bounds = self._boundaries(design_timing, full_benchmark_se)
        design_look_indices = design_look_indices[:len(upper_bounds)]
        upper_bound_by_look_index = {
            look_index: upper_bound
            for look_index, upper_bound in zip(design_look_indices, upper_bounds)
        }
        lower_bound_by_look_index = {
            look_index: lower_bound
            for look_index, lower_bound in zip(design_look_indices, lower_bounds)
        }
        emitted_look_index_by_original_index = {
            original_look_index: emitted_look_index
            for emitted_look_index, original_look_index in enumerate(design_look_indices)
        }

        planned_sample_sizes = list(sampling_plan.cumulative_sample_sizes())
        if len(batch_allocations) != len(planned_sample_sizes):
            raise ValueError(
                "batch_allocations must contain one allocation per planned look: "
                f"allocations={len(batch_allocations)}, planned_looks={len(planned_sample_sizes)}."
            )

        cumulative_scores: List[ScoreLike] = []
        batch_iter = iter(sampler.sample_batches())
        looks_emitted = 0
        for look_index, expected_n_seen in enumerate(planned_sample_sizes):
            try:
                batch = next(batch_iter)
            except StopIteration as exc:
                raise ValueError(
                    f"Sample batch stream ended before planned look {look_index} at {expected_n_seen} samples."
                ) from exc

            cumulative_scores = _append_batch_to_cumulative_scores(
                batch,
                cumulative_scores,
                expected_n_seen,
            )
            if look_index not in upper_bound_by_look_index:
                continue

            emitted_look_index = emitted_look_index_by_original_index[look_index]
            upper_boundary = upper_bound_by_look_index[look_index]
            lower_boundary = lower_bound_by_look_index[look_index]
            aggregate = self.aggregator.compute(cumulative_scores, sampling_plan.dataset_sizes, self.weights)
            if aggregate.variance <= 0.0 or not math.isfinite(aggregate.variance):
                raise ValueError("Aggregate variance must be positive and finite to compute a gsDesign look.")

            estimate = float(aggregate.estimate)
            se = math.sqrt(float(aggregate.variance))
            z_stat = estimate / se
            margin = float(upper_boundary) * se

            looks_emitted += 1
            yield LookResult(
                look_index=emitted_look_index,
                n_seen=expected_n_seen,
                total_examples=sampling_plan.total_available,
                estimate=estimate,
                ci_lower=estimate - margin,
                ci_upper=estimate + margin,
                upper_boundary=float(upper_boundary),
                lower_boundary=None if lower_boundary is None else float(lower_boundary),
                futility_crossed=None if lower_boundary is None else bool(z_stat <= lower_boundary),
                z_stat=float(z_stat),
                samples=tuple(cumulative_scores),
            )

        if looks_emitted != len(design_look_indices):
            raise ValueError(
                "SimpleMeanOfMeansGsDesignEngine did not emit one look per selected design look: "
                f"selected={len(design_look_indices)}, emitted={looks_emitted}."
            )
        _raise_if_extra_batch(batch_iter)


