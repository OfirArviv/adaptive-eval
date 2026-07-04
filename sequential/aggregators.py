from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from sequential.types import AggregationMode, DatasetId, ScoreLike


_INFORMATION_FRACTION_TOLERANCE = 1e-12


def _validate_information_fraction(information_fraction: float) -> float:
    if information_fraction < -_INFORMATION_FRACTION_TOLERANCE or information_fraction > 1.0 + _INFORMATION_FRACTION_TOLERANCE:
        raise ValueError(
            "information_fraction must be in [0, 1]. This usually means the projected "
            "full-benchmark estimator variance is larger than the current estimator variance, "
            "or full_dataset_sizes are inconsistent with the observed scores."
        )
    return min(max(float(information_fraction), 0.0), 1.0)


def _normalized_weights(
    full_dataset_sizes: Mapping[DatasetId, int],
    weights: Optional[Mapping[DatasetId, float]] = None,
) -> Optional[Dict[DatasetId, float]]:
    if not weights:
        return None

    expected_dataset_ids = set(full_dataset_sizes)
    if set(weights) != expected_dataset_ids:
        raise ValueError(
            "weights keys must exactly match full_dataset_sizes keys when weights are provided: "
            f"missing={sorted(expected_dataset_ids - set(weights))}, "
            f"unexpected={sorted(set(weights) - expected_dataset_ids)}."
        )

    numeric_weights: Dict[DatasetId, float] = {}
    invalid_weights = []
    for dataset_id, weight in weights.items():
        try:
            numeric_weight = float(weight)
        except (TypeError, ValueError):
            invalid_weights.append(dataset_id)
            continue
        if not math.isfinite(numeric_weight) or numeric_weight <= 0.0:
            invalid_weights.append(dataset_id)
            continue
        numeric_weights[dataset_id] = numeric_weight

    invalid_weights = sorted(invalid_weights)
    if invalid_weights:
        raise ValueError(f"weights must be finite and positive for every dataset: {invalid_weights}")

    weight_sum = sum(numeric_weights.values())
    return {dataset_id: weight / weight_sum for dataset_id, weight in numeric_weights.items()}


def _group_score_values_by_dataset(scores: Sequence[ScoreLike]) -> Dict[DatasetId, list[float]]:
    values_by_dataset: Dict[DatasetId, list[float]] = {}
    for score in scores:
        values_by_dataset.setdefault(score.dataset_id, []).append(float(score.score))
    return values_by_dataset


@dataclass(frozen=True)
class AggregatedEstimate:
    """
    Aggregated estimate plus information quantities for the chosen estimator.

    `variance` is the variance of the aggregate estimator at the current look, not the raw
    per-example score variance. For a pooled mean this is roughly `sigma^2 / n_seen`; for a
    mean-of-means it is `sum_j w_j^2 * sigma_j^2 / n_j_seen`.

    `max_information` is the projected full-benchmark information using the same variance model and
    the full planned dataset sizes: `max_information = 1 / variance_full`. It is needed to convert
    raw effects into full-benchmark standard errors and to reason about how much information the
    complete benchmark can provide.

    `information_fraction` is the ratio of projected full-benchmark estimator variance to current
    estimator variance: `variance_full / variance`. Equivalently, it is current information divided
    by projected full-benchmark information. It may differ from `n_seen / n_total`, especially for
    mean-of-means where dataset-specific variances and sample counts determine precision.
    """

    estimate: float
    variance: float
    information_fraction: float
    max_information: float


class ScoreAggregator(ABC):
    """Defines how scores are aggregated and how estimator information scales."""

    aggregation_mode: AggregationMode

    @abstractmethod
    def compute(
        self,
        scores: Sequence[ScoreLike],
        full_dataset_sizes: Mapping[DatasetId, int],
        weights: Optional[Mapping[DatasetId, float]] = None,
    ) -> AggregatedEstimate:
        pass


class MeanOfMeansAggregator(ScoreAggregator):
    """Unweighted mean-of-dataset-means (or weighted by provided dataset weights)."""

    aggregation_mode = "mean_of_means"

    def compute(
        self,
        scores: Sequence[ScoreLike],
        full_dataset_sizes: Mapping[DatasetId, int],
        weights: Optional[Mapping[DatasetId, float]] = None,
    ) -> AggregatedEstimate:
        scores_by_dataset_id = _group_score_values_by_dataset(scores)

        normalized_weights = _normalized_weights(full_dataset_sizes, weights)
        if normalized_weights is None:
            normalized_weights = {dataset_id: 1.0 / len(scores_by_dataset_id) for dataset_id in scores_by_dataset_id.keys()}

        estimate = 0.0
        current_estimator_variance = 0.0
        full_estimator_variance = 0.0
        for dataset_id, weight in normalized_weights.items():
            vals = scores_by_dataset_id[dataset_id]
            current_n = len(vals)
            full_n = full_dataset_sizes[dataset_id]
            raw_score_variance = float(np.var(vals, ddof=1)) if current_n > 1 else 0.0
            estimate += float(np.mean(vals)) * weight
            current_estimator_variance += (weight**2) * raw_score_variance / current_n
            full_estimator_variance += (weight**2) * raw_score_variance / full_n

        if (
            not math.isfinite(current_estimator_variance)
            or not math.isfinite(full_estimator_variance)
            or current_estimator_variance <= 0.0
            or full_estimator_variance <= 0.0
        ):
            raise ValueError("MeanOfMeansAggregator requires positive observed variance to estimate information.")

        max_information = 1.0 / full_estimator_variance
        information_fraction = (
            full_estimator_variance / current_estimator_variance
            if current_estimator_variance > 0.0
            else 1.0
        )
        information_fraction = _validate_information_fraction(information_fraction)
        return AggregatedEstimate(float(estimate), float(current_estimator_variance), information_fraction, max_information)


class PooledAggregator(ScoreAggregator):
    """Treat all scores as one pooled sample, ignoring dataset boundaries."""

    aggregation_mode = "pooled_mean"

    def compute(
        self,
        scores: Sequence[ScoreLike],
        full_dataset_sizes: Mapping[DatasetId, int],
        weights: Optional[Mapping[DatasetId, float]] = None,
    ) -> AggregatedEstimate:
        if weights:
            raise ValueError("PooledAggregator does not support dataset weights.")

        scores_by_dataset_id = _group_score_values_by_dataset(scores)
        vals = [value for values in scores_by_dataset_id.values() for value in values]
        n = len(vals)

        mean = float(np.mean(vals))
        raw_score_variance = float(np.var(vals, ddof=1)) if n > 1 else 0.0
        current_estimator_variance = raw_score_variance / n if n > 0 else 0.0
        full_n = sum(full_dataset_sizes.values())
        full_estimator_variance = raw_score_variance / full_n if full_n > 0 else 0.0
        if (
            not math.isfinite(current_estimator_variance)
            or not math.isfinite(full_estimator_variance)
            or current_estimator_variance <= 0.0
            or full_estimator_variance <= 0.0
        ):
            raise ValueError("PooledAggregator requires positive observed variance to estimate information.")

        max_information = 1.0 / full_estimator_variance
        information_fraction = (
            full_estimator_variance / current_estimator_variance
            if current_estimator_variance > 0.0
            else 1.0
        )

        information_fraction = _validate_information_fraction(information_fraction)
        return AggregatedEstimate(mean, float(current_estimator_variance), information_fraction, max_information)
