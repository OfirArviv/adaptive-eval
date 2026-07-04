from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Literal, Mapping, Optional, Sequence, Dict

import numpy as np
from scipy.stats import norm

from sequential.aggregators import ScoreAggregator
from sequential.types import AggregationMode, DatasetId, DatasetSizes, DatasetStreams, ScoreLike


@dataclass(frozen=True)
class FixedSampleEstimate:
    method: Literal["normal_approx", "bootstrap_percentile"]
    aggregation_mode: AggregationMode
    estimate: float
    variance: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    alpha: float
    p_value: Optional[float] = None
    n_boot: Optional[int] = None
    seed: Optional[int] = None


def normal_ci(
    *,
    streams: DatasetStreams,
    dataset_sizes: DatasetSizes,
    aggregator: ScoreAggregator,
    alpha: float = 0.05,
    dataset_weights: Optional[Mapping[DatasetId, float]] = None,
) -> FixedSampleEstimate:
    """Normal-approximation fixed-sample CI for one complete score collection."""
    _validate_alpha(alpha)
    scores = _flatten_streams(streams)
    aggregate = aggregator.compute(scores, dataset_sizes, dataset_weights)
    standard_error = math.sqrt(aggregate.variance)
    z_value = float(norm.ppf(1.0 - alpha / 2.0))
    ci_lower = aggregate.estimate - z_value * standard_error
    ci_upper = aggregate.estimate + z_value * standard_error
    p_value = float(2.0 * norm.sf(abs(aggregate.estimate / standard_error)))

    return FixedSampleEstimate(
        method="normal_approx",
        aggregation_mode=aggregator.aggregation_mode,
        estimate=float(aggregate.estimate),
        variance=float(aggregate.variance),
        standard_error=float(standard_error),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        p_value=float(p_value),
        alpha=float(alpha),
    )


def bootstrap_ci(
    *,
    streams: DatasetStreams,
    dataset_sizes: DatasetSizes,
    aggregator: ScoreAggregator,
    alpha: float = 0.05,
    n_boot: int = 1000,
    seed: int = 0,
    dataset_weights: Optional[Mapping[DatasetId, float]] = None,
) -> FixedSampleEstimate:
    """Percentile bootstrap fixed-sample CI, resampling within each dataset."""
    _validate_alpha(alpha)
    if n_boot <= 0:
        raise ValueError("n_boot must be positive.")

    scores = _flatten_streams(streams)
    estimate = aggregator.compute(scores, dataset_sizes, dataset_weights).estimate
    rng = np.random.default_rng(seed)
    scores_by_dataset = _scores_by_dataset(scores)
    boot_estimates: List[float] = []
    for _ in range(n_boot):
        sampled_scores: List[ScoreLike] = []
        for dataset_scores in scores_by_dataset.values():
            sample_indices = rng.integers(0, len(dataset_scores), size=len(dataset_scores))
            sampled_scores.extend(dataset_scores[index] for index in sample_indices)
        boot_estimates.append(
            float(aggregator.compute(sampled_scores, dataset_sizes, dataset_weights).estimate)
        )

    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)
    bootstrap_values = np.asarray(boot_estimates, dtype=float)
    variance = float(np.var(bootstrap_values, ddof=1)) if len(bootstrap_values) > 1 else 0.0
    ci_lower = float(np.percentile(boot_estimates, lower_q))
    ci_upper = float(np.percentile(boot_estimates, upper_q))

    return FixedSampleEstimate(
        method="bootstrap_percentile",
        aggregation_mode=aggregator.aggregation_mode,
        estimate=float(estimate),
        variance=variance,
        standard_error=math.sqrt(variance),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        alpha=float(alpha),
        p_value=None,
        n_boot=int(n_boot),
        seed=int(seed),
    )


def _validate_alpha(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in the open interval (0, 1).")


def _flatten_streams(streams: DatasetStreams) -> List[ScoreLike]:
    scores = [score for dataset_scores in streams.values() for score in dataset_scores]
    if not scores:
        raise ValueError("At least one score is required.")
    return scores


def _scores_by_dataset(scores: Sequence[ScoreLike]) -> Dict[DatasetId, List[ScoreLike]]:
    scores_by_dataset: Dict[DatasetId, List[ScoreLike]] = {}
    for score in scores:
        scores_by_dataset.setdefault(score.dataset_id, []).append(score)
    return scores_by_dataset
