from __future__ import annotations

import math

from sequential.types import DatasetId, DatasetSizes, DatasetStreams, ScoreLike


def validate_dataset_source(
    data_map: DatasetStreams,
    dataset_sizes: DatasetSizes,
) -> None:
    """
    Validate that stream buckets and declared sizes describe the same benchmark.

    This catches missing or extra datasets before samplers consume streams.
    SamplingPlan owns declared-size validation.
    """

    dataset_ids_with_streams = set(data_map.keys())
    dataset_ids_with_declared_size = set(dataset_sizes.keys())

    missing_streams = sorted(dataset_ids_with_declared_size - dataset_ids_with_streams)
    if missing_streams:
        raise ValueError(f"dataset_sizes contains dataset ids missing from data_map: {missing_streams}")

    missing_sizes = sorted(dataset_ids_with_streams - dataset_ids_with_declared_size)
    if missing_sizes:
        raise ValueError(f"data_map contains dataset ids missing from dataset_sizes: {missing_sizes}")


def validate_emitted_score(dataset_id: DatasetId, score: ScoreLike) -> None:
    """
    Validate that a sampled score matches its bucket and has a finite numeric value.

    Samplers rely on this to catch invalid records at the stream boundary, before
    aggregation mixes datasets or computes variance.
    """

    if score.dataset_id != dataset_id:
        raise ValueError(
            f"Sample dataset_id={score.dataset_id!r} does not match its bucket key {dataset_id!r}."
        )
    try:
        value = float(score.score)
    except (TypeError, ValueError):
        raise ValueError(
            f"Scores must be finite numeric values: dataset_id={score.dataset_id!r}, "
            f"instance_id={score.instance_id!r}, score={score.score!r}."
        ) from None
    if not math.isfinite(value):
        raise ValueError(
            f"Scores must be finite: dataset_id={score.dataset_id!r}, "
            f"instance_id={score.instance_id!r}, score={score.score!r}."
        )


def full_benchmark_standard_error_from_max_information(max_information: float) -> float:
    """
    Convert projected full-benchmark information into the full-benchmark standard error.

    This is non-trivial because information is defined for the estimator, not for raw examples:

        information = 1 / Var(estimator)
        standard_error = sqrt(Var(estimator)) = 1 / sqrt(information)

    Simple pooled-mean example:
        If all examples are pooled, Var(mean_full) = sigma^2 / N.
        Therefore max_information = N / sigma^2, and this function returns sigma / sqrt(N).

    Mean-of-means example:
        If the benchmark estimate is the average of dataset means,
        Var(full) = sum_j w_j^2 * sigma_j^2 / N_j.
        The aggregator hides those dataset-level details and returns
        max_information = 1 / Var(full). This function then returns sqrt(Var(full)), the standard
        error of the full mean-of-means benchmark estimate.

    The sequential engines use this helper to turn an aggregator's projected full-benchmark
    information into the raw-effect standard error needed for futility planning and boundary
    construction.
    """

    if not math.isfinite(max_information) or max_information <= 0.0:
        raise ValueError(f"max_information must be finite and positive, got {max_information!r}.")
    return 1.0 / math.sqrt(max_information)
