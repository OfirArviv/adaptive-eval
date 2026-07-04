from __future__ import annotations

import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from case_studies.common.configs import PairStrategy, SamplerName, SummaryGroup

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
from case_studies.common.base import BaseCaseStudy
from case_studies.common.utils import (
    log_stage_duration,
    log_stage_start,
    make_aggregator,
    progress_iterator,
    replicate_iterator,
    select_model_pairs,
)
from case_studies.common.schemas import CaseStudyResult
from data.benchmark import Benchmark
from data.core import BenchmarkBundle, PairwiseBundle
from data.views import MissingPolicy, OrderedBenchmarkView, PairwiseOrderedBenchmarkView
from fixed_sample import normal_ci
from sequential.aggregators import ScoreAggregator
from sequential.samplers import NeymanSampler, RoundRobinSampler, SamplingPlan
from sequential.types import AggregationMode, ScoreLike


TargetMode = Literal["single_model", "pairwise_difference"]
MilestoneKind = Literal["precision", "ci_half_width"]


@dataclass(frozen=True)
class CIWidthPrecisionCurveInputConfig:
    target_mode: TargetMode = "pairwise_difference"
    top_k: int = 10
    pair_strategy: PairStrategy = "all"
    alpha: float = 0.05
    aggregation_mode: AggregationMode = "mean_of_means"
    sampler_name: SamplerName = "round_robin"
    precision_milestones: Sequence[float] = ()
    ci_half_width_milestones: Sequence[float] = ()
    initial_size_per_dataset: int = 100
    batch_size: int = 100
    n_replicates: int = 100
    missing_policy: MissingPolicy = "error"
    use_cache: bool = True
    verbose: bool = True


@dataclass(frozen=True)
class CIWidthPrecisionCurveStudyConfig:
    benchmark_id: str
    benchmark_version: str
    target_mode: TargetMode
    top_k: int
    pair_strategy: PairStrategy
    n_replicates: int
    batch_size: int
    alpha: float
    aggregation_mode: AggregationMode
    sampler_name: SamplerName
    initial_size_per_dataset: int
    precision_milestones: list[float]
    ci_half_width_milestones: list[float]
    missing_policy: MissingPolicy
    use_cache: bool
    verbose: bool


@dataclass(frozen=True)
class CIWidthPrecisionCurveState:
    benchmark: Benchmark
    aggregator: ScoreAggregator
    targets: list["_PrecisionTarget"]
    pair_tuples: list[tuple[str, str]]


@dataclass(frozen=True)
class CIWidthPrecisionCurveRow:
    benchmark_id: str
    benchmark_version: str
    target_mode: TargetMode
    target_id: str
    rank: int
    replicate_index: int
    requested_sample_count: int | None
    effective_sample_fraction: float
    samples_used: int
    total_available: int
    estimate: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    ci_width: float
    ci_half_width: float
    aggregation_mode: str
    alpha: float


@dataclass(frozen=True)
class CIWidthPrecisionCurveSummaryRow:
    benchmark_id: str
    group_label: SummaryGroup
    requested_sample_count: int | None
    n_rows: int
    n_pairs: int
    n_replicates: int
    median_effective_sample_fraction: float
    median_ci_width: float
    p25_ci_width: float
    p75_ci_width: float
    median_ci_half_width: float
    p25_ci_half_width: float
    p75_ci_half_width: float
    mean_samples_used: float
    median_samples_used: float


@dataclass(frozen=True)
class MilestoneSummaryRow:
    benchmark_id: str
    group_label: SummaryGroup
    milestone_kind: MilestoneKind
    milestone_target: float
    milestone_percentage: float
    milestone_ci_half_width: float
    median_samples_needed: float | None
    median_sample_fraction_needed: float | None


@dataclass(frozen=True)
class _PrecisionTarget:
    target_id: str
    rank: int | None


StudySummaryRow = CIWidthPrecisionCurveSummaryRow | MilestoneSummaryRow


class CIWidthPrecisionCurveCaseStudy(
    BaseCaseStudy[
        CIWidthPrecisionCurveInputConfig,
        CIWidthPrecisionCurveStudyConfig,
        CIWidthPrecisionCurveState,
        CIWidthPrecisionCurveRow,
        StudySummaryRow,
    ]
):
    study_name = "ci_width_precision_curve"

    def validate_input_config(self, config: CIWidthPrecisionCurveInputConfig) -> None:
        if config.target_mode not in {"single_model", "pairwise_difference"}:
            raise ValueError("target_mode must be 'single_model' or 'pairwise_difference'.")
        if config.top_k < 2:
            raise ValueError("top_k must be at least 2.")
        if config.n_replicates <= 0:
            raise ValueError("n_replicates must be positive.")
        if config.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not (0.0 < config.alpha < 1.0):
            raise ValueError("alpha must be in the open interval (0, 1).")
        if config.aggregation_mode not in {"mean_of_means", "pooled_mean"}:
            raise ValueError("aggregation_mode must be 'mean_of_means' or 'pooled_mean'.")
        if config.sampler_name not in {"round_robin", "neyman"}:
            raise ValueError("sampler_name must be 'round_robin' or 'neyman'.")
        if config.initial_size_per_dataset <= 0:
            raise ValueError("initial_size_per_dataset must be positive.")
        if config.sampler_name == "neyman" and config.initial_size_per_dataset < 30:
            raise ValueError("Neyman sampler requires initial_size_per_dataset to be at least 30.")
        for milestone in config.precision_milestones:
            if not (0.0 < float(milestone) <= 1.0):
                raise ValueError("precision_milestones must be in the interval (0, 1].")
        for milestone in config.ci_half_width_milestones:
            if float(milestone) <= 0.0:
                raise ValueError("ci_half_width_milestones must be positive.")

    def build_state(
        self,
        bundle: BenchmarkBundle,
        benchmark: Benchmark,
        config: CIWidthPrecisionCurveInputConfig,
    ) -> CIWidthPrecisionCurveState:
        aggregator = make_aggregator(config.aggregation_mode)
        if config.target_mode == "pairwise_difference":
            selection = select_model_pairs(bundle, top_k=config.top_k, pair_strategy=config.pair_strategy)
            pair_tuples = [(pair.model_a, pair.model_b) for pair in selection.pairs]
            targets = [
                _PrecisionTarget(
                    target_id=pair.pair_id,
                    rank=abs(selection.rank_by_model[pair.model_b] - selection.rank_by_model[pair.model_a]),
                )
                for pair in selection.pairs
            ]
        else:
            selected_model_ids, rank_by_model = _select_models(bundle, config.top_k)
            targets = [
                _PrecisionTarget(
                    target_id=model_id,
                    rank=rank_by_model[model_id],
                )
                for model_id in selected_model_ids
            ]
            pair_tuples = []
        return CIWidthPrecisionCurveState(
            benchmark=benchmark,
            aggregator=aggregator,
            targets=targets,
            pair_tuples=pair_tuples,
        )

    def build_study_config(
        self,
        bundle: BenchmarkBundle,
        config: CIWidthPrecisionCurveInputConfig,
        state: CIWidthPrecisionCurveState,
    ) -> CIWidthPrecisionCurveStudyConfig:
        return CIWidthPrecisionCurveStudyConfig(
            benchmark_id=bundle.benchmark_id,
            benchmark_version=bundle.version,
            target_mode=config.target_mode,
            top_k=config.top_k,
            pair_strategy=config.pair_strategy,
            n_replicates=config.n_replicates,
            batch_size=config.batch_size,
            alpha=config.alpha,
            aggregation_mode=config.aggregation_mode,
            sampler_name=config.sampler_name,
            initial_size_per_dataset=config.initial_size_per_dataset,
            precision_milestones=[float(milestone) for milestone in config.precision_milestones],
            ci_half_width_milestones=[float(milestone) for milestone in config.ci_half_width_milestones],
            missing_policy=config.missing_policy,
            use_cache=config.use_cache,
            verbose=config.verbose,
        )

    def build_raw_rows(
        self,
        state: CIWidthPrecisionCurveState,
        config: CIWidthPrecisionCurveInputConfig,
        study_config: CIWidthPrecisionCurveStudyConfig,
    ) -> list[CIWidthPrecisionCurveRow]:
        rows: list[CIWidthPrecisionCurveRow] = []
        for replicate_index in replicate_iterator(config.n_replicates, config.verbose):
            replicate_number = replicate_index + 1
            ordering_started_at = time.perf_counter()
            log_stage_start(
                self.study_name,
                config.verbose,
                f"replicate {replicate_number}/{config.n_replicates} ordering bundle",
            )
            if config.target_mode == "pairwise_difference":
                assert state.pair_tuples
                ordered_bundle = PairwiseOrderedBenchmarkView(
                    benchmark=state.benchmark,
                    pairs=state.pair_tuples,
                    seed=replicate_index,
                    missing_policy=config.missing_policy,
                ).load(use_cache=config.use_cache)
            else:
                ordered_bundle = OrderedBenchmarkView(
                    benchmark=state.benchmark,
                    seed=replicate_index,
                ).load(use_cache=config.use_cache)
            log_stage_duration(
                self.study_name,
                config.verbose,
                f"replicate {replicate_number}/{config.n_replicates} ordering bundle",
                ordering_started_at,
            )

            replicate_rows_started_at = time.perf_counter()
            log_stage_start(
                self.study_name,
                config.verbose,
                f"replicate {replicate_number}/{config.n_replicates} target rows",
            )
            replicate_rows = (
                _replicate_rows(
                    ordered_bundle=ordered_bundle,
                    targets=state.targets,
                    target_mode=config.target_mode,
                    replicate_index=replicate_index,
                    aggregator=state.aggregator,
                    alpha=config.alpha,
                    sampler_name=config.sampler_name,
                    aggregation_mode=config.aggregation_mode,
                    initial_size_per_dataset=config.initial_size_per_dataset,
                    batch_size=config.batch_size,
                    verbose=config.verbose,
                )
            )
            rows.extend(replicate_rows)
            log_stage_duration(
                self.study_name,
                config.verbose,
                f"replicate {replicate_number}/{config.n_replicates} target rows",
                replicate_rows_started_at,
            )
        return rows

    def build_summary_rows(
        self,
        raw_rows: Sequence[CIWidthPrecisionCurveRow],
        state: CIWidthPrecisionCurveState,
        config: CIWidthPrecisionCurveInputConfig,
        study_config: CIWidthPrecisionCurveStudyConfig,
    ) -> list[StudySummaryRow]:
        rows: list[StudySummaryRow] = []
        schedule_keys = {
            row.requested_sample_count
            for row in raw_rows
        }
        for requested_count in sorted(
            schedule_keys,
            key=lambda item: -1 if item is None else int(item),
        ):
            schedule_rows = [
                row
                for row in raw_rows
                if row.requested_sample_count == requested_count
            ]
            rows.append(_summarize_group(study_config.benchmark_id, "all", schedule_rows))
            adjacent_rows = [
                row
                for row in schedule_rows
                if config.target_mode == "pairwise_difference"
                and row.rank == 1
            ]
            if adjacent_rows:
                rows.append(_summarize_group(study_config.benchmark_id, "adjacent_pairs", adjacent_rows))
        rows.extend(
            _summarize_milestones(
                benchmark_id=study_config.benchmark_id,
                raw_rows=raw_rows,
                target_mode=config.target_mode,
                precision_milestones=study_config.precision_milestones,
                ci_half_width_milestones=study_config.ci_half_width_milestones,
            )
        )
        return rows

    def build_narrative(
        self,
        raw_rows: Sequence[CIWidthPrecisionCurveRow],
        summary_rows: Sequence[StudySummaryRow],
        state: CIWidthPrecisionCurveState,
        config: CIWidthPrecisionCurveInputConfig,
        study_config: CIWidthPrecisionCurveStudyConfig,
    ) -> str:
        all_rows = [
            row
            for row in summary_rows
            if isinstance(row, CIWidthPrecisionCurveSummaryRow) and row.group_label == "all"
        ]
        first_row = min(all_rows, key=lambda row: row.median_effective_sample_fraction)
        last_row = max(all_rows, key=lambda row: row.median_effective_sample_fraction)
        return (
            f"CI width precision curve study completed for {study_config.benchmark_id}. "
            f"The median CI width moves from {first_row.median_ci_width:.4g} at "
            f"{first_row.median_effective_sample_fraction:.1%} of the benchmark to "
            f"{last_row.median_ci_width:.4g} at full budget."
        )

    def build_figures(
        self,
        raw_rows: Sequence[CIWidthPrecisionCurveRow],
        summary_rows: Sequence[StudySummaryRow],
        state: CIWidthPrecisionCurveState,
        config: CIWidthPrecisionCurveInputConfig,
        study_config: CIWidthPrecisionCurveStudyConfig,
    ) -> dict[str, object]:
        import matplotlib.pyplot as plt

        group_label = "adjacent_pairs" if config.pair_strategy == "adjacent" else "all"
        plot_rows = [
            row
            for row in summary_rows
            if isinstance(row, CIWidthPrecisionCurveSummaryRow) and row.group_label == group_label
        ]
        if not plot_rows:
            return {}
        figure, axis = plt.subplots(figsize=(7.5, 4.5))
        plot_rows = sorted(plot_rows, key=lambda row: row.median_samples_used)
        x_values = [row.median_samples_used for row in plot_rows]
        median_half_widths = [row.median_ci_half_width for row in plot_rows]
        p25_half_widths = [row.p25_ci_half_width for row in plot_rows]
        p75_half_widths = [row.p75_ci_half_width for row in plot_rows]
        axis.fill_between(
            x_values,
            p25_half_widths,
            p75_half_widths,
            color="#5f85c6",
            alpha=0.18,
            linewidth=0,
        )
        axis.plot(
            x_values,
            median_half_widths,
            marker="o",
            color="#2c6aa0",
            linewidth=2.0,
        )
        final_row = max(plot_rows, key=lambda row: row.median_samples_used)
        axis.axhline(
            y=final_row.median_ci_half_width,
            color="#444444",
            linestyle=":",
            linewidth=1.5,
            alpha=0.8,
        )
        x_max = max(row.median_samples_used for row in plot_rows)
        axis.text(
            0.02,
            0.98,
            f"full benchmark\n+/-{final_row.median_ci_half_width:.4g}\n{x_max:,.0f} samples",
            transform=axis.transAxes,
            color="#444444",
            fontsize=9,
            ha="left",
            va="top",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#444444", "alpha": 0.9},
        )
        milestone_rows = [
            row
            for row in summary_rows
            if isinstance(row, MilestoneSummaryRow)
            and row.group_label == group_label
            and row.median_samples_needed is not None
            and row.median_sample_fraction_needed is not None
        ]
        milestone_rows = sorted(milestone_rows, key=lambda row: float(row.median_samples_needed))
        y_max = max(row.median_ci_half_width for row in plot_rows)
        annotation_levels = [0.98, 0.86, 0.74, 0.62]
        for index, row in enumerate(milestone_rows):
            x = float(row.median_samples_needed)
            saved_percent = 100.0 * (1.0 - float(row.median_sample_fraction_needed))
            color = "#c65f5f" if row.milestone_kind == "precision" else "#5f85c6"
            if row.milestone_kind == "precision":
                label = (
                    f"{int(round(row.milestone_target * 100))}% precision "
                    f"(+/-{row.milestone_ci_half_width:.4g})"
                )
            else:
                label = (
                    f"+/-{row.milestone_target:.2g} points "
                    f"({row.milestone_percentage * 100:.0f}% precision)"
                )
            axis.axvline(x=x, color=color, linestyle="--", linewidth=1.5, alpha=0.8)
            axis.text(
                x,
                y_max * annotation_levels[index % len(annotation_levels)],
                f"{label}\n{x:,.0f} samples\nsave {saved_percent:.1f}%",
                color=color,
                fontsize=9,
                ha="left",
                va="top",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": color, "alpha": 0.9},
            )
        axis.set_title(f"{study_config.benchmark_id}: CI Width Precision Curve")
        axis.set_xlabel("Samples Used")
        axis.set_ylabel("Median CI Half-Width")
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        return {"ci_width_curve": figure}

def _select_models(bundle: BenchmarkBundle, top_k: int) -> tuple[list[str], dict[str, int]]:
    ranked_models = bundle.rank_models(complete_only=True)
    if len(ranked_models) < top_k:
        raise ValueError(f"Requested top_k={top_k}, but benchmark has only {len(ranked_models)} complete models.")
    ranked_model_ids = [model_id for model_id, _score in ranked_models]
    return ranked_model_ids[:top_k], {
        model_id: rank
        for rank, model_id in enumerate(ranked_model_ids, start=1)
    }


def _replicate_rows(
    *,
    ordered_bundle: BenchmarkBundle | PairwiseBundle,
    targets: Sequence[_PrecisionTarget],
    target_mode: TargetMode,
    replicate_index: int,
    aggregator: ScoreAggregator,
    alpha: float,
    sampler_name: SamplerName,
    aggregation_mode: AggregationMode,
    initial_size_per_dataset: int,
    batch_size: int,
    verbose: bool,
) -> list[CIWidthPrecisionCurveRow]:
    rows: list[CIWidthPrecisionCurveRow] = []
    target_label = "Model pairs" if target_mode == "pairwise_difference" else "Models"
    target_unit = "pair" if target_mode == "pairwise_difference" else "model"
    for target in progress_iterator(
        targets,
        verbose=verbose,
        desc=f"{target_label} (replicate {replicate_index + 1})",
        unit=target_unit,
    ):
        full_streams = ordered_bundle.to_streams(target.target_id)
        if target_mode == "pairwise_difference":
            assert isinstance(ordered_bundle, PairwiseBundle)
            full_dataset_sizes = ordered_bundle.dataset_sizes(target.target_id)
        else:
            assert isinstance(ordered_bundle, BenchmarkBundle)
            full_dataset_sizes = ordered_bundle.dataset_sizes_for_model(target.target_id)
        benchmark_id = ordered_bundle.benchmark_id
        benchmark_version = ordered_bundle.version
        total_available = sum(full_dataset_sizes.values())
        sampling_plan = SamplingPlan(
            dataset_sizes=full_dataset_sizes,
            initial_size_per_dataset=initial_size_per_dataset,
            batch_size=batch_size,
        )
        sampler = _make_sampler(
            sampler_name=sampler_name,
            aggregation_mode=aggregation_mode,
            streams=full_streams,
            sampling_plan=sampling_plan,
        )
        cumulative_streams: dict[str, list[ScoreLike]] = {
            dataset_id: []
            for dataset_id in full_dataset_sizes
        }
        for batch in sampler.sample_batches():
            for score in batch:
                cumulative_streams[score.dataset_id].append(score)
            dataset_sizes = {
                dataset_id: len(scores)
                for dataset_id, scores in cumulative_streams.items()
            }
            samples_used = sum(dataset_sizes.values())
            estimate = normal_ci(
                streams=cumulative_streams,
                dataset_sizes=dataset_sizes,
                aggregator=aggregator,
                alpha=alpha,
            )
            ci_width = float(estimate.ci_upper - estimate.ci_lower)
            rows.append(
                CIWidthPrecisionCurveRow(
                    benchmark_id=benchmark_id,
                    benchmark_version=benchmark_version,
                    target_mode=target_mode,
                    target_id=target.target_id,
                    rank=target.rank,
                    replicate_index=replicate_index,
                    requested_sample_count=samples_used,
                    effective_sample_fraction=float(samples_used / total_available),
                    samples_used=samples_used,
                    total_available=total_available,
                    estimate=float(estimate.estimate),
                    standard_error=float(estimate.standard_error),
                    ci_lower=float(estimate.ci_lower),
                    ci_upper=float(estimate.ci_upper),
                    ci_width=ci_width,
                    ci_half_width=float(ci_width / 2.0),
                    aggregation_mode=aggregator.aggregation_mode,
                    alpha=float(alpha),
                )
            )
    return rows


def _make_sampler(
    *,
    sampler_name: SamplerName,
    aggregation_mode: AggregationMode,
    streams,
    sampling_plan: SamplingPlan,
):
    if sampler_name == "round_robin":
        return RoundRobinSampler(streams, sampling_plan)
    if sampler_name == "neyman":
        return NeymanSampler(streams, sampling_plan, aggregation_mode)
    raise ValueError("sampler_name must be 'round_robin' or 'neyman'.")


def _summarize_group(
    benchmark_id: str,
    group_label: SummaryGroup,
    rows: Sequence[CIWidthPrecisionCurveRow],
) -> CIWidthPrecisionCurveSummaryRow:
    effective_sample_fractions = np.asarray([row.effective_sample_fraction for row in rows], dtype=float)
    ci_widths = np.asarray([row.ci_width for row in rows], dtype=float)
    ci_half_widths = np.asarray([row.ci_half_width for row in rows], dtype=float)
    samples_used = np.asarray([float(row.samples_used) for row in rows], dtype=float)
    target_ids = {row.target_id for row in rows}
    return CIWidthPrecisionCurveSummaryRow(
        benchmark_id=benchmark_id,
        group_label=group_label,
        requested_sample_count=rows[0].requested_sample_count,
        n_rows=len(rows),
        n_pairs=len(target_ids),
        n_replicates=len({row.replicate_index for row in rows}),
        median_effective_sample_fraction=float(np.percentile(effective_sample_fractions, 50)),
        median_ci_width=float(np.percentile(ci_widths, 50)),
        p25_ci_width=float(np.percentile(ci_widths, 25)),
        p75_ci_width=float(np.percentile(ci_widths, 75)),
        median_ci_half_width=float(np.percentile(ci_half_widths, 50)),
        p25_ci_half_width=float(np.percentile(ci_half_widths, 25)),
        p75_ci_half_width=float(np.percentile(ci_half_widths, 75)),
        mean_samples_used=float(np.mean(samples_used)),
        median_samples_used=float(np.percentile(samples_used, 50)),
    )


def _summarize_milestones(
    *,
    benchmark_id: str,
    raw_rows: Sequence[CIWidthPrecisionCurveRow],
    target_mode: TargetMode,
    precision_milestones: Sequence[float],
    ci_half_width_milestones: Sequence[float],
) -> list[MilestoneSummaryRow]:
    rows: list[MilestoneSummaryRow] = []
    grouped_rows = _group_rows_by_target_and_replicate(raw_rows)
    rows.extend(
        _milestone_rows_for_group(
            benchmark_id=benchmark_id,
            group_label="all",
            grouped_rows=grouped_rows,
            precision_milestones=precision_milestones,
            ci_half_width_milestones=ci_half_width_milestones,
        )
    )
    if target_mode == "pairwise_difference":
        adjacent_grouped_rows = {
            key: target_rows
            for key, target_rows in grouped_rows.items()
            if target_rows[0].rank == 1
        }
        if adjacent_grouped_rows:
            rows.extend(
                _milestone_rows_for_group(
                    benchmark_id=benchmark_id,
                    group_label="adjacent_pairs",
                    grouped_rows=adjacent_grouped_rows,
                    precision_milestones=precision_milestones,
                    ci_half_width_milestones=ci_half_width_milestones,
                )
            )
    return rows


def _group_rows_by_target_and_replicate(
    raw_rows: Sequence[CIWidthPrecisionCurveRow],
) -> dict[tuple[str, int], list[CIWidthPrecisionCurveRow]]:
    grouped: dict[tuple[str, int], list[CIWidthPrecisionCurveRow]] = {}
    for row in raw_rows:
        grouped.setdefault((row.target_id, row.replicate_index), []).append(row)
    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda row: row.samples_used)
    return grouped


def _milestone_rows_for_group(
    *,
    benchmark_id: str,
    group_label: SummaryGroup,
    grouped_rows: dict[tuple[str, int], list[CIWidthPrecisionCurveRow]],
    precision_milestones: Sequence[float],
    ci_half_width_milestones: Sequence[float],
) -> list[MilestoneSummaryRow]:
    rows: list[MilestoneSummaryRow] = []
    for milestone in precision_milestones:
        rows.append(
            _build_milestone_summary_row(
                benchmark_id=benchmark_id,
                group_label=group_label,
                milestone_kind="precision",
                milestone_target=float(milestone),
                resolved_pairs=[
                    _first_precision_milestone_pair(target_rows, float(milestone))
                    for target_rows in grouped_rows.values()
                ],
            )
        )
    for milestone in ci_half_width_milestones:
        rows.append(
            _build_milestone_summary_row(
                benchmark_id=benchmark_id,
                group_label=group_label,
                milestone_kind="ci_half_width",
                milestone_target=float(milestone),
                resolved_pairs=[
                    _first_ci_half_width_milestone_pair(target_rows, float(milestone))
                    for target_rows in grouped_rows.values()
                ],
            )
        )
    return rows


def _first_precision_milestone_pair(
    rows: Sequence[CIWidthPrecisionCurveRow],
    milestone: float,
) -> tuple[CIWidthPrecisionCurveRow, CIWidthPrecisionCurveRow] | None:
    final_row = rows[-1]
    final_ci_width = final_row.ci_width
    for row in rows:
        achieved_precision_fraction = final_ci_width / row.ci_width
        if achieved_precision_fraction >= milestone:
            return row, final_row
    return None


def _first_ci_half_width_milestone_pair(
    rows: Sequence[CIWidthPrecisionCurveRow],
    milestone: float,
 ) -> tuple[CIWidthPrecisionCurveRow, CIWidthPrecisionCurveRow] | None:
    final_row = rows[-1]
    for row in rows:
        if row.ci_half_width <= milestone:
            return row, final_row
    return None


def _build_milestone_summary_row(
    *,
    benchmark_id: str,
    group_label: SummaryGroup,
    milestone_kind: MilestoneKind,
    milestone_target: float,
    resolved_pairs: Sequence[tuple[CIWidthPrecisionCurveRow, CIWidthPrecisionCurveRow] | None],
) -> MilestoneSummaryRow:
    resolved = [pair for pair in resolved_pairs if pair is not None]
    if not resolved:
        return MilestoneSummaryRow(
            benchmark_id=benchmark_id,
            group_label=group_label,
            milestone_kind=milestone_kind,
            milestone_target=milestone_target,
            milestone_percentage=float("nan"),
            milestone_ci_half_width=float("nan"),
            median_samples_needed=None,
            median_sample_fraction_needed=None,
        )

    achieved_precision_fractions = np.asarray(
        [
            _achieved_precision_fraction(row, final_row)
            for row, final_row in resolved
        ],
        dtype=float,
    )
    achieved_ci_half_widths = np.asarray([row.ci_half_width for row, _final_row in resolved], dtype=float)
    samples_needed = np.asarray([float(row.samples_used) for row, _final_row in resolved], dtype=float)
    sample_fractions_needed = np.asarray(
        [row.effective_sample_fraction for row, _final_row in resolved],
        dtype=float,
    )
    return MilestoneSummaryRow(
        benchmark_id=benchmark_id,
        group_label=group_label,
        milestone_kind=milestone_kind,
        milestone_target=milestone_target,
        milestone_percentage=float(np.percentile(achieved_precision_fractions, 50)),
        milestone_ci_half_width=float(np.percentile(achieved_ci_half_widths, 50)),
        median_samples_needed=float(np.percentile(samples_needed, 50)),
        median_sample_fraction_needed=float(np.percentile(sample_fractions_needed, 50)),
    )


def _achieved_precision_fraction(
    row: CIWidthPrecisionCurveRow,
    final_row: CIWidthPrecisionCurveRow,
) -> float:
    return float(final_row.ci_width / row.ci_width)


def build_sampler_comparison_figure(
    round_robin_result: CaseStudyResult,
    neyman_result: CaseStudyResult,
) -> object | None:
    import matplotlib.pyplot as plt

    group_label = _group_label_for_result(round_robin_result)
    rr_rows = _curve_summary_rows(round_robin_result.summary_rows, group_label=group_label)
    ny_rows = _curve_summary_rows(neyman_result.summary_rows, group_label=group_label)
    if not rr_rows or not ny_rows:
        return None

    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    _plot_summary_curve(axis, rr_rows, label="round robin", color="#2c6aa0")
    _plot_summary_curve(axis, ny_rows, label="Neyman", color="#c65f5f")
    axis.set_title(f"{round_robin_result.dataset_label}: Round Robin vs Neyman")
    axis.set_xlabel("Samples Used")
    axis.set_ylabel("Median CI Half-Width")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    return figure


def build_samples_saved_vs_milestone_figure(
    round_robin_result: CaseStudyResult,
    neyman_result: CaseStudyResult,
) -> object | None:
    import matplotlib.pyplot as plt

    group_label = _group_label_for_result(round_robin_result)
    rr_milestones = _milestone_summary_rows(round_robin_result.summary_rows, group_label=group_label)
    ny_milestones = _milestone_summary_rows(neyman_result.summary_rows, group_label=group_label)
    if not rr_milestones or not ny_milestones:
        return None

    figure, (precision_axis, ci_axis) = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    _plot_saved_vs_milestones(
        precision_axis,
        rr_milestones,
        label="round robin",
        color="#2c6aa0",
        milestone_kind="precision",
    )
    _plot_saved_vs_milestones(
        precision_axis,
        ny_milestones,
        label="Neyman",
        color="#c65f5f",
        milestone_kind="precision",
    )
    _plot_saved_vs_milestones(
        ci_axis,
        rr_milestones,
        label="round robin",
        color="#2c6aa0",
        milestone_kind="ci_half_width",
    )
    _plot_saved_vs_milestones(
        ci_axis,
        ny_milestones,
        label="Neyman",
        color="#c65f5f",
        milestone_kind="ci_half_width",
    )
    precision_axis.set_title("Precision milestones")
    precision_axis.set_xlabel("Milestone target")
    precision_axis.set_ylabel("Saved samples (%)")
    precision_axis.set_ylim(0.0, 100.0)
    precision_axis.grid(True, alpha=0.3)
    ci_axis.set_title("CI half-width milestones")
    ci_axis.set_xlabel("Milestone target")
    ci_axis.grid(True, alpha=0.3)
    ci_axis.legend()
    precision_axis.legend()
    figure.suptitle(f"{round_robin_result.dataset_label}: Samples Saved vs Milestone")
    figure.tight_layout()
    return figure


def build_precision_saved_figure(
    round_robin_result: CaseStudyResult,
    neyman_result: CaseStudyResult,
) -> object | None:
    import matplotlib.pyplot as plt

    group_label = _group_label_for_result(round_robin_result)
    rr_milestones = _milestone_summary_rows(round_robin_result.summary_rows, group_label=group_label)
    ny_milestones = _milestone_summary_rows(neyman_result.summary_rows, group_label=group_label)
    if not rr_milestones or not ny_milestones:
        return None

    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    _plot_saved_vs_milestones(
        axis,
        rr_milestones,
        label="round robin",
        color="#2c6aa0",
        milestone_kind="precision",
    )
    _plot_saved_vs_milestones(
        axis,
        ny_milestones,
        label="Neyman",
        color="#c65f5f",
        milestone_kind="precision",
    )
    axis.set_title(f"{round_robin_result.dataset_label}: Precision Milestones vs Samples Saved")
    axis.set_xlabel("Precision milestone")
    axis.set_ylabel("Saved samples (%)")
    axis.set_ylim(0.0, 100.0)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure


def _group_label_for_result(result: CaseStudyResult) -> str:
    return "adjacent_pairs" if result.config.get("pair_strategy") == "adjacent" else "all"


def _curve_summary_rows(summary_rows: Sequence[Mapping[str, Any]], *, group_label: str) -> list[Mapping[str, Any]]:
    rows = [
        row
        for row in summary_rows
        if row.get("group_label") == group_label and "median_samples_used" in row
    ]
    return sorted(rows, key=lambda row: float(row["median_samples_used"]))


def _milestone_summary_rows(summary_rows: Sequence[Mapping[str, Any]], *, group_label: str) -> list[Mapping[str, Any]]:
    rows = [
        row
        for row in summary_rows
        if row.get("group_label") == group_label and row.get("milestone_kind")
    ]
    return sorted(rows, key=lambda row: (row["milestone_kind"], float(row["milestone_target"])))


def _plot_summary_curve(
    axis,
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
    color: str,
) -> None:
    x_values = [float(row["median_samples_used"]) for row in rows]
    median_half_widths = [float(row["median_ci_half_width"]) for row in rows]
    p25_half_widths = [float(row["p25_ci_half_width"]) for row in rows]
    p75_half_widths = [float(row["p75_ci_half_width"]) for row in rows]
    axis.fill_between(x_values, p25_half_widths, p75_half_widths, color=color, alpha=0.12, linewidth=0)
    axis.plot(x_values, median_half_widths, marker="o", color=color, linewidth=2.0, label=label)


def _plot_saved_vs_milestones(
    axis,
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
    color: str,
    milestone_kind: MilestoneKind,
) -> None:
    filtered_rows = [row for row in rows if row.get("milestone_kind") == milestone_kind]
    if not filtered_rows:
        return
    x_values = [float(row["milestone_target"]) for row in filtered_rows]
    saved_values = [
        100.0 * (1.0 - float(row["median_sample_fraction_needed"]))
        for row in filtered_rows
        if row.get("median_sample_fraction_needed") not in {None, ""}
    ]
    if len(saved_values) != len(x_values):
        return
    axis.plot(x_values, saved_values, marker="o", color=color, linewidth=2.0, label=label)


if __name__ == "__main__":
    from case_studies.common.artifacts import save_case_study_result
    from data.registry import get_benchmark

    round_robin_result = CIWidthPrecisionCurveCaseStudy().run(
        get_benchmark("open_vlm_leaderboard"),
        CIWidthPrecisionCurveInputConfig(
            target_mode="pairwise_difference",
            sampler_name="round_robin",
            top_k=50,
            n_replicates=1,
            batch_size=100,
            aggregation_mode="pooled_mean",
            precision_milestones=[0.8, 0.9],
            ci_half_width_milestones=[0.02],
            pair_strategy="all",
            alpha=0.05
        ),
    )
    save_case_study_result(round_robin_result)
    exit(0)
    neyman_result = CIWidthPrecisionCurveCaseStudy().run(
        get_benchmark("open_vlm_leaderboard"),
        CIWidthPrecisionCurveInputConfig(
            target_mode="pairwise_difference",
            sampler_name="neyman",
            top_k=10,
            n_replicates=10,
            batch_size=100,
            aggregation_mode="mean_of_means",
            precision_milestones=[0.8, 0.9],
            ci_half_width_milestones=[0.02],
            pair_strategy="all",
            alpha=0.05
        ),
    )
    combined_figures = dict(round_robin_result.figures)
    comparison_figure = build_sampler_comparison_figure(round_robin_result, neyman_result)
    if comparison_figure is not None:
        combined_figures["sampler_comparison_curve"] = comparison_figure
    saved_vs_milestone_figure = build_samples_saved_vs_milestone_figure(round_robin_result, neyman_result)
    if saved_vs_milestone_figure is not None:
        combined_figures["samples_saved_vs_milestone"] = saved_vs_milestone_figure
    precision_saved_figure = build_precision_saved_figure(round_robin_result, neyman_result)
    if precision_saved_figure is not None:
        combined_figures["precision_saved_vs_milestone"] = precision_saved_figure

    combined_result = replace(round_robin_result, figures=combined_figures)
    paths = save_case_study_result(combined_result, run_id="top_10_all_round_robin_vs_neyman")
