from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from case_studies.ci_width_precision_curve import (
    CIWidthPrecisionCurveCaseStudy,
    CIWidthPrecisionCurveInputConfig,
    CIWidthPrecisionCurveState,
    CIWidthPrecisionCurveStudyConfig,
    CIWidthPrecisionCurveSummaryRow,
    MilestoneSummaryRow,
    StudySummaryRow,
    TargetMode,
)
from case_studies.common.utils import make_aggregator, replicate_iterator
from data.core import BenchmarkBundle, PairwiseBundle
from data.views import OrderedBenchmarkView, PairwiseOrderedBenchmarkView
from sequential import RgsDesignSequentialEngine
from sequential.engines import SequentialEngine
from sequential.samplers import NeymanSampler, RoundRobinSampler, SamplingPlan
from sequential.types import AggregationMode, LookResult, SequentialTestType, SpendingFunction


EngineName = Literal["r_gsdesign"]


def _make_sampler(
    sampler_name: str,
    aggregation_mode: AggregationMode,
    streams,
    sampling_plan: SamplingPlan,
):
    if sampler_name == "round_robin":
        return RoundRobinSampler(streams, sampling_plan)
    if sampler_name == "neyman":
        return NeymanSampler(streams, sampling_plan, aggregation_mode)
    raise ValueError("sampler_name must be 'round_robin' or 'neyman'.")


@dataclass(frozen=True)
class SequentialPrecisionCurveInputConfig(CIWidthPrecisionCurveInputConfig):
    beta: float = 0.2
    engine_name: EngineName = "r_gsdesign"
    spending_function: SpendingFunction = "pocock"


@dataclass(frozen=True)
class SequentialPrecisionCurveStudyConfig(CIWidthPrecisionCurveStudyConfig):
    beta: float = 0.2
    engine_name: EngineName = "r_gsdesign"
    spending_function: SpendingFunction = "pocock"


@dataclass(frozen=True)
class SequentialPrecisionCurveRow:
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
    beta: float
    engine_name: EngineName
    spending_function: SpendingFunction
    look_index: int
    z_stat: float | None = None
    upper_boundary: float | None = None
    lower_boundary: float | None = None


class SequentialPrecisionCurveCaseStudy(
    CIWidthPrecisionCurveCaseStudy,
):
    study_name = "sequential_precision_curve"

    def validate_input_config(self, config: SequentialPrecisionCurveInputConfig) -> None:
        super().validate_input_config(config)
        if not (0.0 < config.beta < 1.0):
            raise ValueError("beta must be in the open interval (0, 1).")
        if config.engine_name != "r_gsdesign":
            raise ValueError("engine_name must be 'r_gsdesign'.")
        if config.spending_function not in {"pocock", "obrien_fleming", "wang_tsiatis"}:
            raise ValueError(
                "spending_function must be 'pocock', 'obrien_fleming', or 'wang_tsiatis'."
            )

    def build_study_config(
        self,
        bundle: BenchmarkBundle,
        config: SequentialPrecisionCurveInputConfig,
        state: CIWidthPrecisionCurveState,
    ) -> SequentialPrecisionCurveStudyConfig:
        return SequentialPrecisionCurveStudyConfig(
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
            beta=config.beta,
            engine_name=config.engine_name,
            spending_function=config.spending_function,
        )

    def build_raw_rows(
        self,
        state: CIWidthPrecisionCurveState,
        config: SequentialPrecisionCurveInputConfig,
        study_config: SequentialPrecisionCurveStudyConfig,
    ) -> list[SequentialPrecisionCurveRow]:
        rows: list[SequentialPrecisionCurveRow] = []
        for replicate_index in replicate_iterator(config.n_replicates, config.verbose):
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
            engine = _make_engine(
                alpha=config.alpha,
                beta=config.beta,
                aggregation_mode=config.aggregation_mode,
                spending_function=config.spending_function,
            )
            rows.extend(
                _replicate_rows_sequential(
                    ordered_bundle=ordered_bundle,
                    targets=state.targets,
                    target_mode=config.target_mode,
                    replicate_index=replicate_index,
                    config=study_config,
                    engine=engine,
                    initial_size_per_dataset=config.initial_size_per_dataset,
                    batch_size=config.batch_size,
                    sampler_name=config.sampler_name,
                )
            )
        return rows

    def build_narrative(
        self,
        raw_rows: Sequence[SequentialPrecisionCurveRow],
        summary_rows: Sequence[StudySummaryRow],
        state: CIWidthPrecisionCurveState,
        config: SequentialPrecisionCurveInputConfig,
        study_config: SequentialPrecisionCurveStudyConfig,
    ) -> str:
        all_rows = [
            row
            for row in summary_rows
            if isinstance(row, CIWidthPrecisionCurveSummaryRow) and row.group_label == "all"
        ]
        first_row = min(all_rows, key=lambda row: row.median_effective_sample_fraction)
        last_row = max(all_rows, key=lambda row: row.median_effective_sample_fraction)
        return (
            f"Sequential precision curve study completed for {study_config.benchmark_id}. "
            f"The median sequential CI half-width moves from {first_row.median_ci_half_width:.4g} at "
            f"{first_row.median_effective_sample_fraction:.1%} of the benchmark to "
            f"{last_row.median_ci_half_width:.4g} at full budget."
        )

    def build_figures(
        self,
        raw_rows: Sequence[SequentialPrecisionCurveRow],
        summary_rows: Sequence[StudySummaryRow],
        state: CIWidthPrecisionCurveState,
        config: SequentialPrecisionCurveInputConfig,
        study_config: SequentialPrecisionCurveStudyConfig,
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
        axis.plot(
            [row.median_samples_used for row in plot_rows],
            [row.median_ci_half_width for row in plot_rows],
            marker="o",
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
            x_max,
            final_row.median_ci_half_width,
            f"full benchmark: +/-{final_row.median_ci_half_width:.4g} ({x_max:,.0f} samples)",
            color="#444444",
            fontsize=9,
            ha="right",
            va="bottom",
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
        axis.set_title(f"{study_config.benchmark_id}: Sequential Precision Curve")
        axis.set_xlabel("Samples Used")
        axis.set_ylabel("Median Sequential CI Half-Width")
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        return {"sequential_precision_curve": figure}


def _make_engine(
    *,
    alpha: float,
    beta: float,
        aggregation_mode: AggregationMode,
        spending_function: SpendingFunction,
) -> SequentialEngine:
    return RgsDesignSequentialEngine(
        alpha=alpha,
        beta=beta,
        aggregator=make_aggregator(aggregation_mode),
        test_type=SequentialTestType.TWO_SIDED,
        spending_function=spending_function,
        timing_mode="planned_information",
    )


def _replicate_rows_sequential(
    *,
    ordered_bundle: BenchmarkBundle | PairwiseBundle,
    targets: Sequence[object],
    target_mode: TargetMode,
    replicate_index: int,
    config: SequentialPrecisionCurveStudyConfig,
    engine: SequentialEngine,
    initial_size_per_dataset: int,
    batch_size: int,
    sampler_name: str,
) -> list[SequentialPrecisionCurveRow]:
    rows: list[SequentialPrecisionCurveRow] = []
    for target in targets:
        target_id = getattr(target, "target_id")
        rank = getattr(target, "rank")
        full_streams = ordered_bundle.to_streams(target_id)
        if target_mode == "pairwise_difference":
            assert isinstance(ordered_bundle, PairwiseBundle)
            full_dataset_sizes = ordered_bundle.dataset_sizes(target_id)
        else:
            assert isinstance(ordered_bundle, BenchmarkBundle)
            full_dataset_sizes = ordered_bundle.dataset_sizes_for_model(target_id)
        sampling_plan = SamplingPlan(
            dataset_sizes=full_dataset_sizes,
            initial_size_per_dataset=initial_size_per_dataset,
            batch_size=batch_size,
        )
        sampler = _make_sampler(sampler_name, config.aggregation_mode, full_streams, sampling_plan)
        try:
            for look in engine.iter_looks(sampler=sampler):
                rows.append(_row_from_look(ordered_bundle, target_mode, target_id, rank, replicate_index, config, look))
        except Exception as exc:
            raise ValueError(
                "Sequential precision curve failed while building the gsDesign look trace. "
                f"target_id={target_id!r}, replicate_index={replicate_index}, "
                f"total_available={sampling_plan.total_available}, "
                f"initial_size_per_dataset={initial_size_per_dataset}, "
                f"batch_size={batch_size}, "
                f"planned_looks={len(sampling_plan.cumulative_sample_sizes())}. "
                "The study uses a pilot-frozen planned-information timing schedule; if gsDesign "
                "still rejects this run, inspect the target-level benchmark path and variance profile."
            ) from exc
    return rows


def _row_from_look(
    ordered_bundle: BenchmarkBundle | PairwiseBundle,
    target_mode: TargetMode,
    target_id: str,
    rank: int,
    replicate_index: int,
    config: SequentialPrecisionCurveStudyConfig,
    look: LookResult,
) -> SequentialPrecisionCurveRow:
    standard_error = float("nan")
    if look.ci_upper > look.ci_lower and 0.0 < config.alpha < 1.0:
        z_value = _two_sided_z_value(config.alpha)
        if z_value > 0:
            standard_error = look.halfwidth / z_value
    return SequentialPrecisionCurveRow(
        benchmark_id=ordered_bundle.benchmark_id,
        benchmark_version=ordered_bundle.version,
        target_mode=target_mode,
        target_id=target_id,
        rank=rank,
        replicate_index=replicate_index,
        requested_sample_count=look.n_seen,
        effective_sample_fraction=float(look.n_seen / look.total_examples),
        samples_used=look.n_seen,
        total_available=look.total_examples,
        estimate=float(look.estimate),
        standard_error=standard_error,
        ci_lower=float(look.ci_lower),
        ci_upper=float(look.ci_upper),
        ci_width=float(look.ci_upper - look.ci_lower),
        ci_half_width=float(look.halfwidth),
        aggregation_mode=config.aggregation_mode,
        alpha=float(config.alpha),
        beta=float(config.beta),
        engine_name=config.engine_name,
        spending_function=config.spending_function,
        look_index=look.look_index,
        z_stat=look.z_stat,
        upper_boundary=look.upper_boundary,
        lower_boundary=look.lower_boundary,
    )


def _two_sided_z_value(alpha: float) -> float:
    try:
        from scipy.stats import norm
    except Exception:
        return float("nan")
    return float(norm.ppf(1.0 - alpha / 2.0))


if __name__ == "__main__":
    from case_studies.common.artifacts import save_case_study_result
    from data.registry import get_benchmark

    result = SequentialPrecisionCurveCaseStudy().run(
        get_benchmark("swe_bench_verified"),
        SequentialPrecisionCurveInputConfig(
            target_mode="pairwise_difference",
            sampler_name="round_robin",
            top_k=50,
            n_replicates=10,
            batch_size=100,
            aggregation_mode="pooled_mean",
            precision_milestones=[0.8, 0.9],
            ci_half_width_milestones=[0.02],
            pair_strategy="all",
            alpha=0.05
        ),
    )
    paths = save_case_study_result(result)
