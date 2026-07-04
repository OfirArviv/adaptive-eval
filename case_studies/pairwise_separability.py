from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Literal, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from case_studies.common.base import BaseCaseStudy
from case_studies.common.utils import (
    make_aggregator,
    progress_iterator,
    replicate_iterator,
    select_model_pairs,
)
from case_studies.common.schemas import CaseStudyResult
from data.benchmark import Benchmark
from data.registry import get_benchmark
from data.views import MissingPolicy, PairwiseBenchmarkView, PairwiseOrderedBenchmarkView
from fixed_sample import FixedSampleEstimate, normal_ci
from sequential import (
    EfficacyRule,
    EquivalenceRule,
    FutilityRule,
    NeymanSampler,
    SamplingPlan,
    SimpleMeanOfMeansGsDesignEngine,
    run_sequential_test,
)
from sequential.aggregators import ScoreAggregator
from sequential.types import AggregationMode, SpendingFunction

FullBaselineLabel = Literal["separable", "equivalent_under_mdes", "unresolved"]
PairwiseSeparabilityGroup = Literal[
    "all",
    "separable",
    "non_separable",
    "equivalent_under_mdes",
    "unresolved",
    "separable_positive_diff",
    "separable_negative_diff",
    "non_separable_positive_diff",
    "non_separable_negative_diff",
    "separable_over_futility_mdes",
    "separable_under_futility_mdes",
    "non_separable_over_futility_mdes",
    "non_separable_under_futility_mdes",
]


@dataclass(frozen=True)
class PairwiseSeparabilityInputConfig:
    initial_look_per_dataset: int
    batch_size: int
    top_k: int = 10
    n_replicates: int = 100
    equivalence_margin: float = 0.02
    futility_mdes: float = 0.005
    alpha: float = 0.05
    beta: float = 0.1
    aggregation_mode: AggregationMode = "mean_of_means"
    spending_function: SpendingFunction = "pocock"
    futility_spending_function: SpendingFunction = "pocock"
    enable_futility: bool = False
    missing_policy: MissingPolicy = "error"
    use_cache: bool = True
    verbose: bool = True


@dataclass(frozen=True)
class PairwiseSeparabilityStudyConfig:
    benchmark_id: str
    benchmark_version: str
    top_k: int
    n_replicates: int
    equivalence_margin: float
    futility_mdes: float
    alpha: float
    beta: float
    aggregation_mode: AggregationMode
    initial_look_per_dataset: int
    batch_size: int
    spending_function: SpendingFunction
    futility_spending_function: SpendingFunction
    enable_futility: bool
    missing_policy: MissingPolicy
    use_cache: bool
    verbose: bool
    selected_model_ids: list[str]
    pair_ids: list[str]


@dataclass(frozen=True)
class PairwiseSeparabilityState:
    ordered_pairs: list[tuple[str, str]]
    pair_ids: list[str]
    selected_model_ids: list[str]
    aggregator: ScoreAggregator
    fixed_baselines: dict[str, FixedSampleEstimate]
    leaderboard_gaps: dict[str, float]
    benchmark: Benchmark


@dataclass(frozen=True)
class PairwiseSeparabilityRawRow:
    benchmark_id: str
    benchmark_version: str
    model_a: str
    model_b: str
    pair_id: str
    leaderboard_gap: float
    replicate_index: int
    ordering_seed: int
    mdes: float
    equivalence_margin: float
    futility_mdes: float
    full_diff: float
    full_ci_lower: float
    full_ci_upper: float
    full_baseline_label: FullBaselineLabel
    sequential_stop_reason: str
    samples_used: int
    total_available: int
    sample_fraction: float
    stop_look: int
    total_planned_looks: int
    sequential_estimate: float
    sequential_ci_lower: float
    sequential_ci_upper: float
    agreement_with_full_baseline: bool
    aggregation_mode: AggregationMode


@dataclass(frozen=True)
class PairwiseSeparabilitySummaryRow:
    benchmark_id: str
    mdes: float
    group_label: PairwiseSeparabilityGroup
    n_runs: int
    n_pairs: int
    efficacy_count: int
    equivalence_count: int
    futility_count: int
    max_samples_count: int
    efficacy_rate: float
    equivalence_rate: float
    futility_rate: float
    max_samples_rate: float
    efficacy_correct_rate: float
    equivalence_correct_rate: float
    futility_correct_rate: float
    max_samples_correct_rate: float
    mean_sample_fraction: float
    median_sample_fraction: float
    p25_sample_fraction: float
    p75_sample_fraction: float
    sequential_full_baseline_agreement_rate: float
    full_separable_rate: float
    full_equivalent_rate: float
    full_unresolved_rate: float


class PairwiseSeparabilityCaseStudy(
    BaseCaseStudy[
        PairwiseSeparabilityInputConfig,
        PairwiseSeparabilityStudyConfig,
        PairwiseSeparabilityState,
        PairwiseSeparabilityRawRow,
        PairwiseSeparabilitySummaryRow,
    ]
):
    study_name = "pairwise_separability"

    def validate_input_config(self, config: PairwiseSeparabilityInputConfig) -> None:
        if config.top_k < 2:
            raise ValueError("top_k must be at least 2.")
        if config.n_replicates <= 0:
            raise ValueError("n_replicates must be positive.")
        if config.equivalence_margin <= 0.0:
            raise ValueError("equivalence_margin must be positive.")
        if config.futility_mdes <= 0.0:
            raise ValueError("futility_mdes must be positive.")
        if not (0.0 < config.alpha < 1.0):
            raise ValueError("alpha must be in the open interval (0, 1).")
        if not (0.0 < config.beta < 1.0):
            raise ValueError("beta must be in the open interval (0, 1).")
        if config.aggregation_mode != "mean_of_means":
            raise ValueError("Simple gsDesign pairwise separability currently requires aggregation_mode='mean_of_means'.")
        if config.initial_look_per_dataset <= 0:
            raise ValueError("initial_look_per_dataset must be positive.")
        if config.initial_look_per_dataset < NeymanSampler.MIN_INITIAL_SAMPLES_PER_DATASET:
            raise ValueError(
                "Simple gsDesign pairwise separability uses NeymanSampler, so "
                "initial_look_per_dataset must be at least "
                f"{NeymanSampler.MIN_INITIAL_SAMPLES_PER_DATASET}."
            )
        if config.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

    def build_state(
        self,
        bundle,
        benchmark: Benchmark,
        config: PairwiseSeparabilityInputConfig,
    ) -> PairwiseSeparabilityState:
        selection = select_model_pairs(bundle, top_k=config.top_k, pair_strategy="all")
        ordered_pairs = [(pair.model_a, pair.model_b) for pair in selection.pairs]
        aggregator = make_aggregator(config.aggregation_mode)
        pairwise = PairwiseBenchmarkView(
            benchmark=benchmark,
            pairs=ordered_pairs,
            missing_policy=config.missing_policy,
        ).load(use_cache=config.use_cache)
        fixed_baselines = {
            pair_id: normal_ci(
                streams=pairwise.to_streams(pair_id),
                dataset_sizes=pairwise.dataset_sizes(pair_id),
                aggregator=aggregator,
                alpha=config.alpha,
            )
            for pair_id in pairwise.pair_ids
        }
        ranking_scores = dict(bundle.rank_models(complete_only=True))
        leaderboard_gaps = {
            f"{model_a};{model_b}": abs(ranking_scores[model_a] - ranking_scores[model_b])
            for model_a, model_b in ordered_pairs
        }
        return PairwiseSeparabilityState(
            ordered_pairs=ordered_pairs,
            pair_ids=[pair.pair_id for pair in selection.pairs],
            selected_model_ids=selection.selected_model_ids,
            aggregator=aggregator,
            fixed_baselines=fixed_baselines,
            leaderboard_gaps=leaderboard_gaps,
            benchmark=benchmark,
        )

    def build_study_config(
        self,
        bundle,
        config: PairwiseSeparabilityInputConfig,
        state: PairwiseSeparabilityState,
    ) -> PairwiseSeparabilityStudyConfig:
        return PairwiseSeparabilityStudyConfig(
            benchmark_id=bundle.benchmark_id,
            benchmark_version=bundle.version,
            top_k=config.top_k,
            n_replicates=config.n_replicates,
            equivalence_margin=float(config.equivalence_margin),
            futility_mdes=float(config.futility_mdes),
            alpha=config.alpha,
            beta=config.beta,
            aggregation_mode=config.aggregation_mode,
            initial_look_per_dataset=config.initial_look_per_dataset,
            batch_size=config.batch_size,
            spending_function=config.spending_function,
            futility_spending_function=config.futility_spending_function,
            enable_futility=config.enable_futility,
            missing_policy=config.missing_policy,
            use_cache=config.use_cache,
            verbose=config.verbose,
            selected_model_ids=state.selected_model_ids,
            pair_ids=state.pair_ids,
        )

    def build_raw_rows(
        self,
        state: PairwiseSeparabilityState,
        config: PairwiseSeparabilityInputConfig,
        study_config: PairwiseSeparabilityStudyConfig,
    ) -> list[PairwiseSeparabilityRawRow]:
        rows: list[PairwiseSeparabilityRawRow] = []
        for replicate_index in replicate_iterator(config.n_replicates, config.verbose):
            ordered_pairwise = PairwiseOrderedBenchmarkView(
                benchmark=state.benchmark,
                pairs=state.ordered_pairs,
                seed=replicate_index,
                missing_policy=config.missing_policy,
            ).load(use_cache=config.use_cache)
            pair_iterator = progress_iterator(
                study_config.pair_ids,
                verbose=config.verbose,
                desc=f"Replicate {replicate_index + 1}/{config.n_replicates} pairs",
                unit="pair",
            )
            replicate_rows_before = len(rows)
            for pair_id in pair_iterator:
                model_a, model_b = pair_id.split(";", maxsplit=1)
                rows.extend(
                    _run_pair_replicate(
                        benchmark_id=study_config.benchmark_id,
                        benchmark_version=study_config.benchmark_version,
                        pair_id=pair_id,
                        model_a=model_a,
                        model_b=model_b,
                        replicate_index=replicate_index,
                        ordered_pairwise=ordered_pairwise,
                        fixed_baseline=state.fixed_baselines[pair_id],
                        leaderboard_gap=state.leaderboard_gaps[pair_id],
                        input_config=config,
                        aggregator=state.aggregator,
                    )
                )
        return rows

    def build_summary_rows(
        self,
        raw_rows: Sequence[PairwiseSeparabilityRawRow],
        state: PairwiseSeparabilityState,
        config: PairwiseSeparabilityInputConfig,
        study_config: PairwiseSeparabilityStudyConfig,
    ) -> list[PairwiseSeparabilitySummaryRow]:
        summaries: list[PairwiseSeparabilitySummaryRow] = []
        for mdes in sorted({row.mdes for row in raw_rows}):
            mdes_rows = [row for row in raw_rows if row.mdes == mdes]
            summaries.append(_summarize_group(study_config.benchmark_id, mdes, "all", mdes_rows))
            non_separable_rows = [row for row in mdes_rows if row.full_baseline_label != "separable"]
            if non_separable_rows:
                summaries.append(_summarize_group(study_config.benchmark_id, mdes, "non_separable", non_separable_rows))
            for label in ("separable", "equivalent_under_mdes", "unresolved"):
                label_rows = [row for row in mdes_rows if row.full_baseline_label == label]
                if label_rows:
                    summaries.append(_summarize_group(study_config.benchmark_id, mdes, label, label_rows))
            direction_groups = [
                ("separable_positive_diff", _is_separable_and_positive_diff),
                ("separable_negative_diff", _is_separable_and_negative_diff),
                ("non_separable_positive_diff", _is_non_separable_and_positive_diff),
                ("non_separable_negative_diff", _is_non_separable_and_negative_diff),
            ]
            for group_label, predicate in direction_groups:
                group_rows = [row for row in mdes_rows if predicate(row)]
                if group_rows:
                    summaries.append(_summarize_group(study_config.benchmark_id, mdes, group_label, group_rows))
            split_groups = [
                ("separable_over_futility_mdes", _is_separable_and_over_futility_mdes),
                ("separable_under_futility_mdes", _is_separable_and_under_futility_mdes),
                ("non_separable_over_futility_mdes", _is_non_separable_and_over_futility_mdes),
                ("non_separable_under_futility_mdes", _is_non_separable_and_under_futility_mdes),
            ]
            for group_label, predicate in split_groups:
                group_rows = [row for row in mdes_rows if predicate(row)]
                if group_rows:
                    summaries.append(_summarize_group(study_config.benchmark_id, mdes, group_label, group_rows))
        return summaries

    def build_narrative(
        self,
        raw_rows: Sequence[PairwiseSeparabilityRawRow],
        summary_rows: Sequence[PairwiseSeparabilitySummaryRow],
        state: PairwiseSeparabilityState,
        config: PairwiseSeparabilityInputConfig,
        study_config: PairwiseSeparabilityStudyConfig,
    ) -> str:
        lines = [f"Pairwise separability study completed for {study_config.benchmark_id}."]
        for row in summary_rows:
            if row.group_label != "all":
                continue
            resolved_rate = row.efficacy_rate + row.equivalence_rate + row.futility_rate
            saved_fraction = 1.0 - row.mean_sample_fraction
            lines.append(
                (
                    f"At MDES={row.mdes:.4g}, {row.n_pairs} pairs were evaluated "
                    f"across {row.n_runs} sequential runs. Sequential stopping resolved "
                    f"{resolved_rate:.1%} with mean sample savings of {saved_fraction:.1%}."
                )
            )
        return "\n\n".join(lines)

    def build_figures(
        self,
        raw_rows: Sequence[PairwiseSeparabilityRawRow],
        summary_rows: Sequence[PairwiseSeparabilitySummaryRow],
        state: PairwiseSeparabilityState,
        config: PairwiseSeparabilityInputConfig,
        study_config: PairwiseSeparabilityStudyConfig,
    ) -> dict[str, object]:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(7.5, 4.5))
        color_by_reason = {
            "efficacy": "#1f77b4",
            "equivalence": "#ff7f0e",
            "futility": "#ff7f0e",
            "max_samples": "#7f7f7f",
        }
        legend_labels: set[str] = set()
        for row in raw_rows:
            if row.sequential_stop_reason == "equivalence":
                label = f"Equivalence (margin={row.equivalence_margin:.3g})"
            elif row.sequential_stop_reason == "futility":
                label = f"Futility (MDES={row.futility_mdes:.3g})"
            elif row.sequential_stop_reason == "max_samples":
                label = "Full Benchmark"
            else:
                label = row.sequential_stop_reason.replace("_", " ").title()
            axis.scatter(
                row.leaderboard_gap,
                row.sample_fraction,
                color=color_by_reason.get(row.sequential_stop_reason, "#7f7f7f"),
                alpha=0.75,
                s=32,
                label=label if label not in legend_labels else None,
            )
            legend_labels.add(label)

        axis.set_title(f"{study_config.benchmark_id}: Pair Difficulty Determines Required Budget")
        axis.set_xlabel("Absolute Leaderboard Score Gap")
        axis.set_ylabel("Sample Fraction Used Before Stopping")
        axis.grid(True, alpha=0.3)
        if raw_rows:
            axis.set_ylim(bottom=0.0, top=1.05)
        if legend_labels:
            axis.legend(frameon=False)
        figure.tight_layout()
        return {"pair_difficulty_budget": figure}


def run_pairwise_separability_study(
    benchmark: Benchmark,
    *,
    initial_look_per_dataset: int,
    batch_size: int,
    top_k: int = 10,
    n_replicates: int = 100,
    equivalence_margin: float = 0.02,
    futility_mdes: float = 0.005,
    alpha: float = 0.05,
    beta: float = 0.1,
    aggregation_mode: AggregationMode = "mean_of_means",
    spending_function: SpendingFunction = "pocock",
    futility_spending_function: SpendingFunction = "pocock",
    enable_futility: bool = False,
    missing_policy: MissingPolicy = "report",
    use_cache: bool = True,
    verbose: bool = True,
) -> CaseStudyResult:
    return PairwiseSeparabilityCaseStudy().run(
        benchmark,
        PairwiseSeparabilityInputConfig(
            initial_look_per_dataset=initial_look_per_dataset,
            batch_size=batch_size,
            top_k=top_k,
            n_replicates=n_replicates,
            equivalence_margin=float(equivalence_margin),
            futility_mdes=float(futility_mdes),
            alpha=alpha,
            beta=beta,
            aggregation_mode=aggregation_mode,
            spending_function=spending_function,
            futility_spending_function=futility_spending_function,
            enable_futility=enable_futility,
            missing_policy=missing_policy,
            use_cache=use_cache,
            verbose=verbose,
        ),
    )


def _run_pair_replicate(
    *,
    benchmark_id: str,
    benchmark_version: str,
    pair_id: str,
    model_a: str,
    model_b: str,
    leaderboard_gap: float,
    replicate_index: int,
    ordered_pairwise,
    fixed_baseline: FixedSampleEstimate,
    input_config: PairwiseSeparabilityInputConfig,
    aggregator: ScoreAggregator,
) -> list[PairwiseSeparabilityRawRow]:
    streams = ordered_pairwise.to_streams(pair_id)
    dataset_sizes = ordered_pairwise.dataset_sizes(pair_id)
    sampling_plan = SamplingPlan(
        dataset_sizes=dataset_sizes,
        initial_size_per_dataset=input_config.initial_look_per_dataset,
        batch_size=input_config.batch_size,
    )
    active_mdes = (
        float(input_config.futility_mdes)
        if input_config.enable_futility
        else float(input_config.equivalence_margin)
    )
    engine_kwargs = dict(
        alpha=input_config.alpha,
        aggregator=aggregator,
        spending_function=input_config.spending_function,
        enable_futility=input_config.enable_futility,
        futility_mdes=float(input_config.futility_mdes),
        beta=input_config.beta,
        futility_spending_function=input_config.futility_spending_function,
    )
    engine = SimpleMeanOfMeansGsDesignEngine(**engine_kwargs)
    sampler = NeymanSampler(
        datasets_streams=streams,
        sampling_plan=sampling_plan,
        aggregation_mode=input_config.aggregation_mode,
    )
    if input_config.enable_futility:
        rules = [EfficacyRule(), FutilityRule()]
    else:
        rules = [EfficacyRule(), EquivalenceRule(margin=float(input_config.equivalence_margin))]
    outcome = run_sequential_test(
        data_map=streams,
        sampler=sampler,
        sampling_plan=sampling_plan,
        engine=engine,
        rules=rules,
    )
    final_look = outcome.looks[-1]
    baseline_label = _baseline_label(
        fixed_baseline.ci_lower,
        fixed_baseline.ci_upper,
        float(input_config.equivalence_margin),
    )
    sequential_label = _sequential_label(outcome.stop_reason.value)
    return [
        PairwiseSeparabilityRawRow(
            benchmark_id=benchmark_id,
            benchmark_version=benchmark_version,
            model_a=model_a,
            model_b=model_b,
            pair_id=pair_id,
            leaderboard_gap=float(leaderboard_gap),
            replicate_index=replicate_index,
            ordering_seed=replicate_index,
            mdes=active_mdes,
            equivalence_margin=float(input_config.equivalence_margin),
            futility_mdes=float(input_config.futility_mdes),
            full_diff=float(fixed_baseline.estimate),
            full_ci_lower=float(fixed_baseline.ci_lower),
            full_ci_upper=float(fixed_baseline.ci_upper),
            full_baseline_label=baseline_label,
            sequential_stop_reason=outcome.stop_reason.value,
            samples_used=outcome.samples_used,
            total_available=outcome.total_available,
            sample_fraction=float(outcome.samples_used / outcome.total_available),
            stop_look=outcome.stop_look,
            total_planned_looks=outcome.total_planned_looks,
            sequential_estimate=float(final_look.estimate),
            sequential_ci_lower=float(final_look.ci_lower),
            sequential_ci_upper=float(final_look.ci_upper),
            agreement_with_full_baseline=(sequential_label == baseline_label),
            aggregation_mode=input_config.aggregation_mode,
        )
    ]


def _baseline_label(ci_lower: float, ci_upper: float, mdes: float) -> FullBaselineLabel:
    if ci_lower > 0.0 or ci_upper < 0.0:
        return "separable"
    if ci_lower >= -mdes and ci_upper <= mdes:
        return "equivalent_under_mdes"
    return "unresolved"


def _sequential_label(stop_reason: str) -> FullBaselineLabel:
    if stop_reason == "efficacy":
        return "separable"
    if stop_reason in {"equivalence", "futility"}:
        return "equivalent_under_mdes"
    return "unresolved"


def _summarize_group(
    benchmark_id: str,
    mdes: float,
    group_label: PairwiseSeparabilityGroup,
    rows: Sequence[PairwiseSeparabilityRawRow],
) -> PairwiseSeparabilitySummaryRow:
    sample_fractions = [row.sample_fraction for row in rows]
    stop_reasons = [row.sequential_stop_reason for row in rows]
    full_labels = [row.full_baseline_label for row in rows]
    pair_ids = {row.pair_id for row in rows}
    efficacy_count = sum(value == "efficacy" for value in stop_reasons)
    equivalence_count = sum(value == "equivalence" for value in stop_reasons)
    futility_count = sum(value == "futility" for value in stop_reasons)
    max_samples_count = sum(value == "max_samples" for value in stop_reasons)
    return PairwiseSeparabilitySummaryRow(
        benchmark_id=benchmark_id,
        mdes=float(mdes),
        group_label=group_label,
        n_runs=len(rows),
        n_pairs=len(pair_ids),
        efficacy_count=efficacy_count,
        equivalence_count=equivalence_count,
        futility_count=futility_count,
        max_samples_count=max_samples_count,
        efficacy_rate=float(efficacy_count / len(rows)),
        equivalence_rate=float(equivalence_count / len(rows)),
        futility_rate=float(futility_count / len(rows)),
        max_samples_rate=float(max_samples_count / len(rows)),
        efficacy_correct_rate=_rule_correct_rate(rows, "efficacy"),
        equivalence_correct_rate=_rule_correct_rate(rows, "equivalence"),
        futility_correct_rate=_rule_correct_rate(rows, "futility"),
        max_samples_correct_rate=_rule_correct_rate(rows, "max_samples"),
        mean_sample_fraction=float(np.mean(sample_fractions)),
        median_sample_fraction=float(median(sample_fractions)),
        p25_sample_fraction=float(np.quantile(sample_fractions, 0.25)),
        p75_sample_fraction=float(np.quantile(sample_fractions, 0.75)),
        sequential_full_baseline_agreement_rate=float(np.mean([row.agreement_with_full_baseline for row in rows])),
        full_separable_rate=float(sum(value == "separable" for value in full_labels) / len(rows)),
        full_equivalent_rate=float(sum(value == "equivalent_under_mdes" for value in full_labels) / len(rows)),
        full_unresolved_rate=float(sum(value == "unresolved" for value in full_labels) / len(rows)),
    )


def _rule_correct_rate(
    rows: Sequence[PairwiseSeparabilityRawRow],
    stop_reason: str,
) -> float:
    matching_rows = [row for row in rows if row.sequential_stop_reason == stop_reason]
    if not matching_rows:
        return 0.0
    if stop_reason == "efficacy":
        return float(np.mean([row.full_baseline_label == "separable" for row in matching_rows]))
    if stop_reason == "equivalence":
        return float(np.mean([row.full_baseline_label == "equivalent_under_mdes" for row in matching_rows]))
    if stop_reason == "futility":
        return float(np.mean([row.full_baseline_label != "separable" for row in matching_rows]))
    if stop_reason == "max_samples":
        return float(np.mean([row.full_baseline_label == "unresolved" for row in matching_rows]))
    raise ValueError(f"Unsupported stop reason for summary: {stop_reason}")


def _is_over_futility_mdes(row: PairwiseSeparabilityRawRow) -> bool:
    return abs(row.full_diff) >= row.futility_mdes


def _is_positive_diff(row: PairwiseSeparabilityRawRow) -> bool:
    return row.full_diff > 0.0


def _is_negative_diff(row: PairwiseSeparabilityRawRow) -> bool:
    return row.full_diff < 0.0


def _is_separable_and_positive_diff(row: PairwiseSeparabilityRawRow) -> bool:
    return row.full_baseline_label == "separable" and _is_positive_diff(row)


def _is_separable_and_negative_diff(row: PairwiseSeparabilityRawRow) -> bool:
    return row.full_baseline_label == "separable" and _is_negative_diff(row)


def _is_non_separable_and_positive_diff(row: PairwiseSeparabilityRawRow) -> bool:
    return row.full_baseline_label != "separable" and _is_positive_diff(row)


def _is_non_separable_and_negative_diff(row: PairwiseSeparabilityRawRow) -> bool:
    return row.full_baseline_label != "separable" and _is_negative_diff(row)


def _is_separable_and_over_futility_mdes(row: PairwiseSeparabilityRawRow) -> bool:
    return row.full_baseline_label == "separable" and _is_over_futility_mdes(row)


def _is_separable_and_under_futility_mdes(row: PairwiseSeparabilityRawRow) -> bool:
    return row.full_baseline_label == "separable" and not _is_over_futility_mdes(row)


def _is_non_separable_and_over_futility_mdes(row: PairwiseSeparabilityRawRow) -> bool:
    return row.full_baseline_label != "separable" and _is_over_futility_mdes(row)


def _is_non_separable_and_under_futility_mdes(row: PairwiseSeparabilityRawRow) -> bool:
    return row.full_baseline_label != "separable" and not _is_over_futility_mdes(row)


if __name__ == "__main__":
    from case_studies.common.artifacts import save_case_study_result

    result = run_pairwise_separability_study(
        benchmark=get_benchmark("open_vlm_leaderboard"),
        initial_look_per_dataset=100,
        batch_size=100,
        top_k=50,
        n_replicates=1,
        equivalence_margin=0.02,
        futility_mdes=0.005,
        aggregation_mode="mean_of_means",
        enable_futility=True,
    )
    paths = save_case_study_result(result)
    print(f"Saved pairwise separability artifacts to {paths.root_dir}")
