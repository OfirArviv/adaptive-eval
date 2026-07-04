from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
from scipy.stats import norm
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
from case_studies.common.base import BaseCaseStudy
from case_studies.common.utils import make_aggregator, progress_iterator, select_model_pairs
from case_studies.common.configs import PairStrategy, SummaryGroup
from data.benchmark import Benchmark
from data.core import PairwiseBundle
from data.registry import get_benchmark
from data.views import MissingPolicy, build_pairwise_bundle
from fixed_sample import normal_ci
from sequential.aggregators import ScoreAggregator
from sequential.types import AggregationMode
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class BenchmarkResolutionInputConfig:
    top_k: int
    pair_strategy: PairStrategy
    alpha: float
    beta: float
    aggregation_mode: AggregationMode
    missing_policy: MissingPolicy = "error"
    use_cache: bool = True
    include_plots: bool = True
    verbose: bool = True


@dataclass(frozen=True)
class BenchmarkResolutionStudyConfig:
    benchmark_id: str
    benchmark_version: str
    top_k: int
    pair_strategy: PairStrategy
    alpha: float
    beta: float
    aggregation_mode: AggregationMode
    selected_model_ids: list[str]
    pair_ids: list[str]
    missing_policy: MissingPolicy
    use_cache: bool
    include_plots: bool
    verbose: bool


@dataclass(frozen=True)
class BenchmarkResolutionState:
    pairwise: PairwiseBundle
    aggregator: ScoreAggregator
    rank_by_model: dict[str, int]
    selected_model_ids: list[str]
    pair_ids: list[str]


@dataclass(frozen=True)
class BenchmarkResolutionPairRow:
    benchmark_id: str
    benchmark_version: str
    model_a: str
    model_b: str
    pair_id: str
    rank_a: int
    rank_b: int
    rank_gap: int
    full_diff: float
    abs_full_diff: float
    standard_error: float
    variance: float
    ci_lower: float
    ci_upper: float
    p_value: float
    full_benchmark_mde: float
    # abs_full_diff / full_benchmark_mde; values above 1.0 mean the observed gap
    # is larger than the benchmark's estimated minimum detectable effect.
    resolution_ratio: float
    is_resolvable_by_mde: bool
    is_statistically_separable: bool
    aggregation_mode: str
    alpha: float
    beta: float
    total_available: int
    dataset_count: int


@dataclass(frozen=True)
class BenchmarkResolutionSummaryRow:
    benchmark_id: str
    group_label: SummaryGroup
    n_pairs: int
    median_abs_diff: float
    median_mde: float
    mean_mde: float
    p25_mde: float
    p75_mde: float
    statistically_separable_rate: float
    resolvable_by_mde_rate: float
    below_resolution_rate: float
    median_resolution_ratio: float
    p25_resolution_ratio: float
    p75_resolution_ratio: float


class BenchmarkResolutionCaseStudy(
    BaseCaseStudy[
        BenchmarkResolutionInputConfig,
        BenchmarkResolutionStudyConfig,
        BenchmarkResolutionState,
        BenchmarkResolutionPairRow,
        BenchmarkResolutionSummaryRow,
    ]
):
    study_name = "benchmark_resolution"

    def validate_input_config(self, config: BenchmarkResolutionInputConfig) -> None:
        if config.top_k < 2:
            raise ValueError("top_k must be at least 2.")
        if config.pair_strategy not in {"all", "adjacent"}:
            raise ValueError("pair_strategy must be 'all' or 'adjacent'.")
        if not (0.0 < config.alpha < 1.0):
            raise ValueError("alpha must be in the open interval (0, 1).")
        if not (0.0 < config.beta < 1.0):
            raise ValueError("beta must be in the open interval (0, 1).")
        if config.aggregation_mode not in {"mean_of_means", "pooled_mean"}:
            raise ValueError("aggregation_mode must be 'mean_of_means' or 'pooled_mean'.")

    def build_state(
        self,
        bundle,
        benchmark: Benchmark,
        config: BenchmarkResolutionInputConfig,
    ) -> BenchmarkResolutionState:
        selection = select_model_pairs(bundle, top_k=config.top_k, pair_strategy=config.pair_strategy)
        pairwise = build_pairwise_bundle(
            bundle=bundle,
            pairs=[(pair.model_a, pair.model_b) for pair in selection.pairs],
            missing_policy=config.missing_policy,
            verbose=config.verbose,
        )
        return BenchmarkResolutionState(
            pairwise=pairwise,
            aggregator=make_aggregator(config.aggregation_mode),
            rank_by_model=selection.rank_by_model,
            selected_model_ids=selection.selected_model_ids,
            pair_ids=[pair.pair_id for pair in selection.pairs],
        )

    def build_study_config(
        self,
        bundle,
        config: BenchmarkResolutionInputConfig,
        state: BenchmarkResolutionState,
    ) -> BenchmarkResolutionStudyConfig:
        return BenchmarkResolutionStudyConfig(
            benchmark_id=bundle.benchmark_id,
            benchmark_version=bundle.version,
            top_k=config.top_k,
            pair_strategy=config.pair_strategy,
            alpha=config.alpha,
            beta=config.beta,
            aggregation_mode=config.aggregation_mode,
            missing_policy=config.missing_policy,
            use_cache=config.use_cache,
            include_plots=config.include_plots,
            verbose=config.verbose,
            selected_model_ids=state.selected_model_ids,
            pair_ids=state.pair_ids,
        )

    def build_raw_rows(
        self,
        state: BenchmarkResolutionState,
        config: BenchmarkResolutionInputConfig,
        study_config: BenchmarkResolutionStudyConfig,
    ) -> list[BenchmarkResolutionPairRow]:
        return [
            _analyze_pair(
                benchmark_id=state.pairwise.benchmark_id,
                benchmark_version=state.pairwise.version,
                pair_id=pair_id,
                model_a=pair_id.split(";", maxsplit=1)[0],
                model_b=pair_id.split(";", maxsplit=1)[1],
                rank_by_model=state.rank_by_model,
                pairwise=state.pairwise,
                aggregator=state.aggregator,
                alpha=config.alpha,
                beta=config.beta,
            )
            for pair_id in progress_iterator(
                state.pair_ids,
                verbose=config.verbose,
                desc="Pairs",
                unit="pair",
            )
        ]

    def build_summary_rows(
        self,
        raw_rows: Sequence[BenchmarkResolutionPairRow],
        state: BenchmarkResolutionState,
        config: BenchmarkResolutionInputConfig,
        study_config: BenchmarkResolutionStudyConfig,
    ) -> list[BenchmarkResolutionSummaryRow]:
        groups: list[tuple[SummaryGroup, list[BenchmarkResolutionPairRow]]] = [("all", list(raw_rows))]
        if config.pair_strategy == "all":
            adjacent_rows = [row for row in raw_rows if row.rank_gap == 1]
            if adjacent_rows:
                groups.append(("adjacent_pairs", adjacent_rows))
        return [
            _summarize_group(study_config.benchmark_id, group_label, group_rows)
            for group_label, group_rows in groups
            if group_rows
        ]

    def build_narrative(
        self,
        raw_rows: Sequence[BenchmarkResolutionPairRow],
        summary_rows: Sequence[BenchmarkResolutionSummaryRow],
        state: BenchmarkResolutionState,
        config: BenchmarkResolutionInputConfig,
        study_config: BenchmarkResolutionStudyConfig,
    ) -> str:
        overall = next(row for row in summary_rows if row.group_label == "all")
        separable_count = sum(row.is_statistically_separable for row in raw_rows)
        resolvable_count = sum(row.is_resolvable_by_mde for row in raw_rows)
        return (
            f"Benchmark resolution study completed for {study_config.benchmark_id}. "
            f"Among {overall.n_pairs} top-model pairs, {separable_count} had a full-benchmark "
            f"confidence interval excluding zero and {resolvable_count} had an observed difference "
            f"at least as large as the estimated full-benchmark MDE. "
            f"The median full-benchmark MDE was {overall.median_mde:.4g}."
        )

    def build_figures(
        self,
        raw_rows: Sequence[BenchmarkResolutionPairRow],
        summary_rows: Sequence[BenchmarkResolutionSummaryRow],
        state: BenchmarkResolutionState,
        config: BenchmarkResolutionInputConfig,
        study_config: BenchmarkResolutionStudyConfig,
    ) -> dict[str, object]:
        if not config.include_plots:
            return {}
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(2, 2, figsize=(14.0, 8.0))
        flat_axes = axes.ravel()
        _plot_separability_counts(flat_axes[0], raw_rows, study_config)
        _plot_mde_distribution(flat_axes[1], raw_rows)
        _plot_observed_gap_vs_full_benchmark_mde(flat_axes[2], raw_rows)
        _plot_rank_gap_vs_observed_gap(flat_axes[3], raw_rows)
        figure.suptitle(f"{study_config.benchmark_id}: Benchmark Resolution", y=1.02)
        figure.tight_layout()
        return {"benchmark_resolution_overview": figure}

def _analyze_pair(
    *,
    benchmark_id: str,
    benchmark_version: str,
    pair_id: str,
    model_a: str,
    model_b: str,
    rank_by_model: dict[str, int],
    pairwise: PairwiseBundle,
    aggregator: ScoreAggregator,
    alpha: float,
    beta: float,
) -> BenchmarkResolutionPairRow:
    rank_a = rank_by_model[model_a]
    rank_b = rank_by_model[model_b]
    dataset_sizes = pairwise.dataset_sizes(pair_id)
    estimate = normal_ci(
        streams=pairwise.to_streams(pair_id),
        dataset_sizes=dataset_sizes,
        aggregator=aggregator,
        alpha=alpha,
    )
    full_benchmark_mde = float(
        (norm.ppf(1.0 - estimate.alpha / 2.0) + norm.ppf(1.0 - beta))
        * estimate.standard_error
    )
    abs_full_diff = abs(estimate.estimate)
    return BenchmarkResolutionPairRow(
        benchmark_id=benchmark_id,
        benchmark_version=benchmark_version,
        model_a=model_a,
        model_b=model_b,
        pair_id=pair_id,
        rank_a=rank_a,
        rank_b=rank_b,
        rank_gap=abs(rank_b - rank_a),
        full_diff=float(estimate.estimate),
        abs_full_diff=float(abs_full_diff),
        standard_error=float(estimate.standard_error),
        variance=float(estimate.variance),
        ci_lower=float(estimate.ci_lower),
        ci_upper=float(estimate.ci_upper),
        p_value=float(estimate.p_value) if estimate.p_value is not None else float("nan"),
        full_benchmark_mde=full_benchmark_mde,
        resolution_ratio=float(abs_full_diff / full_benchmark_mde),
        is_resolvable_by_mde=bool(abs_full_diff >= full_benchmark_mde),
        is_statistically_separable=bool(estimate.ci_lower > 0.0 or estimate.ci_upper < 0.0),
        aggregation_mode=estimate.aggregation_mode,
        alpha=float(estimate.alpha),
        beta=float(beta),
        total_available=sum(dataset_sizes.values()),
        dataset_count=len(dataset_sizes),
    )


def _summarize_group(
    benchmark_id: str,
    group_label: SummaryGroup,
    rows: Sequence[BenchmarkResolutionPairRow],
) -> BenchmarkResolutionSummaryRow:
    abs_diffs = [row.abs_full_diff for row in rows]
    mdes = [row.full_benchmark_mde for row in rows]
    ratios = [row.resolution_ratio for row in rows]
    return BenchmarkResolutionSummaryRow(
        benchmark_id=benchmark_id,
        group_label=group_label,
        n_pairs=len(rows),
        median_abs_diff=float(np.percentile(np.asarray(abs_diffs, dtype=float), 50)),
        median_mde=float(np.percentile(np.asarray(mdes, dtype=float), 50)),
        mean_mde=float(np.mean(mdes)),
        p25_mde=float(np.percentile(np.asarray(mdes, dtype=float), 25)),
        p75_mde=float(np.percentile(np.asarray(mdes, dtype=float), 75)),
        statistically_separable_rate=float(sum(row.is_statistically_separable for row in rows) / len(rows)),
        resolvable_by_mde_rate=float(sum(row.is_resolvable_by_mde for row in rows) / len(rows)),
        below_resolution_rate=float(sum(not row.is_resolvable_by_mde for row in rows) / len(rows)),
        median_resolution_ratio=float(np.percentile(np.asarray(ratios, dtype=float), 50)),
        p25_resolution_ratio=float(np.percentile(np.asarray(ratios, dtype=float), 25)),
        p75_resolution_ratio=float(np.percentile(np.asarray(ratios, dtype=float), 75)),
    )


def _plot_separability_counts(
    axis,
    rows: Sequence[BenchmarkResolutionPairRow],
    study_config: BenchmarkResolutionStudyConfig,
) -> None:
    plot_rank_gaps = sorted({row.rank_gap for row in rows if row.rank_gap <= 10})
    separable_rates = [
        float(
            sum(row.is_statistically_separable for row in rows if row.rank_gap == rank_gap)
            / sum(1 for row in rows if row.rank_gap == rank_gap)
        )
        for rank_gap in plot_rank_gaps
    ]
    counts = [
        sum(1 for row in rows if row.rank_gap == rank_gap)
        for rank_gap in plot_rank_gaps
    ]
    bars = axis.bar(
        plot_rank_gaps,
        separable_rates,
        color="#2f6f4e",
        alpha=0.9,
        width=0.8,
    )
    for bar, rate, count in zip(bars, separable_rates, counts):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{rate:.0%}\n(n={count})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    total_separable_rate = (
        sum(row.is_statistically_separable for row in rows) / len(rows)
        if rows else 0.0
    )
    axis.text(
        0.02,
        0.98,
        f"Overall separability: {total_separable_rate:.1%}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#2f6f4e", "alpha": 0.9},
    )
    axis.set_title("Separability by Rank Gap")
    axis.set_xlabel("Rank Gap")
    axis.set_ylabel("CI Separable Rate")
    axis.set_xticks(plot_rank_gaps)
    axis.set_ylim(0.0, min(1.2, max(1.05, max(separable_rates) + 0.12 if separable_rates else 1.05)))
    axis.set_yticks(np.linspace(0.0, 1.0, 6))
    axis.set_yticklabels([f"{tick:.0%}" for tick in np.linspace(0.0, 1.0, 6)])
    axis.grid(axis="y", alpha=0.25)


def _plot_mde_distribution(axis, rows: Sequence[BenchmarkResolutionPairRow]) -> None:
    mdes = [row.full_benchmark_mde for row in rows]
    bin_width = _nice_bin_width(mdes)
    decimals = max(0, -int(math.floor(math.log10(bin_width)))) if bin_width < 1.0 else 0
    min_edge = math.floor(min(mdes) / bin_width) * bin_width
    max_edge = math.ceil(max(mdes) / bin_width) * bin_width
    edges = np.arange(min_edge, max_edge + (bin_width * 0.5), bin_width)
    counts, _ = np.histogram(mdes, bins=edges)
    centers = edges[:-1] + (bin_width / 2.0)
    bars = axis.bar(centers, counts, width=bin_width * 0.92, color="#4c78a8", edgecolor="white", align="center")
    total = len(mdes)
    for bar, count in zip(bars, counts):
        percentage = count / total if total else 0.0
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{percentage:.0%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    median_mde = float(np.median(mdes))
    axis.axvline(
        median_mde,
        color="#222222",
        linestyle="--",
        linewidth=1.2,
        label=f"Median full-benchmark MDE = {median_mde:.4g}",
    )
    axis.set_title("Full Benchmark MDE")
    axis.set_xlabel("Full-Benchmark MDE")
    axis.set_ylabel("Pairs")
    axis.set_xticks(centers)
    axis.set_xticklabels([f"{start:.{decimals}f}" for start in edges[:-1]], rotation=45, ha="right")
    axis.set_ylim(0, max(counts) * 1.25 if counts.size and max(counts) > 0 else 1)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)


def _nice_bin_width(values: Sequence[float]) -> float:
    span = max(values) - min(values)
    if span <= 0.0:
        return 0.001
    target_bins = min(12, max(4, int(round(len(values) ** 0.5))))
    raw_width = span / target_bins
    exponent = math.floor(math.log10(raw_width))
    base = 10 ** exponent
    candidates = [1.0, 2.0, 5.0, 10.0]
    return min((factor * base for factor in candidates), key=lambda candidate: abs(candidate - raw_width))


def _plot_observed_gap_vs_full_benchmark_mde(
    axis,
    rows: Sequence[BenchmarkResolutionPairRow],
) -> None:
    from matplotlib.ticker import FormatStrFormatter

    separable_rows = [row for row in rows if row.is_statistically_separable]
    unresolved_rows = [row for row in rows if not row.is_statistically_separable]
    for label, series, color in (
        ("CI separable", separable_rows, "#2f6f4e"),
        ("Not separable", unresolved_rows, "#b7b7b7"),
    ):
        if not series:
            continue
        axis.scatter(
            [row.abs_full_diff for row in series],
            [row.full_benchmark_mde for row in series],
            label=label,
            color=color,
            alpha=0.85,
        )

    max_value = max(
        max(row.abs_full_diff for row in rows),
        max(row.full_benchmark_mde for row in rows),
    )
    axis.plot([0.0, max_value], [0.0, max_value], color="#222222", linestyle="--", linewidth=1.0)
    axis.set_title("Absolute Observed Gap vs Full-Benchmark MDE")
    axis.set_xlabel("Absolute Observed Gap")
    axis.set_ylabel("Full-Benchmark MDE")
    max_mde = max(row.full_benchmark_mde for row in rows)
    axis.set_xlim(left=0.0)
    axis.set_xticks(np.arange(0.0, math.ceil(max_value / 0.01) * 0.01 + 0.005, 0.01))
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.set_yticks(np.arange(0.0, math.ceil(max_mde / 0.01) * 0.01 + 0.005, 0.01))
    axis.set_ylim(0.0, math.ceil(max_mde / 0.01) * 0.01)
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)


def _plot_rank_gap_vs_observed_gap(
    axis,
    rows: Sequence[BenchmarkResolutionPairRow],
) -> None:
    from matplotlib.ticker import FormatStrFormatter

    separable_rows = [row for row in rows if row.is_statistically_separable]
    unresolved_rows = [row for row in rows if not row.is_statistically_separable]
    for label, series, color, marker in (
        ("CI separable", separable_rows, "#2f6f4e", "o"),
        ("Not separable", unresolved_rows, "#b7b7b7", "x"),
    ):
        if not series:
            continue
        axis.scatter(
            [row.rank_gap for row in series],
            [row.abs_full_diff for row in series],
            label=label,
            color=color,
            marker=marker,
            alpha=0.85,
        )

    rank_gaps = sorted({row.rank_gap for row in rows})
    median_ratios = [
        float(np.median([row.abs_full_diff for row in rows if row.rank_gap == rank_gap]))
        for rank_gap in rank_gaps
    ]
    axis.plot(rank_gaps, median_ratios, color="#222222", linewidth=1.4, marker="s", label="median")
    axis.set_title("Rank Gap vs Absolute Observed Gap")
    axis.set_xlabel("Rank Gap")
    axis.set_ylabel("Absolute Absolute Observed Gap")
    axis.set_xticks(np.arange(2, max(rank_gaps) + 1, 2))
    axis.set_xticks(np.arange(1, max(rank_gaps) + 1, 1), minor=True)
    axis.set_xlim(0.5, max(rank_gaps) + 0.5)
    axis.tick_params(axis="x", which="minor", length=3)
    axis.set_yticks(np.arange(
        math.floor(min(row.abs_full_diff for row in rows) / 0.01) * 0.01,
        math.ceil(max(row.abs_full_diff for row in rows) / 0.01) * 0.01 + 0.005,
        0.01,
    ))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.grid(True, alpha=0.25)

    axis.legend(frameon=False, loc="upper left")


if __name__ == "__main__":
    from case_studies.common.artifacts import save_case_study_result
    result = BenchmarkResolutionCaseStudy().run(
        get_benchmark("open_vlm_leaderboard"),
        BenchmarkResolutionInputConfig(
            top_k=50,
            pair_strategy="all",
            alpha=0.05,
            beta=0.2,
            aggregation_mode="mean_of_means"
        ),
    )

    paths = save_case_study_result(result)
