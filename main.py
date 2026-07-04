from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping

from case_studies.common.artifacts import save_case_study_result
from case_studies.benchmark_resolution import (
    BenchmarkResolutionCaseStudy,
    BenchmarkResolutionInputConfig,
)
from case_studies.ci_width_precision_curve import (
    CIWidthPrecisionCurveCaseStudy,
    CIWidthPrecisionCurveInputConfig,
)
from case_studies.pairwise_separability import (
    PairwiseSeparabilityInputConfig,
    run_pairwise_separability_study,
)
from case_studies.common.schemas import CaseStudyArtifactPaths, CaseStudyResult
from data.benchmark import Benchmark
from data.registry import get_benchmark

BENCHMARK_ID = "open_vlm_leaderboard"
OUTPUT_ROOT = Path("outputs/case_studies")

CI_WIDTH_PAPER_CONFIG = CIWidthPrecisionCurveInputConfig(
    target_mode="pairwise_difference",
    sampler_name="neyman",
    top_k=50,
    n_replicates=3,
    initial_size_per_dataset=100,
    batch_size=100,
    aggregation_mode="mean_of_means",
    precision_milestones=(0.8, 0.9),
    ci_half_width_milestones=(0.02,),
    pair_strategy="all",
    alpha=0.05,
    verbose=True,
)

BENCHMARK_RESOLUTION_PAPER_CONFIG = BenchmarkResolutionInputConfig(
    top_k=50,
    pair_strategy="all",
    alpha=0.05,
    beta=0.2,
    aggregation_mode="mean_of_means",
    include_plots=True,
    verbose=True,
)

PAIRWISE_SEPARABILITY_PAPER_CONFIG = PairwiseSeparabilityInputConfig(
    initial_look_per_dataset=100,
    batch_size=100,
    top_k=50,
    n_replicates=3,
    equivalence_margin=0.02,
    futility_mdes=0.005,
    alpha=0.05,
    beta=0.1,
    aggregation_mode="mean_of_means",
    enable_futility=False,
    verbose=True,
)


def get_open_vlm_paper_benchmark() -> Benchmark:
    return get_benchmark(BENCHMARK_ID)


def get_open_vlm_paper_configs(
    *,
    use_cache: bool = True,
    verbose: bool = True,
) -> Mapping[str, object]:
    return {
        "ci_width_precision_curve": replace(
            CI_WIDTH_PAPER_CONFIG,
            use_cache=use_cache,
            verbose=verbose,
        ),
        "benchmark_resolution": replace(
            BENCHMARK_RESOLUTION_PAPER_CONFIG,
            use_cache=use_cache,
            verbose=verbose,
        ),
        "pairwise_separability": replace(
            PAIRWISE_SEPARABILITY_PAPER_CONFIG,
            use_cache=use_cache,
            verbose=verbose,
        ),
    }


def run_open_vlm_paper_studies(
    *,
    benchmark: Benchmark | None = None,
    use_cache: bool = True,
    verbose: bool = True,
) -> dict[str, CaseStudyResult]:
    resolved_benchmark = benchmark or get_open_vlm_paper_benchmark()
    configs = get_open_vlm_paper_configs(use_cache=use_cache, verbose=verbose)
    ci_width_config = configs["ci_width_precision_curve"]
    benchmark_resolution_config = configs["benchmark_resolution"]
    pairwise_config = configs["pairwise_separability"]

    return {
        "ci_width_precision_curve": CIWidthPrecisionCurveCaseStudy().run(
            resolved_benchmark,
            ci_width_config,
        ),
        "benchmark_resolution": BenchmarkResolutionCaseStudy().run(
            resolved_benchmark,
            benchmark_resolution_config,
        ),
        "pairwise_separability": run_pairwise_separability_study(
            benchmark=resolved_benchmark,
            initial_look_per_dataset=pairwise_config.initial_look_per_dataset,
            batch_size=pairwise_config.batch_size,
            top_k=pairwise_config.top_k,
            n_replicates=pairwise_config.n_replicates,
            equivalence_margin=pairwise_config.equivalence_margin,
            futility_mdes=pairwise_config.futility_mdes,
            alpha=pairwise_config.alpha,
            beta=pairwise_config.beta,
            aggregation_mode=pairwise_config.aggregation_mode,
            spending_function=pairwise_config.spending_function,
            futility_spending_function=pairwise_config.futility_spending_function,
            enable_futility=pairwise_config.enable_futility,
            missing_policy=pairwise_config.missing_policy,
            use_cache=pairwise_config.use_cache,
            verbose=pairwise_config.verbose,
        ),
    }


def save_open_vlm_paper_studies(
    *,
    benchmark: Benchmark | None = None,
    use_cache: bool = True,
    verbose: bool = True,
    output_root: Path | str = OUTPUT_ROOT,
) -> dict[str, CaseStudyArtifactPaths]:
    results = run_open_vlm_paper_studies(
        benchmark=benchmark,
        use_cache=use_cache,
        verbose=verbose,
    )
    return {
        study_name: save_case_study_result(result, output_root=output_root)
        for study_name, result in results.items()
    }


def main() -> None:
    paths = save_open_vlm_paper_studies()
    for study_name, artifact_paths in paths.items():
        print(f"{study_name}: {artifact_paths.root_dir}")


if __name__ == "__main__":
    main()
