from __future__ import annotations

from data.benchmark import Benchmark
from data.benchmarks import (
    AlpacaEval20,
    ArenaHard,
    HelmBenchmark,
    MmluPro,
    OpenLlmLeaderboard,
    OpenVlmLeaderboard,
    SweBench,
)


OPEN_VLM_LEADERBOARD_ALIASES = {
    "open_vlm_leaderboard_ocr_bench": ("ocr_bench",),
    "open_vlm_leaderboard_ai2d": ("ai2d",),
    "open_vlm_leaderboard_mmmu_val": ("mmmu_val",),
    "open_vlm_leaderboard_hallusion_bench": ("hallusion_bench",),
    "open_vlm_leaderboard_mmstar": ("mmstar",),
    "open_vlm_leaderboard_mmbench": ("mmbench_v11",),
}
OPEN_LLM_LEADERBOARD_ALIASES = {
    f"open_llm_leaderboard_{dataset_id}": (dataset_id,)
    for dataset_id in OpenLlmLeaderboard.subsets
}
HELM_ALIASES = {
    dataset_id: (dataset_id,)
    for dataset_id in HelmBenchmark.categories
}
SWE_BENCH_ALIASES = {
    f"swe_bench_{split}": (split,)
    for split in SweBench.splits
}
BENCHMARK_IDS = (
    "alpaca_eval_2.0",
    "mmlu_pro",
    "auto_rag_1st_batch",
    "arena_hard",
    *SWE_BENCH_ALIASES,
    *HELM_ALIASES,
    "helm",
    *OPEN_LLM_LEADERBOARD_ALIASES,
    "open_llm_leaderboard",
    *OPEN_VLM_LEADERBOARD_ALIASES,
    "open_vlm_leaderboard",
)


def get_benchmark(benchmark_id: str = "open_vlm_leaderboard", *, version: str | None = None) -> Benchmark:
    if benchmark_id == "alpaca_eval_2.0":
        return AlpacaEval20(version=version or AlpacaEval20.version)
    if benchmark_id == "open_vlm_leaderboard":
        return OpenVlmLeaderboard(version=version or "local_2025_03_05")
    if benchmark_id in OPEN_VLM_LEADERBOARD_ALIASES:
        return OpenVlmLeaderboard(
            version=version or "local_2025_03_05",
            benchmark_id=benchmark_id,
            dataset_ids=OPEN_VLM_LEADERBOARD_ALIASES[benchmark_id],
        )
    if benchmark_id == "open_llm_leaderboard":
        return OpenLlmLeaderboard(version=version or "local")
    if benchmark_id in OPEN_LLM_LEADERBOARD_ALIASES:
        return OpenLlmLeaderboard(
            version=version or "local",
            benchmark_id=benchmark_id,
            dataset_ids=OPEN_LLM_LEADERBOARD_ALIASES[benchmark_id],
        )
    if benchmark_id == "helm":
        return HelmBenchmark(version=version or "agg_data_v0.2.2_v3")
    if benchmark_id in HELM_ALIASES:
        return HelmBenchmark(
            version=version or "agg_data_v0.2.2_v3",
            benchmark_id=benchmark_id,
            dataset_ids=HELM_ALIASES[benchmark_id],
        )
    if benchmark_id == "mmlu_pro":
        return MmluPro(version=version or MmluPro.revision)
    if benchmark_id == "arena_hard":
        return ArenaHard(version=version or ArenaHard.revision)
    if benchmark_id in SWE_BENCH_ALIASES:
        return SweBench(
            version=version or SweBench.revision,
            benchmark_id=benchmark_id,
            dataset_ids=SWE_BENCH_ALIASES[benchmark_id],
        )
    raise NotImplementedError(f"Unsupported benchmark_id: {benchmark_id}")
