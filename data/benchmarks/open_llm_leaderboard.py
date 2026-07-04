from __future__ import annotations

from statistics import mean
from typing import Any

from data.benchmark import Benchmark
from data.cache import RAW_ROOT
from data.core import BenchmarkBundle, BenchmarkRecord, BenchmarkSpec
from data.sources import LocalFileSource, RawSnapshot


class OpenLlmLeaderboard(Benchmark):
    benchmark_id = "open_llm_leaderboard"
    subsets = ("ifeval", "bbh", "math", "gpqa", "musr", "mmlu_pro")
    scenarios = (
        "leaderboard_bbh_boolean_expressions",
        "leaderboard_bbh_causal_judgement",
        "leaderboard_bbh_date_understanding",
        "leaderboard_bbh_disambiguation_qa",
        "leaderboard_bbh_formal_fallacies",
        "leaderboard_bbh_geometric_shapes",
        "leaderboard_bbh_hyperbaton",
        "leaderboard_bbh_logical_deduction_five_objects",
        "leaderboard_bbh_logical_deduction_seven_objects",
        "leaderboard_bbh_logical_deduction_three_objects",
        "leaderboard_bbh_movie_recommendation",
        "leaderboard_bbh_navigate",
        "leaderboard_bbh_object_counting",
        "leaderboard_bbh_penguins_in_a_table",
        "leaderboard_bbh_reasoning_about_colored_objects",
        "leaderboard_bbh_ruin_names",
        "leaderboard_bbh_salient_translation_error_detection",
        "leaderboard_bbh_snarks",
        "leaderboard_bbh_sports_understanding",
        "leaderboard_bbh_temporal_sequences",
        "leaderboard_bbh_tracking_shuffled_objects_five_objects",
        "leaderboard_bbh_tracking_shuffled_objects_seven_objects",
        "leaderboard_bbh_tracking_shuffled_objects_three_objects",
        "leaderboard_bbh_web_of_lies",
        "leaderboard_gpqa_diamond",
        "leaderboard_gpqa_extended",
        "leaderboard_gpqa_main",
        "leaderboard_ifeval",
        "leaderboard_math_algebra_hard",
        "leaderboard_math_counting_and_prob_hard",
        "leaderboard_math_geometry_hard",
        "leaderboard_math_intermediate_algebra_hard",
        "leaderboard_math_num_theory_hard",
        "leaderboard_math_prealgebra_hard",
        "leaderboard_math_precalculus_hard",
        "leaderboard_mmlu_pro",
        "leaderboard_musr_murder_mysteries",
        "leaderboard_musr_object_placements",
        "leaderboard_musr_team_allocation",
    )

    def __init__(
        self,
        version: str = "local",
        *,
        benchmark_id: str = "open_llm_leaderboard",
        dataset_ids: tuple[str, ...] | None = None,
        top_k: int = 50,
    ) -> None:
        self.benchmark_id = benchmark_id
        self.selected_dataset_ids = dataset_ids or self.subsets
        self.top_k = top_k
        self.spec = BenchmarkSpec(
            benchmark_id=benchmark_id,
            version=version,
            raw_source=LocalFileSource(
                source_id=f"{benchmark_id}:{version}",
                path=RAW_ROOT / "open_llm_leaderboard" / version / "open-llm-leaderboard.csv",
            ),
            dataset_ids=self.selected_dataset_ids,
            aggregation_mode="mean_of_means",
            parser_version="open_llm_leaderboard_v1",
        )

    def parse(self, raw: RawSnapshot) -> BenchmarkBundle:
        import pandas as pd
        from datasets import load_dataset
        from tqdm.auto import tqdm

        leaderboard = pd.read_csv(raw.files[0].path)
        leaderboard["Average ⬆️"] = pd.to_numeric(leaderboard["Average ⬆️"], errors="coerce")
        models = leaderboard.nlargest(self.top_k, "Average ⬆️")["Model"].dropna().tolist()
        tasks = [
            (str(model).replace("/", "__"), scenario)
            for model in models
            for scenario in self.scenarios
            if self._dataset_id_for_scenario(scenario) in self.selected_dataset_ids
        ]
        records: list[BenchmarkRecord] = []
        warnings: list[str] = []

        progress = tqdm(
            tasks,
            desc=f"Loading {self.benchmark_id}",
            unit="dataset",
            dynamic_ncols=True,
        )
        for model_id, scenario in progress:
            progress.set_postfix_str(f"{model_id}/{scenario}", refresh=False)
            dataset_path = f"open-llm-leaderboard/{model_id}-details"
            if dataset_path == "open-llm-leaderboard/mistralai__Mixtral-8x22B-Instruct-v0.1-details":
                continue
            dataset_id = self._dataset_id_for_scenario(scenario)
            try:
                dataset = load_dataset(dataset_path, name=f"{model_id}__{scenario}", split="latest")
            except Exception as exc:
                warnings.append(f"{dataset_path}/{scenario}: {exc}")
                continue
            metric = self._metric_for_scenario(scenario)
            for row in dataset:
                records.append(
                    BenchmarkRecord(
                        dataset_id=dataset_id,
                        model_id=model_id,
                        instance_id=f"{scenario}_{row['doc_id']}",
                        score=self._score(row[metric], scenario),
                        grouping=scenario,
                    )
                )

        return BenchmarkBundle.from_records(
            spec=self.spec,
            records=records,
            manifest_hash=self.spec.manifest_hash,
            warnings=tuple(warnings),
        )

    @staticmethod
    def _dataset_id_for_scenario(scenario: str) -> str:
        for subset in OpenLlmLeaderboard.subsets:
            if subset in scenario:
                return subset
        raise ValueError(f"Unsupported Open LLM scenario: {scenario}")

    @staticmethod
    def _metric_for_scenario(scenario: str) -> str:
        if "ifeval" in scenario:
            return "inst_level_strict_acc"
        if "mmlu_pro" in scenario:
            return "acc"
        if "math" in scenario:
            return "exact_match"
        return "acc_norm"

    @staticmethod
    def _score(value: Any, scenario: str) -> float:
        if "ifeval" in scenario:
            return float(mean(1.0 if item is True else 0.0 for item in value))
        return float(value)
