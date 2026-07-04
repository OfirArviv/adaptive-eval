from __future__ import annotations

from data.benchmark import Benchmark
from data.core import BenchmarkBundle, BenchmarkRecord, BenchmarkSpec
from data.sources import CompositeSource, GitHubTreeSource, RawSnapshot, RawUrlSource


class AlpacaEval20(Benchmark):
    benchmark_id = "alpaca_eval_2.0"
    version = "v0.6.5"

    def __init__(self, version: str = "v0.6.5", leaderboard_filter: str = "verified") -> None:
        self.leaderboard_filter = leaderboard_filter
        self.spec = BenchmarkSpec(
            benchmark_id=self.benchmark_id,
            version=version,
            raw_source=CompositeSource(
                source_id=f"{self.benchmark_id}:{version}:{leaderboard_filter}",
                children=(
                    RawUrlSource(
                        source_id=f"{self.benchmark_id}:{version}:leaderboard",
                        url=(
                            "https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/refs/tags/"
                            f"{version}/docs/data_AlpacaEval_2/weighted_alpaca_eval_gpt4_turbo_leaderboard.csv"
                        ),
                        filename="weighted_alpaca_eval_gpt4_turbo_leaderboard.csv",
                    ),
                    GitHubTreeSource(
                        source_id=f"{self.benchmark_id}:{version}:annotations",
                        repo_id="tatsu-lab/alpaca_eval",
                        revision=version,
                        pattern=r"^results/[^/]+/weighted_alpaca_eval_gpt4_turbo/annotations\.json$",
                    ),
                ),
            ),
            dataset_ids=("alpaca_eval_2.0",),
            aggregation_mode="pooled_mean",
            parser_version="alpaca_eval_2_v1",
        )

    def parse(self, raw: RawSnapshot) -> BenchmarkBundle:
        import pandas as pd

        leaderboard_path = raw.one_file("weighted_alpaca_eval_gpt4_turbo_leaderboard.csv")
        leaderboard = pd.read_csv(leaderboard_path)
        annotations = {
            file.logical_path: file
            for file in raw.files
            if file.logical_path.endswith("weighted_alpaca_eval_gpt4_turbo/annotations.json")
        }

        records: list[BenchmarkRecord] = []
        warnings: list[str] = []
        selected = leaderboard[leaderboard["filter"] == self.leaderboard_filter]
        for row in selected.to_dict("records"):
            model_name = str(row["name"])
            samples = row.get("samples")
            if not isinstance(samples, str):
                warnings.append(f"Missing AlpacaEval samples URL for {model_name}.")
                continue
            annotation_path = (
                samples.split("main/")[1].split("model_outputs.json")[0]
                + "weighted_alpaca_eval_gpt4_turbo/annotations.json"
            )
            file = annotations.get(annotation_path)
            if file is None:
                warnings.append(f"Missing AlpacaEval annotations for {model_name}.")
                continue
            rows = pd.read_json(file.path)
            model_id = str(rows.iloc[0]["generator_2"])
            for row in rows.to_dict("records"):
                if row.get("generator_1") != "gpt4_1106_preview" or row.get("generator_2") != model_id:
                    continue
                preference = row.get("preference")
                if preference is None or preference < 1 or preference > 2:
                    preference = 1.5
                records.append(
                    BenchmarkRecord(
                        dataset_id="alpaca_eval_2.0",
                        model_id=model_id,
                        instance_id=str(row["instruction"]),
                        score=float(preference) - 1.0,
                        metadata={"source_file": file.logical_path},
                    )
                )
        return BenchmarkBundle.from_records(
            spec=self.spec,
            records=records,
            manifest_hash=self.spec.manifest_hash,
            warnings=tuple(warnings),
        )
