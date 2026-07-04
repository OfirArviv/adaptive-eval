from __future__ import annotations

from data.benchmark import Benchmark
from data.cache import RAW_ROOT
from data.core import BenchmarkBundle, BenchmarkRecord, BenchmarkSpec
from data.sources import LocalGlobSource, RawSnapshot


class HelmBenchmark(Benchmark):
    benchmark_id = "helm"
    models = (
        "ai21_j1-jumbo",
        "cohere_medium-20220720",
        "together_t0pp",
        "together_opt-175b",
        "openai_curie",
        "openai_text-davinci-003",
        "cohere_small-20220720",
        "together_opt-66b",
        "anthropic_stanford-online-all-v4-s3",
        "openai_davinci",
        "openai_ada",
        "cohere_command-xlarge-beta",
        "openai_text-ada-001",
        "ai21_j1-grande-v2-beta",
        "together_gpt-j-6b",
        "together_bloom",
        "cohere_xlarge-20220609",
        "ai21_j1-grande",
        "ai21_j1-large",
        "openai_text-curie-001",
        "AlephAlpha_luminous-base",
        "AlephAlpha_luminous-extended",
        "cohere_large-20220720",
        "AlephAlpha_luminous-supreme",
        "together_ul2",
        "together_t5-11b",
        "cohere_medium-20221108",
        "microsoft_TNLGv2_7B",
        "openai_babbage",
        "openai_text-davinci-002",
        "cohere_command-medium-beta",
        "cohere_xlarge-20221108",
        "openai_text-babbage-001",
        "together_glm",
        "together_gpt-neox-20b",
        "microsoft_TNLGv2_530B",
        "together_yalm",
    )
    categories = {
        "helm_qa": (
            "question answering",
            ("mmlu", "boolq", "narrative_qa", "natural_qa:mode=closedbook", "natural_qa:mode=openbook", "quac", "openbookqa", "truthful_qa"),
        ),
        "helm_summarization": ("summarization", ("summarization_cnndm", "summarization_xsum")),
        "helm_ir": ("information_retrieval", ("msmarco:track=regular", "msmarco:track=trec")),
        "helm_sentiment_analysis": ("sentiment_analysis", ("imdb",)),
        "helm_toxicity_detection": ("toxicity_detection", ("civil_comments",)),
        "helm_text_classification": ("text_classification", ("raft",)),
    }

    def __init__(
        self,
        version: str = "agg_data_v0.2.2_v3",
        *,
        benchmark_id: str = "helm",
        dataset_ids: tuple[str, ...] | None = None,
    ) -> None:
        self.benchmark_id = benchmark_id
        self.selected_dataset_ids = dataset_ids or tuple(self.categories)
        self.spec = BenchmarkSpec(
            benchmark_id=benchmark_id,
            version=version,
            raw_source=LocalGlobSource(
                source_id=f"{benchmark_id}:{version}",
                pattern=RAW_ROOT / "helm" / version / "*.csv",
            ),
            dataset_ids=self.selected_dataset_ids,
            aggregation_mode="mean_of_means",
            parser_version="helm_v1",
        )

    def parse(self, raw: RawSnapshot) -> BenchmarkBundle:
        import pandas as pd

        frames = []
        for file in raw.files:
            if file.path.suffix == ".csv":
                frame = pd.read_csv(file.path)
                frame["scenario"] = frame["data_augmentations"].map(self._scenario_name)
                frame["balanced_score"] = self._balanced_scores(frame)
                frames.append(frame)
        table = pd.concat(frames, ignore_index=True)
        table = table[table["unique_id_inc_seed"].str.endswith("_0")]

        records: list[BenchmarkRecord] = []
        for dataset_id in self.selected_dataset_ids:
            _, scenarios = self.categories[dataset_id]
            subset = table[table["scenario"].isin(scenarios)]
            for model_id in self.models:
                model_rows = subset[subset["model"] == model_id]
                for row in model_rows.to_dict("records"):
                    records.append(
                        BenchmarkRecord(
                            dataset_id=dataset_id,
                            model_id=model_id,
                            instance_id=str(row["unique_id_inc_seed"]),
                            score=float(row["balanced_score"]),
                            grouping=str(row["scenario"]),
                        )
                    )
        return BenchmarkBundle.from_records(
            spec=self.spec,
            records=records,
            manifest_hash=self.spec.manifest_hash,
        )

    @staticmethod
    def _scenario_name(value: str) -> str:
        value = value.split("v0.0.2/")[-1].split("v0.2.2/")[-1].split("v0.2.3/")[-1]
        if "msmarco" in value:
            return value.split(",")[0]
        if "natural_qa" in value:
            return value.split(",model")[0].replace("_longans", "")
        if "hellaswag" in value:
            return "hellaswag"
        if "openbookqa" in value:
            return "openbookqa"
        return value.split(":")[0]

    @staticmethod
    def _balanced_scores(frame):
        if "sub_split" not in frame.columns or len(frame["sub_split"].dropna().unique()) <= 1:
            return frame["score"].values.tolist()
        subsplit_count = frame.groupby("sub_split")["unique_id_inc_seed"].count().to_dict()
        weights = {key: sum(subsplit_count.values()) / (len(subsplit_count) * value) for key, value in subsplit_count.items()}
        return frame["score"].mul(frame["sub_split"].map(weights)).values.tolist()
