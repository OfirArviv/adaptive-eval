from __future__ import annotations

import json

from data.benchmark import Benchmark
from data.core import BenchmarkBundle, BenchmarkRecord, BenchmarkSpec
from data.sources import GitHubTreeSource, RawSnapshot


class MmluPro(Benchmark):
    benchmark_id = "mmlu_pro"
    revision = "b7b9ffd84b2c21a5bfcf174fc65e5e6d74ca09a8"

    def __init__(self, version: str = revision) -> None:
        self.spec = BenchmarkSpec(
            benchmark_id=self.benchmark_id,
            version=version,
            raw_source=GitHubTreeSource(
                source_id=f"{self.benchmark_id}:{version}",
                repo_id="TIGER-AI-Lab/MMLU-Pro",
                revision=version,
                pattern=r"^eval_results/[^/]+\.json$",
            ),
            dataset_ids=("mmlu_pro",),
            aggregation_mode="pooled_mean",
            parser_version="mmlu_pro_v1",
        )

    def parse(self, raw: RawSnapshot) -> BenchmarkBundle:
        records: list[BenchmarkRecord] = []
        for file in raw.files:
            with file.path.open("r", encoding="utf-8") as file_obj:
                rows = json.load(file_obj)
            model_id = file.path.stem.split("model_outputs_")[-1].lower()
            for row in rows:
                if not isinstance(row, dict):
                    continue
                records.append(
                    BenchmarkRecord(
                        dataset_id="mmlu_pro",
                        model_id=model_id,
                        instance_id=str(row["question_id"]),
                        score=float(row["answer"] == row["pred"]),
                        grouping=str(row.get("category")) if row.get("category") is not None else None,
                        metadata={"source_file": file.logical_path},
                    )
                )
        return BenchmarkBundle.from_records(
            spec=self.spec,
            records=records,
            manifest_hash=self.spec.manifest_hash,
        )
