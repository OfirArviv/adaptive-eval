from __future__ import annotations

import json

from data.benchmark import Benchmark
from data.core import BenchmarkBundle, BenchmarkRecord, BenchmarkSpec
from data.sources import HfHubFilesSource, RawSnapshot


class ArenaHard(Benchmark):
    benchmark_id = "arena_hard"
    revision = "03b91ca"
    score_mapper = {"A=B": 0.0, "A>B": 1.0, "A>>B": 3.0, "B>A": -1.0, "B>>A": -3.0}

    def __init__(self, version: str = revision) -> None:
        self.spec = BenchmarkSpec(
            benchmark_id=self.benchmark_id,
            version=version,
            raw_source=HfHubFilesSource(
                source_id=f"{self.benchmark_id}:{version}",
                repo_id="lmsys/arena-hard-browser",
                repo_type="space",
                revision=version,
                pattern="data/arena-hard-v0.1/model_judgment/gpt-4-1106-preview/*.jsonl",
            ),
            dataset_ids=("arena_hard",),
            aggregation_mode="pooled_mean",
            parser_version="arena_hard_v1",
        )

    def parse(self, raw: RawSnapshot) -> BenchmarkBundle:
        records: list[BenchmarkRecord] = []
        warnings: list[str] = []
        for file in raw.files:
            model_id = file.path.stem.lower()
            with file.path.open("r", encoding="utf-8") as file_obj:
                for line in file_obj:
                    row = json.loads(line)
                    try:
                        game_1 = self.score_mapper[row["games"][0]["score"]]
                        game_2 = self.score_mapper[row["games"][1]["score"]]
                    except KeyError:
                        warnings.append(f"Skipping invalid ArenaHard score in {file.logical_path}.")
                        continue
                    records.append(
                        BenchmarkRecord(
                            dataset_id="arena_hard",
                            model_id=model_id,
                            instance_id=str(row["question_id"]),
                            score=(game_2 - game_1) / 2.0,
                            metadata={"source_file": file.logical_path},
                        )
                    )
        return BenchmarkBundle.from_records(
            spec=self.spec,
            records=records,
            manifest_hash=self.spec.manifest_hash,
            warnings=tuple(warnings),
        )
