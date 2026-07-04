from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from data.benchmark import Benchmark
from data.core import BenchmarkBundle, BenchmarkRecord, BenchmarkSpec
from data.sources import CompositeSource, GitHubTreeSource, HfDatasetSource, RawFile, RawSnapshot


class SweBench(Benchmark):
    benchmark_id = "swe_bench"
    revision = "main"
    splits = ("lite", "verified", "test")
    hf_repo_by_split = {
        "lite": "SWE-bench/SWE-bench_Lite",
        "verified": "SWE-bench/SWE-bench_Verified",
        "test": "SWE-bench/SWE-bench",
    }

    def __init__(
        self,
        version: str = revision,
        *,
        benchmark_id: str = "swe_bench",
        dataset_ids: tuple[str, ...] | None = None,
    ) -> None:
        selected_dataset_ids = dataset_ids or self.splits
        _validate_splits(selected_dataset_ids)
        split_pattern = "|".join(re.escape(split) for split in selected_dataset_ids)
        self.spec = BenchmarkSpec(
            benchmark_id=benchmark_id,
            version=version,
            raw_source=CompositeSource(
                source_id=f"{benchmark_id}:{version}:{','.join(selected_dataset_ids)}",
                children=(
                    GitHubTreeSource(
                        source_id=f"{benchmark_id}:{version}:evaluation",
                        repo_id="SWE-bench/experiments",
                        revision=version,
                        pattern=(
                            rf"^evaluation/({split_pattern})/[^/]+/"
                            r"(metadata\.ya?ml|results/results\.json)$"
                        ),
                    ),
                    *(
                        HfDatasetSource(
                            source_id=f"{benchmark_id}:{version}:{split}:instances",
                            repo_id=self.hf_repo_by_split[split],
                            split="test",
                            filename=f"{split}.parquet",
                        )
                        for split in selected_dataset_ids
                    ),
                ),
            ),
            dataset_ids=selected_dataset_ids,
            aggregation_mode="mean_of_means",
            parser_version="swe_bench_v1",
        )

    def parse(self, raw: RawSnapshot) -> BenchmarkBundle:
        instances_by_split = _instances_by_split(raw, self.spec.dataset_ids)
        submissions = _submissions(raw, self.spec.dataset_ids)
        records: list[BenchmarkRecord] = []
        warnings: list[str] = []

        for (split, submission), files in sorted(submissions.items()):
            model_id, metadata_warning = _model_id(files.get("metadata"), submission)
            if metadata_warning is not None:
                warnings.append(metadata_warning)
            results_file = files.get("results")
            if results_file is None:
                warnings.append(f"Missing SWE-bench results.json for {split}/{submission}.")
                continue
            resolved = _resolved_instance_ids(results_file.path)
            for instance in instances_by_split.get(split, ()):
                instance_id = str(instance["instance_id"])
                repo = _optional_string(instance.get("repo"))
                metadata = {
                    "source_file": results_file.logical_path,
                    "submission": submission,
                }
                if repo is not None:
                    metadata["repo"] = repo
                created_at = _optional_string(instance.get("created_at"))
                if created_at is not None:
                    metadata["created_at"] = created_at
                records.append(
                    BenchmarkRecord(
                        dataset_id=split,
                        model_id=model_id,
                        instance_id=instance_id,
                        score=1.0 if instance_id in resolved else 0.0,
                        grouping=repo,
                        split=split,
                        metadata=metadata,
                    )
                )

        return BenchmarkBundle.from_records(
            spec=self.spec,
            records=records,
            manifest_hash=self.spec.manifest_hash,
            warnings=tuple(warnings),
        )


def _validate_splits(dataset_ids: tuple[str, ...]) -> None:
    unknown = sorted(set(dataset_ids) - set(SweBench.splits))
    if unknown:
        raise ValueError(f"Unsupported SWE-bench splits: {unknown}")


def _instances_by_split(raw: RawSnapshot, splits: tuple[str, ...]) -> dict[str, tuple[dict[str, Any], ...]]:
    return {
        split: _read_instances(_one_file(raw, f"{split}.parquet"))
        for split in splits
    }


def _read_instances(path: Path) -> tuple[dict[str, Any], ...]:
    import pandas as pd
    return tuple(pd.read_parquet(path).to_dict("records"))


def _submissions(
    raw: RawSnapshot,
    splits: tuple[str, ...],
) -> dict[tuple[str, str], dict[str, RawFile]]:
    split_set = set(splits)
    submissions: dict[tuple[str, str], dict[str, RawFile]] = {}
    for file in raw.files:
        parts = file.logical_path.split("/")
        if len(parts) < 4 or parts[0] != "evaluation" or parts[1] not in split_set:
            continue
        split = parts[1]
        submission = parts[2]
        if parts[3:] == ["results", "results.json"]:
            submissions.setdefault((split, submission), {})["results"] = file
        elif parts[3] in {"metadata.yaml", "metadata.yml"}:
            submissions.setdefault((split, submission), {})["metadata"] = file
    return submissions


def _model_id(metadata_file: RawFile | None, submission: str) -> tuple[str, str | None]:
    if metadata_file is None:
        return submission, f"Missing SWE-bench metadata file for {submission}; using submission folder."
    metadata = _read_metadata(metadata_file.path)
    info = metadata.get("info")
    if isinstance(info, dict):
        name = _optional_string(info.get("name"))
        if name is not None:
            return name, None
    return submission, f"Missing SWE-bench metadata info/name for {submission}; using submission folder."


def _read_metadata(path: Path) -> dict[str, Any]:
    import yaml
    with path.open("r", encoding="utf-8") as file_obj:
        metadata = yaml.safe_load(file_obj) or {}
    if not isinstance(metadata, dict):
        return {}
    return metadata


def _resolved_instance_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as file_obj:
        results = json.load(file_obj)
    return {str(instance_id) for instance_id in results["resolved"]}


def _one_file(raw: RawSnapshot, logical_path: str) -> Path:
    matches = [file.path for file in raw.files if file.logical_path == logical_path]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one raw file named {logical_path!r}, found {len(matches)}.")
    return matches[0]


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value)
    return value if value else None
