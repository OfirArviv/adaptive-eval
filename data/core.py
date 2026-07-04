from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, Sequence

if TYPE_CHECKING:
    from data.sources import RawSource


AggregationMode = Literal["mean_of_means", "pooled_mean"]


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    version: str
    raw_source: RawSource
    dataset_ids: tuple[str, ...]
    aggregation_mode: AggregationMode
    parser_version: str = "v1"

    @property
    def benchmark_version(self) -> str:
        return self.version

    @property
    def manifest(self) -> dict[str, Any]:
        from data.sources import source_manifest

        return {
            "benchmark_id": self.benchmark_id,
            "version": self.version,
            "raw_source": source_manifest(self.raw_source),
            "dataset_ids": list(self.dataset_ids),
            "aggregation_mode": self.aggregation_mode,
            "parser_version": self.parser_version,
        }

    @property
    def manifest_hash(self) -> str:
        from data.cache import stable_json_hash

        return stable_json_hash(self.manifest)


@dataclass(frozen=True)
class BenchmarkRecord:
    dataset_id: str
    model_id: str
    instance_id: str
    score: float
    grouping: str | None = None
    split: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class BenchmarkDataset:
    dataset_id: str
    records: tuple[BenchmarkRecord, ...]

    @cached_property
    def by_model(self) -> dict[str, tuple[BenchmarkRecord, ...]]:
        grouped: dict[str, list[BenchmarkRecord]] = {}
        for record in self.records:
            grouped.setdefault(record.model_id, []).append(record)
        return {model_id: tuple(records) for model_id, records in grouped.items()}


@dataclass(frozen=True)
class BenchmarkBundle:
    benchmark_id: str
    version: str
    dataset_ids: tuple[str, ...]
    records: tuple[BenchmarkRecord, ...]
    aggregation_mode: AggregationMode
    manifest_hash: str
    missing_datasets_by_model: dict[str, tuple[str, ...]] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def benchmark_version(self) -> str:
        return self.version

    @cached_property
    def datasets(self) -> dict[str, BenchmarkDataset]:
        grouped: dict[str, list[BenchmarkRecord]] = {dataset_id: [] for dataset_id in self.dataset_ids}
        for record in self.records:
            grouped.setdefault(record.dataset_id, []).append(record)
        return {
            dataset_id: BenchmarkDataset(dataset_id=dataset_id, records=tuple(records))
            for dataset_id, records in grouped.items()
        }

    @cached_property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(sorted({record.model_id for record in self.records}))

    @property
    def total_records(self) -> int:
        return len(self.records)

    @property
    def total_models(self) -> int:
        return len(self.model_ids)

    @property
    def total_datasets(self) -> int:
        return len(self.dataset_ids)

    @cached_property
    def complete_model_ids(self) -> tuple[str, ...]:
        return tuple(
            model_id
            for model_id in self.model_ids
            if all(self.by_model_dataset.get(model_id, {}).get(dataset_id) for dataset_id in self.dataset_ids)
        )

    @cached_property
    def by_model(self) -> dict[str, tuple[BenchmarkRecord, ...]]:
        grouped: dict[str, list[BenchmarkRecord]] = {}
        for record in self.records:
            grouped.setdefault(record.model_id, []).append(record)
        return {model_id: tuple(records) for model_id, records in grouped.items()}

    @cached_property
    def by_model_dataset(self) -> dict[str, dict[str, tuple[BenchmarkRecord, ...]]]:
        grouped: dict[str, dict[str, list[BenchmarkRecord]]] = {}
        for record in self.records:
            grouped.setdefault(record.model_id, {}).setdefault(record.dataset_id, []).append(record)
        return {
            model_id: {
                dataset_id: tuple(records)
                for dataset_id, records in per_dataset.items()
            }
            for model_id, per_dataset in grouped.items()
        }

    @cached_property
    def dataset_sizes(self) -> dict[str, int]:
        return {
            dataset_id: len({record.instance_id for record in self.datasets[dataset_id].records})
            for dataset_id in self.dataset_ids
        }

    @cached_property
    def records_by_dataset(self) -> dict[str, int]:
        return {
            dataset_id: len(self.datasets[dataset_id].records)
            for dataset_id in self.dataset_ids
        }

    @cached_property
    def models_by_dataset(self) -> dict[str, int]:
        return {
            dataset_id: len(self.datasets[dataset_id].by_model)
            for dataset_id in self.dataset_ids
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "version": self.version,
            "total_records": self.total_records,
            "total_models": self.total_models,
            "total_datasets": self.total_datasets,
            "records_by_dataset": self.records_by_dataset,
            "models_by_dataset": self.models_by_dataset,
            "missing_datasets_by_model": {
                model_id: list(dataset_ids)
                for model_id, dataset_ids in self.missing_datasets_by_model.items()
            },
            "warnings": list(self.warnings),
        }

    def get_dataset(self, dataset_id: str) -> BenchmarkDataset:
        return self.datasets[dataset_id]

    def rank_models(self, *, complete_only: bool = True) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        model_ids = self.complete_model_ids if complete_only else self.model_ids
        for model_id in model_ids:
            datasets = self.by_model_dataset[model_id]
            if self.aggregation_mode == "pooled_mean":
                values = [record.score for records in datasets.values() for record in records]
                if values:
                    scores[model_id] = sum(values) / len(values)
                continue

            dataset_means = [
                sum(record.score for record in records) / len(records)
                for records in datasets.values()
                if records
            ]
            if dataset_means:
                scores[model_id] = sum(dataset_means) / len(dataset_means)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)

    def to_streams(self, model_id: str) -> dict[str, tuple[BenchmarkRecord, ...]]:
        return {
            dataset_id: tuple(self.by_model_dataset.get(model_id, {}).get(dataset_id, ()))
            for dataset_id in self.dataset_ids
        }

    def dataset_sizes_for_model(self, model_id: str) -> dict[str, int]:
        return {
            dataset_id: len(records)
            for dataset_id, records in self.to_streams(model_id).items()
        }

    def with_records(self, records: Sequence[BenchmarkRecord]) -> "BenchmarkBundle":
        return BenchmarkBundle(
            benchmark_id=self.benchmark_id,
            version=self.version,
            dataset_ids=self.dataset_ids,
            records=tuple(records),
            aggregation_mode=self.aggregation_mode,
            manifest_hash=self.manifest_hash,
            missing_datasets_by_model=self.missing_datasets_by_model,
            warnings=self.warnings,
        )

    def with_manifest_hash(self, manifest_hash: str) -> "BenchmarkBundle":
        return BenchmarkBundle(
            benchmark_id=self.benchmark_id,
            version=self.version,
            dataset_ids=self.dataset_ids,
            records=self.records,
            aggregation_mode=self.aggregation_mode,
            manifest_hash=manifest_hash,
            missing_datasets_by_model=self.missing_datasets_by_model,
            warnings=self.warnings,
        )

    @classmethod
    def from_records(
        cls,
        *,
        spec: BenchmarkSpec,
        records: Sequence[BenchmarkRecord],
        manifest_hash: str,
        missing_datasets_by_model: dict[str, tuple[str, ...]] | None = None,
        warnings: tuple[str, ...] = (),
    ) -> "BenchmarkBundle":
        record_tuple, warnings = sanitize_benchmark_records(
            records,
            benchmark_id=spec.benchmark_id,
            warnings=warnings,
            context="normalization",
        )
        unknown_dataset_ids = sorted({record.dataset_id for record in record_tuple} - set(spec.dataset_ids))
        if unknown_dataset_ids:
            raise ValueError(f"Records contain datasets not declared by spec: {unknown_dataset_ids}")
        return cls(
            benchmark_id=spec.benchmark_id,
            version=spec.version,
            dataset_ids=spec.dataset_ids,
            records=record_tuple,
            aggregation_mode=spec.aggregation_mode,
            manifest_hash=manifest_hash,
            missing_datasets_by_model=missing_datasets_by_model or {},
            warnings=warnings,
        )


def sanitize_benchmark_records(
    records: Sequence[BenchmarkRecord],
    *,
    benchmark_id: str,
    warnings: Sequence[str] = (),
    context: str,
) -> tuple[tuple[BenchmarkRecord, ...], tuple[str, ...]]:
    input_records = tuple(records)
    filtered_records = tuple(record for record in input_records if math.isfinite(float(record.score)))
    dropped_non_finite = len(input_records) - len(filtered_records)
    warning_tuple = tuple(warnings)
    if not dropped_non_finite:
        return filtered_records, warning_tuple

    message = (
        f"{benchmark_id}: dropped {dropped_non_finite} records with non-finite scores "
        f"during {context}."
    )
    print(f"[data] {message}")
    return filtered_records, (*warning_tuple, message)


@dataclass(frozen=True)
class PairwiseRecord:
    pair_id: str
    model_a: str
    model_b: str
    dataset_id: str
    instance_id: str
    score_a: float
    score_b: float
    diff: float
    grouping: str | None = None

    @property
    def score(self) -> float:
        return self.diff


@dataclass(frozen=True)
class PairwiseBundle:
    benchmark_id: str
    version: str
    source_manifest_hash: str
    pairset_hash: str
    records: tuple[PairwiseRecord, ...]

    @property
    def benchmark_version(self) -> str:
        return self.version

    @cached_property
    def pair_ids(self) -> tuple[str, ...]:
        return tuple(sorted({record.pair_id for record in self.records}))

    @cached_property
    def dataset_ids(self) -> tuple[str, ...]:
        return tuple(sorted({record.dataset_id for record in self.records}))

    @cached_property
    def by_pair_dataset(self) -> dict[str, dict[str, tuple[PairwiseRecord, ...]]]:
        grouped: dict[str, dict[str, list[PairwiseRecord]]] = {}
        for record in self.records:
            grouped.setdefault(record.pair_id, {}).setdefault(record.dataset_id, []).append(record)
        return {
            pair_id: {
                dataset_id: tuple(records)
                for dataset_id, records in per_dataset.items()
            }
            for pair_id, per_dataset in grouped.items()
        }

    def dataset_sizes(self, pair_id: str) -> dict[str, int]:
        return {
            dataset_id: len(records)
            for dataset_id, records in self.by_pair_dataset[pair_id].items()
        }

    def to_streams(self, pair_id: str) -> dict[str, tuple[PairwiseRecord, ...]]:
        return dict(self.by_pair_dataset[pair_id])

    def with_records(self, records: Sequence[PairwiseRecord]) -> "PairwiseBundle":
        return PairwiseBundle(
            benchmark_id=self.benchmark_id,
            version=self.version,
            source_manifest_hash=self.source_manifest_hash,
            pairset_hash=self.pairset_hash,
            records=tuple(records),
        )
