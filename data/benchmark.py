from __future__ import annotations

from abc import ABC, abstractmethod

from data.cache import (
    normalized_cache_dir,
    read_json,
    read_records_parquet,
    stable_json_hash,
    write_json,
    write_records_parquet,
)
from data.core import BenchmarkBundle, BenchmarkSpec, sanitize_benchmark_records
from data.sources import RawSnapshot, portable_path, resolve_raw_source


class Benchmark(ABC):
    """One benchmark lifecycle: raw data -> normalized records."""

    spec: BenchmarkSpec

    def raw(self, *, use_cache: bool = True) -> RawSnapshot:
        return resolve_raw_source(self.spec.raw_source, use_cache=use_cache)

    def load(self, *, use_cache: bool = True) -> BenchmarkBundle:
        raw = self.raw(use_cache=use_cache)
        manifest_hash = self._normalized_manifest_hash(raw)
        cache_dir = normalized_cache_dir(self.spec.benchmark_id, self.spec.version, manifest_hash)
        records_path = cache_dir / "records.parquet"
        manifest_path = cache_dir / "manifest.json"

        if use_cache and records_path.exists() and manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest.get("manifest_hash") == manifest_hash:
                diagnostics = manifest.get("diagnostics", {})
                records, warnings = sanitize_benchmark_records(
                    read_records_parquet(records_path),
                    benchmark_id=self.spec.benchmark_id,
                    warnings=tuple(str(item) for item in diagnostics.get("warnings", ())),
                    context="cache load",
                )
                return BenchmarkBundle(
                    benchmark_id=self.spec.benchmark_id,
                    version=self.spec.version,
                    dataset_ids=self.spec.dataset_ids,
                    records=records,
                    aggregation_mode=self.spec.aggregation_mode,
                    manifest_hash=manifest_hash,
                    missing_datasets_by_model={
                        str(model_id): tuple(str(dataset_id) for dataset_id in dataset_ids)
                        for model_id, dataset_ids in diagnostics.get("missing_datasets_by_model", {}).items()
                    },
                    warnings=warnings,
                )

        bundle = self.parse(raw).with_manifest_hash(manifest_hash)
        if use_cache:
            write_json(
                manifest_path,
                {
                    **self.spec.manifest,
                    "manifest_hash": manifest_hash,
                    "raw_source_hash": raw.source_hash,
                    "raw_source_id": raw.source_id,
                    "raw_root": portable_path(raw.root),
                    "diagnostics": bundle.diagnostics(),
                },
            )
            write_records_parquet(records_path, bundle.records)
        return bundle

    @abstractmethod
    def parse(self, raw: RawSnapshot) -> BenchmarkBundle:
        raise NotImplementedError

    def _normalized_manifest_hash(self, raw: RawSnapshot) -> str:
        return stable_json_hash(
            {
                "benchmark_spec": self.spec.manifest,
                "raw_source_hash": raw.source_hash,
            }
        )
