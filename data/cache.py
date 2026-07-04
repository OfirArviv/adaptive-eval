from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Iterable

from data.core import (
    BenchmarkRecord,
    PairwiseRecord,
)


DATA_PACKAGE_ROOT = Path(__file__).resolve().parent
RAW_ROOT = DATA_PACKAGE_ROOT / "raw"
CACHE_ROOT = DATA_PACKAGE_ROOT / "cache"
TRANSFORM_VERSION = "v1"


def stable_json_hash(value: Any, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def write_json(path: Path, value: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(value, file_obj, indent=2, sort_keys=True, default=str)
        file_obj.write("\n")


def benchmark_cache_root(benchmark_id: str, version: str, manifest_hash: str) -> Path:
    return CACHE_ROOT / benchmark_id / version / manifest_hash


def normalized_cache_dir(benchmark_id: str, version: str, manifest_hash: str) -> Path:
    return benchmark_cache_root(benchmark_id, version, manifest_hash) / "normalized"


def ordering_cache_dir(benchmark_id: str, version: str, manifest_hash: str) -> Path:
    return benchmark_cache_root(benchmark_id, version, manifest_hash) / "ordering"


def pairwise_cache_dir(benchmark_id: str, version: str, manifest_hash: str, pairset_hash: str) -> Path:
    return benchmark_cache_root(benchmark_id, version, manifest_hash) / "pairwise" / f"pairset={pairset_hash}"


def pairwise_ordering_cache_dir(benchmark_id: str, version: str, manifest_hash: str, pairset_hash: str) -> Path:
    return benchmark_cache_root(benchmark_id, version, manifest_hash) / "pairwise_ordering" / f"pairset={pairset_hash}"


def require_pyarrow():
    import pyarrow as pa
    import pyarrow.parquet as pq
    return pa, pq


def _metadata_to_json(metadata: dict[str, Any] | None) -> str | None:
    if metadata is None:
        return None
    return json.dumps(metadata, sort_keys=True, default=str)


def _metadata_from_json(value: str | None) -> dict[str, Any] | None:
    if value in (None, ""):
        return None
    return json.loads(value)


def write_records_parquet(path: Path, records: Iterable[BenchmarkRecord]) -> None:
    pa, pq = require_pyarrow()
    rows = [
        {
            "dataset_id": record.dataset_id,
            "model_id": record.model_id,
            "instance_id": record.instance_id,
            "score": float(record.score),
            "grouping": record.grouping,
            "split": record.split,
            "metadata_json": _metadata_to_json(record.metadata),
        }
        for record in records
    ]
    ensure_dir(path.parent)
    pq.write_table(pa.Table.from_pylist(rows), path)


def read_records_parquet(path: Path) -> tuple[BenchmarkRecord, ...]:
    _, pq = require_pyarrow()
    table = pq.read_table(path)
    return tuple(
        BenchmarkRecord(
            dataset_id=str(row["dataset_id"]),
            model_id=str(row["model_id"]),
            instance_id=str(row["instance_id"]),
            score=float(row["score"]),
            grouping=row.get("grouping"),
            split=row.get("split"),
            metadata=_metadata_from_json(row.get("metadata_json")),
        )
        for row in table.to_pylist()
    )


def write_ordering_parquet(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    pa, pq = require_pyarrow()
    ensure_dir(path.parent)
    pq.write_table(pa.Table.from_pylist(list(rows)), path)


def read_ordering_parquet(path: Path) -> tuple[dict[str, Any], ...]:
    _, pq = require_pyarrow()
    return tuple(pq.read_table(path).to_pylist())


def write_pairwise_parquet(path: Path, records: Iterable[PairwiseRecord]) -> None:
    pa, pq = require_pyarrow()
    rows = [
        {
            "pair_id": record.pair_id,
            "model_a": record.model_a,
            "model_b": record.model_b,
            "dataset_id": record.dataset_id,
            "instance_id": record.instance_id,
            "score_a": float(record.score_a),
            "score_b": float(record.score_b),
            "diff": float(record.diff),
            "grouping": record.grouping,
        }
        for record in records
    ]
    ensure_dir(path.parent)
    pq.write_table(pa.Table.from_pylist(rows), path)


def read_pairwise_parquet(path: Path) -> tuple[PairwiseRecord, ...]:
    _, pq = require_pyarrow()
    table = pq.read_table(path)
    return tuple(
        PairwiseRecord(
            pair_id=str(row["pair_id"]),
            model_a=str(row["model_a"]),
            model_b=str(row["model_b"]),
            dataset_id=str(row["dataset_id"]),
            instance_id=str(row["instance_id"]),
            score_a=float(row["score_a"]),
            score_b=float(row["score_b"]),
            diff=float(row["diff"]),
            grouping=row.get("grouping"),
        )
        for row in table.to_pylist()
    )
