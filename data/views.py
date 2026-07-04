from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from data.benchmark import Benchmark
from data.cache import (
    TRANSFORM_VERSION,
    ordering_cache_dir,
    pairwise_cache_dir,
    pairwise_ordering_cache_dir,
    read_ordering_parquet,
    read_pairwise_parquet,
    stable_json_hash,
    write_json,
    write_ordering_parquet,
    write_pairwise_parquet,
)
from data.core import BenchmarkBundle, PairwiseBundle, PairwiseRecord


MissingPolicy = Literal["error", "intersect", "report"]


@dataclass(frozen=True)
class OrderingEntry:
    dataset_id: str
    instance_id: str
    order_index: int


@dataclass(frozen=True)
class OrderedBenchmarkView:
    benchmark: Benchmark
    seed: int

    def load(self, *, use_cache: bool = True) -> BenchmarkBundle:
        bundle = self.benchmark.load(use_cache=use_cache)
        cache_path = ordering_cache_dir(bundle.benchmark_id, bundle.version, bundle.manifest_hash) / (
            f"scope=dataset_seed={self.seed}_{TRANSFORM_VERSION}.parquet"
        )
        if use_cache and cache_path.exists():
            ordering = _read_ordering(cache_path)
        else:
            ordering = _build_ordering(bundle, self.seed)
            if use_cache:
                write_ordering_parquet(cache_path, _ordering_rows(ordering))
        return _apply_ordering(bundle, ordering)


@dataclass(frozen=True)
class PairwiseBenchmarkView:
    benchmark: Benchmark
    pairs: Sequence[tuple[str, str]]
    missing_policy: MissingPolicy = "error"

    def load(self, *, use_cache: bool = True) -> PairwiseBundle:
        bundle = self.benchmark.load(use_cache=use_cache)
        cache_hash = _pairwise_hash(self.pairs, self.missing_policy)
        cache_dir = pairwise_cache_dir(bundle.benchmark_id, bundle.version, bundle.manifest_hash, cache_hash)
        records_path = cache_dir / "records.parquet"
        manifest_path = cache_dir / "manifest.json"

        if use_cache and records_path.exists():
            return PairwiseBundle(
                benchmark_id=bundle.benchmark_id,
                version=bundle.version,
                source_manifest_hash=bundle.manifest_hash,
                pairset_hash=cache_hash,
                records=read_pairwise_parquet(records_path),
            )

        pairwise = _build_pairwise(bundle, self.pairs, self.missing_policy, cache_hash)
        if use_cache:
            write_json(
                manifest_path,
                {
                    "benchmark_id": bundle.benchmark_id,
                    "version": bundle.version,
                    "source_manifest_hash": bundle.manifest_hash,
                    "pairset_hash": cache_hash,
                    "missing_policy": self.missing_policy,
                    "transform_version": TRANSFORM_VERSION,
                    "pairs": [{"model_a": a, "model_b": b} for a, b in self.pairs],
                },
            )
            write_pairwise_parquet(records_path, pairwise.records)
        return pairwise


@dataclass(frozen=True)
class PairwiseOrderedBenchmarkView:
    benchmark: Benchmark
    pairs: Sequence[tuple[str, str]]
    seed: int
    missing_policy: MissingPolicy = "error"

    def load(self, *, use_cache: bool = True) -> PairwiseBundle:
        pairwise = PairwiseBenchmarkView(
            benchmark=self.benchmark,
            pairs=self.pairs,
            missing_policy=self.missing_policy,
        ).load(use_cache=use_cache)
        cache_dir = pairwise_ordering_cache_dir(
            pairwise.benchmark_id,
            pairwise.version,
            pairwise.source_manifest_hash,
            pairwise.pairset_hash,
        )
        cache_path = cache_dir / f"scope=pair_dataset_seed={self.seed}_{TRANSFORM_VERSION}.parquet"

        if use_cache and cache_path.exists():
            rows = read_ordering_parquet(cache_path)
        else:
            rows = _build_pairwise_ordering(pairwise, self.seed)
            if use_cache:
                write_ordering_parquet(cache_path, rows)
        return _apply_pairwise_ordering(pairwise, rows)


def build_pairwise_bundle(
    bundle: BenchmarkBundle,
    pairs: Sequence[tuple[str, str]],
    missing_policy: MissingPolicy = "error",
    verbose: bool = False,
) -> PairwiseBundle:
    return _build_pairwise(
        bundle=bundle,
        pairs=pairs,
        missing_policy=missing_policy,
        pairset_hash=_pairwise_hash(pairs, missing_policy),
        verbose=verbose,
    )


def _dataset_seed(seed: int, dataset_id: str) -> int:
    return seed + sum(ord(ch) for ch in dataset_id) * 9973


def _build_ordering(bundle: BenchmarkBundle, seed: int) -> tuple[OrderingEntry, ...]:
    entries: list[OrderingEntry] = []
    for dataset_id in bundle.dataset_ids:
        instance_ids = sorted({record.instance_id for record in bundle.datasets[dataset_id].records})
        rng = np.random.default_rng(_dataset_seed(seed, dataset_id))
        for index, instance_id in enumerate(rng.permutation(instance_ids).tolist()):
            entries.append(OrderingEntry(dataset_id=dataset_id, instance_id=str(instance_id), order_index=index))
    return tuple(entries)


def _apply_ordering(bundle: BenchmarkBundle, ordering: tuple[OrderingEntry, ...]) -> BenchmarkBundle:
    order_index = {(entry.dataset_id, entry.instance_id): entry.order_index for entry in ordering}
    return bundle.with_records(
        sorted(
            bundle.records,
            key=lambda record: (
                record.model_id,
                record.dataset_id,
                order_index[(record.dataset_id, record.instance_id)],
            ),
        )
    )


def _read_ordering(path) -> tuple[OrderingEntry, ...]:
    return tuple(
        OrderingEntry(
            dataset_id=str(row["dataset_id"]),
            instance_id=str(row["instance_id"]),
            order_index=int(row["order_index"]),
        )
        for row in read_ordering_parquet(path)
    )


def _ordering_rows(ordering: tuple[OrderingEntry, ...]) -> tuple[dict[str, object], ...]:
    return tuple(
        {"dataset_id": entry.dataset_id, "instance_id": entry.instance_id, "order_index": entry.order_index}
        for entry in ordering
    )


def _pairwise_hash(pairs: Sequence[tuple[str, str]], missing_policy: MissingPolicy) -> str:
    return stable_json_hash(
        {
            "pairs": [{"model_a": a, "model_b": b} for a, b in pairs],
            "missing_policy": missing_policy,
            "transform_version": TRANSFORM_VERSION,
        }
    )


def _build_pairwise(
    bundle: BenchmarkBundle,
    pairs: Sequence[tuple[str, str]],
    missing_policy: MissingPolicy,
    pairset_hash: str,
    verbose: bool = False,
) -> PairwiseBundle:
    by_key = {(record.model_id, record.dataset_id, record.instance_id): record for record in bundle.records}
    records: list[PairwiseRecord] = []

    for model_a, model_b in _progress_iterator(pairs, verbose=verbose, desc="Build pairwise bundle", unit="pair"):
        if model_a not in bundle.model_ids or model_b not in bundle.model_ids:
            raise ValueError(f"Unknown model pair: {model_a!r}, {model_b!r}")

        pair_id = f"{model_a};{model_b}"
        a_keys = {
            (record.dataset_id, record.instance_id)
            for record in bundle.by_model[model_a]
        }
        b_keys = {
            (record.dataset_id, record.instance_id)
            for record in bundle.by_model[model_b]
        }
        if missing_policy == "error" and a_keys != b_keys:
            raise ValueError(
                f"Pair {pair_id} has mismatched instances: "
                f"missing_from_b={len(a_keys - b_keys)}, missing_from_a={len(b_keys - a_keys)}."
            )

        for dataset_id, instance_id in sorted(a_keys & b_keys):
            left = by_key[(model_a, dataset_id, instance_id)]
            right = by_key[(model_b, dataset_id, instance_id)]
            records.append(
                PairwiseRecord(
                    pair_id=pair_id,
                    model_a=model_a,
                    model_b=model_b,
                    dataset_id=dataset_id,
                    instance_id=instance_id,
                    score_a=float(left.score),
                    score_b=float(right.score),
                    diff=float(left.score) - float(right.score),
                    grouping=left.grouping or right.grouping,
                )
            )

    return PairwiseBundle(
        benchmark_id=bundle.benchmark_id,
        version=bundle.version,
        source_manifest_hash=bundle.manifest_hash,
        pairset_hash=pairset_hash,
        records=tuple(records),
    )


def _progress_iterator(items, *, verbose: bool, desc: str, unit: str):
    if not verbose:
        return items
    from tqdm.auto import tqdm
    return tqdm(items, desc=desc, unit=unit, dynamic_ncols=True, file=sys.stdout)


def _build_pairwise_ordering(pairwise: PairwiseBundle, seed: int) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for pair_id, datasets in pairwise.by_pair_dataset.items():
        for dataset_id, records in datasets.items():
            instance_ids = sorted({record.instance_id for record in records})
            rng = np.random.default_rng(_dataset_seed(seed, f"{pair_id}:{dataset_id}"))
            for index, instance_id in enumerate(rng.permutation(instance_ids).tolist()):
                rows.append(
                    {
                        "pair_id": pair_id,
                        "dataset_id": dataset_id,
                        "instance_id": str(instance_id),
                        "order_index": index,
                    }
                )
    return tuple(rows)


def _apply_pairwise_ordering(pairwise: PairwiseBundle, rows: tuple[dict[str, object], ...]) -> PairwiseBundle:
    order_index = {
        (str(row["pair_id"]), str(row["dataset_id"]), str(row["instance_id"])): int(row["order_index"])
        for row in rows
    }
    return pairwise.with_records(
        sorted(
            pairwise.records,
            key=lambda record: (
                record.pair_id,
                record.dataset_id,
                order_index[(record.pair_id, record.dataset_id, record.instance_id)],
            ),
        )
    )
