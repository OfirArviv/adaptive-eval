from __future__ import annotations

import hashlib
import itertools
import json
import time
import sys
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Iterable, Literal, Mapping, TypeVar, Sequence, Dict, List, Optional
from case_studies.common.configs import PairStrategy
from case_studies.common.schemas import ModelPair
from data.core import BenchmarkBundle
from sequential import  MeanOfMeansAggregator, PooledAggregator
from sequential.aggregators import ScoreAggregator
from sequential.types import AggregationMode

T = TypeVar("T")
RUN_ID_IGNORED_KEYS = {
    "benchmark_id",
    "benchmark_version",
    "selected_model_ids",
    "pair_ids",
    "missing_policy",
    "use_cache",
    "verbose",
    "include_plots",
}
MAX_RUN_ID_LENGTH = 120
RUN_ID_HASH_LENGTH = 10

def replicate_iterator(n_replicates: int, verbose: bool):
    return progress_iterator(range(n_replicates), verbose=verbose, desc="Replicates", unit="replicate")


def progress_iterator(items: Iterable[T], *, verbose: bool, desc: str, unit: str):
    if not verbose:
        return items
    from tqdm.auto import tqdm
    return tqdm(items, desc=desc, unit=unit, dynamic_ncols=True, file=sys.stdout)


def log_stage_duration(study_name: str, verbose: bool, label: str, started_at: float) -> None:
    if not verbose:
        return
    elapsed = time.perf_counter() - started_at
    print(f"[{study_name}] {label}: {elapsed:.2f}s", flush=True)


def log_stage_start(study_name: str, verbose: bool, label: str) -> None:
    if not verbose:
        return
    print(f"[{study_name}] starting {label}...", flush=True)


def make_case_study_run_id(config: Mapping[str, Any] | Any) -> str:
    if is_dataclass(config):
        config_dict = asdict(config)
    elif isinstance(config, Mapping):
        config_dict = dict(config)
    else:
        raise TypeError("config must be a dataclass or mapping.")

    stable_config = {
        key: config_dict[key]
        for key in sorted(config_dict)
        if key not in RUN_ID_IGNORED_KEYS
    }
    parts = [f"{key}_{_slugify_value(stable_config[key])}" for key in sorted(stable_config)]
    run_id = "__".join(parts) if parts else "run"
    if len(run_id) <= MAX_RUN_ID_LENGTH:
        return run_id
    digest = hashlib.sha256(
        json.dumps(stable_config, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()[:RUN_ID_HASH_LENGTH]
    prefix_length = MAX_RUN_ID_LENGTH - RUN_ID_HASH_LENGTH - 2
    prefix = run_id[:prefix_length].rstrip("_")
    return f"{prefix}__{digest}"



def _slugify_value(value: Any) -> str:
    text = str(value).strip()
    safe = "".join(character if character.isalnum() or character in "._-" else "_" for character in text)
    return safe.strip("_") or "none"



@dataclass(frozen=True)
class SelectedModelPairs:
    ranked_model_ids: List[str]
    selected_model_ids: List[str]
    rank_by_model: Dict[str, int]
    pairs: List[ModelPair]


def create_model_pairs(
    model_ids: Sequence[str],
    strategy: PairStrategy,
    baseline_model_id: Optional[str] = None,
) -> List[ModelPair]:
    resolved_model_ids = list(model_ids)
    if len(set(resolved_model_ids)) != len(resolved_model_ids):
        raise ValueError("model_ids must be unique.")

    if strategy == "all":
        return [
            ModelPair(model_a=model_a, model_b=model_b)
            for model_a, model_b in itertools.combinations(resolved_model_ids, 2)
        ]

    if strategy == "adjacent":
        return [
            ModelPair(model_a=model_a, model_b=model_b)
            for model_a, model_b in zip(resolved_model_ids, resolved_model_ids[1:])
        ]

    if strategy == "baseline_vs_all":
        if baseline_model_id is None:
            raise ValueError("baseline_model_id is required for baseline_vs_all pairs.")
        if baseline_model_id not in resolved_model_ids:
            raise ValueError(f"Unknown baseline_model_id: {baseline_model_id!r}")
        return [
            ModelPair(model_a=baseline_model_id, model_b=model_id)
            for model_id in resolved_model_ids
            if model_id != baseline_model_id
        ]

    raise ValueError(f"Unknown pair strategy: {strategy!r}")


def select_model_pairs(
    bundle: BenchmarkBundle,
    *,
    top_k: int,
    pair_strategy: PairStrategy = "all",
    baseline_model_id: Optional[str] = None,
    complete_only: bool = True,
) -> SelectedModelPairs:
    if top_k < 2:
        raise ValueError("top_k must be at least 2.")

    ranked_models = bundle.rank_models(complete_only=complete_only)
    if len(ranked_models) < top_k:
        raise ValueError(
            f"Requested top_k={top_k}, but benchmark has only {len(ranked_models)} "
            f"{'complete ' if complete_only else ''}models."
        )

    ranked_model_ids = [model_id for model_id, _score in ranked_models]
    selected_model_ids = ranked_model_ids[:top_k]
    return SelectedModelPairs(
        ranked_model_ids=ranked_model_ids,
        selected_model_ids=selected_model_ids,
        rank_by_model={
            model_id: rank
            for rank, model_id in enumerate(ranked_model_ids, start=1)
        },
        pairs=create_model_pairs(
            selected_model_ids,
            pair_strategy,
            baseline_model_id=baseline_model_id,
        ),
    )



def make_aggregator(aggregation_mode: AggregationMode) -> ScoreAggregator:
    if aggregation_mode == "mean_of_means":
        return MeanOfMeansAggregator()
    if aggregation_mode == "pooled_mean":
        return PooledAggregator()
    raise ValueError("aggregation_mode must be 'mean_of_means' or 'pooled_mean'.")
