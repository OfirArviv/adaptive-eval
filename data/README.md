# Data Layer

This package loads benchmark raw data into one normalized row format and builds cached views for repeated experiments.

The mental model is:

```text
BenchmarkSpec -> RawSource -> RawSnapshot -> Benchmark.load() -> BenchmarkBundle -> Views
```

## Main Objects

### `Benchmark`

`Benchmark` is the lifecycle object for one benchmark.

```python
benchmark = OpenVlmLeaderboard()
bundle = benchmark.load()
raw = benchmark.raw()
```

Every benchmark class owns:

- `spec`: benchmark identity, raw source, dataset ids, aggregation mode, parser version.
- `parse(raw)`: benchmark-specific conversion from raw files to normalized records.

The base class owns:

- raw-source resolution via `raw()`
- normalized Parquet cache lookup/write via `load()`
- normalized cache key construction from benchmark spec + raw snapshot hash

Benchmark-specific parsing stays inside the benchmark class. There is no scorer registry and no dataset parser config.

### `BenchmarkSpec`

`BenchmarkSpec` is the benchmark-level cache contract.

It lives in `core.py` with the normalized benchmark data types.

Fields:

- `benchmark_id`: stable id, e.g. `open_vlm_leaderboard`.
- `version`: benchmark/raw snapshot version, e.g. `local_2025_03_05`.
- `raw_source`: typed source declaration from `sources.py`.
- `dataset_ids`: expected normalized datasets and their order.
- `aggregation_mode`: default ranking aggregation, currently `mean_of_means` or `pooled_mean`.
- `parser_version`: bump this when parsing/scoring logic changes.

The spec is intentionally small. Benchmark-specific details like file patterns, excluded models, and scoring logic live in the benchmark class.

### `RawSource`, `RawSnapshot`, `RawFile`

Raw source declarations and raw-source resolution live in `sources.py`.

Examples:

```python
LocalDirectorySource(...)
LocalGlobSource(...)
GitHubTreeSource(...)
HfDatasetSource(...)
CompositeSource(...)
```

A `RawSource` says where data comes from. A `RawSnapshot` says what local files exist after resolving that source.

```text
RawSource
  declaration: local directory, glob, GitHub tree, HF dataset, etc.

RawSnapshot
  resolved local raw input for one benchmark/version

RawFile
  one file inside the resolved snapshot
```

`RawSnapshot` has:

- `source_id`
- `source_hash`
- `root`
- `files`

`RawFile` has:

- `path`: absolute local path
- `logical_path`: path relative to the snapshot root

Parsers can use either:

```python
raw.root / "OpenVLM.json"
```

or:

```python
for file in raw.files:
    ...
```

Use `raw.root` when the raw snapshot has meaningful structure, like Open VLM. Use `raw.files` when the source is a resolved file list, like GitHub results, local globs, or materialized HF datasets.

## Raw Resolution

`resolve_raw_source(source)` turns a `RawSource` into a `RawSnapshot`.

Supported source types:

- `LocalDirectorySource`: indexes all files under a local directory.
- `LocalFileSource`: resolves one local file.
- `LocalGlobSource`: resolves a local glob into files.
- `GitHubTreeSource`: downloads matching files from a GitHub repo revision into `raw/_resolved/...`.
- `RawUrlSource`: downloads one URL into `raw/_resolved/...`.
- `HfHubFilesSource`: downloads matching files from Hugging Face Hub.
- `HfDatasetSource`: loads a Hugging Face dataset split and materializes it to local Parquet.
- `CompositeSource`: combines multiple sources into one snapshot.

For HF datasets, parsers should not call `load_dataset()` directly. The source resolver materializes the dataset first:

```text
HfDatasetSource -> dataset.parquet -> RawSnapshot(files=(RawFile(...),))
```

Then the benchmark parser reads the local Parquet file. This keeps network behavior and cache behavior out of parsing.

## Normalized Data

All benchmarks normalize into `BenchmarkRecord` rows:

```python
BenchmarkRecord(
    dataset_id="ai2d",
    model_id="GPT4o_MINI",
    instance_id="123",
    score=1.0,
    grouping=None,
    split=None,
    metadata={"source_file": "..."},
)
```

The canonical stored shape is flat records because it works well for Parquet, filtering, ordering, pairwise joins, and debugging.

`BenchmarkBundle` wraps those records and exposes grouped views:

- `bundle.records`
- `bundle.datasets`
- `bundle.by_model`
- `bundle.by_model_dataset`
- `bundle.dataset_sizes`
- `bundle.rank_models()`
- `bundle.to_streams(model_id)`

`BenchmarkDataset` is a convenience view over one dataset:

```python
dataset = bundle.get_dataset("ai2d")
records_by_model = dataset.by_model
```

## Caching

Cache files live under:

```text
data/cache/<benchmark_id>/<version>/<manifest_hash>/
```

The normalized cache is:

```text
normalized/
  manifest.json
  records.parquet
```

The normalized manifest hash includes:

- benchmark spec manifest
- resolved raw snapshot hash

This means the normalized cache changes when:

- raw files change
- parser version changes
- dataset ids change
- aggregation mode changes
- raw source declaration changes

`manifest.json` also stores diagnostics:

- record counts
- model counts
- records per dataset
- models per dataset
- missing datasets by model
- warnings

There is no separate quality-report object or quality-report cache file.

## Views

Views are derived representations over a loaded benchmark. They are separate from the benchmark class because experiments repeatedly create different views over the same benchmark.

### Ordered Benchmark View

```python
ordered = OrderedBenchmarkView(benchmark, seed=17).load()
```

This returns a `BenchmarkBundle` with records ordered by dataset-specific seeded order.

Ordering cache:

```text
ordering/
  scope=dataset_seed=<seed>_<transform_version>.parquet
```

The ordering plan stores:

```text
dataset_id
instance_id
order_index
```

### Pairwise Benchmark View

```python
pairwise = PairwiseBenchmarkView(
    benchmark,
    pairs=[("model_a", "model_b")],
    missing_policy="error",
).load()
```

Pairwise matching is by:

```text
model_id + dataset_id + instance_id
```

The pairwise score is:

```python
diff = score_a - score_b
```

Pairwise cache:

```text
pairwise/pairset=<pairset_hash>/
  manifest.json
  records.parquet
```

### Pairwise Ordered View

```python
ordered_pairwise = PairwiseOrderedBenchmarkView(
    benchmark,
    pairs=[("model_a", "model_b")],
    seed=17,
).load()
```

Pairwise ordering is cached separately because pairwise records have `pair_id` as part of the ordering scope.

```text
pairwise_ordering/pairset=<pairset_hash>/
  scope=pair_dataset_seed=<seed>_<transform_version>.parquet
```

## Current Benchmarks

### Registered IDs

`BENCHMARK_IDS` contains the legacy-compatible labels:

```text
alpaca_eval_2.0
mmlu_pro
auto_rag_1st_batch
arena_hard
helm_qa
helm_summarization
helm_ir
helm_sentiment_analysis
helm_toxicity_detection
helm_text_classification
helm
open_llm_leaderboard_ifeval
open_llm_leaderboard_bbh
open_llm_leaderboard_math
open_llm_leaderboard_gpqa
open_llm_leaderboard_musr
open_llm_leaderboard_mmlu_pro
open_llm_leaderboard
open_vlm_leaderboard_ocr_bench
open_vlm_leaderboard_ai2d
open_vlm_leaderboard_mmmu_val
open_vlm_leaderboard_hallusion_bench
open_vlm_leaderboard_mmstar
open_vlm_leaderboard_mmbench
open_vlm_leaderboard
```

Single-dataset legacy labels are implemented as aliases over the same benchmark classes. For example, `open_vlm_leaderboard_ai2d` uses the Open VLM parser but declares only `dataset_ids=("ai2d",)`.

### Open VLM Leaderboard

File:

```text
benchmarks/open_vlm_leaderboard/benchmark.py
```

Source:

```python
LocalTreeSource(
    root=RAW_ROOT / "open_vlm_leaderboard" / "local_2025_03_05",
    patterns=("OpenVLM.json", "EvalRecords/**/*AI2D_TEST_openai_result.xlsx", ...)
)
```

Parser uses `raw.root` because Open VLM has a meaningful directory structure. The source indexes only the files used by the selected dataset ids, so unrelated files do not invalidate caches.

```text
OpenVLM.json
EvalRecords/<model_dir>/<result files>
```

### MMLU-Pro

File:

```text
benchmarks/mmlu_pro.py
```

Source:

```python
GitHubTreeSource(
    repo_id="TIGER-AI-Lab/MMLU-Pro",
    revision=<commit>,
    pattern="^eval_results/[^/]+\\.json$",
)
```

Parser iterates `raw.files` because GitHub resolution produces a file list.

### Auto RAG First Batch

File:

```text
benchmarks/auto_rag.py
```

Source:

```python
LocalGlobSource(
    pattern=Path("data/datasets/auto_rag_1st_batch/WatsonX benchmark results/*.csv")
)
```

Parser iterates `raw.files` because the source is a glob of CSV files.

### Other Migrated Benchmarks

- `AlpacaEval20`: uses the tagged AlpacaEval leaderboard CSV plus GitHub annotation files.
- `ArenaHard`: uses `HfHubFilesSource` against the Arena Hard Hugging Face Space JSONL judgments.
- `HelmBenchmark`: uses local HELM aggregate CSVs and supports both category aliases and the aggregate `helm` id.
- `OpenLlmLeaderboard`: uses the local scraped leaderboard CSV to pick top models, then loads the matching Hugging Face details datasets for the requested subsets.

## Adding A Benchmark

Create a benchmark class:

```python
class MyBenchmark(Benchmark):
    def __init__(self, version: str = "local") -> None:
        self.spec = BenchmarkSpec(
            benchmark_id="my_benchmark",
            version=version,
            raw_source=LocalDirectorySource(...),
            dataset_ids=("dataset_a", "dataset_b"),
            aggregation_mode="mean_of_means",
            parser_version="my_benchmark_v1",
        )

    def parse(self, raw: RawSnapshot) -> BenchmarkBundle:
        records = []
        ...
        return BenchmarkBundle.from_records(
            spec=self.spec,
            records=records,
            manifest_hash=self.spec.manifest_hash,
        )
```

Then register it in `data/registry.py` if it should be available through `get_benchmark()`.
