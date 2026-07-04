from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Mapping

from data.benchmark import Benchmark
from data.cache import RAW_ROOT, read_json
from data.core import BenchmarkBundle, BenchmarkRecord, BenchmarkSpec
from data.sources import LocalTreeSource, RawSnapshot, portable_path


class OpenVlmLeaderboard(Benchmark):
    benchmark_id = "open_vlm_leaderboard"
    excluded_models = ("Gemini-1.0-Pro",)
    dataset_files = {
        "ocr_bench": "*OCRBench.xlsx",
        "ai2d": "*AI2D_TEST_openai_result.xlsx",
        "mmmu_val": "*MMMU_DEV_VAL_openai_result.xlsx",
        "hallusion_bench": "*HallusionBench_auxmatch.xlsx",
        "mmstar": "*MMStar_openai_result.xlsx",
        "mmbench_v11": "*MMBench_V11.xlsx",
    }

    def __init__(
        self,
        version: str = "local_2025_03_05",
        *,
        benchmark_id: str | None = None,
        dataset_ids: tuple[str, ...] | None = None,
    ) -> None:
        self.benchmark_id = benchmark_id or self.__class__.benchmark_id
        self.selected_dataset_ids = dataset_ids or tuple(self.dataset_files)
        self.spec = BenchmarkSpec(
            benchmark_id=self.benchmark_id,
            version=version,
            raw_source=LocalTreeSource(
                source_id=f"{self.benchmark_id}:{version}",
                root=RAW_ROOT / self.__class__.benchmark_id / version,
                patterns=self._raw_patterns(),
                metadata={"metadata_file": "OpenVLM.json", "records_dir": "EvalRecords"},
            ),
            dataset_ids=self.selected_dataset_ids,
            aggregation_mode="mean_of_means",
            parser_version="open_vlm_v2",
        )

    def parse(self, raw: RawSnapshot) -> BenchmarkBundle:
        payload = read_json(raw.root / "OpenVLM.json")
        records_root = raw.root / "EvalRecords"
        records: list[BenchmarkRecord] = []
        missing: dict[str, tuple[str, ...]] = {}
        warnings: list[str] = []

        for model_id, model_payload in sorted(payload.get("results", {}).items()):
            if model_id in self.excluded_models:
                continue
            meta = model_payload.get("META", {})
            dir_name = meta.get("dir_name")
            if not dir_name:
                continue
            model_dir = records_root / str(dir_name)
            missing_for_model: list[str] = []

            for dataset_id in self.selected_dataset_ids:
                pattern = self.dataset_files[dataset_id]
                try:
                    file_path = self._find_one(model_dir, pattern)
                    records.extend(self._parse_dataset(dataset_id, model_id, file_path))
                except FileNotFoundError as exc:
                    missing_for_model.append(dataset_id)
                    warnings.append(str(exc))

            if missing_for_model:
                missing[str(model_id)] = tuple(missing_for_model)

        return BenchmarkBundle.from_records(
            spec=self.spec,
            records=records,
            manifest_hash=self.spec.manifest_hash,
            missing_datasets_by_model=missing,
            warnings=tuple(warnings),
        )

    def _parse_dataset(self, dataset_id: str, model_id: str, file_path: Path) -> list[BenchmarkRecord]:
        table = self._read_table(file_path)
        if dataset_id == "mmbench_v11":
            table = table[(table["split"] == "test") & table["answer"].notna()]

        rows: list[BenchmarkRecord] = []
        for row in table.to_dict("records"):
            rows.append(
                BenchmarkRecord(
                    dataset_id=dataset_id,
                    model_id=model_id,
                    instance_id=str(row["index"]),
                    score=float(self._score(dataset_id, row)),
                    grouping=self._optional(row.get("category")) if dataset_id == "mmstar" else None,
                    split=self._optional(row.get("split")) if dataset_id == "mmbench_v11" else None,
                    metadata={"source_file": portable_path(file_path)},
                )
            )
        return rows

    def _score(self, dataset_id: str, row: Mapping[str, Any]) -> float:
        if dataset_id in {"ai2d", "mmmu_val", "mmstar"}:
            return float(row["hit"])
        if dataset_id == "hallusion_bench":
            return 1.0 if bool(row["score"]) else 0.0
        if dataset_id == "ocr_bench":
            return self._score_ocr(row)
        if dataset_id == "mmbench_v11":
            return self._score_mmbench_v11(row)
        raise ValueError(f"Unsupported Open VLM dataset: {dataset_id}")

    def _score_ocr(self, row: Mapping[str, Any]) -> float:
        prediction = str(row["prediction"])
        answers = ast.literal_eval(str(row["answer"]))
        if row.get("category") == "Handwritten Mathematical Expression Recognition":
            normalized = prediction.strip().replace("\n", " ").replace(" ", "")
            return float(any(str(answer).strip().replace("\n", " ").replace(" ", "") in normalized for answer in answers))

        normalized = prediction.lower().strip().replace("\n", " ")
        return float(any(str(answer).lower().strip().replace("\n", " ") in normalized for answer in answers))

    def _score_mmbench_v11(self, row: Mapping[str, Any]) -> float:
        prediction = re.sub(r"[^a-zA-Z\s]", "", str(row["prediction"])).strip()
        answer = str(row["answer"]).strip()
        if prediction in {"A", "B", "C", "D"}:
            return float(prediction == answer)

        answer_text = re.sub(r"[^a-zA-Z\s]", "", str(row[answer]))
        answer_text = re.sub(r"\s+", " ", answer_text).lower().strip()
        normalized = re.sub(r"\s+", " ", prediction).lower().strip()
        answer_lower = answer.lower()

        if f"is {answer_lower} " in normalized or normalized.startswith(f"{answer_lower} "):
            return 1.0
        if "i dont know" in normalized or "i cant predict" in normalized:
            return 0.0
        if answer_text in normalized:
            return 1.0

        for option in ("A", "B", "C", "D"):
            option_lower = option.lower()
            if option != answer and str(row.get(option)) != "nan":
                option_text = re.sub(r"[^a-zA-Z\s]", "", str(row[option]))
                option_text = re.sub(r"\s+", " ", option_text).lower().strip()
                if option_text in normalized:
                    return 0.0
            if f" {option_lower} " in normalized or normalized.startswith(f"{option_lower} ") or normalized.endswith(option_lower):
                return 0.0
        return 0.0

    @staticmethod
    def _find_one(root: Path, pattern: str) -> Path:
        matches = sorted(root.glob(pattern))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected exactly one {pattern} under {portable_path(root)}, found {len(matches)}.")
        return matches[0]

    @staticmethod
    def _read_table(path: Path):
        import pandas as pd

        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        raise ValueError(f"Unsupported table format: {path}")

    @staticmethod
    def _optional(value: Any) -> str | None:
        if value is None or str(value) == "nan":
            return None
        return str(value)

    def _raw_patterns(self) -> tuple[str, ...]:
        return (
            "OpenVLM.json",
            *(
                f"EvalRecords/**/*{self.dataset_files[dataset_id].lstrip('*')}"
                for dataset_id in self.selected_dataset_ids
            ),
        )
