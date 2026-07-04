from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import Field, asdict
from typing import Any, ClassVar, Generic, Mapping, Protocol, Sequence, TypeVar

from case_studies.common.schemas import CaseStudyResult
from case_studies.common.utils import log_stage_duration, log_stage_start
from data.benchmark import Benchmark
from data.core import BenchmarkBundle


class DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


InputConfig = TypeVar("InputConfig", bound=DataclassInstance)
StudyConfig = TypeVar("StudyConfig", bound=DataclassInstance)
StudyState = TypeVar("StudyState", bound=DataclassInstance)
StudyRawRow = TypeVar("StudyRawRow", bound=DataclassInstance)
StudySummaryRow = TypeVar("StudySummaryRow", bound=DataclassInstance)


class BaseCaseStudy(
    ABC,
    Generic[InputConfig, StudyConfig, StudyState, StudyRawRow, StudySummaryRow],
):
    study_name: str

    def run(self, benchmark: Benchmark, config: InputConfig) -> CaseStudyResult:
        started_at = time.perf_counter()
        self.validate_input_config(config)
        load_started_at = time.perf_counter()
        log_stage_start(self.study_name, self.verbose(config), "benchmark load")
        bundle = benchmark.load(use_cache=self.use_cache(config))
        log_stage_duration(self.study_name, self.verbose(config), "benchmark load", load_started_at)
        log_stage_start(self.study_name, self.verbose(config), "state build")
        state_started_at = time.perf_counter()
        state = self.build_state(bundle, benchmark, config)
        log_stage_duration(self.study_name, self.verbose(config), "state build", state_started_at)
        log_stage_start(self.study_name, self.verbose(config), "study config build")
        config_started_at = time.perf_counter()
        study_config = self.build_study_config(bundle, config, state)
        log_stage_duration(self.study_name, self.verbose(config), "study config build", config_started_at)
        log_stage_start(self.study_name, self.verbose(config), "raw rows")
        raw_rows_started_at = time.perf_counter()
        raw_rows = self.build_raw_rows(state, config, study_config)
        log_stage_duration(self.study_name, self.verbose(config), "raw rows", raw_rows_started_at)
        log_stage_start(self.study_name, self.verbose(config), "summary rows")
        summary_rows_started_at = time.perf_counter()
        summary_rows = self.build_summary_rows(raw_rows, state, config, study_config)
        log_stage_duration(self.study_name, self.verbose(config), "summary rows", summary_rows_started_at)
        log_stage_start(self.study_name, self.verbose(config), "narrative")
        narrative_started_at = time.perf_counter()
        narrative = self.build_narrative(raw_rows, summary_rows, state, config, study_config)
        log_stage_duration(self.study_name, self.verbose(config), "narrative", narrative_started_at)
        log_stage_start(self.study_name, self.verbose(config), "figures")
        figures_started_at = time.perf_counter()
        figures = self.build_figures(raw_rows, summary_rows, state, config, study_config)
        log_stage_duration(self.study_name, self.verbose(config), "figures", figures_started_at)
        log_stage_duration(self.study_name, self.verbose(config), "total study runtime", started_at)
        return CaseStudyResult(
            study_name=self.study_name,
            dataset_label=bundle.benchmark_id,
            config=asdict(study_config),
            input_config=asdict(config),
            raw_rows=[asdict(row) for row in raw_rows],
            summary_rows=[asdict(row) for row in summary_rows],
            narrative=narrative,
            figures=figures,
        )

    def use_cache(self, config: InputConfig) -> bool:
        return bool(getattr(config, "use_cache"))

    def verbose(self, config: InputConfig) -> bool:
        return bool(getattr(config, "verbose", False))

    @abstractmethod
    def validate_input_config(self, config: InputConfig) -> None:
        raise NotImplementedError

    @abstractmethod
    def build_state(
        self,
        bundle: BenchmarkBundle,
        benchmark: Benchmark,
        config: InputConfig,
    ) -> StudyState:
        raise NotImplementedError

    @abstractmethod
    def build_study_config(
        self,
        bundle: BenchmarkBundle,
        config: InputConfig,
        state: StudyState,
    ) -> StudyConfig:
        raise NotImplementedError

    @abstractmethod
    def build_raw_rows(
        self,
        state: StudyState,
        config: InputConfig,
        study_config: StudyConfig,
    ) -> list[StudyRawRow]:
        raise NotImplementedError

    @abstractmethod
    def build_summary_rows(
        self,
        raw_rows: Sequence[StudyRawRow],
        state: StudyState,
        config: InputConfig,
        study_config: StudyConfig,
    ) -> list[StudySummaryRow]:
        raise NotImplementedError

    @abstractmethod
    def build_narrative(
        self,
        raw_rows: Sequence[StudyRawRow],
        summary_rows: Sequence[StudySummaryRow],
        state: StudyState,
        config: InputConfig,
        study_config: StudyConfig,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_figures(
        self,
        raw_rows: Sequence[StudyRawRow],
        summary_rows: Sequence[StudySummaryRow],
        state: StudyState,
        config: InputConfig,
        study_config: StudyConfig,
    ) -> Mapping[str, Any]:
        raise NotImplementedError
