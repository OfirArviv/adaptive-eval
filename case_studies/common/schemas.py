from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class CaseStudyArtifactPaths:
    root_dir: Path
    config_path: Path
    raw_csv_path: Path
    summary_csv_path: Path
    narrative_path: Path
    figure_paths: List[Path] = field(default_factory=list)
    input_config_path: Optional[Path] = None


@dataclass(frozen=True)
class CaseStudyResult:
    study_name: str
    dataset_label: str
    config: Dict[str, Any]
    raw_rows: List[Dict[str, Any]]
    summary_rows: List[Dict[str, Any]]
    narrative: str
    input_config: Optional[Dict[str, Any]] = None
    figures: Mapping[str, Any] = field(default_factory=dict)
    figure_paths: Sequence[Path | str] = field(default_factory=tuple)


@dataclass(frozen=True)
class ModelPair:
    model_a: str
    model_b: str

    @property
    def pair_id(self) -> str:
        return f"{self.model_a};{self.model_b}"
