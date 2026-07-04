from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from case_studies.common.schemas import CaseStudyArtifactPaths, CaseStudyResult
from case_studies.common.utils import make_case_study_run_id

_REPO_ROOT = Path(__file__).resolve().parents[2]


def save_case_study_result(
    result: CaseStudyResult,
    run_id: Optional[str] = None,
    output_root: Path | str = "outputs/case_studies",
) -> CaseStudyArtifactPaths:
    resolved_run_id = run_id or make_case_study_run_id(result.config)
    root_dir = (
        _resolve_output_root(output_root)
        / _safe_path_part(result.study_name)
        / _safe_path_part(result.dataset_label)
        / _safe_path_part(resolved_run_id)
    )
    figures_dir = root_dir / "figures"
    root_dir.mkdir(parents=True, exist_ok=True)

    config_path = root_dir / "config.json"
    raw_csv_path = root_dir / "raw.csv"
    summary_csv_path = root_dir / "summary.csv"
    narrative_path = root_dir / "narrative.md"
    input_config_path = root_dir / "input_config.json" if result.input_config is not None else None

    config_path.write_text(
        json.dumps(result.config, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    if input_config_path is not None:
        input_config_path.write_text(
            json.dumps(result.input_config, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    _write_csv(raw_csv_path, result.raw_rows)
    _write_csv(summary_csv_path, result.summary_rows)
    narrative_path.write_text(result.narrative, encoding="utf-8")

    figure_paths = _save_figures(
        figures_dir=figures_dir,
        figures=result.figures,
        source_paths=result.figure_paths,
    )

    print(f"Saved study to {root_dir}")
    for figure_path in figure_paths:
        print(f"Saved figure to {figure_path}")

    return CaseStudyArtifactPaths(
        root_dir=root_dir,
        config_path=config_path,
        raw_csv_path=raw_csv_path,
        summary_csv_path=summary_csv_path,
        narrative_path=narrative_path,
        figure_paths=figure_paths,
        input_config_path=input_config_path,
    )


def _resolve_output_root(output_root: Path | str) -> Path:
    root = Path(output_root)
    if root.is_absolute():
        return root
    return _REPO_ROOT / root


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = _fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        if not fieldnames:
            return
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({fieldname: row.get(fieldname) for fieldname in fieldnames})


def _fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(str(key))
    return fieldnames


def _save_figures(
    figures_dir: Path,
    figures: Mapping[str, Any],
    source_paths: Iterable[Path | str],
) -> list[Path]:
    saved_paths: list[Path] = []
    source_path_list = list(source_paths)

    if figures or source_path_list:
        figures_dir.mkdir(parents=True, exist_ok=True)

    for name, figure in figures.items():
        path = figures_dir / f"{_safe_figure_name(name)}.png"
        if not hasattr(figure, "savefig"):
            raise TypeError(f"Figure {name!r} does not provide a savefig method.")
        figure.savefig(path, format="png", bbox_inches="tight")
        saved_paths.append(path)

    for index, source_path in enumerate(source_path_list):
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Figure path does not exist: {source}")
        suffix = source.suffix.lower()
        filename = source.name if suffix == ".png" else f"figure_{index}.png"
        destination = figures_dir / _safe_figure_name(filename, keep_suffix=True)
        shutil.copyfile(source, destination)
        saved_paths.append(destination)

    return saved_paths


def _safe_path_part(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")
    return safe or "unnamed"


def _safe_figure_name(value: str, keep_suffix: bool = False) -> str:
    safe = _safe_path_part(value)
    if keep_suffix:
        return safe
    return safe.removesuffix(".png")
