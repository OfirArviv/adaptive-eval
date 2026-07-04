from __future__ import annotations
import base64
import fnmatch
import os
import re
import shutil
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypeAlias
from dotenv import load_dotenv

from data.cache import DATA_PACKAGE_ROOT, RAW_ROOT, ensure_dir, stable_json_hash


REPO_ROOT = DATA_PACKAGE_ROOT.parent


@dataclass(frozen=True)
class LocalDirectorySource:
    source_id: str
    root: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalTreeSource:
    source_id: str
    root: Path
    patterns: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalFileSource:
    source_id: str
    path: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalGlobSource:
    source_id: str
    pattern: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GitHubTreeSource:
    source_id: str
    repo_id: str
    revision: str
    pattern: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RawUrlSource:
    source_id: str
    url: str
    filename: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HfHubFilesSource:
    source_id: str
    repo_id: str
    pattern: str
    repo_type: Literal["dataset", "space", "model"] = "dataset"
    revision: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HfDatasetSource:
    source_id: str
    repo_id: str
    split: str
    config: str | None = None
    revision: str | None = None
    filename: str = "dataset.parquet"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompositeSource:
    source_id: str
    children: tuple[RawSource, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


RawSource: TypeAlias = (
    LocalDirectorySource
    | LocalTreeSource
    | LocalFileSource
    | LocalGlobSource
    | GitHubTreeSource
    | RawUrlSource
    | HfHubFilesSource
    | HfDatasetSource
    | CompositeSource
)


@dataclass(frozen=True)
class RawFile:
    path: Path
    logical_path: str


@dataclass(frozen=True)
class RawSnapshot:
    source_id: str
    source_hash: str
    root: Path
    files: tuple[RawFile, ...]

    def one_file(self, pattern: str) -> Path:
        matches = [file.path for file in self.files if fnmatch.fnmatch(file.logical_path, pattern)]
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected exactly one raw file matching {pattern!r}, found {len(matches)}.")
        return matches[0]


def source_manifest(source: RawSource) -> dict[str, Any]:
    if isinstance(source, LocalDirectorySource):
        data: dict[str, Any] = {
            "kind": "LocalDirectorySource",
            "source_id": source.source_id,
            "root": portable_path(source.root),
            "metadata": source.metadata,
        }
    elif isinstance(source, LocalTreeSource):
        data = {
            "kind": "LocalTreeSource",
            "source_id": source.source_id,
            "root": portable_path(source.root),
            "patterns": list(source.patterns),
            "metadata": source.metadata,
        }
    elif isinstance(source, LocalFileSource):
        data = {
            "kind": "LocalFileSource",
            "source_id": source.source_id,
            "path": portable_path(source.path),
            "metadata": source.metadata,
        }
    elif isinstance(source, LocalGlobSource):
        data = {
            "kind": "LocalGlobSource",
            "source_id": source.source_id,
            "pattern": portable_path(source.pattern),
            "metadata": source.metadata,
        }
    elif isinstance(source, GitHubTreeSource):
        data = {
            "kind": "GitHubTreeSource",
            "source_id": source.source_id,
            "repo_id": source.repo_id,
            "revision": source.revision,
            "pattern": source.pattern,
            "metadata": source.metadata,
        }
    elif isinstance(source, RawUrlSource):
        data = {
            "kind": "RawUrlSource",
            "source_id": source.source_id,
            "url": source.url,
            "filename": source.filename,
            "metadata": source.metadata,
        }
    elif isinstance(source, HfHubFilesSource):
        data = {
            "kind": "HfHubFilesSource",
            "source_id": source.source_id,
            "repo_id": source.repo_id,
            "pattern": source.pattern,
            "repo_type": source.repo_type,
            "revision": source.revision,
            "metadata": source.metadata,
        }
    elif isinstance(source, HfDatasetSource):
        data = {
            "kind": "HfDatasetSource",
            "source_id": source.source_id,
            "repo_id": source.repo_id,
            "split": source.split,
            "config": source.config,
            "revision": source.revision,
            "filename": source.filename,
            "metadata": source.metadata,
        }
    elif isinstance(source, CompositeSource):
        data = {
            "kind": "CompositeSource",
            "source_id": source.source_id,
            "children": [source_manifest(child) for child in source.children],
            "metadata": source.metadata,
        }
    else:
        raise TypeError(f"Unsupported raw source: {source!r}")
    return data


def resolve_raw_source(source: RawSource, *, use_cache: bool = True) -> RawSnapshot:
    if isinstance(source, LocalDirectorySource):
        return _local_directory(source)
    if isinstance(source, LocalTreeSource):
        return _local_tree(source)
    if isinstance(source, LocalFileSource):
        return _local_file(source)
    if isinstance(source, LocalGlobSource):
        return _local_glob(source)
    if isinstance(source, RawUrlSource):
        return _raw_url(source, use_cache=use_cache)
    if isinstance(source, GitHubTreeSource):
        return _github_tree(source, use_cache=use_cache)
    if isinstance(source, HfHubFilesSource):
        return _hf_hub_files(source, use_cache=use_cache)
    if isinstance(source, HfDatasetSource):
        return _hf_dataset(source, use_cache=use_cache)
    if isinstance(source, CompositeSource):
        return _composite(source, use_cache=use_cache)
    raise TypeError(f"Unsupported raw source: {source!r}")


def _local_directory(source: LocalDirectorySource) -> RawSnapshot:
    root = _resolve_path(source.root)
    if not root.is_dir():
        raise FileNotFoundError(f"Raw directory does not exist: {root}")
    return _snapshot_from_files(source.source_id, root, _files_under(root), source)


def _local_tree(source: LocalTreeSource) -> RawSnapshot:
    root = _resolve_path(source.root)
    if not root.is_dir():
        raise FileNotFoundError(f"Raw directory does not exist: {root}")
    files = {
        path
        for pattern in source.patterns
        for path in root.glob(pattern)
        if path.is_file()
    }
    if not files:
        raise FileNotFoundError(f"Raw tree source matched no files: {root}")
    return _snapshot_from_files(source.source_id, root, tuple(sorted(files)), source)


def _local_file(source: LocalFileSource) -> RawSnapshot:
    path = _resolve_path(source.path)
    if not path.is_file():
        raise FileNotFoundError(f"Raw file does not exist: {path}")
    return _snapshot_from_files(source.source_id, path.parent, (path,), source)


def _local_glob(source: LocalGlobSource) -> RawSnapshot:
    pattern = _resolve_path(source.pattern)
    files = tuple(sorted(path for path in pattern.parent.glob(pattern.name) if path.is_file()))
    if not files:
        raise FileNotFoundError(f"Raw glob matched no files: {pattern}")
    return _snapshot_from_files(source.source_id, pattern.parent, files, source)


def _raw_url(source: RawUrlSource, *, use_cache: bool) -> RawSnapshot:
    target = _download_root(source) / source.filename
    if not target.exists() or not use_cache:
        from urllib.request import urlopen

        ensure_dir(target.parent)
        with urlopen(source.url) as response, target.open("wb") as file_obj:
            shutil.copyfileobj(response, file_obj)
    return _snapshot_from_files(source.source_id, target.parent, (target,), source)


def _github_tree(source: GitHubTreeSource, *, use_cache: bool) -> RawSnapshot:
    import requests

    root = _download_root(source)
    if not use_cache or not any(root.rglob("*")):
        headers = _github_headers()
        if "Authorization" in headers:
            verify_github_token()
        api_url = f"https://api.github.com/repos/{source.repo_id}/git/trees/{source.revision}?recursive=1"
        response = requests.get(api_url, headers=headers, timeout=60)
        _raise_for_github_status(response)
        for item in response.json().get("tree", []):
            logical_path = item.get("path")
            if item.get("type") != "blob" or not logical_path or not _matches(logical_path, source.pattern):
                continue
            target = root / logical_path
            ensure_dir(target.parent)
            file_response = requests.get(item["url"], headers=headers, timeout=60)
            _raise_for_github_status(file_response)
            target.write_bytes(base64.b64decode(file_response.json()["content"]))
    return _snapshot_from_files(source.source_id, root, _files_under(root), source)


def _hf_hub_files(source: HfHubFilesSource, *, use_cache: bool) -> RawSnapshot:
    from huggingface_hub import HfApi, hf_hub_download

    root = _download_root(source)
    api = HfApi()
    for logical_path in api.list_repo_files(source.repo_id, repo_type=source.repo_type, revision=source.revision):
        if not _matches(logical_path, source.pattern):
            continue
        target = root / logical_path
        if target.exists() and use_cache:
            continue
        downloaded = hf_hub_download(
            repo_id=source.repo_id,
            repo_type=source.repo_type,
            revision=source.revision,
            filename=logical_path,
        )
        ensure_dir(target.parent)
        shutil.copy2(downloaded, target)
    return _snapshot_from_files(source.source_id, root, _files_under(root), source)


def _hf_dataset(source: HfDatasetSource, *, use_cache: bool) -> RawSnapshot:
    from datasets import load_dataset

    target = _download_root(source) / source.filename
    if not target.exists() or not use_cache:
        dataset = load_dataset(source.repo_id, source.config, split=source.split, revision=source.revision)
        ensure_dir(target.parent)
        dataset.to_parquet(str(target))
    return _snapshot_from_files(source.source_id, target.parent, (target,), source)


def _composite(source: CompositeSource, *, use_cache: bool) -> RawSnapshot:
    snapshots = tuple(resolve_raw_source(child, use_cache=use_cache) for child in source.children)
    files = tuple(file for snapshot in snapshots for file in snapshot.files)
    root = snapshots[0].root if snapshots else _download_root(source)
    return RawSnapshot(
        source_id=source.source_id,
        source_hash=stable_json_hash(
            {"source": source_manifest(source), "children": [snapshot.source_hash for snapshot in snapshots]}
        ),
        root=root,
        files=files,
    )


def _snapshot_from_files(source_id: str, root: Path, paths: tuple[Path, ...], source: RawSource) -> RawSnapshot:
    files = tuple(
        RawFile(
            path=path.resolve(),
            logical_path=path.resolve().relative_to(root.resolve()).as_posix(),
        )
        for path in sorted(paths)
        if path.is_file()
    )
    if not files:
        raise FileNotFoundError(f"Raw source {source_id!r} contains no files.")
    return RawSnapshot(
        source_id=source_id,
        source_hash=_source_hash(source, files),
        root=root.resolve(),
        files=files,
    )


def _files_under(root: Path) -> tuple[Path, ...]:
    return tuple(path for path in sorted(root.rglob("*")) if path.is_file())


def _download_root(source: RawSource) -> Path:
    return ensure_dir(RAW_ROOT / "_resolved" / source.__class__.__name__ / stable_json_hash(source_manifest(source)))


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (DATA_PACKAGE_ROOT.parent / path).resolve()


def portable_path(path: Path | str) -> str:
    path_obj = Path(path)
    absolute = path_obj if path_obj.is_absolute() else (REPO_ROOT / path_obj)
    try:
        return absolute.relative_to(DATA_PACKAGE_ROOT).as_posix()
    except ValueError:
        pass
    try:
        return absolute.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _matches(value: str, pattern: str) -> bool:
    if fnmatch.fnmatch(value, pattern):
        return True
    return re.match(pattern, value) is not None


def _github_headers() -> dict[str, str]:
    token = _github_token()
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = _github_auth_header(token)
    return headers


def _github_token() -> str | None:
    load_dotenv()
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if token is None:
        return None
    token = token.strip().strip("\"'")
    return token or None


def _github_auth_header(token: str) -> str:
    lowered = token.lower()
    if lowered.startswith("bearer ") or lowered.startswith("token "):
        return token
    return f"Bearer {token}"


@lru_cache(maxsize=1)
def verify_github_token() -> dict[str, Any]:
    token = _github_token()
    if not token:
        raise RuntimeError("GITHUB_TOKEN or GH_TOKEN is required for GitHub verification.")
    import requests

    response = requests.get("https://api.github.com/rate_limit", headers=_github_headers(), timeout=30)
    _raise_for_github_status(response)
    return response.json()


def _raise_for_github_status(response: Any) -> None:
    if response.status_code == 401:
        raise RuntimeError(
            "GitHub authentication failed. The configured GITHUB_TOKEN/GH_TOKEN is invalid, expired, "
            "or malformed. Use a raw token value like `github_pat_...` or `ghp_...`; if you include a "
            "prefix, only `Bearer ...` or `token ...` is accepted."
        ) from None
    if response.status_code == 403 and "rate limit" in response.text.lower():
        if os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN"):
            raise RuntimeError(
                "GitHub API rate limit exceeded even with an auth token. "
                "Wait for the limit to reset or use a token with a higher limit."
            ) from None
        raise RuntimeError(
            "GitHub API rate limit exceeded. Set GITHUB_TOKEN or GH_TOKEN in your environment or .env file."
        ) from None
    response.raise_for_status()


def _source_hash(source: RawSource, files: tuple[RawFile, ...]) -> str:
    metadata = getattr(source, "metadata", {})
    hash_files = bool(metadata.get("hash_files", False))
    return stable_json_hash(
        {
            "source": source_manifest(source),
            "files": [
                {
                    "logical_path": file.logical_path,
                    "size": file.path.stat().st_size,
                    "mtime_ns": file.path.stat().st_mtime_ns,
                    **({"sha256": _file_sha256(file.path)} if hash_files else {}),
                }
                for file in files
            ],
        }
    )


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
