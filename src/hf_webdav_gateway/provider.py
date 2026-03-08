from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import PurePosixPath
from typing import Iterable

from huggingface_hub import HfApi, hf_hub_download
from wsgidav.dav_provider import DAVCollection, DAVNonCollection, DAVProvider

from hf_webdav_gateway.config import RepoMount


TYPE_SEGMENTS = {
    "model": "models",
    "dataset": "datasets",
    "space": "spaces",
}


@dataclass(frozen=True)
class EntryInfo:
    name: str
    repo_path: str
    is_dir: bool
    size: int | None = None
    etag: str | None = None
    modified: float | None = None


@dataclass(frozen=True)
class MountRecord:
    path_parts: tuple[str, str, str]
    mount: RepoMount


class HfGatewayBackend:
    def __init__(self, mounts: Iterable[RepoMount]) -> None:
        self.api = HfApi()
        self.records = [_make_mount_record(mount) for mount in mounts]
        self.mounts_by_root = {record.path_parts: record.mount for record in self.records}
        self.children_index = _build_children_index(self.records)

    def get_root_children(self, prefix: tuple[str, ...]) -> list[str]:
        return self.children_index.get(prefix, [])

    def get_mount(self, path_parts: tuple[str, ...]) -> RepoMount | None:
        if len(path_parts) < 3:
            return None
        return self.mounts_by_root.get((path_parts[0], path_parts[1], path_parts[2]))

    @lru_cache(maxsize=1024)
    def list_dir(self, mount_root: tuple[str, str, str], repo_path: str) -> dict[str, EntryInfo]:
        mount = self.mounts_by_root[mount_root]
        normalized = _normalize_repo_path(repo_path)
        entries: dict[str, EntryInfo] = {}
        for item in self.api.list_repo_tree(
            repo_id=mount.repo_id,
            path_in_repo=normalized,
            repo_type=mount.repo_type,
            revision=mount.revision,
            recursive=False,
            expand=True,
            token=self._token_for_mount(mount),
        ):
            item_path = getattr(item, "path", "")
            if not item_path:
                continue
            name = PurePosixPath(item_path).name
            entries[name] = EntryInfo(
                name=name,
                repo_path=item_path,
                is_dir=_is_directory_item(item),
                size=getattr(item, "size", None),
                etag=_extract_etag(item),
                modified=_extract_modified_timestamp(item),
            )
        return entries

    def get_entry(self, mount_root: tuple[str, str, str], repo_path: str) -> EntryInfo | None:
        normalized = _normalize_repo_path(repo_path)
        if normalized == "":
            return EntryInfo(name=mount_root[2], repo_path="", is_dir=True)
        parent = str(PurePosixPath(normalized).parent)
        if parent == ".":
            parent = ""
        name = PurePosixPath(normalized).name
        return self.list_dir(mount_root, parent).get(name)

    def open_file(self, mount_root: tuple[str, str, str], repo_path: str):
        mount = self.mounts_by_root[mount_root]
        local_path = hf_hub_download(
            repo_id=mount.repo_id,
            filename=_normalize_repo_path(repo_path),
            repo_type=mount.repo_type,
            revision=mount.revision,
            token=self._token_for_mount(mount),
        )
        return open(local_path, "rb")

    def guess_content_type(self, repo_path: str) -> str:
        content_type, _ = mimetypes.guess_type(repo_path)
        return content_type or "application/octet-stream"

    def _token_for_mount(self, mount: RepoMount) -> str | None:
        if mount.token:
            return mount.token
        if not mount.token_env:
            return None
        token = os.getenv(mount.token_env, "").strip()
        return token or None


class HfWebDavProvider(DAVProvider):
    def __init__(self, backend: HfGatewayBackend) -> None:
        super().__init__()
        self.backend = backend

    def is_readonly(self) -> bool:
        return True

    def get_resource_inst(self, path: str, environ):
        parts = tuple(part for part in path.strip("/").split("/") if part)
        if not parts:
            return VirtualCollection(path or "/", environ, self, parts)

        mount = self.backend.get_mount(parts)
        if mount is None and len(parts) <= 2:
            children = self.backend.get_root_children(parts)
            if children:
                return VirtualCollection(path, environ, self, parts)
            return None

        if mount is None:
            return None

        mount_root = (parts[0], parts[1], parts[2])
        repo_path = "/".join(parts[3:])
        entry = self.backend.get_entry(mount_root, repo_path)
        if entry is None:
            return None
        if entry.is_dir:
            return RepoCollection(path, environ, self, mount_root, repo_path)
        return RepoFile(path, environ, self, mount_root, repo_path, entry)


class VirtualCollection(DAVCollection):
    def __init__(self, path: str, environ, provider: HfWebDavProvider, prefix: tuple[str, ...]) -> None:
        super().__init__(path, environ)
        self.provider = provider
        self.prefix = prefix

    def get_member_names(self):
        return self.provider.backend.get_root_children(self.prefix)

    def get_member(self, name: str):
        child_prefix = (*self.prefix, name)
        child_path = _child_webdav_path(self.path, name)
        mount = self.provider.backend.get_mount(child_prefix)
        if mount is not None:
            return RepoCollection(
                child_path,
                self.environ,
                self.provider,
                (child_prefix[0], child_prefix[1], child_prefix[2]),
                "",
            )
        children = self.provider.backend.get_root_children(child_prefix)
        if children:
            return VirtualCollection(child_path, self.environ, self.provider, child_prefix)
        return None

    def get_etag(self):
        return None


class RepoCollection(DAVCollection):
    def __init__(
        self,
        path: str,
        environ,
        provider: HfWebDavProvider,
        mount_root: tuple[str, str, str],
        repo_path: str,
    ) -> None:
        super().__init__(path, environ)
        self.provider = provider
        self.mount_root = mount_root
        self.repo_path = _normalize_repo_path(repo_path)

    def get_member_names(self):
        return list(self.provider.backend.list_dir(self.mount_root, self.repo_path).keys())

    def get_member(self, name: str):
        entry = self.provider.backend.list_dir(self.mount_root, self.repo_path).get(name)
        if entry is None:
            return None
        child_repo_path = entry.repo_path
        child_path = _child_webdav_path(self.path, name)
        if entry.is_dir:
            return RepoCollection(child_path, self.environ, self.provider, self.mount_root, child_repo_path)
        return RepoFile(child_path, self.environ, self.provider, self.mount_root, child_repo_path, entry)

    def get_etag(self):
        return None


class RepoFile(DAVNonCollection):
    def __init__(
        self,
        path: str,
        environ,
        provider: HfWebDavProvider,
        mount_root: tuple[str, str, str],
        repo_path: str,
        entry: EntryInfo,
    ) -> None:
        super().__init__(path, environ)
        self.provider = provider
        self.mount_root = mount_root
        self.repo_path = _normalize_repo_path(repo_path)
        self.entry = entry

    def get_content_length(self):
        return self.entry.size

    def get_content_type(self):
        return self.provider.backend.guess_content_type(self.repo_path)

    def get_content(self):
        return self.provider.backend.open_file(self.mount_root, self.repo_path)

    def get_etag(self):
        return self.entry.etag

    def support_etag(self):
        return self.entry.etag is not None

    def support_ranges(self):
        return False

    def get_last_modified(self):
        return self.entry.modified


def _make_mount_record(mount: RepoMount) -> MountRecord:
    username, repo_name = mount.repo_id.split("/", 1)
    return MountRecord(path_parts=(username, TYPE_SEGMENTS[mount.repo_type], repo_name), mount=mount)


def _build_children_index(records: Iterable[MountRecord]) -> dict[tuple[str, ...], list[str]]:
    index: dict[tuple[str, ...], set[str]] = {}
    for record in records:
        username, repo_type_segment, repo_name = record.path_parts
        index.setdefault(tuple(), set()).add(username)
        index.setdefault((username,), set()).add(repo_type_segment)
        index.setdefault((username, repo_type_segment), set()).add(repo_name)
    return {key: sorted(value) for key, value in index.items()}


def _normalize_repo_path(repo_path: str) -> str:
    text = repo_path.strip().strip("/")
    if not text:
        return ""
    normalized = str(PurePosixPath(text))
    if normalized == ".":
        return ""
    return normalized


def _child_webdav_path(parent: str, name: str) -> str:
    base = parent.rstrip("/")
    return f"{base}/{name}" if base else f"/{name}"


def _is_directory_item(item) -> bool:
    kind = getattr(item, "type", None)
    if isinstance(kind, str):
        lowered = kind.lower()
        if "folder" in lowered or "directory" in lowered:
            return True
        if "file" in lowered:
            return False
    return "folder" in item.__class__.__name__.lower()


def _extract_etag(item) -> str | None:
    for attr in ("blob_id", "oid", "sha"):
        value = getattr(item, attr, None)
        if value:
            return str(value)
    return None


def _extract_modified_timestamp(item) -> float | None:
    last_commit = getattr(item, "last_commit", None)
    if last_commit is None:
        return None
    date = getattr(last_commit, "date", None)
    if date is None:
        return None
    timestamp = getattr(date, "timestamp", None)
    if callable(timestamp):
        value = timestamp()
        if isinstance(value, (int, float)):
            return float(value)
    return None
