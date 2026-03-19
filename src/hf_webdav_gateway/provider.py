from __future__ import annotations

import io
import mimetypes
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import PurePosixPath
import threading
import time
from typing import Iterable
from urllib.parse import unquote

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import RemoteEntryNotFoundError
from wsgidav.dav_error import DAVError, HTTP_BAD_REQUEST, HTTP_FORBIDDEN, HTTP_INTERNAL_ERROR, HTTP_METHOD_NOT_ALLOWED
from wsgidav.dav_provider import DAVCollection, DAVNonCollection, DAVProvider

from hf_webdav_gateway.api_monitor import track_api_call

try:
    # Optional: batch commit API (preferred for large COPY/MOVE)
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete  # type: ignore
except Exception:  # pragma: no cover
    CommitOperationAdd = None  # type: ignore
    CommitOperationDelete = None  # type: ignore

from hf_webdav_gateway.config import RepoMount


TYPE_SEGMENTS = {
    "model": "models",
    "dataset": "datasets",
    "space": "spaces",
}

PLACEHOLDER_FILE = os.getenv("HF_WEBDAV_PLACEHOLDER_FILE", ".gitkeep").strip() or ".gitkeep"

# Keep newly created directories visible for a short period to avoid immediate client retries.
_OPTIMISTIC_DIR_TTL_SECONDS = int(os.getenv("HF_WEBDAV_OPTIMISTIC_DIR_TTL", "600") or "600")


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
        # Optimistic directory overlay used to make just-created folders visible
        # immediately (Windows Explorer issues PROPFIND right after MKCOL).
        # Key: (mount_root, parent_repo_path) -> {child_dir_name: last_seen_ts}
        self._optimistic_dirs: dict[tuple[tuple[str, str, str], str], dict[str, float]] = {}
        self._optimistic_dirs_lock = threading.Lock()
        self._optimistic_last_gc = 0.0
        self._last_cache_scan = 0.0
        self._empty_file_path: str | None = None

    def _get_empty_file_path(self) -> str:
        """Return a stable local path to an empty file.

        Some huggingface_hub versions are more reliable with file paths than
        file-like objects for commit operations.
        """
        if self._empty_file_path and os.path.exists(self._empty_file_path):
            return self._empty_file_path
        # Create once and keep it for the process lifetime.
        fd, path = tempfile.mkstemp(prefix="hf-webdav-empty-", suffix=".bin")
        os.close(fd)
        # Ensure empty
        try:
            with open(path, "wb"):
                pass
        except Exception:
            pass
        self._empty_file_path = path
        return path

    def invalidate_mount_cache(self, mount_root: tuple[str, str, str]) -> None:
        self.list_dir.cache_clear()

        # Best-effort cleanup of optimistic entries for this mount.
        now = time.time()
        if now - self._optimistic_last_gc > 60:
            with self._optimistic_dirs_lock:
                expired_keys = []
                for key, entries in self._optimistic_dirs.items():
                    if key[0] != mount_root:
                        continue
                    expired_names = [name for name, ts in entries.items() if now - ts > _OPTIMISTIC_DIR_TTL_SECONDS]
                    for name in expired_names:
                        entries.pop(name, None)
                    if not entries:
                        expired_keys.append(key)
                for key in expired_keys:
                    self._optimistic_dirs.pop(key, None)
                self._optimistic_last_gc = now

        mount = self.mounts_by_root.get(mount_root)
        if mount is None:
            return
        token = self._token_for_mount(mount)
        if not token:
            return
        scan_cache_dir = os.getenv("HF_HOME", "/data/hf-home")
        # scan_cache_dir can be expensive on large caches; throttle it.
        if time.time() - float(self._last_cache_scan) > 60:
            try:
                self.api.scan_cache_dir(cache_dir=scan_cache_dir).delete_revisions()
            except Exception:
                pass
            self._last_cache_scan = time.time()

    def ensure_placeholder(self, mount_root: tuple[str, str, str], placeholder_path: str, *, commit_message: str) -> None:
        """Ensure a placeholder file exists.

        HF Hub is git-backed and does not support empty folders. We materialize
        folders by writing a placeholder file (default: .gitkeep).
        """
        mount = self.mounts_by_root[mount_root]
        token = self._token_for_mount(mount)
        if not token:
            raise PermissionError("Operation requires a token-backed repository entry.")

        path_in_repo = _normalize_repo_path(placeholder_path)
        if not path_in_repo:
            raise ValueError("Invalid placeholder path")

        if CommitOperationAdd is not None and hasattr(self.api, "create_commit"):
            try:
                with track_api_call("create_commit", "mkcol", repo_id=mount.repo_id):
                    self.api.create_commit(
                        repo_id=mount.repo_id,
                        repo_type=mount.repo_type,
                        revision=mount.revision,
                        token=token,
                        commit_message=commit_message,
                        operations=[
                            CommitOperationAdd(
                                path_in_repo=path_in_repo,
                                path_or_fileobj=self._get_empty_file_path(),
                            )
                        ],
                    )
                return
            except Exception as exc:
                if _is_conflict_error(exc):
                    return
                raise

        # Fallback: upload_file
        try:
            with track_api_call("upload_file", "mkcol", repo_id=mount.repo_id):
                self.api.upload_file(
                    # Use a stable local empty file path for best compatibility.
                    path_or_fileobj=self._get_empty_file_path(),
                    path_in_repo=path_in_repo,
                    repo_id=mount.repo_id,
                    repo_type=mount.repo_type,
                    revision=mount.revision,
                    token=token,
                    commit_message=commit_message,
                )
        except Exception as exc:
            if _is_conflict_error(exc):
                return
            raise

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
        try:
            with track_api_call("list_repo_tree", "propfind", repo_id=mount.repo_id):
                iterator = self.api.list_repo_tree(
                    repo_id=mount.repo_id,
                    path_in_repo=normalized,
                    repo_type=mount.repo_type,
                    revision=mount.revision,
                    recursive=False,
                    expand=True,
                    token=self._token_for_mount(mount),
                )
                for item in iterator:
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
        except RemoteEntryNotFoundError:
            # Missing directory on HF Hub should behave like an empty listing.
            entries = {}

        # Merge optimistic folders (best-effort).
        key = (mount_root, normalized)
        optimistic = self._optimistic_dirs.get(key)
        if optimistic:
            now = time.time()
            expired = [name for name, ts in optimistic.items() if now - ts > _OPTIMISTIC_DIR_TTL_SECONDS]
            if expired:
                with self._optimistic_dirs_lock:
                    current = self._optimistic_dirs.get(key)
                    if current is not None:
                        for name in expired:
                            current.pop(name, None)
                        if not current:
                            self._optimistic_dirs.pop(key, None)
                        optimistic = self._optimistic_dirs.get(key)
            if optimistic:
                for child in sorted(optimistic.keys()):
                    if child in entries:
                        continue
                    child_path = "/".join(part for part in (normalized, child) if part)
                    entries[child] = EntryInfo(name=child, repo_path=child_path, is_dir=True)
        return entries

    def mark_dir_created(self, mount_root: tuple[str, str, str], repo_path: str) -> None:
        """Record a directory as existing, to avoid immediate client retries."""
        normalized = _normalize_repo_path(repo_path)
        if not normalized:
            return
        parent = str(PurePosixPath(normalized).parent)
        if parent == ".":
            parent = ""
        name = PurePosixPath(normalized).name
        key = (mount_root, parent)
        with self._optimistic_dirs_lock:
            self._optimistic_dirs.setdefault(key, {})[name] = time.time()
        # Ensure follow-up list_dir is not served from an old cache entry.
        self.list_dir.cache_clear()

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
        normalized = _normalize_repo_path(repo_path)
        print(
            f"[webdav-get] repo_id={mount.repo_id} path={normalized} status=begin",
            flush=True,
        )
        with track_api_call("hf_hub_download", "file_download", repo_id=mount.repo_id):
            local_path = hf_hub_download(
                repo_id=mount.repo_id,
                filename=normalized,
                repo_type=mount.repo_type,
                revision=mount.revision,
                token=self._token_for_mount(mount),
            )
        print(
            f"[webdav-get] repo_id={mount.repo_id} path={normalized} status=ok local_path={local_path}",
            flush=True,
        )
        return open(local_path, "rb")

    def guess_content_type(self, repo_path: str) -> str:
        content_type, _ = mimetypes.guess_type(repo_path)
        return content_type or "application/octet-stream"

    def write_file(self, mount_root: tuple[str, str, str], repo_path: str, data: bytes) -> None:
        """Write file content from bytes (for small files)."""
        mount = self.mounts_by_root[mount_root]
        normalized = _normalize_repo_path(repo_path)
        token = self._token_for_mount(mount)
        if not token:
            print(
                f"[webdav-put-error] repo_id={mount.repo_id} path={normalized} reason=missing_token",
                flush=True,
            )
            raise PermissionError("Writing requires a token-backed repository entry.")
        print(
            f"[webdav-put] repo_id={mount.repo_id} path={normalized} bytes={len(data)} status=uploading",
            flush=True,
        )
        with track_api_call("upload_file", "file_upload", repo_id=mount.repo_id):
            self.api.upload_file(
                path_or_fileobj=data,
                path_in_repo=normalized,
                repo_id=mount.repo_id,
                repo_type=mount.repo_type,
                revision=mount.revision,
                token=token,
                commit_message=f"Update {normalized} via WebDAV gateway",
            )
        self.invalidate_mount_cache(mount_root)
        print(
            f"[webdav-put] repo_id={mount.repo_id} path={normalized} bytes={len(data)} status=ok",
            flush=True,
        )

    def write_file_from_path(self, mount_root: tuple[str, str, str], repo_path: str, local_path: str) -> None:
        """Write file content from a local file path (streaming, memory-efficient for large files).

        This method is used for large file uploads to avoid loading the entire file into memory.
        """
        mount = self.mounts_by_root[mount_root]
        normalized = _normalize_repo_path(repo_path)
        token = self._token_for_mount(mount)
        if not token:
            print(
                f"[webdav-put-error] repo_id={mount.repo_id} path={normalized} reason=missing_token",
                flush=True,
            )
            raise PermissionError("Writing requires a token-backed repository entry.")

        # Get file size for logging
        file_size = 0
        try:
            file_size = os.path.getsize(local_path)
        except OSError:
            pass

        print(
            f"[webdav-put] repo_id={mount.repo_id} path={normalized} bytes={file_size} status=uploading",
            flush=True,
        )
        try:
            with track_api_call("upload_file", "file_upload", repo_id=mount.repo_id):
                self.api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=normalized,
                    repo_id=mount.repo_id,
                    repo_type=mount.repo_type,
                    revision=mount.revision,
                    token=token,
                    commit_message=f"Update {normalized} via WebDAV gateway",
                )
            self.invalidate_mount_cache(mount_root)
            print(
                f"[webdav-put] repo_id={mount.repo_id} path={normalized} bytes={file_size} status=ok",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[webdav-put-error] repo_id={mount.repo_id} path={normalized} error={exc}",
                flush=True,
            )
            raise

    def copy_folder(self, mount_root: tuple[str, str, str], src_repo_path: str, dest_repo_path: str) -> None:
        """Recursively copy a folder inside the same repository.

        Note: HF Hub does not provide a server-side folder copy API.
        We implement it as download + upload.
        """
        mount = self.mounts_by_root[mount_root]
        token = self._token_for_mount(mount)
        if not token:
            raise PermissionError("Copying requires a token-backed repository entry.")

        src_norm = _normalize_repo_path(src_repo_path)
        dest_norm = _normalize_repo_path(dest_repo_path)
        print(
            f"[webdav-copy-folder] repo_id={mount.repo_id} src={src_norm or '/'} dest={dest_norm or '/'} status=begin",
            flush=True,
        )

        try:
            with track_api_call("list_repo_tree", "folder_copy", repo_id=mount.repo_id):
                iterator = self.api.list_repo_tree(
                    repo_id=mount.repo_id,
                    path_in_repo=src_norm,
                    repo_type=mount.repo_type,
                    revision=mount.revision,
                    recursive=True,
                    token=token,
                )
        except RemoteEntryNotFoundError:
            print(
                f"[webdav-copy-folder] repo_id={mount.repo_id} src={src_norm or '/'} dest={dest_norm or '/'} status=missing-src",
                flush=True,
            )
            return

        max_ops = int(os.getenv("HF_WEBDAV_MAX_COMMIT_OPS", "200") or "200")

        # Prefer batch create_commit to avoid N commits for N files.
        if CommitOperationAdd is not None and hasattr(self.api, "create_commit"):
            ops = []
            committed = 0
            commits = 0
            files = 0
            try:
                for item in iterator:
                    src_file_path = getattr(item, "path", "")
                    if not src_file_path:
                        continue
                    if _is_directory_item(item):
                        continue

                    if src_norm:
                        relative_path = src_file_path[len(src_norm) :].lstrip("/")
                    else:
                        relative_path = src_file_path
                    dest_file_path = "/".join(part for part in (dest_norm, relative_path) if part)

                    local_path = hf_hub_download(
                        repo_id=mount.repo_id,
                        filename=src_file_path,
                        repo_type=mount.repo_type,
                        revision=mount.revision,
                        token=token,
                    )
                    ops.append(CommitOperationAdd(path_in_repo=dest_file_path, path_or_fileobj=local_path))
                    files += 1

                    if len(ops) >= max_ops:
                        self.api.create_commit(
                            repo_id=mount.repo_id,
                            repo_type=mount.repo_type,
                            revision=mount.revision,
                            token=token,
                            commit_message=f"Copy folder {src_norm or '/'} to {dest_norm or '/'} via WebDAV gateway",
                            operations=ops,
                        )
                        committed += len(ops)
                        commits += 1
                        ops = []
                        print(
                            f"[webdav-copy-folder] repo_id={mount.repo_id} src={src_norm or '/'} dest={dest_norm or '/'} committed={committed}",
                            flush=True,
                        )

                if ops:
                    self.api.create_commit(
                        repo_id=mount.repo_id,
                        repo_type=mount.repo_type,
                        revision=mount.revision,
                        token=token,
                        commit_message=f"Copy folder {src_norm or '/'} to {dest_norm or '/'} via WebDAV gateway",
                        operations=ops,
                    )
                    committed += len(ops)
                    commits += 1

            except RemoteEntryNotFoundError:
                print(
                    f"[webdav-copy-folder] repo_id={mount.repo_id} src={src_norm or '/'} dest={dest_norm or '/'} status=missing-src",
                    flush=True,
                )
                return

            self.invalidate_mount_cache(mount_root)
            print(
                f"[webdav-copy-folder] repo_id={mount.repo_id} src={src_norm or '/'} dest={dest_norm or '/'} files={files} commits={commits} status=ok",
                flush=True,
            )
            return

        # Fallback: per-file upload (slower, more commits)
        copied = 0
        try:
            with track_api_call("list_repo_tree", "folder_copy", repo_id=mount.repo_id):
                iterator = self.api.list_repo_tree(
                    repo_id=mount.repo_id,
                    path_in_repo=src_norm,
                    repo_type=mount.repo_type,
                    revision=mount.revision,
                    recursive=True,
                    token=token,
                )
        except RemoteEntryNotFoundError:
            print(
                f"[webdav-copy-folder] repo_id={mount.repo_id} src={src_norm or '/'} dest={dest_norm or '/'} status=missing-src",
                flush=True,
            )
            return

        for item in iterator:
            src_file_path = getattr(item, "path", "")
            if not src_file_path:
                continue
            if _is_directory_item(item):
                continue
            if src_norm:
                relative_path = src_file_path[len(src_norm) :].lstrip("/")
            else:
                relative_path = src_file_path
            dest_file_path = "/".join(part for part in (dest_norm, relative_path) if part)
            local_path = hf_hub_download(
                repo_id=mount.repo_id,
                filename=src_file_path,
                repo_type=mount.repo_type,
                revision=mount.revision,
                token=token,
            )
            self.api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=dest_file_path,
                repo_id=mount.repo_id,
                repo_type=mount.repo_type,
                revision=mount.revision,
                token=token,
                commit_message=f"Copy {src_file_path} to {dest_file_path} via WebDAV gateway",
            )
            copied += 1
            if copied % 50 == 0:
                print(
                    f"[webdav-copy-folder] repo_id={mount.repo_id} src={src_norm or '/'} dest={dest_norm or '/'} copied={copied}",
                    flush=True,
                )

        self.invalidate_mount_cache(mount_root)
        print(
            f"[webdav-copy-folder] repo_id={mount.repo_id} src={src_norm or '/'} dest={dest_norm or '/'} files={copied} commits={copied} status=ok",
            flush=True,
        )

    def copy_file(self, mount_root: tuple[str, str, str], src_repo_path: str, dest_repo_path: str) -> None:
        """Copy a single file within the same repository."""
        mount = self.mounts_by_root[mount_root]
        token = self._token_for_mount(mount)
        if not token:
            raise PermissionError("Copying requires a token-backed repository entry.")

        src_norm = _normalize_repo_path(src_repo_path)
        dest_norm = _normalize_repo_path(dest_repo_path)
        if not src_norm or not dest_norm:
            raise ValueError("Invalid source or destination path")
        if src_norm == dest_norm:
            return

        with track_api_call("hf_hub_download", "file_copy", repo_id=mount.repo_id):
            local_path = hf_hub_download(
                repo_id=mount.repo_id,
                filename=src_norm,
                repo_type=mount.repo_type,
                revision=mount.revision,
                token=token,
            )

        if CommitOperationAdd is not None and hasattr(self.api, "create_commit"):
            with track_api_call("create_commit", "file_copy", repo_id=mount.repo_id):
                self.api.create_commit(
                    repo_id=mount.repo_id,
                    repo_type=mount.repo_type,
                    revision=mount.revision,
                    token=token,
                    commit_message=f"Copy {src_norm} to {dest_norm} via WebDAV gateway",
                    operations=[CommitOperationAdd(path_in_repo=dest_norm, path_or_fileobj=local_path)],
                )
        else:
            with track_api_call("upload_file", "file_copy", repo_id=mount.repo_id):
                self.api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=dest_norm,
                    repo_id=mount.repo_id,
                    repo_type=mount.repo_type,
                    revision=mount.revision,
                    token=token,
                    commit_message=f"Copy {src_norm} to {dest_norm} via WebDAV gateway",
                )

        self.invalidate_mount_cache(mount_root)

    def move_file(self, mount_root: tuple[str, str, str], src_repo_path: str, dest_repo_path: str) -> None:
        """Move a single file within the same repository.

        Prefer a single create_commit(Add+Delete) when available.
        """
        mount = self.mounts_by_root[mount_root]
        token = self._token_for_mount(mount)
        if not token:
            raise PermissionError("Moving requires a token-backed repository entry.")

        src_norm = _normalize_repo_path(src_repo_path)
        dest_norm = _normalize_repo_path(dest_repo_path)
        if not src_norm or not dest_norm:
            raise ValueError("Invalid source or destination path")
        if src_norm == dest_norm:
            return

        with track_api_call("hf_hub_download", "file_move", repo_id=mount.repo_id):
            local_path = hf_hub_download(
                repo_id=mount.repo_id,
                filename=src_norm,
                repo_type=mount.repo_type,
                revision=mount.revision,
                token=token,
            )

        if CommitOperationAdd is not None and CommitOperationDelete is not None and hasattr(self.api, "create_commit"):
            with track_api_call("create_commit", "file_move", repo_id=mount.repo_id):
                self.api.create_commit(
                    repo_id=mount.repo_id,
                    repo_type=mount.repo_type,
                    revision=mount.revision,
                    token=token,
                    commit_message=f"Move {src_norm} to {dest_norm} via WebDAV gateway",
                    operations=[
                        CommitOperationAdd(path_in_repo=dest_norm, path_or_fileobj=local_path),
                        CommitOperationDelete(path_in_repo=src_norm),
                    ],
                )
        else:
            with track_api_call("upload_file", "file_move", repo_id=mount.repo_id):
                self.api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=dest_norm,
                    repo_id=mount.repo_id,
                    repo_type=mount.repo_type,
                    revision=mount.revision,
                    token=token,
                    commit_message=f"Move {src_norm} to {dest_norm} via WebDAV gateway",
                )
            self.delete_path(mount_root, src_norm, is_dir=False)

        self.invalidate_mount_cache(mount_root)

    def ensure_placeholder_dir(self, mount_root: tuple[str, str, str], dir_repo_path: str) -> None:
        """Ensure a directory remains materialized by placing a placeholder inside."""
        normalized_dir = _normalize_repo_path(dir_repo_path)
        if not normalized_dir:
            return
        placeholder_path = f"{normalized_dir}/{PLACEHOLDER_FILE}"
        self.ensure_placeholder(
            mount_root,
            placeholder_path,
            commit_message=f"Keep folder {normalized_dir} via WebDAV gateway",
        )

    def move_folder(self, mount_root: tuple[str, str, str], src_repo_path: str, dest_repo_path: str) -> None:
        """Recursively move a folder inside the same repository.

        Implemented as a single (or chunked) create_commit with add + delete.
        """
        mount = self.mounts_by_root[mount_root]
        token = self._token_for_mount(mount)
        if not token:
            raise PermissionError("Moving requires a token-backed repository entry.")

        src_norm = _normalize_repo_path(src_repo_path)
        dest_norm = _normalize_repo_path(dest_repo_path)
        print(
            f"[webdav-move-folder] repo_id={mount.repo_id} src={src_norm or '/'} dest={dest_norm or '/'} status=begin",
            flush=True,
        )
        
        # First pass: collect file paths (lightweight, only strings)
        file_paths: list[str] = []
        try:
            with track_api_call("list_repo_tree", "folder_move", repo_id=mount.repo_id):
                for item in self.api.list_repo_tree(
                    repo_id=mount.repo_id,
                    path_in_repo=src_norm,
                    repo_type=mount.repo_type,
                    revision=mount.revision,
                    recursive=True,
                    token=token,
                ):
                    path = getattr(item, "path", "")
                    if not path:
                        continue
                    if _is_directory_item(item):
                        continue
                    file_paths.append(path)
        except RemoteEntryNotFoundError:
            print(
                f"[webdav-move-folder] repo_id={mount.repo_id} src={src_norm or '/'} dest={dest_norm or '/'} status=missing-src",
                flush=True,
            )
            return

        if CommitOperationAdd is None or CommitOperationDelete is None or not hasattr(self.api, "create_commit"):
            # Fallback: copy then delete folder (two commits at least)
            self.copy_folder(mount_root, src_norm, dest_norm)
            self.delete_path(mount_root, src_norm, is_dir=True)
            self.invalidate_mount_cache(mount_root)
            print(
                f"[webdav-move-folder] repo_id={mount.repo_id} src={src_norm or '/'} dest={dest_norm or '/'} files={len(file_paths)} status=ok-fallback",
                flush=True,
            )
            return

        add_ops = []
        for src_file_path in file_paths:
            if src_norm:
                relative_path = src_file_path[len(src_norm) :].lstrip("/")
            else:
                relative_path = src_file_path
            dest_file_path = "/".join(part for part in (dest_norm, relative_path) if part)
            local_path = hf_hub_download(
                repo_id=mount.repo_id,
                filename=src_file_path,
                repo_type=mount.repo_type,
                revision=mount.revision,
                token=token,
            )
            add_ops.append(CommitOperationAdd(path_in_repo=dest_file_path, path_or_fileobj=local_path))

        max_ops = int(os.getenv("HF_WEBDAV_MAX_COMMIT_OPS", "200") or "200")
        add_commits = 0
        committed = 0

        # Phase 1: add/copy all files first
        for i in range(0, len(add_ops), max_ops):
            chunk = add_ops[i : i + max_ops]
            self.api.create_commit(
                repo_id=mount.repo_id,
                repo_type=mount.repo_type,
                revision=mount.revision,
                token=token,
                commit_message=f"Move folder (copy) {src_norm or '/'} to {dest_norm or '/'} via WebDAV gateway",
                operations=chunk,
            )
            add_commits += 1
            committed += len(chunk)
            if committed < len(add_ops):
                print(
                    f"[webdav-move-folder] repo_id={mount.repo_id} src={src_norm or '/'} dest={dest_norm or '/'} copied={committed}",
                    flush=True,
                )

        # Phase 2: delete source (prefer folder delete op)
        delete_ops = []
        try:
            delete_ops.append(CommitOperationDelete(path_in_repo=src_norm, is_folder=True))  # type: ignore[call-arg]
        except TypeError:
            for src_file_path in file_paths:
                delete_ops.append(CommitOperationDelete(path_in_repo=src_file_path))

        del_commits = 0
        for i in range(0, len(delete_ops), max_ops):
            chunk = delete_ops[i : i + max_ops]
            self.api.create_commit(
                repo_id=mount.repo_id,
                repo_type=mount.repo_type,
                revision=mount.revision,
                token=token,
                commit_message=f"Move folder (delete) {src_norm or '/'} via WebDAV gateway",
                operations=chunk,
            )
            del_commits += 1

        self.invalidate_mount_cache(mount_root)
        print(
            f"[webdav-move-folder] repo_id={mount.repo_id} src={src_norm or '/'} dest={dest_norm or '/'} files={len(file_paths)} commits={add_commits + del_commits} status=ok",
            flush=True,
        )

    def delete_path(self, mount_root: tuple[str, str, str], repo_path: str, is_dir: bool) -> None:
        mount = self.mounts_by_root[mount_root]
        token = self._token_for_mount(mount)
        normalized = _normalize_repo_path(repo_path)
        if not token:
            print(
                f"[webdav-delete-error] repo_id={mount.repo_id} path={normalized} reason=missing_token",
                flush=True,
            )
            raise PermissionError("Deleting requires a token-backed repository entry.")

        try:
            if is_dir:
                delete_folder = getattr(self.api, "delete_folder", None)
                if not callable(delete_folder):
                    raise NotImplementedError("Directory deletion is not supported by this huggingface_hub version.")
                try:
                    delete_folder(
                        path_in_repo=normalized,
                        repo_id=mount.repo_id,
                        repo_type=mount.repo_type,
                        revision=mount.revision,
                        token=token,
                        commit_message=f"Delete folder {normalized} via WebDAV gateway",
                    )
                except RemoteEntryNotFoundError:
                    # Idempotent delete: if it doesn't exist remotely, treat as success.
                    pass
            else:
                delete_file = getattr(self.api, "delete_file", None)
                if not callable(delete_file):
                    raise NotImplementedError("File deletion is not supported by this huggingface_hub version.")
                try:
                    delete_file(
                        path_in_repo=normalized,
                        repo_id=mount.repo_id,
                        repo_type=mount.repo_type,
                        revision=mount.revision,
                        token=token,
                        commit_message=f"Delete {normalized} via WebDAV gateway",
                    )
                except RemoteEntryNotFoundError:
                    # Idempotent delete.
                    pass

                # If the deleted file was the last entry in its parent directory, HF Hub will
                # effectively drop that now-empty folder. Windows clients often expect the
                # folder to remain, so we create a placeholder file to keep it materialized.
                parent = str(PurePosixPath(normalized).parent)
                if parent == ".":
                    parent = ""
                deleted_name = PurePosixPath(normalized).name
                if parent and deleted_name != PLACEHOLDER_FILE:
                    try:
                        remaining = self.list_dir(mount_root, parent)
                    except Exception:
                        remaining = {}
                    if not remaining:
                        try:
                            self.ensure_placeholder_dir(mount_root, parent)
                        except Exception:
                            # Best-effort: even if this fails, the delete already succeeded.
                            pass
            self.invalidate_mount_cache(mount_root)
            print(
                f"[webdav-delete] repo_id={mount.repo_id} path={normalized} kind={'dir' if is_dir else 'file'} status=ok",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[webdav-delete-error] repo_id={mount.repo_id} path={normalized} kind={'dir' if is_dir else 'file'} error={exc}",
                flush=True,
            )
            raise

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
        return False

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

    def create_empty_resource(self, name: str):
        child_path = _child_webdav_path(self.path, name)
        child_repo_path = "/".join(part for part in (self.repo_path, name) if part)
        entry = EntryInfo(name=name, repo_path=child_repo_path, is_dir=False, size=0)
        return RepoFile(child_path, self.environ, self.provider, self.mount_root, child_repo_path, entry)

    def create_collection(self, name: str):
        child_path = _child_webdav_path(self.path, name)
        child_repo_path = "/".join(part for part in (self.repo_path, name) if part)
        mount = self.provider.backend.mounts_by_root[self.mount_root]
        
        # HuggingFace Hub doesn't support empty folders - create a placeholder file.
        token = self.provider.backend._token_for_mount(mount)
        if token:
            placeholder_path = f"{child_repo_path}/{PLACEHOLDER_FILE}" if child_repo_path else f"{name}/{PLACEHOLDER_FILE}"
            try:
                self.provider.backend.ensure_placeholder(
                    self.mount_root,
                    placeholder_path,
                    commit_message=f"Create folder {child_repo_path} via WebDAV gateway",
                )
                self.provider.backend.list_dir.cache_clear()
                print(
                    f"[webdav-mkcol] repo_id={mount.repo_id} path={child_repo_path or '/'} status=created",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[webdav-mkcol-error] repo_id={mount.repo_id} path={child_repo_path or '/'} error={exc}",
                    flush=True,
                )
                raise
        else:
            print(
                f"[webdav-mkcol-error] repo_id={mount.repo_id} path={child_repo_path or '/'} reason=missing_token",
                flush=True,
            )
            raise PermissionError("Creating folders requires a token-backed repository entry.")
        
        return RepoCollection(child_path, self.environ, self.provider, self.mount_root, child_repo_path)

    def handle_copy(self, dest_path, *, depth_infinity=False, overwrite=False):
        """Handle COPY request for a collection (folder).

        Note: GatewayApp handles COPY directly for compatibility. This is kept as a
        fallback when running without GatewayApp.
        """
        mount = self.provider.backend.mounts_by_root.get(self.mount_root)
        if mount is None:
            raise DAVError(HTTP_FORBIDDEN, "Source repository not found.")
        
        token = self.provider.backend._token_for_mount(mount)
        if not token:
            raise DAVError(HTTP_FORBIDDEN, "Copying requires a token-backed repository entry.")
        
        # Parse destination path (may be a full URL)
        dest_parts = _parse_webdav_dest_path(dest_path)
        if len(dest_parts) < 4:
            raise DAVError(HTTP_BAD_REQUEST, "Invalid destination path.")
        
        dest_mount_root = (dest_parts[0], dest_parts[1], dest_parts[2])
        dest_repo_path = "/".join(dest_parts[3:])
        
        # Check if source and destination are in the same repo
        if dest_mount_root != self.mount_root:
            raise DAVError(HTTP_FORBIDDEN, "Cross-repository copy is not supported.")
        
        try:
            self.provider.backend.copy_folder(self.mount_root, self.repo_path, dest_repo_path)
        except PermissionError as exc:
            raise DAVError(HTTP_FORBIDDEN, str(exc)) from exc
        except Exception as exc:
            print(
                f"[webdav-copy-error] repo_id={mount.repo_id} src={self.repo_path} dest={dest_repo_path} error={exc}",
                flush=True,
            )
            raise DAVError(HTTP_INTERNAL_ERROR, f"Failed to copy folder: {exc}") from exc

        self.provider.backend.list_dir.cache_clear()
        # Return True means handled, wsgidav will set response.
        return True

    def handle_move(self, dest_path):
        """Handle MOVE request for a collection (folder).

        Note: GatewayApp handles MOVE directly for compatibility. This is kept as a
        fallback when running without GatewayApp.
        """
        mount = self.provider.backend.mounts_by_root.get(self.mount_root)
        if mount is None:
            raise DAVError(HTTP_FORBIDDEN, "Source repository not found.")
        
        token = self.provider.backend._token_for_mount(mount)
        if not token:
            raise DAVError(HTTP_FORBIDDEN, "Moving requires a token-backed repository entry.")
        
        # Parse destination path (may be a full URL)
        dest_parts = _parse_webdav_dest_path(dest_path)
        if len(dest_parts) < 4:
            raise DAVError(HTTP_BAD_REQUEST, "Invalid destination path.")
        
        dest_mount_root = (dest_parts[0], dest_parts[1], dest_parts[2])
        dest_repo_path = "/".join(dest_parts[3:])
        
        # Check if source and destination are in the same repo
        if dest_mount_root != self.mount_root:
            raise DAVError(HTTP_FORBIDDEN, "Cross-repository move is not supported.")
        
        try:
            move = getattr(self.provider.backend, "move_folder", None)
            if callable(move):
                move(self.mount_root, self.repo_path, dest_repo_path)
            else:
                self.provider.backend.copy_folder(self.mount_root, self.repo_path, dest_repo_path)
                self.provider.backend.delete_path(self.mount_root, self.repo_path, is_dir=True)
        except PermissionError as exc:
            raise DAVError(HTTP_FORBIDDEN, str(exc)) from exc
        except Exception as exc:
            print(f"[webdav-move-error] src={self.repo_path} dest={dest_repo_path} error={exc}", flush=True)
            raise DAVError(HTTP_INTERNAL_ERROR, f"Failed to move: {exc}")

        self.provider.backend.list_dir.cache_clear()
        return True

    def delete(self):
        if not self.repo_path:
            raise DAVError(HTTP_METHOD_NOT_ALLOWED, "Repository roots cannot be deleted through WebDAV.")
        try:
            self.provider.backend.delete_path(self.mount_root, self.repo_path, is_dir=True)
        except PermissionError as exc:
            raise DAVError(HTTP_FORBIDDEN, str(exc)) from exc
        except NotImplementedError as exc:
            raise DAVError(HTTP_METHOD_NOT_ALLOWED, str(exc)) from exc


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

    def begin_write(self, *, content_type=None):
        return _UploadBuffer(self.provider.backend, self.mount_root, self.repo_path)

    def end_write(self, with_errors: bool) -> None:
        pass

    def delete(self):
        try:
            self.provider.backend.delete_path(self.mount_root, self.repo_path, is_dir=False)
        except PermissionError as exc:
            raise DAVError(HTTP_FORBIDDEN, str(exc)) from exc
        except NotImplementedError as exc:
            raise DAVError(HTTP_METHOD_NOT_ALLOWED, str(exc)) from exc


class _UploadBuffer(io.BufferedWriter):
    """A file-like buffer that streams uploads to a temp file, avoiding memory blowup.

    Instead of buffering the entire upload in memory (which causes OOM for large files),
    this class writes directly to a temporary file and streams it to HuggingFace Hub.
    """

    def __init__(self, backend: HfGatewayBackend, mount_root: tuple[str, str, str], repo_path: str) -> None:
        self._backend = backend
        self._mount_root = mount_root
        self._repo_path = repo_path
        self._closed = False
        self._temp_file = tempfile.NamedTemporaryFile(
            mode='wb',
            delete=False,
            prefix='hf-webdav-upload-',
            suffix='.tmp'
        )
        self._temp_path = self._temp_file.name
        super().__init__(self._temp_file)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        # Flush and close the temp file first
        try:
            self.flush()
        except Exception:
            pass
        try:
            self._temp_file.close()
        except Exception:
            pass

        # Upload from the temp file path (avoids reading entire file into memory)
        try:
            self._backend.write_file_from_path(self._mount_root, self._repo_path, self._temp_path)
        except Exception:
            raise
        finally:
            # Clean up the temp file
            try:
                os.remove(self._temp_path)
            except OSError:
                pass


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


def _parse_webdav_dest_path(dest_path: str) -> tuple[str, ...]:
    """Parse Destination header value into path segments.

    wsgidav may pass an absolute URL (e.g. http://host/dav/u/models/r/file)
    or a raw path. We only need the path portion.
    """
    text = (dest_path or "").strip()
    if "://" in text:
        # crude but sufficient: split at first '/' after scheme+host
        try:
            after = text.split("://", 1)[1]
            idx = after.find("/")
            text = after[idx:] if idx >= 0 else "/"
        except Exception:
            text = "/"
    text = unquote(text)
    parts = tuple(part for part in text.strip("/").split("/") if part)
    if parts and parts[0] == "dav":
        parts = parts[1:]
    return parts


def _is_conflict_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("409" in msg) or ("conflict" in msg) or ("already exists" in msg)


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
