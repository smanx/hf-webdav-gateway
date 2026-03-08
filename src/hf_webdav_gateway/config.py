from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml


VALID_REPO_TYPES = {"model", "dataset", "space"}
REPO_TYPE_ALIASES = {
    "models": "model",
    "datasets": "dataset",
    "spaces": "space",
}


@dataclass(frozen=True)
class RepoMount:
    alias: str
    repo_id: str
    repo_type: str = "model"
    revision: str = "main"
    token_env: str | None = None


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8080


@dataclass(frozen=True)
class GatewayConfig:
    server: ServerConfig
    repositories: tuple[RepoMount, ...]


def load_config(path: str | Path) -> GatewayConfig:
    raw = _load_yaml_config(path)

    server_raw = raw.get("server") or {}
    repositories_raw = raw.get("repositories") or []

    env_server = _load_server_from_env()
    env_repositories = _load_repositories_from_env()

    if env_server:
        server_raw = {**server_raw, **env_server}
    if env_repositories:
        repositories_raw = env_repositories

    repositories = _parse_repositories(repositories_raw) if repositories_raw else []
    server = _parse_server(server_raw)

    return GatewayConfig(server=server, repositories=tuple(repositories))


def _load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Top-level config must be a mapping.")
    return raw


def _load_server_from_env() -> dict[str, Any]:
    server: dict[str, Any] = {}
    host = os.getenv("HF_WEBDAV_HOST", "").strip()
    port = os.getenv("HF_WEBDAV_PORT", "").strip()
    if host:
        server["host"] = host
    if port:
        server["port"] = port
    return server


def _load_repositories_from_env() -> list[dict[str, Any]]:
    raw = os.getenv("HF_WEBDAV_REPOSITORIES", "").strip()
    if not raw:
        return []

    repositories: list[dict[str, Any]] = []
    for index, chunk in enumerate(raw.split(";"), start=1):
        item = chunk.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split("|")]
        if len(parts) < 4:
            raise ValueError(
                "Each HF_WEBDAV_REPOSITORIES item must be 'alias|repo_id|repo_type|revision' "
                "or 'alias|repo_id|repo_type|revision|token_env'."
            )

        repo: dict[str, Any] = {
            "alias": parts[0],
            "repo_id": parts[1],
            "repo_type": parts[2] or "model",
            "revision": parts[3] or "main",
        }
        if len(parts) >= 5 and parts[4]:
            repo["token_env"] = parts[4]

        repositories.append(repo)

    if not repositories:
        raise ValueError("HF_WEBDAV_REPOSITORIES is set but contains no valid repository entries.")
    return repositories


def _parse_server(server_raw: dict[str, Any]) -> ServerConfig:
    return ServerConfig(
        host=str(server_raw.get("host", "127.0.0.1")).strip() or "127.0.0.1",
        port=int(server_raw.get("port", 8080)),
    )


def _parse_repositories(repositories_raw: list[dict[str, Any]]) -> list[RepoMount]:
    
    seen_aliases: set[str] = set()
    repositories: list[RepoMount] = []

    for index, item in enumerate(repositories_raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Repository entry #{index} must be a mapping.")

        alias = str(item.get("alias", "")).strip()
        repo_id = str(item.get("repo_id", "")).strip()
        repo_type = str(item.get("repo_type", "model")).strip().lower() or "model"
        repo_type = REPO_TYPE_ALIASES.get(repo_type, repo_type)
        revision = str(item.get("revision", "main")).strip() or "main"
        token_env = item.get("token_env")
        token_env = str(token_env).strip() if token_env else None

        _validate_alias(alias, index)

        if alias in seen_aliases:
            raise ValueError(f"Duplicate alias '{alias}' in repository config.")
        if not repo_id or "/" not in repo_id:
            raise ValueError(f"Repository '{alias}' must define a valid repo_id like owner/name.")
        if repo_type not in VALID_REPO_TYPES:
            raise ValueError(
                f"Repository '{alias}' has invalid repo_type '{repo_type}'. "
                f"Expected one of: {', '.join(sorted(VALID_REPO_TYPES))}."
            )

        repositories.append(
            RepoMount(
                alias=alias,
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                token_env=token_env,
            )
        )
        seen_aliases.add(alias)
    return repositories


def _validate_alias(alias: str, index: int) -> None:
    if not alias:
        raise ValueError(f"Repository entry #{index} is missing alias.")
    if "/" in alias or "\\" in alias:
        raise ValueError(f"Repository alias '{alias}' cannot contain path separators.")
    if alias in {".", ".."}:
        raise ValueError(f"Repository alias '{alias}' is reserved.")
