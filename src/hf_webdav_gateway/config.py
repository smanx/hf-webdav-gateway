from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
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
    token: str | None = None
    discovery_source: str = "anonymous"
    discovery_status: str = "ok"
    discovery_message: str = ""


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8080


@dataclass(frozen=True)
class GatewayConfig:
    server: ServerConfig
    repositories: tuple[RepoMount, ...]
    discover_users: tuple[dict[str, str], ...] = ()


def load_config(path: str | Path) -> GatewayConfig:
    raw = _load_yaml_config(path)

    server_raw = raw.get("server") or {}

    env_server = _load_server_from_env()
    discover_users = _load_repositories_from_env()

    if env_server:
        server_raw = {**server_raw, **env_server}

    server = _parse_server(server_raw)

    return GatewayConfig(
        server=server,
        repositories=tuple(),
        discover_users=tuple(discover_users),
    )


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


def _load_repositories_from_env() -> list[dict[str, str]]:
    raw = os.getenv("HF_WEBDAV_REPOSITORIES", "").strip()
    if not raw:
        return []

    lowered = raw.lower()
    if lowered in {"0", "false", "no", "off", "disable", "disabled"}:
        return []

    users: list[dict[str, str]] = []
    seen: set[str] = set()
    for chunk in re.split(r"[;,\r\n]+", raw):
        item = chunk.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split("|")]
        if len(parts) < 1 or len(parts) > 2:
            raise ValueError(
                "HF_WEBDAV_REPOSITORIES now expects token-first entries like 'hf_xxx;hf_yyy', "
                "and if an entry is not a token it is treated as a username."
            )

        entry_value = parts[0]
        token = parts[1] if len(parts) == 2 else ""
        dedupe_key = f"{entry_value.lower()}|{token.lower()}"
        if dedupe_key in seen:
            continue
        users.append({"value": entry_value, "token": token})
        seen.add(dedupe_key)
    return users


def _parse_server(server_raw: dict[str, Any]) -> ServerConfig:
    return ServerConfig(
        host=str(server_raw.get("host", "127.0.0.1")).strip() or "127.0.0.1",
        port=int(server_raw.get("port", 8080)),
    )
