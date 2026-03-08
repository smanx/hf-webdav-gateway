---
title: HF WebDAV Gateway
emoji: "📁"
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Hugging Face WebDAV Gateway

Expose one or more Hugging Face repositories as a single read-only WebDAV server.

Each configured repository is mounted as a top-level folder:

- `/models` -> `owner/model-repo`
- `/datasets` -> `owner/dataset-repo`
- `/spaces-assets` -> `owner/space-repo`

This project is useful when you want to browse or download files from Hugging Face
with any WebDAV client, such as Windows Explorer, macOS Finder, rclone, or davfs2.

## Features

- Read-only WebDAV view over Hugging Face repositories
- Multiple repositories mounted at the same time
- Supports `model`, `dataset`, and `space` repo types
- Uses the local Hugging Face cache for downloaded files
- Simple YAML or environment-variable configuration

## Project Layout

- `config.yaml` - tracked default configuration used by Docker and Spaces
- `requirements.txt` - Python dependencies
- `run.py` - local startup entry point
- `Dockerfile` - container image for Docker and Spaces
- `docker-compose.yml` - local container deployment example
- `src/hf_webdav_gateway/config.py` - config loader and validation
- `src/hf_webdav_gateway/provider.py` - WebDAV provider backed by Hugging Face
- `src/hf_webdav_gateway/server.py` - WSGI server bootstrap

## Quick Start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Configure repositories with the tracked `config.yaml` or override them with environment variables.

Option A: edit the tracked `config.yaml`:

```bash
notepad config.yaml
```

Option B: configure everything directly with env vars:

```bash
set HF_WEBDAV_HOST=127.0.0.1
set HF_WEBDAV_PORT=8080
set HF_WEBDAV_REPOSITORIES=models|openai/whisper-large-v3|model|main;datasets|HuggingFaceFW/fineweb|dataset|main;private-models|your-org/secret-model|model|main|HF_TOKEN
```

3. If needed, export a Hugging Face token for private repositories:

```bash
set HF_TOKEN=hf_xxx
```

4. Start the server:

```bash
python run.py --config config.yaml
```

If `HF_WEBDAV_REPOSITORIES` is set, it overrides the repository list from `config.yaml`.
If neither place defines repositories, the service still starts and the homepage shows an empty mount list until you add some.

5. Connect a WebDAV client to:

```text
http://127.0.0.1:8080/dav
```

## Example Mounts

```yaml
server:
  host: 127.0.0.1
  port: 8080

repositories:
  - alias: models
    repo_id: openai/whisper-large-v3
    repo_type: model
    revision: main

  - alias: datasets
    repo_id: HuggingFaceFW/fineweb
    repo_type: dataset
    revision: main

  - alias: private-models
    repo_id: your-org/secret-model
    repo_type: model
    token_env: HF_TOKEN
```

`repositories` in `config.yaml` is optional. You can leave it out entirely and define everything with `HF_WEBDAV_REPOSITORIES` later.

## Environment Variable Format

- `HF_WEBDAV_HOST` - optional, defaults to `127.0.0.1`
- `HF_WEBDAV_PORT` - optional, defaults to `8080`
- `HF_WEBDAV_REPOSITORIES` - full repository list
- `HF_WEBDAV_USERNAME` - optional, defaults to `admin`
- `HF_WEBDAV_PASSWORD` - optional, defaults to `admin`

`HF_WEBDAV_REPOSITORIES` format:

```text
alias|repo_id|repo_type|revision[|token_env];alias2|repo_id2|repo_type2|revision2[|token_env2]
```

Example:

```bash
set HF_WEBDAV_REPOSITORIES=models|openai/whisper-large-v3|model|main;datasets|HuggingFaceFW/fineweb|dataset|main;spaces-assets|my-user/my-space|space|main;private-models|your-org/secret-model|model|main|HF_TOKEN
```

This lets you fully define multiple `alias` / `repo_id` / `repo_type` / `revision` entries from environment variables alone. Accepted `repo_type` values are `model`, `dataset`, and `space`, and the common plural forms `models`, `datasets`, and `spaces` are normalized automatically.

In Hugging Face Spaces, `HF_WEBDAV_REPOSITORIES` may be stored in either `Variables` or `Secrets`.
The app reads the single environment variable name `HF_WEBDAV_REPOSITORIES`.

## WebDAV Authentication

`/dav` is protected by HTTP Basic Auth by default.

Default credentials:

```text
admin / admin
```

Override them with:

```bash
set HF_WEBDAV_USERNAME=admin
set HF_WEBDAV_PASSWORD=change-me
```

Behavior:

- `/dav` always requires username and password
- `/` stays public so the Space homepage can still explain the mount points
- `/healthz` stays public for simple liveness checks
- WsgiDAV built-in auth is disabled; only this outer auth layer is used

WebDAV clients should connect to `/dav` and use the configured credentials.

## Docker

Build and run locally:

```bash
docker build -t hf-webdav-gateway .
docker run --rm -p 8080:7860 -e HF_WEBDAV_USERNAME=admin -e HF_WEBDAV_PASSWORD=admin -e HF_WEBDAV_REPOSITORIES="models|openai/whisper-large-v3|model|main;datasets|HuggingFaceFW/fineweb|dataset|main" -v %cd%/.hf_cache:/data/hf-home hf-webdav-gateway
```

Authenticated example:

```bash
docker run --rm -p 8080:7860 -e HF_WEBDAV_USERNAME=admin -e HF_WEBDAV_PASSWORD=change-me -e HF_WEBDAV_REPOSITORIES="models|openai/whisper-large-v3|model|main" -v %cd%/.hf_cache:/data/hf-home hf-webdav-gateway
```

Or with Compose:

```bash
docker compose up --build
```

Notes:

- The container listens on `7860` by default, which also matches Hugging Face Spaces Docker conventions.
- Hugging Face cache is stored in `/data/hf-home`; mount it as a volume to persist downloads.
- Override `HF_WEBDAV_PORT` if you need a different internal port outside Spaces.
- Open `/` for the landing page and `/dav` for the actual WebDAV endpoint.

## Hugging Face Spaces

This project also works in a Docker Space.

- Keep `Dockerfile` at the repository root
- Keep the tracked `config.yaml` at the repository root
- Add Space Variables or Secrets such as `HF_WEBDAV_REPOSITORIES` and `HF_TOKEN`
- Add `HF_WEBDAV_USERNAME` and `HF_WEBDAV_PASSWORD` if you want `/dav` protected
- If you do not set auth variables, the default credentials are still `admin` / `admin`
- The app auto-detects Space runtime hints and uses `PORT` when Hugging Face injects it
- Default bind host becomes `0.0.0.0` when `SPACE_ID` is present
- The Space homepage shows mounted repos and the WebDAV path at `/dav`

Recommended usage in Spaces:

- Put `HF_WEBDAV_REPOSITORIES` in `Secrets` when it contains private repository mappings or anything you do not want shown in the UI
- You may also store `HF_WEBDAV_REPOSITORIES` in `Variables` for public mappings
- If you define the same variable name in both places, Spaces generally injects the `Secret` value

Minimal Space configuration example:

```text
HF_WEBDAV_REPOSITORIES=models|openai/whisper-large-v3|model|main;spaces-assets|username/my-space-assets|space|main
```

## Notes

- The current implementation is read-only on purpose. It focuses on stable browsing and download behavior.
- File content is fetched through `huggingface_hub` and cached locally.
- For production exposure, put this behind a reverse proxy such as Nginx or Caddy.
- Root path `/` is a human-friendly index page; WebDAV clients should connect to `/dav`.

## Future Extensions

- Basic authentication in front of the WebDAV endpoint
- Write support using Hugging Face commit APIs
- Per-repository access control
- Optional metadata caching TTL settings
