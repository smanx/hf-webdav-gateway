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

[中文说明](README.zh-CN.md)

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

2. Configure the server, then let the app discover repositories for one or more Hugging Face users through `HF_WEBDAV_REPOSITORIES`.

Option A: edit the tracked `config.yaml` for host and port only:

```bash
notepad config.yaml
```

Then set one or more user entries:

```bash
set HF_WEBDAV_REPOSITORIES=smanx|hf_xxx;other-user|hf_yyy
```

3. If needed, export a Hugging Face token for private repositories:

```bash
set HF_TOKEN=hf_xxx
```

4. Start the server:

```bash
python run.py --config config.yaml
```

Repository discovery is controlled only by `HF_WEBDAV_REPOSITORIES`.
`config.yaml` no longer defines repository mappings.

5. Connect a WebDAV client to:

```text
http://127.0.0.1:8080/dav
```

## Path Layout

Discovered repositories are exposed under this fixed structure:

```text
/dav/<username>/models/<repo-name>
/dav/<username>/datasets/<repo-name>
/dav/<username>/spaces/<repo-name>
```

Example:

```text
/dav/smanx/models/my-model
/dav/smanx/datasets/phoenix-data
/dav/smanx/spaces/demo-space
```

## Environment Variable Format

- `HF_WEBDAV_HOST` - optional, defaults to `127.0.0.1`
- `HF_WEBDAV_PORT` - optional, defaults to `8080`
- `HF_WEBDAV_REPOSITORIES` - one or more token-first entries separated by `;`, `,`, or new lines
- `HF_WEBDAV_USERNAME` - optional, defaults to `admin`
- `HF_WEBDAV_PASSWORD` - optional, defaults to `admin`

Format:

```text
HF_WEBDAV_REPOSITORIES=hf_xxx;hf_yyy
```

These separators are all supported between entries:

- `;`
- `,`
- new lines

Example with commas:

```text
HF_WEBDAV_REPOSITORIES=hf_xxx,hf_yyy
```

Example with new lines:

```text
HF_WEBDAV_REPOSITORIES=hf_xxx
hf_yyy
```

When set, the app first treats each entry as a Hugging Face token. If the token resolves successfully, the username is inferred automatically. If the token does not resolve, the same value is treated as a username and only public repositories are queried.

Behavior:

- You do not specify repository names manually
- You do not specify `model` / `dataset` / `space` manually
- You do not specify mount paths manually
- Paths are always exposed as `/username/models/repo-name`, `/username/datasets/repo-name`, and `/username/spaces/repo-name`
- Multiple users are supported in one variable, separated by `;`, `,`, or new lines
- The preferred format is just `token`
- If an entry is not a valid token, it is treated as a username
- A compatibility form `username|token` is also accepted

Disable discovery by leaving `HF_WEBDAV_REPOSITORIES` unset, or by setting one of these values:

```text
0
false
no
off
disable
disabled
```

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
docker run --rm -p 8080:7860 -e HF_WEBDAV_USERNAME=admin -e HF_WEBDAV_PASSWORD=admin -e HF_WEBDAV_REPOSITORIES="hf_xxx;hf_yyy" -v %cd%/.hf_cache:/data/hf-home hf-webdav-gateway
```

Authenticated example:

```bash
docker run --rm -p 8080:7860 -e HF_WEBDAV_USERNAME=admin -e HF_WEBDAV_PASSWORD=change-me -e HF_WEBDAV_REPOSITORIES="hf_xxx;public-user" -v %cd%/.hf_cache:/data/hf-home hf-webdav-gateway
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
- Add Space Variables or Secrets such as `HF_WEBDAV_REPOSITORIES`
- Add `HF_WEBDAV_USERNAME` and `HF_WEBDAV_PASSWORD` if you want `/dav` protected
- If you do not set auth variables, the default credentials are still `admin` / `admin`
- The app auto-detects Space runtime hints and uses `PORT` when Hugging Face injects it
- Default bind host becomes `0.0.0.0` when `SPACE_ID` is present
- The Space homepage shows mounted repos and the WebDAV path at `/dav`

Recommended usage in Spaces:

- Put `HF_WEBDAV_REPOSITORIES` in `Secrets` when you enable discovery for a private account
- You may also store `HF_WEBDAV_REPOSITORIES` in `Variables` for public mappings
- If you define the same variable name in both places, Spaces generally injects the `Secret` value

Minimal Space configuration example:

```text
HF_WEBDAV_REPOSITORIES=hf_xxx;hf_yyy
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
