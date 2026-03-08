from __future__ import annotations

import argparse
import base64
import html
import os
from urllib.parse import parse_qs
from pathlib import Path
from typing import cast
from wsgiref.simple_server import make_server
from wsgiref.util import request_uri

from huggingface_hub import HfApi
from wsgidav.wsgidav_app import WsgiDAVApp

from hf_webdav_gateway.config import GatewayConfig, RepoMount, load_config
from hf_webdav_gateway.provider import HfGatewayBackend, HfWebDavProvider


DISCOVERY_EVENTS: list[dict[str, object]] = []
REPO_METADATA: dict[str, dict[str, object]] = {}


def build_app(config_path: str | Path):
    DISCOVERY_EVENTS.clear()
    REPO_METADATA.clear()
    config = load_config(config_path)
    mounts = _resolve_mounts(config)
    config = GatewayConfig(server=config.server, repositories=tuple(mounts), discover_users=tuple())
    backend = HfGatewayBackend(config.repositories)
    provider = HfWebDavProvider(backend)

    dav_app = WsgiDAVApp(
        {
            "provider_mapping": {"/": provider},
            "simple_dc": {"user_mapping": {"*": True}},
            "http_authenticator": {
                "accept_basic": False,
                "accept_digest": False,
                "trusted_auth_header": "REMOTE_USER",
            },
            "verbose": 1,
        }
    )
    app = GatewayApp(config, dav_app)
    return app, config.server.host, config.server.port


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Expose Hugging Face repositories over WebDAV.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML config file. Optional if env vars are used.",
    )
    args = parser.parse_args(argv)

    if not os.getenv("HF_WEBDAV_CONFIG", "").strip() and args.config == "config.yaml":
        args.config = os.getenv("HF_WEBDAV_CONFIG", "config.yaml")

    space_port = os.getenv("PORT", "").strip()
    if space_port and not os.getenv("HF_WEBDAV_PORT", "").strip():
        os.environ["HF_WEBDAV_PORT"] = space_port
    if os.getenv("SPACE_ID", "").strip() and not os.getenv("HF_WEBDAV_HOST", "").strip():
        os.environ["HF_WEBDAV_HOST"] = "0.0.0.0"

    app, host, port = build_app(args.config)

    with make_server(host, port, app) as httpd:
        print(f"Hugging Face WebDAV gateway listening on http://{host}:{port}/")
        httpd.serve_forever()

    return 0


class GatewayApp:
    def __init__(self, config: GatewayConfig, dav_app: WsgiDAVApp) -> None:
        self.config = config
        self.dav_app = dav_app
        self.username = os.getenv("HF_WEBDAV_USERNAME", "admin").strip() or "admin"
        self.password = os.getenv("HF_WEBDAV_PASSWORD", "admin")

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "") or "/"
        if path == "/space-action":
            return self._handle_space_action(environ, start_response)
        if path == "/":
            return self._serve_home(environ, start_response)
        if path == "/healthz":
            body = b"ok\n"
            start_response(
                "200 OK",
                [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))],
            )
            return [body]
        if path.startswith("/dav"):
            if self._auth_enabled() and not self._is_authorized(environ):
                return self._unauthorized(start_response)
            forwarded = dict(environ)
            forwarded["REMOTE_USER"] = self.username
            forwarded["SCRIPT_NAME"] = f"{environ.get('SCRIPT_NAME', '')}/dav".rstrip("/")
            forwarded_path = path[len("/dav") :] or "/"
            forwarded["PATH_INFO"] = forwarded_path
            return self.dav_app(forwarded, start_response)

        body = b"Not Found\n"
        start_response(
            "404 Not Found",
            [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))],
        )
        return [body]

    def _serve_home(self, environ, start_response):
        language = _detect_language(environ)
        text = _get_home_text(language)
        title = cast(str, text["title"])
        eyebrow = cast(str, text["eyebrow"])
        headline = cast(str, text["headline"])
        intro = cast(str, text["intro"])
        webdav_root = cast(str, text["webdav_root"])
        health_check = cast(str, text["health_check"])
        webdav_auth = cast(str, text["webdav_auth"])
        mounted_repositories = cast(str, text["mounted_repositories"])
        alias_label = cast(str, text["alias"])
        repo_id_label = cast(str, text["repo_id"])
        type_label = cast(str, text["type"])
        revision_label = cast(str, text["revision"])
        webdav_path_label = cast(str, text["webdav_path"])
        hint = cast(str, text["hint"])
        lang_note = cast(str, text["lang_note"])
        summary_users = cast(str, text["summary_users"])
        summary_repos = cast(str, text["summary_repos"])
        auth_enabled = cast(str, text["auth_enabled"])
        auth_disabled = cast(str, text["auth_disabled"])
        discovery_status = cast(str, text["discovery_status"])
        discovery_source = cast(str, text["discovery_source"])
        discovery_message = cast(str, text["discovery_message"])
        hf_link_label = cast(str, text["hf_link_label"])
        space_actions = cast(str, text["space_actions"])
        runtime_status_label = cast(str, text["runtime_status_label"])
        repo_size_label = cast(str, text["repo_size_label"])
        start_label = cast(str, text["start_label"])
        resume_label = cast(str, text["resume_label"])
        pause_label = cast(str, text["pause_label"])
        restart_label = cast(str, text["restart_label"])
        issue_title = cast(str, text["issue_title"])
        issue_empty = cast(str, text["issue_empty"])
        base_url = request_uri(environ, include_query=False).rstrip("/")
        dav_url = f"{base_url}/dav"
        auth_status = auth_enabled if self._auth_enabled() else auth_disabled
        issues = [event for event in DISCOVERY_EVENTS if event.get("severity") == "error"]
        rows = "\n".join(
            _render_repo_row(mount, text, DISCOVERY_EVENTS, start_label, pause_label, resume_label, restart_label)
            for mount in self.config.repositories
        )
        body = f"""<!doctype html>
<html lang=\"{html.escape(language)}\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --panel: rgba(255, 252, 247, 0.88);
      --ink: #1f2937;
      --muted: #5f6b7a;
      --line: rgba(31, 41, 55, 0.12);
      --accent: #0f766e;
      --accent-2: #c2410c;
      --shadow: 0 20px 60px rgba(41, 37, 36, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.14), transparent 32%),
        radial-gradient(circle at bottom right, rgba(194,65,12,0.16), transparent 26%),
        linear-gradient(135deg, #f8f2e8, #ece7df 52%, #f3ede4);
      min-height: 100vh;
    }}
    .shell {{
      max-width: 1040px;
      margin: 0 auto;
      padding: 48px 20px 64px;
    }}
    .hero, .panel {{
      background: var(--panel);
      backdrop-filter: blur(8px);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
    }}
    .hero {{ padding: 32px; }}
    .hero-top {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 18px; }}
    .eyebrow {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(15,118,110,0.1);
      color: var(--accent);
      font: 600 12px/1.2 "Trebuchet MS", sans-serif;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    h1 {{ margin: 18px 0 12px; font-size: clamp(2.2rem, 4vw, 4.1rem); line-height: 0.96; }}
    p {{ color: var(--muted); font-size: 1.05rem; line-height: 1.7; }}
    .grid {{ display: grid; gap: 20px; margin-top: 20px; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }}
    .panel {{ padding: 24px; }}
    .k {{ margin: 0 0 8px; font: 600 12px/1.2 "Trebuchet MS", sans-serif; letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent-2); }}
    .langbar {{ display: flex; justify-content: flex-end; gap: 10px; flex-wrap: wrap; }}
    .langlink {{
      text-decoration: none; padding: 8px 12px; border-radius: 999px; border: 1px solid var(--line);
      background: rgba(255,255,255,0.62); color: var(--ink); font: 600 12px/1.2 "Trebuchet MS", sans-serif;
      letter-spacing: 0.06em; text-transform: uppercase;
    }}
    .langlink.active {{ background: rgba(15,118,110,0.12); color: var(--accent); border-color: rgba(15,118,110,0.28); }}
    .langnote {{ margin-top: 8px; color: var(--muted); font-size: 0.92rem; }}
    .url {{
      display: block; overflow-wrap: anywhere; text-decoration: none; color: var(--ink);
      padding: 14px 16px; border-radius: 16px; background: rgba(255,255,255,0.7); border: 1px solid var(--line);
      font-family: "Courier New", monospace;
    }}
    .summary {{ margin-top: 18px; display: flex; flex-wrap: wrap; gap: 12px; }}
    .chip {{
      display: inline-flex; align-items: center; gap: 8px; padding: 10px 12px; border-radius: 999px;
      background: rgba(255,255,255,0.68); border: 1px solid var(--line); color: var(--ink);
      font: 600 13px/1.2 "Trebuchet MS", sans-serif;
    }}
    .issues {{ margin-top: 20px; display: grid; gap: 12px; }}
    .issue {{
      padding: 16px 18px; border-radius: 18px; border: 1px solid rgba(185, 28, 28, 0.16);
      background: rgba(254, 242, 242, 0.92); color: #7f1d1d;
    }}
    .issue strong {{ display: block; margin-bottom: 6px; }}
    .dot {{ width: 8px; height: 8px; border-radius: 999px; background: var(--accent); }}
    .repo-list {{ display: grid; gap: 14px; margin-top: 12px; }}
    .repo-card {{ padding: 18px; border-radius: 20px; border: 1px solid var(--line); background: rgba(255,255,255,0.72); }}
    .repo-head {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 14px; flex-wrap: wrap; }}
    .repo-title {{ margin: 0; font-size: 1.08rem; line-height: 1.35; }}
    .repo-sub {{ margin-top: 6px; color: var(--muted); font-size: 0.95rem; overflow-wrap: anywhere; }}
    .pill {{ display: inline-flex; align-items: center; gap: 6px; padding: 7px 10px; border-radius: 999px; border: 1px solid var(--line); background: rgba(255,255,255,0.82); font: 600 12px/1.2 "Trebuchet MS", sans-serif; color: var(--ink); }}
    .repo-meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px 16px; margin-top: 14px; }}
    .meta-item {{ min-width: 0; }}
    .meta-k {{ margin-bottom: 4px; color: var(--muted); font: 600 11px/1.2 "Trebuchet MS", sans-serif; letter-spacing: 0.06em; text-transform: uppercase; }}
    .meta-v {{ overflow-wrap: anywhere; }}
    .meta-v code, .repo-sub code {{ white-space: pre-wrap; }}
    .actions {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .actions form {{ margin: 0; }}
    .actions button {{ padding: 8px 12px; border-radius: 999px; border: 1px solid var(--line); background: rgba(255,255,255,0.9); color: var(--ink); cursor: pointer; }}
    code {{ font-family: "Courier New", monospace; }}
    .hint {{ margin-top: 22px; font-size: 0.96rem; }}
    .path {{ color: var(--accent); font-weight: 700; }}
    @media (max-width: 640px) {{
      .shell {{ padding: 24px 14px 40px; }}
      .hero, .panel {{ border-radius: 18px; }}
      .repo-card {{ padding: 16px; }}
    }}
  </style>
</head>
<body>
  <main class=\"shell\">
    <section class=\"hero\">
      <div class=\"hero-top\">
        <div>
          <span class=\"eyebrow\">{html.escape(eyebrow)}</span>
          <h1>{html.escape(headline)}</h1>
        </div>
        <div>
          <div class=\"langbar\">
            <a class=\"langlink{' active' if language == 'en' else ''}\" href=\"/?lang=en\">English</a>
            <a class=\"langlink{' active' if language == 'zh-CN' else ''}\" href=\"/?lang=zh-CN\">中文</a>
          </div>
          <div class=\"langnote\">{html.escape(lang_note)}</div>
        </div>
      </div>
      <p>{intro.format(path='<code>/</code>', dav='<span class="path">/dav</span>')}</p>
      <div class=\"summary\">
        <div class=\"chip\"><span class=\"dot\"></span>{html.escape(summary_users.format(count=len({mount.repo_id.split('/', 1)[0] for mount in self.config.repositories})) )}</div>
        <div class=\"chip\"><span class=\"dot\"></span>{html.escape(summary_repos.format(count=len(self.config.repositories)))}</div>
      </div>
      <div class=\"issues\">{_render_issues(issues, issue_title, issue_empty)}</div>
      <div class=\"grid\">
        <section class=\"panel\">
          <div class=\"k\">{html.escape(webdav_root)}</div>
          <a class=\"url\" href=\"{html.escape(dav_url)}\">{html.escape(dav_url)}</a>
        </section>
        <section class=\"panel\">
          <div class=\"k\">{html.escape(health_check)}</div>
          <a class=\"url\" href=\"{html.escape(base_url + '/healthz')}\">{html.escape(base_url + '/healthz')}</a>
        </section>
        <section class=\"panel\">
          <div class=\"k\">{html.escape(webdav_auth)}</div>
          <div class=\"url\">{html.escape(auth_status)}</div>
        </section>
      </div>
    </section>
    <section class=\"panel\" style=\"margin-top: 20px;\">
      <div class=\"k\">{html.escape(mounted_repositories)}</div>
      <div class="repo-list">
        {rows or _empty_row(text)}
      </div>
      <p class=\"hint\">{hint.format(example=html.escape(dav_url) + '/username/models/repository')}</p>
    </section>
  </main>
</body>
</html>
""".encode("utf-8")
        start_response(
            "200 OK",
            [("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(body)))],
        )
        return [body]

    def _auth_enabled(self) -> bool:
        return True

    def _handle_space_action(self, environ, start_response):
        if environ.get("REQUEST_METHOD", "GET").upper() != "POST":
            body = b"Method Not Allowed\n"
            start_response("405 Method Not Allowed", [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))])
            return [body]
        if self._auth_enabled() and not self._is_authorized(environ):
            return self._unauthorized(start_response)

        params = parse_qs(environ.get("QUERY_STRING", ""), keep_blank_values=False)
        repo_id = (params.get("repo_id") or [""])[0].strip()
        action = (params.get("action") or [""])[0].strip().lower()
        mount = next((item for item in self.config.repositories if item.repo_id == repo_id and item.repo_type == "space"), None)
        if not repo_id or mount is None:
            body = b"Unknown Space repository\n"
            start_response("404 Not Found", [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))])
            return [body]
        if not mount.token:
            body = b"Space action requires a token-backed entry\n"
            start_response("403 Forbidden", [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))])
            return [body]

        api = HfApi(token=mount.token)
        try:
            if action == "start":
                api.restart_space(repo_id, token=mount.token)
            elif action == "resume":
                api.restart_space(repo_id, token=mount.token)
            elif action == "pause":
                api.pause_space(repo_id, token=mount.token)
            elif action == "restart":
                api.restart_space(repo_id, token=mount.token)
            else:
                body = b"Unknown action\n"
                start_response("400 Bad Request", [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))])
                return [body]
            print(f"[space-action] repo_id={repo_id} action={action} status=ok", flush=True)
        except Exception as exc:
            print(f"[space-action-error] repo_id={repo_id} action={action} error={exc}", flush=True)
            body = f"Space action failed: {exc}\n".encode("utf-8")
            start_response("500 Internal Server Error", [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))])
            return [body]

        body = f"Space action completed: {action}\n".encode("utf-8")
        start_response("200 OK", [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))])
        return [body]

    def _is_authorized(self, environ) -> bool:
        header = environ.get("HTTP_AUTHORIZATION", "")
        if not header.startswith("Basic "):
            return False
        token = header[6:].strip()
        try:
            decoded = base64.b64decode(token).decode("utf-8")
        except Exception:
            return False
        provided_username, sep, provided_password = decoded.partition(":")
        if not sep:
            return False
        return provided_username == self.username and provided_password == self.password

    def _unauthorized(self, start_response):
        body = b"Authentication required\n"
        start_response(
            "401 Unauthorized",
            [
                ("Content-Type", "text/plain; charset=utf-8"),
                ("Content-Length", str(len(body))),
                ("WWW-Authenticate", 'Basic realm="HF WebDAV"'),
            ],
        )
        return [body]


def _render_repo_row(
    mount: RepoMount,
    text: dict[str, object],
    events: list[dict[str, object]],
    start_label: str,
    pause_label: str,
    resume_label: str,
    restart_label: str,
) -> str:
    repo_id = mount.repo_id
    repo_type = mount.repo_type
    revision = mount.revision
    username, repo_name = repo_id.split("/", 1)
    dav_path = f"/dav/{username}/{repo_type}s/{repo_name}"
    repo_url = f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else 'spaces/' if repo_type == 'space' else ''}{repo_id}"
    type_labels = cast(dict[str, str], text["type_labels"])
    status_labels = cast(dict[str, str], text["status_labels"])
    source_labels = cast(dict[str, str], text["source_labels"])
    metadata = REPO_METADATA.get(repo_id, {})
    runtime_stage = str(metadata.get("runtime_stage", "-"))
    repo_size = str(metadata.get("repo_size", "-"))
    runtime_lower = runtime_stage.lower()
    status = "ok"
    source = "anonymous"
    message = ""
    for event in events:
        if event.get("username") == username:
            status = cast(str, event.get("status", status))
            source = cast(str, event.get("source", source))
            message = cast(str, event.get("message", message))
    actions = ""
    if repo_type == "space" and mount.token:
        escaped_repo_id = html.escape(repo_id)
        action_buttons = [
            f'<form method="post" action="/space-action?repo_id={escaped_repo_id}&action=restart"><button type="submit">{html.escape(restart_label)}</button></form>'
        ]
        if runtime_lower in {"paused", "sleeping", "stopped", "suspended"}:
            action_buttons.insert(
                0,
                f'<form method="post" action="/space-action?repo_id={escaped_repo_id}&action=resume"><button type="submit">{html.escape(resume_label)}</button></form>',
            )
        else:
            action_buttons.insert(
                0,
                f'<form method="post" action="/space-action?repo_id={escaped_repo_id}&action=pause"><button type="submit">{html.escape(pause_label)}</button></form>',
            )
            if runtime_lower in {"-", "build", "building", "no_app_file", "runtime_error", "error"}:
                action_buttons.insert(
                    0,
                    f'<form method="post" action="/space-action?repo_id={escaped_repo_id}&action=start"><button type="submit">{html.escape(start_label)}</button></form>',
                )
        actions = f'<div class="actions">{"".join(action_buttons)}</div>'
    return (
        '<article class="repo-card">'
        '<div class="repo-head">'
        '<div>'
        f'<h3 class="repo-title">{html.escape(repo_name)}</h3>'
        f'<div class="repo-sub"><code>{html.escape(repo_id)}</code></div>'
        '</div>'
        f'<span class="pill">{html.escape(type_labels.get(repo_type, repo_type))}</span>'
        '</div>'
        '<div class="repo-meta">'
        f'{_meta_item(alias_label := cast(str, text["alias"]), f"<code>{html.escape(username)}</code>")}'
        f'{_meta_item(revision_label := cast(str, text["revision"]), f"<code>{html.escape(revision)}</code>")}'
        f'{_meta_item(webdav_path_label := cast(str, text["webdav_path"]), f"<code>{html.escape(dav_path)}</code>")}'
        f'{_meta_item(hf_link_label := cast(str, text["hf_link_label"]), f"<a href=\"{html.escape(repo_url)}\" target=\"_blank\" rel=\"noreferrer\">{html.escape(repo_url)}</a>")}'
        f'{_meta_item(runtime_status_label := cast(str, text["runtime_status_label"]), html.escape(runtime_stage))}'
        f'{_meta_item(repo_size_label := cast(str, text["repo_size_label"]), html.escape(repo_size))}'
        f'{_meta_item(discovery_status_label := cast(str, text["discovery_status"]), html.escape(status_labels.get(status, status)))}'
        f'{_meta_item(discovery_source_label := cast(str, text["discovery_source"]), html.escape(source_labels.get(source, source)))}'
        f'{_meta_item(discovery_message_label := cast(str, text["discovery_message"]), html.escape(message or "-"))}'
        f'{_meta_item(space_actions_label := cast(str, text["space_actions"]), actions or "-")}'
        '</div>'
        '</article>'
    )


def _empty_row(text: dict[str, object]) -> str:
    empty_mounts = cast(str, text["empty_mounts"])
    return f'<div class="repo-card"><span style="color:#5f6b7a;">{html.escape(empty_mounts)}</span></div>'


def _meta_item(label: str, value: str) -> str:
    return (
        '<div class="meta-item">'
        f'<div class="meta-k">{html.escape(label)}</div>'
        f'<div class="meta-v">{value}</div>'
        '</div>'
    )


def _render_issues(issues: list[dict[str, object]], title: str, empty_text: str) -> str:
    if not issues:
        return f'<div class="issue"><strong>{html.escape(title)}</strong>{html.escape(empty_text)}</div>'
    rendered = []
    for issue in issues:
        username = cast(str, issue.get("username", "unknown"))
        message = cast(str, issue.get("message", ""))
        rendered.append(
            f'<div class="issue"><strong>{html.escape(username)}</strong>{html.escape(message)}</div>'
        )
    return "".join(rendered)


def _detect_language(environ) -> str:
    query = environ.get("QUERY_STRING", "")
    for part in query.split("&"):
        if not part:
            continue
        key, _, value = part.partition("=")
        if key == "lang":
            value = value.strip()
            if value in {"en", "zh-CN"}:
                return value
    accept_language = environ.get("HTTP_ACCEPT_LANGUAGE", "")
    if "zh" in accept_language.lower():
        return "zh-CN"
    return "en"


def _get_home_text(language: str) -> dict[str, object]:
    if language == "zh-CN":
        return {
            "title": "HF WebDAV Gateway",
            "eyebrow": "Hugging Face x WebDAV",
            "headline": "把 Hugging Face 仓库整理成清晰可读的 WebDAV 目录。",
            "intro": "首页位于 {path}，方便查看当前服务状态；真正给 WebDAV 客户端连接的入口是 {dav}。连接后，你会按“用户名 / 仓库类型 / 仓库名”的结构浏览内容。",
            "webdav_root": "WebDAV 入口",
            "health_check": "健康检查",
            "webdav_auth": "访问认证",
            "mounted_repositories": "已发现仓库",
            "alias": "用户名",
            "repo_id": "仓库标识",
            "type": "仓库类型",
            "revision": "版本分支",
            "webdav_path": "WebDAV 路径",
            "hint": "示例路径：<code>{example}</code>。目录结构固定为“用户名 / 仓库类型 / 仓库名”。如果某个用户没有列出任何仓库，通常是因为该用户没有公开仓库，或者需要提供对应 token。",
            "auth_enabled": "已开启",
            "auth_disabled": "未开启",
            "empty_mounts": "暂时还没有发现可访问的仓库。请检查 HF_WEBDAV_REPOSITORIES 中的用户名是否正确，以及私有仓库是否填写了对应 token。",
            "lang_note": "可随时切换中英文界面，不影响 WebDAV 访问路径。",
            "summary_users": "已配置用户 {count} 个",
            "summary_repos": "已发现仓库 {count} 个",
            "type_labels": {"model": "模型", "dataset": "数据集", "space": "Space"},
            "discovery_status": "发现状态",
            "discovery_source": "识别方式",
            "discovery_message": "说明",
            "issue_title": "配置提醒",
            "issue_empty": "当前没有发现明显的 token 或用户名错误。",
            "status_labels": {"ok": "正常", "warning": "需注意", "error": "错误"},
            "source_labels": {"token": "Token 自动识别", "anonymous": "按用户名匿名访问", "inline-pair": "用户名 + Token"},
            "hf_link_label": "HF 页面",
            "space_actions": "Space 操作",
            "runtime_status_label": "运行状态",
            "repo_size_label": "占用大小",
            "start_label": "启动",
            "resume_label": "恢复",
            "pause_label": "暂停",
            "restart_label": "重启",
        }
    return {
        "title": "HF WebDAV Gateway",
        "eyebrow": "Hugging Face x WebDAV",
        "headline": "Browse Hugging Face repositories through a cleaner WebDAV layout.",
        "intro": "This landing page sits at {path} so you can quickly inspect the service. The actual WebDAV endpoint is {dav}, where repositories are organized as username / repository type / repository name.",
        "webdav_root": "WebDAV Root",
        "health_check": "Health Check",
        "webdav_auth": "Access Control",
        "mounted_repositories": "Mounted Repositories",
        "alias": "Username",
        "repo_id": "Repo ID",
        "type": "Repository Type",
        "revision": "Revision",
        "webdav_path": "WebDAV Path",
        "hint": "Example path: <code>{example}</code>. The fixed layout is username / repository type / repository name. If a configured user has no repositories listed, the account may have no public repositories or it may require a token.",
        "auth_enabled": "Enabled",
        "auth_disabled": "Disabled",
        "empty_mounts": "No accessible repositories were discovered yet. Check the usernames in HF_WEBDAV_REPOSITORIES and provide tokens for private accounts if needed.",
        "lang_note": "Switching language only changes this page, not the WebDAV paths.",
        "summary_users": "Configured users: {count}",
        "summary_repos": "Discovered repositories: {count}",
        "type_labels": {"model": "Model", "dataset": "Dataset", "space": "Space"},
        "discovery_status": "Discovery Status",
        "discovery_source": "Source",
        "discovery_message": "Details",
        "issue_title": "Configuration Notes",
        "issue_empty": "No obvious token or username errors were detected.",
        "status_labels": {"ok": "OK", "warning": "Warning", "error": "Error"},
        "source_labels": {"token": "Token resolved", "anonymous": "Anonymous username", "inline-pair": "Username + token"},
        "hf_link_label": "HF Page",
        "space_actions": "Space Actions",
        "runtime_status_label": "Runtime",
        "repo_size_label": "Size",
        "start_label": "Start",
        "resume_label": "Resume",
        "pause_label": "Pause",
        "restart_label": "Restart",
    }


def _resolve_mounts(config: GatewayConfig) -> list[RepoMount]:
    if not config.discover_users:
        return []
    return _discover_user_mounts(config.discover_users)


def _discover_user_mounts(user_entries: tuple[dict[str, str], ...]) -> list[RepoMount]:

    mounts = []

    for entry in user_entries:
        raw_value = entry["value"]
        fallback_token = entry.get("token") or None
        username = raw_value
        token = None
        api = None
        token_identity = None
        token_namespaces: set[str] = set()

        if _looks_like_hf_token(raw_value):
            token = raw_value
            api = HfApi(token=token)
            try:
                token_identity = api.whoami(token=token)
                token_user = str(token_identity.get("name", "")).strip()
                if token_user:
                    username = token_user
                    token_namespaces.add(token_user.lower())
                for org in token_identity.get("orgs", []) or []:
                    org_name = str(org.get("name", "")).strip()
                    if org_name:
                        token_namespaces.add(org_name.lower())
                print(
                    "[discover-token] "
                    f"source=token resolved_username={username} namespaces={sorted(token_namespaces)}",
                    flush=True,
                )
                DISCOVERY_EVENTS.append(
                    {
                        "username": username,
                        "status": "ok",
                        "source": "token",
                        "message": "Username resolved automatically from token.",
                    }
                )
            except Exception as exc:
                print(
                    "[discover-token-error] "
                    f"source=token error={exc}",
                    flush=True,
                )
                DISCOVERY_EVENTS.append(
                    {
                        "username": raw_value,
                        "status": "error",
                        "source": "token",
                        "message": f"Token validation failed: {exc}",
                        "severity": "error",
                    }
                )
                token = None
                username = raw_value

        if token is None and fallback_token:
            token = fallback_token
            api = HfApi(token=token)
            print(
                "[discover-token] "
                f"source=inline-pair username={username}",
                flush=True,
            )
            DISCOVERY_EVENTS.append(
                {
                    "username": username,
                    "status": "ok",
                    "source": "inline-pair",
                    "message": "Using explicit username with inline token.",
                }
            )

        if token is None and not _looks_like_hf_token(raw_value):
            DISCOVERY_EVENTS.append(
                {
                    "username": username,
                    "status": "ok",
                    "source": "anonymous",
                    "message": "Querying public repositories without a token.",
                }
            )

        if api is None:
            api = HfApi(token=token)

        try:
            discovered_counts = {"model": 0, "dataset": 0, "space": 0}
            for repo_type, iterator in (
                ("model", api.list_models(author=username, token=token)),
                ("dataset", api.list_datasets(author=username, token=token)),
                ("space", api.list_spaces(author=username, token=token)),
            ):
                for item in iterator:
                    repo_id = str(getattr(item, "id", "")).strip()
                    if not repo_id:
                        continue
                    mounts.append(
                        RepoMount(
                            alias=repo_id.split("/", 1)[-1].strip() or repo_type,
                            repo_id=repo_id,
                            repo_type=repo_type,
                            revision="main",
                            token_env=None,
                            token=token,
                            discovery_source="token" if token else "anonymous",
                            discovery_status="ok",
                            discovery_message="",
                        )
                    )
                    REPO_METADATA[repo_id] = _load_repo_metadata(api, repo_id, repo_type, token)
                    discovered_counts[repo_type] += 1
            total = sum(discovered_counts.values())
            auth_mode = "token" if token else "anonymous"
            print(
                "[discover] "
                f"username={username} auth={auth_mode} "
                f"models={discovered_counts['model']} datasets={discovered_counts['dataset']} spaces={discovered_counts['space']} total={total}",
                flush=True,
            )
            if token and total == 0:
                if username.lower() not in token_namespaces:
                    print(
                        "[discover-warning] "
                        f"username={username} token_namespace_mismatch namespaces={sorted(token_namespaces)}",
                        flush=True,
                    )
                    DISCOVERY_EVENTS.append(
                        {
                            "username": username,
                            "status": "warning",
                            "source": "token",
                            "message": f"Token belongs to a different namespace: {sorted(token_namespaces)}",
                        }
                    )
                else:
                    print(
                        "[discover-warning] "
                        f"username={username} token_valid_but_no_visible_repositories",
                        flush=True,
                    )
                    DISCOVERY_EVENTS.append(
                        {
                            "username": username,
                            "status": "warning",
                            "source": "token",
                            "message": "Token is valid, but no visible repositories were found.",
                        }
                    )
        except Exception as exc:
            auth_mode = "token" if token else "anonymous"
            print(
                "[discover-error] "
                f"username={username} auth={auth_mode} error={exc}",
                flush=True,
            )
            DISCOVERY_EVENTS.append(
                {
                    "username": username,
                    "status": "error",
                    "source": "token" if token else "anonymous",
                    "message": f"Discovery failed: {exc}",
                    "severity": "error",
                }
            )
            if token is None:
                print(
                    "[discover-skip] "
                    f"username={username} reason=not_found_or_not_public",
                    flush=True,
                )
                DISCOVERY_EVENTS.append(
                    {
                        "username": username,
                        "status": "warning",
                        "source": "anonymous",
                        "message": "Username could not be resolved to any public repositories, so this entry was skipped.",
                    }
                )
    return mounts


def _looks_like_hf_token(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered.startswith("hf_")


def _load_repo_metadata(api: HfApi, repo_id: str, repo_type: str, token: str | None) -> dict[str, object]:
    metadata: dict[str, object] = {"runtime_stage": "-", "repo_size": "-"}
    try:
        info = api.repo_info(repo_id=repo_id, repo_type=repo_type, token=token, expand=["usedStorage"])
        used_storage = getattr(info, "usedStorage", None)
        if used_storage in (None, ""):
            siblings = getattr(info, "siblings", None) or []
            total_size = 0
            for sibling in siblings:
                size = getattr(sibling, "size", None)
                if isinstance(size, int):
                    total_size += size
            if total_size > 0:
                used_storage = total_size
        metadata["repo_size"] = _format_bytes(used_storage)
    except Exception as exc:
        print(f"[repo-metadata-warning] repo_id={repo_id} field=size error={exc}", flush=True)

    if repo_type == "space":
        try:
            runtime = api.get_space_runtime(repo_id, token=token)
            metadata["runtime_stage"] = str(getattr(runtime, "stage", "-") or "-")
        except Exception as exc:
            print(f"[repo-metadata-warning] repo_id={repo_id} field=runtime error={exc}", flush=True)

    return metadata


def _format_bytes(value) -> str:
    if not isinstance(value, (int, float)) or value < 0:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return "-"
