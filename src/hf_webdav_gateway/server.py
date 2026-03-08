from __future__ import annotations

import argparse
import base64
import html
import os
from pathlib import Path
from wsgiref.simple_server import make_server
from wsgiref.util import request_uri

from wsgidav.wsgidav_app import WsgiDAVApp

from hf_webdav_gateway.config import GatewayConfig, load_config
from hf_webdav_gateway.provider import HfGatewayBackend, HfWebDavProvider


def build_app(config_path: str | Path):
    config = load_config(config_path)
    backend = HfGatewayBackend(config.repositories)
    provider = HfWebDavProvider(backend)

    dav_app = WsgiDAVApp(
        {
            "provider_mapping": {"/": provider},
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
        base_url = request_uri(environ, include_query=False).rstrip("/")
        dav_url = f"{base_url}/dav"
        auth_status = text["auth_enabled"] if self._auth_enabled() else text["auth_disabled"]
        rows = "\n".join(
            _render_repo_row(
                mount.alias,
                mount.repo_id,
                mount.repo_type,
                mount.revision,
                mount.token_env,
                text,
            )
            for mount in self.config.repositories
        )
        body = f"""<!doctype html>
<html lang=\"{html.escape(language)}\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{html.escape(text['title'])}</title>
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
    .langbar {{ display: flex; justify-content: flex-end; gap: 10px; margin-bottom: 14px; }}
    .langlink {{
      text-decoration: none; padding: 8px 12px; border-radius: 999px; border: 1px solid var(--line);
      background: rgba(255,255,255,0.62); color: var(--ink); font: 600 12px/1.2 "Trebuchet MS", sans-serif;
      letter-spacing: 0.06em; text-transform: uppercase;
    }}
    .langlink.active {{ background: rgba(15,118,110,0.12); color: var(--accent); border-color: rgba(15,118,110,0.28); }}
    .url {{
      display: block; overflow-wrap: anywhere; text-decoration: none; color: var(--ink);
      padding: 14px 16px; border-radius: 16px; background: rgba(255,255,255,0.7); border: 1px solid var(--line);
      font-family: "Courier New", monospace;
    }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ text-align: left; padding: 14px 10px; border-top: 1px solid var(--line); vertical-align: top; }}
    th {{ font: 600 12px/1.2 "Trebuchet MS", sans-serif; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }}
    code {{ font-family: "Courier New", monospace; }}
    .hint {{ margin-top: 22px; font-size: 0.96rem; }}
    .path {{ color: var(--accent); font-weight: 700; }}
    @media (max-width: 640px) {{
      .shell {{ padding: 24px 14px 40px; }}
      .hero, .panel {{ border-radius: 18px; }}
      th, td {{ padding: 12px 8px; font-size: 0.94rem; }}
    }}
  </style>
</head>
<body>
  <main class=\"shell\">
    <div class=\"langbar\">
      <a class=\"langlink{' active' if language == 'en' else ''}\" href=\"/?lang=en\">English</a>
      <a class=\"langlink{' active' if language == 'zh-CN' else ''}\" href=\"/?lang=zh-CN\">中文</a>
    </div>
    <section class=\"hero\">
      <span class=\"eyebrow\">{html.escape(text['eyebrow'])}</span>
      <h1>{html.escape(text['headline'])}</h1>
      <p>{text['intro'].format(path='<code>/</code>', dav='<span class="path">/dav</span>')}</p>
      <div class=\"grid\">
        <section class=\"panel\">
          <div class=\"k\">{html.escape(text['webdav_root'])}</div>
          <a class=\"url\" href=\"{html.escape(dav_url)}\">{html.escape(dav_url)}</a>
        </section>
        <section class=\"panel\">
          <div class=\"k\">{html.escape(text['health_check'])}</div>
          <a class=\"url\" href=\"{html.escape(base_url + '/healthz')}\">{html.escape(base_url + '/healthz')}</a>
        </section>
        <section class=\"panel\">
          <div class=\"k\">{html.escape(text['webdav_auth'])}</div>
          <div class=\"url\">{html.escape(auth_status)}</div>
        </section>
      </div>
    </section>
    <section class=\"panel\" style=\"margin-top: 20px;\">
      <div class=\"k\">{html.escape(text['mounted_repositories'])}</div>
      <table>
        <thead>
          <tr>
            <th>{html.escape(text['alias'])}</th>
            <th>{html.escape(text['repo_id'])}</th>
            <th>{html.escape(text['type'])}</th>
            <th>{html.escape(text['revision'])}</th>
            <th>{html.escape(text['webdav_path'])}</th>
          </tr>
        </thead>
        <tbody>
          {rows or _empty_row(text)}
        </tbody>
      </table>
      <p class=\"hint\">{text['hint'].format(example=html.escape(dav_url) + '/models')}</p>
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
    alias: str,
    repo_id: str,
    repo_type: str,
    revision: str,
    token_env: str | None,
    text: dict[str, str],
) -> str:
    token_label = f" <code>({html.escape(text['token_env'])}: {html.escape(token_env)})</code>" if token_env else ""
    dav_path = f"/dav/{alias}"
    return (
        "<tr>"
        f"<td><code>{html.escape(alias)}</code></td>"
        f"<td><code>{html.escape(repo_id)}</code>{token_label}</td>"
        f"<td>{html.escape(repo_type)}</td>"
        f"<td><code>{html.escape(revision)}</code></td>"
        f"<td><code>{html.escape(dav_path)}</code></td>"
        "</tr>"
    )


def _empty_row(text: dict[str, str]) -> str:
    return (
        "<tr>"
        f"<td colspan=\"5\"><span style=\"color:#5f6b7a;\">{html.escape(text['empty_mounts'])}</span></td>"
        "</tr>"
    )


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


def _get_home_text(language: str) -> dict[str, str]:
    if language == "zh-CN":
        return {
            "title": "HF WebDAV Gateway",
            "eyebrow": "Hugging Face x WebDAV",
            "headline": "通过清晰的 WebDAV 入口浏览已挂载仓库。",
            "intro": "这个页面在 {path} 提供可读首页，真正的 WebDAV 文件系统位于 {dav}。你可以使用任意 WebDAV 客户端连接并浏览下方列出的仓库。",
            "webdav_root": "WebDAV 根路径",
            "health_check": "健康检查",
            "webdav_auth": "WebDAV 认证",
            "mounted_repositories": "已挂载仓库",
            "alias": "别名",
            "repo_id": "仓库 ID",
            "type": "类型",
            "revision": "版本",
            "webdav_path": "WebDAV 路径",
            "hint": "示例路径：<code>{example}</code>。私有仓库仍可通过服务配置中的 token 环境变量访问。",
            "auth_enabled": "已启用",
            "auth_disabled": "未启用",
            "token_env": "令牌环境变量",
            "empty_mounts": "当前还没有配置任何仓库挂载。请设置 config.yaml 或 HF_WEBDAV_REPOSITORIES。",
        }
    return {
        "title": "HF WebDAV Gateway",
        "eyebrow": "Hugging Face x WebDAV",
        "headline": "Browse mounted repos through a clean WebDAV endpoint.",
        "intro": "This page provides a friendly landing view at {path}, while the actual WebDAV filesystem lives under {dav}. Use any WebDAV client to connect and browse the repositories listed below.",
        "webdav_root": "WebDAV Root",
        "health_check": "Health Check",
        "webdav_auth": "WebDAV Auth",
        "mounted_repositories": "Mounted Repositories",
        "alias": "Alias",
        "repo_id": "Repo ID",
        "type": "Type",
        "revision": "Revision",
        "webdav_path": "WebDAV Path",
        "hint": "Example path: <code>{example}</code>. Private repositories can still use token environment variables defined in the server config.",
        "auth_enabled": "Enabled",
        "auth_disabled": "Disabled",
        "token_env": "Token Env",
        "empty_mounts": "No repository mounts are configured yet. Set config.yaml or HF_WEBDAV_REPOSITORIES to add some.",
    }
