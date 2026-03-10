from __future__ import annotations

import argparse
import base64
import html
import io
import mimetypes
import os
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote
from urllib.request import Request, urlopen
from pathlib import Path
from typing import cast
from wsgiref.util import request_uri

from cheroot.wsgi import Server
from huggingface_hub import HfApi, hf_hub_url
from wsgidav import util as wsgidav_util
from wsgidav.wsgidav_app import WsgiDAVApp

from hf_webdav_gateway.config import GatewayConfig, RepoMount, load_config
from hf_webdav_gateway.provider import HfGatewayBackend, HfWebDavProvider


DISCOVERY_EVENTS: list[dict[str, object]] = []
REPO_METADATA: dict[str, dict[str, object]] = {}

# 全局变量用于定时刷新
_backend: HfGatewayBackend | None = None
_provider: HfWebDavProvider | None = None
_config_path: str | Path = "config.yaml"
_refresh_interval: int = 300  # 默认 5 分钟
_current_mounts: tuple[RepoMount, ...] = tuple()  # 当前最新的 mounts

# Windows WebDAV client may fire MKCOL concurrently/repeatedly.
# Use per-path locks to avoid redundant placeholder commits.
_mkcol_locks: dict[str, threading.Lock] = {}
_mkcol_locks_guard = threading.Lock()
PASSTHROUGH_RESPONSE_HEADERS = {
    "accept-ranges",
    "cache-control",
    "content-length",
    "content-range",
    "content-type",
    "etag",
    "last-modified",
}


def _refresh_mounts() -> tuple[RepoMount, ...]:
    """刷新仓库列表，返回最新的 mounts"""
    global _backend, _provider, DISCOVERY_EVENTS, REPO_METADATA, _current_mounts
    
    if _backend is None:
        return _current_mounts
    
    try:
        config = load_config(_config_path)
        mounts = _resolve_mounts(config)
        
        # 更新 backend 的挂载信息
        from hf_webdav_gateway.provider import _make_mount_record, _build_children_index
        _backend.records = [_make_mount_record(mount) for mount in mounts]
        _backend.mounts_by_root = {record.path_parts: record.mount for record in _backend.records}
        _backend.children_index = _build_children_index(_backend.records)
        _backend.list_dir.cache_clear()
        
        # 更新全局 mounts
        _current_mounts = tuple(mounts)
        
        total = len(mounts)
        print(f"[refresh] status=ok repos={total}", flush=True)
        return _current_mounts
    except Exception as exc:
        print(f"[refresh-error] error={exc}", flush=True)
        return _current_mounts


def _refresh_loop() -> None:
    """后台刷新循环"""
    while True:
        time.sleep(_refresh_interval)
        print("[refresh] starting scheduled refresh...", flush=True)
        _refresh_mounts()


def build_app(config_path: str | Path):
    global _backend, _provider, _config_path, _refresh_interval, _current_mounts
    
    DISCOVERY_EVENTS.clear()
    REPO_METADATA.clear()
    config = load_config(config_path)
    _config_path = config_path
    
    # 读取刷新间隔配置（分钟）
    refresh_env = os.getenv("HF_WEBDAV_REFRESH_INTERVAL", "").strip()
    if refresh_env:
        try:
            _refresh_interval = int(refresh_env) * 60  # 转换为秒
        except ValueError:
            pass
    
    mounts = _resolve_mounts(config)
    config = GatewayConfig(server=config.server, repositories=tuple(mounts), discover_users=tuple())
    _current_mounts = config.repositories
    _backend = HfGatewayBackend(config.repositories)
    _provider = HfWebDavProvider(_backend)

    # 获取认证用户名和密码
    username = os.getenv("HF_WEBDAV_USERNAME", "admin").strip() or "admin"
    password = os.getenv("HF_WEBDAV_PASSWORD", "admin")

    dav_app = WsgiDAVApp(
        {
            "provider_mapping": {"/": _provider},
            # 配置 simple_dc 允许写入
            "simple_dc": {
                "user_mapping": {
                    "*": {
                        username: {
                            "password": password,
                            "roles": [],
                        }
                    }
                }
            },
            "http_authenticator": {
                "accept_basic": True,
                "accept_digest": False,
                "trusted_auth_header": None,
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

    # 启动后台刷新线程
    refresh_thread = threading.Thread(target=_refresh_loop, daemon=True, name="refresh-worker")
    refresh_thread.start()
    print(f"[refresh] interval={_refresh_interval // 60} minutes thread=started", flush=True)

    # 使用 cheroot 多线程服务器，支持 100-continue
    server = Server(
        bind_addr=(host, port),
        wsgi_app=app,
        numthreads=10,
        max=-1,  # 无限制请求数
        request_queue_size=20,
        timeout=300,  # 5 分钟超时
        shutdown_timeout=5,
    )
    
    print(f"Hugging Face WebDAV gateway listening on http://{host}:{port}/")
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n[shutdown] received interrupt", flush=True)
        server.stop()

    return 0


class GatewayApp:
    def __init__(self, config: GatewayConfig, dav_app: WsgiDAVApp) -> None:
        self.config = config
        self.dav_app = dav_app
        self.username = os.getenv("HF_WEBDAV_USERNAME", "admin").strip() or "admin"
        self.password = os.getenv("HF_WEBDAV_PASSWORD", "admin")

    def __call__(self, environ, start_response):
        raw_path = environ.get("PATH_INFO", "") or "/"
        path = _normalize_wsgi_path(raw_path)
        method = (environ.get("REQUEST_METHOD", "GET") or "GET").upper()
        if path == "/space-action":
            return self._handle_space_action(environ, start_response)
        if path == "/":
            # 首页也需要认证
            if self._auth_enabled() and not self._is_authorized(environ):
                return self._unauthorized(start_response)
            return self._serve_home(environ, start_response)
        if path == "/healthz":
            body = b"ok\n"
            start_response(
                "200 OK",
                [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))],
            )
            return [body]
        if path.startswith("/dav"):
            if method in {"PROPFIND", "GET", "PUT", "DELETE", "MKCOL", "MOVE", "COPY", "PROPPATCH", "LOCK", "UNLOCK"}:
                print(f"[webdav-request] method={method} path={path}", flush=True)
            if self._auth_enabled() and not self._is_authorized(environ):
                return self._unauthorized(start_response)
            if method in {"GET", "HEAD"}:
                streamed = self._maybe_stream_file(path, method, environ, start_response)
                if streamed is not None:
                    return streamed
            if method == "MKCOL":
                forwarded_path = path[len("/dav") :] or "/"
                if self._handle_mkcol(forwarded_path, start_response):
                    return getattr(self, '_mkcol_response', [])

            if method == "COPY":
                forwarded_path = path[len("/dav") :] or "/"
                if self._handle_copy(forwarded_path, environ, start_response):
                    return getattr(self, "_copy_response", [])

            if method == "MOVE":
                forwarded_path = path[len("/dav") :] or "/"
                if self._handle_move(forwarded_path, environ, start_response):
                    return getattr(self, "_move_response", [])

            forwarded = dict(environ)
            forwarded["REMOTE_USER"] = self.username
            forwarded["SCRIPT_NAME"] = f"{environ.get('SCRIPT_NAME', '')}/dav".rstrip("/")
            forwarded_path = path[len("/dav") :] or "/"
            forwarded["PATH_INFO"] = _to_wsgi_path(forwarded_path)
            
            return self.dav_app(forwarded, start_response)

        body = b"Not Found\n"
        start_response(
            "404 Not Found",
            [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))],
        )
        return [body]

    def _serve_home(self, environ, start_response):
        # 访问首页时刷新仓库列表
        _refresh_mounts()
        
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
        # 使用刷新后的最新仓库列表
        rows = "\n".join(
            _render_repo_row(mount, text, DISCOVERY_EVENTS, start_label, pause_label, resume_label, restart_label)
            for mount in _current_mounts
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
    .actions button {{ padding: 8px 12px; border-radius: 999px; border: 1px solid var(--line); background: rgba(255,255,255,0.9); color: var(--ink); cursor: pointer; transition: all 0.2s; }}
    .actions button:hover {{ background: rgba(15,118,110,0.1); border-color: var(--accent); }}
    .actions button:disabled {{ opacity: 0.6; cursor: not-allowed; }}
    .actions button.loading {{ background: rgba(15,118,110,0.15); }}
    .actions button.success {{ background: rgba(34,197,94,0.2); border-color: #22c55e; }}
    .actions button.error {{ background: rgba(239,68,68,0.2); border-color: #ef4444; }}
    .toast {{ position: fixed; bottom: 20px; right: 20px; padding: 12px 20px; border-radius: 12px; color: #fff; font-size: 14px; z-index: 1000; opacity: 0; transform: translateY(20px); transition: all 0.3s; }}
    .toast.show {{ opacity: 1; transform: translateY(0); }}
    .toast.success {{ background: #22c55e; }}
    .toast.error {{ background: #ef4444; }}
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
        <div class=\"chip\"><span class=\"dot\"></span>{html.escape(summary_users.format(count=len({mount.repo_id.split('/', 1)[0] for mount in _current_mounts})) )}</div>
        <div class=\"chip\"><span class=\"dot\"></span>{html.escape(summary_repos.format(count=len(_current_mounts)))}</div>
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
  <div id="toast" class="toast"></div>
  <script>
    (function() {{
      var auth = btoa('{html.escape(self.username)}:{html.escape(self.password)}');
      
      function showToast(msg, type) {{
        var toast = document.getElementById('toast');
        toast.textContent = msg;
        toast.className = 'toast ' + type + ' show';
        setTimeout(function() {{ toast.classList.remove('show'); }}, 3000);
      }}
      
      document.querySelectorAll('.action-btn').forEach(function(btn) {{
        btn.addEventListener('click', function() {{
          var repo = this.dataset.repo;
          var action = this.dataset.action;
          var btnEl = this;
          
          if (btnEl.disabled) return;
          
          btnEl.disabled = true;
          btnEl.classList.add('loading');
          
          fetch('/space-action?repo_id=' + encodeURIComponent(repo) + '&action=' + action, {{
            method: 'POST',
            headers: {{ 'Authorization': 'Basic ' + auth }}
          }})
          .then(function(r) {{ return r.text().then(function(t) {{ return {{ ok: r.ok, text: t }}; }}); }})
          .then(function(result) {{
            btnEl.classList.remove('loading');
            if (result.ok) {{
              btnEl.classList.add('success');
              showToast(result.text.trim() || 'OK', 'success');
            }} else {{
              btnEl.classList.add('error');
              showToast(result.text.trim() || 'Error', 'error');
            }}
            setTimeout(function() {{
              btnEl.classList.remove('success', 'error');
              btnEl.disabled = false;
            }}, 2000);
          }})
          .catch(function(e) {{
            btnEl.classList.remove('loading');
            btnEl.classList.add('error');
            showToast('Request failed: ' + e.message, 'error');
            setTimeout(function() {{
              btnEl.classList.remove('error');
              btnEl.disabled = false;
            }}, 2000);
          }});
        }});
      }});
    }})();
  </script>
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

    def _handle_mkcol(self, forwarded_path: str, start_response) -> bool:
        """Handle MKCOL (create collection/folder) request directly.
        
        HuggingFace Hub doesn't support empty folders, so we create a .gitkeep placeholder file.
        This method handles all MKCOL requests directly instead of passing to wsgidav,
        because wsgidav's existence checks can cause issues with HF API caching delays.
        
        Returns True if handled, False to passthrough to wsgidav.
        Response body is stored in _mkcol_response if needed.
        """
        normalized = (forwarded_path or "/").rstrip("/") or "/"
        
        # Parse path to get mount info
        parts = tuple(part for part in normalized.strip("/").split("/") if part)
        
        # Debug: log parsed path info
        print(f"[webdav-mkcol-debug] normalized={normalized} parts={parts} backend_exists={_backend is not None} mounts_count={len(_current_mounts)}", flush=True)
        
        if len(parts) < 4:
            # Do not allow MKCOL outside repository paths.
            # Returning 405 prevents Windows Explorer from looping on virtual paths.
            print(f"[webdav-mkcol] path={normalized} status=not-allowed", flush=True)
            start_response("405 Method Not Allowed", [("Content-Length", "0")])
            return True
        
        # Check if this is a repository root
        repo_roots = {
            f"/{mount.repo_id.split('/', 1)[0]}/{mount.repo_type}s/{mount.repo_id.split('/', 1)[1]}".rstrip("/")
            for mount in _current_mounts
        }
        if normalized in repo_roots:
            # Repository roots already exist. Some clients expect MKCOL to be idempotent.
            print(f"[webdav-mkcol] path={normalized} status=already-exists", flush=True)
            start_response("201 Created", [("Content-Length", "0")])
            return True
        
        # Find the mount for this path using backend's mounts_by_root
        mount_root = (parts[0], parts[1], parts[2])
        mount = None
        
        # First try using backend's mounts_by_root for faster lookup
        if _backend is not None:
            mount = _backend.mounts_by_root.get(mount_root)
            if mount is None:
                # Debug: log available keys
                available_roots = list(_backend.mounts_by_root.keys())[:5]
                print(f"[webdav-mkcol-debug] mount_root={mount_root} available_roots={available_roots}", flush=True)
        
        # Fallback to searching _current_mounts
        if mount is None:
            for m in _current_mounts:
                m_root = (m.repo_id.split('/', 1)[0], f"{m.repo_type}s", m.repo_id.split('/', 1)[1])
                if m_root == mount_root:
                    mount = m
                    break
        
        if mount is None:
            # Repository not found - return error instead of passthrough
            # This prevents infinite loops
            print(f"[webdav-mkcol-error] path={normalized} mount_root={mount_root} current_mounts_count={len(_current_mounts)} reason=repo-not-found", flush=True)
            body = b"Repository not found\n"
            start_response("404 Not Found", [
                ("Content-Type", "text/plain; charset=utf-8"),
                ("Content-Length", str(len(body)))
            ])
            self._mkcol_response = [body]
            return True
        
        # Get token
        token = _token_for_mount(mount)
        if not token:
            print(f"[webdav-mkcol-error] path={normalized} reason=missing-token", flush=True)
            body = b"Creating folders requires a token-backed repository entry\n"
            start_response("403 Forbidden", [
                ("Content-Type", "text/plain; charset=utf-8"),
                ("Content-Length", str(len(body)))
            ])
            self._mkcol_response = [body]
            return True
        
        # Calculate folder path within repo
        folder_path = "/".join(parts[3:]) if len(parts) > 3 else ""

        placeholder_name = os.getenv("HF_WEBDAV_PLACEHOLDER_FILE", ".gitkeep").strip() or ".gitkeep"
        placeholder_path = f"{folder_path}/{placeholder_name}" if folder_path else placeholder_name

        lock_key = f"{mount.repo_type}:{mount.repo_id}:{mount.revision}:{folder_path}"
        with _mkcol_locks_guard:
            lock = _mkcol_locks.get(lock_key)
            if lock is None:
                lock = threading.Lock()
                _mkcol_locks[lock_key] = lock

        # Create the folder by uploading a placeholder file.
        # Note: Due to hiding the placeholder in listings, backend existence checks may be
        # inconsistent for placeholder-only folders. We treat the upstream conflict as the
        # authoritative "already exists" signal.
        with lock:
            try:
                if _backend is not None:
                    # Avoid returning 201 for obvious existing directories.
                    _backend.list_dir.cache_clear()
                    existing = _backend.get_entry(mount_root, folder_path)
                    if existing is not None and existing.is_dir:
                        print(
                            f"[webdav-mkcol] repo_id={mount.repo_id} path={folder_path or '/'} status=already-exists",
                            flush=True,
                        )
                        start_response("201 Created", [("Content-Length", "0")])
                        self._mkcol_response = []
                        return True

                api = HfApi(token=token)
                api.upload_file(
                    path_or_fileobj=io.BytesIO(b""),
                    path_in_repo=placeholder_path,
                    repo_id=mount.repo_id,
                    repo_type=mount.repo_type,
                    revision=mount.revision,
                    token=token,
                    commit_message=f"Create folder {folder_path or '/'} via WebDAV gateway",
                )

                if _backend is not None:
                    # Make the new directory visible immediately for follow-up PROPFIND.
                    mark = getattr(_backend, "mark_dir_created", None)
                    if callable(mark):
                        mark(mount_root, folder_path)
                    _backend.list_dir.cache_clear()

                print(f"[webdav-mkcol] repo_id={mount.repo_id} path={folder_path or '/'} status=created", flush=True)
                start_response("201 Created", [("Content-Length", "0")])
                self._mkcol_response = []
                return True

            except Exception as exc:
                # If placeholder already exists, treat MKCOL as idempotent success.
                message = str(exc).lower()
                if "already exists" in message or "409" in message or "conflict" in message:
                    print(
                        f"[webdav-mkcol] repo_id={mount.repo_id} path={folder_path or '/'} status=already-exists",
                        flush=True,
                    )
                    start_response("201 Created", [("Content-Length", "0")])
                    self._mkcol_response = []
                    return True
                print(
                    f"[webdav-mkcol-error] repo_id={mount.repo_id} path={folder_path or '/'} error={exc}",
                    flush=True,
                )
                body = f"Failed to create folder: {exc}\n".encode("utf-8")
                start_response(
                    "500 Internal Server Error",
                    [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))],
                )
                self._mkcol_response = [body]
                return True

    def _handle_copy(self, forwarded_path: str, environ, start_response) -> bool:
        """Handle COPY request directly for repository paths.

        Some clients (e.g. openlist) rely on COPY for folder copy and may retry
        aggressively when the server implementation is incomplete.
        """
        if _backend is None:
            return False

        raw_src = forwarded_path or "/"
        src_normalized = (raw_src or "/").lstrip("/")
        src_parts = tuple(part for part in src_normalized.split("/") if part)
        if len(src_parts) < 4:
            return False

        mount_root = (src_parts[0], src_parts[1], src_parts[2])
        mount = _backend.mounts_by_root.get(mount_root)
        if mount is None:
            return False

        token = _token_for_mount(mount)
        if not token:
            body = b"Copy requires a token-backed repository entry\n"
            start_response(
                "403 Forbidden",
                [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))],
            )
            self._copy_response = [body]
            return True

        dest_header = environ.get("HTTP_DESTINATION", "") or ""
        dest_path = _normalize_destination_path(dest_header)
        if not dest_path:
            body = b"Missing Destination header\n"
            start_response(
                "400 Bad Request",
                [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))],
            )
            self._copy_response = [body]
            return True

        # Destination is expected to include /dav prefix. Strip it.
        if dest_path.startswith("/dav/"):
            dest_forwarded = dest_path[len("/dav") :]
        elif dest_path == "/dav":
            dest_forwarded = "/"
        else:
            dest_forwarded = dest_path

        dest_parts = tuple(part for part in dest_forwarded.strip("/").split("/") if part)
        if len(dest_parts) < 4:
            body = b"Invalid Destination path\n"
            start_response(
                "400 Bad Request",
                [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))],
            )
            self._copy_response = [body]
            return True

        dest_mount_root = (dest_parts[0], dest_parts[1], dest_parts[2])
        if dest_mount_root != mount_root:
            body = b"Cross-repository copy is not supported\n"
            start_response(
                "403 Forbidden",
                [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))],
            )
            self._copy_response = [body]
            return True

        src_repo_path = "/".join(src_parts[3:])
        dest_repo_path = "/".join(dest_parts[3:])
        overwrite = (environ.get("HTTP_OVERWRITE", "") or "").strip().upper() != "F"

        print(
            "[webdav-copy-request] "
            f"repo_id={mount.repo_id} src={src_repo_path} dest={dest_repo_path} overwrite={overwrite} destination={dest_header}",
            flush=True,
        )

        # Precondition check when overwrite is disabled.
        if not overwrite:
            existing = _backend.get_entry(mount_root, dest_repo_path)
            if existing is not None:
                body = b"Destination exists and overwrite is disabled\n"
                start_response(
                    "412 Precondition Failed",
                    [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))],
                )
                self._copy_response = [body]
                return True

        try:
            # Treat trailing slash as a hint that the source is a collection.
            src_is_collection = (raw_src or "").endswith("/")
            entry = _backend.get_entry(mount_root, src_repo_path)
            if entry is not None:
                src_is_collection = src_is_collection or entry.is_dir

            if src_is_collection:
                _backend.copy_folder(mount_root, src_repo_path, dest_repo_path)
            else:
                api = HfApi(token=token)
                url = hf_hub_url(
                    repo_id=mount.repo_id,
                    filename=src_repo_path,
                    repo_type=mount.repo_type,
                    revision=mount.revision,
                )
                with urlopen(Request(url, headers={"Authorization": f"Bearer {token}"}), timeout=120) as resp:
                    data = resp.read()
                api.upload_file(
                    path_or_fileobj=data,
                    path_in_repo=dest_repo_path,
                    repo_id=mount.repo_id,
                    repo_type=mount.repo_type,
                    revision=mount.revision,
                    token=token,
                    commit_message=f"Copy {src_repo_path} to {dest_repo_path} via WebDAV gateway",
                )

            start_response("201 Created", [("Content-Length", "0")])
            self._copy_response = []
            return True
        except Exception as exc:
            print(
                f"[webdav-copy-error] repo_id={mount.repo_id} src={src_repo_path} dest={dest_repo_path} error={exc}",
                flush=True,
            )
            body = f"Failed to copy: {exc}\n".encode("utf-8")
            start_response(
                "500 Internal Server Error",
                [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))],
            )
            self._copy_response = [body]
            return True

    def _handle_move(self, forwarded_path: str, environ, start_response) -> bool:
        """Handle MOVE request directly for repository paths.

        Many clients use MOVE for folder rename/move and may retry aggressively.
        We implement MOVE as copy + delete within the same repository.
        """
        if _backend is None:
            return False

        raw_src = forwarded_path or "/"
        src_normalized = (raw_src or "/").lstrip("/")
        src_parts = tuple(part for part in src_normalized.split("/") if part)
        if len(src_parts) < 4:
            return False

        mount_root = (src_parts[0], src_parts[1], src_parts[2])
        mount = _backend.mounts_by_root.get(mount_root)
        if mount is None:
            return False

        token = _token_for_mount(mount)
        if not token:
            body = b"Move requires a token-backed repository entry\n"
            start_response(
                "403 Forbidden",
                [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))],
            )
            self._move_response = [body]
            return True

        dest_header = environ.get("HTTP_DESTINATION", "") or ""
        dest_path = _normalize_destination_path(dest_header)
        if not dest_path:
            body = b"Missing Destination header\n"
            start_response(
                "400 Bad Request",
                [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))],
            )
            self._move_response = [body]
            return True

        if dest_path.startswith("/dav/"):
            dest_forwarded = dest_path[len("/dav") :]
        elif dest_path == "/dav":
            dest_forwarded = "/"
        else:
            dest_forwarded = dest_path

        dest_parts = tuple(part for part in dest_forwarded.strip("/").split("/") if part)
        if len(dest_parts) < 4:
            body = b"Invalid Destination path\n"
            start_response(
                "400 Bad Request",
                [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))],
            )
            self._move_response = [body]
            return True

        dest_mount_root = (dest_parts[0], dest_parts[1], dest_parts[2])
        if dest_mount_root != mount_root:
            body = b"Cross-repository move is not supported\n"
            start_response(
                "403 Forbidden",
                [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))],
            )
            self._move_response = [body]
            return True

        src_repo_path = "/".join(src_parts[3:])
        dest_repo_path = "/".join(dest_parts[3:])
        overwrite = (environ.get("HTTP_OVERWRITE", "") or "").strip().upper() != "F"

        print(
            "[webdav-move-request] "
            f"repo_id={mount.repo_id} src={src_repo_path} dest={dest_repo_path} overwrite={overwrite} destination={dest_header}",
            flush=True,
        )

        if not overwrite:
            existing = _backend.get_entry(mount_root, dest_repo_path)
            if existing is not None:
                body = b"Destination exists and overwrite is disabled\n"
                start_response(
                    "412 Precondition Failed",
                    [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))],
                )
                self._move_response = [body]
                return True

        try:
            src_is_collection = (raw_src or "").endswith("/")
            entry = _backend.get_entry(mount_root, src_repo_path)
            if entry is not None:
                src_is_collection = src_is_collection or entry.is_dir

            if src_is_collection:
                _backend.copy_folder(mount_root, src_repo_path, dest_repo_path)
                _backend.delete_path(mount_root, src_repo_path, is_dir=True)
            else:
                api = HfApi(token=token)
                url = hf_hub_url(
                    repo_id=mount.repo_id,
                    filename=src_repo_path,
                    repo_type=mount.repo_type,
                    revision=mount.revision,
                )
                with urlopen(Request(url, headers={"Authorization": f"Bearer {token}"}), timeout=120) as resp:
                    data = resp.read()
                api.upload_file(
                    path_or_fileobj=data,
                    path_in_repo=dest_repo_path,
                    repo_id=mount.repo_id,
                    repo_type=mount.repo_type,
                    revision=mount.revision,
                    token=token,
                    commit_message=f"Move {src_repo_path} to {dest_repo_path} via WebDAV gateway",
                )
                _backend.delete_path(mount_root, src_repo_path, is_dir=False)

            start_response("201 Created", [("Content-Length", "0")])
            self._move_response = []
            return True
        except Exception as exc:
            print(
                f"[webdav-move-error] repo_id={mount.repo_id} src={src_repo_path} dest={dest_repo_path} error={exc}",
                flush=True,
            )
            body = f"Failed to move: {exc}\n".encode("utf-8")
            start_response(
                "500 Internal Server Error",
                [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))],
            )
            self._move_response = [body]
            return True


    def _maybe_stream_file(self, path: str, method: str, environ, start_response):
        parsed = _parse_dav_file_request(path)
        if parsed is None or _backend is None:
            return None
        mount_root, repo_path = parsed
        entry = _backend.get_entry(mount_root, repo_path)
        if entry is None or entry.is_dir:
            return None

        mount = _backend.mounts_by_root.get(mount_root)
        if mount is None:
            return None

        file_url = hf_hub_url(
            repo_id=mount.repo_id,
            filename=repo_path,
            repo_type=mount.repo_type,
            revision=mount.revision,
        )
        headers = {"User-Agent": "hf-webdav-gateway/1.0"}
        requested_range = environ.get("HTTP_RANGE", "")
        if requested_range:
            headers["Range"] = requested_range
        token = _token_for_mount(mount)
        if token:
            headers["Authorization"] = f"Bearer {token}"

        print(
            f"[webdav-stream] repo_id={mount.repo_id} path={repo_path} method={method} range={requested_range or '-'}",
            flush=True,
        )

        request = Request(file_url, method=method, headers=headers)
        try:
            upstream = urlopen(request, timeout=120)
        except HTTPError as exc:
            body = exc.read() if method != "HEAD" else b""
            message = body if body else f"Upstream media request failed: {exc}".encode("utf-8", errors="replace")
            start_response(
                f"{exc.code} {exc.reason}",
                [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(message)))],
            )
            print(
                f"[webdav-stream-error] repo_id={mount.repo_id} path={repo_path} status={exc.code} error={exc}",
                flush=True,
            )
            return [] if method == "HEAD" else [message]
        except URLError as exc:
            message = f"Upstream media request failed: {exc}".encode("utf-8", errors="replace")
            start_response(
                "502 Bad Gateway",
                [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(message)))],
            )
            print(
                f"[webdav-stream-error] repo_id={mount.repo_id} path={repo_path} status=502 error={exc}",
                flush=True,
            )
            return [message]

        response_headers = _select_passthrough_headers(upstream.headers)
        if not any(name.lower() == "content-type" for name, _ in response_headers):
            response_headers.append(("Content-Type", mimetypes.guess_type(repo_path)[0] or "application/octet-stream"))
        if not any(name.lower() == "accept-ranges" for name, _ in response_headers):
            response_headers.append(("Accept-Ranges", "bytes"))

        start_response(f"{upstream.status} {upstream.reason}", response_headers)
        if method == "HEAD":
            upstream.close()
            return []
        return _stream_upstream_response(upstream)

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
        mount = next((item for item in _current_mounts if item.repo_id == repo_id and item.repo_type == "space"), None)
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
    alias_label = cast(str, text["alias"])
    revision_label = cast(str, text["revision"])
    webdav_path_label = cast(str, text["webdav_path"])
    hf_link_label = cast(str, text["hf_link_label"])
    runtime_status_label = cast(str, text["runtime_status_label"])
    repo_size_label = cast(str, text["repo_size_label"])
    discovery_status_label = cast(str, text["discovery_status"])
    discovery_source_label = cast(str, text["discovery_source"])
    discovery_message_label = cast(str, text["discovery_message"])
    space_actions_label = cast(str, text["space_actions"])
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
            f'<button type="button" class="action-btn" data-repo="{escaped_repo_id}" data-action="restart">{html.escape(restart_label)}</button>'
        ]
        if runtime_lower in {"paused", "sleeping", "stopped", "suspended"}:
            action_buttons.insert(
                0,
                f'<button type="button" class="action-btn" data-repo="{escaped_repo_id}" data-action="resume">{html.escape(resume_label)}</button>',
            )
        else:
            action_buttons.insert(
                0,
                f'<button type="button" class="action-btn" data-repo="{escaped_repo_id}" data-action="pause">{html.escape(pause_label)}</button>',
            )
            if runtime_lower in {"-", "build", "building", "no_app_file", "runtime_error", "error"}:
                action_buttons.insert(
                    0,
                    f'<button type="button" class="action-btn" data-repo="{escaped_repo_id}" data-action="start">{html.escape(start_label)}</button>',
                )
        actions = f'<div class="actions">{"".join(action_buttons)}</div>'
    repo_link_html = f'<a href="{html.escape(repo_url)}" target="_blank" rel="noreferrer">{html.escape(repo_url)}</a>'
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
        f'{_meta_item(alias_label, f"<code>{html.escape(username)}</code>")}'
        f'{_meta_item(revision_label, f"<code>{html.escape(revision)}</code>")}'
        f'{_meta_item(webdav_path_label, f"<code>{html.escape(dav_path)}</code>")}'
        f'{_meta_item(hf_link_label, repo_link_html)}'
        f'{_meta_item(runtime_status_label, html.escape(runtime_stage))}'
        f'{_meta_item(repo_size_label, html.escape(repo_size))}'
        f'{_meta_item(discovery_status_label, html.escape(status_labels.get(status, status)))}'
        f'{_meta_item(discovery_source_label, html.escape(source_labels.get(source, source)))}'
        f'{_meta_item(discovery_message_label, html.escape(message or "-"))}'
        f'{_meta_item(space_actions_label, actions or "-")}'
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


def _normalize_wsgi_path(path: str) -> str:
    if not path:
        return "/"
    try:
        return path.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return path


def _to_wsgi_path(path: str) -> str:
    try:
        return wsgidav_util.unicode_to_wsgi(path)
    except Exception:
        return path


def _parse_dav_file_request(path: str) -> tuple[tuple[str, str, str], str] | None:
    if not path.startswith("/dav"):
        return None
    parts = tuple(part for part in path[len("/dav") :].strip("/").split("/") if part)
    if len(parts) < 4:
        return None
    if parts[1] not in {"models", "datasets", "spaces"}:
        return None
    return (parts[0], parts[1], parts[2]), "/".join(parts[3:])


def _token_for_mount(mount: RepoMount) -> str | None:
    if mount.token:
        return mount.token
    if not mount.token_env:
        return None
    token = os.getenv(mount.token_env, "").strip()
    return token or None


def _select_passthrough_headers(headers) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    for name, value in headers.items():
        if name.lower() in PASSTHROUGH_RESPONSE_HEADERS:
            selected.append((name, value))
    return selected


def _stream_upstream_response(upstream, chunk_size: int = 1024 * 256):
    try:
        while True:
            chunk = upstream.read(chunk_size)
            if not chunk:
                break
            yield chunk
    finally:
        upstream.close()


def _normalize_destination_path(value: str) -> str:
    """Normalize a WebDAV Destination header to an absolute path."""
    text = (value or "").strip()
    if not text:
        return ""
    if "://" in text:
        try:
            after = text.split("://", 1)[1]
            idx = after.find("/")
            text = after[idx:] if idx >= 0 else "/"
        except Exception:
            text = "/"
    if not text.startswith("/"):
        text = "/" + text
    return unquote(text)


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
