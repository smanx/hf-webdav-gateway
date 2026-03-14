"""API call monitoring module for tracking HuggingFace API usage.

This module provides functionality to monitor and log all API calls made to
HuggingFace Hub, helping identify rate limiting issues (429 errors) and
optimize API usage patterns.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from functools import wraps
import contextlib

# Environment variable configuration
API_LOGGING_ENABLED = os.getenv("HF_WEBDAV_API_LOGGING", "1").strip().lower() not in ("0", "false", "no", "off")
API_STATS_MAX = int(os.getenv("HF_WEBDAV_API_STATS_MAX", "1000") or "1000")


@dataclass
class ApiCallRecord:
    """Represents a single API call record."""
    timestamp: float  # Unix timestamp
    api_name: str  # e.g., "list_models", "list_repo_tree"
    repo_id: str | None  # Repository ID if applicable
    source: str  # What triggered this call, e.g., "propfind", "home_refresh"
    duration_ms: float  # Duration in milliseconds
    status: str  # "ok", "429", "error", etc.
    error_msg: str | None = None  # Error message if failed


@dataclass
class ApiStats:
    """Statistics for API calls."""
    total_calls: int = 0
    total_429_errors: int = 0
    total_errors: int = 0
    calls_by_api: dict[str, int] = field(default_factory=dict)
    errors_429_by_api: dict[str, int] = field(default_factory=dict)
    last_refresh_time: float | None = None


class ApiMonitor:
    """Thread-safe API call monitor with statistics tracking."""

    def __init__(self, max_records: int = API_STATS_MAX):
        self._records: deque[ApiCallRecord] = deque(maxlen=max_records)
        self._stats = ApiStats()
        self._lock = threading.Lock()

    def record_call(
        self,
        api_name: str,
        source: str,
        duration_ms: float,
        status: str = "ok",
        repo_id: str | None = None,
        error_msg: str | None = None,
    ) -> None:
        """Record an API call.

        Args:
            api_name: Name of the API method called
            source: What triggered this call
            duration_ms: Duration in milliseconds
            status: Result status ("ok", "429", "error", etc.)
            repo_id: Repository ID if applicable
            error_msg: Error message if failed
        """
        record = ApiCallRecord(
            timestamp=time.time(),
            api_name=api_name,
            repo_id=repo_id,
            source=source,
            duration_ms=duration_ms,
            status=status,
            error_msg=error_msg,
        )

        with self._lock:
            self._records.append(record)
            self._stats.total_calls += 1

            # Update per-API counts
            self._stats.calls_by_api[api_name] = self._stats.calls_by_api.get(api_name, 0) + 1

            # Track errors
            if status == "429":
                self._stats.total_429_errors += 1
                self._stats.errors_429_by_api[api_name] = self._stats.errors_429_by_api.get(api_name, 0) + 1
            elif status not in ("ok", "skipped"):
                self._stats.total_errors += 1

        # Log to console if enabled
        if API_LOGGING_ENABLED:
            self._print_log(record)

    def _print_log(self, record: ApiCallRecord) -> None:
        """Print a log message to console."""
        ts = datetime.fromtimestamp(record.timestamp).strftime("%H:%M:%S.%f")[:12]
        repo_part = f" repo={record.repo_id}" if record.repo_id else ""
        error_part = f" error={record.error_msg}" if record.error_msg else ""
        status_str = f"[{record.status}]" if record.status != "ok" else ""

        print(
            f"[api-call] ts={ts} api={record.api_name} source={record.source}"
            f"{repo_part} duration={record.duration_ms:.0f}ms{status_str}{error_part}",
            flush=True,
        )

    def mark_refresh(self) -> None:
        """Mark that a repository refresh has occurred."""
        with self._lock:
            self._stats.last_refresh_time = time.time()

    def get_stats(self) -> dict[str, Any]:
        """Get current statistics as a dictionary."""
        with self._lock:
            recent_records = list(reversed(self._records))  # Most recent first

        # Calculate time since last refresh
        last_refresh_ago = None
        if self._stats.last_refresh_time:
            last_refresh_ago = time.time() - self._stats.last_refresh_time

        return {
            "summary": {
                "total_calls": self._stats.total_calls,
                "total_429_errors": self._stats.total_429_errors,
                "total_errors": self._stats.total_errors,
                "last_refresh_ago_seconds": last_refresh_ago,
            },
            "by_api": dict(self._stats.calls_by_api),
            "errors_429_by_api": dict(self._stats.errors_429_by_api),
            "recent_calls": [
                {
                    "timestamp": r.timestamp,
                    "time": datetime.fromtimestamp(r.timestamp).isoformat(),
                    "api": r.api_name,
                    "repo_id": r.repo_id,
                    "source": r.source,
                    "duration_ms": round(r.duration_ms, 1),
                    "status": r.status,
                    "error": r.error_msg,
                }
                for r in recent_records
            ],
        }

    def get_stats_html(self) -> str:
        """Get statistics as an HTML page."""
        stats = self.get_stats()
        summary = stats["summary"]
        by_api = stats["by_api"]
        errors_429 = stats["errors_429_by_api"]
        recent = stats["recent_calls"]

        # Build by-API section
        api_rows = []
        for api_name in sorted(by_api.keys()):
            count = by_api[api_name]
            err_429 = errors_429.get(api_name, 0)
            err_str = f" ({err_429} x 429)" if err_429 else ""
            api_rows.append(f"<tr><td>{api_name}</td><td>{count}{err_str}</td></tr>")

        # Build recent calls section
        recent_rows = []
        for r in recent[:50]:  # Last 50 calls (most recent first)
            ts = datetime.fromtimestamp(r["timestamp"]).strftime("%H:%M:%S")
            repo = r["repo_id"] or "-"
            status_class = "status-error" if r["status"] != "ok" else "status-ok"
            status_str = r["status"] if r["status"] != "ok" else ""
            recent_rows.append(
                f"<tr class='{status_class}'>"
                f"<td>{ts}</td><td>{r['api']}</td><td>{repo}</td><td>{r['source']}</td>"
                f"<td>{r['duration_ms']:.0f}ms</td><td>{status_str}</td>"
                f"</tr>"
            )

        # Last refresh time
        last_refresh = "-"
        if summary["last_refresh_ago_seconds"] is not None:
            secs = int(summary["last_refresh_ago_seconds"])
            if secs < 60:
                last_refresh = f"{secs}s ago"
            elif secs < 3600:
                last_refresh = f"{secs // 60}m ago"
            else:
                last_refresh = f"{secs // 3600}h {(secs % 3600) // 60}m ago"

        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>API Stats - HF WebDAV Gateway</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --panel: rgba(255, 252, 247, 0.88);
      --ink: #1f2937;
      --muted: #5f6b7a;
      --line: rgba(31, 41, 55, 0.12);
      --accent: #0f766e;
      --error: #dc2626;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      color: var(--ink);
      background: var(--bg);
      min-height: 100vh;
      padding: 24px;
    }}
    .container {{ max-width: 1000px; margin: 0 auto; }}
    h1 {{ margin: 0 0 24px; font-size: 1.5rem; }}
    .grid {{ display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); margin-bottom: 24px; }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 16px;
    }}
    .card-value {{ font-size: 2rem; font-weight: 600; }}
    .card-label {{ color: var(--muted); font-size: 0.875rem; }}
    .card.error {{ border-color: rgba(220, 38, 38, 0.3); }}
    .card.error .card-value {{ color: var(--error); }}
    table {{ width: 100%; border-collapse: collapse; background: var(--panel); border-radius: 12px; overflow: hidden; }}
    th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--line); }}
    th {{ background: rgba(15, 118, 110, 0.08); font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }}
    .status-error {{ background: rgba(220, 38, 38, 0.06); }}
    .status-ok {{ }}
    code {{ font-family: "SF Mono", Consolas, monospace; font-size: 0.875em; }}
    .section {{ margin-bottom: 24px; }}
    .section-title {{ font-size: 0.875rem; font-weight: 600; color: var(--accent); margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.05em; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>API Call Statistics</h1>
    
    <div class="grid">
      <div class="card">
        <div class="card-value">{summary['total_calls']}</div>
        <div class="card-label">Total Calls</div>
      </div>
      <div class="card error">
        <div class="card-value">{summary['total_429_errors']}</div>
        <div class="card-label">429 Errors</div>
      </div>
      <div class="card">
        <div class="card-value">{summary['total_errors']}</div>
        <div class="card-label">Other Errors</div>
      </div>
      <div class="card">
        <div class="card-value">{last_refresh}</div>
        <div class="card-label">Last Refresh</div>
      </div>
    </div>

    <div class="section">
      <div class="section-title">Calls by API</div>
      <table>
        <tr><th>API</th><th>Calls</th></tr>
        {''.join(api_rows)}
      </table>
    </div>

    <div class="section">
      <div class="section-title">Recent Calls (newest first, max 50)</div>
      <table>
        <tr><th>Time</th><th>API</th><th>Repo</th><th>Source</th><th>Duration</th><th>Status</th></tr>
        {''.join(recent_rows)}
      </table>
    </div>
  </div>
</body>
</html>
"""

    def clear(self) -> None:
        """Clear all records and statistics."""
        with self._lock:
            self._records.clear()
            self._stats = ApiStats()


# Global monitor instance
_monitor = ApiMonitor()


def get_monitor() -> ApiMonitor:
    """Get the global API monitor instance."""
    return _monitor


@contextlib.contextmanager
def track_api_call(api_name: str, source: str, repo_id: str | None = None):
    """Context manager to track an API call.

    Usage:
        with track_api_call("list_models", "home_refresh") as tracker:
            result = api.list_models(...)
            tracker["status"] = "ok"

    Or simply (status will be "ok" if no exception):
        with track_api_call("list_repo_tree", "propfind", repo_id="user/repo"):
            result = api.list_repo_tree(...)
    """
    tracker = {"status": "ok", "error_msg": None}
    start_time = time.time()

    try:
        yield tracker
    except Exception as e:
        # Check for 429 error
        error_str = str(e).lower()
        if "429" in error_str or "too many requests" in error_str or "rate limit" in error_str:
            tracker["status"] = "429"
        else:
            tracker["status"] = "error"
        tracker["error_msg"] = str(e)[:200]  # Truncate long error messages
        raise
    finally:
        duration_ms = (time.time() - start_time) * 1000
        _monitor.record_call(
            api_name=api_name,
            source=source,
            duration_ms=duration_ms,
            status=tracker["status"],
            repo_id=repo_id,
            error_msg=tracker["error_msg"],
        )
