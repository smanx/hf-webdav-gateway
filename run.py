from pathlib import Path
import os
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _log_startup() -> None:
    print("[startup] hf-webdav-gateway entrypoint reached", flush=True)
    print(f"[startup] python={sys.version.split()[0]}", flush=True)
    print(f"[startup] root={ROOT}", flush=True)
    print(f"[startup] src_exists={SRC.exists()}", flush=True)
    print(f"[startup] cwd={Path.cwd()}", flush=True)
    print(f"[startup] port_env={os.getenv('PORT', '')}", flush=True)
    print(f"[startup] space_id={os.getenv('SPACE_ID', '')}", flush=True)
    print(f"[startup] hf_webdav_host={os.getenv('HF_WEBDAV_HOST', '')}", flush=True)
    print(f"[startup] hf_webdav_port={os.getenv('HF_WEBDAV_PORT', '')}", flush=True)
    print(
        f"[startup] repos_env_set={bool(os.getenv('HF_WEBDAV_REPOSITORIES', '').strip())}",
        flush=True,
    )


from hf_webdav_gateway.server import main


if __name__ == "__main__":
    _log_startup()
    raise SystemExit(main())
