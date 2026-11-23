"""Server settings, from the environment (``KARNA_*``), overridable by ``karna serve`` flags."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_data_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "karna" / "server"


@dataclass
class Settings:
    #: jobs live here as plain directories, so they survive restarts
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("KARNA_DATA_DIR") or _default_data_dir()))
    host: str = field(default_factory=lambda: os.environ.get("KARNA_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.environ.get("KARNA_PORT", "8000")))
    #: loaded in the background at start; others load on first use
    model: str = field(default_factory=lambda: os.environ.get("KARNA_MODEL", "karna-base"))
    device: str | None = field(default_factory=lambda: os.environ.get("KARNA_DEVICE") or None)
    #: when set, /v1/* (except /v1/info) needs "Authorization: Bearer <key>" or "X-API-Key: <key>"
    api_key: str | None = field(default_factory=lambda: os.environ.get("KARNA_API_KEY") or None)
    #: longer audio is refused: each file is decoded in one pass, and training used 1-25 s clips
    max_seconds: float = field(default_factory=lambda: float(os.environ.get("KARNA_MAX_SECONDS", "30")))
    max_upload_mb: float = field(default_factory=lambda: float(os.environ.get("KARNA_MAX_UPLOAD_MB", "50")))
    #: finished jobs kept on disk (oldest deleted first)
    keep_jobs: int = field(default_factory=lambda: int(os.environ.get("KARNA_KEEP_JOBS", "200")))
