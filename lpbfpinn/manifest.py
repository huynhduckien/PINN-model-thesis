"""Manifest JSON cho mỗi lượt chạy: cấu hình, phiên bản thư viện, commit git, thời điểm."""
from __future__ import annotations

import datetime as _dt
import json
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _git_head() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        import torch
        versions["torch"] = torch.__version__
    except ImportError:
        pass
    try:
        import deepxde
        versions["deepxde"] = deepxde.__version__
    except ImportError:
        pass
    return versions


def write_manifest(path: str | Path, run_name: str, config: Any, extra: dict[str, Any] | None = None) -> Path:
    payload = {
        "run": run_name,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "git_head": _git_head(),
        "versions": _library_versions(),
        "config": asdict(config) if is_dataclass(config) else config,
        "extra": extra or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
