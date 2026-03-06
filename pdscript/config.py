from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "transcription" / "config" / "podcast.yaml"


def load_config(config_path: str | Path | None = None, *, required: bool = True) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve() if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Missing config file: {path}. "
                "Create it from transcription/config/podcast.template.yaml."
            )
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping/object: {path}")
    return raw


def get_cfg(cfg: dict[str, Any], dotted: str, default: Any = "") -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def choose_value(*values: Any, default: Any = "") -> Any:
    for v in values:
        if v is None:
            continue
        if isinstance(v, str):
            if v.strip():
                return v.strip()
            continue
        return v
    return default

