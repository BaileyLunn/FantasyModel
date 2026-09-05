"""Load and validate project configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML or JSON config; resolve relative paths against project root."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.is_absolute():
        cfg_path = project_root() / cfg_path
    text = cfg_path.read_text(encoding="utf-8")
    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        cfg = yaml.safe_load(text)
    elif cfg_path.suffix.lower() == ".json":
        cfg = json.loads(text)
    else:
        raise ValueError(f"Unsupported config format: {cfg_path}")
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping")
    root = project_root()
    paths = cfg.setdefault("paths", {})
    for key in ("raw_dir", "processed_dir", "sample_dir", "models_dir", "reports_dir"):
        if key in paths and paths[key] and not Path(paths[key]).is_absolute():
            paths[key] = str(root / paths[key])
    return cfg


def scoring_dict(cfg: dict[str, Any]) -> dict[str, float]:
    return dict(cfg.get("scoring", {}))
