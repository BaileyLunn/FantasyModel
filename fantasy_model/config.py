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


SCORING_PROFILES = ("half_ppr", "ppr", "standard")
# short tags used in artifact / report file names
PROFILE_TAGS = {"half_ppr": "half", "ppr": "ppr", "standard": "std"}
PROFILE_ALIASES = {
    "half": "half_ppr", "half-ppr": "half_ppr", "half_ppr": "half_ppr", "0.5": "half_ppr",
    "ppr": "ppr", "full": "ppr", "full_ppr": "ppr", "full-ppr": "ppr", "1": "ppr",
    "standard": "standard", "std": "standard", "non-ppr": "standard", "0": "standard",
}


def resolve_profile_name(name: str) -> str:
    key = str(name).strip().lower()
    if key not in PROFILE_ALIASES:
        raise ValueError(f"unknown scoring profile {name!r}; choose from {SCORING_PROFILES}")
    return PROFILE_ALIASES[key]


def scoring_profile(cfg: dict[str, Any]) -> str:
    """Active scoring profile name (``scoring.profile`` in config, overridable via --scoring)."""
    sc = cfg.get("scoring", {}) or {}
    return resolve_profile_name(sc.get("profile", "half_ppr")) if isinstance(sc, dict) else "half_ppr"


def set_scoring_profile(cfg: dict[str, Any], profile: str | None) -> dict[str, Any]:
    """Return a copy of ``cfg`` with the active scoring profile switched (no-op for None)."""
    if not profile:
        return cfg
    out = dict(cfg)
    out["scoring"] = {**(cfg.get("scoring") or {}), "profile": resolve_profile_name(profile)}
    return out


def scoring_dict(cfg: dict[str, Any]) -> dict[str, float]:
    """Scoring map for the active profile.

    Config layout::

        scoring:
          profile: half_ppr          # half_ppr | ppr | standard
          base: {pass_yd: 0.04, ...} # shared rules
          profiles: {half_ppr: {rec: 0.5}, ppr: {rec: 1.0}, standard: {rec: 0.0}}

    A flat legacy mapping (``scoring: {pass_yd: ..., rec: 0.5}``) is still accepted.
    """
    sc = dict(cfg.get("scoring", {}) or {})
    if "base" not in sc and "profiles" not in sc:
        return {k: float(v) for k, v in sc.items() if k != "profile"}
    prof = scoring_profile(cfg)
    out = {k: float(v) for k, v in (sc.get("base") or {}).items()}
    out.update({k: float(v) for k, v in ((sc.get("profiles") or {}).get(prof) or {}).items()})
    return out


def profile_tag(cfg: dict[str, Any]) -> str:
    return PROFILE_TAGS[scoring_profile(cfg)]


def model_path(cfg: dict[str, Any], validation: bool = False) -> Path:
    """Default artifact path for the active scoring profile, e.g. models/fantasy_hgb_ppr.joblib."""
    d = Path(cfg.get("paths", {}).get("models_dir", project_root() / "models"))
    return d / f"fantasy_hgb_{profile_tag(cfg)}{'_val' if validation else ''}.joblib"
