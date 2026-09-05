"""Load processed or sample player-game panels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fantasy_model.config import project_root


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def load_player_games(cfg: dict[str, Any], prefer_sample: bool = False) -> tuple[pd.DataFrame, str]:
    """Load the best available panel.

    Returns (dataframe, source_label).
    """
    paths = cfg.get("paths", {})
    root = project_root()
    processed = Path(paths.get("processed_dir", root / "data" / "processed"))
    sample = Path(paths.get("sample_dir", root / "data" / "sample"))

    candidates_real = [
        processed / "player_games.parquet",
        processed / "player_games.csv",
    ]
    candidates_sample = [
        sample / "player_games.csv",
        sample / "player_games.parquet",
    ]

    if prefer_sample:
        ordered = candidates_sample + candidates_real
    else:
        ordered = candidates_real + candidates_sample

    path = _first_existing(ordered)
    if path is None:
        raise FileNotFoundError(
            "No player_games data found. Run scripts/fetch_data.py or use data/sample/."
        )
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    label = "sample" if "sample" in path.parts else "processed"
    return df, f"{label}:{path}"
