"""Team abbreviation normalization.

nflverse is inconsistent across releases for relocated franchises: ``games.csv`` (schedules)
and snap counts / depth charts keep the historical code (OAK, SD, STL), while the weekly
player-stats release uses the current code (LV, LAC, LA) for every season. Normalizing both
sides to the current franchise code makes 2016–2019 rows join their schedule.
"""

from __future__ import annotations

import pandas as pd

# historical / alternate code -> current nflverse franchise code
TEAM_ALIASES: dict[str, str] = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
    "LAR": "LA",
    "WSH": "WAS",
    "JAC": "JAX",
}

TEAM_COLUMNS = ("team", "recent_team", "opponent_team", "home_team", "away_team", "club_code")


def normalize_team(code: object) -> object:
    if code is None or (isinstance(code, float) and pd.isna(code)):
        return code
    s = str(code).strip().upper()
    return TEAM_ALIASES.get(s, s)


def normalize_team_series(s: pd.Series) -> pd.Series:
    return s.map(normalize_team)


def normalize_team_columns(df: pd.DataFrame, cols: tuple[str, ...] = TEAM_COLUMNS) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = normalize_team_series(out[c])
    return out
