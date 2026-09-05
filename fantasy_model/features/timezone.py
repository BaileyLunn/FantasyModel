"""Timezone shift features vs last game / home TZ."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

# Primary IANA-ish offset hours from UTC for team home cities (standard time approximation).
# We use fixed US offsets (ignore DST) for a stable delta feature; DST noise is secondary.
TEAM_TZ_OFFSET_HOURS: dict[str, float] = {
    "ARI": -7.0,
    "ATL": -5.0,
    "BAL": -5.0,
    "BUF": -5.0,
    "CAR": -5.0,
    "CHI": -6.0,
    "CIN": -5.0,
    "CLE": -5.0,
    "DAL": -6.0,
    "DEN": -7.0,
    "DET": -5.0,
    "GB": -6.0,
    "HOU": -6.0,
    "IND": -5.0,
    "JAX": -5.0,
    "KC": -6.0,
    "LA": -8.0,
    "LAR": -8.0,
    "LAC": -8.0,
    "LV": -8.0,
    "MIA": -5.0,
    "MIN": -6.0,
    "NE": -5.0,
    "NO": -6.0,
    "NYG": -5.0,
    "NYJ": -5.0,
    "PHI": -5.0,
    "PIT": -5.0,
    "SEA": -8.0,
    "SF": -8.0,
    "TB": -5.0,
    "TEN": -6.0,
    "WAS": -5.0,
    "WSH": -5.0,
}


def _norm_team(team: object) -> str:
    if team is None:
        return ""
    try:
        if team != team:
            return ""
    except Exception:
        pass
    s = str(team).strip()
    if s.lower() in {"nan", "none", ""}:
        return ""
    return s.upper()


def team_tz_offset(team: str, table: Mapping[str, float] | None = None) -> float:
    table = table or TEAM_TZ_OFFSET_HOURS
    return float(table.get(_norm_team(team), -5.0))


def timezone_delta_hours(from_team_or_venue: str, to_team_or_venue: str) -> float:
    """Hours of clock shift traveling from venue A home TZ to venue B home TZ.

    Positive means clocks are later at destination (eastward travel).
    Example: SEA (-8) -> NYG (-5) => +3.
    """
    return team_tz_offset(to_team_or_venue) - team_tz_offset(from_team_or_venue)


def add_timezone_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add tz_vs_home and tz_vs_prev_game.

    Expects: team/recent_team, home_team/home, and optionally prev_home_team.
    If prev_home_team missing, compute from sorted player history when player_id present.
    """
    out = df.copy()
    team_col = "team" if "team" in out.columns else "recent_team"
    home_col = "home_team" if "home_team" in out.columns else "home"
    if team_col not in out.columns or home_col not in out.columns:
        out["tz_vs_home"] = 0.0
        out["tz_vs_prev_game"] = 0.0
        return out

    teams = out[team_col].astype(str).str.upper()
    homes = out[home_col].astype(str).str.upper()
    out["tz_vs_home"] = [
        timezone_delta_hours(t, h) for t, h in zip(teams.tolist(), homes.tolist(), strict=False)
    ]

    if "prev_home_team" not in out.columns and "player_id" in out.columns:
        sort_cols = [c for c in ("season", "week", "game_id", "_row_id") if c in out.columns]
        tmp = out.sort_values(sort_cols) if sort_cols else out.copy()
        prev = tmp.groupby("player_id")[home_col].shift(1)
        out.loc[tmp.index, "prev_home_team"] = prev

    if "prev_home_team" in out.columns:
        prev = out["prev_home_team"].astype(str).str.upper().replace({"NAN": "", "NONE": ""})
        out["tz_vs_prev_game"] = [
            timezone_delta_hours(p, h) if p else 0.0
            for p, h in zip(prev.tolist(), homes.tolist(), strict=False)
        ]
    else:
        out["tz_vs_prev_game"] = 0.0
    return out
