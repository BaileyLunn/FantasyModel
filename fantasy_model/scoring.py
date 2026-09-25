"""Fantasy scoring utilities (ESPN half-PPR by default)."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

# Column aliases commonly present in nflverse weekly / play-by-play aggregates
STAT_COLUMNS = {
    "pass_yd": ["passing_yards", "pass_yds", "pass_yards"],
    "pass_td": ["passing_tds", "pass_td", "pass_tds"],
    "pass_int": ["interceptions", "passing_interceptions", "pass_int", "ints"],
    "rush_yd": ["rushing_yards", "rush_yds", "rush_yards"],
    "rush_td": ["rushing_tds", "rush_td", "rush_tds"],
    "rec": ["receptions", "rec"],
    "rec_yd": ["receiving_yards", "rec_yds", "receiving_yds"],
    "rec_td": ["receiving_tds", "rec_td", "rec_tds"],
    "fumble_lost": ["fantasy_points_fumbles_lost", "fumbles_lost", "fum_lost"],
    "two_pt": ["two_point_conversions", "two_pt", "two_point"],
}

FUMBLE_LOST_PARTS = [
    "fumbles_lost",
    "rushing_fumbles_lost",
    "receiving_fumbles_lost",
    "sack_fumbles_lost",
]

TWO_PT_PARTS = [
    "two_point_conversions",
    "passing_2pt_conversions",
    "rushing_2pt_conversions",
    "receiving_2pt_conversions",
]


def _resolve_series(df: pd.DataFrame, aliases: list[str]) -> pd.Series:
    for name in aliases:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype=float)


def _sum_parts(df: pd.DataFrame, parts: list[str]) -> pd.Series:
    total = pd.Series(0.0, index=df.index, dtype=float)
    found = False
    for name in parts:
        if name in df.columns:
            total = total + pd.to_numeric(df[name], errors="coerce").fillna(0.0)
            found = True
    return total if found else total


def fantasy_points(df: pd.DataFrame, scoring: Mapping[str, float]) -> pd.Series:
    """Compute fantasy points for each row given a scoring map.

    Expected scoring keys: pass_yd, pass_td, pass_int, rush_yd, rush_td,
    rec, rec_yd, rec_td, fumble_lost, two_pt, and optional yardage bonuses.
    Special-teams TDs counted as 6 when ``special_teams_tds`` is present.
    """
    pts = pd.Series(0.0, index=df.index, dtype=float)
    for key, aliases in STAT_COLUMNS.items():
        if key in ("fumble_lost", "two_pt"):
            continue
        rate = float(scoring.get(key, 0.0))
        pts = pts + _resolve_series(df, aliases) * rate

    # Fumbles / 2PT: prefer aggregated parts (nflverse weekly)
    fum = _sum_parts(df, FUMBLE_LOST_PARTS)
    if fum.eq(0).all() and "fumbles_lost" not in df.columns:
        fum = _resolve_series(df, STAT_COLUMNS["fumble_lost"])
    pts = pts + fum * float(scoring.get("fumble_lost", 0.0))

    two = _sum_parts(df, TWO_PT_PARTS)
    pts = pts + two * float(scoring.get("two_pt", 0.0))

    if "special_teams_tds" in df.columns:
        pts = pts + pd.to_numeric(df["special_teams_tds"], errors="coerce").fillna(0.0) * 6.0

    # Optional bonuses
    pass_yds = _resolve_series(df, STAT_COLUMNS["pass_yd"])
    rush_yds = _resolve_series(df, STAT_COLUMNS["rush_yd"])
    rec_yds = _resolve_series(df, STAT_COLUMNS["rec_yd"])
    if float(scoring.get("pass_300_bonus", 0.0)):
        pts = pts + (pass_yds >= 300).astype(float) * float(scoring["pass_300_bonus"])
    if float(scoring.get("rush_100_bonus", 0.0)):
        pts = pts + (rush_yds >= 100).astype(float) * float(scoring["rush_100_bonus"])
    if float(scoring.get("rec_100_bonus", 0.0)):
        pts = pts + (rec_yds >= 100).astype(float) * float(scoring["rec_100_bonus"])
    return pts


def ensure_fantasy_points(df: pd.DataFrame, scoring: Mapping[str, float], col: str = "fantasy_points") -> pd.DataFrame:
    """Recompute fantasy_points from box-score stats using the configured scoring map."""
    out = df.copy()
    out[col] = fantasy_points(out, scoring)
    # Forward (not-yet-played) rows have no box score: keep the label missing so that
    # shifted/expanding features and training never treat them as 0-point games.
    if "is_projection" in out.columns:
        proj = pd.to_numeric(out["is_projection"], errors="coerce").fillna(0).astype(bool)
        out.loc[proj, col] = float("nan")
    return out
