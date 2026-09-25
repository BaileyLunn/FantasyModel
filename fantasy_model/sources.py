"""nflverse raw-data sources with per-season fallbacks.

``nfl_data_py.import_weekly_data`` reads the legacy ``player_stats/player_stats_{season}``
release, which nflverse stopped publishing after 2024 (2025+ returns HTTP 404). The
replacement release is ``stats_player/stats_player_week_{season}.parquet``; this module
loads it and harmonizes its columns to the legacy weekly schema used by the pipeline.
"""

from __future__ import annotations

import pandas as pd

NFLVERSE_RELEASES = "https://github.com/nflverse/nflverse-data/releases/download"
STATS_PLAYER_WEEK_URL = NFLVERSE_RELEASES + "/stats_player/stats_player_week_{season}.parquet"
INJURIES_URL = NFLVERSE_RELEASES + "/injuries/injuries_{season}.parquet"
WEEKLY_ROSTER_URL = NFLVERSE_RELEASES + "/weekly_rosters/roster_weekly_{season}.parquet"
SCHEDULES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"

# stats_player column -> legacy player_stats (nfl_data_py weekly) column
STATS_PLAYER_RENAMES = {
    "team": "recent_team",
    "passing_interceptions": "interceptions",
    "sacks_suffered": "sacks",
    "sack_yards_lost": "sack_yards",
}

# A player-game counts as an offensive appearance (legacy player_stats semantics) when
# any of these is non-zero. Pure special-teams / defensive snaps are dropped.
OFFENSIVE_INVOLVEMENT_COLS = [
    "attempts",
    "carries",
    "targets",
    "receptions",
    "sacks",
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "special_teams_tds",
    "rushing_fumbles_lost",
    "receiving_fumbles_lost",
    "sack_fumbles_lost",
]


def harmonize_stats_player(df: pd.DataFrame, keep_uninvolved: bool = False) -> pd.DataFrame:
    """Rename stats_player columns to the legacy weekly schema and drop non-offensive rows."""
    out = df.rename(columns={k: v for k, v in STATS_PLAYER_RENAMES.items() if k in df.columns})
    # stats_player carries game_id; the panel builder joins schedules on season/week/team
    if "game_id" in out.columns:
        out = out.rename(columns={"game_id": "stats_game_id"})
    if not keep_uninvolved:
        cols = [c for c in OFFENSIVE_INVOLVEMENT_COLS if c in out.columns]
        involved = out[cols].apply(pd.to_numeric, errors="coerce").fillna(0).ne(0).any(axis=1)
        out = out.loc[involved].copy()
    out["stats_source"] = "nflverse:stats_player_week"
    return out.reset_index(drop=True)


def load_weekly_season(season: int, prefer: str = "auto") -> tuple[pd.DataFrame, str]:
    """Load one season of weekly player stats in the legacy schema.

    prefer: ``auto`` (nfl_data_py then stats_player), ``nfl_data_py``, or ``stats_player``.
    Raises if no source has the season.
    """
    errors: list[str] = []
    if prefer in ("auto", "nfl_data_py"):
        try:
            import nfl_data_py as nfl

            df = nfl.import_weekly_data([season])
            df["stats_source"] = "nfl_data_py:player_stats"
            return df, "nfl_data_py:player_stats"
        except Exception as e:  # noqa: BLE001
            errors.append(f"nfl_data_py: {e}")
            if prefer == "nfl_data_py":
                raise
    try:
        raw = pd.read_parquet(STATS_PLAYER_WEEK_URL.format(season=season))
        return harmonize_stats_player(raw), "nflverse:stats_player_week"
    except Exception as e:  # noqa: BLE001
        errors.append(f"stats_player: {e}")
    raise RuntimeError(f"weekly {season} unavailable: {'; '.join(errors)}")


def load_injuries_season(season: int) -> pd.DataFrame:
    return pd.read_parquet(INJURIES_URL.format(season=season))


def load_weekly_roster(season: int) -> pd.DataFrame:
    return pd.read_parquet(WEEKLY_ROSTER_URL.format(season=season))


def load_schedules(seasons: list[int]) -> pd.DataFrame:
    games = pd.read_csv(SCHEDULES_URL)
    return games[games["season"].isin(seasons)].reset_index(drop=True)
