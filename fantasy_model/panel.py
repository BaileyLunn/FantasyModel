"""Build the player-game panel (weekly stats + schedule + injuries).

Shared by ``scripts/fetch_data.py`` (history) and ``fantasy_model.project`` (forward rows).
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from fantasy_model.scoring import ensure_fantasy_points
from fantasy_model.teams import normalize_team_columns

SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]

SCHED_COLS = (
    "game_id", "season", "week", "gameday", "home_team", "away_team",
    "roof", "temp", "wind", "spread_line", "total_line",
    "home_moneyline", "away_moneyline",
)


def team_schedule(schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team) with game context."""
    cols = [c for c in SCHED_COLS if c in schedules.columns]
    sched = normalize_team_columns(schedules[cols].drop_duplicates())
    home = sched.copy()
    away = sched.copy()
    home["team"] = home["home_team"]
    away["team"] = away["away_team"]
    return pd.concat([home, away], ignore_index=True)


def prepare_injuries(inj: pd.DataFrame, id_col: str = "player_id") -> pd.DataFrame:
    """Collapse nflverse injury reports to one status/type per player-week."""
    id_right = "gsis_id" if "gsis_id" in inj.columns else ("player_id" if "player_id" in inj.columns else None)
    if id_right is None:
        return pd.DataFrame(columns=[id_col, "season", "week", "injury_status", "injury_type"])
    cols = [id_right]
    for c in (
        "season", "week", "report_status", "report_primary_injury",
        "practice_status", "practice_primary_injury",
        "injury_status", "injury_type",
    ):
        if c in inj.columns and c not in cols:
            cols.append(c)
    inj2 = inj[cols].copy()
    if id_right != id_col:
        inj2 = inj2.rename(columns={id_right: id_col})
    for c in ("season", "week"):
        if c in inj2.columns:
            inj2[c] = pd.to_numeric(inj2[c], errors="coerce")
    # Prefer official report_status; fall back to practice designation
    if "report_status" in inj2.columns:
        inj2["injury_status"] = inj2["report_status"]
    if "injury_status" in inj2.columns and "practice_status" in inj2.columns:
        inj2["injury_status"] = inj2["injury_status"].fillna(inj2["practice_status"])
    if "report_primary_injury" in inj2.columns:
        inj2["injury_type"] = inj2["report_primary_injury"]
    if "injury_type" in inj2.columns and "practice_primary_injury" in inj2.columns:
        inj2["injury_type"] = inj2["injury_type"].fillna(inj2["practice_primary_injury"])
    keep = [c for c in (id_col, "season", "week", "injury_status", "injury_type") if c in inj2.columns]
    return inj2[keep].drop_duplicates([c for c in (id_col, "season", "week") if c in keep])


def build_panel(
    weekly: pd.DataFrame,
    schedules: pd.DataFrame | None,
    injuries: pd.DataFrame | None,
    scoring: Mapping[str, float],
    raw_dir: str | None = None,
    add_zero_stat_rows: bool = False,
) -> pd.DataFrame:
    """Join weekly player rows to schedule + injuries; REG season skill positions only.

    Team codes on both sides are normalized to the current franchise code (OAK->LV, SD->LAC,
    STL->LA) so relocated-team seasons join their schedule. When ``raw_dir`` is given, depth-chart
    rank and prior snap-share context (``fantasy_model.features.usage``) are attached, and with
    ``add_zero_stat_rows`` snap-count appearances without any offensive stat are added as 0-point
    player-games (flag ``zero_stat_appearance``).
    """
    weekly = weekly.copy()
    if add_zero_stat_rows and raw_dir is not None:
        from fantasy_model.features.usage import rostered_no_snap_rows, zero_stat_appearances

        zero = pd.concat([zero_stat_appearances(weekly, raw_dir), rostered_no_snap_rows(weekly, raw_dir)],
                         ignore_index=True, sort=False)
        weekly["zero_stat_appearance"] = 0
        if not zero.empty:
            if "team" in weekly.columns and "recent_team" not in weekly.columns:
                zero = zero.rename(columns={"recent_team": "team"})
            weekly = pd.concat([weekly, zero], ignore_index=True, sort=False)
    if "recent_team" in weekly.columns and "team" not in weekly.columns:
        weekly = weekly.rename(columns={"recent_team": "team"})
    weekly = normalize_team_columns(weekly)
    if "player_display_name" in weekly.columns and "player_name" not in weekly.columns:
        weekly["player_name"] = weekly["player_display_name"]
    for c in ("season", "week"):
        if c in weekly.columns:
            weekly[c] = pd.to_numeric(weekly[c], errors="coerce")

    if schedules is not None and not schedules.empty:
        team_sched = team_schedule(schedules)
        keys = [c for c in ("season", "week", "team") if c in weekly.columns and c in team_sched.columns]
        panel = weekly.merge(team_sched, on=keys, how="left", suffixes=("", "_sched"))
        if "zero_row_type" in panel.columns:
            # rostered-no-snap rows for a team on its bye week are not games
            bye = panel["zero_row_type"].eq("rostered_no_snap") & panel["game_id"].isna()
            panel = panel.loc[~bye].copy()
        if {"home_team", "away_team", "team"} <= set(panel.columns):
            derived = panel["home_team"].where(panel["team"] != panel["home_team"], panel["away_team"])
            if "opponent_team" in panel.columns:
                # legacy player_stats sets opponent_team == team on special-teams-TD-only rows
                bad = panel["opponent_team"].isna() | (panel["opponent_team"] == panel["team"])
                panel["opponent_team"] = panel["opponent_team"].where(~bad, derived)
            else:
                panel["opponent_team"] = derived
    else:
        panel = weekly

    if injuries is not None and not injuries.empty and "player_id" in panel.columns:
        inj2 = prepare_injuries(injuries, "player_id")
        drop = [c for c in ("injury_status", "injury_type") if c in panel.columns]
        panel = panel.drop(columns=drop)
        merge_keys = [c for c in ("player_id", "season", "week") if c in panel.columns and c in inj2.columns]
        panel = panel.merge(inj2, on=merge_keys, how="left")

    panel = ensure_fantasy_points(panel, scoring)
    if "position" in panel.columns:
        panel = panel[panel["position"].astype(str).str.upper().isin(SKILL_POSITIONS)].copy()
    if "season_type" in panel.columns:
        panel = panel[panel["season_type"].astype(str).str.upper().eq("REG")].copy()
    panel = panel.reset_index(drop=True)
    if raw_dir is not None:
        from fantasy_model.features.usage import add_usage_context

        panel = add_usage_context(panel, raw_dir, schedules)
    return panel
