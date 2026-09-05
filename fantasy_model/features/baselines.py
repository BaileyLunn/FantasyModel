"""Baselines: opponent defense vs position, rest days, rolling usage, season avgs, Vegas."""

from __future__ import annotations

import pandas as pd


def add_baseline_features(df: pd.DataFrame, rolling_games: int = 3) -> pd.DataFrame:
    out = df.copy()
    team_col = "team" if "team" in out.columns else "recent_team"
    sort_cols = [c for c in ("season", "week", "_row_id") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols + (["player_id"] if "player_id" in out.columns else []))

    # Rest days from prior game date if available
    if "gameday" in out.columns and "player_id" in out.columns:
        out["gameday"] = pd.to_datetime(out["gameday"], errors="coerce")
        prev_day = out.groupby("player_id")["gameday"].shift(1)
        out["rest_days"] = (out["gameday"] - prev_day).dt.days.fillna(7).clip(lower=0, upper=21)
    elif "week" in out.columns and "player_id" in out.columns:
        prev_week = out.groupby(["player_id", "season"])["week"].shift(1) if "season" in out.columns else out.groupby("player_id")["week"].shift(1)
        out["rest_days"] = ((out["week"] - prev_week).fillna(1) * 7).clip(lower=0, upper=21)
    else:
        out["rest_days"] = 7.0

    # Rolling usage / FP (shifted — prior games only)
    if "player_id" in out.columns and "fantasy_points" in out.columns:
        g = out.groupby("player_id")["fantasy_points"]
        out["fp_roll"] = g.transform(lambda s: s.shift(1).rolling(rolling_games, min_periods=1).mean())
        out["fp_season_avg"] = out.groupby(["player_id", "season"])["fantasy_points"].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        ) if "season" in out.columns else out["fp_roll"]
        out["fp_roll"] = out["fp_roll"].fillna(0.0)
        out["fp_season_avg"] = out["fp_season_avg"].fillna(0.0)
    else:
        out["fp_roll"] = 0.0
        out["fp_season_avg"] = 0.0

    for usage_col, out_col in (
        ("targets", "targets_roll"),
        ("carries", "carries_roll"),
        ("rushing_attempts", "carries_roll"),
        ("receptions", "rec_roll"),
    ):
        if usage_col in out.columns and "player_id" in out.columns and out_col not in out.columns:
            out[out_col] = (
                out.groupby("player_id")[usage_col]
                .transform(lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(rolling_games, min_periods=1).mean())
                .fillna(0.0)
            )

    # Opponent defense vs position: mean FP allowed to that position in prior games
    opp_col = None
    for c in ("opponent_team", "opponent", "opp"):
        if c in out.columns:
            opp_col = c
            break
    if opp_col is None and team_col in out.columns:
        # Derive opponent from home/away
        home_col = "home_team" if "home_team" in out.columns else "home"
        away_col = "away_team" if "away_team" in out.columns else "away"
        if home_col in out.columns and away_col in out.columns:
            teams = out[team_col].astype(str).str.upper()
            homes = out[home_col].astype(str).str.upper()
            aways = out[away_col].astype(str).str.upper()
            out["opponent_team"] = [
                a if t == h else h for t, h, a in zip(teams, homes, aways, strict=False)
            ]
            opp_col = "opponent_team"

    if opp_col and "position" in out.columns and "fantasy_points" in out.columns:
        # For each game row, defense = opponent; points scored by offense players against them
        tmp = out.copy()
        # Points "allowed" = fantasy points by players facing this defense
        keys = [opp_col, "position"]
        tmp = tmp.sort_values(sort_cols) if sort_cols else tmp
        # Expanding mean of FP by players vs this defense at this position, shifted
        # Approximate: group by opponent+position across all rows chronologically
        allowed = (
            tmp.groupby(keys, dropna=False)["fantasy_points"]
            .transform(lambda s: s.shift(1).expanding(min_periods=5).mean())
        )
        out["opp_pos_fp_allowed"] = allowed.fillna(out.get("fp_season_avg", 0.0))
    else:
        out["opp_pos_fp_allowed"] = out.get("fp_season_avg", 0.0)

    # Vegas lines when freely available on schedule joins
    for c in ("spread_line", "total_line", "home_moneyline", "away_moneyline"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            out[c] = pd.NA
    # Implied team total heuristic
    if "spread_line" in out.columns and "total_line" in out.columns and team_col in out.columns:
        home_col = "home_team" if "home_team" in out.columns else "home"
        if home_col in out.columns:
            is_home = out[team_col].astype(str).str.upper() == out[home_col].astype(str).str.upper()
            # spread_line typically from home perspective in nflverse
            spread = pd.to_numeric(out["spread_line"], errors="coerce")
            total = pd.to_numeric(out["total_line"], errors="coerce")
            home_tt = (total - spread) / 2.0
            away_tt = (total + spread) / 2.0
            out["implied_team_total"] = home_tt.where(is_home, away_tt)
        else:
            out["implied_team_total"] = pd.to_numeric(out["total_line"], errors="coerce") / 2.0
    else:
        out["implied_team_total"] = pd.NA

    return out
