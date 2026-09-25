"""Baselines: opponent defense vs position, rest days, rolling usage, season avgs, Vegas."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fantasy_model.features.history import prior_weeks_mean


def implied_team_total(team, home_team, spread_line, total_line) -> pd.Series:
    """Vegas implied points for ``team``: total/2 + (team's own expected margin)/2.

    nflverse ``spread_line`` = expected home margin (home favored > 0), so the team's margin is
    ``+spread`` at home and ``-spread`` on the road. Falls back to total/2 when home is unknown.
    """
    total = pd.to_numeric(pd.Series(total_line), errors="coerce").reset_index(drop=True)
    spread = pd.to_numeric(pd.Series(spread_line), errors="coerce").reset_index(drop=True)
    idx = total_line.index if isinstance(total_line, pd.Series) else None
    if home_team is None:
        res = total / 2.0
    else:
        t = pd.Series(team).astype(str).str.upper().reset_index(drop=True)
        h = pd.Series(home_team).astype(str).str.upper().reset_index(drop=True)
        margin = np.where(t == h, spread, -spread)
        res = total / 2.0 + pd.Series(margin, dtype=float) / 2.0
    if idx is not None:
        res.index = idx
    return res


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

    if opp_col and "position" in out.columns and "fantasy_points" in out.columns and "season" in out.columns:
        # FP allowed by this defense to this position in STRICTLY EARLIER weeks. (The old row-wise
        # shift(1) let a WR2 row see the WR1's same-game points vs the same defense.)
        allowed = prior_weeks_mean(out, [opp_col, "position"], "fantasy_points", min_count=5)
        out["opp_pos_fp_allowed"] = allowed.fillna(out.get("fp_season_avg", 0.0))
    else:
        out["opp_pos_fp_allowed"] = out.get("fp_season_avg", 0.0)

    # Vegas lines when freely available on schedule joins
    for c in ("spread_line", "total_line", "home_moneyline", "away_moneyline"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            out[c] = pd.NA
    # Implied team total. nflverse ``spread_line`` is the HOME team's expected margin
    # (positive = home favored; it correlates +0.44 with ``result`` = home - away score).
    # home implied = total/2 + spread/2, away implied = total/2 - spread/2.
    if team_col in out.columns:
        out["implied_team_total"] = implied_team_total(
            out[team_col], out.get("home_team", out.get("home")), out["spread_line"], out["total_line"]
        )
    else:
        out["implied_team_total"] = pd.NA

    return out
