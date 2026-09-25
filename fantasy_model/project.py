"""Forward projections for an upcoming (or in-progress) week.

Rows are built from the nflverse weekly roster (status ``ACT``, QB/RB/WR/TE) for teams on the
schedule that week, joined to schedule (Vegas lines, roof, weather if published) and the
week's injury report. They are appended to the played-games history with
``is_projection=1`` (label kept missing), features are built with the same leakage-safe
pipeline, and the saved model predicts. Games of that week that were already played are
still predicted from pre-game features; their actual points are attached for comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from fantasy_model import sources
from fantasy_model.config import load_config, scoring_dict
from fantasy_model.data import load_player_games
from fantasy_model.features.pipeline import build_feature_matrix
from fantasy_model.panel import SKILL_POSITIONS, build_panel
from fantasy_model.scoring import fantasy_points


def _roster(cfg: dict[str, Any], season: int, refresh: bool) -> pd.DataFrame:
    path = Path(cfg["paths"]["raw_dir"]) / f"roster_weekly_{season}.parquet"
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        sources.load_weekly_roster(season).to_parquet(path, index=False)
    return pd.read_parquet(path)


def forward_rows(roster: pd.DataFrame, schedules: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    games = schedules[(schedules["season"] == season) & (schedules["week"] == week)]
    if games.empty:
        raise ValueError(f"no schedule rows for {season} week {week}")
    opp = {}
    for _, g in games.iterrows():
        opp[g["home_team"]] = g["away_team"]
        opp[g["away_team"]] = g["home_team"]
    r = roster[
        (roster["season"] == season)
        & (roster["week"] == week)
        & (roster["status"].astype(str).str.upper() == "ACT")
        & (roster["position"].isin(SKILL_POSITIONS))
        & (roster["team"].isin(opp.keys()))
        & roster["gsis_id"].notna()
    ].drop_duplicates("gsis_id")
    return pd.DataFrame(
        {
            "player_id": r["gsis_id"].to_numpy(),
            "player_name": r["full_name"].to_numpy(),
            "player_display_name": r["full_name"].to_numpy(),
            "position": r["position"].to_numpy(),
            "position_group": r["position"].to_numpy(),
            "recent_team": r["team"].to_numpy(),
            "opponent_team": r["team"].map(opp).to_numpy(),
            "season": season,
            "week": week,
            "season_type": "REG",
            "is_projection": 1,
        }
    )


def project_week(
    season: int,
    week: int,
    cfg: dict[str, Any] | None = None,
    config_path: str | None = None,
    model_path: str | None = None,
    refresh_roster: bool = True,
    out_path: str | None = None,
) -> pd.DataFrame:
    cfg = cfg or load_config(config_path)
    scoring = scoring_dict(cfg)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    path = Path(model_path) if model_path else Path(cfg["paths"]["models_dir"]) / "fantasy_hgb.joblib"
    artifact = joblib.load(path)

    hist, _ = load_player_games(cfg)
    hist = hist[(hist["season"] < season) | ((hist["season"] == season) & (hist["week"] < week))].copy()
    hist["is_projection"] = 0

    schedules = pd.read_csv(raw_dir / "schedules.csv", low_memory=False)
    inj_path = raw_dir / "injuries.csv"
    injuries = pd.read_csv(inj_path, low_memory=False) if inj_path.exists() else None
    fwd = forward_rows(_roster(cfg, season, refresh_roster), schedules, season, week)
    fwd_panel = build_panel(fwd, schedules, injuries, scoring)

    df = pd.concat([hist, fwd_panel], ignore_index=True, sort=False)
    X_all, _, _ = build_feature_matrix(df, scoring, cfg)
    proj_mask = df["is_projection"].fillna(0).astype(int).eq(1).to_numpy()
    X = X_all.loc[proj_mask, artifact["feature_cols"]].to_numpy(dtype=float)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(artifact["impute_medians"], inds[1])
    preds = artifact["model"].predict(X)

    out = df.loc[proj_mask].copy()
    out["predicted_fp"] = np.round(preds, 2)
    prior = hist.groupby("player_id").size()
    out["n_prior_games"] = out["player_id"].map(prior).fillna(0).astype(int)
    prior_season = hist[hist["season"] == season].groupby("player_id").size()
    out["n_prior_games_this_season"] = out["player_id"].map(prior_season).fillna(0).astype(int)
    feats = X_all.loc[proj_mask]
    out["is_home"] = feats["is_home"].to_numpy() if "is_home" in feats else np.nan
    out["implied_team_total"] = feats["implied_team_total"].to_numpy() if "implied_team_total" in feats else np.nan

    # Attach actual half-PPR points for games of this week already played (if published)
    game_played = pd.Series(False, index=out.index)
    out["actual_fp"] = np.nan
    wk_path = raw_dir / "weekly.csv"
    if wk_path.exists():
        wk = pd.read_csv(wk_path, low_memory=False)
        wk = wk[(wk["season"] == season) & (wk["week"] == week)].copy()
        if not wk.empty:
            wk = wk.assign(actual_fp=fantasy_points(wk, scoring).round(2))
            played_teams = set(wk["recent_team"].dropna())
            game_played = out["team"].isin(played_teams)
            act = wk.drop_duplicates("player_id").set_index("player_id")["actual_fp"]
            out["actual_fp"] = out["player_id"].map(act)
            # Active but no offensive stats in a completed game -> 0 points
            out.loc[game_played & out["actual_fp"].isna(), "actual_fp"] = 0.0
    out["game_status"] = np.where(game_played, "played", "upcoming")

    cols = [
        "season", "week", "game_id", "gameday", "player_id", "player_name", "position", "team",
        "opponent_team", "is_home", "spread_line", "total_line", "implied_team_total",
        "injury_status", "injury_type", "n_prior_games", "n_prior_games_this_season",
        "predicted_fp", "game_status", "actual_fp",
    ]
    result = out[[c for c in cols if c in out.columns]].sort_values("predicted_fp", ascending=False)
    result = result.reset_index(drop=True)
    if out_path is None:
        out_path = str(Path(cfg["paths"]["reports_dir"]) / f"projections_{season}_w{week}.csv")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    result.attrs["out_path"] = out_path
    return result
