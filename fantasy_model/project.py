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
from fantasy_model.config import load_config, model_path as default_model_path, profile_tag, scoring_dict
from fantasy_model.data import load_player_games
from fantasy_model.features.pipeline import build_feature_matrix
from fantasy_model.panel import SKILL_POSITIONS, build_panel
from fantasy_model.scoring import fantasy_points
from fantasy_model.teams import normalize_team, normalize_team_series
from fantasy_model.train import apply_interval


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
        h, a = normalize_team(g["home_team"]), normalize_team(g["away_team"])
        opp[h] = a
        opp[a] = h
    roster = roster.assign(team=normalize_team_series(roster["team"]))
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


ROLE_DEPTH_CUTOFF = {"QB": 1, "RB": 2, "WR": 3, "TE": 2}
RULED_OUT = {"out", "injured reserve", "ir"}


def likely_role(position: pd.Series, depth_rank: pd.Series, snap_roll3: pd.Series) -> pd.Series:
    """True when the depth chart or recent snap share says the player has a real offensive role."""
    cut = position.astype(str).str.upper().map(ROLE_DEPTH_CUTOFF).fillna(2)
    dr = pd.to_numeric(depth_rank, errors="coerce")
    return (dr <= cut) | (pd.to_numeric(snap_roll3, errors="coerce").fillna(0) >= 0.35)


def local_drivers(model, X: np.ndarray, reference: np.ndarray, feature_cols: list[str], top: int = 3) -> list[str]:
    """Per-row top features by single-feature substitution: effect = f(x) - f(x with feature j set to
    its training median). A cheap local attribution for tree models (NOT SHAP; ignores interactions)."""
    base = model.predict(X)
    effects = np.zeros_like(X, dtype=float)
    for j in range(X.shape[1]):
        Xj = X.copy()
        Xj[:, j] = reference[j]
        effects[:, j] = base - model.predict(Xj)
    out = []
    for i in range(X.shape[0]):
        order = np.argsort(-np.abs(effects[i]))[:top]
        out.append("; ".join(f"{feature_cols[j]}={X[i, j]:.3g} ({effects[i, j]:+.1f})" for j in order))
    return out


def projection_features(
    season: int,
    week: int,
    cfg: dict[str, Any],
    refresh_roster: bool = True,
    hist: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build forward rows for (season, week) and their leakage-safe features.

    Returns (hist_before_week, projection_rows, projection_feature_rows). ``hist`` defaults to the
    processed panel; only rows strictly before the projected week are used.
    """
    scoring = scoring_dict(cfg)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    if hist is None:
        hist, _ = load_player_games(cfg)
    hist = hist[(hist["season"] < season) | ((hist["season"] == season) & (hist["week"] < week))].copy()
    hist["is_projection"] = 0

    schedules = pd.read_csv(raw_dir / "schedules.csv", low_memory=False)
    inj_path = raw_dir / "injuries.csv"
    injuries = pd.read_csv(inj_path, low_memory=False) if inj_path.exists() else None
    fwd = forward_rows(_roster(cfg, season, refresh_roster), schedules, season, week)
    fwd_panel = build_panel(fwd, schedules, injuries, scoring, raw_dir=str(raw_dir),
                            snap_ewm_halflife=cfg.get("features", {}).get("ewm_halflife_games"))

    df = pd.concat([hist, fwd_panel], ignore_index=True, sort=False)
    X_all, _, _ = build_feature_matrix(df, scoring, cfg)
    proj_mask = df["is_projection"].fillna(0).astype(int).eq(1).to_numpy()
    return hist, df.loc[proj_mask].copy(), X_all.loc[proj_mask]


def ruled_out_mask(out: pd.DataFrame) -> pd.Series:
    status = out.get("injury_status", pd.Series("", index=out.index)).fillna("").astype(str).str.strip().str.lower()
    return status.isin(RULED_OUT)


def actual_points(out: pd.DataFrame, raw_dir: Path, season: int, week: int, scoring: dict[str, float]) -> tuple[pd.Series, pd.Series]:
    """(actual_fp, game_played) for projection rows from the raw weekly stats; active players with no
    stat line in a completed game get 0."""
    game_played = pd.Series(False, index=out.index)
    actual = pd.Series(np.nan, index=out.index)
    wk_path = Path(raw_dir) / "weekly.csv"
    if wk_path.exists():
        wk = pd.read_csv(wk_path, low_memory=False)
        wk = wk[(wk["season"] == season) & (wk["week"] == week)].copy()
        if not wk.empty:
            wk = wk.assign(actual_fp=fantasy_points(wk, scoring).round(2))
            played_teams = set(normalize_team_series(wk["recent_team"]).dropna())
            game_played = out["team"].isin(played_teams)
            act = wk.drop_duplicates("player_id").set_index("player_id")["actual_fp"]
            actual = out["player_id"].map(act)
            actual[game_played & actual.isna()] = 0.0
    return actual, game_played


def project_week(
    season: int,
    week: int,
    cfg: dict[str, Any] | None = None,
    config_path: str | None = None,
    model_path: str | None = None,
    refresh_roster: bool = True,
    out_path: str | None = None,
    drivers: bool = True,
) -> pd.DataFrame:
    cfg = cfg or load_config(config_path)
    scoring = scoring_dict(cfg)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    path = Path(model_path) if model_path else default_model_path(cfg)
    artifact = joblib.load(path)

    hist, out, feats = projection_features(season, week, cfg, refresh_roster=refresh_roster)
    feature_cols = artifact["feature_cols"]
    X = feats[feature_cols].to_numpy(dtype=float)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(artifact["impute_medians"], inds[1])
    preds = artifact["model"].predict(X)
    lo, hi = apply_interval(preds, artifact.get("interval"))

    out["model_projection"] = np.round(preds, 2)
    out["low_80"] = np.round(lo, 2)
    out["high_80"] = np.round(hi, 2)
    for c in ("is_home", "implied_team_total", "depth_rank_filled", "snap_pct_last", "snap_pct_roll3", "snap_pct_season",
              "snap_pct_ewm", "fp_roll", "fp_ewm"):
        out[c] = feats[c].to_numpy() if c in feats else np.nan
    out["depth_rank"] = pd.to_numeric(out.get("depth_rank"), errors="coerce")
    prior = hist.groupby("player_id").size()
    out["n_prior_games"] = out["player_id"].map(prior).fillna(0).astype(int)
    prior_season = hist[hist["season"] == season].groupby("player_id").size()
    out["n_prior_games_this_season"] = out["player_id"].map(prior_season).fillna(0).astype(int)
    out["likely_role"] = likely_role(out["position"], out["depth_rank"], out["snap_pct_roll3"])

    # The model never sees ruled-out players (they have no played-game rows), so it cannot learn
    # "Out". Rows are kept; projection is set to 0 when the nflverse report says Out/IR.
    ruled_out = ruled_out_mask(out)
    out["projection"] = out["model_projection"].where(~ruled_out, 0.0)
    out.loc[ruled_out, ["low_80", "high_80"]] = 0.0
    out["availability_note"] = np.where(ruled_out, "ruled out on nflverse injury report -> 0", "")
    if drivers and len(X):
        out["top_drivers"] = local_drivers(artifact["model"], X, artifact["impute_medians"], feature_cols)

    # Attach actual points for games of this week already played (if published)
    out["actual_fp"], game_played = actual_points(out, raw_dir, season, week, scoring)
    out["game_status"] = np.where(game_played, "played", "upcoming")
    out["scoring_profile"] = artifact.get("scoring_profile", profile_tag(cfg))

    cols = [
        "season", "week", "game_id", "gameday", "player_id", "player_name", "position", "team",
        "opponent_team", "is_home", "spread_line", "total_line", "implied_team_total",
        "depth_rank", "likely_role", "snap_pct_last", "snap_pct_roll3", "snap_pct_season", "snap_pct_ewm",
        "fp_roll", "fp_ewm", "injury_status", "injury_type", "n_prior_games", "n_prior_games_this_season",
        "projection", "low_80", "high_80", "model_projection", "availability_note",
        "game_status", "actual_fp", "scoring_profile", "top_drivers",
    ]
    result = out[[c for c in cols if c in out.columns]].sort_values("projection", ascending=False)
    result = result.reset_index(drop=True)
    if out_path is None:
        out_path = str(Path(cfg["paths"]["reports_dir"]) / f"projections_{season}_w{week}_{profile_tag(cfg)}.csv")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    result.attrs["out_path"] = out_path
    return result
