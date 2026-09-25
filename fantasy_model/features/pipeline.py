"""Assemble feature matrix without label leakage."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fantasy_model.features.baselines import add_baseline_features
from fantasy_model.features.injury import add_injury_features
from fantasy_model.features.teammates import add_teammate_features
from fantasy_model.features.timezone import add_timezone_features
from fantasy_model.features.travel import add_travel_features
from fantasy_model.features.weather import add_weather_features
from fantasy_model.scoring import ensure_fantasy_points

FEATURE_COLUMNS = [
    "injury_severity",
    "injury_analog_fp",
    "injury_analog_delta",
    "is_indoor_game",
    "temp_filled",
    "wind_filled",
    "temp_missing",
    "wind_missing",
    "is_home",
    "travel_miles",
    "tz_vs_home",
    "tz_vs_prev_game",
    "team_qb_fp_roll",
    "is_backup_qb_game",
    "team_target_share",
    "team_rush_share",
    "rest_days",
    "fp_roll",
    "fp_season_avg",
    "opp_pos_fp_allowed",
    "implied_team_total",
    "spread_line",
    "total_line",
    "targets_roll",
    "carries_roll",
    "rec_roll",
    # depth chart / snap usage (fantasy_model.features.usage; attached at panel build)
    "depth_rank_filled",
    "depth_listed",
    "snap_pct_last",
    "snap_pct_roll3",
    "snap_pct_season",
    "snap_games_season",
    "has_snap_history",
]

UNLISTED_DEPTH_RANK = 9  # players absent from the depth chart are treated as deep reserves


def add_usage_features(out: pd.DataFrame) -> pd.DataFrame:
    """Model-ready depth/snap columns. Explicit fills (not train-median imputation): a player missing
    from the depth chart is a deep reserve, and no prior snaps means 0 share, not the median."""
    out = out.copy()
    dr = pd.to_numeric(out["depth_rank"], errors="coerce") if "depth_rank" in out.columns else pd.Series(float("nan"), index=out.index)
    out["depth_listed"] = dr.notna().astype(int)
    out["depth_rank_filled"] = dr.clip(upper=UNLISTED_DEPTH_RANK).fillna(UNLISTED_DEPTH_RANK)
    last = pd.to_numeric(out.get("snap_pct_last"), errors="coerce") if "snap_pct_last" in out.columns else pd.Series(float("nan"), index=out.index)
    out["has_snap_history"] = last.notna().astype(int)
    for c in ("snap_pct_last", "snap_pct_roll3", "snap_pct_season", "snap_games_season"):
        vals = pd.to_numeric(out[c], errors="coerce") if c in out.columns else pd.Series(float("nan"), index=out.index)
        out[c] = vals.fillna(0.0)
    return out


def build_feature_matrix(
    df: pd.DataFrame,
    scoring: dict[str, float],
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.Series | None, list[str]]:
    """Return X, y (if fantasy_points computable), and feature name list.

    All rolling / expanding features use shift(1) internally to avoid leakage.
    Row order / index are restored to match the input ``df``.
    """
    cfg = cfg or {}
    feat_cfg = cfg.get("features", {})
    model_cfg = cfg.get("model", {})
    rolling = int(model_cfg.get("rolling_games", 3))

    original_index = df.index
    out = df.copy().reset_index(drop=True)
    out["_row_id"] = range(len(out))

    out = ensure_fantasy_points(out, scoring)

    if feat_cfg.get("weather_enabled", True):
        out = add_weather_features(out)
    if feat_cfg.get("travel_enabled", True):
        out = add_travel_features(out)
    if feat_cfg.get("timezone_enabled", True):
        out = add_timezone_features(out)
    if feat_cfg.get("injury_enabled", True):
        out = add_injury_features(out)
    if feat_cfg.get("teammates_enabled", True):
        out = add_teammate_features(out, rolling_games=rolling)
    if feat_cfg.get("baselines_enabled", True):
        out = add_baseline_features(out, rolling_games=rolling)
    if feat_cfg.get("usage_enabled", True):
        out = add_usage_features(out)

    # Restore input order after sorts/merges inside feature helpers
    if "_row_id" not in out.columns:
        raise RuntimeError("feature pipeline dropped _row_id; check merges")
    if out["_row_id"].duplicated().any():
        # Merges should be many-to-one; if duplicates appear, keep first
        out = out.drop_duplicates("_row_id", keep="first")
    out = out.sort_values("_row_id").reset_index(drop=True)
    if len(out) != len(df):
        raise RuntimeError(f"feature pipeline changed row count {len(df)} -> {len(out)}")

    if "position" in out.columns:
        # Stable codes independent of row order
        pos = out["position"].astype(str).str.upper()
        mapping = {p: i for i, p in enumerate(["QB", "RB", "TE", "WR"])}
        out["position_code"] = pos.map(mapping).fillna(-1).astype(int)
    else:
        out["position_code"] = 0

    feature_cols = [c for c in FEATURE_COLUMNS if c in out.columns]
    feature_cols.append("position_code")
    seen: set[str] = set()
    feature_cols = [c for c in feature_cols if not (c in seen or seen.add(c))]

    X = out[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = out["fantasy_points"] if "fantasy_points" in out.columns else None
    meta_cols = [
        c
        for c in (
            "player_id",
            "player_name",
            "player_display_name",
            "season",
            "week",
            "team",
            "position",
        )
        if c in out.columns
    ]
    X_meta = out[meta_cols].copy() if meta_cols else pd.DataFrame(index=out.index)
    X = pd.concat([X_meta, X], axis=1)
    X.index = original_index
    if y is not None:
        y.index = original_index
    return X, y, feature_cols
