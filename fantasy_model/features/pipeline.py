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
    # recent form: exponentially weighted means of prior games (features.recent_form_enabled)
    "fp_ewm",
    "targets_ewm",
    "carries_ewm",
    "rec_ewm",
    "snap_pct_ewm",
]

RECENT_FORM_COLUMNS = ["fp_ewm", "targets_ewm", "carries_ewm", "rec_ewm", "snap_pct_ewm"]
_EWM_SOURCES = {"fp_ewm": "fantasy_points", "targets_ewm": "targets", "carries_ewm": "carries", "rec_ewm": "receptions"}


def add_recent_form_features(out: pd.DataFrame, halflife_games: float = 5.0) -> pd.DataFrame:
    """EWMA of a player's PRIOR games (fantasy points, targets, carries, receptions).

    ``shift(1)`` inside each player's time-ordered history, then ``ewm(halflife, ignore_na=True)``:
    a row only sees games strictly before it (one row per player-week), across season boundaries.
    Zero-stat rows (0-point appearances / rostered-no-snap) count as 0 usage; unlabelled projection
    rows are skipped (NaN). Players with no prior games get 0, like ``fp_roll``.
    """
    out = out.copy()
    if "player_id" not in out.columns:
        for c in _EWM_SOURCES:
            out[c] = 0.0
        return out
    sort_cols = [c for c in ("season", "week", "_row_id") if c in out.columns]
    order = out.sort_values(sort_cols).index if sort_cols else out.index
    srt = out.loc[order]
    zero = pd.to_numeric(srt.get("zero_stat_appearance"), errors="coerce").fillna(0).eq(1) \
        if "zero_stat_appearance" in srt.columns else pd.Series(False, index=srt.index)
    pid = srt["player_id"]
    for col, src in _EWM_SOURCES.items():
        if src not in srt.columns:
            out[col] = 0.0
            continue
        v = pd.to_numeric(srt[src], errors="coerce")
        if src != "fantasy_points":
            v = v.mask(zero & v.isna(), 0.0)
        prev = v.groupby(pid, sort=False).shift(1)
        ew = prev.groupby(pid, sort=False).transform(lambda x: x.ewm(halflife=halflife_games, ignore_na=True).mean())
        out[col] = ew.reindex(out.index).fillna(0.0)
    return out

# Position-specific usage (features.position_specific_enabled): EWMA of PRIOR games of role stats that
# matter for one position more than others (QB rushing share, RB/WR/TE target + air-yards share, QB
# passing volume). Same shift(1) + ewm(halflife) scheme as the recent-form features.
POSITION_SPECIFIC_COLUMNS = [
    "rush_share_ewm",        # player carries / team carries (QB designed-run share, RB workload)
    "rush_yds_ewm",          # QB rushing production
    "target_share_ewm",      # player targets / team targets (RB/WR/TE)
    "air_yards_share_ewm",   # share of team air yards (WR/TE)
    "rec_air_yards_ewm",
    "pass_att_ewm",          # QB passing volume
    "pass_air_yards_ewm",
]


def add_position_specific_features(out: pd.DataFrame, halflife_games: float = 5.0) -> pd.DataFrame:
    """EWMA of a player's prior-game role stats. Team shares are computed within each team-game from the
    labelled rows (skill positions only), so a row never sees its own game. Labelled rows with no value
    (e.g. no targets) count as 0; unlabelled projection rows are skipped. No prior games -> 0."""
    out = out.copy()
    need = {"player_id", "season", "week"}
    if not need.issubset(out.columns):
        for c in POSITION_SPECIFIC_COLUMNS:
            out[c] = 0.0
        return out
    team_col = "team" if "team" in out.columns else "recent_team"
    fp = pd.to_numeric(out.get("fantasy_points"), errors="coerce") if "fantasy_points" in out.columns else pd.Series(0.0, index=out.index)
    labelled = fp.notna()

    def num(c):
        v = pd.to_numeric(out[c], errors="coerce") if c in out.columns else pd.Series(float("nan"), index=out.index)
        return v.where(~labelled, v.fillna(0.0))

    carries, targets, rec_air = num("carries"), num("targets"), num("receiving_air_yards")
    keys = [out[team_col], out["season"], out["week"]] if team_col in out.columns else [out["season"], out["week"]]

    def share(v):
        tot = v.where(labelled, 0.0).groupby(keys).transform("sum")
        return (v / tot.where(tot > 0)).where(labelled, float("nan")).where(~labelled | (tot > 0), 0.0)

    src = {
        "rush_share_ewm": share(carries),
        "rush_yds_ewm": num("rushing_yards"),
        "target_share_ewm": share(targets),
        "air_yards_share_ewm": share(rec_air.clip(lower=0)),
        "rec_air_yards_ewm": rec_air,
        "pass_att_ewm": num("attempts"),
        "pass_air_yards_ewm": num("passing_air_yards"),
    }
    sort_cols = [c for c in ("season", "week", "_row_id") if c in out.columns]
    order = out.sort_values(sort_cols).index
    pid = out.loc[order, "player_id"]
    for col, v in src.items():
        prev = v.loc[order].groupby(pid, sort=False).shift(1)
        ew = prev.groupby(pid, sort=False).transform(lambda x: x.ewm(halflife=halflife_games, ignore_na=True).mean())
        out[col] = ew.reindex(out.index).fillna(0.0)
    return out


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
    for c in ("snap_pct_last", "snap_pct_roll3", "snap_pct_season", "snap_games_season", "snap_pct_ewm"):
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
    recent_form = bool(feat_cfg.get("recent_form_enabled", False))
    if recent_form:
        out = add_recent_form_features(out, float(feat_cfg.get("ewm_halflife_games", 5.0)))
    pos_specific = bool(feat_cfg.get("position_specific_enabled", False))
    if pos_specific:
        out = add_position_specific_features(out, float(feat_cfg.get("ewm_halflife_games", 5.0)))

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
        from fantasy_model.model import POSITION_CODES as mapping  # QB 0, RB 1, TE 2, WR 3
        out["position_code"] = pos.map(mapping).fillna(-1).astype(int)
    else:
        out["position_code"] = 0

    feature_cols = [c for c in FEATURE_COLUMNS if c in out.columns and (recent_form or c not in RECENT_FORM_COLUMNS)]
    if pos_specific:
        feature_cols += [c for c in POSITION_SPECIFIC_COLUMNS if c in out.columns]
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
