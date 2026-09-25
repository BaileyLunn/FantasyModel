"""Teammate context: QB for skill players, usage shares, backup QB effects."""

from __future__ import annotations

import pandas as pd


def add_teammate_features(df: pd.DataFrame, rolling_games: int = 3) -> pd.DataFrame:
    """Add QB fantasy context and prior-week usage shares (no same-game leakage)."""
    out = df.copy()
    if "_row_id" not in out.columns:
        out["_row_id"] = range(len(out))

    team_col = "team" if "team" in out.columns else "recent_team"
    if team_col not in out.columns:
        out["team_qb_fp_roll"] = 0.0
        out["is_backup_qb_game"] = 0
        out["team_target_share"] = 0.0
        out["team_rush_share"] = 0.0
        return out

    sort_cols = [c for c in ("season", "week", "_row_id") if c in out.columns]
    out = out.sort_values(sort_cols)

    # Team QB rolling FP from prior team games only. Aggregate to one value per team-week (the
    # top QB's points) BEFORE shifting: the old row-wise shift let the 2nd QB row of a team-week
    # (and hence every skill player's mean) see the other QB's same-game points.
    out["team_qb_fp_roll"] = 0.0
    if "position" in out.columns and "fantasy_points" in out.columns:
        qb_mask = out["position"].astype(str).str.upper().eq("QB")
        qb = out.loc[qb_mask].copy()
        group_keys = [c for c in (team_col, "season", "week") if c in qb.columns]
        if not qb.empty and len(group_keys) == 3:
            tw = qb.groupby(group_keys, as_index=False)["fantasy_points"].max().sort_values(["season", "week"])
            tw["team_qb_fp_roll"] = tw.groupby(team_col)["fantasy_points"].transform(
                lambda s: s.shift(1).rolling(rolling_games, min_periods=1).mean()
            )
            out = out.drop(columns=["team_qb_fp_roll"], errors="ignore")
            out = out.merge(tw[group_keys + ["team_qb_fp_roll"]], on=group_keys, how="left")
    out["team_qb_fp_roll"] = out["team_qb_fp_roll"].fillna(0.0)

    # Backup QB heuristic from prior week team QB production
    out["is_backup_qb_game"] = 0
    if "position" in out.columns and "passing_yards" in out.columns:
        qb_mask = out["position"].astype(str).str.upper().eq("QB")
        tmp = out.loc[qb_mask, [team_col, "season", "week", "passing_yards"]].copy()
        if not tmp.empty:
            tmp["_py"] = pd.to_numeric(tmp["passing_yards"], errors="coerce").fillna(0.0)
            keys = [c for c in (team_col, "season", "week") if c in tmp.columns]
            week_max = tmp.groupby(keys)["_py"].transform("max")
            tmp["low_qb"] = ((tmp["_py"] < 100) & (week_max < 150)).astype(int)
            team_week_flag = tmp.groupby(keys, as_index=False)["low_qb"].max()
            team_week_flag = team_week_flag.sort_values(keys)
            team_week_flag["is_backup_qb_game"] = (
                team_week_flag.groupby(team_col)["low_qb"].shift(1).fillna(0).astype(int)
            )
            out = out.drop(columns=["is_backup_qb_game"], errors="ignore")
            out = out.merge(team_week_flag[keys + ["is_backup_qb_game"]], on=keys, how="left")
            out["is_backup_qb_game"] = out["is_backup_qb_game"].fillna(0).astype(int)

    # Prior rolling usage shares
    for raw, share_name in (
        ("targets", "team_target_share"),
        ("carries", "team_rush_share"),
        ("rushing_attempts", "team_rush_share"),
    ):
        if raw not in out.columns or "player_id" not in out.columns:
            continue
        if share_name == "team_rush_share" and raw == "rushing_attempts" and "carries" in out.columns:
            continue
        vals = pd.to_numeric(out[raw], errors="coerce").fillna(0.0)
        keys = [c for c in (team_col, "season", "week") if c in out.columns]
        if keys:
            team_tot = vals.groupby([out[k] for k in keys]).transform("sum")
        else:
            team_tot = vals
        share = (vals / team_tot.replace(0, pd.NA)).fillna(0.0)
        out["_share_raw"] = share
        out[share_name] = (
            out.groupby("player_id")["_share_raw"]
            .transform(lambda s: s.shift(1).rolling(rolling_games, min_periods=1).mean())
            .fillna(0.0)
        )
        out = out.drop(columns=["_share_raw"], errors="ignore")

    if "team_target_share" not in out.columns:
        out["team_target_share"] = 0.0
    if "team_rush_share" not in out.columns:
        out["team_rush_share"] = 0.0

    return out
