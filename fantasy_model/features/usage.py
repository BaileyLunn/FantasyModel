"""Depth-chart rank and prior snap-share context from nflverse (pre-game information only).

Sources (nflverse-data releases, downloaded to ``data/raw/nflverse_extra/``):

* ``depth_charts/depth_charts_{2016..2024}.parquet`` — weekly depth charts (``depth_team`` 1/2/3
  per ``depth_position``), keyed by ``club_code`` / ``week`` / ``gsis_id``.
* ``depth_charts/depth_charts_{2025,2026}.parquet`` — new format: timestamped (``dt``) daily ESPN
  snapshots with ``pos_abb`` / ``pos_rank``. For each team-week we use the **latest snapshot
  taken before game day** (so it is pre-game for played weeks and "as of now" for upcoming ones).
* ``snap_counts/snap_counts_{season}.parquet`` — PFR offensive snap share per player-game,
  mapped to gsis ids through ``players/players.parquet`` (``pfr_id``).

Output columns attached to the panel (all known before kickoff of the row's game):

* ``depth_rank`` — 1 = top of the team's depth chart at the player's position (WR1..WRn,
  RB1.., QB1.., TE1..); NaN when the player is not listed.
* ``snap_pct_last`` / ``snap_pct_roll3`` — offensive snap share in the previous game / mean of the
  previous 3 games the player appeared in (0 for special-teams-only appearances).
* ``snap_pct_season`` — mean prior offensive snap share this season (NaN before first appearance).
* ``snap_games_season`` — prior games this season with a snap-count row (a proxy for being active).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fantasy_model.teams import normalize_team_series

NFLVERSE_RELEASES = "https://github.com/nflverse/nflverse-data/releases/download"
DEPTH_URL = NFLVERSE_RELEASES + "/depth_charts/depth_charts_{season}.parquet"
SNAPS_URL = NFLVERSE_RELEASES + "/snap_counts/snap_counts_{season}.parquet"
ROSTER_URL = NFLVERSE_RELEASES + "/weekly_rosters/roster_weekly_{season}.parquet"
PLAYERS_URL = NFLVERSE_RELEASES + "/players/players.parquet"

SKILL = ["QB", "RB", "WR", "TE"]
USAGE_COLUMNS = ["depth_rank", "snap_pct_last", "snap_pct_roll3", "snap_pct_season", "snap_games_season"]


def extra_dir(raw_dir: str | Path) -> Path:
    return Path(raw_dir) / "nflverse_extra"


def download_usage_sources(raw_dir: str | Path, seasons: list[int], players: bool = True) -> list[str]:
    """Download depth charts + snap counts for ``seasons`` (and players.parquet). Returns errors."""
    d = extra_dir(raw_dir)
    d.mkdir(parents=True, exist_ok=True)
    errors = []
    for y in seasons:
        for url, name in ((DEPTH_URL, "depth_charts"), (SNAPS_URL, "snap_counts")):
            try:
                pd.read_parquet(url.format(season=y)).to_parquet(d / f"{name}_{y}.parquet", index=False)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{name} {y}: {e}")
        try:
            pd.read_parquet(ROSTER_URL.format(season=y)).to_parquet(Path(raw_dir) / f"roster_weekly_{y}.parquet", index=False)
        except Exception as e:  # noqa: BLE001
            errors.append(f"roster_weekly {y}: {e}")
    if players:
        try:
            pd.read_parquet(PLAYERS_URL).to_parquet(d / "players.parquet", index=False)
        except Exception as e:  # noqa: BLE001
            errors.append(f"players: {e}")
    return errors


# ----------------------------------------------------------------------------- depth charts

def legacy_depth_ranks(dc: pd.DataFrame) -> pd.DataFrame:
    """2016–2024 weekly format -> (player_id, season, week, depth_rank)."""
    d = dc[(dc["formation"].astype(str) == "Offense") & dc["position"].isin(SKILL) & dc["gsis_id"].notna()].copy()
    if "game_type" in d.columns:
        d = d[d["game_type"].astype(str) == "REG"]
    d["team"] = normalize_team_series(d["club_code"])
    d["depth_team"] = pd.to_numeric(d["depth_team"], errors="coerce")
    d = d.dropna(subset=["week", "depth_team"])
    d["week"] = d["week"].astype(int)
    # a player may be listed at several slots (e.g. LWR and SWR): keep his best listing
    d = d.sort_values("depth_team").drop_duplicates(["season", "week", "team", "gsis_id"])
    d = d.sort_values(["season", "week", "team", "position", "depth_team", "jersey_number"])
    d["depth_rank"] = d.groupby(["season", "week", "team", "position"]).cumcount() + 1
    return d.rename(columns={"gsis_id": "player_id"})[["player_id", "season", "week", "team", "depth_rank"]]


def snapshot_depth_ranks(dc: pd.DataFrame, schedules: pd.DataFrame, season: int) -> pd.DataFrame:
    """2025+ timestamped snapshots -> latest snapshot strictly before each team's game day."""
    d = dc[dc["pos_abb"].isin(SKILL) & dc["gsis_id"].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["player_id", "season", "week", "team", "depth_rank"])
    d["team"] = normalize_team_series(d["team"])
    d["snap_date"] = pd.to_datetime(d["dt"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    sched = schedules[(schedules["season"] == season) & (schedules["game_type"].astype(str) == "REG")] \
        if "game_type" in schedules.columns else schedules[schedules["season"] == season]
    tw = pd.concat(
        [sched[["week", "gameday", "home_team"]].rename(columns={"home_team": "team"}),
         sched[["week", "gameday", "away_team"]].rename(columns={"away_team": "team"})],
        ignore_index=True,
    )
    tw["team"] = normalize_team_series(tw["team"])
    tw["gameday"] = pd.to_datetime(tw["gameday"])
    snaps = d[["team", "snap_date"]].drop_duplicates().sort_values("snap_date")
    tw = tw.sort_values("gameday")
    # latest snapshot taken before 00:00 ET of game day
    pick = pd.merge_asof(tw, snaps, left_on="gameday", right_on="snap_date", by="team",
                         direction="backward", allow_exact_matches=False)
    pick = pick.dropna(subset=["snap_date"])
    m = d.merge(pick[["team", "week", "snap_date"]], on=["team", "snap_date"], how="inner")
    m = m.sort_values("pos_rank").drop_duplicates(["week", "team", "gsis_id"])
    # re-rank inside position so ranks are contiguous (1..n) like the legacy format
    m = m.sort_values(["week", "team", "pos_abb", "pos_rank"])
    m["depth_rank"] = m.groupby(["week", "team", "pos_abb"]).cumcount() + 1
    m["season"] = season
    return m.rename(columns={"gsis_id": "player_id"})[["player_id", "season", "week", "team", "depth_rank"]]


def load_depth_ranks(raw_dir: str | Path, schedules: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    d = extra_dir(raw_dir)
    frames = []
    for y in seasons:
        p = d / f"depth_charts_{y}.parquet"
        if not p.exists():
            continue
        dc = pd.read_parquet(p)
        frames.append(snapshot_depth_ranks(dc, schedules, y) if "dt" in dc.columns else legacy_depth_ranks(dc))
    if not frames:
        return pd.DataFrame(columns=["player_id", "season", "week", "team", "depth_rank"])
    out = pd.concat(frames, ignore_index=True)
    out["season"] = out["season"].astype("int64")
    out["week"] = out["week"].astype("int64")
    out["player_id"] = out["player_id"].astype(str)
    return out


# ----------------------------------------------------------------------------- snap counts

def load_snaps(raw_dir: str | Path, seasons: list[int]) -> pd.DataFrame:
    d = extra_dir(raw_dir)
    frames = [pd.read_parquet(d / f"snap_counts_{y}.parquet") for y in seasons if (d / f"snap_counts_{y}.parquet").exists()]
    if not frames or not (d / "players.parquet").exists():
        return pd.DataFrame(columns=["player_id", "season", "week", "offense_pct"])
    s = pd.concat(frames, ignore_index=True)
    s = s[s["game_type"].astype(str) == "REG"]
    players = pd.read_parquet(d / "players.parquet", columns=["gsis_id", "pfr_id"]).dropna()
    idmap = players.drop_duplicates("pfr_id").set_index("pfr_id")["gsis_id"]
    s["player_id"] = s["pfr_player_id"].map(idmap)
    s = s.dropna(subset=["player_id"])
    s["team"] = normalize_team_series(s["team"])
    s = s.sort_values("offense_snaps", ascending=False).drop_duplicates(["player_id", "season", "week"])
    s["season"] = s["season"].astype("int64")
    s["week"] = s["week"].astype("int64")
    return s[["player_id", "season", "week", "team", "offense_snaps", "offense_pct"]].reset_index(drop=True)


def prior_snap_features(snaps: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """For each (player_id, season, week) in ``keys``: snap-share summaries of games strictly before it."""
    keys = keys[["player_id", "season", "week"]].drop_duplicates().copy()
    if snaps.empty:
        for c in USAGE_COLUMNS[1:]:
            keys[c] = np.nan
        return keys
    s = snaps.sort_values(["player_id", "season", "week"]).copy()
    g = s.groupby("player_id")["offense_pct"]
    s["snap_pct_last"] = s["offense_pct"]
    s["snap_pct_roll3"] = g.transform(lambda x: x.rolling(3, min_periods=1).mean())
    gs = s.groupby(["player_id", "season"])["offense_pct"]
    s["snap_pct_season"] = gs.transform(lambda x: x.expanding().mean())
    s["snap_games_season"] = gs.cumcount() + 1
    s["_t"] = (s["season"].astype("int64") * 100 + s["week"].astype("int64")).astype("int64")
    keys["_t"] = (pd.to_numeric(keys["season"]) * 100 + pd.to_numeric(keys["week"])).astype("int64")
    s["player_id"] = s["player_id"].astype(str)
    keys["player_id"] = keys["player_id"].astype(str)
    # as-of join: latest snap game with _t strictly less than the row's _t
    left = keys.sort_values("_t")
    right = s[["player_id", "_t", "season", "snap_pct_last", "snap_pct_roll3", "snap_pct_season", "snap_games_season"]] \
        .rename(columns={"season": "_snap_season"}).sort_values("_t")
    m = pd.merge_asof(left, right, on="_t", by="player_id", direction="backward", allow_exact_matches=False)
    other_season = m["_snap_season"] != m["season"]
    m.loc[other_season, "snap_pct_season"] = np.nan
    m.loc[other_season, "snap_games_season"] = 0
    m["snap_games_season"] = m["snap_games_season"].fillna(0)
    return m.drop(columns=["_t", "_snap_season"])


def add_usage_context(panel: pd.DataFrame, raw_dir: str | Path, schedules: pd.DataFrame | None) -> pd.DataFrame:
    """Attach depth_rank + prior snap-share columns to a player-game panel (in place of any old ones)."""
    out = panel.drop(columns=[c for c in USAGE_COLUMNS if c in panel.columns])
    if out.empty or "player_id" not in out.columns:
        return out
    out["season"] = pd.to_numeric(out["season"]).astype("int64")
    out["week"] = pd.to_numeric(out["week"]).astype("int64")
    out["player_id"] = out["player_id"].astype(str)
    seasons = sorted(map(int, pd.to_numeric(out["season"], errors="coerce").dropna().unique()))
    snap_seasons = list(range(min(seasons) - 1, max(seasons) + 1))
    snaps = load_snaps(raw_dir, snap_seasons)
    feats = prior_snap_features(snaps, out)
    out = out.merge(feats, on=["player_id", "season", "week"], how="left")
    sched = schedules if schedules is not None else pd.DataFrame()
    depth = load_depth_ranks(raw_dir, sched, seasons) if not sched.empty else pd.DataFrame()
    if not depth.empty:
        dr = depth.sort_values("depth_rank").drop_duplicates(["player_id", "season", "week"])
        out = out.merge(dr[["player_id", "season", "week", "depth_rank"]], on=["player_id", "season", "week"], how="left")
    else:
        out["depth_rank"] = np.nan
    return out


# ----------------------------------------------------------------------------- zero-stat games

def zero_stat_appearances(weekly: pd.DataFrame, raw_dir: str | Path) -> pd.DataFrame:
    """Player-games where a QB/RB/WR/TE was on the field (PFR snap counts) but recorded no
    offensive stat, so the weekly stats release has no row for him. These are real 0-point games;
    without them the training set only contains games where a player touched the ball, which
    biases backups' projections upward. Only (season, week) pairs already present in ``weekly``
    are used, so --max-week caps are respected.
    """
    d = extra_dir(raw_dir)
    if weekly.empty or not (d / "players.parquet").exists():
        return pd.DataFrame()
    wk = weekly[["season", "week"]].apply(pd.to_numeric, errors="coerce").dropna().astype("int64").drop_duplicates()
    snaps = load_snaps(raw_dir, sorted(wk["season"].unique().tolist()))
    if snaps.empty:
        return pd.DataFrame()
    snaps = snaps.merge(wk, on=["season", "week"], how="inner")
    have = weekly[["player_id", "season", "week"]].copy()
    have["player_id"] = have["player_id"].astype(str)
    have[["season", "week"]] = have[["season", "week"]].apply(pd.to_numeric, errors="coerce").astype("int64")
    m = snaps.merge(have.drop_duplicates(), on=["player_id", "season", "week"], how="left", indicator=True)
    m = m[m["_merge"] == "left_only"].drop(columns="_merge")
    players = pd.read_parquet(d / "players.parquet", columns=["gsis_id", "display_name", "position"]) \
        .drop_duplicates("gsis_id").rename(columns={"gsis_id": "player_id"})
    m = m.merge(players, on="player_id", how="left")
    m = m[m["position"].isin(SKILL)]
    raw = pd.concat([pd.read_parquet(d / f"snap_counts_{y}.parquet", columns=["season", "week", "team", "opponent", "game_type"])
                     for y in sorted(m["season"].unique()) if (d / f"snap_counts_{y}.parquet").exists()], ignore_index=True)
    raw = raw[raw["game_type"].astype(str) == "REG"].drop_duplicates(["season", "week", "team"])
    raw["team"] = normalize_team_series(raw["team"])
    raw["opponent_team"] = normalize_team_series(raw["opponent"])
    m = m.merge(raw[["season", "week", "team", "opponent_team"]], on=["season", "week", "team"], how="left")
    return pd.DataFrame({
        "player_id": m["player_id"].to_numpy(),
        "player_name": m["display_name"].to_numpy(),
        "player_display_name": m["display_name"].to_numpy(),
        "position": m["position"].to_numpy(),
        "position_group": m["position"].to_numpy(),
        "recent_team": m["team"].to_numpy(),
        "opponent_team": m["opponent_team"].to_numpy(),
        "season": m["season"].to_numpy(),
        "week": m["week"].to_numpy(),
        "season_type": "REG",
        "zero_stat_appearance": 1,
        "zero_row_type": "snap_no_stats",
        "stats_source": "nflverse:snap_counts(zero-stat appearance)",
    })


ROSTERED_STATUSES = ("ACT", "INA")  # 53-man active list incl. game-day inactives (2019+ split them)


def rostered_no_snap_rows(weekly: pd.DataFrame, raw_dir: str | Path) -> pd.DataFrame:
    """Player-weeks where a QB/RB/WR/TE was on the active roster (nflverse weekly roster status
    ACT, or INA = game-day inactive from 2019 on) but has neither a stats row nor a snap-count row:
    a real 0-point week (e.g. the QB2 who dressed but never played). Forward projections are made
    for every ACT roster player *before* inactives are known, so training on the same population
    keeps backups' projections honest. Only (season, week) pairs present in ``weekly`` are used.
    Bye weeks are removed later by the schedule join in ``build_panel``.
    """
    raw_dir = Path(raw_dir)
    wk = weekly[["season", "week"]].apply(pd.to_numeric, errors="coerce").dropna().astype("int64").drop_duplicates()
    frames = []
    for y in sorted(wk["season"].unique()):
        p = raw_dir / f"roster_weekly_{y}.parquet"
        if not p.exists():
            continue
        r = pd.read_parquet(p)
        r = r[r["position"].isin(SKILL) & r["status"].astype(str).str.upper().isin(ROSTERED_STATUSES) & r["gsis_id"].notna()]
        if "game_type" in r.columns:
            r = r[r["game_type"].astype(str) == "REG"]
        frames.append(r[["season", "week", "team", "position", "gsis_id", "full_name"]])
    if not frames:
        return pd.DataFrame()
    r = pd.concat(frames, ignore_index=True)
    r["season"] = r["season"].astype("int64")
    r["week"] = pd.to_numeric(r["week"]).astype("int64")
    r = r.merge(wk, on=["season", "week"], how="inner").rename(columns={"gsis_id": "player_id"})
    r["player_id"] = r["player_id"].astype(str)
    r["team"] = normalize_team_series(r["team"])
    r = r.drop_duplicates(["player_id", "season", "week"])
    have = weekly[["player_id", "season", "week"]].copy()
    have["player_id"] = have["player_id"].astype(str)
    have[["season", "week"]] = have[["season", "week"]].apply(pd.to_numeric, errors="coerce").astype("int64")
    snaps = load_snaps(raw_dir, sorted(wk["season"].unique().tolist()))[["player_id", "season", "week"]]
    seen = pd.concat([have, snaps], ignore_index=True).drop_duplicates()
    m = r.merge(seen, on=["player_id", "season", "week"], how="left", indicator=True)
    m = m[m["_merge"] == "left_only"]
    return pd.DataFrame({
        "player_id": m["player_id"].to_numpy(),
        "player_name": m["full_name"].to_numpy(),
        "player_display_name": m["full_name"].to_numpy(),
        "position": m["position"].to_numpy(),
        "position_group": m["position"].to_numpy(),
        "recent_team": m["team"].to_numpy(),
        "season": m["season"].to_numpy(),
        "week": m["week"].to_numpy(),
        "season_type": "REG",
        "zero_stat_appearance": 1,
        "zero_row_type": "rostered_no_snap",
        "stats_source": "nflverse:weekly_rosters(rostered, no snap)",
    })
