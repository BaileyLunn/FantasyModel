"""In-season update plumbing: source harmonization, projection labels, forward rows."""

import numpy as np
import pandas as pd

from fantasy_model.project import forward_rows
from fantasy_model.scoring import ensure_fantasy_points
from fantasy_model.sources import harmonize_stats_player

HALF_PPR = {"pass_yd": 0.04, "pass_td": 4.0, "pass_int": -2.0, "rush_yd": 0.1, "rush_td": 6.0,
            "rec": 0.5, "rec_yd": 0.1, "rec_td": 6.0, "fumble_lost": -2.0, "two_pt": 2.0}


def test_harmonize_stats_player_renames_and_filters():
    raw = pd.DataFrame(
        [
            {"player_id": "a", "team": "KC", "game_id": "g1", "passing_interceptions": 2, "attempts": 30,
             "passing_yards": 250, "sacks_suffered": 1, "carries": 0, "targets": 0},
            # special-teams only appearance -> dropped (legacy player_stats had no such rows)
            {"player_id": "b", "team": "KC", "game_id": "g1", "passing_interceptions": 0, "attempts": 0,
             "passing_yards": 0, "sacks_suffered": 0, "carries": 0, "targets": 0},
        ]
    )
    out = harmonize_stats_player(raw)
    assert list(out["player_id"]) == ["a"]
    assert {"recent_team", "interceptions", "sacks", "stats_game_id"} <= set(out.columns)
    assert "team" not in out.columns and "game_id" not in out.columns


def test_interceptions_scored_from_stats_player_column():
    df = pd.DataFrame([{"passing_yards": 250, "passing_tds": 2, "passing_interceptions": 1}])
    pts = ensure_fantasy_points(df, HALF_PPR)["fantasy_points"].iloc[0]
    assert abs(pts - (10 + 8 - 2)) < 1e-9


def test_projection_rows_have_missing_label():
    df = pd.DataFrame(
        [
            {"receptions": 5, "receiving_yards": 50, "is_projection": 0},
            {"receptions": np.nan, "receiving_yards": np.nan, "is_projection": 1},
        ]
    )
    out = ensure_fantasy_points(df, HALF_PPR)
    assert out["fantasy_points"].iloc[0] == 7.5
    assert np.isnan(out["fantasy_points"].iloc[1])


def test_forward_rows_active_skill_players_with_opponent():
    roster = pd.DataFrame(
        [
            {"season": 2026, "week": 3, "team": "BUF", "position": "QB", "status": "ACT", "gsis_id": "1", "full_name": "A"},
            {"season": 2026, "week": 3, "team": "LAC", "position": "WR", "status": "ACT", "gsis_id": "2", "full_name": "B"},
            {"season": 2026, "week": 3, "team": "LAC", "position": "WR", "status": "RES", "gsis_id": "3", "full_name": "C"},
            {"season": 2026, "week": 3, "team": "LAC", "position": "K", "status": "ACT", "gsis_id": "4", "full_name": "D"},
            {"season": 2026, "week": 2, "team": "LAC", "position": "RB", "status": "ACT", "gsis_id": "5", "full_name": "E"},
        ]
    )
    sched = pd.DataFrame([{"season": 2026, "week": 3, "home_team": "BUF", "away_team": "LAC"}])
    out = forward_rows(roster, sched, 2026, 3)
    assert sorted(out["player_id"]) == ["1", "2"]
    assert dict(zip(out["player_id"], out["opponent_team"])) == {"1": "LAC", "2": "BUF"}
    assert (out["is_projection"] == 1).all()
