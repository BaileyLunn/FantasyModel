"""Regression tests for the 2026-09-25 bug-fix pass."""

import numpy as np
import pandas as pd
import pytest

from fantasy_model.config import load_config, model_path, scoring_dict, set_scoring_profile
from fantasy_model.features.baselines import add_baseline_features, implied_team_total
from fantasy_model.features.history import prior_weeks_mean
from fantasy_model.features.pipeline import add_usage_features
from fantasy_model.features.teammates import add_teammate_features
from fantasy_model.features.travel import travel_distance_miles
from fantasy_model.features.usage import legacy_depth_ranks, prior_snap_features, snapshot_depth_ranks
from fantasy_model.panel import build_panel
from fantasy_model.teams import normalize_team
from fantasy_model.train import apply_interval, residual_interval_table

HALF = {"pass_yd": 0.04, "pass_td": 4.0, "pass_int": -2.0, "rush_yd": 0.1, "rush_td": 6.0,
        "rec": 0.5, "rec_yd": 0.1, "rec_td": 6.0, "fumble_lost": -2.0, "two_pt": 2.0}


# 1. Vegas implied team total --------------------------------------------------------------

def test_implied_total_favorite_gets_more_points():
    # nflverse: spread_line = expected home margin. DET -7 at home vs NO, total 49.5 (2026 wk1 row)
    home = implied_team_total(pd.Series(["DET"]), pd.Series(["DET"]), pd.Series([7.0]), pd.Series([49.5]))
    away = implied_team_total(pd.Series(["NO"]), pd.Series(["DET"]), pd.Series([7.0]), pd.Series([49.5]))
    assert home.iloc[0] == pytest.approx(28.25) and away.iloc[0] == pytest.approx(21.25)
    # road favorite: CHI -3 at CAR (spread_line -3), total 47.5
    fav = implied_team_total(pd.Series(["CHI"]), pd.Series(["CAR"]), pd.Series([-3.0]), pd.Series([47.5]))
    dog = implied_team_total(pd.Series(["CAR"]), pd.Series(["CAR"]), pd.Series([-3.0]), pd.Series([47.5]))
    assert fav.iloc[0] == pytest.approx(25.25) and dog.iloc[0] == pytest.approx(22.25)
    assert fav.iloc[0] + dog.iloc[0] == pytest.approx(47.5)


def test_baseline_feature_uses_fixed_sign():
    df = pd.DataFrame({
        "player_id": ["a", "b"], "season": 2026, "week": 1, "team": ["DET", "NO"], "position": "WR",
        "home_team": "DET", "away_team": "NO", "opponent_team": ["NO", "DET"],
        "spread_line": 7.0, "total_line": 49.5, "fantasy_points": [10.0, 5.0], "_row_id": [0, 1],
    })
    out = add_baseline_features(df).set_index("player_id")
    assert out.loc["a", "implied_team_total"] > out.loc["b", "implied_team_total"]


# 3. Relocated-team abbreviations ------------------------------------------------------------

def test_normalize_team_codes():
    assert [normalize_team(x) for x in ("OAK", "SD", "STL", "LAR", "LV", "KC")] == ["LV", "LAC", "LA", "LA", "LV", "KC"]


def test_oakland_schedule_joins_lv_weekly_row():
    weekly = pd.DataFrame([{"player_id": "p", "position": "WR", "recent_team": "LV", "opponent_team": "SD",
                            "season": 2017, "week": 1, "season_type": "REG", "receptions": 3, "receiving_yards": 40}])
    sched = pd.DataFrame([{"game_id": "2017_01_OAK_SD", "season": 2017, "week": 1, "gameday": "2017-09-10",
                           "home_team": "SD", "away_team": "OAK", "spread_line": -3.0, "total_line": 50.0}])
    panel = build_panel(weekly, sched, None, HALF)
    assert panel["game_id"].iloc[0] == "2017_01_OAK_SD"
    assert panel["home_team"].iloc[0] == "LAC" and panel["opponent_team"].iloc[0] == "LAC"


def test_special_teams_row_opponent_equal_team_is_repaired():
    weekly = pd.DataFrame([{"player_id": "p", "position": "WR", "recent_team": "MIA", "opponent_team": "MIA",
                            "season": 2016, "week": 5, "season_type": "REG", "special_teams_tds": 1}])
    sched = pd.DataFrame([{"game_id": "g", "season": 2016, "week": 5, "home_team": "MIA", "away_team": "TEN"}])
    assert build_panel(weekly, sched, None, HALF)["opponent_team"].iloc[0] == "TEN"


def test_travel_uses_historic_oakland_venue():
    # KC at Raiders: Oakland in 2019, Las Vegas from 2020
    assert travel_distance_miles("KC", "LV", season=2019) != pytest.approx(travel_distance_miles("KC", "LV", season=2021))
    assert travel_distance_miles("LV", "SF", season=2019) < 30  # Oakland -> Santa Clara


# 4. Same-week leakage in group features -----------------------------------------------------

def test_prior_weeks_mean_excludes_same_week():
    df = pd.DataFrame({"season": 2020, "week": [1, 1, 2, 2], "g": "x", "v": [10.0, 20.0, 99.0, 0.0]})
    out = prior_weeks_mean(df, ["g"], "v", min_count=1)
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])  # nothing strictly earlier
    assert out.iloc[2] == pytest.approx(15.0) and out.iloc[3] == pytest.approx(15.0)


def test_team_qb_feature_does_not_see_same_week_qb():
    df = pd.DataFrame({
        "player_id": ["qb1", "qb2", "wr", "qb1", "wr"], "team": "KC", "season": 2020,
        "week": [1, 1, 1, 2, 2], "position": ["QB", "QB", "WR", "QB", "WR"],
        "fantasy_points": [20.0, 2.0, 10.0, 30.0, 8.0], "_row_id": range(5),
    })
    out = add_teammate_features(df).set_index(["player_id", "week"])
    assert out.loc[("qb2", 1), "team_qb_fp_roll"] == 0.0  # old code leaked qb1's 20 here
    assert out.loc[("wr", 1), "team_qb_fp_roll"] == 0.0
    assert out.loc[("wr", 2), "team_qb_fp_roll"] == pytest.approx(20.0)


# 2. Depth chart / snaps ---------------------------------------------------------------------

def test_legacy_depth_ranks_contiguous_within_position():
    dc = pd.DataFrame({
        "season": 2019, "club_code": "OAK", "week": 1.0, "game_type": "REG", "formation": "Offense",
        "position": ["WR", "WR", "WR", "WR", "QB", "QB"], "depth_team": ["1", "1", "2", "1", "1", "2"],
        "gsis_id": ["w1", "w2", "w3", "w1", "q1", "q2"], "jersey_number": ["10", "11", "12", "10", "4", "8"],
    })
    r = legacy_depth_ranks(dc).set_index("player_id")
    assert r.loc["q1", "depth_rank"] == 1 and r.loc["q2", "depth_rank"] == 2
    assert r.loc["w3", "depth_rank"] == 3 and set(r.loc[["w1", "w2"], "depth_rank"]) == {1, 2}
    assert (r["team"] == "LV").all()


def test_snapshot_depth_uses_latest_snapshot_before_game_day():
    dc = pd.DataFrame({
        "dt": ["2026-09-20T12:00:00Z", "2026-09-26T12:00:00Z", "2026-09-28T12:00:00Z"],
        "team": "GB", "gsis_id": "p", "pos_abb": "RB", "pos_rank": [3, 1, 2],
    })
    sched = pd.DataFrame({"season": 2026, "week": [3], "gameday": ["2026-09-27"], "home_team": ["GB"],
                          "away_team": ["ATL"], "game_type": ["REG"]})
    r = snapshot_depth_ranks(dc, sched, 2026)
    assert len(r) == 1 and r["depth_rank"].iloc[0] == 1  # the 09-26 snapshot, not 09-28


def test_prior_snap_features_strictly_prior():
    snaps = pd.DataFrame({"player_id": "p", "season": 2026, "week": [1, 2], "offense_pct": [0.2, 0.8]})
    keys = pd.DataFrame({"player_id": "p", "season": 2026, "week": [1, 2, 3]})
    f = prior_snap_features(snaps, keys).set_index("week")
    assert np.isnan(f.loc[1, "snap_pct_last"])
    assert f.loc[2, "snap_pct_last"] == pytest.approx(0.2)
    assert f.loc[3, "snap_pct_last"] == pytest.approx(0.8) and f.loc[3, "snap_pct_roll3"] == pytest.approx(0.5)
    assert f.loc[3, "snap_games_season"] == 2


def test_usage_features_unlisted_is_deep_reserve_not_median():
    out = add_usage_features(pd.DataFrame({"depth_rank": [1.0, np.nan], "snap_pct_last": [0.9, np.nan]}))
    assert list(out["depth_rank_filled"]) == [1.0, 9.0] and list(out["depth_listed"]) == [1, 0]
    assert list(out["snap_pct_last"]) == [0.9, 0.0] and list(out["has_snap_history"]) == [1, 0]


# 5. Scoring profiles ------------------------------------------------------------------------

def test_scoring_profiles():
    cfg = load_config()
    assert scoring_dict(set_scoring_profile(cfg, "half_ppr"))["rec"] == 0.5
    ppr = scoring_dict(set_scoring_profile(cfg, "ppr"))
    assert ppr["rec"] == 1.0 and ppr["pass_td"] == 4.0 and ppr["rec_yd"] == 0.1
    assert scoring_dict(set_scoring_profile(cfg, "standard"))["rec"] == 0.0
    assert model_path(set_scoring_profile(cfg, "full")).name == "fantasy_hgb_ppr.joblib"
    with pytest.raises(ValueError):
        set_scoring_profile(cfg, "tep")


def test_flat_legacy_scoring_config_still_works():
    assert scoring_dict({"scoring": dict(HALF)})["rec"] == 0.5


# Interval -----------------------------------------------------------------------------------

def test_residual_interval_covers_about_80pct():
    rng = np.random.default_rng(0)
    pred = rng.uniform(0, 20, 5000)
    actual = pred + rng.normal(0, 1 + pred / 5)
    tab = residual_interval_table(pred, actual)
    lo, hi = apply_interval(pred, tab)
    cov = np.mean((actual >= lo) & (actual <= hi))
    assert 0.77 < cov < 0.83
    assert (hi - lo)[pred > 15].mean() > (hi - lo)[pred < 5].mean()  # wider for bigger projections


def test_zero_point_rows_for_backups(tmp_path):
    from fantasy_model.features.usage import rostered_no_snap_rows, zero_stat_appearances

    ex = tmp_path / "nflverse_extra"
    ex.mkdir()
    pd.DataFrame({"gsis_id": ["s", "b", "q2"], "pfr_id": ["S1", "B1", "Q1"], "display_name": ["Star", "Blocker", "QB Two"],
                  "position": ["WR", "TE", "QB"]}).to_parquet(ex / "players.parquet")
    pd.DataFrame({"season": 2024, "week": 1, "game_type": "REG", "team": "OAK", "opponent": "KC",
                  "pfr_player_id": ["S1", "B1"], "offense_snaps": [60.0, 40.0], "offense_pct": [0.9, 0.6]}
                 ).to_parquet(ex / "snap_counts_2024.parquet")
    pd.DataFrame({"season": 2024, "week": 1, "game_type": "REG", "team": "LV", "status": ["ACT", "ACT", "ACT", "RES"],
                  "position": ["WR", "TE", "QB", "RB"], "gsis_id": ["s", "b", "q2", "ir"], "full_name": ["Star", "Blocker", "QB Two", "Hurt"]}
                 ).to_parquet(tmp_path / "roster_weekly_2024.parquet")
    weekly = pd.DataFrame({"player_id": ["s"], "season": [2024], "week": [1]})  # only the star has a stats row
    z = zero_stat_appearances(weekly, tmp_path)
    assert list(z["player_id"]) == ["b"] and z["recent_team"].iloc[0] == "LV" and z["opponent_team"].iloc[0] == "KC"
    r = rostered_no_snap_rows(weekly, tmp_path)
    assert list(r["player_id"]) == ["q2"]  # dressed QB2 with no snaps; RES (IR) player excluded
    panel = build_panel(pd.concat([weekly.assign(position="WR", recent_team="LV", season_type="REG", receptions=5), z, r]),
                        None, None, HALF)
    assert panel.set_index("player_id").loc[["b", "q2"], "fantasy_points"].eq(0).all()
