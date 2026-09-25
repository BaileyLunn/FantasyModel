"""Recency bias: training sample weights + recent-form EWMA features (no leakage)."""

import numpy as np
import pandas as pd
import pytest

from fantasy_model.config import load_config
from fantasy_model.features.pipeline import add_recent_form_features, build_feature_matrix
from fantasy_model.features.usage import prior_snap_features
from fantasy_model.train import fit_weighted
from fantasy_model.weights import recency_weights, season_time, training_recency

HALF = {"pass_yd": 0.04, "pass_td": 4.0, "pass_int": -2.0, "rush_yd": 0.1, "rush_td": 6.0,
        "rec": 0.5, "rec_yd": 0.1, "rec_td": 6.0, "fumble_lost": -2.0, "two_pt": 2.0}


# ---------------------------------------------------------------- sample weights

def test_weights_monotone_in_time():
    seasons = np.repeat(np.arange(2016, 2027), 18)
    weeks = np.tile(np.arange(1, 19), 11)
    w = recency_weights(seasons, weeks, 3.0)
    order = np.argsort(season_time(seasons, weeks))
    assert np.all(np.diff(w[order]) > 0)          # strictly newer -> strictly heavier
    assert w.mean() == pytest.approx(1.0)


def test_weights_half_life_and_old_seasons_still_count():
    w = recency_weights(np.array([2016, 2022, 2025]), np.array([1, 1, 1]), 3.0)
    assert w[1] / w[2] == pytest.approx(0.5)       # 3 seasons older = half weight
    assert w[0] / w[2] == pytest.approx(0.125)     # 2016 vs 2025: 1/8, not ~0
    # within-season week matters too
    w2 = recency_weights(np.array([2025, 2025]), np.array([1, 18]), 3.0)
    assert w2[0] < w2[1]


def test_weights_uniform_when_disabled_and_config_parse():
    s, wk = np.array([2016, 2025]), np.array([1, 1])
    assert np.allclose(recency_weights(s, wk, None), 1.0)
    assert np.allclose(recency_weights(s, wk, 0), 1.0)
    assert training_recency({"training": {"recency_half_life_seasons": None}})[0] is None
    hl, wps = training_recency(load_config())
    assert hl is not None and 1.0 <= hl <= 6.0 and wps == 18


def test_fit_weighted_passes_weights():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    season = np.repeat([2020, 2025], 200)
    y = np.where(season == 2025, 10.0, 0.0) + X[:, 0]
    cfg = {"model": {"max_iter": 20, "min_samples_leaf": 5}, "training": {"recency_half_life_seasons": 1.0}}
    _, _, w = fit_weighted(cfg, X, y, season, np.ones(400))
    assert w[season == 2025].mean() / w[season == 2020].mean() == pytest.approx(32.0)


# ---------------------------------------------------------------- EWMA features

def _history():
    return pd.DataFrame({
        "player_id": ["a"] * 5 + ["b"] * 3,
        "season": [2025, 2025, 2025, 2026, 2026, 2026, 2026, 2026],
        "week": [16, 17, 18, 1, 2, 1, 2, 3],
        "fantasy_points": [10.0, 20.0, 0.0, 30.0, 5.0, 8.0, 12.0, np.nan],  # b wk3 = unplayed projection row
        "targets": [5, 8, np.nan, 10, 2, 3, 4, np.nan],
        "carries": [0, 1, np.nan, 0, 0, 10, 12, np.nan],
        "receptions": [4, 6, np.nan, 7, 1, 2, 3, np.nan],
        "zero_stat_appearance": [0, 0, 1, 0, 0, 0, 0, 0],
        "_row_id": range(8),
    })


def test_ewm_uses_only_prior_games():
    df = _history()
    base = add_recent_form_features(df, 3.0)
    # First game of each player has no history
    assert base.loc[0, "fp_ewm"] == 0.0 and base.loc[5, "fp_ewm"] == 0.0
    # Second game = exactly the first game's value
    assert base.loc[1, "fp_ewm"] == pytest.approx(10.0)
    # Changing a game's own outcome (and later ones) never changes that row's features
    for i in range(len(df)):
        mod = df.copy()
        later = (mod["player_id"] == mod.loc[i, "player_id"]) & (mod["_row_id"] >= i)
        mod.loc[later, ["fantasy_points", "targets", "carries", "receptions"]] = 999.0
        out = add_recent_form_features(mod, 3.0)
        for c in ("fp_ewm", "targets_ewm", "carries_ewm", "rec_ewm"):
            assert out.loc[i, c] == pytest.approx(base.loc[i, c]), (i, c)


def test_ewm_values_weight_recent_games_more_and_zero_rows_count():
    out = add_recent_form_features(_history(), 3.0)
    # row 3 (a, 2026 wk1) sees 10, 20, 0 (zero-stat week counts as 0 targets) -> weights a^2, a, 1
    a = 0.5 ** (1 / 3.0)
    exp_fp = (10 * a**2 + 20 * a + 0) / (a**2 + a + 1)
    assert out.loc[3, "fp_ewm"] == pytest.approx(exp_fp)
    exp_tg = (5 * a**2 + 8 * a + 0) / (a**2 + a + 1)
    assert out.loc[3, "targets_ewm"] == pytest.approx(exp_tg)
    # more recent-heavy than the plain mean of the same games
    assert out.loc[4, "fp_ewm"] > np.mean([10, 20, 0, 30])
    # projection row (b wk3) gets features from b's played games only
    assert out.loc[7, "fp_ewm"] == pytest.approx((8 * a + 12) / (a + 1))


def test_ewm_order_independent_of_input_row_order():
    df = _history()
    shuffled = df.sample(frac=1.0, random_state=1)
    a = add_recent_form_features(df, 3.0).sort_values("_row_id")["fp_ewm"].to_numpy()
    b = add_recent_form_features(shuffled, 3.0).sort_values("_row_id")["fp_ewm"].to_numpy()
    assert np.allclose(a, b)


def test_snap_ewm_is_strictly_prior():
    snaps = pd.DataFrame({"player_id": ["a"] * 3, "season": [2026] * 3, "week": [1, 2, 3],
                          "offense_pct": [1.0, 0.0, 0.5]})
    keys = pd.DataFrame({"player_id": ["a"] * 4, "season": [2026] * 4, "week": [1, 2, 3, 4]})
    out = prior_snap_features(snaps, keys, ewm_halflife=1.0).set_index("week")
    assert np.isnan(out.loc[1, "snap_pct_ewm"])
    assert out.loc[2, "snap_pct_ewm"] == pytest.approx(1.0)
    assert out.loc[3, "snap_pct_ewm"] == pytest.approx(0.5 / 1.5)  # weights 0.5 (wk1), 1 (wk2)
    assert out.loc[4, "snap_pct_ewm"] == pytest.approx((0.25 * 1 + 0.5 * 0 + 1 * 0.5) / 1.75)


def test_recent_form_toggle_in_feature_matrix():
    df = _history().assign(position="WR", team="KC")
    on = {"features": {"recent_form_enabled": True, "ewm_halflife_games": 3.0}}
    off = {"features": {"recent_form_enabled": False}}
    _, _, cols_on = build_feature_matrix(df, HALF, on)
    _, _, cols_off = build_feature_matrix(df, HALF, off)
    assert {"fp_ewm", "targets_ewm", "carries_ewm", "rec_ewm"} <= set(cols_on)
    assert not {"fp_ewm", "snap_pct_ewm"} & set(cols_off)


def test_default_config_turns_recency_on_consistently():
    from fantasy_model.features.usage import DEFAULT_EWM_HALFLIFE_GAMES

    cfg = load_config()
    assert cfg["features"]["recent_form_enabled"] is True
    assert float(cfg["features"]["ewm_halflife_games"]) == DEFAULT_EWM_HALFLIFE_GAMES
    assert training_recency(cfg)[0] == 4.0


def test_update_week_overrides():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from update_week import apply_recency_overrides

    cfg = load_config()
    assert apply_recency_overrides(cfg, None, None)["training"]["recency_half_life_seasons"] == 4
    off = apply_recency_overrides(cfg, 0, 0)
    assert off["training"]["recency_half_life_seasons"] is None and off["features"]["recent_form_enabled"] is False
    assert cfg["training"]["recency_half_life_seasons"] == 4  # original untouched
