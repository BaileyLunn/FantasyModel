"""Per-position models: strategy config, router, fitting, per-position intervals, drivers, features."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from fantasy_model.features.pipeline import (
    POSITION_SPECIFIC_COLUMNS, add_position_specific_features, build_feature_matrix,
)
from fantasy_model.model import POSITION_CODES, PositionRouterModel, per_position_flags, position_cfg
from fantasy_model.project import local_drivers
from fantasy_model.train import (
    apply_interval, fit_strategy, fit_weighted, residual_interval_table, residual_interval_tables,
)

HALF = {"pass_yd": 0.04, "pass_td": 4.0, "pass_int": -2.0, "rush_yd": 0.1, "rush_td": 6.0,
        "rec": 0.5, "rec_yd": 0.1, "rec_td": 6.0, "fumble_lost": -2.0, "two_pt": 2.0}
COLS = ["f0", "f1", "position_code"]


def test_strategy_flags():
    assert per_position_flags({}) == {"QB": False, "RB": False, "TE": False, "WR": False}
    assert per_position_flags({"model": {"strategy": "pooled", "per_position": {"QB": True}}})["QB"] is False
    assert all(per_position_flags({"model": {"strategy": "per_position"}}).values())
    hyb = per_position_flags({"model": {"strategy": "hybrid", "per_position": {"QB": True, "TE": True}}})
    assert hyb == {"QB": True, "RB": False, "TE": True, "WR": False}
    assert per_position_flags({"model": {"per_position": {"RB": True}}})["RB"] is True  # mapping alone = hybrid
    with pytest.raises(ValueError):
        per_position_flags({"model": {"strategy": "bogus"}})


def test_position_cfg_overrides_only_that_position():
    cfg = {"model": {"max_depth": 6, "random_state": 42, "position_params": {"QB": {"max_depth": 3}}}}
    assert position_cfg(cfg, "QB")["model"]["max_depth"] == 3
    assert position_cfg(cfg, "RB")["model"]["max_depth"] == 6
    assert position_cfg(cfg, "QB", seed=7)["model"]["random_state"] == 7
    assert cfg["model"]["max_depth"] == 6


def _synthetic(n=1600, seed=0):
    rng = np.random.default_rng(seed)
    code = rng.integers(0, 4, n)
    f0, f1 = rng.normal(size=n), rng.normal(size=n)
    # QB (0) depends on f1 only; others on f0
    y = np.where(code == 0, 5 * f1, 3 * f0) + 10 + rng.normal(scale=0.1, size=n)
    X = np.column_stack([f0, f1, code]).astype(float)
    X[::50, 1] = np.nan  # some missing values -> imputed with global medians
    season = np.repeat([2023, 2024], n // 2)
    return X, y, season, np.ones(n)


CFG = {"model": {"max_iter": 60, "min_samples_leaf": 5, "random_state": 0}, "training": {"recency_half_life_seasons": 4}}


def test_pooled_strategy_is_unchanged_plain_model():
    X, y, s, w = _synthetic()
    m, med, _ = fit_strategy(CFG, X, y, s, w, COLS)
    ref, med_ref, _ = fit_weighted(CFG, X, y, s, w)
    assert isinstance(m, HistGradientBoostingRegressor)
    assert np.allclose(med, med_ref)
    Xi = np.where(np.isnan(X), med, X)
    assert np.allclose(m.predict(Xi), ref.predict(Xi))


def test_hybrid_routes_flagged_positions_and_pooled_rest():
    X, y, s, w = _synthetic()
    cfg = {**CFG, "model": {**CFG["model"], "strategy": "hybrid", "per_position": {"QB": True}}}
    router, med, _ = fit_strategy(cfg, X, y, s, w, COLS)
    assert isinstance(router, PositionRouterModel) and router.positions_separate == ["QB"]
    ref, _, _ = fit_weighted(CFG, X, y, s, w)
    Xi = np.where(np.isnan(X), med, X)
    p = router.predict(Xi)
    qb = Xi[:, 2] == POSITION_CODES["QB"]
    assert np.allclose(p[~qb], ref.predict(Xi[~qb]))                        # pooled positions = pooled model
    assert np.allclose(p[qb], router.models["QB"].predict(Xi[qb][:, [0, 1]]))  # QB = its own model, no position_code
    assert router.models["QB"].n_features_in_ == 2


def test_full_per_position_has_no_pooled_model_and_errors_on_unknown_position():
    X, y, s, w = _synthetic()
    cfg = {**CFG, "model": {**CFG["model"], "strategy": "per_position",
                            "position_params": {"TE": {"max_depth": 2}}}}
    router, med, _ = fit_strategy(cfg, X, y, s, w, COLS)
    assert router.pooled is None and router.positions_separate == ["QB", "RB", "TE", "WR"]
    assert router.models["TE"].max_depth == 2 and router.models["QB"].max_depth != 2
    bad = np.array([[0.0, 0.0, -1.0]])
    with pytest.raises(ValueError):
        router.predict(bad)


def test_position_specific_features_only_reach_per_position_models():
    X, y, s, w = _synthetic()
    X = np.column_stack([X[:, :2], np.random.default_rng(1).normal(size=len(X)), X[:, 2]])
    cols = ["f0", "f1", POSITION_SPECIFIC_COLUMNS[0], "position_code"]
    cfg = {**CFG, "model": {**CFG["model"], "strategy": "hybrid", "per_position": {"QB": True}}}
    router, _, _ = fit_strategy(cfg, X, y, s, w, cols)
    assert router.pooled_cols == [0, 1, 3] and router.position_cols["QB"] == [0, 1, 2]


def test_per_position_intervals():
    rng = np.random.default_rng(0)
    pos = np.array(["QB"] * 500 + ["TE"] * 500)
    pred = rng.uniform(5, 20, 1000)
    actual = pred + np.where(pos == "QB", rng.normal(scale=8, size=1000), rng.normal(scale=2, size=1000))
    t = residual_interval_tables(pred, actual, pos)
    assert set(t["by_position"]) == {"QB", "TE"}
    lo, hi = apply_interval(pred, t, pos)
    width = hi - lo
    assert width[pos == "QB"].mean() > 2.5 * width[pos == "TE"].mean()
    for p in ("QB", "TE"):
        m = pos == p
        cov = np.mean((actual[m] >= lo[m]) & (actual[m] <= hi[m]))
        assert 0.75 <= cov <= 0.85
    # positions without a table (and calls without positions) fall back to the pooled table
    lo2, hi2 = apply_interval(pred[:3], t, np.array(["WR"] * 3))
    lo3, hi3 = apply_interval(pred[:3], residual_interval_table(pred, actual))
    assert np.allclose(lo2, lo3) and np.allclose(hi2, hi3)
    lo4, _ = apply_interval(pred[:3], t)
    assert np.allclose(lo4, lo3)


def test_local_drivers_skip_position_code_for_router():
    X, y, s, w = _synthetic()
    cfg = {**CFG, "model": {**CFG["model"], "strategy": "per_position"}}
    router, med, _ = fit_strategy(cfg, X, y, s, w, COLS)
    Xi = np.where(np.isnan(X), med, X)[:20]
    drv = local_drivers(router, Xi, med, COLS, top=3)
    assert all("position_code" not in d.split("; ")[0] or "(+0.0)" in d for d in drv)
    qb_rows = np.where(Xi[:, 2] == 0)[0]
    assert all(drv[i].startswith("f1=") for i in qb_rows)  # QB model depends on f1


def _hist():
    return pd.DataFrame({
        "player_id": ["q", "r", "q", "r", "q", "r"],
        "team": ["KC"] * 6,
        "position": ["QB", "RB"] * 3,
        "season": [2026] * 6,
        "week": [1, 1, 2, 2, 3, 3],
        "fantasy_points": [20.0, 10.0, 15.0, 12.0, np.nan, np.nan],   # week 3 = projection rows
        "carries": [5, 15, 2, 18, np.nan, np.nan],
        "rushing_yards": [30, 60, 10, 80, np.nan, np.nan],
        "targets": [np.nan, 4, np.nan, 2, np.nan, np.nan],
        "receiving_air_yards": [np.nan, 10, np.nan, 5, np.nan, np.nan],
        "attempts": [35, np.nan, 30, np.nan, np.nan, np.nan],
        "passing_air_yards": [250, np.nan, 200, np.nan, np.nan, np.nan],
        "_row_id": range(6),
    })


def test_position_specific_features_prior_only():
    df = _hist()
    base = add_position_specific_features(df, 1.0)
    assert (base.loc[[0, 1], POSITION_SPECIFIC_COLUMNS] == 0).all().all()  # first game: no history
    assert base.loc[2, "rush_share_ewm"] == pytest.approx(5 / 20)
    assert base.loc[3, "target_share_ewm"] == pytest.approx(1.0)
    a = 0.5
    assert base.loc[4, "rush_share_ewm"] == pytest.approx((5 / 20 * a + 2 / 20) / (a + 1))
    assert base.loc[4, "pass_att_ewm"] == pytest.approx((35 * a + 30) / (a + 1))
    for i in range(len(df)):  # a row's own (and later) outcomes never change its features
        mod = df.copy()
        later = mod["_row_id"] >= i
        mod.loc[later & mod["fantasy_points"].notna(), ["carries", "targets", "rushing_yards", "attempts"]] = 99.0
        out = add_position_specific_features(mod, 1.0)
        assert np.allclose(out.loc[i, POSITION_SPECIFIC_COLUMNS].astype(float), base.loc[i, POSITION_SPECIFIC_COLUMNS].astype(float)), i


def test_position_specific_toggle():
    df = _hist()
    _, _, on = build_feature_matrix(df, HALF, {"features": {"position_specific_enabled": True}})
    _, _, off = build_feature_matrix(df, HALF, {"features": {}})
    assert set(POSITION_SPECIFIC_COLUMNS) <= set(on) and not set(POSITION_SPECIFIC_COLUMNS) & set(off)
    assert on[-1] == "position_code"


def test_cli_per_position_override():
    from fantasy_model.config import apply_per_position_override, load_config

    cfg = load_config()
    assert apply_per_position_override(cfg, None) is cfg
    assert per_position_flags(apply_per_position_override(cfg, "none")) == {p: False for p in POSITION_CODES}
    assert all(per_position_flags(apply_per_position_override(cfg, "all")).values())
    hyb = per_position_flags(apply_per_position_override(cfg, "qb, TE"))
    assert hyb == {"QB": True, "RB": False, "TE": True, "WR": False}
    with pytest.raises(ValueError):
        apply_per_position_override(cfg, "QB,K")
    parser_mod = __import__("fantasy_model.cli", fromlist=["build_parser"])
    args = parser_mod.build_parser().parse_args(["--per-position", "QB", "train"])
    assert args.per_position == "QB"


def test_position_params_per_scoring_profile():
    cfg = {"scoring": {"profile": "ppr"},
           "model": {"max_depth": 6, "min_samples_leaf": 25,
                     "position_params": {"QB": {"max_depth": 4}, "ppr": {"QB": {"min_samples_leaf": 60}},
                                         "half_ppr": {"QB": {"min_samples_leaf": 150}}}}}
    m = position_cfg(cfg, "QB")["model"]
    assert (m["max_depth"], m["min_samples_leaf"]) == (4, 60)
    half = {**cfg, "scoring": {"profile": "half_ppr"}}
    assert position_cfg(half, "QB")["model"]["min_samples_leaf"] == 150
    assert position_cfg(cfg, "RB")["model"]["min_samples_leaf"] == 25


def test_train_evaluate_end_to_end_on_sample(tmp_path):
    """Hybrid artifact from train_model works with evaluate_slice and has per-position intervals."""
    import copy

    import joblib

    from fantasy_model.config import load_config
    from fantasy_model.data import load_player_games
    from fantasy_model.evaluate import evaluate_slice
    from fantasy_model.train import train_model

    cfg = copy.deepcopy(load_config())
    cfg["paths"]["models_dir"] = cfg["paths"]["reports_dir"] = str(tmp_path)
    cfg["model"].update({"strategy": "hybrid", "per_position": {"QB": True}, "max_iter": 30})
    df, src = load_player_games(cfg, prefer_sample=True)
    if src != "sample" and "sample" not in str(src):
        pytest.skip("sample fixture not available")
    last = int(df["season"].max())
    out = tmp_path / "m.joblib"
    m = train_model(cfg=cfg, prefer_sample=True, test_season=last, refit_full=True, model_out=str(out), write_reports=False)
    art = joblib.load(out)
    assert isinstance(art["model"], PositionRouterModel) and art["model"].positions_separate == ["QB"]
    assert art["strategy"]["per_position"]["QB"] is True
    if m["n_test"] >= 200:
        assert "by_position" in art["interval"]
    rep = evaluate_slice(last, None, cfg=cfg, model_path=str(out), df=df)
    assert rep["overall"]["n"] > 0 and rep["strategy"]["per_position"]["QB"] is True
