"""Model factory: HistGradientBoosting (default), optional LightGBM/XGBoost."""

from __future__ import annotations

from typing import Any

import numpy as np


def make_model(cfg: dict[str, Any]):
    model_cfg = cfg.get("model", {})
    backend = str(model_cfg.get("backend", "hist_gradient_boosting")).lower()
    rs = int(model_cfg.get("random_state", 42))

    if backend == "lightgbm":
        try:
            import lightgbm as lgb

            return lgb.LGBMRegressor(
                n_estimators=int(model_cfg.get("max_iter", 200)),
                learning_rate=float(model_cfg.get("learning_rate", 0.08)),
                max_depth=int(model_cfg.get("max_depth", 6)),
                random_state=rs,
                verbosity=-1,
            )
        except ImportError as e:
            raise ImportError("lightgbm not installed; pip install lightgbm") from e

    if backend == "xgboost":
        try:
            import xgboost as xgb

            return xgb.XGBRegressor(
                n_estimators=int(model_cfg.get("max_iter", 200)),
                learning_rate=float(model_cfg.get("learning_rate", 0.08)),
                max_depth=int(model_cfg.get("max_depth", 6)),
                random_state=rs,
                objective="reg:squarederror",
            )
        except ImportError as e:
            raise ImportError("xgboost not installed; pip install xgboost") from e

    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_iter=int(model_cfg.get("max_iter", 200)),
        max_depth=int(model_cfg.get("max_depth", 6)),
        learning_rate=float(model_cfg.get("learning_rate", 0.08)),
        min_samples_leaf=int(model_cfg.get("min_samples_leaf", 20)),
        random_state=rs,
    )


def time_based_split(
    seasons: np.ndarray | list,
    test_season: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Train on seasons < test_season; test on == test_season. No shuffle."""
    seasons = np.asarray(seasons)
    train_idx = np.where(seasons < test_season)[0]
    test_idx = np.where(seasons == test_season)[0]
    return train_idx, test_idx
