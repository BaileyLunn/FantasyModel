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


# ----------------------------------------------------------------------------- per-position models

POSITION_CODES = {"QB": 0, "RB": 1, "TE": 2, "WR": 3}  # must match features.pipeline position_code
POSITIONS = tuple(POSITION_CODES)


def per_position_flags(cfg: dict[str, Any]) -> dict[str, bool]:
    """Which positions get their own model. Config (``model``)::

        strategy: pooled | per_position | hybrid
        per_position: {QB: true, RB: false, WR: false, TE: false}   # used by strategy hybrid

    ``pooled`` (default) -> all False; ``per_position`` -> all True; ``hybrid`` -> the mapping.
    A ``per_position`` mapping without ``strategy`` is treated as hybrid.
    """
    m = (cfg or {}).get("model", {}) or {}
    strategy = str(m.get("strategy", "") or "").lower()
    mapping = m.get("per_position") or {}
    if isinstance(mapping, bool):
        mapping = {p: mapping for p in POSITIONS}
    if strategy in ("", "hybrid") and mapping:
        return {p: bool(mapping.get(p, False)) for p in POSITIONS}
    if strategy == "per_position":
        return {p: True for p in POSITIONS}
    if strategy not in ("", "pooled", "hybrid"):
        raise ValueError(f"unknown model.strategy {strategy!r}; use pooled | per_position | hybrid")
    return {p: False for p in POSITIONS}


def position_cfg(cfg: dict[str, Any], position: str, seed: int | None = None) -> dict[str, Any]:
    """Config for one position's model: ``model`` hyper-parameters overridden by ``model.position_params[pos]``
    and then by ``model.position_params[<scoring profile>][pos]`` (tuned per profile)."""
    m = dict((cfg or {}).get("model", {}) or {})
    pp = m.get("position_params") or {}
    m.update(pp.get(position) or {})
    # optional per-scoring-profile overrides: position_params: {ppr: {QB: {...}}, half_ppr: {QB: {...}}}
    prof = str(((cfg or {}).get("scoring") or {}).get("profile", "") or "")
    m.update(((pp.get(prof) or {}).get(position) or {}) if prof else {})
    if seed is not None:
        m["random_state"] = int(seed)
    return {**cfg, "model": m}


class PositionRouterModel:
    """Routes each row to its position's model (by the ``position_code`` column); positions without a
    dedicated model use the pooled model (trained on all positions). ``predict(X)`` takes the full
    artifact feature matrix, so evaluate / predict / project / drivers work unchanged.

    ``pooled_cols`` / ``position_cols`` are column indices into X (the pooled model does not see
    position-specific features)."""

    def __init__(self, pooled, pooled_cols, models: dict[str, Any], position_cols: dict[str, list[int]], pos_col: int):
        self.pooled = pooled
        self.pooled_cols = list(pooled_cols)
        self.models = dict(models)
        self.position_cols = {k: list(v) for k, v in position_cols.items()}
        self.pos_col = int(pos_col)

    @property
    def positions_separate(self) -> list[str]:
        return sorted(self.models)

    def model_for(self, position: str):
        return self.models.get(position, self.pooled)

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.full(len(X), np.nan)
        code = X[:, self.pos_col] if len(X) else np.array([])
        routed = np.zeros(len(X), dtype=bool)
        for p, m in self.models.items():
            mask = code == POSITION_CODES[p]
            if mask.any():
                out[mask] = m.predict(X[mask][:, self.position_cols[p]])
            routed |= mask
        rest = ~routed
        if rest.any():
            if self.pooled is None:
                raise ValueError("rows with a position that has no dedicated model and no pooled model")
            out[rest] = self.pooled.predict(X[rest][:, self.pooled_cols])
        return out
