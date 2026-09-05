"""Evaluate a saved model on held-out season or full recompute metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fantasy_model.config import load_config, scoring_dict
from fantasy_model.data import load_player_games
from fantasy_model.features.pipeline import build_feature_matrix
from fantasy_model.model import time_based_split


def evaluate_model(
    cfg: dict[str, Any] | None = None,
    prefer_sample: bool = False,
    config_path: str | None = None,
    model_path: str | None = None,
) -> dict[str, Any]:
    cfg = cfg or load_config(config_path)
    scoring = scoring_dict(cfg)
    models_dir = Path(cfg["paths"]["models_dir"])
    path = Path(model_path) if model_path else models_dir / "fantasy_hgb.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}; run train first")

    artifact = joblib.load(path)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    med = artifact["impute_medians"]

    df, source = load_player_games(cfg, prefer_sample=prefer_sample)
    X_all, y, _ = build_feature_matrix(df, scoring, cfg)
    seasons = df["season"].to_numpy()
    test_season = int(cfg.get("model", {}).get("test_season", int(np.max(seasons))))
    _, test_idx = time_based_split(seasons, test_season)
    if len(test_idx) == 0:
        test_idx = np.arange(len(df))

    X = X_all.iloc[test_idx][feature_cols].to_numpy(dtype=float)
    y_true = y.iloc[test_idx].to_numpy(dtype=float)
    inds = np.where(np.isnan(X))
    X = X.copy()
    X[inds] = np.take(med, inds[1])
    pred = model.predict(X)

    by_pos: dict[str, Any] = {}
    if "position" in df.columns:
        pos = df.iloc[test_idx]["position"].astype(str).str.upper()
        for p in sorted(pos.unique()):
            m = pos == p
            if m.sum() == 0:
                continue
            by_pos[p] = {
                "n": int(m.sum()),
                "mae": float(mean_absolute_error(y_true[m], pred[m])),
                "rmse": float(mean_squared_error(y_true[m], pred[m]) ** 0.5),
            }

    report = {
        "source": source,
        "model_path": str(path),
        "test_season": test_season,
        "n": int(len(test_idx)),
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
        "r2": float(r2_score(y_true, pred)) if len(np.unique(y_true)) > 1 else None,
        "by_position": by_pos,
        "train_metrics_cached": artifact.get("metrics", {}),
    }
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / "eval_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(out)
    return report


def main(config_path: str | None = None, prefer_sample: bool = False) -> None:
    report = evaluate_model(config_path=config_path, prefer_sample=prefer_sample)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
