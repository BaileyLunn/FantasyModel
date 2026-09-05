"""Train fantasy points model with time-based split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fantasy_model.config import load_config, project_root, scoring_dict
from fantasy_model.data import load_player_games
from fantasy_model.features.pipeline import build_feature_matrix
from fantasy_model.model import make_model, time_based_split


def train_model(
    cfg: dict[str, Any] | None = None,
    prefer_sample: bool = False,
    config_path: str | None = None,
) -> dict[str, Any]:
    cfg = cfg or load_config(config_path)
    scoring = scoring_dict(cfg)
    df, source = load_player_games(cfg, prefer_sample=prefer_sample)

    X_all, y, feature_cols = build_feature_matrix(df, scoring, cfg)
    if y is None:
        raise ValueError("Target fantasy_points unavailable")

    if "season" not in X_all.columns and "season" in df.columns:
        X_all = X_all.copy()
        X_all["season"] = df["season"].values

    seasons = X_all["season"].to_numpy() if "season" in X_all.columns else df["season"].to_numpy()
    test_season = int(cfg.get("model", {}).get("test_season", int(np.max(seasons))))
    train_idx, test_idx = time_based_split(seasons, test_season)

    min_rows = int(cfg.get("model", {}).get("min_train_rows", 50))
    if len(train_idx) < min_rows:
        # Fallback for tiny sample: last 20% chronologically as test
        order = np.argsort(seasons)
        cut = max(1, int(len(order) * 0.8))
        train_idx, test_idx = order[:cut], order[cut:]
        split_note = "chronological_80_20_fallback"
    else:
        split_note = f"train_season_lt_{test_season}"

    X_train = X_all.iloc[train_idx][feature_cols].to_numpy(dtype=float)
    X_test = X_all.iloc[test_idx][feature_cols].to_numpy(dtype=float)
    y_train = y.iloc[train_idx].to_numpy(dtype=float)
    y_test = y.iloc[test_idx].to_numpy(dtype=float)

    # Impute NaNs with column medians from train only
    med = np.nanmedian(X_train, axis=0)
    med = np.where(np.isnan(med), 0.0, med)

    def _fill(a: np.ndarray) -> np.ndarray:
        out = a.copy()
        inds = np.where(np.isnan(out))
        out[inds] = np.take(med, inds[1])
        return out

    X_train = _fill(X_train)
    X_test = _fill(X_test)

    model = make_model(cfg)
    model.fit(X_train, y_train)
    pred = model.predict(X_test) if len(test_idx) else np.array([])

    metrics: dict[str, Any] = {
        "source": source,
        "split": split_note,
        "test_season": test_season,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "feature_cols": feature_cols,
        "seasons_in_data": sorted(map(int, np.unique(seasons))),
    }
    if len(test_idx):
        metrics.update(
            {
                "mae": float(mean_absolute_error(y_test, pred)),
                "rmse": float(mean_squared_error(y_test, pred) ** 0.5),
                "r2": float(r2_score(y_test, pred)) if len(np.unique(y_test)) > 1 else None,
                "mean_actual": float(np.mean(y_test)),
                "mean_pred": float(np.mean(pred)),
            }
        )

    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "impute_medians": med,
        "scoring": scoring,
        "cfg_model": cfg.get("model", {}),
        "metrics": metrics,
    }
    out_path = models_dir / "fantasy_hgb.joblib"
    joblib.dump(artifact, out_path)
    metrics["model_path"] = str(out_path)

    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "train_metrics.json"
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metrics["report_path"] = str(report_path)

    # Sample predictions CSV for the test fold
    if len(test_idx):
        meta = X_all.iloc[test_idx].copy()
        meta["actual_fp"] = y_test
        meta["predicted_fp"] = pred
        pred_path = reports_dir / "test_predictions.csv"
        keep = [c for c in meta.columns if c in feature_cols or c in (
            "player_id", "player_name", "player_display_name", "season", "week",
            "team", "position", "actual_fp", "predicted_fp",
        )]
        meta[keep].to_csv(pred_path, index=False)
        metrics["predictions_path"] = str(pred_path)

    return metrics


def main(config_path: str | None = None, prefer_sample: bool = False) -> None:
    metrics = train_model(config_path=config_path, prefer_sample=prefer_sample)
    print(json.dumps({k: v for k, v in metrics.items() if k != "feature_cols"}, indent=2))
    print("features:", len(metrics.get("feature_cols", [])))


if __name__ == "__main__":
    main()
