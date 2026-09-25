"""Evaluate a saved model on held-out season or full recompute metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fantasy_model.config import load_config, model_path as default_model_path, profile_tag, scoring_dict
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
    path = Path(model_path) if model_path else default_model_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}; run train first")

    artifact = joblib.load(path)
    note = None
    # A refit_full production artifact has seen the holdout season; score its validation twin.
    if model_path is None and (artifact.get("metrics") or {}).get("refit_full"):
        val = path.with_name(path.stem + "_val" + path.suffix)
        if val.exists():
            path = val
            artifact = joblib.load(val)
            note = "production model is refit on all data; holdout scored with validation model"
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    med = artifact["impute_medians"]

    df, source = load_player_games(cfg, prefer_sample=prefer_sample)
    df = df.reset_index(drop=True)
    X_all, y, _ = build_feature_matrix(df, scoring, cfg)
    seasons = df["season"].to_numpy()
    test_season = int(cfg.get("model", {}).get("test_season", int(np.max(seasons))))
    _, test_idx = time_based_split(seasons, test_season)
    if len(test_idx) == 0:
        test_idx = np.arange(len(df))
    test_idx = test_idx[y.iloc[test_idx].notna().to_numpy()]

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
        "note": note,
        "train_metrics_cached": artifact.get("metrics", {}),
    }
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / f"eval_report_{profile_tag(cfg)}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(out)
    return report


def _metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
        "r2": float(r2_score(y_true, pred)) if len(np.unique(y_true)) > 1 else None,
    }


def evaluate_slice(
    season: int,
    weeks: list[int] | None = None,
    cfg: dict[str, Any] | None = None,
    config_path: str | None = None,
    model_path: str | None = None,
    df: pd.DataFrame | None = None,
    out_path: str | None = None,
) -> dict[str, Any]:
    """Score a saved model on played games of ``season`` (optionally only ``weeks``).

    Features are built on the full panel so rolling/opponent features see all prior games,
    exactly as they would have pre-game. Reports overall, per-week and per-position
    MAE / RMSE / R². Rows without a label (projection rows) are skipped.
    """
    cfg = cfg or load_config(config_path)
    scoring = scoring_dict(cfg)
    path = Path(model_path) if model_path else default_model_path(cfg)
    artifact = joblib.load(path)
    if df is None:
        df, _ = load_player_games(cfg)
    df = df.reset_index(drop=True)
    X_all, y, _ = build_feature_matrix(df, scoring, cfg)
    mask = (df["season"] == season) & y.notna()
    if weeks:
        mask &= df["week"].isin(weeks)
    idx = np.where(mask.to_numpy())[0]
    if len(idx) == 0:
        raise ValueError(f"no labelled rows for season={season} weeks={weeks}")
    X = X_all.iloc[idx][artifact["feature_cols"]].to_numpy(dtype=float)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(artifact["impute_medians"], inds[1])
    pred = artifact["model"].predict(X)
    y_true = y.iloc[idx].to_numpy(dtype=float)
    sub = df.iloc[idx]
    wk = sub["week"].to_numpy()
    pos = sub["position"].astype(str).str.upper().to_numpy()

    report: dict[str, Any] = {
        "scoring_profile": artifact.get("scoring_profile"),
        "model_path": str(path),
        "model_train_seasons": artifact.get("metrics", {}).get("train_seasons")
        or artifact.get("metrics", {}).get("seasons_in_data"),
        "season": season,
        "weeks": sorted(map(int, np.unique(wk))),
        "overall": _metrics(y_true, pred),
        "by_position": {p: _metrics(y_true[pos == p], pred[pos == p]) for p in sorted(np.unique(pos))},
        "by_week": {},
    }
    if "zero_stat_appearance" in sub.columns:
        keep = pd.to_numeric(sub["zero_stat_appearance"], errors="coerce").fillna(0).to_numpy() != 1
        report["stats_rows_only"] = {
            **_metrics(y_true[keep], pred[keep]),
            "by_position": {p: _metrics(y_true[keep & (pos == p)], pred[keep & (pos == p)]) for p in sorted(np.unique(pos[keep]))},
            "by_week": {str(int(w)): _metrics(y_true[keep & (wk == w)], pred[keep & (wk == w)]) for w in sorted(np.unique(wk[keep]))},
            "note": "excludes zero-stat snap-count appearances; comparable to earlier panels",
        }
    for w in sorted(np.unique(wk)):
        m = wk == w
        report["by_week"][str(int(w))] = {
            **_metrics(y_true[m], pred[m]),
            "by_position": {p: _metrics(y_true[m & (pos == p)], pred[m & (pos == p)]) for p in sorted(np.unique(pos[m]))},
        }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report_path"] = str(out_path)
    return report


def main(config_path: str | None = None, prefer_sample: bool = False) -> None:
    report = evaluate_model(config_path=config_path, prefer_sample=prefer_sample)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
