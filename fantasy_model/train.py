"""Train fantasy points model with time-based split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fantasy_model.config import load_config, model_path, profile_tag, scoring_dict, scoring_profile
from fantasy_model.data import load_player_games
from fantasy_model.features.pipeline import build_feature_matrix
from fantasy_model.model import make_model, time_based_split
from fantasy_model.weights import recency_weights, training_recency, weight_summary

INTERVAL_QUANTILES = (0.10, 0.90)


def residual_interval_table(pred: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    """Empirical residual quantiles by projection bin (quantile bins of the prediction).

    Used to attach an honest, out-of-sample 80% range to projections: for a projection falling in
    bin b, low/high = projection + q10/q90 of (actual - predicted) on the holdout rows in bin b.
    """
    pred = np.asarray(pred, dtype=float)
    resid = np.asarray(actual, dtype=float) - pred
    edges = np.unique(np.quantile(pred, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, len(edges) - 2)
    lo, hi, n = [], [], []
    for i in range(len(edges) - 1):
        r = resid[b == i]
        lo.append(float(np.quantile(r, INTERVAL_QUANTILES[0])) if len(r) else float("nan"))
        hi.append(float(np.quantile(r, INTERVAL_QUANTILES[1])) if len(r) else float("nan"))
        n.append(int(len(r)))
    inside = None
    if len(pred):
        lo_a, hi_a = np.array(lo)[b], np.array(hi)[b]
        inside = float(np.mean((resid >= lo_a) & (resid <= hi_a)))
    return {
        "edges": [float(e) for e in edges], "resid_lo": lo, "resid_hi": hi, "n": n,
        "quantiles": list(INTERVAL_QUANTILES), "holdout_coverage": inside,
        "note": "empirical 80% range from holdout residuals of the validation model, binned by projection",
    }


def apply_interval(pred: np.ndarray, table: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=float)
    if not table:
        return np.full_like(pred, np.nan), np.full_like(pred, np.nan)
    edges = np.array(table["edges"], dtype=float)
    b = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, len(edges) - 2)
    return pred + np.array(table["resid_lo"])[b], pred + np.array(table["resid_hi"])[b]


def impute(X: np.ndarray, med: np.ndarray) -> np.ndarray:
    out = np.array(X, dtype=float, copy=True)
    inds = np.where(np.isnan(out))
    out[inds] = np.take(med, inds[1])
    return out


def fit_weighted(cfg: dict[str, Any], X: np.ndarray, y: np.ndarray, season, week):
    """Median-impute (medians from these rows) and fit with recency sample weights
    (``training.recency_half_life_seasons``; uniform when unset). Returns (model, medians, weights)."""
    med = np.nanmedian(X, axis=0) if len(X) else np.zeros(X.shape[1])
    med = np.where(np.isnan(med), 0.0, med)
    half_life, wps = training_recency(cfg)
    w = recency_weights(season, week, half_life, wps)
    model = make_model(cfg)
    model.fit(impute(X, med), y, sample_weight=w)
    return model, med, w


def train_model(
    cfg: dict[str, Any] | None = None,
    prefer_sample: bool = False,
    config_path: str | None = None,
    test_season: int | None = None,
    refit_full: bool | None = None,
    model_out: str | None = None,
    write_reports: bool = True,
) -> dict[str, Any]:
    """Validate on a held-out season, then (optionally) refit on all labelled rows.

    * Validation: train on seasons < ``test_season``; score ``test_season`` (metrics reported).
    * ``refit_full`` (config ``model.refit_full``): the saved production artifact is refit on
      every labelled row (all seasons, incl. the holdout and current in-season weeks) with the
      same hyper-parameters. The validation-only model is saved alongside as ``*_val.joblib``.
    """
    cfg = cfg or load_config(config_path)
    scoring = scoring_dict(cfg)
    df, source = load_player_games(cfg, prefer_sample=prefer_sample)
    df = df.reset_index(drop=True)
    model_cfg = cfg.get("model", {})
    if refit_full is None:
        refit_full = bool(model_cfg.get("refit_full", False))

    X_all, y, feature_cols = build_feature_matrix(df, scoring, cfg)
    if y is None:
        raise ValueError("Target fantasy_points unavailable")

    if "season" not in X_all.columns and "season" in df.columns:
        X_all = X_all.copy()
        X_all["season"] = df["season"].values

    seasons = X_all["season"].to_numpy() if "season" in X_all.columns else df["season"].to_numpy()
    if test_season is None:
        test_season = int(model_cfg.get("test_season", int(np.max(seasons))))
    train_idx, test_idx = time_based_split(seasons, test_season)
    labelled = y.notna().to_numpy()
    train_idx = train_idx[labelled[train_idx]]
    test_idx = test_idx[labelled[test_idx]]

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

    half_life, wps = training_recency(cfg)
    wk_all = df["week"].to_numpy() if "week" in df.columns else np.ones(len(df))
    model, med, w_train = fit_weighted(cfg, X_train, y_train, seasons[train_idx], wk_all[train_idx])
    X_test = impute(X_test, med)
    pred = model.predict(X_test) if len(test_idx) else np.array([])

    metrics: dict[str, Any] = {
        "scoring_profile": scoring_profile(cfg),
        "scoring": scoring,
        "source": source,
        "split": split_note,
        "test_season": test_season,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "feature_cols": feature_cols,
        "seasons_in_data": sorted(map(int, np.unique(seasons))),
        "recency": {
            "half_life_seasons": half_life,
            "weeks_per_season": wps,
            "relative_mean_weight_by_season": weight_summary(seasons[train_idx], w_train),
            "ewm_halflife_games": (cfg.get("features", {}) or {}).get("ewm_halflife_games")
            if (cfg.get("features", {}) or {}).get("recent_form_enabled") else None,
        },
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

    if len(test_idx) and "zero_stat_appearance" in df.columns:
        # Comparable to pre-2026-09-25 metrics: only player-games that have a weekly stats row
        zs = pd.to_numeric(df.iloc[test_idx]["zero_stat_appearance"], errors="coerce").fillna(0).to_numpy() == 1
        keep = ~zs
        pos_k = df.iloc[test_idx]["position"].astype(str).str.upper().to_numpy()[keep]
        metrics["stats_rows_only"] = {
            "n": int(keep.sum()),
            "mae": float(mean_absolute_error(y_test[keep], pred[keep])),
            "rmse": float(mean_squared_error(y_test[keep], pred[keep]) ** 0.5),
            "r2": float(r2_score(y_test[keep], pred[keep])),
            "by_position": {
                p: {
                    "n": int((pos_k == p).sum()),
                    "mae": float(mean_absolute_error(y_test[keep][pos_k == p], pred[keep][pos_k == p])),
                    "rmse": float(mean_squared_error(y_test[keep][pos_k == p], pred[keep][pos_k == p]) ** 0.5),
                    "r2": float(r2_score(y_test[keep][pos_k == p], pred[keep][pos_k == p])),
                }
                for p in sorted(np.unique(pos_k))
            },
            "note": "excludes zero-stat snap-count appearances; comparable to earlier panels",
        }
        metrics["zero_stat_rows_in_test"] = int(zs.sum())

    if len(test_idx) and "position" in df.columns:
        pos = df.iloc[test_idx]["position"].astype(str).str.upper().to_numpy()
        metrics["by_position"] = {
            p: {
                "n": int((pos == p).sum()),
                "mae": float(mean_absolute_error(y_test[pos == p], pred[pos == p])),
                "rmse": float(mean_squared_error(y_test[pos == p], pred[pos == p]) ** 0.5),
                "r2": float(r2_score(y_test[pos == p], pred[pos == p])),
            }
            for p in sorted(np.unique(pos))
        }
    metrics["train_seasons"] = sorted(map(int, np.unique(seasons[train_idx])))

    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(model_out) if model_out else model_path(cfg)
    interval = residual_interval_table(pred, y_test) if len(test_idx) >= 200 else None
    if interval:
        metrics["interval_holdout_coverage"] = interval["holdout_coverage"]
    artifact = {
        "scoring_profile": scoring_profile(cfg),
        "interval": interval,
        "model": model,
        "feature_cols": feature_cols,
        "impute_medians": med,
        "scoring": scoring,
        "cfg_model": cfg.get("model", {}),
        "metrics": metrics,
    }

    if refit_full and len(test_idx):
        val_path = out_path.with_name(out_path.stem + "_val" + out_path.suffix)
        joblib.dump(artifact, val_path)
        metrics["validation_model_path"] = str(val_path)
        all_idx = np.where(labelled)[0]
        X_full_raw = X_all.iloc[all_idx][feature_cols].to_numpy(dtype=float)
        full_model, med_full, w_full = fit_weighted(
            cfg, X_full_raw, y.iloc[all_idx].to_numpy(dtype=float), seasons[all_idx], wk_all[all_idx]
        )
        sub = df.iloc[all_idx]
        last_season = int(sub["season"].max())
        metrics["refit_full"] = {
            "n_train": int(len(all_idx)),
            "train_seasons": sorted(map(int, sub["season"].unique())),
            "last_season_weeks": sorted(map(int, sub.loc[sub["season"] == last_season, "week"].unique())),
            "relative_mean_weight_by_season": weight_summary(seasons[all_idx], w_full),
            "note": "production artifact refit on all labelled rows; holdout metrics above come from the validation model",
        }
        artifact = {**artifact, "model": full_model, "impute_medians": med_full, "metrics": metrics}
    else:
        metrics["refit_full"] = None

    joblib.dump(artifact, out_path)
    metrics["model_path"] = str(out_path)

    if not write_reports:
        return metrics

    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    tag = profile_tag(cfg)
    report_path = reports_dir / f"train_metrics_{tag}.json"
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metrics["report_path"] = str(report_path)

    # Sample predictions CSV for the test fold
    if len(test_idx):
        meta = X_all.iloc[test_idx].copy()
        meta["actual_fp"] = y_test
        meta["predicted_fp"] = pred
        pred_path = reports_dir / f"test_predictions_{tag}.csv"
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
