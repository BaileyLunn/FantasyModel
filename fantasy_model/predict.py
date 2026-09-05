"""Predict expected fantasy points for a season/week."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from fantasy_model.config import load_config, scoring_dict
from fantasy_model.data import load_player_games
from fantasy_model.features.pipeline import build_feature_matrix


def predict_week(
    season: int,
    week: int,
    player: str | None = None,
    cfg: dict[str, Any] | None = None,
    prefer_sample: bool = False,
    config_path: str | None = None,
    model_path: str | None = None,
) -> pd.DataFrame:
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
    mask = (df["season"] == season) & (df["week"] == week)
    sub = df.loc[mask].copy()
    if player:
        name_cols = [c for c in ("player_name", "player_display_name", "player_id") if c in sub.columns]
        pl = player.lower()
        hit = pd.Series(False, index=sub.index)
        for c in name_cols:
            hit = hit | sub[c].astype(str).str.lower().str.contains(pl, na=False)
        sub = sub.loc[hit]
    if sub.empty:
        return pd.DataFrame()

    # Build features on full history for correct rolling stats, then slice
    X_all, _, _ = build_feature_matrix(df, scoring, cfg)
    X_sub = X_all.loc[sub.index, feature_cols].to_numpy(dtype=float)
    inds = np.where(np.isnan(X_sub))
    X_sub = X_sub.copy()
    X_sub[inds] = np.take(med, inds[1])
    preds = model.predict(X_sub)

    out = sub.copy()
    out["predicted_fp"] = preds
    out["data_source"] = source
    keep = [
        c
        for c in (
            "season", "week", "player_id", "player_name", "player_display_name",
            "team", "position", "predicted_fp", "fantasy_points", "data_source",
        )
        if c in out.columns
    ]
    result = out[keep].sort_values("predicted_fp", ascending=False)
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{season}_w{week}" + (f"_{player.replace(' ', '_')}" if player else "")
    out_path = reports_dir / f"predict_{tag}.csv"
    result.to_csv(out_path, index=False)
    return result


def main(
    season: int,
    week: int,
    player: str | None = None,
    prefer_sample: bool = False,
    config_path: str | None = None,
) -> None:
    df = predict_week(
        season=season,
        week=week,
        player=player,
        prefer_sample=prefer_sample,
        config_path=config_path,
    )
    if df.empty:
        print(json.dumps({"error": "no rows matched", "season": season, "week": week, "player": player}))
        return
    print(df.head(30).to_string(index=False))
    print(f"\n... {len(df)} rows")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    p.add_argument("--player", type=str, default=None)
    args = p.parse_args()
    main(args.season, args.week, args.player)
