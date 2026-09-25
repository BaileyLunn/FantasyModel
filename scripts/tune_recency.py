#!/usr/bin/env python3
"""Grid over recency settings: training sample-weight half-life x recent-form EWMA half-life.

For every (ewm_halflife_games, recency_half_life_seasons) pair and scoring profile:
  * 2025 holdout: train 2016–2024 (weights referenced to the last training week), score 2025 REG.
  * 2026 in-season: train 2016–2025, score 2026 weeks in the panel.
  * Roster backtest: same thru-2025 model, every ACT roster QB/RB/WR/TE of 2026 weeks 1..N
    (projection rows built exactly like ``project_week``; Out/IR -> 0; no-stat players = 0).
Metrics on stats rows (comparable to earlier reports) and all rows. Writes
reports/recency_tuning_<tag>.json. No production artifacts are touched.

  python scripts/tune_recency.py --scoring ppr
  python scripts/tune_recency.py --scoring ppr half_ppr --ewm off 3 --half-lives none 3
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fantasy_model.config import load_config, profile_tag, scoring_dict, set_scoring_profile
from fantasy_model.data import load_player_games
from fantasy_model.features.pipeline import build_feature_matrix
from fantasy_model.features.usage import load_snaps, prior_snap_features
from fantasy_model.project import actual_points, projection_features, ruled_out_mask
from fantasy_model.train import fit_weighted, impute


def _m(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"n": int(len(y)), "mae": round(float(mean_absolute_error(y, p)), 4),
            "rmse": round(float(mean_squared_error(y, p) ** 0.5), 4), "r2": round(float(r2_score(y, p)), 4),
            "bias": round(float(np.mean(p - y)), 4)}


def _mean_runs(runs: list[dict]) -> dict:
    """Average metric dicts over seeds (same nested structure, 'seed' dropped)."""
    def rec(ds):
        if isinstance(ds[0], dict):
            return {k: rec([d[k] for d in ds]) for k in ds[0] if k != "seed"}
        if isinstance(ds[0], int):
            return ds[0]
        return round(float(np.mean(ds)), 4)
    return rec(runs)


def _parse_hl(v: str):
    return None if v.lower() in ("none", "off", "0") else float(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scoring", nargs="+", default=["ppr"])
    ap.add_argument("--ewm", nargs="+", default=["off", "2", "3", "5"], help="EWMA half-life in games ('off' = no recent-form features)")
    ap.add_argument("--half-lives", nargs="+", default=["none", "1", "2", "3", "4", "6"], help="sample-weight half-life in seasons")
    ap.add_argument("--roster-weeks", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--suffix", default="")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="model.random_state values (HGB early-stopping split); metrics are averaged over seeds")
    a = ap.parse_args(argv)

    base = load_config()
    raw_dir = Path(base["paths"]["raw_dir"])
    panel0, _ = load_player_games(base)
    panel0 = panel0.reset_index(drop=True)
    seasons_all = sorted(map(int, panel0["season"].unique()))
    snaps = load_snaps(raw_dir, list(range(min(seasons_all) - 1, max(seasons_all) + 1)))

    for prof in a.scoring:
        results = []
        for ewm in a.ewm:
            ewm_hl = _parse_hl(ewm)
            cfg = copy.deepcopy(set_scoring_profile(base, prof))
            cfg.setdefault("features", {})["recent_form_enabled"] = ewm_hl is not None
            if ewm_hl is not None:
                cfg["features"]["ewm_halflife_games"] = ewm_hl
            scoring = scoring_dict(cfg)
            panel = panel0.copy()
            if ewm_hl is not None:
                sf = prior_snap_features(snaps, panel, ewm_halflife=ewm_hl)[["player_id", "season", "week", "snap_pct_ewm"]]
                panel = panel.drop(columns=["snap_pct_ewm"], errors="ignore")
                panel["player_id"] = panel["player_id"].astype(str)
                panel = panel.merge(sf, on=["player_id", "season", "week"], how="left")
            t0 = time.time()
            X_all, y, fcols = build_feature_matrix(panel, scoring, cfg)
            season = panel["season"].to_numpy()
            week = panel["week"].to_numpy()
            lab = y.notna().to_numpy()
            zs = pd.to_numeric(panel["zero_stat_appearance"], errors="coerce").fillna(0).to_numpy() == 1
            Xn = X_all[fcols].to_numpy(dtype=float)
            yn = y.to_numpy(dtype=float)
            fwd = []
            for w in a.roster_weeks:
                _, out, feats = projection_features(2026, w, cfg, refresh_roster=False, hist=panel)
                act, played = actual_points(out, raw_dir, 2026, w, scoring)
                keep = played.to_numpy() & act.notna().to_numpy()
                fwd.append((feats[fcols].to_numpy(dtype=float)[keep], act.to_numpy()[keep], ruled_out_mask(out).to_numpy()[keep]))
            print(f"[{prof} ewm={ewm}] features built in {time.time() - t0:.0f}s ({len(fcols)} cols)", flush=True)

            for hl in a.half_lives:
                hl_v = _parse_hl(hl)
                runs = []
                for seed in (a.seeds or [cfg.get("model", {}).get("random_state", 42)]):
                    c2 = copy.deepcopy(cfg)
                    c2.setdefault("training", {})["recency_half_life_seasons"] = hl_v
                    c2.setdefault("model", {})["random_state"] = int(seed)
                    r = {"seed": int(seed)}
                    # 2025 holdout
                    tr = np.where(lab & (season < 2025))[0]
                    te = np.where(lab & (season == 2025))[0]
                    model, med, _ = fit_weighted(c2, Xn[tr], yn[tr], season[tr], week[tr])
                    p = model.predict(impute(Xn[te], med))
                    k = ~zs[te]
                    r["holdout_2025"] = {"stats_rows": _m(yn[te][k], p[k]), "all_rows": _m(yn[te], p)}
                    # 2026 in-season + roster backtest
                    tr = np.where(lab & (season < 2026))[0]
                    te = np.where(lab & (season == 2026))[0]
                    model, med, _ = fit_weighted(c2, Xn[tr], yn[tr], season[tr], week[tr])
                    p = model.predict(impute(Xn[te], med))
                    k = ~zs[te]
                    r["in_season_2026"] = {"stats_rows": _m(yn[te][k], p[k]), "all_rows": _m(yn[te], p)}
                    ps, ys = [], []
                    for Xf, af, ro in fwd:
                        pf = model.predict(impute(Xf, med))
                        ps.append(np.where(ro, 0.0, pf))
                        ys.append(af)
                    r["roster_backtest_2026"] = _m(np.concatenate(ys), np.concatenate(ps))
                    runs.append(r)
                row = {"ewm_halflife_games": ewm_hl, "recency_half_life_seasons": hl_v, **_mean_runs(runs)}
                if len(runs) > 1:
                    row["per_seed"] = runs
                results.append(row)
                s = lambda d: f"{d['mae']:.3f}/{d['rmse']:.3f}/{d['r2']:.3f}"
                print(f"  {prof} ewm={ewm:>4} hl={hl:>4}  2025 {s(row['holdout_2025']['stats_rows'])}  "
                      f"2026 {s(row['in_season_2026']['stats_rows'])}  roster {s(row['roster_backtest_2026'])}", flush=True)
        out = Path(base["paths"]["reports_dir"]) / f"recency_tuning_{profile_tag(set_scoring_profile(base, prof))}{a.suffix}.json"
        out.write_text(json.dumps({"scoring_profile": prof, "grid": results,
                                   "note": "MAE/RMSE/R2; stats_rows comparable to README tables; roster backtest = all ACT players 2026 wk" + ",".join(map(str, a.roster_weeks))}, indent=2))
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
