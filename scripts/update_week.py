#!/usr/bin/env python3
"""In-season weekly update: add week N, score the old model on it, retrain, project week N+1.

Run after week N's games (Tuesday is safest — nflverse stats are refreshed nightly and
stat corrections land early in the week):

  python scripts/update_week.py --season 2026 --week 3
  python scripts/update_week.py --season 2026 --week 3 --skip-download   # reuse data/raw
  python scripts/update_week.py --season 2026 --week 3 --no-train        # score + project only

Steps
 1. Refresh raw nflverse data for ``--season`` and rebuild the panel capped at week N
    (history seasons are read from data/raw; use --refresh-all to re-download them).
 For each scoring profile in --scoring (default half_ppr and ppr):
 2. Score the CURRENT production model on week N before refitting (true out-of-sample,
    as long as that model was trained through week N-1) -> reports/oos/prod_{tag}_on_{S}_w{N}.json
 3. Retrain (config: validation on model.test_season, refit_full on all labelled rows).
    Recency bias is ON by default for every profile: training sample weights decay with
    ``training.recency_half_life_seasons`` and the recent-form EWMA features use
    ``features.ewm_halflife_games`` (override with --recency-half-life / --ewm-halflife; 0 = off).
    Model strategy follows ``model.strategy`` / ``model.per_position`` (default: QB has its own model,
    RB/WR/TE pooled); ``--per-position none|all|QB,TE`` overrides it for the run.
 4. Project week N+1 -> reports/projections_{S}_w{N+1}_{tag}.csv  (tag = half | ppr | std)
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import joblib

import fetch_data
from fantasy_model.config import apply_per_position_override, load_config, model_path, profile_tag, set_scoring_profile
from fantasy_model.evaluate import evaluate_slice
from fantasy_model.model import per_position_flags
from fantasy_model.project import project_week
from fantasy_model.train import train_model


def apply_recency_overrides(cfg: dict, half_life: float | None, ewm: float | None) -> dict:
    """Recency weighting / EWMA come from the config (on by default); CLI values override."""
    cfg = copy.deepcopy(cfg)
    tr = cfg.setdefault("training", {})
    feats = cfg.setdefault("features", {})
    if half_life is not None:
        tr["recency_half_life_seasons"] = half_life if half_life > 0 else None
    if ewm is not None:
        feats["recent_form_enabled"] = ewm > 0
        if ewm > 0:
            feats["ewm_halflife_games"] = ewm
    return cfg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True, help="Most recent completed week to add")
    p.add_argument("--first-season", type=int, default=2016)
    p.add_argument("--config", default=None)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--refresh-all", action="store_true", help="Re-download every season, not just --season")
    p.add_argument("--no-train", action="store_true")
    p.add_argument("--no-project", action="store_true")
    p.add_argument("--scoring", nargs="+", default=["half_ppr", "ppr"],
                   help="Scoring profiles to score/retrain/project (default: half_ppr ppr)")
    p.add_argument("--recency-half-life", type=float, default=None,
                   help="Override training.recency_half_life_seasons (0 = uniform weights)")
    p.add_argument("--ewm-halflife", type=float, default=None,
                   help="Override features.ewm_halflife_games (0 = recent-form EWMA features off)")
    p.add_argument("--per-position", default=None,
                   help="Override model.strategy for the refit: none | all | e.g. QB,TE (default: config)")
    p.add_argument("--allow-partial", action="store_true", help="Proceed even if some week-N games are not in the stats yet")
    args = p.parse_args(argv)

    cfg = apply_recency_overrides(load_config(args.config), args.recency_half_life, args.ewm_halflife)
    cfg = apply_per_position_override(cfg, args.per_position)
    S, N = args.season, args.week
    seasons = [str(y) for y in range(args.first_season, S + 1)]
    fetch_args = ["--seasons", *seasons, "--max-week", f"{S}:{N}"]
    if args.config:
        fetch_args = ["--config", args.config, *fetch_args]
    if args.skip_download:
        fetch_args.append("--skip-download")
    elif not args.refresh_all:
        fetch_args += ["--refresh-seasons", str(S)]
    if args.ewm_halflife:
        fetch_args += ["--ewm-halflife", str(args.ewm_halflife)]
    rc = fetch_data.main(fetch_args)
    if rc:
        return rc

    meta = json.loads((Path(cfg["paths"]["processed_dir"]) / "fetch_meta.json").read_text())
    wk_meta = meta.get("latest_season_weeks", {}).get(str(N))
    if not wk_meta:
        print(f"Week {N} of {S} has no played rows in the panel yet; stopping.", file=sys.stderr)
        return 2
    import pandas as pd

    sched = pd.read_csv(Path(cfg["paths"]["raw_dir"]) / "schedules.csv", low_memory=False)
    n_sched = int(((sched["season"] == S) & (sched["week"] == N)).sum())
    wk_meta["scheduled_games"] = n_sched
    print(f"week {N}: {wk_meta}")
    if wk_meta["games"] < n_sched and not args.allow_partial:
        print(
            f"Week {N} incomplete in nflverse stats ({wk_meta['games']}/{n_sched} games); "
            "re-run after the week finishes or pass --allow-partial.",
            file=sys.stderr,
        )
        return 3

    summary: dict = {"season": S, "week": N, "week_rows": wk_meta, "profiles": {},
                     "recency": {"half_life_seasons": cfg["training"].get("recency_half_life_seasons"),
                                 "ewm_halflife_games": cfg["features"].get("ewm_halflife_games")
                                 if cfg["features"].get("recent_form_enabled") else None},
                     "per_position": per_position_flags(cfg)}
    print(f"recency settings: {summary['recency']}")
    reports = Path(cfg["paths"]["reports_dir"])
    base_cfg = cfg
    for prof in args.scoring:
        cfg = set_scoring_profile(base_cfg, prof)
        tag = profile_tag(cfg)
        ps: dict = {}
        summary["profiles"][tag] = ps
        prod = model_path(cfg)
        if prod.exists():
            art = joblib.load(prod)
            refit = (art.get("metrics") or {}).get("refit_full") or {}
            trained_through = (
                (max(refit.get("train_seasons", [0])), max(refit.get("last_season_weeks", [0])))
                if refit
                else (max((art.get("metrics") or {}).get("train_seasons", [0])), 99)
            )
            oos = trained_through[0] < S or (trained_through[0] == S and trained_through[1] < N)
            rep = evaluate_slice(S, [N], cfg=cfg, model_path=str(prod),
                                 out_path=str(reports / "oos" / f"prod_{tag}_on_{S}_w{N}.json"))
            ps["pre_refit_score"] = {"out_of_sample": bool(oos), **rep["overall"], "by_position": rep["by_position"]}
            print(tag, json.dumps(ps["pre_refit_score"], indent=2))
            if not args.no_train:
                shutil.copy2(prod, prod.with_name(f"{prod.stem}_before_{S}_w{N}.joblib"))

        if not args.no_train:
            m = train_model(cfg=cfg)
            ps["train"] = {k: m.get(k) for k in ("test_season", "n_train", "n_test", "mae", "rmse", "r2", "refit_full")}
            print(tag, json.dumps(ps["train"], indent=2))

        if not args.no_project:
            try:
                proj = project_week(S, N + 1, cfg=cfg)
                ps["projections"] = {"rows": int(len(proj)), "path": proj.attrs.get("out_path")}
                print(f"{tag} projections: {ps['projections']}")
            except ValueError as e:
                print(f"projection skipped: {e}")

    (reports / "oos").mkdir(parents=True, exist_ok=True)
    (reports / "oos" / f"update_{S}_w{N}.json").write_text(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
