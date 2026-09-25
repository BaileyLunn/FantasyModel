#!/usr/bin/env python3
"""Fetch nflverse weekly stats, schedules, injuries into data/raw and build data/processed/player_games.

Raw files are combined across seasons (``weekly.csv``, ``schedules.csv``, ``injuries.csv``).
Only the seasons being (re)downloaded are replaced; other seasons already on disk are kept,
so an in-season refresh only needs the current season.

Weekly source per season: ``nfl_data_py.import_weekly_data`` (legacy ``player_stats`` release,
2016–2024) with automatic fallback to nflverse ``stats_player/stats_player_week_{season}``
(2025+; the legacy release 404s). See ``fantasy_model/sources.py``.

Usage:
  python scripts/fetch_data.py --seasons 2016 ... 2026
  python scripts/fetch_data.py --seasons 2016 ... 2026 --refresh-seasons 2026        # in-season update
  python scripts/fetch_data.py --seasons 2016 ... 2026 --skip-download               # rebuild panel only
  python scripts/fetch_data.py --seasons 2016 ... 2026 --skip-download --max-week 2026:2   # as-of build
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from fantasy_model.config import load_config, scoring_dict
from fantasy_model.panel import build_panel
from fantasy_model import sources


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch nflverse fantasy panel")
    p.add_argument("--config", default=None)
    p.add_argument("--seasons", nargs="*", type=int, default=None, help="Seasons to include in the panel")
    p.add_argument("--max-seasons", type=int, default=None, help="Keep only the N most recent requested seasons")
    p.add_argument("--skip-download", action="store_true", help="Rebuild processed from existing raw CSVs")
    p.add_argument(
        "--refresh-seasons", nargs="*", type=int, default=None,
        help="Seasons to (re)download; default = all --seasons. Others are read from existing raw files.",
    )
    p.add_argument(
        "--max-week", action="append", default=[], metavar="SEASON:WEEK",
        help="Cap a season at WEEK (as-of builds / reproducible in-season snapshots). Repeatable.",
    )
    p.add_argument("--weekly-source", default="auto", choices=["auto", "nfl_data_py", "stats_player"])
    return p.parse_args(argv)


def _replace_seasons(existing_path: Path, new: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    if existing_path.exists():
        old = pd.read_csv(existing_path, low_memory=False)
        if "season" in old.columns:
            old = old[~pd.to_numeric(old["season"], errors="coerce").isin(seasons)]
        return pd.concat([old, new], ignore_index=True, sort=False)
    return new


def parse_max_week(specs: list[str]) -> dict[int, int]:
    caps: dict[int, int] = {}
    for spec in specs:
        s, w = spec.split(":")
        caps[int(s)] = int(w)
    return caps


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    start = int(cfg["seasons"]["target_start"])
    end = int(cfg["seasons"]["target_end"])
    seasons = args.seasons or list(range(start, end + 1))
    if args.max_seasons:
        seasons = sorted(seasons)[-args.max_seasons :]
    refresh = [] if args.skip_download else (args.refresh_seasons if args.refresh_seasons is not None else seasons)
    caps = parse_max_week(args.max_week)

    weekly_path = raw_dir / "weekly.csv"
    schedules_path = raw_dir / "schedules.csv"
    inj_path = raw_dir / "injuries.csv"
    meta: dict = {
        "requested_seasons": seasons,
        "refreshed_seasons": refresh,
        "downloaded_seasons": [],
        "weekly_sources": {},
        "max_week": {str(k): v for k, v in caps.items()},
        "errors": [],
    }

    if refresh:
        frames = []
        for y in refresh:
            try:
                df, src = sources.load_weekly_season(y, prefer=args.weekly_source)
                frames.append(df)
                meta["downloaded_seasons"].append(y)
                meta["weekly_sources"][str(y)] = src
                print(f"  weekly {y}: rows={len(df)} via {src}")
            except Exception as e:  # noqa: BLE001
                meta["errors"].append(f"weekly {y}: {e}")
                print(f"Weekly {y} unavailable: {e}", file=sys.stderr)
        if frames:
            got = meta["downloaded_seasons"]
            weekly_all = _replace_seasons(weekly_path, pd.concat(frames, ignore_index=True, sort=False), got)
            weekly_all.to_csv(weekly_path, index=False)

        try:
            sched = sources.load_schedules(refresh)
            _replace_seasons(schedules_path, sched, refresh).to_csv(schedules_path, index=False)
            print(f"  schedules rows={len(sched)} for {refresh}")
        except Exception as e:  # noqa: BLE001
            meta["errors"].append(f"schedules: {e}")
            print(f"Schedules fetch failed: {e}", file=sys.stderr)

        inj_frames = []
        for y in refresh:
            try:
                inj_frames.append(sources.load_injuries_season(y))
            except Exception as e:  # noqa: BLE001
                meta["errors"].append(f"injuries {y}: {e}")
                print(f"Injuries {y} skipped: {e}")
        if inj_frames:
            inj_new = pd.concat(inj_frames, ignore_index=True, sort=False)
            inj_seasons = sorted(map(int, inj_new["season"].dropna().unique()))
            _replace_seasons(inj_path, inj_new, inj_seasons).to_csv(inj_path, index=False)
            print(f"  injuries rows={len(inj_new)} for {inj_seasons}")

    if not weekly_path.exists():
        print("No weekly.csv present", file=sys.stderr)
        return 1

    weekly = pd.read_csv(weekly_path, low_memory=False)
    weekly = weekly[pd.to_numeric(weekly["season"], errors="coerce").isin(seasons)]
    for s, w in caps.items():
        weekly = weekly[~((weekly["season"] == s) & (pd.to_numeric(weekly["week"], errors="coerce") > w))]
    schedules = pd.read_csv(schedules_path, low_memory=False) if schedules_path.exists() else pd.DataFrame()
    injuries = pd.read_csv(inj_path, low_memory=False) if inj_path.exists() else None

    panel = build_panel(weekly, schedules, injuries, scoring_dict(cfg))

    out_csv = processed_dir / "player_games.csv"
    panel.to_csv(out_csv, index=False)
    try:
        panel.to_parquet(processed_dir / "player_games.parquet", index=False)
    except Exception:
        pass

    meta["n_rows"] = int(len(panel))
    meta["seasons_present"] = sorted(map(int, panel["season"].dropna().unique())) if "season" in panel.columns else []
    counts = panel.groupby("season").size()
    meta["rows_per_season"] = {str(int(k)): int(v) for k, v in counts.items()}
    latest = int(panel["season"].max())
    wk = panel[panel["season"] == latest].groupby("week").agg(rows=("player_id", "size"), games=("game_id", "nunique"))
    meta["latest_season_weeks"] = {
        str(int(w)): {"rows": int(r.rows), "games": int(r.games)} for w, r in wk.iterrows()
    }
    meta["processed_path"] = str(out_csv)
    (processed_dir / "fetch_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
