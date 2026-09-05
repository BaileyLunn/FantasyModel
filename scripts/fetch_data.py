#!/usr/bin/env python3
"""Fetch nflverse weekly + schedules into data/raw and build data/processed/player_games.

Target seasons: 2016–2025 (inclusive) from config. Missing years are skipped with a note.

Usage:
  python scripts/fetch_data.py
  python scripts/fetch_data.py --seasons 2022 2023 2024
  python scripts/fetch_data.py --max-seasons 3   # quick subset from recent years
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
from fantasy_model.scoring import ensure_fantasy_points


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch nflverse fantasy panel")
    p.add_argument("--config", default=None)
    p.add_argument("--seasons", nargs="*", type=int, default=None)
    p.add_argument("--max-seasons", type=int, default=None, help="Keep only the N most recent requested seasons")
    p.add_argument("--skip-download", action="store_true", help="Rebuild processed from existing raw CSVs")
    return p.parse_args()


def main() -> int:
    args = parse_args()
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

    weekly_path = raw_dir / "weekly.csv"
    schedules_path = raw_dir / "schedules.csv"
    meta = {"requested_seasons": seasons, "downloaded_seasons": [], "errors": []}

    if not args.skip_download:
        try:
            import nfl_data_py as nfl
        except ImportError:
            print("nfl_data_py not installed. pip install nfl-data-py", file=sys.stderr)
            return 1

        print(f"Fetching weekly data for seasons {seasons} ...")
        try:
            weekly = nfl.import_weekly_data(seasons)
            weekly.to_csv(weekly_path, index=False)
            meta["downloaded_seasons"] = sorted(map(int, weekly["season"].dropna().unique()))
            print(f"  weekly rows={len(weekly)} -> {weekly_path}")
        except Exception as e:  # noqa: BLE001
            meta["errors"].append(f"weekly: {e}")
            print(f"Weekly fetch failed: {e}", file=sys.stderr)
            if not weekly_path.exists():
                return 1

        print(f"Fetching schedules for seasons {seasons} ...")
        try:
            schedules = nfl.import_schedules(seasons)
            schedules.to_csv(schedules_path, index=False)
            print(f"  schedules rows={len(schedules)} -> {schedules_path}")
        except Exception as e:  # noqa: BLE001
            meta["errors"].append(f"schedules: {e}")
            print(f"Schedules fetch failed: {e}", file=sys.stderr)

        # Optional injuries (may be incomplete historically)
        try:
            injuries = nfl.import_injuries(seasons)
            inj_path = raw_dir / "injuries.csv"
            injuries.to_csv(inj_path, index=False)
            print(f"  injuries rows={len(injuries)} -> {inj_path}")
        except Exception as e:  # noqa: BLE001
            meta["errors"].append(f"injuries: {e}")
            print(f"Injuries fetch skipped/failed: {e}")

    if not weekly_path.exists():
        print("No weekly.csv present", file=sys.stderr)
        return 1

    weekly = pd.read_csv(weekly_path)
    if schedules_path.exists():
        schedules = pd.read_csv(schedules_path)
    else:
        schedules = pd.DataFrame()

    # Normalize join keys
    if "recent_team" in weekly.columns and "team" not in weekly.columns:
        weekly = weekly.rename(columns={"recent_team": "team"})
    if "player_display_name" in weekly.columns and "player_name" not in weekly.columns:
        weekly["player_name"] = weekly["player_display_name"]

    if not schedules.empty:
        sched_cols = [
            c
            for c in (
                "game_id", "season", "week", "gameday", "home_team", "away_team",
                "roof", "temp", "wind", "spread_line", "total_line",
                "home_moneyline", "away_moneyline",
            )
            if c in schedules.columns
        ]
        sched = schedules[sched_cols].drop_duplicates()
        # Join weekly to schedule on season/week + team in {home, away}
        # Prefer game_id if weekly has it
        if "game_id" in weekly.columns and "game_id" in sched.columns:
            panel = weekly.merge(sched, on="game_id", how="left", suffixes=("", "_sched"))
            # Fill season/week if duplicated
            for c in ("season", "week"):
                if f"{c}_sched" in panel.columns:
                    panel[c] = panel[c].fillna(panel[f"{c}_sched"])
                    panel = panel.drop(columns=[f"{c}_sched"])
        else:
            home = sched.copy()
            away = sched.copy()
            home["team"] = home["home_team"]
            away["team"] = away["away_team"]
            team_sched = pd.concat([home, away], ignore_index=True)
            keys = [c for c in ("season", "week", "team") if c in weekly.columns and c in team_sched.columns]
            panel = weekly.merge(team_sched, on=keys, how="left", suffixes=("", "_sched"))
    else:
        panel = weekly

    # Injuries merge (best-effort)
    inj_path = raw_dir / "injuries.csv"
    if inj_path.exists():
        inj = pd.read_csv(inj_path)
        id_left = "player_id" if "player_id" in panel.columns else None
        id_right = "gsis_id" if "gsis_id" in inj.columns else ("player_id" if "player_id" in inj.columns else None)
        if id_left and id_right:
            cols = [id_right]
            for c in (
                "season", "week", "report_status", "report_primary_injury",
                "practice_status", "practice_primary_injury",
                "injury_status", "injury_type",
            ):
                if c in inj.columns and c not in cols:
                    cols.append(c)
            inj2 = inj[cols].copy()
            if id_right != id_left:
                inj2 = inj2.rename(columns={id_right: id_left})
            for c in ("season", "week"):
                if c in inj2.columns:
                    inj2[c] = pd.to_numeric(inj2[c], errors="coerce")
                if c in panel.columns:
                    panel[c] = pd.to_numeric(panel[c], errors="coerce")
            # Prefer official report_status; fall back to practice designation
            if "report_status" in inj2.columns:
                inj2["injury_status"] = inj2["report_status"]
            if "injury_status" in inj2.columns and "practice_status" in inj2.columns:
                inj2["injury_status"] = inj2["injury_status"].fillna(inj2["practice_status"])
            if "report_primary_injury" in inj2.columns:
                inj2["injury_type"] = inj2["report_primary_injury"]
            if "injury_type" in inj2.columns and "practice_primary_injury" in inj2.columns:
                inj2["injury_type"] = inj2["injury_type"].fillna(inj2["practice_primary_injury"])
            keep = [c for c in (id_left, "season", "week", "injury_status", "injury_type") if c in inj2.columns]
            inj2 = inj2[keep].drop_duplicates([c for c in (id_left, "season", "week") if c in keep])
            merge_keys = [c for c in (id_left, "season", "week") if c in panel.columns and c in inj2.columns]
            panel = panel.merge(inj2, on=merge_keys, how="left")

    scoring = scoring_dict(cfg)
    panel = ensure_fantasy_points(panel, scoring)

    # Keep skill + QB; regular season only (playoff rest/travel differ)
    if "position" in panel.columns:
        panel = panel[panel["position"].astype(str).str.upper().isin(["QB", "RB", "WR", "TE"])].copy()
    if "season_type" in panel.columns:
        panel = panel[panel["season_type"].astype(str).str.upper().eq("REG")].copy()

    out_csv = processed_dir / "player_games.csv"
    panel.to_csv(out_csv, index=False)
    try:
        panel.to_parquet(processed_dir / "player_games.parquet", index=False)
    except Exception:
        pass

    meta["n_rows"] = int(len(panel))
    meta["seasons_present"] = sorted(map(int, panel["season"].dropna().unique())) if "season" in panel.columns else []
    meta["processed_path"] = str(out_csv)
    (processed_dir / "fetch_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
