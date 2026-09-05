#!/usr/bin/env python3
"""Generate synthetic/sample player-game panel for offline CLI smoke tests."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fantasy_model.scoring import fantasy_points

RNG = np.random.default_rng(42)

TEAMS = ["KC", "BUF", "SF", "PHI", "DAL", "MIA", "BAL", "DET", "SEA", "GB", "NO", "MIN", "LAR", "CHI"]
# Include a dome home for weather nulling tests in sample
PLAYERS = [
    ("p_qb_kc", "Patrick Demo", "QB", "KC"),
    ("p_rb_kc", "Isiah Demo", "RB", "KC"),
    ("p_wr_kc", "Rashee Demo", "WR", "KC"),
    ("p_te_kc", "Travis Demo", "TE", "KC"),
    ("p_qb_buf", "Josh Demo", "QB", "BUF"),
    ("p_rb_buf", "James Demo", "RB", "BUF"),
    ("p_wr_buf", "Stefon Demo", "WR", "BUF"),
    ("p_qb_sf", "Brock Demo", "QB", "SF"),
    ("p_rb_sf", "CMC Demo", "RB", "SF"),
    ("p_wr_sf", "Deebo Demo", "WR", "SF"),
    ("p_qb_no", "Derek Demo", "QB", "NO"),
    ("p_wr_no", "Chris Demo", "WR", "NO"),
    ("p_qb_min", "Kirk Demo", "QB", "MIN"),
    ("p_rb_min", "Aaron Demo", "RB", "MIN"),
    ("p_qb_det", "Jared Demo", "QB", "DET"),
    ("p_wr_det", "Amon Demo", "WR", "DET"),
    ("p_qb_phi", "Jalen Demo", "QB", "PHI"),
    ("p_rb_phi", "Saquon Demo", "RB", "PHI"),
    ("p_wr_dal", "CeeDee Demo", "WR", "DAL"),
    ("p_te_phi", "Dallas Demo", "TE", "PHI"),
]

SCORING = {
    "pass_yd": 0.04,
    "pass_td": 4.0,
    "pass_int": -2.0,
    "rush_yd": 0.1,
    "rush_td": 6.0,
    "rec": 0.5,
    "rec_yd": 0.1,
    "rec_td": 6.0,
    "fumble_lost": -2.0,
    "two_pt": 2.0,
}


def _game_pairings(season: int, week: int) -> list[tuple[str, str]]:
    rotated = TEAMS[week % len(TEAMS) :] + TEAMS[: week % len(TEAMS)]
    pairs = []
    for i in range(0, len(rotated) - 1, 2):
        pairs.append((rotated[i], rotated[i + 1]))
    return pairs


def main() -> None:
    rows = []
    for season in (2022, 2023, 2024):
        for week in range(1, 11):  # 10 weeks keeps sample small but trainable
            pairs = _game_pairings(season, week)
            home_of = {}
            away_of = {}
            for home, away in pairs:
                home_of[home] = home
                home_of[away] = home
                away_of[home] = away
                away_of[away] = away
            gameday = pd.Timestamp(year=season, month=9, day=1) + pd.Timedelta(weeks=week - 1)
            for pid, name, pos, team in PLAYERS:
                if team not in home_of:
                    continue
                home_team = home_of[team]
                away_team = away_of[team]
                # Roof / weather
                if home_team in {"NO", "MIN", "DET", "DAL", "ATL", "IND", "LV", "ARI", "HOU"}:
                    roof = "dome"
                    temp, wind = np.nan, np.nan
                else:
                    roof = "outdoors"
                    temp = float(RNG.normal(55, 15))
                    wind = float(max(0, RNG.normal(8, 5)))
                # Stats by position
                if pos == "QB":
                    passing_yards = max(0, RNG.normal(250, 60))
                    passing_tds = max(0, int(RNG.poisson(1.6)))
                    interceptions = max(0, int(RNG.poisson(0.7)))
                    rushing_yards = max(0, RNG.normal(15, 20))
                    rushing_tds = 1 if RNG.random() < 0.1 else 0
                    receptions = receiving_yards = receiving_tds = 0
                    targets = carries = 0
                    rush_att = int(max(0, RNG.normal(3, 2)))
                elif pos == "RB":
                    passing_yards = passing_tds = interceptions = 0
                    rushing_yards = max(0, RNG.normal(70, 35))
                    rushing_tds = 1 if RNG.random() < 0.35 else 0
                    receptions = max(0, int(RNG.normal(3, 2)))
                    receiving_yards = max(0, RNG.normal(25, 20))
                    receiving_tds = 1 if RNG.random() < 0.08 else 0
                    targets = receptions + int(RNG.integers(0, 3))
                    carries = int(max(0, RNG.normal(14, 5)))
                    rush_att = carries
                elif pos == "WR":
                    passing_yards = passing_tds = interceptions = rushing_tds = 0
                    rushing_yards = max(0, RNG.normal(2, 5))
                    receptions = max(0, int(RNG.normal(5, 2)))
                    receiving_yards = max(0, RNG.normal(65, 30))
                    receiving_tds = 1 if RNG.random() < 0.25 else 0
                    targets = receptions + int(RNG.integers(0, 4))
                    carries = rush_att = int(RNG.integers(0, 2))
                else:  # TE
                    passing_yards = passing_tds = interceptions = rushing_yards = rushing_tds = 0
                    receptions = max(0, int(RNG.normal(4, 2)))
                    receiving_yards = max(0, RNG.normal(45, 25))
                    receiving_tds = 1 if RNG.random() < 0.2 else 0
                    targets = receptions + int(RNG.integers(0, 3))
                    carries = rush_att = 0

                injury = RNG.choice(["", "", "", "Questionable", "Doubtful", "Out"], p=[0.7, 0.1, 0.05, 0.08, 0.04, 0.03])
                injury_type = RNG.choice(["", "ankle", "hamstring", "shoulder", "knee"]) if injury else ""
                # Zero production if Out
                if str(injury).lower() == "out":
                    passing_yards = passing_tds = interceptions = 0
                    rushing_yards = rushing_tds = 0
                    receptions = receiving_yards = receiving_tds = 0
                    targets = carries = rush_att = 0

                spread = float(RNG.normal(0, 4))
                total = float(RNG.normal(46, 5))
                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "gameday": gameday.strftime("%Y-%m-%d"),
                        "player_id": pid,
                        "player_name": name,
                        "position": pos,
                        "team": team,
                        "home_team": home_team,
                        "away_team": away_team,
                        "roof": roof,
                        "temp": temp,
                        "wind": wind,
                        "passing_yards": round(passing_yards, 1),
                        "passing_tds": passing_tds,
                        "interceptions": interceptions,
                        "rushing_yards": round(rushing_yards, 1),
                        "rushing_tds": rushing_tds,
                        "receptions": receptions,
                        "receiving_yards": round(receiving_yards, 1),
                        "receiving_tds": receiving_tds,
                        "targets": targets,
                        "carries": carries,
                        "rushing_attempts": rush_att,
                        "fumbles_lost": 1 if RNG.random() < 0.05 else 0,
                        "injury_status": injury,
                        "injury_type": injury_type,
                        "spread_line": round(spread, 1),
                        "total_line": round(total, 1),
                        "game_id": f"{season}_{week}_{away_team}_{home_team}",
                    }
                )

    df = pd.DataFrame(rows)
    df["fantasy_points"] = fantasy_points(df, SCORING)
    out_dir = ROOT / "data" / "sample"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "player_games.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows -> {out}")
    print(f"seasons={sorted(df['season'].unique())} mean_fp={df['fantasy_points'].mean():.2f}")


if __name__ == "__main__":
    main()
