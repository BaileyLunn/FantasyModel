# Update: 2025 season + 2026 weeks 1–2 (built 2026-09-25)

Scoring: ESPN **half-PPR** (`configs/default.yaml`, unchanged). REG season, QB/RB/WR/TE.

## Data added (nflverse, real downloads — nothing synthesized)

| Season / week | Panel rows | Games | Weekly source |
|---|---:|---:|---|
| 2016–2024 | 45,715 | — | `nfl_data_py` legacy `player_stats` (unchanged raw) |
| 2025 REG wk1–18 | 5,285 | 272 | `stats_player/stats_player_week_2025.parquet` |
| 2026 wk1 | 310 | 16 | `stats_player/stats_player_week_2026.parquet` |
| 2026 wk2 | 316 | 16 | same |
| **Total** | **51,626** | | |

* Schedules / Vegas lines / roof / temp / wind: nflverse `games.csv` (2026 wk1–2 fully present, 32 games).
* Injuries: `injuries/injuries_2025.parquet` + `injuries_2026.parquet` (6,760 report rows; panel coverage 2025 17.5%, 2026 10.1%).
* The legacy `player_stats` release 404s from 2025 on, so `nfl_data_py.import_weekly_data` cannot load it;
  `fantasy_model/sources.py` falls back to `stats_player_week` and harmonizes columns
  (`team→recent_team`, `passing_interceptions→interceptions`, `sacks_suffered→sacks`) and drops
  pure special-teams/defensive appearances to match legacy row semantics.
  Cross-check on 2024 (both sources exist): 5,222 matched player-games, mean |Δ half-PPR| = 0.006
  (9 rows differ — stat corrections); 21 extra offensive rows in stats_player.
* 2026 week 3 Thursday game (ATL@GB, 2026-09-24) is in nflverse but **excluded** from training (`--max-week 2026:2`).

## 1. Current model (v1) scored out-of-sample on 2026 weeks 1–2 (before any refit)

v1 artifact reproduced bit-for-bit from `main` (train seasons < 2024, holdout 2024: MAE 4.398 / RMSE 5.890 / R² 0.380).
Features for 2026 rows are built on the full panel (2025 history feeds rolling/opponent features, all `shift(1)`).

| Slice | n | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| 2026 wk1 — all | 310 | 4.71 | 6.62 | 0.316 |
| 2026 wk1 — QB | 37 | 6.88 | 8.57 | 0.176 |
| 2026 wk1 — RB | 78 | 5.19 | 7.88 | 0.223 |
| 2026 wk1 — TE | 69 | 3.90 | 5.32 | 0.034 |
| 2026 wk1 — WR | 126 | 4.23 | 5.68 | 0.256 |
| 2026 wk2 — all | 316 | 4.59 | 6.09 | 0.331 |
| 2026 wk2 — QB | 39 | 7.12 | 8.50 | 0.305 |
| 2026 wk2 — RB | 84 | 3.88 | 4.67 | 0.359 |
| 2026 wk2 — TE | 63 | 4.05 | 5.21 | 0.154 |
| 2026 wk2 — WR | 130 | 4.55 | 6.43 | 0.185 |
| **2026 wk1–2 — all** | 626 | 4.65 | 6.36 | 0.324 |
| 2026 wk1–2 — QB | 76 | 7.00 | 8.53 | 0.252 |
| 2026 wk1–2 — RB | 162 | 4.51 | 6.42 | 0.284 |
| 2026 wk1–2 — TE | 132 | 3.97 | 5.27 | 0.094 |
| 2026 wk1–2 — WR | 256 | 4.39 | 6.07 | 0.217 |

v1 on the full 2025 season (also never seen): MAE 4.40 / RMSE 5.91 / R² 0.361 (n=5,285).

## 2. Retrain

Validation protocol (documented in `configs/default.yaml`): `test_season: 2025`, `refit_full: true`.
1. **Validation model** — train 2016–2024 (45,715 rows), score 2025 REG (5,285 rows) → `models/fantasy_hgb_val.joblib`.
2. **Production model** — same hyper-parameters refit on all 51,626 labelled rows (2016–2025 + 2026 wk1–2) →
   `models/fantasy_hgb.joblib`. No holdout by construction; its first true test is 2026 week 3.

2025 holdout:

| Slice | v1 (train 2016–23) | validation model (train 2016–24) |
|---|---|---|
| 2025 REG all (n=5,285) | 4.40 / 5.91 / 0.361 | 4.38 / 5.90 / 0.362 |
| 2025 QB (n=664) | 6.52 / 7.98 / 0.277 | 6.43 / 7.92 / 0.287 |
| 2025 RB (n=1,360) | 4.60 / 6.32 / 0.336 | 4.58 / 6.30 / 0.340 |
| 2025 TE (n=1,127) | 3.38 / 4.64 / 0.224 | 3.34 / 4.63 / 0.228 |
| 2025 WR (n=2,134) | 4.16 / 5.47 / 0.254 | 4.17 / 5.49 / 0.248 |

Effect of adding a season, tested on 2026 wk1–2 (out-of-sample for both; the 2016–25 model is the
production recipe minus the two 2026 weeks):

| Slice | v1 (train 2016–23) MAE / RMSE / R² | 2016–25 model MAE / RMSE / R² |
|---|---|---|
| 2026 wk1 (n=310) | 4.71 / 6.62 / 0.316 | 4.74 / 6.72 / 0.296 |
| 2026 wk2 (n=316) | 4.59 / 6.09 / 0.331 | 4.49 / 5.97 / 0.357 |
| **2026 wk1–2 (n=626)** | 4.65 / 6.36 / 0.324 | 4.62 / 6.35 / 0.326 |
| wk1–2 QB (n=76) | 7.00 / 8.53 / 0.252 | 6.94 / 8.53 / 0.253 |
| wk1–2 RB (n=162) | 4.51 / 6.42 / 0.284 | 4.53 / 6.42 / 0.283 |
| wk1–2 TE (n=132) | 3.97 / 5.27 / 0.094 | 3.86 / 5.21 / 0.114 |
| wk1–2 WR (n=256) | 4.39 / 6.07 / 0.217 | 4.38 / 6.08 / 0.216 |

**Read:** adding more history moves error only marginally (MAE −0.03 on 626 rows — within noise). Week-1 errors
are larger than week 2 for both models (stale features after the offseason: roster moves, rookies, new roles).

## Caveats

* Pre-existing feature issue (not changed here to keep v1 comparable): `implied_team_total` uses
  `(total − spread)/2` for the home team, but nflverse `spread_line` is positive when the **home** team is favored,
  so the feature is effectively the *opponent's* implied total. Trees can still use it, but it should be fixed in a follow-up.
* 2016–2019 panel rows with relocated-team abbreviations (OAK/SD/STL) have no schedule join (768 rows) — pre-existing.
* Injury coverage remains low (~18% of player-games; 10% so far in 2026). Outdoor temp/wind missing on ~35% of rows.
* Week 3 projections cover every roster `ACT` QB/RB/WR/TE (499 players) — the model has no depth chart, so backups
  with prior starting history (e.g. a former starter now QB2) can be over-projected. Filter with
  `n_prior_games_this_season > 0` for likely contributors.
* 2026 R² by position is noisy (n=63–130 per position-week).
