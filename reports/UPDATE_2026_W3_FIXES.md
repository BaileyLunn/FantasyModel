# Bug-fix pass + depth/snap awareness + scoring profiles (built 2026-09-25)

Same branch / PR as the 2026-wk2 update. Same validation protocol: **train 2016–2024 → score 2025 REG**,
and **train 2016–2025 → score 2026 wk1–2**. Production models refit on everything through 2026 week 2.
All numbers are from real nflverse data; "before" = the pre-fix code at `132b81d` re-run on the same raw data
(half-PPR reproduced the documented 4.38 / 5.90 / 0.362 and 4.62 / 6.35 / 0.326 exactly).

## Fixes

| # | Fix | Where |
|---|-----|-------|
| 1 | **Vegas implied team total sign.** nflverse `spread_line` = expected *home* margin (corr +0.44 with `result`; home moneyline favorite ⇒ spread_line > 0 in 97% of games, rest are pick'ems). Implied = total/2 + own margin/2 (home: +spread, away: −spread). Old code gave favorites the underdog's total. Checked on 2026 wk1: DET −7 vs NO, total 49.5 → DET 28.25 / NO 21.25; CHI −3 at CAR → CHI 25.25 / CAR 22.25. Home implied vs actual home score: MAE 7.37 (fixed) vs 9.08 (old convention). | `features/baselines.py::implied_team_total` |
| 2a | **Depth chart + snap share features** (nflverse `depth_charts` 2016–2026, `snap_counts` 2015–2026 mapped pfr→gsis via `players`, 99.8% mapped): `depth_rank` (1 = top at position; 2025+ ESPN snapshots use the latest snapshot *before game day*), `snap_pct_last`, `snap_pct_roll3`, `snap_pct_season`, `snap_games_season`, `depth_listed`, `has_snap_history`. Strictly pre-game (as-of joins). Coverage: depth rank 95.6% of stats rows, prior snaps 98.5%. | `features/usage.py`, `features/pipeline.py` |
| 2b | **Zero-point games added to training** (root cause of backup over-projection: the panel only had games where a player recorded a stat, so a QB2 row only existed when he actually played). Added 15,275 snap-count appearances with no offensive stat and 16,784 rostered (weekly roster ACT/INA) player-weeks with no snap, as real 0-point rows (`zero_stat_appearance=1`, `zero_row_type`). Bye weeks and IR/RES excluded. This matches the projection population (every ACT roster player). | `features/usage.py`, `panel.py` (`features.zero_stat_rows`) |
| 2c | Projections keep every ACT player but add `depth_rank`, `likely_role` (depth ≤ QB1/RB2/WR3/TE2 or snap share ≥ 35%), snap columns; players on the nflverse report as **Out/IR** get `projection = 0` (the model never sees Out players, so it cannot learn it) with `model_projection` kept. | `project.py` |
| 3 | **Relocated-team codes.** Schedules/snaps/depth charts use OAK/SD (STL), weekly stats use LV/LAC/LA. Normalized everything to current codes → the 768 unjoined 2016–19 rows now join (0 rows without schedule). Travel distance uses Oakland (≤2019) / San Diego (2016) venue coordinates so normalization doesn't distort travel. | `teams.py`, `panel.py`, `features/travel.py` |
| 4a | **Same-week leakage: `opp_pos_fp_allowed`.** Row-wise `shift(1).expanding()` inside (defense, position) let a WR2 row see the WR1's same-game points vs that defense. Now uses strictly earlier weeks. | `features/history.py`, `baselines.py` |
| 4b | **Same-week leakage: `team_qb_fp_roll`.** Row-wise shift over a team's QB rows let the 2nd QB of a team-week (and every teammate's mean) see the other QB's same-game points. Now: one value per team-week (top QB), rolling mean of the previous 3 team games. | `features/teammates.py` |
| 4c | **Same-week leakage: injury analog** (position × severity expanding mean) — same pattern, now prior weeks only. | `features/injury.py` |
| 4d | **Special-teams-TD rows had `opponent_team == team`** in legacy player_stats (22 rows) → opponent re-derived from the schedule. | `panel.py` |
| 4e | Train-median imputation of the new usage features would have made unlisted players look like depth-rank-2 regulars; explicit fills instead (unlisted → rank 9, no snaps → 0). | `features/pipeline.py` |
| 5 | **Scoring profiles**: `scoring.profile: half_ppr | ppr | standard` (config) or `--scoring` CLI flag; artifacts `models/fantasy_hgb_{half,ppr,std}.joblib`, reports/projections suffixed. ESPN league 746996 confirmed **full PPR** (1/rec, pass TD 4, 0.04/pass yd, 0.1/rush-rec yd, no TE premium) → `--scoring ppr`. Config default left at half_ppr as requested. `update_week.py` retrains/projects half + PPR by default. | `config.py`, `cli.py`, `configs/default.yaml` |
| + | Residual-based **80% range** (`low_80`/`high_80`): empirical 10th/90th percentiles of 2025 holdout residuals of the validation model, binned by projection decile. Per-row `top_drivers` = single-feature median-substitution effects (not SHAP). | `train.py`, `project.py` |

(The earlier interception-naming issue was already fixed in `132b81d`; nothing further found there.)

## Metrics

"Stats rows" = the player-games that existed before (a weekly stat line) → directly comparable to the old numbers.
"All rows" adds the new 0-point rows (3,300 in 2025; 393 in 2026 wk1–2) — the population projections are made for.

### Half-PPR — MAE / RMSE / R²

| Slice | Before | After (stats rows) | After (all rows) |
|---|---|---|---|
| 2025 holdout (n=5,285 / 8,585) | 4.38 / 5.90 / 0.362 | **4.16 / 5.86 / 0.372** | 3.07 / 4.77 / 0.512 |
| 2025 QB (664) | 6.43 / 7.92 / 0.287 | 6.03 / 7.88 / 0.295 | 3.73 / 5.82 / 0.612 |
| 2025 RB (1,360) | 4.58 / 6.30 / 0.340 | 4.39 / 6.21 / 0.360 | 3.36 / 5.19 / 0.504 |
| 2025 TE (1,127) | 3.34 / 4.63 / 0.228 | 3.12 / 4.65 / 0.223 | 2.25 / 3.66 / 0.401 |
| 2025 WR (2,134) | 4.17 / 5.49 / 0.248 | 3.99 / 5.45 / 0.259 | 3.11 / 4.60 / 0.408 |
| 2026 wk1–2, trained thru 2025 (n=626 / 1,019) | 4.62 / 6.35 / 0.326 | **4.37 / 6.20 / 0.358** | 3.19 / 5.00 / 0.497 |
| 2026 QB (76) | 6.94 / 8.53 / 0.253 | 6.67 / 8.45 / 0.266 | 3.42 / 5.59 / 0.659 |
| 2026 RB (162) | 4.53 / 6.42 / 0.283 | 4.13 / 5.81 / 0.413 | 3.47 / 5.09 / 0.510 |
| 2026 TE (132) | 3.86 / 5.21 / 0.114 | 3.77 / 5.46 / 0.027 | 2.61 / 4.20 / 0.239 |
| 2026 WR (256) | 4.38 / 6.08 / 0.216 | 4.16 / 6.00 / 0.236 | 3.29 / 5.11 / 0.365 |

### Full PPR — MAE / RMSE / R²

| Slice | Before | After (stats rows) | After (all rows) |
|---|---|---|---|
| 2025 holdout | 4.79 / 6.38 / 0.347 | **4.58 / 6.35 / 0.353** | 3.41 / 5.20 / 0.512 |
| 2025 QB / RB / TE / WR MAE | 6.45 / 4.80 / 3.88 / 4.73 | 6.03 / 4.65 / 3.66 / 4.57 | 3.75 / 3.58 / 2.66 / 3.61 |
| 2026 wk1–2, trained thru 2025 | 5.08 / 6.90 / 0.303 | **4.80 / 6.72 / 0.338** | 3.52 / 5.44 / 0.496 |
| 2026 QB / RB / TE / WR MAE | 6.84 / 4.83 / 4.42 / 5.04 | 6.74 / 4.28 / 4.27 / 4.82 | 3.45 / 3.63 / 2.97 / 3.84 |

Usage-feature ablation (same final panel, usage features off; stats rows): half 2025 4.20 / 5.94 / 0.354, 2026 4.58 / 6.61 / 0.270;
PPR 2025 4.62 / 6.44 / 0.335, 2026 5.03 / 7.15 / 0.250 → depth/snap features account for most of the 2026 gain.
Intermediate development runs are in `reports/ablation/` (panel before zero rows: half 2025 4.31 / 5.79 / 0.386,
2026 4.51 / 6.11 / 0.376 — better R² on stats rows but backups stayed over-projected, see below).

### Roster-based backtest (what projections actually do): every ACT roster player, 2026 wk1–2, models trained thru 2025

n = 904 player-weeks (inactive / no-stat players count as 0). `reports/oos/roster_backtest_2026_w1_w2.json`.

| | Half before | Half after | PPR before | PPR after |
|---|---|---|---|---|
| MAE / RMSE / R² | 4.43 / 5.83 / 0.359 | **3.43 / 5.24 / 0.481** | 4.92 / 6.37 / 0.348 | **3.78 / 5.70 / 0.478** |
| mean bias (pred − actual) | +1.02 | −0.16 | +1.25 | −0.17 |

Mean projection vs actual by depth rank (half-PPR): rank 1 actual 12.18 — before 10.24, after 11.12; rank 2 actual 3.76 — 5.87 → 4.14;
rank 3 actual 2.39 — 4.14 → 2.30; rank ≥4/unlisted actual 1.22 — 3.63 → 1.36. **Backup QBs** (depth 2, n=59): actual 1.30, before 6.65, after 1.66.

Interval honesty: the 80% range covered 80.0% of 2025 holdout rows (in-sample for the table) and **76.7% (half) / 76.3% (PPR)**
of the 2026 wk1–2 roster backtest (out-of-sample) → slightly too narrow; treat as ~75%.

## Week 3 2026 projections

* `reports/projections_2026_w3_ppr.csv` (league scoring) and `reports/projections_2026_w3_half.csv` — 499 ACT QB/RB/WR/TE each.
* Columns: player, team, position, opponent_team, is_home, spread/total/implied_team_total, depth_rank, likely_role, snap shares,
  injury_status/injury_type (nflverse report as of fetch), projection, low_80, high_80, model_projection, availability_note,
  game_status, actual_fp, top_drivers.
* Thursday ATL@GB (30 rows) kept, predicted from pre-game features, `game_status=played` with `actual_fp`.
* Production models: 74,081 train rows for validation (2016–24), refit on 83,685 labelled rows (2016–2025 + 2026 wk1–2).

## Caveats

* Depth charts: 2016–24 weekly charts have no publish timestamp (assumed pre-game); 2025+ ESPN snapshots can be stale or odd
  (e.g. GB lists Josh Jacobs RB4) — `depth_rank` is only as good as the source.
* Rostered-no-snap rows: 2016–18 rosters don't mark game-day inactives (INA), 2019+ do; both are included so the population is
  "active roster before inactives are known", matching forward projections.
* The nflverse week-3 injury report was mostly Wednesday practice data at fetch time; final Friday designations are not in.
  Out/IR → 0 applies only to the nflverse report status.
* TE R² on 2026 stats rows fell (0.114 → 0.027 half; n=132) — noisy slice, but worth watching.
* Neutral-site games (London/Germany/etc., 65 in schedules) still get travel computed to the home team's stadium.
* `low_80/high_80` are empirical, unconditional on player; slightly under-cover out of sample.
