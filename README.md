# FantasyModel

Predict **expected fantasy points** for NFL player-games (pre-game features only) using nflverse history.

**Scoring profiles:** `half_ppr` (config default), `ppr`, `standard` — pick with `--scoring` (e.g. `python -m fantasy_model --scoring ppr project ...`)
or `scoring.profile` in `configs/default.yaml`. ESPN league 746996 is **full PPR** → use `--scoring ppr`.
Each profile has its own artifact: `models/fantasy_hgb_{half,ppr,std}.joblib` (+ `_val`).

**Panel:** 2016–2025 REG + 2026 weeks 1–2, QB/RB/WR/TE: 51,626 stat-line player-games plus 32,059 real 0-point rows
(snap-count appearances with no stat, and active-roster weeks with no snap) = 83,685 rows.
2016–2024 weekly stats come from `nfl_data_py`; 2025+ from nflverse `stats_player_week` (`fantasy_model/sources.py`).
Depth charts, snap counts, weekly rosters and the players id map come from nflverse-data releases (`fantasy_model/features/usage.py`).

## Results (real data, leakage-aware)

Validation: train seasons `< 2025`, test **2025 REG**; in-season check: train `< 2026`, test 2026 wk1–2.
Production artifacts are refit on every labelled row through the latest completed week.

| MAE / RMSE / R² (stat-line rows, comparable to earlier versions) | half-PPR | full PPR |
|---|---|---|
| 2025 holdout (n=5,285) | 4.15 / 5.81 / 0.383 | 4.55 / 6.29 / 0.365 |
| 2026 wk1–2 (n=626) | 4.30 / 6.14 / 0.369 | 4.77 / 6.69 / 0.345 |
| Roster backtest 2026 wk1–2, all ACT players (n=904; 5-seed mean) | 3.32 / 5.16 / 0.497 | 3.73 / 5.67 / 0.484 |

**Per-position models (2026-09-25):** QB now has its own model with position-specific features (rush share, pass volume, air yards, ...).
RB/WR/TE stay on the pooled model (`model.strategy: hybrid`, `model.per_position`). QB beat pooled on the 2024 inner validation and on the 2025 holdout
beyond seed noise (PPR QB RMSE 5.831 → 5.776, MAE 3.763 → 3.745). On 2026 wk1–2 (only 180 QB rows) RMSE was slightly higher, within noise, and MAE lower.
It is a marginal gain; full tables are in `reports/PER_POSITION_2026_09_25.md` (`scripts/per_position_experiment.py`). Before (pooled only):
PPR 4.57 / 6.32 / 0.359, 4.76 / 6.67 / 0.349, roster 3.75 / 5.66 / 0.486 (5-seed); half 4.16 / 5.84 / 0.377, 4.29 / 6.11 / 0.376, roster 3.34 / 5.15 / 0.500.

With recency bias (2026-09-25): training sample weights decay with a 4-season half-life (`training.recency_half_life_seasons`)
and EWMA recent-form features use a 5-game half-life (`features.ewm_halflife_games`). Before recency (same panel):
PPR 4.58 / 6.35 / 0.353, 4.80 / 6.72 / 0.338, roster 3.78 / 5.70 / 0.478; half 4.16 / 5.86 / 0.372, 4.37 / 6.20 / 0.358,
roster 3.43 / 5.24 / 0.481. Tuning grid and rationale: `reports/RECENCY_2026_09_25.md` (`scripts/tune_recency.py`).

Before the 2026-09-25 fix pass (half-PPR): 2025 4.38 / 5.90 / 0.362; 2026 4.62 / 6.35 / 0.326; roster backtest 4.43 / 5.83 / 0.359.
Details, ablations and the list of fixes: `reports/UPDATE_2026_W3_FIXES.md`. Earlier update: `reports/UPDATE_2026_W2.md`.

Reports: `reports/train_metrics_{half,ppr}.json`, `reports/validation_{half,ppr}.json`, `reports/eval_report_{half,ppr}.json`, `reports/oos/*.json`.

> Same-game usage is **not** a feature. Rolling usage / FP / snap share use prior games; group aggregates
> (opponent allowances, injury analogs, team QB form) use strictly earlier *weeks* so validation reflects pre-game information.

## Layout

```
FantasyModel/
  configs/default.yaml
  fantasy_model/           # scoring, features/*, train, evaluate, predict, cli
  fantasy_model/sources.py  # nflverse URLs + stats_player -> legacy schema
  fantasy_model/panel.py    # weekly + schedule + injuries join
  fantasy_model/project.py  # forward (upcoming-week) projections (+ 80% range, drivers)
  fantasy_model/teams.py    # OAK/SD/STL -> LV/LAC/LA normalization
  fantasy_model/features/usage.py    # depth charts, snap counts, zero-point rows
  fantasy_model/features/history.py  # prior-weeks group aggregates (no same-week leakage)
  scripts/validate_models.py         # 2025 holdout + 2026 in-season validation per profile
  scripts/tune_recency.py            # grid: sample-weight half-life x EWMA half-life (+ roster backtest)
  scripts/per_position_experiment.py # pooled vs per-position vs hybrid (+ position features), 5 seeds, 3 checks
  fantasy_model/weights.py           # recency sample weights
  scripts/fetch_data.py
  scripts/update_week.py    # in-season: add week N, score, retrain, project N+1
  scripts/make_sample_data.py
  data/raw/                # gitignored nflverse downloads
  data/processed/          # gitignored player_games panel
  data/sample/             # offline smoke fixtures
  models/fantasy_hgb_{half,ppr}.joblib
  reports/
  tests/
```

## Setup

```bash
cd /workspace/FantasyModel
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
# nfl_data_py pins old numpy/pandas; deps above are intentional on Python 3.13
```

## Fetch → train → predict

```bash
# Full window (2016–2024 via nfl_data_py; 2025+ via stats_player fallback)
python scripts/fetch_data.py --seasons 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026
# Rebuild panel only from existing raw files, as of a given week:
python scripts/fetch_data.py --skip-download --seasons 2016 ... 2026 --max-week 2026:2

python -m fantasy_model --scoring ppr train               # validate on 2025, refit on everything
python -m fantasy_model --scoring half_ppr train
python -m fantasy_model --scoring ppr evaluate            # 2025 holdout (uses models/fantasy_hgb_ppr_val.joblib)
python -m fantasy_model --scoring ppr evaluate --season 2026 --weeks 1 2
python -m fantasy_model --scoring ppr predict --season 2024 --week 10   # played weeks (shows actuals)
python -m fantasy_model --scoring ppr project --season 2026 --week 3    # -> reports/projections_2026_w3_ppr.csv
python scripts/validate_models.py --scoring half_ppr ppr --ablate-usage # validation tables
```

## In-season weekly update

After week N finishes (Tuesday is safest — stat corrections):

```bash
python scripts/update_week.py --season 2026 --week 3
```

1. re-downloads only the current season's nflverse stats/schedule/injuries and rebuilds the panel capped at week N
   (refuses if nflverse doesn't yet have every week-N game; `--allow-partial` to override);
2. scores the current production model on week N **before** refitting → `reports/oos/prod_on_2026_w3.json`
   (flagged `out_of_sample` when the model was trained through week N−1); the old artifact is kept as
   `models/fantasy_hgb_before_2026_w3.joblib`;
3. retrains (2025 validation + full refit) with recency sample weights + EWMA recent-form features on by default
   (`--recency-half-life 0 --ewm-halflife 0` turns them off for a run);
4. projects week N+1 → `reports/projections_2026_w4_{half,ppr}.csv`.
Steps 2–4 run for each profile in `--scoring` (default `half_ppr ppr`).

### Offline smoke (synthetic)

```bash
python scripts/make_sample_data.py
python -m fantasy_model --sample train --model-out /tmp/sample.joblib --no-reports   # don't clobber real models
python -m fantasy_model --sample evaluate --model /tmp/sample.joblib
python -m fantasy_model --sample predict --season 2024 --week 5
```

## Feature families

| Family | Notes |
|--------|-------|
| Injury | Severity from report/practice status + expanding FP analogs by position×severity (statistical only) |
| Weather | Temp/wind; **nulled** for dome / closed / retractable-closed |
| Travel | Haversine miles team home → game stadium; `is_home` |
| Timezone | `tz_vs_home`, `tz_vs_prev_game` (fixed US offsets; DST ignored) |
| Teammates | Prior QB FP, backup-QB heuristic, **rolling prior** target/rush shares |
| Recent form | EWMA (5-game half-life) of prior fantasy points, targets, carries, receptions, snap share (`fp_ewm`, …, `snap_pct_ewm`) |
| Baselines | Opp FP allowed vs position (prior weeks), rest days, rolling/season FP, Vegas spread/total → implied team total (total/2 ± home spread/2) |
| Usage | Depth-chart rank (pre-game), prior snap share (last / 3-game / season), games with snaps this season |
| Position-specific (per-position models only) | EWMA of prior rush share, rush yds, target share, air-yards share, receiving air yards, pass attempts, pass air yards |

Model: `HistGradientBoostingRegressor` (optional `lightgbm` / `xgboost` via config).

Strategy (`model.strategy`): `pooled` = one model with `position_code` as a feature; `per_position` = one model per position;
`hybrid` = positions flagged in `model.per_position` get their own model (hyper-parameters in `model.position_params`, which can be per scoring profile)
and the rest use the pooled model. Default: hybrid with QB separate. Artifacts hold a `PositionRouterModel` (`fantasy_model/model.py`) that routes rows by
position, so `evaluate`, `predict`, `project`, `top_drivers` and `update_week.py` work unchanged. Projection 80% ranges use per-position residual tables.
Override for one run: `python -m fantasy_model --per-position none|all|QB,TE train`, or `scripts/update_week.py ... --per-position none`.
Position-specific features (`features.position_specific_enabled`) feed only the per-position models.

## Tests

```bash
pytest -q
```

Dome weather nulling, timezone deltas, travel distance (incl. Oakland/San Diego venues), scoring profiles, stats_player harmonization, projection-row labels, forward roster rows, Vegas implied-total sign, team-code normalization, same-week leakage guards, depth/snap as-of joins, zero-point rows, residual intervals (`tests/test_fixes_2026_09_25.py`), per-position strategy / router / per-position intervals / position-specific features (`tests/test_per_position.py`).

## Known gaps

- Injury designations cover a minority of player-games; nflverse's current-week report may lag the final Friday designations.
  Projections set Out/IR (per nflverse) to 0; Questionable/Doubtful are left to the model.
- 2025+ depth charts are ESPN snapshots and can be stale; 2016–24 weekly charts have no publish timestamp.
- Schedule `temp`/`wind` missing for a large share of games (domes intentionally nulled; outdoor gaps remain).
- Neutral-site games use the home team's stadium for travel; stadium lat/lon and TZ tables are static approximations.
- `low_80/high_80` are empirical holdout-residual ranges (80% on 2025, ~76% on 2026 wk1–2 out of sample).
- `nfl-data-py` metadata pins conflict with NumPy 2 / Pandas 2+; runtime works with current wheels on CPython 3.13.

## ESPN wiring (next steps)

`configs/default.yaml` records `espn.league_id: 746996`.

1. Map ESPN player IDs ↔ nflverse `player_id` / gsis.
2. Pull league settings & roster (public endpoints or SWID/`espn_s2`).
3. League scoring confirmed full PPR (`--scoring ppr`); no TE premium.
4. Add a roster-filtered start/sit report from `predict`.

## Data credit

nflverse via `nfl_data_py`. Personal / research fantasy use.
