# FantasyModel

Predict **expected fantasy points** for NFL player-games (pre-game features only) using nflverse history.

**Default scoring:** ESPN half-PPR (`0.5` PPR). Configure in `configs/default.yaml`.

**Seasons in the panel:** **2016–2025** regular season + **2026 weeks 1–2** (QB/RB/WR/TE, 51,626 player-games).
2016–2024 weekly stats come from `nfl_data_py`; 2025+ from the nflverse `stats_player_week` release
(the legacy `player_stats` release 404s from 2025 on — handled automatically in `fantasy_model/sources.py`).

## Results (real data, leakage-aware)

Validation: train seasons `< 2025`, test **2025 REG** (`model.test_season`). The production artifact is then
refit on all labelled rows through the latest completed week (`model.refit_full: true`).

| Metric (2025 holdout, half-PPR) | Value |
|--------|------:|
| n_train | 45,715 |
| n_test | 5,285 |
| MAE | **4.38** |
| RMSE | **5.90** |
| R² | **0.36** |

By position (2025 test MAE): QB 6.43 · RB 4.58 · TE 3.34 · WR 4.17

Previous v1 (train < 2024, test 2024): MAE 4.40 · RMSE 5.89 · R² 0.38. v1 scored out-of-sample on 2026 weeks 1–2:
MAE 4.65 · RMSE 6.36 · R² 0.32 (n=626). Full before/after tables: `reports/UPDATE_2026_W2.md`.

Reports: `reports/train_metrics.json`, `reports/eval_report.json`, `reports/oos/*.json`, `reports/SEASONS_USED.md`.

> Same-game usage (targets/carries that week) is **not** used as a feature. Rolling usage / FP / injury analogs / opponent allowances all use `shift(1)` so validation reflects pre-game information.

## Layout

```
FantasyModel/
  configs/default.yaml
  fantasy_model/           # scoring, features/*, train, evaluate, predict, cli
  fantasy_model/sources.py  # nflverse URLs + stats_player -> legacy schema
  fantasy_model/panel.py    # weekly + schedule + injuries join
  fantasy_model/project.py  # forward (upcoming-week) projections
  scripts/fetch_data.py
  scripts/update_week.py    # in-season: add week N, score, retrain, project N+1
  scripts/make_sample_data.py
  data/raw/                # gitignored nflverse downloads
  data/processed/          # gitignored player_games panel
  data/sample/             # offline smoke fixtures
  models/fantasy_hgb.joblib
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

python -m fantasy_model train                         # validate on 2025, refit on everything
python -m fantasy_model evaluate                      # 2025 holdout (uses models/fantasy_hgb_val.joblib)
python -m fantasy_model evaluate --season 2026 --weeks 1 2 --model models/fantasy_hgb.joblib
python -m fantasy_model predict --season 2024 --week 10          # played weeks (shows actuals)
python -m fantasy_model project --season 2026 --week 3           # upcoming week -> reports/projections_2026_w3.csv
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
3. retrains (2025 validation + full refit);
4. projects week N+1 → `reports/projections_2026_w4.csv`.

### Offline smoke (synthetic)

```bash
python scripts/make_sample_data.py
python -m fantasy_model --sample train
python -m fantasy_model --sample evaluate
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
| Baselines | Opp FP allowed vs position, rest days, rolling/season FP, Vegas spread/total → implied team total |

Model: `HistGradientBoostingRegressor` (optional `lightgbm` / `xgboost` via config).

## Tests

```bash
pytest -q
```

Dome weather nulling, timezone deltas, travel distance, half-PPR scoring, stats_player harmonization, projection-row labels, forward roster rows.

## Known gaps

- `implied_team_total` sign: nflverse `spread_line` is positive when the home team is favored, but the feature uses
  `(total − spread)/2` for home teams, so it's effectively the opponent's implied total (pre-existing; fix pending).
- Injury designations cover ~18% of skill player-games (healthy players have null status → severity 0); early “Out” tags are sparse vs Questionable.
- Schedule `temp`/`wind` missing for a large share of games (domes intentionally nulled; outdoor gaps remain).
- 2016–2019 rows for relocated teams (OAK/SD/STL abbreviations) miss the schedule join (768 rows).
- Stadium lat/lon and TZ tables are static approximations.
- Backup-QB flag is a production heuristic, not depth-chart authoritative; forward projections include every
  active-roster player (no depth chart), so filter on `n_prior_games_this_season`.
- `nfl-data-py` metadata pins conflict with NumPy 2 / Pandas 2+; runtime works with current wheels on CPython 3.13.

## ESPN wiring (next steps)

`configs/default.yaml` records `espn.league_id: 746996`.

1. Map ESPN player IDs ↔ nflverse `player_id` / gsis.
2. Pull league settings & roster (public endpoints or SWID/`espn_s2`).
3. Diff league scoring vs `configs/default.yaml`.
4. Add a roster-filtered start/sit report from `predict`.

## Data credit

nflverse via `nfl_data_py`. Personal / research fantasy use.
