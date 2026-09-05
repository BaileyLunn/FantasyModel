# FantasyModel

Predict **expected fantasy points** for NFL player-games (pre-game features only) using nflverse history.

**Default scoring:** ESPN half-PPR (`0.5` PPR). Configure in `configs/default.yaml`.

**Seasons used in the trained model:** **2016–2024** regular season (QB/RB/WR/TE).  
Target window was 2016–2025; **2025 weekly files returned HTTP 404** from nflverse at build time.

## Results (real data, leakage-aware)

Time-based split: train seasons `< 2024`, test **2024 REG**.

| Metric | Value |
|--------|------:|
| n_train | 40,488 |
| n_test | 5,227 |
| MAE | **4.40** |
| RMSE | **5.89** |
| R² | **0.38** |

By position (2024 test MAE): QB 6.01 · RB 4.56 · TE 3.31 · WR 4.35

Reports: `reports/train_metrics.json`, `reports/eval_report.json`, `reports/test_predictions.csv`.

> Same-game usage (targets/carries that week) is **not** used as a feature. Rolling usage / FP / injury analogs / opponent allowances all use `shift(1)` so validation reflects pre-game information.

## Layout

```
FantasyModel/
  configs/default.yaml
  fantasy_model/           # scoring, features/*, train, evaluate, predict, cli
  scripts/fetch_data.py
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
# Full historical window (skips missing years)
python scripts/fetch_data.py --seasons 2016 2017 2018 2019 2020 2021 2022 2023 2024
# Rebuild panel only from existing raw files:
python scripts/fetch_data.py --skip-download --seasons 2016 2017 2018 2019 2020 2021 2022 2023 2024

python -m fantasy_model train
python -m fantasy_model evaluate
python -m fantasy_model predict --season 2024 --week 10
python -m fantasy_model predict --season 2024 --week 10 --player "Mahomes"
```

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

Dome weather nulling, timezone deltas, travel distance, half-PPR scoring.

## Known gaps

- **2025** weekly data unavailable (404) — re-run fetch when nflverse publishes it.
- Injury designations cover ~18% of skill player-games (healthy players have null status → severity 0); early “Out” tags are sparse vs Questionable.
- Schedule `temp`/`wind` missing for a large share of games (domes intentionally nulled; outdoor gaps remain).
- Stadium lat/lon and TZ tables are static approximations.
- Backup-QB flag is a production heuristic, not depth-chart authoritative.
- Predict on historical weeks still shows actual `fantasy_points` for comparison; upcoming weeks need a forward roster/schedule feed.
- `nfl-data-py` metadata pins conflict with NumPy 2 / Pandas 2+; runtime works with current wheels on CPython 3.13.

## ESPN wiring (next steps)

`configs/default.yaml` records `espn.league_id: 746996`.

1. Map ESPN player IDs ↔ nflverse `player_id` / gsis.
2. Pull league settings & roster (public endpoints or SWID/`espn_s2`).
3. Diff league scoring vs `configs/default.yaml`.
4. Add a roster-filtered start/sit report from `predict`.

## Data credit

nflverse via `nfl_data_py`. Personal / research fantasy use.
