# Seasons used

- **Data:** 2016–2025 REG + 2026 REG weeks 1–2 (QB/RB/WR/TE), 51,626 player-games.
  2016–2024 via `nfl_data_py` legacy weekly; 2025–2026 via nflverse `stats_player_week` release.
- **Validation (config `model.test_season: 2025`):** train 2016–2024 (45,715 rows) → 2025 holdout (5,285):
  MAE 4.38 · RMSE 5.90 · R² 0.362 (half-PPR). By position MAE: QB 6.43 · RB 4.58 · TE 3.34 · WR 4.17.
- **Production artifact (`refit_full: true`):** refit on all labelled rows through 2026 week 2.
- **v1 (previous):** train 2016–2023, holdout 2024: MAE 4.40 · RMSE 5.89 · R² 0.38.
  v1 out-of-sample on 2026 wk1–2 (n=626): MAE 4.65 · RMSE 6.36 · R² 0.324.
- Details: `reports/UPDATE_2026_W2.md`, `reports/oos/*.json`.
