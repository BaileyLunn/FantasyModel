# Seasons used

- **Requested target:** 2016–2025
- **Downloaded & trained:** 2016–2024 (9 seasons)
- **Missing:** 2025 (nflverse weekly HTTP 404 at project build)
- **Filter:** `season_type == REG`, positions QB/RB/WR/TE
- **Split:** train season < 2024; test season == 2024
- **Rows:** 40,488 train / 5,227 test
- **Holdout metrics (ESPN half-PPR):** MAE 4.40 · RMSE 5.89 · R² 0.38
- **By position MAE:** QB 6.01 · RB 4.56 · TE 3.31 · WR 4.35
