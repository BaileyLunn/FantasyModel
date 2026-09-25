# Per-position models vs the pooled model (2026-09-25)

Question: are separate QB/RB/WR/TE models more accurate than the single pooled HistGradientBoosting model
(position as a feature)? Script: `scripts/per_position_experiment.py`; raw results (every seed, every metric):
`reports/per_position_{ppr,half}.json`. League scoring is full PPR, so PPR decides; half-PPR is a cross-check.

## Decision

| position | PPR | half-PPR | adopted |
|---|---|---|---|
| QB | per-position model with position-specific features passes the rule | passes the rule | **own model (+ position-specific features)** |
| RB | nothing beats pooled on 2025 | same | pooled |
| WR | per-position is worse on 2025 (+0.035 RMSE, MAE +0.075), better on 2026/roster | same pattern | pooled (fails the 2025 criterion) |
| TE | no 2025 gain; posfeat better on 2026/roster | same | pooled (fails the 2025 criterion) |

Config (`configs/default.yaml`): `model.strategy: hybrid`, `model.per_position: {QB: true, RB: false, WR: false, TE: false}`,
`model.position_params` (QB hyper-parameters, per scoring profile), `features.position_specific_enabled: true`
(these features go only to per-position models; the pooled model is byte-for-byte the old one).

**The QB gain is small and borderline, so treat it as marginal.** On the 2025 holdout, QB RMSE drops 0.056 (PPR; 5.831 → 5.776, about 1%)
and 0.063 (half). Both drops are bigger than seed noise (2 × paired sd = 0.049 / 0.046), and MAE also drops (3.763 → 3.745). The QB model
also beat pooled on the 2024 inner validation (5.854 vs 5.886 PPR), before any holdout was looked at. On 2026 wk1-2 (n=180 QB rows) and in the roster backtest (n=128),
QB **RMSE is higher** (+0.071 and +0.069 PPR), which is inside seed noise (0.136 / 0.170), and **MAE is lower** (3.415 → 3.332, 4.510 → 4.372).
The player-game bootstrap CI for the 2025 PPR QB gain touches zero ([-0.113, +0.000]), so sampling noise is as big as the effect.
Overall (all positions), the adopted mix moves RMSE by -0.010 on 2025, +0.013 on 2026 and +0.011 on the roster backtest. All three are within noise.

### Pre-registered rule (written in the script before results)
Adopt a per-position model for position P only if, on P's rows, compared with pooled:
(1) 2025 holdout mean RMSE (all rows) is lower by more than seed noise (2 × sd of the per-seed paired difference), and MAE is not higher;
(2) 2026 in-season and roster-backtest mean RMSE are not higher by more than seed noise.
The primary candidate was the hybrid's locked choice. `posfeat` counts where it passes and either nothing else qualified for P or it beats `separate` on 2025.
Disclosure: for QB the hybrid locked in `separate` (no position-specific features). That model cut 2025 RMSE more (-0.063) but **raised** MAE (+0.021), so it failed (1).
`posfeat` passed, so it was adopted. Picking among three variants on the 2025 holdout is a mild multiple comparison. The 2024 inner validation, which is independent, also favoured it.

## Protocol
* **pooled**: the current production model (config hyper-parameters, recency weights with half-life 4, EWMA 5).
* **separate**: one HGB per position with the same features minus `position_code`. Hyper-parameters were tuned **per position on pre-holdout data only**
  (time-based CV: train seasons < fold, validate folds 2021, 2022, 2023, mean RMSE). Grid: max_depth 3/4/6 × learning_rate 0.03/0.06 ×
  min_samples_leaf 25/60/150 × max_iter 300/600 (36 combos, seed 42). Tuned models beat per-position models that reuse the pooled hyper-parameters
  by 0.02-0.15 CV RMSE. Winners sit at the grid edge (learning_rate 0.03, min_samples_leaf 150 mostly), so heavier regularization might help a little more.
* **hybrid**: per position, pooled or separate, whichever had the lower 5-seed mean RMSE when trained on seasons ≤ 2023 and validated on 2024.
  Locked before 2025/2026 were scored. PPR: QB, TE, WR separate; RB pooled. Half: QB, WR separate; RB, TE pooled.
* **posfeat** (optional d): separate models plus position-specific EWMA features (same 5-game half-life, prior games only): `rush_share_ewm`
  (carries / team carries, i.e. QB rushing share and RB workload), `rush_yds_ewm`, `target_share_ewm`, `air_yards_share_ewm`, `rec_air_yards_ewm`,
  `pass_att_ewm`, `pass_air_yards_ewm`, all from the already-fetched nflverse weekly stats. Tuned the same way.
* **adopted**: the final per-position mix (QB = posfeat, rest pooled). Re-scored as its own variant with the same locked parameters and seeds.
  The production code path (`train_model`) reproduces its seed-0 2025 metrics exactly (MAE 3.4075 / RMSE 5.1553).
* Checks: 2025 holdout (train < 2025); 2026 wk1-2 (train < 2026); roster backtest (thru-2025 model, every ACT QB/RB/WR/TE of 2026 wk1-2,
  Out/IR → 0). Seeds 0-4 (HGB early-stopping split). Metrics are on all rows (incl. 0-point rows) unless marked "stats rows only".
* **Small samples**: 2026 has 180 QB and 241 TE rows (roster backtest: 128 QB, 222 TE) from just two weeks. Their bootstrap CIs are ±0.1-0.2 RMSE,
  much wider than seed noise, so 2026 per-position differences under about 0.15 RMSE are not informative.

## PPR (primary)

**2025 holdout (train <2025)** - MAE / RMSE ± seed sd / R² / bias (all rows)

| group | n | pooled | separate | hybrid | posfeat | adopted |
|---|---|---|---|---|---|---|
| ALL | 8585 | 3.382 / 5.156 ± 0.008 / 0.520 / +0.17 | 3.423 / 5.157 ± 0.005 / 0.520 / +0.27 | 3.415 / 5.159 ± 0.006 / 0.519 / +0.26 | 3.419 / 5.149 ± 0.006 / 0.521 / +0.25 | 3.379 / 5.146 ± 0.007 / 0.522 / +0.17 |
| QB | 1359 | 3.763 / 5.831 ± 0.017 / 0.612 / +0.16 | 3.784 / 5.769 ± 0.019 / 0.620 / +0.29 | 3.784 / 5.769 ± 0.019 / 0.620 / +0.29 | 3.745 / 5.776 ± 0.024 / 0.619 / +0.17 | 3.745 / 5.776 ± 0.024 / 0.619 / +0.17 |
| RB | 2024 | 3.541 / 5.401 ± 0.006 / 0.535 / +0.09 | 3.575 / 5.393 ± 0.007 / 0.536 / +0.13 | 3.541 / 5.401 ± 0.006 / 0.535 / +0.09 | 3.571 / 5.377 ± 0.006 / 0.539 / +0.11 | 3.541 / 5.401 ± 0.006 / 0.535 / +0.09 |
| TE | 1959 | 2.650 / 4.207 ± 0.012 / 0.442 / -0.02 | 2.655 / 4.212 ± 0.010 / 0.441 / -0.03 | 2.655 / 4.212 ± 0.010 / 0.441 / -0.03 | 2.668 / 4.205 ± 0.006 / 0.443 / -0.02 | 2.650 / 4.207 ± 0.012 / 0.442 / -0.02 |
| WR | 3243 | 3.565 / 5.218 ± 0.009 / 0.446 / +0.34 | 3.640 / 5.253 ± 0.011 / 0.438 / +0.54 | 3.640 / 5.253 ± 0.011 / 0.438 / +0.54 | 3.643 / 5.243 ± 0.009 / 0.440 / +0.54 | 3.565 / 5.218 ± 0.009 / 0.446 / +0.34 |
| ALL, stats rows only | 5285 | 4.559 / 6.308 ± 0.009 / 0.362 / -0.65 | 4.567 / 6.289 ± 0.004 / 0.366 / -0.54 | 4.563 / 6.294 ± 0.008 / 0.365 / -0.55 | 4.563 / 6.285 ± 0.007 / 0.367 / -0.57 | 4.557 / 6.299 ± 0.007 / 0.364 / -0.64 |

**2026 wk1-2 (train <2026)** - MAE / RMSE ± seed sd / R² / bias (all rows)

| group | n | pooled | separate | hybrid | posfeat | adopted |
|---|---|---|---|---|---|---|
| ALL | 1019 | 3.480 / 5.406 ± 0.012 / 0.502 / -0.11 | 3.483 / 5.381 ± 0.021 / 0.506 / -0.02 | 3.473 / 5.367 ± 0.025 / 0.509 / -0.02 | 3.489 / 5.380 ± 0.013 / 0.507 / +0.01 | 3.465 / 5.418 ± 0.014 / 0.500 / -0.13 |
| QB | 180 | 3.415 / 5.426 ± 0.032 / 0.678 / -0.01 | 3.301 / 5.459 ± 0.037 / 0.674 / -0.18 | 3.301 / 5.459 ± 0.037 / 0.674 / -0.18 | 3.332 / 5.497 ± 0.059 / 0.669 / -0.18 | 3.332 / 5.497 ± 0.059 / 0.669 / -0.18 |
| RB | 222 | 3.580 / 5.251 ± 0.037 / 0.554 / +0.18 | 3.628 / 5.315 ± 0.030 / 0.543 / +0.17 | 3.580 / 5.251 ± 0.037 / 0.554 / +0.18 | 3.643 / 5.297 ± 0.035 / 0.546 / +0.23 | 3.580 / 5.251 ± 0.037 / 0.554 / +0.18 |
| TE | 241 | 2.891 / 4.648 ± 0.025 / 0.334 / +0.04 | 2.874 / 4.632 ± 0.012 / 0.338 / -0.03 | 2.874 / 4.632 ± 0.012 / 0.338 / -0.03 | 2.853 / 4.564 ± 0.018 / 0.358 / +0.01 | 2.891 / 4.648 ± 0.025 / 0.334 / +0.04 |
| WR | 376 | 3.829 / 5.914 ± 0.031 / 0.366 / -0.41 | 3.876 / 5.812 ± 0.045 / 0.388 / -0.04 | 3.876 / 5.812 ± 0.045 / 0.388 / -0.04 | 3.882 / 5.835 ± 0.026 / 0.383 / -0.02 | 3.829 / 5.914 ± 0.031 / 0.366 / -0.41 |
| ALL, stats rows only | 626 | 4.780 / 6.681 ± 0.014 / 0.346 / -1.05 | 4.757 / 6.642 ± 0.021 / 0.354 / -0.94 | 4.747 / 6.626 ± 0.027 / 0.357 / -0.93 | 4.758 / 6.637 ± 0.020 / 0.355 / -0.89 | 4.787 / 6.702 ± 0.016 / 0.342 / -1.07 |

**Roster backtest 2026 wk1-2 (train <2026, all ACT players)** - MAE / RMSE ± seed sd / R² / bias (all rows)

| group | n | pooled | separate | hybrid | posfeat | adopted |
|---|---|---|---|---|---|---|
| ALL | 904 | 3.746 / 5.657 ± 0.012 / 0.486 / -0.31 | 3.732 / 5.624 ± 0.018 / 0.492 / -0.23 | 3.725 / 5.613 ± 0.023 / 0.494 / -0.22 | 3.736 / 5.621 ± 0.015 / 0.493 / -0.20 | 3.726 / 5.668 ± 0.015 / 0.484 / -0.35 |
| QB | 128 | 4.510 / 6.409 ± 0.048 / 0.618 / -0.27 | 4.359 / 6.440 ± 0.044 / 0.614 / -0.52 | 4.359 / 6.440 ± 0.044 / 0.614 / -0.52 | 4.372 / 6.478 ± 0.066 / 0.610 / -0.53 | 4.372 / 6.478 ± 0.066 / 0.610 / -0.53 |
| RB | 217 | 3.567 / 5.270 ± 0.036 / 0.554 / +0.00 | 3.597 / 5.319 ± 0.027 / 0.546 / -0.02 | 3.567 / 5.270 ± 0.036 / 0.554 / +0.00 | 3.616 / 5.305 ± 0.032 / 0.548 / +0.03 | 3.567 / 5.270 ± 0.036 / 0.554 / +0.00 |
| TE | 222 | 3.050 / 4.821 ± 0.029 / 0.318 / -0.05 | 3.002 / 4.777 ± 0.016 / 0.331 / -0.15 | 3.002 / 4.777 ± 0.016 / 0.331 / -0.15 | 2.988 / 4.714 ± 0.015 / 0.348 / -0.09 | 3.050 / 4.821 ± 0.029 / 0.318 / -0.05 |
| WR | 337 | 4.028 / 6.087 ± 0.030 / 0.359 / -0.71 | 4.062 / 5.986 ± 0.039 / 0.380 / -0.31 | 4.062 / 5.986 ± 0.039 / 0.380 / -0.31 | 4.064 / 6.005 ± 0.028 / 0.377 / -0.29 | 4.028 / 6.087 ± 0.030 / 0.359 / -0.71 |

**RMSE difference vs pooled** (mean of per-seed paired differences ± sd; negative = better). Bootstrap 90% CI (player-game resampling of seed-averaged predictions) in brackets.

| variant | group | 2025 | 2026 | roster |
|---|---|---|---|---|
| separate | ALL | +0.001 ± 0.011 [-0.018, +0.016] | -0.025 ± 0.019 [-0.075, +0.028] | -0.033 ± 0.020 [-0.090, +0.019] |
| separate | QB | -0.063 ± 0.025 [-0.120, -0.007] | +0.033 ± 0.056 [-0.067, +0.134] | +0.031 ± 0.076 [-0.089, +0.149] |
| separate | RB | -0.008 ± 0.010 [-0.043, +0.025] | +0.064 ± 0.057 [-0.019, +0.142] | +0.049 ± 0.054 [-0.035, +0.129] |
| separate | TE | +0.005 ± 0.012 [-0.027, +0.036] | -0.016 ± 0.031 [-0.132, +0.087] | -0.044 ± 0.034 [-0.164, +0.071] |
| separate | WR | +0.035 ± 0.019 [+0.006, +0.056] | -0.102 ± 0.054 [-0.200, -0.010] | -0.101 ± 0.052 [-0.201, -0.011] |
| hybrid | ALL | +0.003 ± 0.011 [-0.014, +0.016] | -0.038 ± 0.023 [-0.086, +0.012] | -0.045 ± 0.023 [-0.095, +0.005] |
| hybrid | QB | -0.063 ± 0.025 [-0.120, -0.007] | +0.033 ± 0.056 [-0.067, +0.134] | +0.031 ± 0.076 [-0.089, +0.149] |
| hybrid | RB | +0.000 ± 0.000 [+0.000, +0.000] | +0.000 ± 0.000 [+0.000, +0.000] | +0.000 ± 0.000 [+0.000, +0.000] |
| hybrid | TE | +0.005 ± 0.012 [-0.027, +0.036] | -0.016 ± 0.031 [-0.132, +0.087] | -0.044 ± 0.034 [-0.164, +0.071] |
| hybrid | WR | +0.035 ± 0.019 [+0.006, +0.056] | -0.102 ± 0.054 [-0.200, -0.010] | -0.101 ± 0.052 [-0.201, -0.011] |
| posfeat | ALL | -0.006 ± 0.011 [-0.025, +0.008] | -0.026 ± 0.014 [-0.082, +0.029] | -0.036 ± 0.018 [-0.095, +0.018] |
| posfeat | QB | -0.056 ± 0.024 [-0.113, +0.000] | +0.071 ± 0.068 [-0.061, +0.200] | +0.069 ± 0.085 [-0.092, +0.221] |
| posfeat | RB | -0.023 ± 0.008 [-0.062, +0.012] | +0.046 ± 0.059 [-0.040, +0.130] | +0.035 ± 0.054 [-0.051, +0.122] |
| posfeat | TE | -0.002 ± 0.009 [-0.032, +0.027] | -0.084 ± 0.033 [-0.192, +0.011] | -0.107 ± 0.034 [-0.218, -0.001] |
| posfeat | WR | +0.025 ± 0.017 [-0.001, +0.047] | -0.079 ± 0.041 [-0.177, +0.014] | -0.082 ± 0.046 [-0.182, +0.008] |
| adopted | ALL | -0.010 ± 0.004 [-0.020, +0.000] | +0.013 ± 0.012 [-0.010, +0.038] | +0.011 ± 0.014 [-0.015, +0.037] |
| adopted | QB | -0.056 ± 0.024 [-0.113, +0.000] | +0.071 ± 0.068 [-0.061, +0.200] | +0.069 ± 0.085 [-0.092, +0.221] |
| adopted | RB | +0.000 ± 0.000 [+0.000, +0.000] | +0.000 ± 0.000 [+0.000, +0.000] | +0.000 ± 0.000 [+0.000, +0.000] |
| adopted | TE | +0.000 ± 0.000 [+0.000, +0.000] | +0.000 ± 0.000 [+0.000, +0.000] | +0.000 ± 0.000 [+0.000, +0.000] |
| adopted | WR | +0.000 ± 0.000 [+0.000, +0.000] | +0.000 ± 0.000 [+0.000, +0.000] | +0.000 ± 0.000 [+0.000, +0.000] |

## Half-PPR

**2025 holdout (train <2025)** - MAE / RMSE ± seed sd / R² / bias (all rows)

| group | n | pooled | separate | hybrid | posfeat | adopted |
|---|---|---|---|---|---|---|
| ALL | 8585 | 3.060 / 4.748 ± 0.003 / 0.517 / +0.14 | 3.115 / 4.755 ± 0.010 / 0.516 / +0.23 | 3.102 / 4.753 ± 0.010 / 0.516 / +0.21 | 3.094 / 4.740 ± 0.004 / 0.519 / +0.21 | 3.057 / 4.736 ± 0.004 / 0.519 / +0.15 |
| QB | 1359 | 3.759 / 5.826 ± 0.011 / 0.612 / +0.11 | 3.859 / 5.815 ± 0.043 / 0.614 / +0.17 | 3.859 / 5.815 ± 0.043 / 0.614 / +0.17 | 3.742 / 5.763 ± 0.018 / 0.621 / +0.16 | 3.742 / 5.763 ± 0.018 / 0.621 / +0.16 |
| RB | 2024 | 3.327 / 5.145 ± 0.002 / 0.513 / +0.07 | 3.381 / 5.155 ± 0.005 / 0.511 / +0.13 | 3.327 / 5.145 ± 0.002 / 0.513 / +0.07 | 3.375 / 5.138 ± 0.005 / 0.514 / +0.13 | 3.327 / 5.145 ± 0.002 / 0.513 / +0.07 |
| TE | 1959 | 2.243 / 3.642 ± 0.006 / 0.407 / -0.01 | 2.244 / 3.635 ± 0.009 / 0.410 / -0.01 | 2.243 / 3.642 ± 0.006 / 0.407 / -0.01 | 2.252 / 3.637 ± 0.006 / 0.409 / -0.02 | 2.243 / 3.642 ± 0.006 / 0.407 / -0.01 |
| WR | 3243 | 3.094 / 4.575 ± 0.005 / 0.415 / +0.28 | 3.164 / 4.594 ± 0.015 / 0.410 / +0.46 | 3.164 / 4.594 ± 0.015 / 0.410 / +0.46 | 3.156 / 4.593 ± 0.005 / 0.410 / +0.43 | 3.094 / 4.575 ± 0.005 / 0.415 / +0.28 |
| ALL, stats rows only | 5285 | 4.154 / 5.829 ± 0.005 / 0.378 / -0.58 | 4.168 / 5.817 ± 0.008 / 0.381 / -0.52 | 4.164 / 5.820 ± 0.010 / 0.380 / -0.52 | 4.157 / 5.806 ± 0.004 / 0.383 / -0.52 | 4.149 / 5.815 ± 0.004 / 0.381 / -0.57 |

**2026 wk1-2 (train <2026)** - MAE / RMSE ± seed sd / R² / bias (all rows)

| group | n | pooled | separate | hybrid | posfeat | adopted |
|---|---|---|---|---|---|---|
| ALL | 1019 | 3.100 / 4.909 ± 0.016 / 0.514 / -0.10 | 3.107 / 4.908 ± 0.007 / 0.514 / -0.03 | 3.100 / 4.890 ± 0.008 / 0.518 / -0.01 | 3.113 / 4.905 ± 0.016 / 0.515 / -0.02 | 3.083 / 4.924 ± 0.012 / 0.511 / -0.13 |
| QB | 180 | 3.398 / 5.390 ± 0.052 / 0.682 / +0.04 | 3.296 / 5.476 ± 0.035 / 0.672 / -0.12 | 3.296 / 5.476 ± 0.035 / 0.672 / -0.12 | 3.301 / 5.465 ± 0.062 / 0.673 / -0.15 | 3.301 / 5.465 ± 0.062 / 0.673 / -0.15 |
| RB | 222 | 3.315 / 4.905 ± 0.016 / 0.545 / +0.21 | 3.327 / 4.963 ± 0.020 / 0.534 / +0.19 | 3.315 / 4.905 ± 0.016 / 0.545 / +0.21 | 3.350 / 4.966 ± 0.026 / 0.533 / +0.21 | 3.315 / 4.905 ± 0.016 / 0.545 / +0.21 |
| TE | 241 | 2.406 / 3.977 ± 0.015 / 0.317 / -0.03 | 2.428 / 4.005 ± 0.015 / 0.308 / -0.06 | 2.406 / 3.977 ± 0.015 / 0.317 / -0.03 | 2.407 / 3.938 ± 0.018 / 0.331 / -0.02 | 2.406 / 3.977 ± 0.015 / 0.317 / -0.03 |
| WR | 376 | 3.276 / 5.202 ± 0.026 / 0.340 / -0.38 | 3.322 / 5.110 ± 0.021 / 0.364 / -0.08 | 3.322 / 5.110 ± 0.021 / 0.364 / -0.08 | 3.336 / 5.139 ± 0.005 / 0.356 / -0.10 | 3.276 / 5.202 ± 0.026 / 0.340 / -0.38 |
| ALL, stats rows only | 626 | 4.293 / 6.094 ± 0.018 / 0.379 / -0.90 | 4.290 / 6.097 ± 0.008 / 0.379 / -0.81 | 4.302 / 6.084 ± 0.015 / 0.381 / -0.76 | 4.282 / 6.088 ± 0.018 / 0.381 / -0.82 | 4.303 / 6.120 ± 0.011 / 0.374 / -0.92 |

**Roster backtest 2026 wk1-2 (train <2026, all ACT players)** - MAE / RMSE ± seed sd / R² / bias (all rows)

| group | n | pooled | separate | hybrid | posfeat | adopted |
|---|---|---|---|---|---|---|
| ALL | 904 | 3.339 / 5.147 ± 0.015 / 0.500 / -0.28 | 3.341 / 5.149 ± 0.008 / 0.499 / -0.21 | 3.344 / 5.138 ± 0.013 / 0.501 / -0.18 | 3.339 / 5.142 ± 0.018 / 0.501 / -0.21 | 3.323 / 5.162 ± 0.010 / 0.497 / -0.32 |
| QB | 128 | 4.466 / 6.361 ± 0.063 / 0.624 / -0.20 | 4.376 / 6.466 ± 0.041 / 0.611 / -0.42 | 4.376 / 6.466 ± 0.041 / 0.611 / -0.42 | 4.357 / 6.445 ± 0.073 / 0.614 / -0.47 | 4.357 / 6.445 ± 0.073 / 0.614 / -0.47 |
| RB | 217 | 3.314 / 4.941 ± 0.016 / 0.542 / +0.06 | 3.307 / 4.983 ± 0.022 / 0.534 / +0.01 | 3.314 / 4.941 ± 0.016 / 0.542 / +0.06 | 3.329 / 4.984 ± 0.029 / 0.534 / +0.03 | 3.314 / 4.941 ± 0.016 / 0.542 / +0.06 |
| TE | 222 | 2.547 / 4.131 ± 0.017 / 0.302 / -0.10 | 2.541 / 4.136 ± 0.016 / 0.300 / -0.16 | 2.547 / 4.131 ± 0.017 / 0.302 / -0.10 | 2.522 / 4.071 ± 0.017 / 0.322 / -0.11 | 2.547 / 4.131 ± 0.017 / 0.302 / -0.10 |
| WR | 337 | 3.448 / 5.360 ± 0.021 / 0.336 / -0.64 | 3.495 / 5.290 ± 0.029 / 0.353 / -0.30 | 3.495 / 5.290 ± 0.029 / 0.353 / -0.30 | 3.499 / 5.314 ± 0.010 / 0.347 / -0.33 | 3.448 / 5.360 ± 0.021 / 0.336 / -0.64 |

**RMSE difference vs pooled** (mean of per-seed paired differences ± sd; negative = better). Bootstrap 90% CI (player-game resampling of seed-averaged predictions) in brackets.

| variant | group | 2025 | 2026 | roster |
|---|---|---|---|---|
| separate | ALL | +0.006 ± 0.012 [-0.012, +0.020] | -0.001 ± 0.018 [-0.051, +0.047] | +0.003 ± 0.019 [-0.050, +0.051] |
| separate | QB | -0.010 ± 0.047 [-0.067, +0.041] | +0.086 ± 0.050 [-0.031, +0.201] | +0.105 ± 0.063 [-0.039, +0.243] |
| separate | RB | +0.010 ± 0.005 [-0.022, +0.038] | +0.057 ± 0.031 [-0.037, +0.146] | +0.043 ± 0.034 [-0.050, +0.129] |
| separate | TE | -0.007 ± 0.008 [-0.033, +0.019] | +0.028 ± 0.007 [-0.063, +0.111] | +0.005 ± 0.008 [-0.084, +0.094] |
| separate | WR | +0.019 ± 0.016 [-0.004, +0.039] | -0.093 ± 0.026 [-0.181, -0.011] | -0.070 ± 0.034 [-0.159, +0.008] |
| hybrid | ALL | +0.005 ± 0.013 [-0.009, +0.017] | -0.019 ± 0.017 [-0.061, +0.020] | -0.008 ± 0.018 [-0.051, +0.033] |
| hybrid | QB | -0.010 ± 0.047 [-0.067, +0.041] | +0.086 ± 0.050 [-0.031, +0.201] | +0.105 ± 0.063 [-0.039, +0.243] |
| hybrid | RB | +0.000 ± 0.000 [+0.000, +0.000] | +0.000 ± 0.000 [+0.000, +0.000] | +0.000 ± 0.000 [+0.000, +0.000] |
| hybrid | TE | +0.000 ± 0.000 [+0.000, +0.000] | +0.000 ± 0.000 [+0.000, +0.000] | +0.000 ± 0.000 [+0.000, +0.000] |
| hybrid | WR | +0.019 ± 0.016 [-0.004, +0.039] | -0.093 ± 0.026 [-0.181, -0.011] | -0.070 ± 0.034 [-0.159, +0.008] |
| posfeat | ALL | -0.008 ± 0.005 [-0.023, +0.007] | -0.004 ± 0.027 [-0.048, +0.039] | -0.004 ± 0.029 [-0.050, +0.044] |
| posfeat | QB | -0.063 ± 0.023 [-0.111, -0.008] | +0.075 ± 0.083 [-0.021, +0.172] | +0.085 ± 0.103 [-0.035, +0.198] |
| posfeat | RB | -0.007 ± 0.005 [-0.042, +0.023] | +0.061 ± 0.036 [-0.034, +0.153] | +0.043 ± 0.040 [-0.048, +0.141] |
| posfeat | TE | -0.005 ± 0.009 [-0.029, +0.021] | -0.039 ± 0.017 [-0.121, +0.036] | -0.060 ± 0.019 [-0.141, +0.021] |
| posfeat | WR | +0.018 ± 0.008 [+0.001, +0.037] | -0.064 ± 0.028 [-0.138, +0.014] | -0.045 ± 0.028 [-0.119, +0.026] |
| adopted | ALL | -0.012 ± 0.004 [-0.022, -0.002] | +0.015 ± 0.016 [-0.005, +0.035] | +0.015 ± 0.018 [-0.005, +0.037] |
| adopted | QB | -0.063 ± 0.023 [-0.111, -0.008] | +0.075 ± 0.083 [-0.021, +0.172] | +0.085 ± 0.103 [-0.035, +0.198] |
| adopted | RB | +0.000 ± 0.000 [+0.000, +0.000] | +0.000 ± 0.000 [+0.000, +0.000] | +0.000 ± 0.000 [+0.000, +0.000] |
| adopted | TE | +0.000 ± 0.000 [+0.000, +0.000] | +0.000 ± 0.000 [+0.000, +0.000] | +0.000 ± 0.000 [+0.000, +0.000] |
| adopted | WR | +0.000 ± 0.000 [+0.000, +0.000] | +0.000 ± 0.000 [+0.000, +0.000] | +0.000 ± 0.000 [+0.000, +0.000] |

## Production refit (seed 42, through 2026 week 2)
`python -m fantasy_model --scoring {ppr,half_ppr} train`, then `scripts/validate_models.py`, then `project --season 2026 --week 3 --no-refresh-roster`
(same roster snapshot as the previous projections, so only the model changed).

| MAE / RMSE / R², stats rows | PPR pooled (before) | PPR adopted | half pooled (before) | half adopted |
|---|---|---|---|---|
| 2025 holdout (n=5,285) | 4.57 / 6.32 / 0.359 | 4.55 / 6.29 / 0.365 | 4.16 / 5.84 / 0.377 | 4.15 / 5.81 / 0.383 |
| 2026 wk1-2 (n=626) | 4.76 / 6.67 / 0.349 | 4.77 / 6.69 / 0.345 | 4.29 / 6.11 / 0.376 | 4.30 / 6.14 / 0.369 |

Intervals: the 80% residual ranges are now computed **per position** (2025 holdout residuals of the validation model, binned by projection;
the pooled table is kept as a fallback). 2026 wk1-2 out-of-sample coverage, PPR: per-position tables 75.8% (QB 77%, RB 78%, WR 76%, TE 73%),
width 9.07; pooled table 77.0%, width 9.17. The half-PPR numbers are 78.0% and 78.9%.

Week-3 projections: only QBs changed (RB/WR/TE point projections are identical; their low/high ranges moved because of the per-position intervals).
PPR QB mean change -0.28. QBs that moved more than 1.5 points: Jaxson Dart +3.52 (3.39 → 6.91), Caleb Williams -2.64, Drake Maye -2.17,
Geno Smith -2.03, Daniel Jones -1.99, Justin Herbert -1.93, Tua Tagovailoa -1.89, Malik Willis -1.80, J.J. McCarthy -1.54.

## Watch list
* WR and TE per-position models (TE especially with position-specific features) are better on 2026 wk1-2 and the roster backtest
  (roster RMSE: TE posfeat -0.107, WR separate -0.101) but not on the 2025 holdout. Re-run the experiment later in the season, when 2026 has
  more weeks, before switching them.
* Per-position models push 2025 bias up (WR +0.54 vs +0.34) but pull the 2026 WR under-projection toward zero (-0.04 vs -0.41).

Re-run: `python scripts/per_position_experiment.py --scoring ppr half_ppr` (~30 min per profile on 4 threads), then
`--reuse-tuning` to add the `adopted` variant.
