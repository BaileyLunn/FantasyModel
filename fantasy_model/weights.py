"""Recency sample weights for training (exponential decay by season + within-season week)."""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_WEEKS_PER_SEASON = 18


def season_time(season, week, weeks_per_season: int = DEFAULT_WEEKS_PER_SEASON) -> np.ndarray:
    """Continuous time in seasons: 2025 wk1 -> 2025.0, 2025 wk18 -> 2025 + 17/18."""
    s = np.asarray(season, dtype=float)
    w = np.nan_to_num(np.asarray(week, dtype=float), nan=1.0)
    return s + (np.clip(w, 1, None) - 1.0) / float(weeks_per_season)


def recency_weights(
    season,
    week,
    half_life_seasons: float | None,
    weeks_per_season: int = DEFAULT_WEEKS_PER_SEASON,
    reference: float | None = None,
) -> np.ndarray:
    """w = 0.5 ** (age / half_life), age = seasons between the row and ``reference``
    (default: the most recent row). Normalized to mean 1. ``half_life_seasons`` None/<=0 -> uniform.

    Example (half-life 3): a game one season older than the newest counts 0.79x, three seasons
    older 0.5x, nine seasons older (2016 vs 2025) 0.125x — old seasons still contribute.
    """
    t = season_time(season, week, weeks_per_season)
    if t.size == 0:
        return np.ones(0)
    if not half_life_seasons or float(half_life_seasons) <= 0:
        return np.ones_like(t)
    ref = float(np.nanmax(t)) if reference is None else float(reference)
    age = np.clip(ref - t, 0.0, None)
    w = np.power(0.5, age / float(half_life_seasons))
    return w / w.mean()


def training_recency(cfg: dict[str, Any]) -> tuple[float | None, int]:
    tr = (cfg or {}).get("training", {}) or {}
    hl = tr.get("recency_half_life_seasons")
    hl = float(hl) if hl not in (None, "", "none", "None") and float(hl) > 0 else None
    return hl, int(tr.get("recency_weeks_per_season", DEFAULT_WEEKS_PER_SEASON))


def weight_summary(season, weights) -> dict[str, float]:
    """Mean weight per season (relative to the newest season = 1.0) for reports."""
    season = np.asarray(season)
    weights = np.asarray(weights, dtype=float)
    per = {int(s): float(weights[season == s].mean()) for s in np.unique(season)}
    top = per[max(per)] if per else 1.0
    return {str(k): round(v / top, 4) for k, v in per.items()}
