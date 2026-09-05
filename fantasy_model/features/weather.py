"""Weather features with indoor/dome/retractable-closed nulling."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Stadiums that are always indoor (domes) or typically closed retractable.
# Keys are nflverse / common abbreviations. Weather should be nulled when closed.
ALWAYS_INDOOR = {
    "ATL",  # Mercedes-Benz Stadium (retractable; treat closed as indoor default)
    "DET",  # Ford Field
    "HOU",  # NRG (retractable)
    "IND",  # Lucas Oil (retractable)
    "LV",   # Allegiant (retractable / indoor climate)
    "LAR",  # SoFi (open but mild; NOT indoor — keep weather)
    "LAC",  # SoFi shared
    "MIN",  # U.S. Bank Stadium
    "NO",   # Caesars Superdome
    "DAL",  # AT&T Stadium (retractable)
    "ARI",  # State Farm (retractable)
}

# Explicit roof types if column present
INDOOR_ROOF_VALUES = {"dome", "closed", "indoor", "retractable_closed"}


def is_indoor_game(row: pd.Series) -> bool:
    """Return True if outdoor weather should NOT apply."""
    roof = str(row.get("roof", "") or "").strip().lower()
    if roof in INDOOR_ROOF_VALUES:
        return True
    if roof in {"outdoors", "open", "retractable_open"}:
        return False
    # Fallback: home team dome heuristic when roof unknown
    home = str(row.get("home_team", "") or row.get("home", "") or "").upper()
    if home in ALWAYS_INDOOR and roof in {"", "nan", "none"}:
        return True
    # Explicit flag
    if row.get("is_indoor") is True or row.get("is_dome") is True:
        return True
    return False


def null_indoor_weather(df: pd.DataFrame, weather_cols: list[str] | None = None) -> pd.DataFrame:
    """Set weather columns to NaN for indoor/closed-roof games.

    Outdoor weather (temp, wind, humidity, precip) must not influence dome games.
    """
    out = df.copy()
    cols = weather_cols or [
        c
        for c in ("temp", "wind", "humidity", "precip", "weather_temp", "weather_wind", "weather_humidity")
        if c in out.columns
    ]
    if not cols:
        return out
    indoor_mask = out.apply(is_indoor_game, axis=1)
    for c in cols:
        out.loc[indoor_mask, c] = np.nan
    out["is_indoor_game"] = indoor_mask.astype(int)
    return out


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize weather columns and apply indoor nulling."""
    out = df.copy()
    # Harmonize names
    rename = {}
    if "weather_temp" in out.columns and "temp" not in out.columns:
        rename["weather_temp"] = "temp"
    if "weather_wind" in out.columns and "wind" not in out.columns:
        rename["weather_wind"] = "wind"
    if rename:
        out = out.rename(columns=rename)
    out = null_indoor_weather(out)
    # Model-friendly numeric fills: indoor -> neutral 0 wind / missing temp flagged
    if "temp" in out.columns:
        out["temp_missing"] = out["temp"].isna().astype(int)
        out["temp_filled"] = out["temp"].fillna(70.0)  # neutral after nulling
    if "wind" in out.columns:
        out["wind_missing"] = out["wind"].isna().astype(int)
        out["wind_filled"] = out["wind"].fillna(0.0)
    return out
