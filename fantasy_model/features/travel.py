"""Travel distance and home/road features."""

from __future__ import annotations

import math
from typing import Mapping

import pandas as pd

# Approximate stadium coordinates (lat, lon) by team abbreviation.
# Pragmatic static table for distance features; not GPS-precise.
TEAM_COORDS: dict[str, tuple[float, float]] = {
    "ARI": (33.5275, -112.2625),
    "ATL": (33.7554, -84.4010),
    "BAL": (39.2780, -76.6227),
    "BUF": (42.7738, -78.7870),
    "CAR": (35.2258, -80.8528),
    "CHI": (41.8623, -87.6167),
    "CIN": (39.0954, -84.5160),
    "CLE": (41.5061, -81.6995),
    "DAL": (32.7473, -97.0945),
    "DEN": (39.7439, -105.0201),
    "DET": (42.3400, -83.0456),
    "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107),
    "IND": (39.7601, -86.1639),
    "JAX": (30.3239, -81.6373),
    "KC": (39.0489, -94.4839),
    "LA": (33.9535, -118.3392),
    "LAR": (33.9535, -118.3392),
    "LAC": (33.9535, -118.3392),
    "LV": (36.0908, -115.1830),
    "MIA": (25.9580, -80.2389),
    "MIN": (44.9738, -93.2575),
    "NE": (42.0909, -71.2643),
    "NO": (29.9511, -90.0812),
    "NYG": (40.8128, -74.0742),
    "NYJ": (40.8128, -74.0742),
    "PHI": (39.9008, -75.1675),
    "PIT": (40.4468, -80.0158),
    "SEA": (47.5952, -122.3316),
    "SF": (37.4032, -121.9698),
    "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713),
    "WAS": (38.9077, -76.8645),
    "WSH": (38.9077, -76.8645),
}


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two WGS84 points."""
    r = 3958.7613  # Earth radius miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def travel_distance_miles(
    team: str,
    home_team: str,
    coords: Mapping[str, tuple[float, float]] | None = None,
) -> float:
    """Miles from team's home stadium to the game stadium (home_team venue).

    Home games return 0.0. Unknown teams return 0.0 (safe default).
    """
    coords = coords or TEAM_COORDS
    def _u(x: object) -> str:
        if x is None:
            return ""
        try:
            if x != x:  # NaN
                return ""
        except Exception:
            pass
        s = str(x).strip()
        if s.lower() in {"nan", "none", ""}:
            return ""
        return s.upper()

    team_u = _u(team)
    home_u = _u(home_team)
    if not team_u or not home_u or team_u == home_u:
        return 0.0
    if team_u not in coords or home_u not in coords:
        return 0.0
    lat1, lon1 = coords[team_u]
    lat2, lon2 = coords[home_u]
    return haversine_miles(lat1, lon1, lat2, lon2)


def add_travel_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_home, travel_miles for each player-game row.

    Expects columns: team (or recent_team), home_team (or home).
    """
    out = df.copy()
    team_col = "team" if "team" in out.columns else "recent_team"
    home_col = "home_team" if "home_team" in out.columns else "home"
    if team_col not in out.columns or home_col not in out.columns:
        out["is_home"] = 0
        out["travel_miles"] = 0.0
        return out

    teams = out[team_col].astype(str).str.upper()
    homes = out[home_col].astype(str).str.upper()
    out["is_home"] = (teams == homes).astype(int)
    out["travel_miles"] = [
        travel_distance_miles(t, h) for t, h in zip(teams.tolist(), homes.tolist(), strict=False)
    ]
    return out
