"""Injury status + historical analog features (statistical, not medical)."""

from __future__ import annotations

import pandas as pd

from fantasy_model.features.history import prior_weeks_mean

# Coarse status codes used in practice reports / nflverse injury feeds.
# Higher = less likely to produce fantasy points. Statistical only.
STATUS_SEVERITY = {
    "": 0,
    "nan": 0,
    "none": 0,
    "active": 0,
    "probable": 1,
    "questionable": 2,
    "doubtful": 3,
    "out": 4,
    "ir": 5,
    "injured reserve": 5,
    "pup": 5,
    # Practice designations (nflverse)
    "full participation in practice": 0,
    "limited participation in practice": 2,
    "did not practice": 3,
    "did not participate in practice": 3,
    "rest": 1,
}


def normalize_injury_status(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().lower()


def injury_severity(status: object) -> int:
    key = normalize_injury_status(status)
    if key in STATUS_SEVERITY:
        return STATUS_SEVERITY[key]
    # Fuzzy contains
    if "out" in key or "injured reserve" in key:
        return 4
    if "doubt" in key:
        return 3
    if "question" in key or "limited" in key:
        return 2
    if "did not" in key:
        return 3
    if "full participation" in key or "probable" in key:
        return 0 if "full" in key else 1
    return 0


def add_injury_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add injury_severity and position/injury-type historical analog delta.

    Analog: mean fantasy points in prior games with same position + injury bucket,
    computed with expanding mean shifted by 1 (no leakage).
    """
    out = df.copy()
    status_col = None
    for c in ("injury_status", "report_status", "status"):
        if c in out.columns:
            status_col = c
            break
    if status_col is None:
        out["injury_status_norm"] = ""
        out["injury_severity"] = 0
    else:
        out["injury_status_norm"] = out[status_col].map(normalize_injury_status)
        out["injury_severity"] = out["injury_status_norm"].map(injury_severity)

    type_col = "injury_type" if "injury_type" in out.columns else None
    if type_col:
        out["injury_type_norm"] = out[type_col].fillna("").astype(str).str.lower().str.strip()
    else:
        out["injury_type_norm"] = ""

    if "fantasy_points" in out.columns and "position" in out.columns and {"season", "week"} <= set(out.columns):
        # Prior-WEEKS means (row-wise shift leaked same-week outcomes of other players in the group)
        out["injury_analog_fp"] = prior_weeks_mean(out, ["position", "injury_severity"], "fantasy_points", min_count=3)
        overall = prior_weeks_mean(out, [], "fantasy_points", min_count=10)
        out["injury_analog_delta"] = out["injury_analog_fp"] - overall
        out["injury_analog_fp"] = out["injury_analog_fp"].fillna(0.0)
        out["injury_analog_delta"] = out["injury_analog_delta"].fillna(0.0)
    else:
        out["injury_analog_fp"] = 0.0
        out["injury_analog_delta"] = 0.0
    return out
