"""Leakage-safe "prior weeks" aggregates.

Row-wise ``groupby(...).shift(1).expanding()`` is only leakage-free when a group has one row per
week. For groups that contain several players in the same game (opponent defense x position,
position x injury bucket, team QBs), shifting by one *row* lets a row see same-week outcomes of
other players. These helpers aggregate to (group, season, week) first and only use strictly
earlier weeks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _time_key(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["season"], errors="coerce") * 100 + pd.to_numeric(df["week"], errors="coerce")


def prior_weeks_mean(df: pd.DataFrame, keys: list[str], value: str, min_count: int = 1) -> pd.Series:
    """Mean of ``value`` over all rows of the same ``keys`` group in strictly earlier weeks.

    NaN values (e.g. unplayed projection rows) are ignored. Returns a Series aligned to ``df``;
    NaN where fewer than ``min_count`` prior observations exist.
    """
    tmp = pd.DataFrame({"_t": _time_key(df), "_v": pd.to_numeric(df[value], errors="coerce")}, index=df.index)
    for k in keys:
        tmp[k] = df[k].to_numpy()
    gkeys = list(keys) + ["_t"]
    agg = tmp.groupby(gkeys, dropna=False)["_v"].agg(["sum", "count"]).reset_index().sort_values("_t")
    if keys:
        grp = agg.groupby(keys, dropna=False)
        agg["_psum"] = grp["sum"].cumsum() - agg["sum"]
        agg["_pcnt"] = grp["count"].cumsum() - agg["count"]
    else:
        agg["_psum"] = agg["sum"].cumsum() - agg["sum"]
        agg["_pcnt"] = agg["count"].cumsum() - agg["count"]
    agg["_mean"] = np.where(agg["_pcnt"] >= max(min_count, 1), agg["_psum"] / agg["_pcnt"].replace(0, np.nan), np.nan)
    merged = tmp.reset_index().merge(agg[gkeys + ["_mean"]], on=gkeys, how="left").set_index("index")
    return merged["_mean"].reindex(df.index)
