"""Unit tests: dome / indoor weather nulling."""

import pandas as pd

from fantasy_model.features.weather import is_indoor_game, null_indoor_weather


def test_dome_nulls_temp_and_wind():
    df = pd.DataFrame(
        [
            {"home_team": "NO", "roof": "dome", "temp": 88.0, "wind": 15.0},
            {"home_team": "BUF", "roof": "outdoors", "temp": 28.0, "wind": 20.0},
            {"home_team": "DAL", "roof": "closed", "temp": 95.0, "wind": 12.0},
            {"home_team": "SEA", "roof": "outdoors", "temp": 50.0, "wind": 10.0},
        ]
    )
    out = null_indoor_weather(df)
    assert pd.isna(out.loc[0, "temp"]) and pd.isna(out.loc[0, "wind"])
    assert out.loc[1, "temp"] == 28.0 and out.loc[1, "wind"] == 20.0
    assert pd.isna(out.loc[2, "temp"])
    assert out.loc[3, "temp"] == 50.0
    assert out.loc[0, "is_indoor_game"] == 1
    assert out.loc[1, "is_indoor_game"] == 0


def test_retractable_open_keeps_weather():
    row = pd.Series({"home_team": "DAL", "roof": "retractable_open", "temp": 80.0, "wind": 5.0})
    assert is_indoor_game(row) is False
