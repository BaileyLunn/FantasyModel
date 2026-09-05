"""Half-PPR scoring sanity checks."""

import pandas as pd

from fantasy_model.scoring import fantasy_points

HALF_PPR = {
    "pass_yd": 0.04,
    "pass_td": 4.0,
    "pass_int": -2.0,
    "rush_yd": 0.1,
    "rush_td": 6.0,
    "rec": 0.5,
    "rec_yd": 0.1,
    "rec_td": 6.0,
    "fumble_lost": -2.0,
}


def test_wr_half_ppr():
    df = pd.DataFrame(
        [
            {
                "receptions": 6,
                "receiving_yards": 80,
                "receiving_tds": 1,
                "rushing_yards": 0,
                "rushing_tds": 0,
            }
        ]
    )
    # 6*0.5 + 80*0.1 + 6 = 3 + 8 + 6 = 17
    assert abs(float(fantasy_points(df, HALF_PPR).iloc[0]) - 17.0) < 1e-6
