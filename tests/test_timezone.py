"""Unit tests: timezone delta hours."""

from fantasy_model.features.timezone import timezone_delta_hours


def test_west_to_east_positive():
    # SEA UTC-8 -> NYG UTC-5 => +3
    assert timezone_delta_hours("SEA", "NYG") == 3.0


def test_east_to_west_negative():
    assert timezone_delta_hours("BUF", "SF") == -3.0


def test_same_zone_zero():
    assert timezone_delta_hours("KC", "DAL") == 0.0


def test_home_game_zero_vs_self():
    assert timezone_delta_hours("PHI", "PHI") == 0.0
