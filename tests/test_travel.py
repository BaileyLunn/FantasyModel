"""Unit tests: travel distance."""

from fantasy_model.features.travel import haversine_miles, travel_distance_miles


def test_home_game_zero_miles():
    assert travel_distance_miles("KC", "KC") == 0.0


def test_coast_to_coast_reasonable():
    miles = travel_distance_miles("SEA", "MIA")
    # Seattle to Miami stadium ~2700–2800 miles
    assert 2400 < miles < 3100


def test_haversine_symmetry():
    a = haversine_miles(40.0, -74.0, 34.0, -118.0)
    b = haversine_miles(34.0, -118.0, 40.0, -74.0)
    assert abs(a - b) < 1e-6


def test_nearby_afc_north():
    miles = travel_distance_miles("PIT", "CLE")
    assert 80 < miles < 200
