"""
Unit tests for the PFZ Predictor module and Flask application.
"""

import sys
import os
import json
import pytest

# Ensure repo root is on the path so imports resolve correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor.pfz_predictor import (
    predict,
    _haversine_distance,
    _historical_pfz_ratio,
    _coastal_zone_info,
)
import pandas as pd


# ── Unit: haversine ──────────────────────────────────────────────────────────

def test_haversine_same_point():
    assert _haversine_distance(10.0, 80.0, 10.0, 80.0) == 0.0


def test_haversine_known_distance():
    # Distance between (0,0) and (0,1) should be ~111 km
    dist = _haversine_distance(0, 0, 0, 1)
    assert 110 < dist < 112


# ── Unit: historical ratio ───────────────────────────────────────────────────

def test_historical_pfz_ratio_no_data():
    df = pd.DataFrame(
        columns=["latitude", "longitude", "catch_kg", "is_pfz"]
    )
    pfz, total, avg = _historical_pfz_ratio(50.0, 50.0, df)
    assert total == 0
    assert pfz == 0
    assert avg == 0.0


def test_historical_pfz_ratio_with_data():
    df = pd.DataFrame(
        {
            "latitude": [9.0, 9.1, 9.2],
            "longitude": [79.0, 79.1, 79.2],
            "catch_kg": [300, 200, 400],
            "is_pfz": [1, 0, 1],
        }
    )
    pfz, total, avg = _historical_pfz_ratio(9.1, 79.1, df)
    assert total == 3
    assert pfz == 2
    assert abs(avg - 300.0) < 1


# ── Unit: coastal zone lookup ────────────────────────────────────────────────

def test_coastal_zone_info_hit():
    df = pd.DataFrame(
        {
            "zone_name": ["Test Zone"],
            "min_lat": [8.0],
            "max_lat": [10.0],
            "min_lon": [78.0],
            "max_lon": [80.0],
            "depth_m": [50.0],
            "salinity_ppt": [35.0],
            "coast_type": ["Tropical"],
        }
    )
    zone = _coastal_zone_info(9.0, 79.0, df)
    assert zone is not None
    assert zone["zone_name"] == "Test Zone"


def test_coastal_zone_info_miss():
    df = pd.DataFrame(
        {
            "zone_name": ["Test Zone"],
            "min_lat": [8.0],
            "max_lat": [10.0],
            "min_lon": [78.0],
            "max_lon": [80.0],
            "depth_m": [50.0],
            "salinity_ppt": [35.0],
            "coast_type": ["Tropical"],
        }
    )
    zone = _coastal_zone_info(50.0, 50.0, df)
    assert zone is None


# ── Integration: predict() ───────────────────────────────────────────────────

def test_predict_returns_required_keys():
    result = predict(9.5, 79.0)
    assert "is_pfz" in result
    assert "confidence" in result
    assert "classification" in result
    assert "factors" in result
    assert "coastal_zone" in result
    assert "recommendations" in result


def test_predict_is_pfz_bool():
    result = predict(9.5, 79.0)
    assert isinstance(result["is_pfz"], bool)


def test_predict_confidence_range():
    result = predict(9.5, 79.0)
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_classification_consistent():
    result = predict(9.5, 79.0)
    if result["is_pfz"]:
        assert result["classification"] == "Potential Fishing Zone"
    else:
        assert result["classification"] == "Non-Potential Fishing Zone"


def test_predict_recommendations_non_empty():
    result = predict(9.5, 79.0)
    assert len(result["recommendations"]) >= 1


def test_predict_known_pfz_area():
    """Gulf of Mannar coordinates should be classified as PFZ."""
    result = predict(8.5, 78.2)
    assert result["is_pfz"] is True


def test_predict_non_pfz_area():
    """Sundarbans Delta has low salinity – expect non-PFZ."""
    result = predict(21.5, 88.5)
    assert result["is_pfz"] is False


def test_predict_factors_structure():
    result = predict(10.0, 79.8)
    factors = result["factors"]
    for key in [
        "sst_celsius",
        "sst_optimal",
        "chlorophyll_mg_m3",
        "chlorophyll_sufficient",
        "salinity_ppt",
        "salinity_suitable",
        "depth_m",
        "depth_adequate",
        "historical_pfz_ratio",
        "historical_records_nearby",
        "avg_historical_catch_kg",
    ]:
        assert key in factors, f"Missing factor: {key}"


# ── Flask app routes ─────────────────────────────────────────────────────────

@pytest.fixture
def client():
    from app import app as flask_app
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


def test_index_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"AquaZone" in resp.data


def test_predict_post_valid(client):
    resp = client.post(
        "/predict",
        data={"latitude": "9.5", "longitude": "79.0"},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"Prediction Result" in resp.data or b"Fishing Zone" in resp.data


def test_predict_post_missing_lat(client):
    resp = client.post(
        "/predict",
        data={"longitude": "79.0"},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"required" in resp.data.lower() or b"error" in resp.data.lower()


def test_predict_post_invalid_coords(client):
    resp = client.post(
        "/predict",
        data={"latitude": "abc", "longitude": "79.0"},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"Invalid" in resp.data or b"invalid" in resp.data


def test_api_predict_valid(client):
    resp = client.post(
        "/api/predict",
        data=json.dumps({"latitude": 9.5, "longitude": 79.0}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "is_pfz" in data
    assert "confidence" in data


def test_api_predict_missing_params(client):
    resp = client.post(
        "/api/predict",
        data=json.dumps({"latitude": 9.5}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_api_predict_out_of_range(client):
    resp = client.post(
        "/api/predict",
        data=json.dumps({"latitude": 200.0, "longitude": 79.0}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_predict_get_query_params(client):
    resp = client.get("/predict?lat=9.5&lon=79.0")
    assert resp.status_code == 200
    assert b"Fishing Zone" in resp.data
