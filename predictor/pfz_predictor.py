"""
PFZ Predictor Module
Classifies geographic coordinates as Potential Fishing Zone (PFZ) or Non-PFZ
using historical coastal, fishing zone, and marine environmental datasets.
"""

import os
import math
import pandas as pd
import numpy as np

# Dataset paths relative to this file's parent directory
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COASTAL_ZONES_PATH = os.path.join(_BASE_DIR, "data", "coastal_zones.csv")
FISHING_ZONES_PATH = os.path.join(_BASE_DIR, "data", "fishing_zones.csv")
MARINE_DATA_PATH = os.path.join(_BASE_DIR, "data", "marine_data.csv")

# Thresholds for PFZ classification
SST_MIN = 26.0          # °C – minimum sea surface temperature
SST_MAX = 30.5          # °C – maximum sea surface temperature
CHLOROPHYLL_MIN = 0.8   # mg/m³ – minimum chlorophyll-a concentration
SALINITY_MIN = 30.0     # ppt – minimum salinity
SALINITY_MAX = 36.5     # ppt – maximum salinity
DEPTH_MIN = 20.0        # m   – minimum depth for open-sea fishing
PROXIMITY_DEG = 2.0     # degrees – radius to search nearby historical records


def _load_datasets():
    """Load all three reference datasets into DataFrames."""
    coastal = pd.read_csv(COASTAL_ZONES_PATH)
    fishing = pd.read_csv(FISHING_ZONES_PATH)
    marine = pd.read_csv(MARINE_DATA_PATH)
    return coastal, fishing, marine


def _haversine_distance(lat1, lon1, lat2, lon2):
    """Return great-circle distance in km between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _nearest_marine_obs(lat, lon, marine_df):
    """Return the nearest marine observation row for the given coordinates."""
    marine_df = marine_df.copy()
    marine_df["_dist"] = marine_df.apply(
        lambda r: _haversine_distance(lat, lon, r["latitude"], r["longitude"]),
        axis=1,
    )
    return marine_df.loc[marine_df["_dist"].idxmin()]


def _coastal_zone_info(lat, lon, coastal_df):
    """
    Return coastal zone metadata for the given point, or None if not inside
    any defined coastal zone bounding box.
    """
    mask = (
        (coastal_df["min_lat"] <= lat) & (lat <= coastal_df["max_lat"])
        & (coastal_df["min_lon"] <= lon) & (lon <= coastal_df["max_lon"])
    )
    matches = coastal_df[mask]
    if matches.empty:
        return None
    return matches.iloc[0]


def _historical_pfz_ratio(lat, lon, fishing_df, radius_deg=PROXIMITY_DEG):
    """
    Return (pfz_count, total_count, avg_catch_kg) for historical fishing
    records within radius_deg degrees of the query point.
    """
    nearby = fishing_df[
        (fishing_df["latitude"].between(lat - radius_deg, lat + radius_deg))
        & (fishing_df["longitude"].between(lon - radius_deg, lon + radius_deg))
    ]
    if nearby.empty:
        return 0, 0, 0.0
    pfz_count = int(nearby["is_pfz"].sum())
    total_count = len(nearby)
    avg_catch = float(nearby["catch_kg"].mean())
    return pfz_count, total_count, avg_catch


def predict(latitude: float, longitude: float) -> dict:
    """
    Predict whether the given coordinates fall in a Potential Fishing Zone.

    Parameters
    ----------
    latitude  : float  – decimal degrees, positive = North
    longitude : float  – decimal degrees, positive = East

    Returns
    -------
    dict with keys:
        is_pfz          : bool
        confidence      : float  (0–1)
        classification  : str    ("Potential Fishing Zone" | "Non-Potential Fishing Zone")
        factors         : dict   (individual factor scores / values)
        coastal_zone    : str    (name of the matched coastal zone, or "Unknown")
        recommendations : list[str]
    """
    coastal_df, fishing_df, marine_df = _load_datasets()

    # ── Factor 1: nearest marine environment ──────────────────────────────────
    marine = _nearest_marine_obs(latitude, longitude, marine_df)
    sst = float(marine["sst_celsius"])
    chlorophyll = float(marine["chlorophyll_mg_m3"])
    salinity = float(marine["salinity_ppt"])

    sst_ok = SST_MIN <= sst <= SST_MAX
    chl_ok = chlorophyll >= CHLOROPHYLL_MIN
    sal_ok = SALINITY_MIN <= salinity <= SALINITY_MAX

    # ── Factor 2: coastal zone metadata ──────────────────────────────────────
    zone = _coastal_zone_info(latitude, longitude, coastal_df)
    coastal_zone_name = zone["zone_name"] if zone is not None else "Unknown"
    depth = float(zone["depth_m"]) if zone is not None else 0.0
    depth_ok = depth >= DEPTH_MIN

    # Exclude low-salinity zones such as river deltas / backwaters
    zone_salinity = float(zone["salinity_ppt"]) if zone is not None else salinity
    zone_sal_ok = zone_salinity >= SALINITY_MIN

    # ── Factor 3: historical fishing zone data ────────────────────────────────
    pfz_count, total_count, avg_catch = _historical_pfz_ratio(
        latitude, longitude, fishing_df
    )
    if total_count > 0:
        hist_ratio = pfz_count / total_count
    else:
        hist_ratio = 0.5  # neutral prior when no historical data exists

    hist_ok = hist_ratio >= 0.5

    # ── Composite confidence score ────────────────────────────────────────────
    weights = {
        "sst": 0.20,
        "chlorophyll": 0.25,
        "salinity": 0.15,
        "depth": 0.15,
        "historical": 0.25,
    }
    scores = {
        "sst": 1.0 if sst_ok else 0.0,
        "chlorophyll": min(chlorophyll / CHLOROPHYLL_MIN, 1.0),
        "salinity": 1.0 if sal_ok and zone_sal_ok else 0.0,
        "depth": 1.0 if depth_ok else 0.0,
        "historical": hist_ratio,
    }
    confidence = sum(weights[k] * scores[k] for k in weights)

    is_pfz = confidence >= 0.55

    # ── Recommendations ───────────────────────────────────────────────────────
    recommendations = []
    if is_pfz:
        recommendations.append("This area shows high fish availability potential.")
        if avg_catch > 300:
            recommendations.append(
                f"Historical average catch in nearby zones is {avg_catch:.0f} kg – "
                "excellent productivity."
            )
        if chlorophyll >= 1.5:
            recommendations.append(
                "High chlorophyll-a levels indicate abundant plankton – "
                "good feeding conditions for fish."
            )
        if SST_MIN + 1 <= sst <= SST_MAX - 1:
            recommendations.append(
                f"Sea surface temperature ({sst:.1f} °C) is within the optimal range."
            )
    else:
        recommendations.append("This area currently shows low fish availability potential.")
        if not sst_ok:
            recommendations.append(
                f"Sea surface temperature ({sst:.1f} °C) is outside the optimal "
                f"range ({SST_MIN}–{SST_MAX} °C)."
            )
        if not chl_ok:
            recommendations.append(
                f"Chlorophyll-a concentration ({chlorophyll:.2f} mg/m³) is below "
                f"the minimum threshold ({CHLOROPHYLL_MIN} mg/m³)."
            )
        if not sal_ok or not zone_sal_ok:
            recommendations.append(
                "Salinity levels indicate the area may be a river delta, estuary, "
                "or coastal lagoon with reduced open-sea fish populations."
            )
        if not depth_ok:
            recommendations.append(
                "Shallow water depth limits access to offshore fish species."
            )
        if total_count > 0 and hist_ratio < 0.4:
            recommendations.append(
                "Historical records show low fishing success in this region."
            )

    return {
        "is_pfz": is_pfz,
        "confidence": round(confidence, 4),
        "classification": (
            "Potential Fishing Zone" if is_pfz else "Non-Potential Fishing Zone"
        ),
        "factors": {
            "sst_celsius": round(sst, 2),
            "sst_optimal": sst_ok,
            "chlorophyll_mg_m3": round(chlorophyll, 3),
            "chlorophyll_sufficient": chl_ok,
            "salinity_ppt": round(salinity, 2),
            "salinity_suitable": sal_ok and zone_sal_ok,
            "depth_m": round(depth, 1),
            "depth_adequate": depth_ok,
            "historical_pfz_ratio": round(hist_ratio, 3),
            "historical_records_nearby": total_count,
            "avg_historical_catch_kg": round(avg_catch, 1),
        },
        "coastal_zone": coastal_zone_name,
        "recommendations": recommendations,
    }
