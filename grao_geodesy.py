"""
Project G-R.A.O. geodesy module v3.

Purpose
-------
This module adds an IERS-2010-oriented solid Earth tide layer and practical
hooks for comparison against IGETS / local gravimeter data.

Scientific note
---------------
The built-in model is an open, transparent approximation of the degree-2
solid Earth tide vertical displacement and gravity tide proxy. It uses:
- Skyfield/JPL ephemerides for Moon/Sun topocentric positions
- degree-2 Legendre pattern P2(sin altitude)
- inverse-cube distance scaling
- nominal Love-number style vertical scaling

It is NOT a full replacement for official IERS 2010 routines. For higher-grade
processing, the code can optionally call specialized libraries when installed
(e.g. PySolid / pyTMD / pygtide), and it can ingest external IGETS or local
superconducting/relative gravimeter CSV data for validation.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
from skyfield.api import load, wgs84

DATA_DIR = Path("data")
GEODESY_DIR = DATA_DIR / "geodesy"
GEODESY_DIR.mkdir(parents=True, exist_ok=True)
GRAVIMETER_FILE = GEODESY_DIR / "gravimeter_observations.csv"

# Skyfield cache. de421 is sufficient for the preview platform; for publication
# work consider de440/de441 if long-term ephemeris consistency is required.
ts = load.timescale()
eph = load("de421.bsp")
earth = eph["earth"]
moon = eph["moon"]
sun = eph["sun"]


def normalize(value, min_val, max_val):
    if value is None:
        return 0.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    value = max(min_val, min(value, max_val))
    if max_val == min_val:
        return 0.0
    return round(((value - min_val) / (max_val - min_val)) * 100, 2)


def _to_skyfield_time(dt: Optional[datetime] = None):
    dt = dt or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)


def iers2010_solid_earth_tide_approx(lat: float, lon: float, when: Optional[datetime] = None) -> Dict:
    """
    IERS-2010-oriented approximation for solid Earth tide screening.

    Outputs are designed for IRGA correlation work:
    - vertical_displacement_cm_approx: approximate body-tide vertical displacement
    - gravity_tide_microgal_proxy: approximate local gravity-tide signal proxy
    - tide_intensity_score: normalized 0-100 intensity window score

    The formulas are deliberately transparent and conservative. Use them to
    select observation windows; validate against pygtide/PySolid/IGETS/local data.
    """
    t = _to_skyfield_time(when)
    site = earth + wgs84.latlon(float(lat), float(lon))

    moon_app = site.at(t).observe(moon).apparent()
    sun_app = site.at(t).observe(sun).apparent()
    moon_alt, moon_az, _ = moon_app.altaz()
    sun_alt, sun_az, _ = sun_app.altaz()

    earth_pos = earth.at(t)
    moon_distance_km = earth_pos.observe(moon).distance().km
    sun_distance_km = earth_pos.observe(sun).distance().km

    # Degree-2 tide-generating pattern. P2 can be positive or negative; we keep
    # signed values for model comparison and absolute values for intensity score.
    moon_p2 = 0.5 * (3 * math.sin(moon_alt.radians) ** 2 - 1)
    sun_p2 = 0.5 * (3 * math.sin(sun_alt.radians) ** 2 - 1)

    # Inverse-cube scaling around mean distances.
    moon_scale = (384400.0 / moon_distance_km) ** 3
    sun_scale = (149597870.7 / sun_distance_km) ** 3

    # Nominal vertical body tide amplitudes. These constants are not official
    # station corrections; they are a practical preview scale for cm-level body
    # tide displacement. Full IERS processing also includes frequency-dependent
    # Love numbers, ocean loading, pole tide, atmospheric loading, etc.
    moon_vertical_cm = 28.0 * moon_p2 * moon_scale
    sun_vertical_cm = 13.0 * sun_p2 * sun_scale
    vertical_displacement_cm = moon_vertical_cm + sun_vertical_cm

    # Gravity tide proxy, scaled in microGal range for screening/correlation.
    # Typical tidal gravity variations are order tens to hundreds microGal.
    gravity_tide_microgal_proxy = 110.0 * moon_p2 * moon_scale + 50.0 * sun_p2 * sun_scale

    abs_intensity = abs(110.0 * moon_p2 * moon_scale) + abs(50.0 * sun_p2 * sun_scale)
    tide_intensity_score = normalize(abs_intensity, 0, 170)

    return {
        "model": "IERS2010_oriented_degree2_preview",
        "status": "approximation_not_full_IERS2010",
        "vertical_displacement_cm_approx": round(vertical_displacement_cm, 4),
        "moon_vertical_cm_component": round(moon_vertical_cm, 4),
        "sun_vertical_cm_component": round(sun_vertical_cm, 4),
        "gravity_tide_microgal_proxy": round(gravity_tide_microgal_proxy, 4),
        "tide_intensity_score": tide_intensity_score,
        "moon_altitude_deg": round(moon_alt.degrees, 5),
        "moon_azimuth_deg": round(moon_az.degrees, 5),
        "sun_altitude_deg": round(sun_alt.degrees, 5),
        "sun_azimuth_deg": round(sun_az.degrees, 5),
        "moon_distance_km": round(moon_distance_km, 3),
        "sun_distance_km": round(sun_distance_km, 3),
        "moon_p2": round(moon_p2, 8),
        "sun_p2": round(sun_p2, 8),
        "assumptions": [
            "degree-2 tide-generating potential proxy",
            "nominal Love-number-style vertical scaling",
            "inverse-cube Moon/Sun distance scaling",
            "no ocean loading / pole tide / atmospheric loading correction in built-in preview",
        ],
    }


def optional_specialized_tide_model(lat: float, lon: float, when: Optional[datetime] = None) -> Dict:
    """
    Detect optional geodetic/tide libraries and report availability.

    This function intentionally avoids hard dependency on heavy scientific
    packages so the web app remains easy to deploy. If installed, users can
    extend this block to call the package-specific APIs.
    """
    available = {}
    for pkg in ["pygtide", "pysolid", "pyTMD", "pytmd"]:
        try:
            __import__(pkg)
            available[pkg] = True
        except Exception:
            available[pkg] = False

    recommendation = []
    if not any(available.values()):
        recommendation.append("No specialized tide package detected. Using built-in IERS2010-oriented preview model.")
        recommendation.append("For validation-grade work consider installing pygtide, PySolid or pyTMD where compatible with your system.")
    else:
        recommendation.append("At least one specialized package is available; integrate its official API for station-specific tide predictions.")

    return {
        "available_packages": available,
        "active_model": "built_in_preview_until_package_adapter_is_enabled",
        "recommendation": recommendation,
    }


def geodesy_index(lat: float, lon: float, when: Optional[datetime] = None) -> Dict:
    built_in = iers2010_solid_earth_tide_approx(lat, lon, when)
    optional = optional_specialized_tide_model(lat, lon, when)
    return {
        "Geodesy_Index": built_in["tide_intensity_score"],
        "solid_earth_tide": built_in,
        "specialized_libraries": optional,
    }


def append_gravimeter_observation(payload: Dict) -> Dict:
    """
    Append local/IGETS-like gravimeter observations.

    Expected fields:
      timestamp_utc, station, lat, lon,
      observed_gravity_microgal, pressure_hpa, instrument, source,
      quality_flag
    Optional fields are stored inside raw_json.
    """
    timestamp = payload.get("timestamp_utc") or datetime.now(timezone.utc).isoformat()
    row = {
        "timestamp_utc": timestamp,
        "station": payload.get("station") or payload.get("location_name") or "unknown",
        "lat": payload.get("lat"),
        "lon": payload.get("lon"),
        "observed_gravity_microgal": payload.get("observed_gravity_microgal"),
        "pressure_hpa": payload.get("pressure_hpa"),
        "instrument": payload.get("instrument", "unknown"),
        "source": payload.get("source", "local_or_igets_import"),
        "quality_flag": payload.get("quality_flag", "unchecked"),
        "raw_json": json.dumps(payload, ensure_ascii=False),
    }
    exists = GRAVIMETER_FILE.exists()
    with open(GRAVIMETER_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    return {"status": "saved", "file": str(GRAVIMETER_FILE), "row": row}


def import_gravimeter_csv(csv_path: str, source: str = "igets_or_local_csv") -> Dict:
    """Import CSV rows into the local gravimeter observation store."""
    path = Path(csv_path)
    if not path.exists():
        return {"status": "error", "message": f"File not found: {csv_path}"}
    df = pd.read_csv(path)
    count = 0
    for rec in df.to_dict(orient="records"):
        rec["source"] = rec.get("source") or source
        append_gravimeter_observation(rec)
        count += 1
    return {"status": "imported", "rows": count, "target_file": str(GRAVIMETER_FILE)}


def compare_gravimeter_with_model(limit: int = 2000, station: Optional[str] = None) -> Dict:
    """Compare observed local gravity with built-in solid Earth tide proxy."""
    if not GRAVIMETER_FILE.exists():
        return {"status": "no_gravimeter_data", "file": str(GRAVIMETER_FILE)}
    df = pd.read_csv(GRAVIMETER_FILE).tail(limit)
    if station:
        df = df[df["station"] == station]
    if df.empty:
        return {"status": "no_matching_rows", "station": station}

    required = ["timestamp_utc", "lat", "lon", "observed_gravity_microgal"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        return {"status": "missing_columns", "missing_columns": missing_cols}

    rows = []
    for _, row in df.iterrows():
        try:
            dt = pd.to_datetime(row["timestamp_utc"], utc=True).to_pydatetime()
            lat = float(row["lat"])
            lon = float(row["lon"])
            obs = float(row["observed_gravity_microgal"])
            model = iers2010_solid_earth_tide_approx(lat, lon, dt)
            pred = float(model["gravity_tide_microgal_proxy"])
            rows.append({"obs": obs, "pred": pred, "residual": obs - pred})
        except Exception:
            continue

    if len(rows) < 5:
        return {"status": "not_enough_valid_rows", "valid_rows": len(rows), "minimum_rows": 5}

    obs_vals = pd.Series([r["obs"] for r in rows], dtype="float64")
    pred_vals = pd.Series([r["pred"] for r in rows], dtype="float64")
    residuals = obs_vals - pred_vals
    corr = obs_vals.corr(pred_vals)
    rmse = math.sqrt(float((residuals**2).mean()))

    return {
        "status": "ok",
        "station": station or "all",
        "points": len(rows),
        "correlation_observed_vs_model": round(float(corr), 6) if not math.isnan(corr) else None,
        "rmse_microgal": round(rmse, 6),
        "mean_residual_microgal": round(float(residuals.mean()), 6),
        "std_residual_microgal": round(float(residuals.std()), 6),
        "model": "built_in_IERS2010_oriented_degree2_preview",
        "publication_note": "For publication claims, compare against official/validated tide software and instrument corrections before interpreting residuals.",
    }
