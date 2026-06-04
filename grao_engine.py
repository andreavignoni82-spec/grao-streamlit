"""
Data acquisition engine for Project G-R.A.O. Live.

Public sources used:
- Open-Meteo forecast API for pressure, temperature, humidity, wind.
- NOAA SWPC planetary K-index JSON for geomagnetic activity.
- USGS Earthquake API for local seismic proxy.
- Skyfield/JPL de421 ephemerides via grao_science.py.
"""

import csv
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

from grao_science import (
    astronomy_index,
    atmosphere_index,
    magnetic_index,
    seismic_index,
    compute_irga_v1,
    irga_band,
)
from grao_geodesy import geodesy_index

DATA_DIR = Path("data")
HISTORY_DIR = DATA_DIR / "history"
DATA_DIR.mkdir(exist_ok=True)
HISTORY_DIR.mkdir(exist_ok=True)

LOCATIONS_FILE = Path("locations.json")
OUTPUT_FILE = DATA_DIR / "grao_latest.json"
HISTORY_FILE = HISTORY_DIR / "grao_history.csv"


def fetch_open_meteo(lat, lon):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m"
        "&forecast_days=2"
        "&timezone=UTC"
    )
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    data = r.json()

    hourly = data["hourly"]
    pressure = hourly["surface_pressure"]
    temp = hourly["temperature_2m"]
    humidity = hourly["relative_humidity_2m"]
    wind = hourly["wind_speed_10m"]

    return {
        "pressure_now": pressure[0],
        "pressure_range_24h": round(max(pressure[:24]) - min(pressure[:24]), 3),
        "temp_now": temp[0],
        "temp_range_24h": round(max(temp[:24]) - min(temp[:24]), 3),
        "humidity_now": humidity[0],
        "wind_now": wind[0],
        "wind_max_24h": max(wind[:24]),
    }


def fetch_kp_index():
    url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
    try:
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        rows = r.json()
        latest = rows[-1]
        return float(latest[1])
    except Exception:
        return 0.0


def fetch_recent_earthquakes(lat, lon, radius_km=500):
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=24)
    url = (
        "https://earthquake.usgs.gov/fdsnws/event/1/query"
        "?format=geojson"
        f"&starttime={start.isoformat()}"
        f"&endtime={end.isoformat()}"
        f"&latitude={lat}"
        f"&longitude={lon}"
        f"&maxradiuskm={radius_km}"
        "&minmagnitude=2.5"
    )
    try:
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        events = r.json().get("features", [])
        mags = [e["properties"].get("mag") for e in events if e["properties"].get("mag") is not None]
        return {
            "quake_count_24h": len(events),
            "quake_max_mag_24h": max(mags) if mags else 0,
        }
    except Exception:
        return {"quake_count_24h": 0, "quake_max_mag_24h": 0}


def data_quality_flags(*blocks):
    missing = 0
    total = 0
    for block in blocks:
        for _, value in block.items():
            total += 1
            if value is None:
                missing += 1
    completeness = 100 if total == 0 else round(100 * (1 - missing / total), 2)
    return {
        "completeness_percent": completeness,
        "missing_values": missing,
        "quality": "high" if completeness >= 95 else "medium" if completeness >= 80 else "low",
    }


def compute_irga(location):
    lat = float(location["lat"])
    lon = float(location["lon"])

    meteo = fetch_open_meteo(lat, lon)
    kp = fetch_kp_index()
    quakes = fetch_recent_earthquakes(lat, lon)

    g = astronomy_index(lat, lon)
    geo = geodesy_index(lat, lon)
    # Blend original astronomy/tidal proxy with the IERS-2010-oriented geodesy layer.
    # This keeps backward compatibility while giving more weight to local solid Earth tide geometry.
    g["G_Index_original_astronomy"] = g.get("G_Index")
    g["G_Index"] = round(0.55 * g.get("G_Index", 0) + 0.45 * geo.get("Geodesy_Index", 0), 2)
    a = atmosphere_index(meteo)
    m = magnetic_index(kp)
    s = seismic_index(quakes)
    irga = compute_irga_v1(g, a, m, s)

    return {
        "name": location["name"],
        "type": location.get("type", "unknown"),
        "lat": lat,
        "lon": lon,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "irga": irga,
        "band": irga_band(irga),
        "indices": {**g, **a, **m, **s, "Geodesy_Index": geo.get("Geodesy_Index")},
        "geodesy": geo,
        "raw": {**meteo, **quakes, "kp_index": kp},
        "data_quality": data_quality_flags(meteo, quakes, {"kp": kp}),
        "scientific_status": {
            "model_version": "G-R.A.O. Scientific Dashboard v3.0",
            "claim_level": "Exploratory correlation index",
            "energy_claim": False,
            "publishable_use": "Hypothesis generation and field-data correlation",
            "limits": [
                "G_Index blends Skyfield/JPL astronomy with an IERS-2010-oriented degree-2 solid Earth tide preview. Full IERS 2010 station corrections require validated geodetic packages and loading corrections.",
                "Vibration is estimated by wind/seismic proxies until local sensors are added.",
                "IRGA does not prove energy production or new physics.",
            ],
        },
    }


def append_history(records):
    fieldnames = [
        "timestamp_utc", "name", "lat", "lon", "irga", "band",
        "G_Index", "G_Index_original_astronomy", "Geodesy_Index", "A_Index", "M_Index", "S_Index",
        "moon_phase_angle_deg", "moon_distance_km", "pressure_now",
        "pressure_range_24h", "temp_now", "temp_range_24h",
        "wind_max_24h", "kp_index", "quake_count_24h", "quake_max_mag_24h"
    ]
    exists = HISTORY_FILE.exists()
    with open(HISTORY_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for r in records:
            if "error" in r:
                continue
            row = {
                "timestamp_utc": r["timestamp_utc"],
                "name": r["name"],
                "lat": r["lat"],
                "lon": r["lon"],
                "irga": r["irga"],
                "band": r["band"],
                "G_Index": r["indices"].get("G_Index"),
                "G_Index_original_astronomy": r["indices"].get("G_Index_original_astronomy"),
                "Geodesy_Index": r["indices"].get("Geodesy_Index"),
                "A_Index": r["indices"].get("A_Index"),
                "M_Index": r["indices"].get("M_Index"),
                "S_Index": r["indices"].get("S_Index"),
                "moon_phase_angle_deg": r["indices"].get("moon_phase_angle_deg"),
                "moon_distance_km": r["indices"].get("moon_distance_km"),
                "pressure_now": r["raw"].get("pressure_now"),
                "pressure_range_24h": r["raw"].get("pressure_range_24h"),
                "temp_now": r["raw"].get("temp_now"),
                "temp_range_24h": r["raw"].get("temp_range_24h"),
                "wind_max_24h": r["raw"].get("wind_max_24h"),
                "kp_index": r["raw"].get("kp_index"),
                "quake_count_24h": r["raw"].get("quake_count_24h"),
                "quake_max_mag_24h": r["raw"].get("quake_max_mag_24h"),
            }
            writer.writerow(row)


def update_all():
    with open(LOCATIONS_FILE, "r", encoding="utf-8") as f:
        locations = json.load(f)

    results = []
    for loc in locations:
        try:
            results.append(compute_irga(loc))
        except Exception as exc:
            results.append({
                "name": loc.get("name", "unknown"),
                "error": str(exc),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            })

    results_sorted = sorted(results, key=lambda x: x.get("irga", -1), reverse=True)
    payload = {
        "project": "Project G-R.A.O. Live",
        "full_name": "Geo-Atmospheric Resonance Observatory",
        "description": "Exploratory dashboard for geo-atmospheric resonance windows.",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "warning": "IRGA is an experimental index. It is not proof of energy production or new physics.",
        "sources": {
            "meteo": "Open-Meteo API",
            "geomagnetic": "NOAA SWPC planetary K-index",
            "seismic": "USGS Earthquake API",
            "astronomy": "Skyfield with JPL DE421 ephemerides",
            "geodesy": "Built-in IERS-2010-oriented degree-2 solid Earth tide preview; optional PySolid/pyTMD/pygtide adapters",
        },
        "locations": results_sorted,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    append_history(results_sorted)
    return payload


if __name__ == "__main__":
    update_all()
    print(f"Updated {OUTPUT_FILE}")
