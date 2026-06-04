"""
Project G-R.A.O. Scientific Preview v1.0
Geo-Atmospheric Resonance Index utilities.

This module defines exploratory indices for hypothesis generation only.
It does not prove energy production or new physics.
"""

import math
from datetime import datetime, timezone
from skyfield.api import load, wgs84
from skyfield import almanac

# Skyfield downloads/cache de421.bsp on first run.
ts = load.timescale()
eph = load("de421.bsp")
earth = eph["earth"]
moon = eph["moon"]
sun = eph["sun"]


def normalize(value, min_val, max_val):
    """Normalize numeric value to 0-100, clipped."""
    if value is None:
        return 0.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    value = max(min_val, min(value, max_val))
    return round(((value - min_val) / (max_val - min_val)) * 100, 2)


def astronomy_index(lat, lon=0.0):
    """
    Astronomical/tidal proxy index.

    Components:
    - Moon phase angle: 0° new moon, 180° full moon.
    - Syzygy score: high near new/full moon.
    - Earth-Moon distance: high near perigee.
    - Earth-Sun distance: small seasonal contribution.
    - Semidiurnal/diurnal timing proxy.

    Note: this is NOT a full IERS Earth tide model.
    It is an open proxy for screening possible observation windows.
    """
    now = datetime.now(timezone.utc)
    t = ts.utc(now.year, now.month, now.day, now.hour, now.minute, now.second)

    phase_angle = almanac.moon_phase(eph, t).degrees

    # Local topocentric position for an IERS-inspired solid Earth tide proxy.
    # This is NOT a full IERS 2010 implementation, but it follows the physical
    # idea that the local tide-generating potential depends on body direction
    # and inverse cube distance.
    site = earth + wgs84.latlon(float(lat), float(lon))
    moon_alt, _, _ = site.at(t).observe(moon).apparent().altaz()
    sun_alt, _, _ = site.at(t).observe(sun).apparent().altaz()
    syzygy_score = abs(math.cos(math.radians(phase_angle))) * 100

    earth_pos = earth.at(t)
    moon_distance_km = earth_pos.observe(moon).distance().km
    sun_distance_km = earth_pos.observe(sun).distance().km

    moon_distance_score = 100 - normalize(moon_distance_km, 356500, 406700)
    sun_distance_score = 100 - normalize(sun_distance_km, 147_100_000, 152_100_000)

    now_hour = now.hour + now.minute / 60 + now.second / 3600
    semidiurnal_score = abs(math.sin(2 * math.pi * now_hour / 12.42)) * 100
    diurnal_score = abs(math.sin(2 * math.pi * now_hour / 24.84)) * 100

    # Degree-2 tide-generating pattern P2(sin altitude) = 0.5*(3sin²(h)-1).
    # The absolute value is used because we screen intensity windows, not sign.
    moon_p2 = abs(0.5 * (3 * math.sin(moon_alt.radians) ** 2 - 1))
    sun_p2 = abs(0.5 * (3 * math.sin(sun_alt.radians) ** 2 - 1))

    # Relative distance scaling. Lunar tide dominates; solar tide is smaller
    # but significant. Reference values are approximate mean distances.
    moon_inverse_cube = (384400 / moon_distance_km) ** 3
    sun_inverse_cube = (149597870 / sun_distance_km) ** 3
    local_tide_proxy = normalize(85 * moon_p2 * moon_inverse_cube + 35 * sun_p2 * sun_inverse_cube, 0, 120)

    latitude_factor = 0.65 + 0.35 * abs(math.cos(math.radians(lat)))

    g_index = (
        0.30 * syzygy_score
        + 0.20 * moon_distance_score
        + 0.08 * sun_distance_score
        + 0.22 * local_tide_proxy
        + 0.13 * semidiurnal_score
        + 0.07 * diurnal_score
    ) * latitude_factor

    return {
        "G_Index": round(g_index, 2),
        "moon_phase_angle_deg": round(phase_angle, 2),
        "syzygy_score": round(syzygy_score, 2),
        "moon_distance_km": round(moon_distance_km, 0),
        "moon_distance_score": round(moon_distance_score, 2),
        "sun_distance_km": round(sun_distance_km, 0),
        "sun_distance_score": round(sun_distance_score, 2),
        "moon_altitude_deg": round(moon_alt.degrees, 3),
        "sun_altitude_deg": round(sun_alt.degrees, 3),
        "local_tide_proxy": round(local_tide_proxy, 2),
        "semidiurnal_score": round(semidiurnal_score, 2),
        "diurnal_score": round(diurnal_score, 2),
    }


def atmosphere_index(meteo):
    """Atmospheric forcing proxy from pressure, temperature and wind ranges."""
    pressure_score = normalize(meteo.get("pressure_range_24h"), 0, 8)
    temp_score = normalize(meteo.get("temp_range_24h"), 0, 25)
    wind_score = normalize(meteo.get("wind_max_24h"), 0, 80)

    a_index = 0.45 * pressure_score + 0.35 * temp_score + 0.20 * wind_score

    return {
        "A_Index": round(a_index, 2),
        "pressure_score": pressure_score,
        "temperature_score": temp_score,
        "wind_score": wind_score,
    }


def magnetic_index(kp):
    """Geomagnetic proxy from NOAA planetary K-index, normalized 0-100."""
    m_index = normalize(kp, 0, 9)
    return {"M_Index": m_index, "kp": kp}


def seismic_index(quakes):
    """Local seismic activity proxy for previous 24h."""
    raw = quakes.get("quake_count_24h", 0) * 5 + quakes.get("quake_max_mag_24h", 0) * 10
    s_index = normalize(raw, 0, 100)
    return {
        "S_Index": s_index,
        "quake_count_24h": quakes.get("quake_count_24h", 0),
        "quake_max_mag_24h": quakes.get("quake_max_mag_24h", 0),
    }


def compute_irga_v1(g, a, m, s):
    """Combined experimental IRGA score."""
    irga = 0.38 * g["G_Index"] + 0.34 * a["A_Index"] + 0.16 * m["M_Index"] + 0.12 * s["S_Index"]
    return round(irga, 2)


def irga_band(score):
    if score >= 81:
        return "critical_window"
    if score >= 61:
        return "interesting_window"
    if score >= 31:
        return "observable_window"
    return "ordinary_background"
