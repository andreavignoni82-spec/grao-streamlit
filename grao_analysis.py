"""
Project G-R.A.O. analysis module.

Adds:
- History loading
- Simple FFT / periodogram analysis for 12h, 24h, 14.77d, 29.53d bands
- Sensor ingestion storage helpers
- Resonant vs non-resonant comparison metrics

This is an exploratory research tool, not a proof of new physics or energy generation.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
HISTORY_DIR = DATA_DIR / "history"
SENSOR_DIR = DATA_DIR / "sensors"
SENSOR_DIR.mkdir(parents=True, exist_ok=True)
SENSOR_FILE = SENSOR_DIR / "sensor_readings.csv"

TARGET_PERIODS_HOURS = {
    "12h_tidal_atmospheric": 12.42,
    "24h_diurnal_tidal": 24.84,
    "14_77d_syzygy": 14.77 * 24,
    "29_53d_lunar_month": 29.53 * 24,
}


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_history(history_file: Path) -> pd.DataFrame:
    if not history_file.exists():
        return pd.DataFrame()
    df = pd.read_csv(history_file)
    if df.empty or "timestamp_utc" not in df.columns:
        return pd.DataFrame()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"])
    return df


def periodogram_for_location(history_file: Path, location_name: Optional[str] = None, value_col: str = "irga") -> Dict:
    """
    Lightweight periodogram using numpy FFT.
    Requires at least 8 points and reasonably regular sampling.
    Returns power around target tidal/lunar bands.
    """
    df = load_history(history_file)
    if df.empty or value_col not in df.columns:
        return {"status": "no_history", "message": "No historical data available yet."}

    if location_name:
        df = df[df["name"] == location_name]

    df = df.sort_values("timestamp_utc")
    if len(df) < 8:
        return {"status": "not_enough_data", "points": int(len(df)), "minimum_points": 8}

    # Collapse duplicate timestamps, convert to numeric values.
    series = df[["timestamp_utc", value_col]].dropna().copy()
    series[value_col] = pd.to_numeric(series[value_col], errors="coerce")
    series = series.dropna()

    if len(series) < 8:
        return {"status": "not_enough_numeric_data", "points": int(len(series))}

    # Sampling interval in hours from median delta.
    deltas = series["timestamp_utc"].diff().dropna().dt.total_seconds() / 3600
    sample_hours = float(deltas.median()) if len(deltas) else 12.0
    if sample_hours <= 0:
        sample_hours = 12.0

    y = series[value_col].to_numpy(dtype=float)
    y = y - np.mean(y)
    n = len(y)

    freqs = np.fft.rfftfreq(n, d=sample_hours)  # cycles/hour
    spectrum = np.abs(np.fft.rfft(y)) ** 2

    bands = []
    for label, period_h in TARGET_PERIODS_HOURS.items():
        target_freq = 1.0 / period_h
        idx = int(np.argmin(np.abs(freqs - target_freq)))
        bands.append({
            "band": label,
            "target_period_hours": round(period_h, 3),
            "nearest_period_hours": round(float(1 / freqs[idx]), 3) if freqs[idx] > 0 else None,
            "power": round(float(spectrum[idx]), 6),
        })

    top_indices = np.argsort(spectrum)[-8:][::-1]
    peaks = []
    for idx in top_indices:
        if freqs[idx] == 0:
            continue
        peaks.append({
            "period_hours": round(float(1 / freqs[idx]), 3),
            "frequency_cycles_per_hour": round(float(freqs[idx]), 8),
            "power": round(float(spectrum[idx]), 6),
        })

    return {
        "status": "ok",
        "location": location_name or "all",
        "value_col": value_col,
        "points": int(n),
        "sample_hours_median": round(sample_hours, 3),
        "target_bands": bands,
        "top_periodogram_peaks": peaks,
        "warning": "FFT output is exploratory. Use longer, evenly sampled datasets for publication-grade claims.",
    }


def append_sensor_reading(payload: Dict) -> Dict:
    """Append ESP32/Raspberry/sensor reading to CSV."""
    SENSOR_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = payload.get("timestamp_utc") or datetime.now(timezone.utc).isoformat()

    row = {
        "timestamp_utc": timestamp,
        "device_id": payload.get("device_id", "unknown"),
        "location_name": payload.get("location_name", "unknown"),
        "lat": payload.get("lat"),
        "lon": payload.get("lon"),
        "sensor_type": payload.get("sensor_type", "generic"),
        "resonant_value": payload.get("resonant_value"),
        "control_value": payload.get("control_value"),
        "irga": payload.get("irga"),
        "irga_band": payload.get("irga_band"),
        "vibration_rms": payload.get("vibration_rms"),
        "tilt_deg": payload.get("tilt_deg"),
        "magnetic_uT": payload.get("magnetic_uT"),
        "temperature_c": payload.get("temperature_c"),
        "pressure_hpa": payload.get("pressure_hpa"),
        "raw_json": json.dumps(payload, ensure_ascii=False),
    }

    fieldnames = list(row.keys())
    exists = SENSOR_FILE.exists()
    with open(SENSOR_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

    return {"status": "saved", "file": str(SENSOR_FILE), "row": row}


def latest_sensor_readings(limit: int = 100) -> Dict:
    if not SENSOR_FILE.exists():
        return {"status": "no_sensor_data", "readings": []}
    df = pd.read_csv(SENSOR_FILE).tail(limit)
    return {"status": "ok", "count": int(len(df)), "readings": df.to_dict(orient="records")}


def compare_resonant_control(limit: int = 1000) -> Dict:
    """
    Compare resonant and non-resonant/control channels.

    Minimum scientific criterion implemented here:
      IRGA high -> resonant signal increase -> no equivalent increase on control.

    If sensor rows include an ``irga`` value, the function separates high-IRGA
    rows (>=75) from baseline rows (<60). If ``irga`` is missing, it still
    returns global paired-channel metrics but cannot validate the criterion.
    """
    if not SENSOR_FILE.exists():
        return {"status": "no_sensor_data"}
    df = pd.read_csv(SENSOR_FILE).tail(limit)
    for col in ["resonant_value", "control_value", "irga"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df = df.dropna(subset=["resonant_value", "control_value"])
    if len(df) < 5:
        return {"status": "not_enough_paired_data", "points": int(len(df)), "minimum_points": 5}

    resonant = df["resonant_value"].to_numpy(dtype=float)
    control = df["control_value"].to_numpy(dtype=float)
    diff = resonant - control
    ratio = np.mean(np.abs(resonant)) / max(np.mean(np.abs(control)), 1e-9)

    result = {
        "status": "ok",
        "points": int(len(df)),
        "resonant_mean_abs": round(float(np.mean(np.abs(resonant))), 6),
        "control_mean_abs": round(float(np.mean(np.abs(control))), 6),
        "amplification_ratio_abs_mean": round(float(ratio), 6),
        "mean_difference": round(float(np.mean(diff)), 6),
        "std_difference": round(float(np.std(diff)), 6),
        "scientific_minimum_criterion": {
            "definition": "IRGA high -> resonant signal increase -> no equivalent increase on control",
            "high_irga_threshold": 75,
            "baseline_irga_threshold": 60,
            "validated": False,
            "reason": "IRGA-conditioned validation requires sensor rows with irga values."
        },
        "interpretation": "ratio > 1 suggests the resonant channel is more responsive than the control channel; IRGA-conditioned validation is stronger.",
    }

    if "irga" in df.columns and df["irga"].notna().sum() >= 10:
        high = df[df["irga"] >= 75]
        base = df[df["irga"] < 60]
        if len(high) >= 3 and len(base) >= 3:
            high_res = float(np.mean(np.abs(high["resonant_value"])))
            base_res = float(np.mean(np.abs(base["resonant_value"])))
            high_ctrl = float(np.mean(np.abs(high["control_value"])))
            base_ctrl = float(np.mean(np.abs(base["control_value"])))
            resonant_gain = high_res / max(base_res, 1e-9)
            control_gain = high_ctrl / max(base_ctrl, 1e-9)
            differential_gain = resonant_gain / max(control_gain, 1e-9)

            validated = bool(resonant_gain >= 1.20 and control_gain <= 1.10 and differential_gain >= 1.15)
            result["scientific_minimum_criterion"] = {
                "definition": "IRGA high -> resonant signal increase -> no equivalent increase on control",
                "high_irga_threshold": 75,
                "baseline_irga_threshold": 60,
                "high_points": int(len(high)),
                "baseline_points": int(len(base)),
                "resonant_gain_high_vs_baseline": round(resonant_gain, 6),
                "control_gain_high_vs_baseline": round(control_gain, 6),
                "differential_gain": round(differential_gain, 6),
                "validated": validated,
                "decision_rule": "provisional: resonant_gain>=1.20, control_gain<=1.10, differential_gain>=1.15",
                "publication_note": "This is a screening rule. Publication-grade validation should include confidence intervals, permutation tests, longer time series and independent replications.",
            }
        else:
            result["scientific_minimum_criterion"]["reason"] = "Need at least 3 high-IRGA and 3 baseline paired rows."

    return result
