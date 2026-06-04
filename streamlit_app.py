"""
Project G-R.A.O. Streamlit Dashboard v4
Geo-Atmospheric Resonance Observatory

Automatic data refresh: 12 hours via st.cache_data(ttl=43200).
Manual refresh: sidebar button clears the cache and reloads data.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

from grao_engine import update_all, OUTPUT_FILE, HISTORY_FILE
from grao_analysis import periodogram_for_location, latest_sensor_readings, compare_resonant_control
from grao_geodesy import compare_gravimeter_with_model, GRAVIMETER_FILE

st.set_page_config(
    page_title="G-R.A.O. Live Scientific Dashboard",
    page_icon="🌍",
    layout="wide",
)

CACHE_TTL_SECONDS = 12 * 60 * 60


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Aggiornamento dati globali G-R.A.O. in corso...")
def load_live_payload():
    """Update public datasets and cache the payload for 12 hours."""
    return update_all()


def load_latest_from_disk():
    if OUTPUT_FILE.exists():
        return json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
    return None


def history_df() -> pd.DataFrame:
    if not HISTORY_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(HISTORY_FILE)
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    return df


def payload_to_df(payload: dict) -> pd.DataFrame:
    rows = []
    for loc in payload.get("locations", []):
        if "error" in loc:
            rows.append({
                "name": loc.get("name"),
                "error": loc.get("error"),
                "irga": None,
            })
            continue
        indices = loc.get("indices", {})
        raw = loc.get("raw", {})
        geo = loc.get("geodesy", {}).get("solid_earth_tide", {})
        rows.append({
            "name": loc.get("name"),
            "type": loc.get("type"),
            "lat": loc.get("lat"),
            "lon": loc.get("lon"),
            "irga": loc.get("irga"),
            "band": loc.get("band"),
            "G_Index": indices.get("G_Index"),
            "Geodesy_Index": indices.get("Geodesy_Index"),
            "A_Index": indices.get("A_Index"),
            "M_Index": indices.get("M_Index"),
            "S_Index": indices.get("S_Index"),
            "pressure_now": raw.get("pressure_now"),
            "pressure_range_24h": raw.get("pressure_range_24h"),
            "temp_now": raw.get("temp_now"),
            "temp_range_24h": raw.get("temp_range_24h"),
            "kp_index": raw.get("kp_index"),
            "gravity_tide_microgal_proxy": geo.get("gravity_tide_microgal_proxy"),
            "vertical_displacement_cm_approx": geo.get("vertical_displacement_cm_approx"),
            "moon_altitude_deg": geo.get("moon_altitude_deg"),
            "sun_altitude_deg": geo.get("sun_altitude_deg"),
        })
    return pd.DataFrame(rows)


st.sidebar.title("G‑R.A.O. v4")
st.sidebar.caption("Aggiornamento automatico ogni 12 ore tramite cache Streamlit.")

if st.sidebar.button("🔄 Aggiorna ora manualmente"):
    st.cache_data.clear()
    st.rerun()

page = st.sidebar.radio(
    "Sezioni",
    [
        "Dashboard globale",
        "Storico e grafici",
        "Fourier / periodogrammi",
        "Sensori risonante vs controllo",
        "Geodesia / gravimetri",
        "Metodo scientifico",
    ],
)

payload = load_live_payload()
df = payload_to_df(payload)
updated = payload.get("updated_utc", "n/d")

st.title("🌍 Project G‑R.A.O. Live Scientific Dashboard")
st.caption("Geo‑Atmospheric Resonance Observatory — indice sperimentale, non prova di nuova energia.")
st.info(f"Ultimo aggiornamento UTC: {updated}. Cache automatica: 12 ore. Fonte dati live: Open‑Meteo, NOAA SWPC, USGS, Skyfield/JPL, modello geodetico preview IERS‑2010 oriented.")

if page == "Dashboard globale":
    c1, c2, c3, c4 = st.columns(4)
    valid = df.dropna(subset=["irga"]) if "irga" in df.columns else pd.DataFrame()
    c1.metric("Località monitorate", len(df))
    c2.metric("IRGA massimo", f"{valid['irga'].max():.2f}" if not valid.empty else "n/d")
    c3.metric("IRGA medio", f"{valid['irga'].mean():.2f}" if not valid.empty else "n/d")
    c4.metric("Finestre ≥75", int((valid["irga"] >= 75).sum()) if not valid.empty else 0)

    st.subheader("Mappa globale IRGA")
    if not valid.empty:
        fig = px.scatter_geo(
            valid,
            lat="lat",
            lon="lon",
            color="irga",
            size="irga",
            hover_name="name",
            hover_data=["type", "band", "G_Index", "A_Index", "M_Index", "Geodesy_Index"],
            projection="natural earth",
            color_continuous_scale="Turbo",
            title="Finestre di possibile risonanza geo‑atmosferica",
        )
        fig.update_layout(height=620, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Classifica località")
        st.dataframe(
            valid.sort_values("irga", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("Nessun dato IRGA valido disponibile.")

elif page == "Storico e grafici":
    hist = history_df()
    st.subheader("Storico IRGA")
    if hist.empty:
        st.warning("Lo storico è ancora vuoto. Apri/aggiorna la dashboard più volte nel tempo per popolarlo.")
    else:
        names = sorted(hist["name"].dropna().unique().tolist())
        selected = st.multiselect("Località", names, default=names[:4])
        chart_df = hist[hist["name"].isin(selected)] if selected else hist
        fig = px.line(chart_df, x="timestamp_utc", y="irga", color="name", markers=True, title="IRGA nel tempo")
        st.plotly_chart(fig, use_container_width=True)

        cols = [c for c in ["G_Index", "Geodesy_Index", "A_Index", "M_Index", "S_Index"] if c in chart_df.columns]
        if selected and cols:
            one = selected[0]
            sub = hist[hist["name"] == one]
            fig2 = px.line(sub, x="timestamp_utc", y=cols, title=f"Componenti indice — {one}")
            st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(hist.tail(500), use_container_width=True, hide_index=True)

elif page == "Fourier / periodogrammi":
    st.subheader("Analisi Fourier / periodogrammi")
    hist = history_df()
    if hist.empty:
        st.warning("Servono almeno 8 punti storici per avviare una FFT esplorativa.")
    else:
        names = [None] + sorted(hist["name"].dropna().unique().tolist())
        selected = st.selectbox("Località", names, format_func=lambda x: "Tutte" if x is None else x)
        value_col = st.selectbox("Variabile", [c for c in ["irga", "G_Index", "A_Index", "Geodesy_Index", "M_Index", "S_Index"] if c in hist.columns])
        res = periodogram_for_location(HISTORY_FILE, selected, value_col)
        st.json(res)
        if res.get("status") == "ok":
            peaks = pd.DataFrame(res.get("top_periodogram_peaks", []))
            bands = pd.DataFrame(res.get("target_bands", []))
            if not peaks.empty:
                fig = px.bar(peaks, x="period_hours", y="power", title="Picchi periodogramma principali")
                st.plotly_chart(fig, use_container_width=True)
            if not bands.empty:
                st.dataframe(bands, use_container_width=True, hide_index=True)

elif page == "Sensori risonante vs controllo":
    st.subheader("Criterio scientifico minimo")
    st.code("IRGA alto -> aumento segnale sensore risonante -> nessun aumento equivalente sul sensore di controllo")
    comp = compare_resonant_control()
    st.json(comp)
    readings = latest_sensor_readings(limit=300)
    if readings.get("status") == "ok":
        sdf = pd.DataFrame(readings["readings"])
        st.dataframe(sdf, use_container_width=True, hide_index=True)
        for col in ["resonant_value", "control_value", "irga"]:
            if col in sdf.columns:
                sdf[col] = pd.to_numeric(sdf[col], errors="coerce")
        if {"timestamp_utc", "resonant_value", "control_value"}.issubset(sdf.columns):
            sdf["timestamp_utc"] = pd.to_datetime(sdf["timestamp_utc"], utc=True, errors="coerce")
            fig = px.line(sdf, x="timestamp_utc", y=["resonant_value", "control_value"], title="Sensore risonante vs controllo")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Nessun dato sensore ancora presente. Usa il CSV data/sensors/sensor_readings.csv o estendi con API esterna.")

elif page == "Geodesia / gravimetri":
    st.subheader("Confronto modello solid Earth tide vs gravimetri locali/IGETS")
    st.caption("Il confronto usa data/geodesy/gravimeter_observations.csv se presente.")
    res = compare_gravimeter_with_model()
    st.json(res)
    if GRAVIMETER_FILE.exists():
        gdf = pd.read_csv(GRAVIMETER_FILE)
        st.dataframe(gdf.tail(500), use_container_width=True, hide_index=True)
    else:
        st.warning("File gravimetri non presente. Carica osservazioni in data/geodesy/gravimeter_observations.csv.")

elif page == "Metodo scientifico":
    st.subheader("Metodo scientifico e limiti")
    st.markdown(
        """
### Obiettivo
G‑R.A.O. cerca finestre temporali in cui forzanti gravitazionali, atmosferiche, geomagnetiche e locali possano produrre pattern misurabili.

### Criterio minimo
```text
IRGA alto -> aumento segnale sensore risonante -> nessun aumento equivalente sul sensore di controllo
```

### Cosa NON dimostra
- Non dimostra nuova energia.
- Non dimostra nuova fisica.
- Non sostituisce modelli geodetici ufficiali IERS completi.

### Cosa può diventare pubblicabile
Una correlazione ripetibile, con dati storici lunghi, controllo non risonante, intervalli di confidenza, test di permutazione e validazione indipendente.

### Aggiornamento 12 ore
La funzione `load_live_payload()` usa `@st.cache_data(ttl=43200)`: Streamlit mantiene i dati per 12 ore e poi li aggiorna automaticamente alla visita successiva.
        """
    )
    st.json(payload.get("sources", {}))
