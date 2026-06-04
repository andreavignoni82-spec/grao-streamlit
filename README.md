# Project G-R.A.O. Streamlit Dashboard v4

Dashboard scientifica sperimentale per l'indice IRGA: Geo-Atmospheric Resonance Observatory.

## Funzioni incluse

- Aggiornamento automatico ogni 12 ore tramite `st.cache_data(ttl=43200)`.
- Pulsante manuale `Aggiorna ora`.
- Mappa globale interattiva Plotly.
- Grafici storici IRGA e componenti.
- Analisi Fourier/periodogrammi su bande 12h, 24h, 14.77 giorni, 29.53 giorni.
- Modulo geodetico preview IERS-2010 oriented per solid Earth tides.
- Struttura per confronto con IGETS/gravimetri locali.
- Confronto sensore risonante vs sensore di controllo.

## Avvio locale

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Pubblicazione su Streamlit Cloud

1. Crea un repository GitHub, per esempio `grao-streamlit`.
2. Carica tutti i file di questa cartella.
3. Vai su https://share.streamlit.io
4. Clicca `New app`.
5. Seleziona repository, branch `main`, file principale `streamlit_app.py`.
6. Premi `Deploy`.

## Aggiornamento dati ogni 12 ore

La dashboard aggiorna i dati tramite:

```python
@st.cache_data(ttl=43200)
def load_live_payload():
    return update_all()
```

`43200` secondi = 12 ore.

Se vuoi forzare l'aggiornamento usa il pulsante nella sidebar.

## Nota scientifica

IRGA è un indice esplorativo. Non prova nuova energia, non prova nuova fisica e non sostituisce modelli geodetici ufficiali completi. Serve per generare ipotesi e raccogliere serie storiche utili a test futuri.

## Criterio scientifico minimo

```text
IRGA alto -> aumento segnale sensore risonante -> nessun aumento equivalente sul sensore di controllo
```

Il confronto è nella sezione `Sensori risonante vs controllo`.
