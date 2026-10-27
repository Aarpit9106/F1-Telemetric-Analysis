"""F1 Race Intelligence — Streamlit entry point.

    streamlit run app.py

The app only READS saved artefacts (processed data, trained models, experiment
results). It never retrains models on page interactions; use
`python -m src.train` for that.
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="F1 Race Intelligence", page_icon="🏁", layout="wide",
                   initial_sidebar_state="auto")

pages = [
    st.Page("app_pages/overview.py", title="Overview", icon="🏁", default=True),
    st.Page("app_pages/data_explorer.py", title="Data Explorer", icon="📊"),
    st.Page("app_pages/race_explorer.py", title="Race Explorer", icon="🗺️"),
    st.Page("app_pages/model_telemetry.py", title="Model Telemetry", icon="📈"),
    st.Page("app_pages/model_comparison.py", title="Model Comparison", icon="⚖️"),
    st.Page("app_pages/podium_predictor.py", title="Podium Predictor", icon="🏆"),
]
nav = st.navigation(pages, position="sidebar")
with st.sidebar:
    st.caption("Educational ML project. Predictions are estimates from historical data, "
               "not race forecasts or betting advice.")
nav.run()
