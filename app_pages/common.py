"""Shared, cached loaders and UI helpers for the Streamlit pages."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.config import load_config, resolve
from src.predict import ModelUnavailable, available_models, load_artefact
from src.tune_models import MODEL_LABELS

CFG = load_config()

MOBILE_CSS = """
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1200px;}
  [data-testid="stMetricValue"] {font-size: 1.45rem;}
  [data-testid="stMetricLabel"] p {font-size: 0.82rem;}
  .f1-note {font-size: 0.86rem; color: #5b6170; line-height: 1.45;}
  .f1-tag {display:inline-block; padding:2px 8px; border-radius:10px; font-size:0.75rem;
           background:#fde8e7; color:#a30400; margin-right:4px;}
  @media (max-width: 640px) {
    .block-container {padding-left: 0.8rem; padding-right: 0.8rem; padding-top: 1.2rem;}
    h1 {font-size: 1.6rem !important;}
    h2 {font-size: 1.25rem !important;}
    [data-testid="stMetricValue"] {font-size: 1.15rem;}
    button, [role="tab"], [data-baseweb="select"] {min-height: 44px;}
  }
</style>
"""


def setup_page() -> None:
    st.markdown(MOBILE_CSS, unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f"<div class='f1-note'>{text}</div>", unsafe_allow_html=True)


def _proc(name: str):
    return resolve(CFG["data"]["processed_dir"]) / name


@st.cache_data(show_spinner=False)
def load_model_df() -> pd.DataFrame | None:
    p = _proc("model_dataset.csv")
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_study() -> pd.DataFrame | None:
    p = _proc("features_study.csv")
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_processed_json(name: str) -> dict | None:
    p = _proc(name)
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_result_json(name: str) -> dict | None:
    p = resolve(CFG["reports"]["results_dir"]) / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_result_csv(name: str) -> pd.DataFrame | None:
    p = resolve(CFG["reports"]["results_dir"]) / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource(show_spinner=False)
def get_artefact(name: str):
    try:
        return load_artefact(name, CFG)
    except ModelUnavailable:
        return None


def trained_models() -> list[str]:
    return available_models(CFG)


def require_data() -> pd.DataFrame | None:
    df = load_model_df()
    if df is None:
        st.error("Processed data not found. Run `python -m src.data_pipeline` (see README), then reload this page.")
    return df


def require_results() -> dict | None:
    res = load_result_json("model_results.json")
    if res is None:
        st.warning("No saved experiment results yet. Run `python -m src.train` to train the four models; "
                   "no placeholder metrics are shown.")
    return res


def model_label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def fmt(v, pct: bool = False, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    return f"{v:.1%}" if pct else f"{v:.{nd}f}"
