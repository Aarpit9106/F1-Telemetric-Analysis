import streamlit as st

from app_pages.common import (CFG, fmt, load_processed_json, load_result_csv, load_result_json, note, require_data,
                              setup_page)
from src.visualizations import fig_class_balance
from src.eda import class_balance

setup_page()
st.title("F1 Race Intelligence")
st.markdown("**Historical data analysis and podium prediction with machine learning.** "
            "Given only what was known *before* a Grand Prix started, does a driver finish in the top three?")

df = require_data()
if df is None:
    st.stop()
pipe = load_processed_json("pipeline_summary.json") or {}
clean = pipe.get("cleaning", {})

c1, c2, c3, c4 = st.columns(4)
c1.metric("Seasons", f"{df['season'].min()}–{df['season'].max()}")
c2.metric("Grands Prix", f"{df['race_id'].nunique()}")
c3.metric("Classified entries", f"{len(df):,}")
c4.metric("Podium share", fmt(df["podium"].mean(), pct=True))

tab1, tab2, tab3 = st.tabs(["Project", "Results summary", "Limitations"])
with tab1:
    st.subheader("Prediction task")
    st.markdown(
        "- **Unit of observation:** one driver in one Grand Prix.\n"
        "- **Target `podium`:** 1 if the classified finishing position is 1–3, else 0.\n"
        "- **Inclusion policy:** only entries with a numeric classified position are modelled; retirements, "
        "disqualifications and withdrawals are excluded and reported (see Data Explorer).\n"
        "- **Features:** grid slot, circuit, season/round, driver & constructor identity, rolling recent form, "
        "experience and in-season standings — all computed from earlier races only.\n"
        "- **Split (chronological):** train "
        f"{CFG['split']['train_seasons'][0]}–{CFG['split']['train_seasons'][1]}, validation "
        f"{CFG['split']['validation_seasons'][0]}–{CFG['split']['validation_seasons'][1]}, test "
        f"{CFG['split']['test_seasons'][0]}–{CFG['split']['test_seasons'][1]}.\n"
        "- **Models:** Decision Tree, K-Nearest Neighbours, Support Vector Machine, Random Forest.")
    st.plotly_chart(fig_class_balance(class_balance(df)))
    st.subheader("Data source")
    manifest_rows = clean.get("rows_all_seasons")
    note(f"{CFG['data']['source_name']} — <code>{CFG['data']['base_url']}</code>. Seasons "
         f"{CFG['data']['history_start_season']}–{CFG['data']['study_start_season'] - 1} are downloaded only to build "
         f"driver/team history; {CFG['data']['study_start_season']}–{CFG['data']['study_end_season']} are modelled. "
         f"Rows across all cached seasons: {manifest_rows:,}." if manifest_rows else "Run the data pipeline.")
with tab2:
    pref = load_result_json("preferred_model.json")
    cmp = load_result_csv("comparison.csv")
    if cmp is None or pref is None:
        st.warning("Models have not been trained yet — run `python -m src.train`. No placeholder numbers are shown.")
    else:
        st.markdown(f"**Preferred model:** {pref['label']} — chosen by *{pref['criterion']}*.")
        show = cmp[["model", "feature_set", "validation_average_precision", "test_average_precision",
                    "test_f1", "test_roc_auc", "test_top3_hit_rate"]].rename(columns={
            "validation_average_precision": "Val AP", "test_average_precision": "Test AP", "test_f1": "Test F1",
            "test_roc_auc": "Test ROC-AUC", "test_top3_hit_rate": "Test top-3 hit rate", "feature_set": "Features",
            "model": "Model"})
        st.dataframe(show.round(3), hide_index=True)
        note("AP = average precision (area under the precision-recall curve) for the podium class; the podium base "
             "rate is about 17%, so a random ranking scores ≈0.17. Top-3 hit rate = share of each race's three "
             "highest-scored drivers who actually finished on the podium.")
with tab3:
    st.markdown(
        "- Results are **conditional on being classified**: retirements cannot be predicted pre-race and are "
        "excluded from the target (a sensitivity check with all starters is on the Model Comparison page).\n"
        "- Grid slot already embeds qualifying pace; the models do not see weather, tyres, strategy, upgrades or "
        "telemetry.\n"
        "- Regulation changes (2017, 2022) shift team competitiveness; history before a change can mislead.\n"
        "- Driver/team identifiers seen in training can memorise dominant eras; unseen IDs get no identity signal.\n"
        "- Career counts start in 2000 (history seasons), so they are lower bounds for very long careers.\n"
        "- Model scores are **not calibrated probabilities** (balanced class weights distort them).\n"
        "- Timing figures depend on the machine that ran the experiments.")
