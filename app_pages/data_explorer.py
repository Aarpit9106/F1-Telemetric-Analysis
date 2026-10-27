import pandas as pd
import streamlit as st

from app_pages.common import fmt, load_processed_json, load_study, note, require_data, setup_page
from src import eda
from src.feature_engineering import FEATURE_DESCRIPTIONS, FEATURE_GROUPS
from src.visualizations import (fig_class_balance, fig_correlation, fig_deciles, fig_grid_rate, fig_grid_vs_finish,
                                fig_histogram, fig_trend)

setup_page()
st.title("Data Explorer")
df_all = require_data()
study = load_study()
if df_all is None or study is None:
    st.stop()

with st.expander("Filters", expanded=False):
    seasons = sorted(df_all["season"].unique())
    s_range = st.select_slider("Seasons", options=seasons, value=(seasons[0], seasons[-1]))
    teams = st.multiselect("Constructors", sorted(df_all["constructor_name"].unique()))
    drivers = st.multiselect("Drivers", sorted(df_all["driver_name"].unique()))
    circuits = st.multiselect("Circuits", sorted(df_all["circuit_name"].unique()))


def _filter(d: pd.DataFrame) -> pd.DataFrame:
    d = d[d["season"].between(*s_range)]
    if teams:
        d = d[d["constructor_name"].isin(teams)]
    if drivers:
        d = d[d["driver_name"].isin(drivers)]
    if circuits:
        d = d[d["circuit_name"].isin(circuits)]
    return d


df = _filter(df_all)
st_f = _filter(study)
if df.empty:
    st.warning("No rows match these filters.")
    st.stop()
note(f"Showing {len(df):,} classified entries ({df['race_id'].nunique()} races). "
     "All charts are computed live from the processed dataset. Correlations and rates are descriptive, not causal.")

t1, t2, t3, t4 = st.tabs(["Data quality", "Race & grid", "Trends", "Features"])

with t1:
    summ = (load_processed_json("pipeline_summary.json") or {}).get("cleaning", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows (filtered)", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Duplicate driver-race keys", int(df.duplicated(["season", "round", "driver_id"]).sum()))
    st.plotly_chart(fig_class_balance(eda.class_balance(df)))
    st.subheader("Excluded entries (study period, unfiltered)")
    ex = pd.DataFrame(summ.get("exclusions_by_reason", []))
    if not ex.empty:
        st.dataframe(ex, hide_index=True)
        note(f"{summ.get('rows_study_period', 0):,} started or entered rows in the study period; "
             f"{summ.get('rows_included_for_modelling', 0):,} kept for modelling. Non-classified entries are not "
             "turned into ordinary non-podium labels.")
    st.subheader("Missing values and types")
    q = eda.data_quality(df.drop(columns=["exclusion_reason"], errors="ignore"))
    st.dataframe(q[q["missing"] > 0] if st.toggle("Only columns with missing values", value=True) else q)
    mh = eda.missing_history(st_f)
    st.subheader("Missing history (rookies, new teams)")
    cols = st.columns(4)
    cols[0].metric("Rookie entries", mh["rookie_entries"])
    cols[1].metric("Returning-driver entries", mh["returning_entries"])
    cols[2].metric("Team-change entries", mh["team_change_entries"])
    cols[3].metric("New-team entries", mh["teams_without_history_entries"])
    note("Rolling-form features are left missing (not zero) when no previous race exists; the model pipeline "
         "imputes the training median and keeps explicit history-availability flags.")

with t2:
    st.plotly_chart(fig_grid_rate(eda.podium_rate_by_grid(df)))
    note("Each bar is the share of classified entries starting from that slot who finished in the top three.")
    st.plotly_chart(fig_grid_vs_finish(df))
    note("Darker cells = more entries. Points below the dashed line gained places relative to their grid slot.")
    st.subheader("Circuit podium patterns")
    cp = eda.circuit_patterns(df, min_races=2)
    st.dataframe(cp.rename(columns={"mean_podium_grid": "mean grid of podium finishers",
                                    "share_outside_top6": "podium share from grid > 6"}).round(2), hide_index=True)
    note("Circuits at the top have podium finishers starting further back on average — overtaking or "
         "incident-prone tracks — in the filtered seasons.")

with t3:
    dp = st_f.groupby(["season", "driver_name"])["podium_all_starters"].sum().reset_index(name="podiums")
    top_d = dp.groupby("driver_name")["podiums"].sum().nlargest(6).index
    st.plotly_chart(fig_trend(dp[dp["driver_name"].isin(top_d)], "season", "podiums", "driver_name",
                              "Podiums per season — top drivers in selection", "Podiums"))
    tp = st_f.groupby(["season", "constructor_name"])["podium_all_starters"].sum().reset_index(name="podiums")
    top_t = tp.groupby("constructor_name")["podiums"].sum().nlargest(6).index
    st.plotly_chart(fig_trend(tp[tp["constructor_name"].isin(top_t)], "season", "podiums", "constructor_name",
                              "Podiums per season — top constructors in selection", "Podiums"))
    ex = eda.experience_vs_podium(df)
    st.dataframe(ex.assign(rate=ex["rate"].map(lambda v: fmt(v, pct=True))).rename(
        columns={"exp_bin": "prior starts", "rate": "podium rate", "n": "entries"}), hide_index=True)
    note("Experienced drivers reach the podium more often, but this mostly reflects that successful drivers keep "
         "their seats (survivorship), not that experience causes podiums.")

with t4:
    num_feats = [c for g in "BCD" for c in FEATURE_GROUPS[g]["numeric"]] + ["grid_clean"]
    feat = st.selectbox("Feature", num_feats, index=num_feats.index("con_ppr_5"))
    st.caption(FEATURE_DESCRIPTIONS.get(feat, ""))
    st.plotly_chart(fig_histogram(df, feat))
    if df[feat].nunique() > 3:
        st.plotly_chart(fig_deciles(eda.feature_target_deciles(df, feat), feat))
    st.plotly_chart(fig_correlation(eda.correlation(df)))
    with st.expander("Feature dictionary"):
        rows = [{"group": g, "feature": c, "description": FEATURE_DESCRIPTIONS.get(c, "")}
                for g, spec in FEATURE_GROUPS.items() for c in spec["numeric"] + spec["categorical"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
