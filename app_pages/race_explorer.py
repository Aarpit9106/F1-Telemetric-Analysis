import streamlit as st

from app_pages.common import load_study, note, setup_page

setup_page()
st.title("Race Explorer")
note("Historical facts only — official grid, qualifying and classified results from the dataset. "
     "Model predictions live on the Podium Predictor page.")
study = load_study()
if study is None:
    st.error("Processed data not found. Run `python -m src.data_pipeline`.")
    st.stop()

c1, c2 = st.columns(2)
season = c1.selectbox("Season", sorted(study["season"].unique(), reverse=True))
races = study[study["season"] == season][["round", "race_name"]].drop_duplicates().sort_values("round")
label = c2.selectbox("Grand Prix", [f"R{r} · {n}" for r, n in zip(races["round"], races["race_name"])])
rnd = int(label.split(" ·")[0][1:])
race = study[(study["season"] == season) & (study["round"] == rnd)].copy()

c3, c4 = st.columns(2)
team_f = c3.multiselect("Constructor", sorted(race["constructor_name"].unique()))
drv_f = c4.multiselect("Driver", sorted(race["driver_name"].unique()))
if team_f:
    race = race[race["constructor_name"].isin(team_f)]
if drv_f:
    race = race[race["driver_name"].isin(drv_f)]

if race.empty:
    st.info("No entrants match these filters.")
    st.stop()

first = race.iloc[0]
m1, m2, m3 = st.columns(3)
m1.metric("Circuit", first["circuit_name"])
m2.metric("Date", str(first["date"])[:10])
m3.metric("Starters shown", len(race))

race["result"] = race.apply(lambda r: f"P{int(r['finish_position'])}" if r["classified"] else
                            f"{r['position_text']} — {r['status']}", axis=1)
race["places gained"] = (race["grid"] - race["finish_position"]).where(race["classified"])
show = race.sort_values("position")[["result", "driver_name", "constructor_name", "grid", "quali_position",
                                      "places gained", "status", "drv_career_starts", "drv_avg_pos_5"]]
st.dataframe(show.rename(columns={"driver_name": "Driver", "constructor_name": "Constructor", "grid": "Grid",
                                  "quali_position": "Qualifying", "drv_career_starts": "Prior starts",
                                  "drv_avg_pos_5": "Avg pos (prev 5)"}),
             hide_index=True, column_config={"Avg pos (prev 5)": st.column_config.NumberColumn(format="%.1f"),
                                             "places gained": st.column_config.NumberColumn(format="%+d")})
note("Grid 0 in the source means a pit-lane start. Qualifying and grid differ when penalties apply. "
     "'Prior starts' and 'Avg pos (prev 5)' are the pre-race features the models saw.")

with st.expander("Season summary"):
    s = study[study["season"] == season]
    wins = s[s["finish_position"] == 1].groupby("driver_name").size().rename("wins")
    pods = s.groupby("driver_name")["podium_all_starters"].sum().rename("podiums")
    tbl = pods.to_frame().join(wins).fillna(0).astype(int).sort_values(["podiums", "wins"], ascending=False)
    st.dataframe(tbl[tbl["podiums"] > 0])
