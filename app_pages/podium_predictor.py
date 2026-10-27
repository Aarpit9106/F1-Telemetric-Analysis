import plotly.graph_objects as go
import streamlit as st

from app_pages.common import get_artefact, load_result_json, load_study, model_label, note, setup_page, trained_models
from src.predict import describe_score, predict_race, race_entrants

setup_page()
st.title("Podium Predictor")
note("Pick a historical Grand Prix and a model. The model scores every driver who started that race using only "
     "information available before the start. These are <b>estimates, not guaranteed outcomes</b>.")

study = load_study()
models = trained_models()
if study is None:
    st.error("Processed data not found. Run `python -m src.data_pipeline`.")
    st.stop()
if not models:
    st.warning("No trained model artefacts found in `models/`. Run `python -m src.train`.")
    st.stop()

pref = (load_result_json("preferred_model.json") or {}).get("model")
c1, c2, c3 = st.columns([1, 2, 2])
season = c1.selectbox("Season", sorted(study["season"].unique(), reverse=True))
races = study[study["season"] == season][["round", "race_name"]].drop_duplicates().sort_values("round")
if races.empty:
    st.info("No races available for this season.")
    st.stop()
label = c2.selectbox("Grand Prix", [f"R{r} · {n}" for r, n in zip(races["round"], races["race_name"])])
rnd = int(label.split(" ·")[0][1:])
name = c3.selectbox("Model", models, index=models.index(pref) if pref in models else 0, format_func=model_label)

art = get_artefact(name)
if art is None:
    st.error(f"Could not load the {model_label(name)} artefact. Retrain with `python -m src.train`.")
    st.stop()

entrants = race_entrants(study, season, rnd)
try:
    ranked = predict_race(art, entrants)
except (ValueError, KeyError) as err:
    st.error(f"Prediction unavailable: {err}")
    st.stop()

if ranked["in_training_data"].iloc[0]:
    st.warning(f"{season} was part of this model's training data ({art['trained_on_seasons'][0]}–"
               f"{art['trained_on_seasons'][-1]}), so this is an **in-sample** prediction and will look better than "
               "true forecasting. Choose 2023 or 2024 for genuine held-out predictions.")
else:
    st.info(f"{season} is a held-out test season for this model — the model never saw these results.")

driver = st.selectbox("Focus driver", ranked["driver_name"].tolist())
row = ranked[ranked["driver_name"] == driver].iloc[0]
m1, m2, m3 = st.columns(3)
m1.metric("Model verdict", "Podium" if row["predicted_podium"] == 1 else "No podium")
m2.metric("Model rank in race", f"{row['model_rank']} / {len(ranked)}")
m3.metric("Grid slot", int(row["grid"]) if row["grid"] > 0 else "Pit lane")
note(f"{driver}: {describe_score(row['podium_score'], art['score_kind'], name)} "
     f"{'Rookie — no prior starts, so form features were imputed.' if row['drv_career_starts'] == 0 else ''}")

st.subheader("Drivers ranked by podium score")
top = ranked.head(10).iloc[::-1]
fig = go.Figure(go.Bar(x=top["podium_score"], y=top["driver_name"], orientation="h",
                       marker_color=["#e10600" if p else "#8a94a6" for p in top["predicted_podium"]]))
fig.update_layout(height=360, margin=dict(l=8, r=8, t=30, b=8),
                  title="Top 10 by model score (red = predicted podium)",
                  xaxis_title="probability-like score" if art["score_kind"] == "probability" else "SVM margin")
st.plotly_chart(fig)

reveal = st.toggle("Reveal actual result (historical fact)", value=False)
cols = ["model_rank", "driver_name", "constructor_name", "grid", "podium_score", "predicted_podium"]
tbl = ranked[cols].rename(columns={"model_rank": "Rank", "driver_name": "Driver", "constructor_name": "Constructor",
                                   "grid": "Grid", "podium_score": "Score", "predicted_podium": "Predicted podium"})
if reveal:
    tbl["Actual"] = ranked.apply(lambda r: f"P{int(r['finish_position'])}" if r["finish_position"] == r["finish_position"]
                                 else r["position_text"], axis=1)
    hits = int(ranked.head(3)["podium_all_starters"].sum())
    st.markdown(f"**Model's top three contained {hits} of the 3 actual podium finishers.**")
st.dataframe(tbl.round(3), hide_index=True)
note("Scores are relative: use them to rank drivers within a race. The models were trained on classified finishers, "
     "so a driver who later retired was scored as if they would finish.")
