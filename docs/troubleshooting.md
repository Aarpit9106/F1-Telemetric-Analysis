# Troubleshooting and known limitations

| Symptom | Cause | Fix |
|---|---|---|
| `pip install` fails on pandas/numpy | Python older than 3.11 | Install Python 3.11–3.13 and recreate the venv |
| `DataSourceError: Could not reach the data source` | No internet / API down | Use the bundled cache: `python -m src.data_pipeline --offline` |
| `Offline mode: no cached file for …` | Cache missing (e.g. fresh git clone) | Run once with internet: `python -m src.data_pipeline` |
| Many "Rate limited by API; waiting" lines | Jolpica limits ~4 req/s and ~500 req/hour | Normal; the downloader backs off automatically |
| App page says "Processed data not found" | `data/processed` missing (git clone) | `python -m src.data_pipeline` |
| App says model "has not been trained" | `models/*.joblib` missing | `python -m src.train` |
| `ModuleNotFoundError: src` | Running from another folder | Run commands from the project root |
| Error unpickling a model / sklearn version warning | Different scikit-learn version | `pip install -r requirements.txt` (exact pins) or retrain |
| Phone cannot open the app | Firewall / wrong address | `--server.address 0.0.0.0`, allow Python on private networks, use the laptop's IPv4 |
| Timings differ from the report | Different hardware | Expected; timings are machine-dependent |

## Known limitations
- Target is conditional on classification; retirements cannot be predicted pre-race (sensitivity run provided).
- Model scores are not calibrated probabilities (balanced class weights); use them for ranking.
- No weather, tyre, upgrade, practice-pace or telemetry features; FastF1 is future work.
- Career and team counts start in 2000 (history window), so they are lower bounds for long careers.
- Only two test seasons (46 races): wide confidence intervals; best models are indistinguishable from a grid ranking.
- Sprint races are not included.
- Training (`python -m src.train`) is single-machine and takes several minutes; the app never retrains.
- Known failures: none in the test suite at submission time (`python -m pytest` → 38 passed).
