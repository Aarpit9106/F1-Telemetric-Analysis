# Running and deploying the app

## 1. Local (Windows / macOS / Linux)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501. The submission ZIP already contains the cached data, processed tables, trained
models and results, so no download or training is needed for the demo.

## 2. Android phone on the same Wi-Fi (recommended for the viva demo)
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
Find the laptop's IP (`ipconfig` on Windows → IPv4 Address, e.g. 192.168.1.23) and open
`http://192.168.1.23:8501` in Chrome on the phone. If it does not load, allow Python through the Windows
firewall for private networks. The layout stacks columns, enlarges touch targets and shrinks charts below 640 px.

## 3. Streamlit Community Cloud (public URL — optional)
1. Push the repository to GitHub. Because `data/` is git-ignored, either
   - commit the processed files the app needs (`data/processed/features_study.csv`, `model_dataset.csv`,
     `pipeline_summary.json`; ~3 MB) by force-adding them: `git add -f data/processed/*.csv data/processed/*.json`, or
   - run the pipeline in a build step (not supported on Community Cloud), so the first option is simpler.
2. `models/*.joblib` and `reports/results/` are committed (small) so the app needs no training on the server.
3. On https://share.streamlit.io choose *New app* → repository → branch → `app.py`. Pick the same Python version as
   local (3.13 tested; ≥ 3.11 required) under *Advanced settings*.
4. After deployment, open the URL on a desktop browser and an Android phone and check every page.

Limitations: free hosting sleeps after inactivity (first load is slow); the API is never called from the app,
so Jolpica rate limits do not affect the hosted app. **This project has not been deployed to a public URL by the
author of this repository; no live URL is claimed.**

## 4. Regenerating screenshots
```bash
pip install -r requirements-dev.txt && python -m playwright install chromium
streamlit run app.py --server.port 8501 &
python scripts/capture_screenshots.py --url http://localhost:8501
```
