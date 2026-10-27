"""Capture desktop and mobile screenshots of every app page from a RUNNING app.

    streamlit run app.py --server.port 8501 &      # start the app first
    pip install playwright && python -m playwright install chromium
    python scripts/capture_screenshots.py --url http://localhost:8501

Screenshots are written to reports/figures/screenshots/.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PAGES = {"overview": "", "data_explorer": "data_explorer", "race_explorer": "race_explorer",
         "model_telemetry": "model_telemetry", "model_comparison": "model_comparison",
         "podium_predictor": "podium_predictor"}
VIEWPORTS = {"desktop": dict(viewport={"width": 1366, "height": 1500}),
             "mobile": dict(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True,
                            has_touch=True)}


def wait_for_app(page, timeout: float = 60.0) -> None:
    page.wait_for_selector("[data-testid='stMain']", timeout=timeout * 1000)
    deadline = time.time() + timeout
    while time.time() < deadline:   # wait until Streamlit stops running the script
        if page.locator("[data-testid='stStatusWidget']").count() == 0:
            break
        time.sleep(0.5)
    time.sleep(3)                   # let Plotly finish drawing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--out", default="reports/figures/screenshots")
    ap.add_argument("--season", default="2024", help="season to select on the predictor page")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for vname, vopts in VIEWPORTS.items():
            ctx = browser.new_context(**vopts)
            page = ctx.new_page()
            for pname, path in PAGES.items():
                page.goto(f"{args.url.rstrip('/')}/{path}")
                wait_for_app(page)
                page.screenshot(path=str(out / f"{vname}_{pname}.png"), full_page=False)
                print("saved", out / f"{vname}_{pname}.png")
            ctx.close()
        browser.close()


if __name__ == "__main__":
    main()
