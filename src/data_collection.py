"""Reproducible retrieval of historical F1 data from the Jolpica-F1 API.

* Raw API responses are cached as JSON under ``data/raw/<kind>/<season>.json``
  together with retrieval metadata, and are never modified afterwards.
* A ``data/raw/manifest.json`` records source URL, retrieval time, row counts
  and coverage for every cached file.
* When the network is unavailable the cached files are used. If a required
  file is neither cached nor downloadable, a clear ``DataSourceError`` is raised
  — no data is ever fabricated.

Usage::

    python -m src.data_collection            # download anything not cached
    python -m src.data_collection --refresh  # re-download everything
    python -m src.data_collection --offline  # never touch the network
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import load_config, resolve

log = logging.getLogger(__name__)

KINDS = ("results", "qualifying", "schedule")


class DataSourceError(RuntimeError):
    """Raised when data cannot be retrieved from the API or the local cache."""


class SchemaError(ValueError):
    """Raised when an API response does not have the expected structure."""


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _endpoint(kind: str, season: int) -> str:
    if kind == "results":
        return f"{season}/results.json"
    if kind == "qualifying":
        return f"{season}/qualifying.json"
    if kind == "schedule":
        return f"{season}.json"
    raise ValueError(f"Unknown data kind: {kind}")


def fetch_json(url: str, params: dict[str, Any], cfg: dict, session: requests.Session | None = None) -> dict:
    """GET a JSON document with retries, back-off and response validation."""
    dcfg = cfg["data"]
    sess = session or requests.Session()
    last_err: Exception | None = None
    for attempt in range(int(dcfg["max_retries"]) + 1):
        try:
            resp = sess.get(url, params=params, timeout=dcfg["request_timeout_seconds"])
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", 5 * (attempt + 1)))
                log.warning("Rate limited by API; waiting %.0fs", wait)
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                raise DataSourceError(f"Server error {resp.status_code} for {resp.url}")
            if resp.status_code != 200:
                raise DataSourceError(f"HTTP {resp.status_code} for {resp.url}: {resp.text[:200]}")
            payload = resp.json()
            if "MRData" not in payload:
                raise SchemaError(f"Response from {resp.url} has no 'MRData' key")
            return payload
        except (requests.ConnectionError, requests.Timeout, DataSourceError, ValueError) as err:
            if isinstance(err, SchemaError):
                raise
            last_err = err
            time.sleep(min(2 ** attempt, 20))
    raise DataSourceError(
        f"Could not reach the data source at {url} after {dcfg['max_retries'] + 1} attempts "
        f"({last_err}). If you are offline, run with --offline to use cached data."
    )


def fetch_paged_races(kind: str, season: int, cfg: dict, session: requests.Session | None = None) -> tuple[list[dict], dict]:
    """Fetch every page for one season and merge races split across pages."""
    dcfg = cfg["data"]
    base = dcfg["base_url"].rstrip("/")
    url = f"{base}/{_endpoint(kind, season)}"
    limit = int(dcfg["page_limit"])
    offset, total = 0, None
    races: dict[tuple[str, str], dict] = {}
    list_key = {"results": "Results", "qualifying": "QualifyingResults", "schedule": None}[kind]
    n_rows = 0
    while total is None or offset < total:
        payload = fetch_json(url, {"limit": limit, "offset": offset}, cfg, session)
        mr = payload["MRData"]
        total = int(mr.get("total", 0))
        page_races = mr.get("RaceTable", {}).get("Races")
        if page_races is None:
            raise SchemaError(f"{url}: missing RaceTable.Races")
        for race in page_races:
            key = (str(race.get("season")), str(race.get("round")))
            if list_key is None:
                races[key] = race
                n_rows += 1
                continue
            rows = race.get(list_key, [])
            n_rows += len(rows)
            if key in races:
                races[key][list_key].extend(rows)
            else:
                races[key] = race
        offset += limit
        time.sleep(float(dcfg["request_pause_seconds"]))
        if total == 0:
            break
    meta = {
        "kind": kind,
        "season": season,
        "url": url,
        "api_total": total,
        "rows_received": n_rows,
        "races": len(races),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if kind != "schedule" and n_rows != total:
        raise SchemaError(f"{url}: received {n_rows} rows but API reported total={total}")
    ordered = sorted(races.values(), key=lambda r: int(r.get("round", 0)))
    return ordered, meta


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------
def raw_path(kind: str, season: int, cfg: dict) -> Path:
    return resolve(cfg["data"]["raw_dir"]) / kind / f"{season}.json"


def seasons_for(kind: str, cfg: dict) -> list[int]:
    d = cfg["data"]
    start = d["qualifying_start_season"] if kind == "qualifying" else d["history_start_season"]
    return list(range(int(start), int(d["study_end_season"]) + 1))


def download_season(kind: str, season: int, cfg: dict, refresh: bool = False,
                    offline: bool = False, session: requests.Session | None = None) -> dict:
    """Ensure one season/kind is cached; return its metadata."""
    path = raw_path(kind, season, cfg)
    if path.exists() and not refresh:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)["meta"] | {"from_cache": True}
    if offline:
        raise DataSourceError(
            f"Offline mode: no cached file for {kind} {season} at {path}. "
            "Run `python -m src.data_collection` once with network access."
        )
    races, meta = fetch_paged_races(kind, season, cfg, session)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "races": races}, fh)
    log.info("Cached %s %s: %s rows, %s races", kind, season, meta["rows_received"], meta["races"])
    return meta | {"from_cache": False}


def download_all(cfg: dict | None = None, refresh: bool = False, offline: bool = False) -> dict:
    """Download (or verify cache for) every configured season and kind."""
    cfg = cfg or load_config()
    session = requests.Session()
    session.headers["User-Agent"] = "f1-race-intelligence-student-project/1.0"
    manifest: dict[str, Any] = {
        "source": cfg["data"]["source_name"],
        "base_url": cfg["data"]["base_url"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": [],
        "errors": [],
    }
    for kind in KINDS:
        for season in seasons_for(kind, cfg):
            try:
                meta = download_season(kind, season, cfg, refresh, offline, session)
                manifest["files"].append(meta)
            except (DataSourceError, SchemaError) as err:
                manifest["errors"].append({"kind": kind, "season": season, "error": str(err)})
                log.error("%s", err)
    out = resolve(cfg["data"]["raw_dir"]) / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    if manifest["errors"]:
        required = [e for e in manifest["errors"] if e["kind"] == "results"]
        if required:
            raise DataSourceError(
                f"{len(required)} required results season(s) could not be loaded: "
                + "; ".join(f"{e['season']}: {e['error']}" for e in required)
            )
    return manifest


def load_raw(kind: str, season: int, cfg: dict) -> list[dict]:
    path = raw_path(kind, season, cfg)
    if not path.exists():
        raise DataSourceError(f"Missing cached file {path}. Run `python -m src.data_collection`.")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)["races"]


# ---------------------------------------------------------------------------
# Flattening (raw JSON -> tidy tables). Malformed records are logged, not guessed.
# ---------------------------------------------------------------------------
REQUIRED_RESULT_FIELDS = ("position", "positionText", "grid", "status")


def flatten_results(races: list[dict], problems: list[dict] | None = None) -> pd.DataFrame:
    rows = []
    for race in races:
        circuit = race.get("Circuit", {}) or {}
        for r in race.get("Results", []):
            drv, con = r.get("Driver", {}) or {}, r.get("Constructor", {}) or {}
            missing = [f for f in REQUIRED_RESULT_FIELDS if f not in r]
            if not drv.get("driverId") or not con.get("constructorId") or missing:
                if problems is not None:
                    problems.append({"season": race.get("season"), "round": race.get("round"),
                                     "issue": f"malformed result (missing {missing or 'ids'})"})
                continue
            rows.append({
                "season": race.get("season"), "round": race.get("round"),
                "race_name": race.get("raceName"), "date": race.get("date"),
                "circuit_id": circuit.get("circuitId"), "circuit_name": circuit.get("circuitName"),
                "country": (circuit.get("Location") or {}).get("country"),
                "driver_id": drv["driverId"], "driver_code": drv.get("code"),
                "driver_name": f"{drv.get('givenName', '')} {drv.get('familyName', '')}".strip(),
                "constructor_id": con["constructorId"], "constructor_name": con.get("name"),
                "car_number": r.get("number"),
                "grid": r.get("grid"), "position": r.get("position"), "position_text": r.get("positionText"),
                "points": r.get("points"), "laps": r.get("laps"), "status": r.get("status"),
            })
    return pd.DataFrame(rows)


def flatten_qualifying(races: list[dict]) -> pd.DataFrame:
    rows = []
    for race in races:
        for q in race.get("QualifyingResults", []):
            rows.append({
                "season": race.get("season"), "round": race.get("round"),
                "driver_id": (q.get("Driver") or {}).get("driverId"),
                "quali_position": q.get("position"),
                "q1": q.get("Q1"), "q2": q.get("Q2"), "q3": q.get("Q3"),
            })
    return pd.DataFrame(rows)


def flatten_schedule(races: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "season": r.get("season"), "round": r.get("round"), "race_name": r.get("raceName"),
        "date": r.get("date"), "circuit_id": (r.get("Circuit") or {}).get("circuitId"),
    } for r in races])


def build_flat_tables(cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    """Combine cached season files into flat CSV tables in data/processed/."""
    cfg = cfg or load_config()
    problems: list[dict] = []
    res = [flatten_results(load_raw("results", s, cfg), problems) for s in seasons_for("results", cfg)]
    sch = [flatten_schedule(load_raw("schedule", s, cfg)) for s in seasons_for("schedule", cfg)
           if raw_path("schedule", s, cfg).exists()]
    qua = [flatten_qualifying(load_raw("qualifying", s, cfg)) for s in seasons_for("qualifying", cfg)
           if raw_path("qualifying", s, cfg).exists()]
    tables = {
        "results_flat": pd.concat(res, ignore_index=True),
        "schedule_flat": pd.concat(sch, ignore_index=True) if sch else pd.DataFrame(),
        "qualifying_flat": pd.concat(qua, ignore_index=True) if qua else pd.DataFrame(),
        "schema_problems": pd.DataFrame(problems, columns=["season", "round", "issue"]),
    }
    out_dir = resolve(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/cache F1 data from Jolpica-F1")
    parser.add_argument("--refresh", action="store_true", help="re-download even if cached")
    parser.add_argument("--offline", action="store_true", help="use only cached files")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manifest = download_all(refresh=args.refresh, offline=args.offline)
    tables = build_flat_tables()
    print(f"Files: {len(manifest['files'])}, errors: {len(manifest['errors'])}")
    for name, df in tables.items():
        print(f"  {name}: {df.shape}")


if __name__ == "__main__":
    main()
