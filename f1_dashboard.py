"""
VER33 F1 Prediction Dashboard
Single-file Flask app. Run: python f1_dashboard.py
Visit: http://localhost:5000

Features:
- Fetches 2022–2025 historical data + 2026 completed races
- Parallel requests (fast), local cache (no re-downloading)
- Predictions for every remaining 2026 race
- Championship Monte Carlo simulation
- Refresh button updates everything
"""

from flask import Flask, render_template_string, jsonify
import pandas as pd
import numpy as np
import requests
import json
import time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CACHE_DIR = Path("./f1_cache")
CACHE_DIR.mkdir(exist_ok=True)
BASE = "https://api.jolpi.ca/ergast/f1"

# ── 2026 full calendar ────────────────────────────────────────────────────────
CALENDAR_2026 = [
    {"round": 1,  "name": "Australian GP",     "circuit": "albert_park",   "date": "2026-03-15"},
    {"round": 2,  "name": "Chinese GP",        "circuit": "shanghai",      "date": "2026-03-22"},
    {"round": 3,  "name": "Japanese GP",       "circuit": "suzuka",        "date": "2026-04-05"},
    {"round": 4,  "name": "Bahrain GP",        "circuit": "bahrain",       "date": "2026-04-19"},
    {"round": 5,  "name": "Saudi Arabian GP",  "circuit": "jeddah",        "date": "2026-04-26"},
    {"round": 6,  "name": "Miami GP",          "circuit": "miami",         "date": "2026-05-10"},
    {"round": 7,  "name": "Emilia Romagna GP", "circuit": "imola",         "date": "2026-05-24"},
    {"round": 8,  "name": "Monaco GP",         "circuit": "monaco",        "date": "2026-05-31"},
    {"round": 9,  "name": "Canadian GP",       "circuit": "villeneuve",    "date": "2026-06-14"},
    {"round": 10, "name": "Spanish GP",        "circuit": "catalunya",     "date": "2026-06-28"},
    {"round": 11, "name": "Austrian GP",       "circuit": "red_bull_ring", "date": "2026-07-05"},
    {"round": 12, "name": "British GP",        "circuit": "silverstone",   "date": "2026-07-12"},
    {"round": 13, "name": "Hungarian GP",      "circuit": "hungaroring",   "date": "2026-07-26"},
    {"round": 14, "name": "Belgian GP",        "circuit": "spa",           "date": "2026-08-02"},
    {"round": 15, "name": "Dutch GP",          "circuit": "zandvoort",     "date": "2026-08-30"},
    {"round": 16, "name": "Italian GP",        "circuit": "monza",         "date": "2026-09-06"},
    {"round": 17, "name": "Azerbaijan GP",     "circuit": "baku",          "date": "2026-09-20"},
    {"round": 18, "name": "Singapore GP",      "circuit": "marina_bay",    "date": "2026-10-04"},
    {"round": 19, "name": "US GP",             "circuit": "americas",      "date": "2026-10-18"},
    {"round": 20, "name": "Mexico GP",         "circuit": "rodriguez",     "date": "2026-10-25"},
    {"round": 21, "name": "Brazilian GP",      "circuit": "interlagos",    "date": "2026-11-08"},
    {"round": 22, "name": "Las Vegas GP",      "circuit": "vegas",         "date": "2026-11-21"},
    {"round": 23, "name": "Qatar GP",          "circuit": "losail",        "date": "2026-11-29"},
    {"round": 24, "name": "Abu Dhabi GP",      "circuit": "yas_marina",    "date": "2026-12-06"},
]

CIRCUIT_COORDS = {
    "bahrain": (26.03, 50.51), "jeddah": (21.63, 39.10), "albert_park": (-37.85, 144.97),
    "imola": (44.34, 11.72), "miami": (25.96, -80.24), "catalunya": (41.57, 2.26),
    "monaco": (43.73, 7.42), "baku": (40.37, 49.85), "villeneuve": (45.50, -73.52),
    "silverstone": (52.08, -1.02), "hungaroring": (47.58, 19.25), "spa": (50.44, 5.97),
    "zandvoort": (52.39, 4.54), "monza": (45.62, 9.28), "marina_bay": (1.29, 103.86),
    "suzuka": (34.84, 136.54), "losail": (25.49, 51.45), "americas": (30.13, -97.64),
    "rodriguez": (19.40, -99.09), "interlagos": (-23.70, -46.70), "vegas": (36.11, -115.17),
    "yas_marina": (24.47, 54.60), "red_bull_ring": (47.22, 14.76), "shanghai": (31.34, 121.22),
}

# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING — parallel + cached
# ═══════════════════════════════════════════════════════════════════════════════

def cache_path(key):
    return CACHE_DIR / f"{key}.json"

def load_cache(key):
    p = cache_path(key)
    if p.exists() and (time.time() - p.stat().st_mtime) < 3600 * 6:  # 6h TTL
        with open(p) as f:
            return json.load(f)
    return None

def save_cache(key, data):
    with open(cache_path(key), "w") as f:
        json.dump(data, f)

def fetch(endpoint, cache_key=None):
    if cache_key:
        cached = load_cache(cache_key)
        if cached is not None:
            return cached
    url = f"{BASE}/{endpoint}.json?limit=1000"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        if cache_key:
            save_cache(cache_key, data)
        return data
    except Exception as e:
        print(f"  ⚠ fetch failed {endpoint}: {e}")
        return None

def fetch_driver_season(args):
    driver_id, year = args
    key = f"{driver_id}_{year}_results"
    data = fetch(f"{year}/drivers/{driver_id}/results", cache_key=key)
    if not data:
        return []
    races = data["MRData"]["RaceTable"]["Races"]
    rows = []
    for race in races:
        res = race["Results"][0]
        rows.append({
            "driver_id":  driver_id,
            "year":       year,
            "round":      int(race["round"]),
            "circuit_id": race["Circuit"]["circuitId"],
            "race_name":  race["raceName"],
            "grid":       int(res["grid"]),
            "position":   int(res["position"]) if str(res["position"]).isdigit() else None,
            "points":     float(res["points"]),
            "status":     res["status"],
        })
    return rows

def fetch_quali_season(args):
    year, = args if isinstance(args, tuple) else (args,)
    key = f"ver_quali_{year}"
    data = fetch(f"{year}/drivers/max_verstappen/qualifying", cache_key=key)
    if not data:
        return []
    rows = []
    for race in data["MRData"]["RaceTable"]["Races"]:
        q = race["QualifyingResults"][0]
        rows.append({
            "year":       year,
            "round":      int(race["round"]),
            "circuit_id": race["Circuit"]["circuitId"],
            "quali_pos":  int(q["position"]),
        })
    return rows

def fetch_weather(circuit_id, date):
    key = f"wx_{circuit_id}_{date}"
    cached = load_cache(key)
    if cached:
        return cached
    if circuit_id not in CIRCUIT_COORDS:
        return {}
    lat, lon = CIRCUIT_COORDS[circuit_id]
    url = (f"https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={lat}&longitude={lon}&start_date={date}&end_date={date}"
           f"&daily=temperature_2m_max,precipitation_sum,windspeed_10m_max&timezone=auto")
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        d = r.json().get("daily", {})
        result = {
            "temp_max":      d.get("temperature_2m_max", [None])[0],
            "precipitation": d.get("precipitation_sum", [0])[0] or 0,
            "wind_speed":    d.get("windspeed_10m_max", [None])[0],
        }
        save_cache(key, result)
        return result
    except:
        return {}

def collect_all_data():
    """Fetch race + quali data in parallel. Weather loaded separately from cache only."""
    print("Fetching race data...")

    DRIVERS = ["max_verstappen", "lewis_hamilton", "charles_leclerc",
               "sergio_perez", "lando_norris", "george_russell", "carlos_sainz"]
    YEARS = [2022, 2023, 2024, 2025, 2026]

    # Parallel race results
    tasks = [(d, y) for d in DRIVERS for y in YEARS]
    all_rows = []
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(fetch_driver_season, t): t for t in tasks}
        for f in as_completed(futures):
            all_rows.extend(f.result())

    results_df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    print(f"  {len(results_df)} race result rows")

    # Parallel quali (only VER, 5 years)
    quali_rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(fetch_quali_season, (y,)) for y in YEARS]
        for f in as_completed(futures):
            quali_rows.extend(f.result())
    quali_df = pd.DataFrame(quali_rows) if quali_rows else pd.DataFrame()
    print(f"  {len(quali_df)} quali rows")

    # Weather — load from cache only (populated by background thread)
    wx_df = _load_weather_from_cache()

    return results_df, quali_df, pd.DataFrame(), wx_df


def _load_weather_from_cache() -> pd.DataFrame:
    """Load whatever weather is already cached — never blocks on network."""
    try:
        from f1_weather_dates import RACE_DATES_ALL
    except ImportError:
        return pd.DataFrame()
    rows = []
    for key, date in RACE_DATES_ALL.items():
        parts = key.split("_", 1)
        if len(parts) != 2:
            continue
        year, circuit_id = parts
        cached = load_cache(f"wx_{circuit_id}_{date}")
        if cached:
            rows.append({"year": int(year), "circuit_id": circuit_id, **cached})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def prefetch_weather_background():
    """Fire-and-forget: fetch missing weather into cache without blocking dashboard."""
    import threading
    def _run():
        try:
            from f1_weather_dates import RACE_DATES_ALL
        except ImportError:
            return
        missing = [(c, d) for k, d in RACE_DATES_ALL.items()
                   for c in [k.split("_", 1)[1]] if len(k.split("_", 1)) == 2
                   and not load_cache(f"wx_{c}_{d}")]
        if not missing:
            return
        print(f"  [bg] Fetching {len(missing)} missing weather records in background...")
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(lambda t: fetch_weather(*t), missing))
        print("  [bg] Weather cache populated.")
    threading.Thread(target=_run, daemon=True).start()


def build_features(results_df, quali_df, wx_df):
    ver = results_df[results_df["driver_id"] == "max_verstappen"].copy()
    if ver.empty:
        return pd.DataFrame()

    if not quali_df.empty:
        ver = ver.merge(quali_df[["year","round","circuit_id","quali_pos"]],
                        on=["year","round","circuit_id"], how="left")
    else:
        ver["quali_pos"] = ver["grid"]

    ver = ver.sort_values(["year","round"]).reset_index(drop=True)
    ver["win"]    = (ver["position"] == 1).astype(float)
    ver["podium"] = (ver["position"] <= 3).astype(float)
    ver["dnf"]    = (ver["status"] != "Finished").astype(float)
    ver["pole"]   = (ver["grid"] == 1).astype(float)

    # ── Recency weight: exponential decay — 2025 counts ~4x more than 2022 ──
    max_year = ver["year"].max()
    ver["recency_w"] = ver["year"].apply(lambda y: 0.55 ** (max_year - y))

    # ── Rolling form: last 5 races globally (across seasons) ──
    ver["roll_win_5"]    = ver["win"].shift(1).rolling(5, min_periods=1).mean()
    ver["roll_pts_5"]    = ver["points"].shift(1).rolling(5, min_periods=1).mean()
    ver["roll_podium_5"] = ver["podium"].shift(1).rolling(5, min_periods=1).mean()
    ver["roll_pos_5"]    = ver["position"].shift(1).rolling(5, min_periods=1).mean()

    # ── Circuit stats — weighted by recency ──
    circ_rows = []
    for cid, g in ver.groupby("circuit_id"):
        w = g["recency_w"]
        total_w = w.sum()
        circ_rows.append({
            "circuit_id":       cid,
            "circ_win_rate":    (g["win"]    * w).sum() / total_w,
            "circ_podium_rate": (g["podium"] * w).sum() / total_w,
            "circ_avg_finish":  (g["position"].fillna(20) * w).sum() / total_w,
            "circ_dnf_rate":    (g["dnf"]    * w).sum() / total_w,
            "circ_races":       len(g),
        })
    circ_df = pd.DataFrame(circ_rows)
    ver = ver.merge(circ_df, on="circuit_id", how="left")

    # Weather merge
    if not wx_df.empty:
        ver = ver.merge(wx_df, on=["year","circuit_id"], how="left")

    return ver


# ═══════════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════════

FEATS_BASE = [
    "grid", "quali_pos", "pole",
    "circ_win_rate", "circ_podium_rate", "circ_avg_finish", "circ_dnf_rate",
    "roll_win_5", "roll_pts_5", "roll_podium_5", "roll_pos_5",
]
FEATS_WX = ["temp_max", "precipitation", "wind_speed"]

def train(df):
    feats = FEATS_BASE + [f for f in FEATS_WX if f in df.columns]
    clean = df.dropna(subset=["position","win"])
    X   = clean[feats].fillna(clean[feats].median()).fillna(0)
    w   = clean["recency_w"].fillna(1.0)   # sample weights
    y_pos = clean["position"]
    y_win = clean["win"].astype(int)
    med   = clean[feats].median().fillna(0)

    rf_pos = RandomForestRegressor(n_estimators=400, max_depth=7,
                                    min_samples_leaf=2, random_state=42)
    rf_pos.fit(X, y_pos, sample_weight=w)

    rf_win = Pipeline([("sc", StandardScaler()),
                       ("rf", RandomForestClassifier(n_estimators=300, max_depth=6,
                                                      min_samples_leaf=2, random_state=42))])
    rf_win.fit(X, y_win, rf__sample_weight=w)

    return rf_pos, rf_win, feats, med


def _circuit_profile(df, circuit_id):
    """Weighted circuit stats for a specific circuit."""
    hist = df[df["circuit_id"] == circuit_id].copy()
    if hist.empty:
        # Fall back to overall recent form
        recent = df[df["year"] >= df["year"].max() - 1]
        return {
            "circ_win_rate":    float(recent["win"].mean()),
            "circ_podium_rate": float(recent["podium"].mean()),
            "circ_avg_finish":  float(recent["position"].fillna(20).mean()),
            "circ_dnf_rate":    float(recent["dnf"].mean()),
            "circ_races":       0,
        }
    w = hist["recency_w"]
    total_w = w.sum()
    return {
        "circ_win_rate":    float((hist["win"]    * w).sum() / total_w),
        "circ_podium_rate": float((hist["podium"] * w).sum() / total_w),
        "circ_avg_finish":  float((hist["position"].fillna(20) * w).sum() / total_w),
        "circ_dnf_rate":    float((hist["dnf"]    * w).sum() / total_w),
        "circ_races":       len(hist),
    }


def predict_remaining(df, rf_pos, rf_win, feats, med):
    completed_rounds = set(df[df["year"] == 2026]["round"].tolist()) if not df.empty else set()

    # Recent form: last 5 completed races globally
    recent5 = df.dropna(subset=["position"]).tail(5)
    roll_win_5    = float(recent5["win"].mean())
    roll_pts_5    = float(recent5["points"].mean())
    roll_podium_5 = float(recent5["podium"].mean())
    roll_pos_5    = float(recent5["position"].mean())

    predictions = []
    for race in CALENDAR_2026:
        rnd = race["round"]
        cid = race["circuit"]

        # Completed — use actual result
        actual = None
        if rnd in completed_rounds:
            row = df[(df["year"] == 2026) & (df["round"] == rnd)]
            if not row.empty:
                pos = row.iloc[0]["position"]
                actual = int(pos) if pd.notna(pos) else None

        cp = _circuit_profile(df, cid)

        # Quali: use actual if completed, else assume pole (optimistic baseline)
        quali_pos, grid = 1, 1
        if rnd in completed_rounds:
            q_row = df[(df["year"] == 2026) & (df["round"] == rnd)]
            if not q_row.empty:
                qp = q_row.iloc[0].get("quali_pos", 1)
                gp = q_row.iloc[0].get("grid", 1)
                quali_pos = int(qp) if pd.notna(qp) else 1
                grid      = int(gp) if pd.notna(gp) else 1

        inp = {
            "grid":            grid,
            "quali_pos":       quali_pos,
            "pole":            int(grid == 1),
            "circ_win_rate":   cp["circ_win_rate"],
            "circ_podium_rate":cp["circ_podium_rate"],
            "circ_avg_finish": cp["circ_avg_finish"],
            "circ_dnf_rate":   cp["circ_dnf_rate"],
            "roll_win_5":      roll_win_5,
            "roll_pts_5":      roll_pts_5,
            "roll_podium_5":   roll_podium_5,
            "roll_pos_5":      roll_pos_5,
        }
        if "temp_max" in feats:
            wx_hist = df[(df["circuit_id"] == cid) & df["temp_max"].notna()] if "temp_max" in df.columns else pd.DataFrame()
            inp["temp_max"]      = float(wx_hist["temp_max"].mean())      if not wx_hist.empty else 25.0
            inp["precipitation"] = float(wx_hist["precipitation"].mean()) if not wx_hist.empty else 0.5
            inp["wind_speed"]    = float(wx_hist["wind_speed"].mean())    if not wx_hist.empty else 20.0

        X_in     = pd.DataFrame([inp])[feats].fillna(med).fillna(0)
        pred_pos = float(rf_pos.predict(X_in)[0])
        win_prob = float(rf_win.predict_proba(X_in)[0][1])

        # Podium prob: model win + weighted circuit podium history
        pod_prob = min(0.98, win_prob + cp["circ_podium_rate"] * 0.6)

        # Update rolling form after each completed race
        if actual is not None:
            pts = {1:25,2:18,3:15,4:12,5:10,6:8,7:6,8:4,9:2,10:1}.get(actual, 0)
            roll_win_5    = roll_win_5    * 0.8 + (1 if actual == 1 else 0) * 0.2
            roll_pts_5    = roll_pts_5    * 0.8 + pts * 0.2
            roll_podium_5 = roll_podium_5 * 0.8 + (1 if actual <= 3 else 0) * 0.2
            roll_pos_5    = roll_pos_5    * 0.8 + actual * 0.2

        predictions.append({
            "round":          rnd,
            "name":           race["name"],
            "circuit":        cid,
            "date":           race["date"],
            "completed":      rnd in completed_rounds,
            "actual":         actual,
            "pred_pos":       round(max(1, pred_pos), 1),
            "win_prob":       round(win_prob * 100, 1),
            "podium_prob":    round(pod_prob * 100, 1),
            "win_rate_hist":  round(cp["circ_win_rate"] * 100, 1),
        })

    return predictions


def monte_carlo_championship(predictions, n=10000):
    remaining = [p for p in predictions if not p["completed"]]
    pts_map   = {1:25,2:18,3:15,4:12,5:10,6:8,7:6,8:4,9:2,10:1}
    completed_pts = sum(
        pts_map.get(p["actual"], 0)
        for p in predictions if p["completed"] and p["actual"]
    )

    rng = np.random.default_rng(42)
    totals = []

    for _ in range(n):
        season_pts = completed_pts
        for race in remaining:
            wp  = min(0.95, race["win_prob"]    / 100)
            pp  = min(0.95, race["podium_prob"] / 100) - wp
            pp  = max(0, pp)
            dnf = race.get("win_rate_hist", 10) / 100 * 0.12   # DNF scales with circuit difficulty
            dnf = min(0.12, max(0.04, dnf))
            tp  = max(0, 0.85 - wp - pp - dnf)
            out = max(0, 1 - wp - pp - tp - dnf)
            probs = np.array([wp, pp, tp, out, dnf])
            probs /= probs.sum()
            # Points outcomes: win, podium(avg), top10(avg), outside points, DNF
            outcome = rng.choice([25, 15, 7, 0, 0], p=probs)
            season_pts += outcome
            if outcome >= 10 and rng.random() < 0.22:
                season_pts += 1  # fastest lap
        totals.append(season_pts)

    totals = np.array(totals)
    return {
        "mean":            round(float(totals.mean()), 0),
        "p10":             round(float(np.percentile(totals, 10)), 0),
        "p50":             round(float(np.percentile(totals, 50)), 0),
        "p90":             round(float(np.percentile(totals, 90)), 0),
        "completed_pts":   completed_pts,
        "remaining_races": len(remaining),
        "hist":            np.histogram(totals, bins=30)[0].tolist(),
        "hist_edges":      [round(x) for x in np.histogram(totals, bins=30)[1].tolist()],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE — rebuilt on refresh
# ═══════════════════════════════════════════════════════════════════════════════

import pickle

MODEL_PATH     = CACHE_DIR / "model.pkl"
DATA_HASH_PATH = CACHE_DIR / "data_hash.txt"

def _data_hash(df):
    """Hash based on number of rows + last race round — changes when new results arrive."""
    return f"{len(df)}_{df['year'].max()}_{df['round'].max() if 'round' in df.columns else 0}"

def save_model(rf_pos, rf_win, feats, med, df):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"rf_pos": rf_pos, "rf_win": rf_win, "feats": feats, "med": med}, f)
    with open(DATA_HASH_PATH, "w") as f:
        f.write(_data_hash(df))
    print("  ✓ Model saved to cache")

def load_model(df):
    """Load cached model if data hasn't changed. Returns None if stale or missing."""
    if not MODEL_PATH.exists() or not DATA_HASH_PATH.exists():
        return None
    with open(DATA_HASH_PATH) as f:
        saved_hash = f.read().strip()
    if saved_hash != _data_hash(df):
        print("  New race data detected — retraining model...")
        return None
    print("  ✓ Using cached model (no new race data)")
    with open(MODEL_PATH, "rb") as f:
        m = pickle.load(f)
    return m["rf_pos"], m["rf_win"], m["feats"], m["med"]


STATE = {"predictions": [], "championship": {}, "last_updated": None, "loading": False, "error": None}

def refresh_data():
    STATE["loading"] = True
    STATE["error"] = None
    try:
        print("\n── Refreshing data ──")
        results_df, quali_df, ver_df, wx_df = collect_all_data()
        df = build_features(results_df, quali_df, wx_df)
        if df.empty:
            STATE["error"] = "No data returned from API — check your internet connection."
            print("  No data available.")
            return

        # Use cached model if data hasn't changed, retrain only when new races appear
        cached = load_model(df)
        if cached:
            rf_pos, rf_win, feats, med = cached
        else:
            rf_pos, rf_win, feats, med = train(df)
            save_model(rf_pos, rf_win, feats, med, df)

        STATE["predictions"]  = predict_remaining(df, rf_pos, rf_win, feats, med)
        STATE["championship"] = monte_carlo_championship(STATE["predictions"])
        STATE["last_updated"] = time.strftime("%Y-%m-%d %H:%M")
        print(f"  ✓ {len(STATE['predictions'])} race predictions ready")
        prefetch_weather_background()
    except Exception as e:
        import traceback
        STATE["error"] = str(e)
        print(f"  ✗ Refresh error: {e}\n{traceback.format_exc()}")
    finally:
        STATE["loading"] = False


# ═══════════════════════════════════════════════════════════════════════════════
# HTML
# ═══════════════════════════════════════════════════════════════════════════════

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MAX 33 · 2026 ORACLE</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;600&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,700;1,400&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root {
  --orange: #FF5F00;
  --orange-dim: rgba(255,95,0,.15);
  --bg: #070b14;
  --bg2: #0c1220;
  --bg3: #141c2e;
  --border: #1c2840;
  --text: #dde3f0;
  --muted: #445570;
  --green: #00ff87;
  --mono: 'IBM Plex Mono', monospace;
  --head: 'Bebas Neue', sans-serif;
  --body: 'DM Sans', sans-serif;
}
* { margin:0; padding:0; box-sizing:border-box; }
html { scroll-behavior: smooth; }
body { background:var(--bg); color:var(--text); font-family:var(--body); min-height:100vh; }

/* top accent stripe */
body::before {
  content:''; position:fixed; top:0; left:0; right:0; height:3px;
  background: linear-gradient(90deg, var(--orange) 0%, #ff8c00 50%, var(--orange) 100%);
  z-index:200;
}

/* ── HEADER ─────────────────────────────────────────── */
header {
  display:flex; align-items:stretch; justify-content:space-between;
  background:var(--bg2); border-bottom:1px solid var(--border);
  position:sticky; top:3px; z-index:100;
  overflow:hidden;
}
.logo-block { display:flex; align-items:stretch; }
.logo-number {
  font-family:var(--head); font-size:54px; line-height:1;
  background:var(--orange); color:#fff;
  padding:6px 22px 2px; display:flex; align-items:center;
  letter-spacing:1px; user-select:none;
  text-shadow: 2px 2px 0 rgba(0,0,0,.3);
}
.logo-text {
  padding:10px 18px; display:flex; flex-direction:column; justify-content:center; gap:3px;
}
.logo-title {
  font-family:var(--head); font-size:22px; letter-spacing:5px; line-height:1;
}
.logo-tagline {
  font-family:var(--mono); font-size:8px; color:var(--orange);
  letter-spacing:2.5px; text-transform:uppercase;
}
.header-right {
  display:flex; align-items:center; gap:14px; padding:0 22px;
}
.last-updated {
  font-family:var(--mono); font-size:9px; color:var(--muted); letter-spacing:1px;
}
.live-dot {
  width:7px; height:7px; border-radius:50%;
  background:var(--green); box-shadow:0 0 7px var(--green);
  animation:blink 2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }
.refresh-btn {
  padding:9px 20px;
  background:transparent; border:1.5px solid var(--orange);
  color:var(--orange); font-family:var(--head); font-size:16px;
  letter-spacing:3px; text-transform:uppercase; cursor:pointer;
  transition:all .2s; white-space:nowrap;
}
.refresh-btn:hover {
  background:var(--orange); color:#fff;
  box-shadow: 0 0 24px rgba(255,95,0,.3);
}
.refresh-btn:disabled { opacity:.35; cursor:wait; }

/* ── MAIN ────────────────────────────────────────────── */
main {
  max-width:1320px; margin:0 auto; padding:24px 20px;
  display:flex; flex-direction:column; gap:18px;
}

/* ── NEXT RACE SPOTLIGHT ─────────────────────────────── */
.spotlight {
  background:var(--bg2); border:1px solid var(--border);
  border-left:4px solid var(--orange); border-radius:6px;
  padding:18px 24px;
  display:grid;
  grid-template-columns: 1fr auto auto auto;
  gap:0 40px;
  align-items:center;
}
.sp-label { font-family:var(--mono); font-size:8px; color:var(--orange); letter-spacing:3px; text-transform:uppercase; margin-bottom:4px; }
.sp-name { font-family:var(--head); font-size:34px; letter-spacing:2px; line-height:1; }
.sp-meta { font-family:var(--mono); font-size:10px; color:var(--muted); margin-top:3px; }
.sp-divider { width:1px; height:48px; background:var(--border); }
.sp-stat { text-align:center; }
.sp-stat-val { font-family:var(--head); font-size:40px; color:var(--orange); line-height:1; }
.sp-stat-lbl { font-family:var(--mono); font-size:8px; color:var(--muted); letter-spacing:2px; margin-top:2px; }

/* ── CHAMPIONSHIP BANNER ─────────────────────────────── */
.champ-grid {
  display:grid; grid-template-columns:repeat(5,1fr);
  background:var(--bg2); border:1px solid var(--border); border-radius:6px;
  overflow:hidden; position:relative;
}
/* watermark 33 */
.champ-grid::after {
  content:'33'; font-family:var(--head); font-size:220px;
  color:rgba(255,95,0,.03); position:absolute; right:-8px; top:-30px;
  line-height:1; pointer-events:none; user-select:none;
}
.cm {
  padding:18px 20px; border-right:1px solid var(--border);
  position:relative; z-index:1;
}
.cm:last-child { border-right:none; }
.cm-label { font-family:var(--mono); font-size:8px; color:var(--muted); text-transform:uppercase; letter-spacing:2px; margin-bottom:6px; }
.cm-value { font-family:var(--head); font-size:48px; line-height:1; }
.cm-value.hi { color:var(--orange); }
.cm-sub { font-family:var(--mono); font-size:9px; color:var(--muted); margin-top:4px; }

/* ── CHART CARD ──────────────────────────────────────── */
.chart-card {
  background:var(--bg2); border:1px solid var(--border); border-radius:6px;
  padding:18px 22px;
}
.card-label {
  font-family:var(--mono); font-size:8px; color:var(--muted);
  text-transform:uppercase; letter-spacing:2.5px; margin-bottom:14px;
  display:flex; align-items:center; gap:10px;
}
.card-label::before { content:''; width:14px; height:2px; background:var(--orange); flex-shrink:0; }

/* ── RACE TABLE ──────────────────────────────────────── */
.races-card {
  background:var(--bg2); border:1px solid var(--border); border-radius:6px; overflow:hidden;
}
.races-header {
  padding:13px 22px; border-bottom:1px solid var(--border);
  display:flex; align-items:center; justify-content:space-between;
}
table { width:100%; border-collapse:collapse; }
thead tr { background:var(--bg3); }
th {
  font-family:var(--mono); font-size:8px; color:var(--muted);
  text-transform:uppercase; letter-spacing:1.5px;
  padding:10px 14px; text-align:left;
  border-bottom:1px solid var(--border); white-space:nowrap;
}
td {
  padding:11px 14px;
  border-bottom:1px solid rgba(28,40,64,.5);
  font-size:13px; vertical-align:middle;
}
tr:last-child td { border-bottom:none; }
tr.done { opacity:.42; }
tr.nxt td { background:rgba(255,95,0,.05); }
tr.nxt td:first-child { border-left:3px solid var(--orange); }
tr:not(.done):not(.nxt):hover td { background:rgba(255,95,0,.025); transition:background .15s; }

.rnd { font-family:var(--mono); font-size:10px; color:var(--muted); }
.race-nm { font-weight:500; }
.dt { font-family:var(--mono); font-size:10px; color:var(--muted); white-space:nowrap; }
.ppos { font-family:var(--head); font-size:24px; line-height:1; }
.apos { font-family:var(--head); font-size:22px; line-height:1; color:var(--green); }

.barrow { display:flex; align-items:center; gap:8px; }
.barbg { flex:1; height:4px; background:var(--bg3); border-radius:2px; overflow:hidden; min-width:50px; }
.barfill { height:100%; border-radius:2px; }
.barval { font-family:var(--mono); font-size:10px; min-width:34px; text-align:right; }

.badge {
  font-family:var(--mono); font-size:8px; padding:3px 8px;
  border-radius:2px; text-transform:uppercase; letter-spacing:1px; white-space:nowrap;
}
.b-done { background:rgba(0,255,135,.07); color:var(--green); border:1px solid rgba(0,255,135,.18); }
.b-next { background:rgba(255,95,0,.1); color:var(--orange); border:1px solid rgba(255,95,0,.28); }
.b-up   { color:var(--muted); border:1px solid var(--border); }

/* ── FOOTER ──────────────────────────────────────────── */
footer {
  text-align:center; padding:20px 16px;
  font-family:var(--mono); font-size:9px; color:var(--muted); letter-spacing:2px;
  border-top:1px solid var(--border); margin-top:8px;
}

/* ── LOADING OVERLAY ─────────────────────────────────── */
.overlay {
  position:fixed; inset:0; background:rgba(7,11,20,.92);
  display:none; flex-direction:column; align-items:center; justify-content:center; gap:20px;
  z-index:999;
}
.spinner {
  width:44px; height:44px;
  border:2px solid var(--border); border-top-color:var(--orange);
  border-radius:50%; animation:spin .7s linear infinite;
}
@keyframes spin { to { transform:rotate(360deg); } }
.load-msg { font-family:var(--mono); font-size:10px; color:var(--muted); letter-spacing:2px; text-transform:uppercase; }

@keyframes fadein { from{opacity:0;transform:translateY(7px)} to{opacity:1;transform:translateY(0)} }
.anim { animation:fadein .3s ease forwards; }

/* ── RESPONSIVE ──────────────────────────────────────── */
@media(max-width:960px) {
  .champ-grid { grid-template-columns:repeat(3,1fr); }
  .spotlight  { grid-template-columns:1fr 1fr; gap:12px; }
  .sp-divider { display:none; }
}
@media(max-width:640px) {
  .logo-number { font-size:40px; padding:4px 14px 2px; }
  .logo-title  { font-size:17px; }
  .champ-grid  { grid-template-columns:1fr 1fr; }
  th,td        { padding:8px 10px; }
  .sp-name     { font-size:24px; }
}
</style>
</head>
<body>

<div class="overlay" id="overlay">
  <div class="spinner"></div>
  <div class="load-msg" id="load-msg">summoning max from the simulator...</div>
</div>

<header>
  <div class="logo-block">
    <div class="logo-number">33</div>
    <div class="logo-text">
      <div class="logo-title">MAX VERSTAPPEN · 2026 ORACLE</div>
      <div class="logo-tagline">VER5TAPPEN TRUTHER &nbsp;·&nbsp; #3 was taken &nbsp;·&nbsp; we don't acknowledge #3</div>
    </div>
  </div>
  <div class="header-right">
    <div class="last-updated" id="last-updated">—</div>
    <div class="live-dot"></div>
    <button class="refresh-btn" id="ref-btn" onclick="doRefresh()">↺ Refresh</button>
  </div>
</header>

<main id="main">
  <div style="text-align:center;padding:80px 20px;font-family:var(--mono);color:var(--muted);font-size:10px;letter-spacing:3px;text-transform:uppercase">
    hit refresh · the oracle awaits
  </div>
</main>

<footer>
  MAX 33 ORACLE &nbsp;·&nbsp; BUILT FOR THE VERSTAPPEN-PILLED &nbsp;·&nbsp; POWERED BY THE COLLECTIVE DESPAIR OF EVERYONE ELSE ON THE GRID
</footer>

<script>
const LOAD_MSGS = [
  "summoning max from the simulator...",
  "calculating the inevitable...",
  "counting laps hamilton couldn't keep up...",
  "fetching 10,000 parallel timelines...",
  "asking the model nicely to say p1...",
  "threatening the api with drs...",
  "weather check: will it rain? (doesn't matter, max wins anyway)",
];

let mcChart = null, msgTick = null;

function posColor(pos) {
  if (pos <= 1) return 'var(--green)';
  if (pos <= 3) return 'var(--orange)';
  if (pos <= 6) return 'var(--text)';
  return 'var(--muted)';
}

async function doRefresh() {
  const btn = document.getElementById('ref-btn');
  const overlay = document.getElementById('overlay');
  const msgEl = document.getElementById('load-msg');
  btn.disabled = true; btn.textContent = '... loading';
  overlay.style.display = 'flex';

  let mi = 0;
  msgEl.textContent = LOAD_MSGS[0];
  msgTick = setInterval(() => { mi = (mi+1) % LOAD_MSGS.length; msgEl.textContent = LOAD_MSGS[mi]; }, 2400);

  try {
    const res  = await fetch('/api/refresh', {method:'POST'});
    const data = await res.json();
    clearInterval(msgTick);
    data.error ? showError(data.error) : render(data);
  } catch(e) {
    clearInterval(msgTick);
    alert('network error: ' + e);
  }

  btn.disabled = false; btn.textContent = '↺ Refresh';
  overlay.style.display = 'none';
}

function showError(err) {
  document.getElementById('main').innerHTML = `
    <div style="background:var(--bg2);border:1px solid rgba(255,95,0,.4);border-radius:6px;padding:28px 30px">
      <div style="color:var(--orange);font-family:var(--mono);font-size:8px;letter-spacing:3px;text-transform:uppercase;margin-bottom:10px">⚠ something exploded (not max's fault)</div>
      <pre style="font-family:var(--mono);font-size:11px;white-space:pre-wrap;color:var(--text)">${err}</pre>
      <div style="color:var(--muted);font-family:var(--mono);font-size:9px;margin-top:12px">check the terminal · blame the api · maybe blame perez</div>
    </div>`;
}

function render(d) {
  document.getElementById('last-updated').textContent = 'synced ' + d.last_updated;
  const c     = d.championship;
  const preds = d.predictions;
  const nxt   = preds.find(p => !p.completed);
  const done  = preds.filter(p => p.completed).length;

  // ── Spotlight
  const spotlight = nxt ? `
    <div class="spotlight anim">
      <div>
        <div class="sp-label">Next Victim</div>
        <div class="sp-name">${nxt.name}</div>
        <div class="sp-meta">Round ${nxt.round} &nbsp;·&nbsp; ${nxt.date} &nbsp;·&nbsp; historical win rate: ${nxt.win_rate_hist}%</div>
      </div>
      <div class="sp-divider"></div>
      <div class="sp-stat">
        <div class="sp-stat-val">${nxt.win_prob}%</div>
        <div class="sp-stat-lbl">win prob</div>
      </div>
      <div class="sp-stat">
        <div class="sp-stat-val">${nxt.podium_prob}%</div>
        <div class="sp-stat-lbl">podium prob</div>
      </div>
      <div class="sp-stat">
        <div class="sp-stat-val">P${nxt.pred_pos}</div>
        <div class="sp-stat-lbl">predicted</div>
      </div>
    </div>` : `
    <div class="spotlight anim" style="border-left-color:var(--green)">
      <div><div class="sp-label" style="color:var(--green)">Season Status</div>
      <div class="sp-name">SEASON COMPLETE</div>
      <div class="sp-meta">all 24 rounds done · how many did he win?</div></div>
    </div>`;

  // ── Table rows
  const rows = preds.map(p => {
    const isNxt = p === nxt;
    const posCell = (p.completed && p.actual)
      ? `<span class="apos">P${p.actual}</span>`
      : `<span class="ppos" style="color:${posColor(p.pred_pos)}">P${p.pred_pos}</span>`;
    const stat = p.completed
      ? `<span class="badge b-done">done ✓</span>`
      : isNxt
        ? `<span class="badge b-next">▶ next</span>`
        : `<span class="badge b-up">upcoming</span>`;
    return `<tr class="${p.completed?'done':''} ${isNxt?'nxt':''}">
      <td><span class="rnd">R${p.round}</span></td>
      <td class="race-nm">${p.name}</td>
      <td class="dt">${p.date}</td>
      <td>${posCell}</td>
      <td>
        <div class="barrow">
          <div class="barbg"><div class="barfill" style="width:${Math.min(100,p.win_prob*2)}%;background:var(--orange)"></div></div>
          <span class="barval" style="color:var(--orange)">${p.win_prob}%</span>
        </div>
      </td>
      <td>
        <div class="barrow">
          <div class="barbg"><div class="barfill" style="width:${p.podium_prob}%;background:#ff9030"></div></div>
          <span class="barval" style="color:#ff9030">${p.podium_prob}%</span>
        </div>
      </td>
      <td class="dt">${p.win_rate_hist}%</td>
      <td>${stat}</td>
    </tr>`;
  }).join('');

  document.getElementById('main').innerHTML = `
    ${spotlight}

    <div class="champ-grid anim">
      <div class="cm">
        <div class="cm-label">projected points</div>
        <div class="cm-value hi">${c.mean}</div>
        <div class="cm-sub">full season avg</div>
      </div>
      <div class="cm">
        <div class="cm-label">points banked</div>
        <div class="cm-value">${c.completed_pts}</div>
        <div class="cm-sub">${done} race${done!==1?'s':''} done</div>
      </div>
      <div class="cm">
        <div class="cm-label">best case (P90)</div>
        <div class="cm-value">${c.p90}</div>
        <div class="cm-sub">top 10% of runs</div>
      </div>
      <div class="cm">
        <div class="cm-label">worst case (P10)</div>
        <div class="cm-value">${c.p10}</div>
        <div class="cm-sub">bottom 10% of runs</div>
      </div>
      <div class="cm">
        <div class="cm-label">races left</div>
        <div class="cm-value">${c.remaining_races}</div>
        <div class="cm-sub">of 24 total</div>
      </div>
    </div>

    <div class="chart-card anim">
      <div class="card-label">10,000 season simulations &nbsp;·&nbsp; championship points distribution</div>
      <canvas id="mcChart" height="72"></canvas>
    </div>

    <div class="races-card anim">
      <div class="races-header">
        <div class="card-label" style="margin:0">2026 Race Predictions &nbsp;·&nbsp; all 24 rounds</div>
        <span style="font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:2px">${done}/24 COMPLETED</span>
      </div>
      <table>
        <thead><tr>
          <th>#</th><th>Race</th><th>Date</th><th>Pos</th>
          <th>Win %</th><th>Podium %</th><th>Hist Win %</th><th>Status</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;

  // Chart
  if (mcChart) { mcChart.destroy(); mcChart = null; }
  const ctx = document.getElementById('mcChart').getContext('2d');
  mcChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: c.hist_edges.slice(0,-1).map(v => Math.round(v)),
      datasets: [{
        data: c.hist,
        backgroundColor: 'rgba(255,95,0,.6)',
        borderColor: 'rgba(255,95,0,.85)',
        borderWidth: 1, borderRadius: 2,
      }]
    },
    options: {
      plugins: {
        legend: {display:false},
        tooltip: {
          callbacks: { label: ctx => `${ctx.raw} simulations` },
          bodyFont: {family:'IBM Plex Mono'}, titleFont: {family:'IBM Plex Mono'}
        }
      },
      scales: {
        x: { ticks:{color:'#445570',font:{family:'IBM Plex Mono',size:9}}, grid:{color:'#1c2840'} },
        y: { ticks:{color:'#445570',font:{family:'IBM Plex Mono',size:9}}, grid:{color:'#1c2840'} }
      }
    }
  });
}

window.onload = doRefresh;
</script>
</body>
</html>
"""

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return HTML

@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    refresh_data()
    return jsonify({
        "predictions":  STATE["predictions"],
        "championship": STATE["championship"],
        "last_updated": STATE["last_updated"],
        "error":        STATE["error"],
    })

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  VER33 F1 Dashboard · http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=False, port=5000)