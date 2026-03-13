"""
MAX 33 · F1 2026 Oracle
Run: python f1_dashboard.py
Visit: http://localhost:5000

Project structure:
  f1_dashboard.py            ← this file (Flask app + ML logic)
  f1_weather_competitors.py  ← race date lookup for weather
  templates/index.html       ← HTML template
  static/style.css           ← styles
  static/app.js              ← frontend JS
  f1_cache/                  ← auto-created cache directory
"""

from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import requests
import json
import time
import pickle
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

    tasks = [(d, y) for d in DRIVERS for y in YEARS]
    all_rows = []
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(fetch_driver_season, t): t for t in tasks}
        for f in as_completed(futures):
            all_rows.extend(f.result())

    results_df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    print(f"  {len(results_df)} race result rows")

    quali_rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(fetch_quali_season, (y,)) for y in YEARS]
        for f in as_completed(futures):
            quali_rows.extend(f.result())
    quali_df = pd.DataFrame(quali_rows) if quali_rows else pd.DataFrame()
    print(f"  {len(quali_df)} quali rows")

    wx_df = _load_weather_from_cache()
    return results_df, quali_df, pd.DataFrame(), wx_df


def _load_weather_from_cache() -> pd.DataFrame:
    """Load whatever weather is already cached — never blocks on network."""
    try:
        from f1_weather_competitors import RACE_DATES_ALL
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
            from f1_weather_competitors import RACE_DATES_ALL
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


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

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

    max_year = ver["year"].max()
    ver["recency_w"] = ver["year"].apply(lambda y: 0.55 ** (max_year - y))

    ver["roll_win_5"]    = ver["win"].shift(1).rolling(5, min_periods=1).mean()
    ver["roll_pts_5"]    = ver["points"].shift(1).rolling(5, min_periods=1).mean()
    ver["roll_podium_5"] = ver["podium"].shift(1).rolling(5, min_periods=1).mean()
    ver["roll_pos_5"]    = ver["position"].shift(1).rolling(5, min_periods=1).mean()

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
    X     = clean[feats].fillna(clean[feats].median()).fillna(0)
    w     = clean["recency_w"].fillna(1.0)
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
    hist = df[df["circuit_id"] == circuit_id].copy()
    if hist.empty:
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

    recent5       = df.dropna(subset=["position"]).tail(5)
    roll_win_5    = float(recent5["win"].mean())
    roll_pts_5    = float(recent5["points"].mean())
    roll_podium_5 = float(recent5["podium"].mean())
    roll_pos_5    = float(recent5["position"].mean())

    predictions = []
    for race in CALENDAR_2026:
        rnd = race["round"]
        cid = race["circuit"]

        actual = None
        if rnd in completed_rounds:
            row = df[(df["year"] == 2026) & (df["round"] == rnd)]
            if not row.empty:
                pos = row.iloc[0]["position"]
                actual = int(pos) if pd.notna(pos) else None

        cp = _circuit_profile(df, cid)

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
        pod_prob = min(0.98, win_prob + cp["circ_podium_rate"] * 0.6)

        if actual is not None:
            pts = {1:25,2:18,3:15,4:12,5:10,6:8,7:6,8:4,9:2,10:1}.get(actual, 0)
            roll_win_5    = roll_win_5    * 0.8 + (1 if actual == 1 else 0) * 0.2
            roll_pts_5    = roll_pts_5    * 0.8 + pts * 0.2
            roll_podium_5 = roll_podium_5 * 0.8 + (1 if actual <= 3 else 0) * 0.2
            roll_pos_5    = roll_pos_5    * 0.8 + actual * 0.2

        predictions.append({
            "round":         rnd,
            "name":          race["name"],
            "circuit":       cid,
            "date":          race["date"],
            "completed":     rnd in completed_rounds,
            "actual":        actual,
            "pred_pos":      round(max(1, pred_pos), 1),
            "win_prob":      round(win_prob * 100, 1),
            "podium_prob":   round(pod_prob * 100, 1),
            "win_rate_hist": round(cp["circ_win_rate"] * 100, 1),
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
            dnf = race.get("win_rate_hist", 10) / 100 * 0.12
            dnf = min(0.12, max(0.04, dnf))
            tp  = max(0, 0.85 - wp - pp - dnf)
            out = max(0, 1 - wp - pp - tp - dnf)
            probs = np.array([wp, pp, tp, out, dnf])
            probs /= probs.sum()
            outcome = rng.choice([25, 15, 7, 0, 0], p=probs)
            season_pts += outcome
            if outcome >= 10 and rng.random() < 0.22:
                season_pts += 1
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
# MODEL CACHE
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PATH     = CACHE_DIR / "model.pkl"
DATA_HASH_PATH = CACHE_DIR / "data_hash.txt"

def _data_hash(df):
    return f"{len(df)}_{df['year'].max()}_{df['round'].max() if 'round' in df.columns else 0}"

def save_model(rf_pos, rf_win, feats, med, df):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"rf_pos": rf_pos, "rf_win": rf_win, "feats": feats, "med": med}, f)
    with open(DATA_HASH_PATH, "w") as f:
        f.write(_data_hash(df))
    print("  ✓ Model saved to cache")

def load_model(df):
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


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE — rebuilt on refresh
# ═══════════════════════════════════════════════════════════════════════════════

STATE = {"predictions": [], "championship": {}, "last_updated": None, "loading": False, "error": None}

def refresh_data():
    STATE["loading"] = True
    STATE["error"]   = None
    try:
        print("\n── Refreshing data ──")
        results_df, quali_df, _, wx_df = collect_all_data()
        df = build_features(results_df, quali_df, wx_df)
        if df.empty:
            STATE["error"] = "No data returned from API — check your internet connection."
            return

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
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")

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
    print("  MAX 33 Oracle · http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=False, port=5000)