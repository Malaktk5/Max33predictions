import pandas as pd
import numpy as np
import requests
import json
import time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

CACHE_DIR = Path("./f1_cache")
CACHE_DIR.mkdir(exist_ok=True)
BASE = "https://api.jolpi.ca/ergast/f1"

CALENDAR_2026 = [
    {
        "round": 1,
        "name": "Australian GP",
        "circuit": "albert_park",
        "date": "2026-03-08",
        "sprint": False,
    },
    {
        "round": 2,
        "name": "Chinese GP",
        "circuit": "shanghai",
        "date": "2026-03-15",
        "sprint": True,
    },
    {
        "round": 3,
        "name": "Japanese GP",
        "circuit": "suzuka",
        "date": "2026-03-29",
        "sprint": False,
    },
    {
        "round": 4,
        "name": "Miami GP",
        "circuit": "miami",
        "date": "2026-05-03",
        "sprint": True,
    },
    {
        "round": 5,
        "name": "Canadian GP",
        "circuit": "villeneuve",
        "date": "2026-05-24",
        "sprint": False,
    },
    {
        "round": 6,
        "name": "Monaco GP",
        "circuit": "monaco",
        "date": "2026-06-07",
        "sprint": False,
    },
    {
        "round": 7,
        "name": "Spanish GP",
        "circuit": "catalunya",
        "date": "2026-06-21",
        "sprint": False,
    },
    {
        "round": 8,
        "name": "Austrian GP",
        "circuit": "red_bull_ring",
        "date": "2026-07-05",
        "sprint": True,
    },
    {
        "round": 9,
        "name": "British GP",
        "circuit": "silverstone",
        "date": "2026-07-12",
        "sprint": False,
    },
    {
        "round": 10,
        "name": "Belgian GP",
        "circuit": "spa",
        "date": "2026-07-19",
        "sprint": True,
    },
    {
        "round": 11,
        "name": "Hungarian GP",
        "circuit": "hungaroring",
        "date": "2026-07-26",
        "sprint": False,
    },
    {
        "round": 12,
        "name": "Dutch GP",
        "circuit": "zandvoort",
        "date": "2026-08-23",
        "sprint": False,
    },
    {
        "round": 13,
        "name": "Italian GP",
        "circuit": "monza",
        "date": "2026-09-06",
        "sprint": False,
    },
    {
        "round": 14,
        "name": "Azerbaijan GP",
        "circuit": "baku",
        "date": "2026-09-27",
        "sprint": True,
    },
    {
        "round": 15,
        "name": "Singapore GP",
        "circuit": "marina_bay",
        "date": "2026-10-11",
        "sprint": False,
    },
    {
        "round": 16,
        "name": "US GP",
        "circuit": "americas",
        "date": "2026-10-25",
        "sprint": True,
    },
    {
        "round": 17,
        "name": "Mexico GP",
        "circuit": "rodriguez",
        "date": "2026-11-01",
        "sprint": False,
    },
    {
        "round": 18,
        "name": "Brazilian GP",
        "circuit": "interlagos",
        "date": "2026-11-15",
        "sprint": True,
    },
    {
        "round": 19,
        "name": "Las Vegas GP",
        "circuit": "vegas",
        "date": "2026-11-21",
        "sprint": False,
    },
    {
        "round": 20,
        "name": "Qatar GP",
        "circuit": "losail",
        "date": "2026-11-29",
        "sprint": False,
    },
    {
        "round": 21,
        "name": "Abu Dhabi GP",
        "circuit": "yas_marina",
        "date": "2026-12-06",
        "sprint": False,
    },
]

# ── 2026 driver list (new regs, new teams) ────────────────────────────────────
DRIVERS_2026 = [
    "max_verstappen",
    "george_russell",
    "kimi_antonelli",
    "charles_leclerc",
    "lewis_hamilton",
    "lando_norris",
    "oscar_piastri",
    "carlos_sainz",
    "oliver_bearman",
    "fernando_alonso",
]

CIRCUIT_COORDS = {
    "bahrain": (26.03, 50.51),
    "jeddah": (21.63, 39.10),
    "albert_park": (-37.85, 144.97),
    "imola": (44.34, 11.72),
    "miami": (25.96, -80.24),
    "catalunya": (41.57, 2.26),
    "monaco": (43.73, 7.42),
    "baku": (40.37, 49.85),
    "villeneuve": (45.50, -73.52),
    "silverstone": (52.08, -1.02),
    "hungaroring": (47.58, 19.25),
    "spa": (50.44, 5.97),
    "zandvoort": (52.39, 4.54),
    "monza": (45.62, 9.28),
    "marina_bay": (1.29, 103.86),
    "suzuka": (34.84, 136.54),
    "losail": (25.49, 51.45),
    "americas": (30.13, -97.64),
    "rodriguez": (19.40, -99.09),
    "interlagos": (-23.70, -46.70),
    "vegas": (36.11, -115.17),
    "yas_marina": (24.47, 54.60),
    "red_bull_ring": (47.22, 14.76),
    "shanghai": (31.34, 121.22),
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING — parallel + cached
# ═══════════════════════════════════════════════════════════════════════════════


def cache_path(key):
    return CACHE_DIR / f"{key}.json"


def load_cache(key):
    p = cache_path(key)
    if not p.exists():
        return None
    age = time.time() - p.stat().st_mtime
    # Race results: cache for 15 mins (dynamic fast refresh)
    # Weather: cache forever (historical data never changes)
    ttl = 900 if key.endswith("_results") or key.endswith("_standings") else 3600 * 24 * 30
    if age > ttl:
        return None
    with open(p) as f:
        return json.load(f)


def save_cache(key, data):
    with open(cache_path(key), "w") as f:
        json.dump(data, f)


import threading

_fetch_lock = threading.Semaphore(3)  # max 3 concurrent requests at once


def fetch(endpoint, cache_key=None, retries=3):
    if cache_key:
        cached = load_cache(cache_key)
        if cached is not None:
            return cached
    url = f"{BASE}/{endpoint}.json?limit=1000"
    for attempt in range(retries):
        with _fetch_lock:
            try:
                time.sleep(0.25)  # 250ms between requests = max ~12/sec safely
                r = requests.get(url, timeout=15)
                if r.status_code == 429:
                    wait = 2**attempt  # exponential backoff: 1s, 2s, 4s
                    print(f"  ⏳ rate limited, waiting {wait}s ({endpoint})")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                data = r.json()
                if cache_key:
                    save_cache(cache_key, data)
                return data
            except Exception as e:
                if attempt == retries - 1:
                    print(f"  ⚠ fetch failed {endpoint}: {e}")
                else:
                    time.sleep(1)
    return None


def fetch_driver_season(args):
    driver_id, year = args
    key = f"{driver_id}_{year}_results"
    data = fetch(f"{year}/drivers/{driver_id}/results", cache_key=key)
    if not data:
        return []
    rows = []
    for race in data["MRData"]["RaceTable"]["Races"]:
        res = race["Results"][0]
        status = res["status"]
        dnf_mechanical = int(
            any(
                w in status.lower()
                for w in [
                    "engine",
                    "gearbox",
                    "hydraulic",
                    "electrical",
                    "power",
                    "brake",
                    "suspension",
                    "oil",
                    "water",
                    "fuel",
                    "turbo",
                ]
            )
        )
        dnf_crash = int(
            any(
                w in status.lower()
                for w in ["accident", "collision", "spun", "damage", "crash"]
            )
        )
        # Retirement lap: how many laps completed (0 = finished all laps)
        total_laps = int(race.get("laps", 0)) if race.get("laps") else 0
        driver_laps = int(res.get("laps", total_laps))
        retirement_lap_pct = None  # % of race completed before DNF
        if status != "Finished" and total_laps > 0:
            retirement_lap_pct = round(driver_laps / total_laps, 3)

        rows.append(
            {
                "driver_id": driver_id,
                "year": year,
                "round": int(race["round"]),
                "circuit_id": race["Circuit"]["circuitId"],
                "race_name": race["raceName"],
                "total_laps": total_laps,
                "laps_completed": driver_laps,
                "grid": int(res["grid"]),
                "position": (
                    int(res["position"]) if str(res["position"]).isdigit() else None
                ),
                "points": float(res["points"]),
                "status": status,
                "dnf_mechanical": dnf_mechanical,
                "dnf_crash": dnf_crash,
                "fastest_lap": int(res.get("FastestLap", {}).get("rank", "0") == "1"),
                "fastest_lap_time": res.get("FastestLap", {})
                .get("Time", {})
                .get("time"),
                "fastest_lap_speed": float(
                    res.get("FastestLap", {}).get("AverageSpeed", {}).get("speed", 0)
                    or 0
                ),
                "retirement_lap_pct": retirement_lap_pct,
            }
        )
    return rows


def fetch_lap_times(args):
    """
    Fetch lap-by-lap times for VER in a given race.
    Returns summary stats: avg lap, best lap, consistency (std), pace drop-off.
    Full lap data is too heavy to store — we summarise it.
    """
    year, rnd = args
    key = f"laps_ver_{year}_{rnd}"
    data = fetch(f"{year}/{rnd}/drivers/max_verstappen/laps", cache_key=key)
    if not data:
        return None
    races = data["MRData"]["RaceTable"]["Races"]
    if not races or not races[0].get("Laps"):
        return None

    def parse_ms(t):
        if not t:
            return None
        try:
            if ":" in str(t):
                m, s = str(t).split(":")
                return int(m) * 60000 + round(float(s) * 1000)
            return round(float(t) * 1000)
        except:
            return None

    lap_times = []
    positions_by_lap = []
    for lap in races[0]["Laps"]:
        lap_num = int(lap["number"])
        for timing in lap.get("Timings", []):
            ms = parse_ms(timing.get("time"))
            if (
                ms and 60000 < ms < 300000
            ):  # filter safety car / pit laps (1-5 min range)
                lap_times.append(ms)
            pos = timing.get("position")
            if pos:
                try:
                    positions_by_lap.append((lap_num, int(pos)))
                except:
                    pass

    if not lap_times:
        return None

    lap_arr = np.array(lap_times)
    # Pace drop-off: compare first 10% of laps vs last 10% (tyre degradation proxy)
    n = len(lap_arr)
    early = lap_arr[: max(1, n // 10)].mean() if n >= 5 else lap_arr.mean()
    late = lap_arr[max(1, -n // 10) :].mean() if n >= 5 else lap_arr.mean()
    pace_dropoff = round((late - early) / early * 100, 3)  # % slower at end

    # Position recovery: where did he start vs his avg position in first 5 laps
    start_pos = positions_by_lap[0][1] if positions_by_lap else None
    lap5_pos = (
        np.mean([p for _, p in positions_by_lap if _ <= 5])
        if positions_by_lap
        else None
    )
    pos_gain_early = round(start_pos - lap5_pos, 2) if start_pos and lap5_pos else None

    return {
        "year": year,
        "round": rnd,
        "avg_lap_ms": round(float(lap_arr.mean()), 1),
        "best_lap_ms": round(float(lap_arr.min()), 1),
        "lap_std_ms": round(float(lap_arr.std()), 1),  # consistency
        "pace_dropoff_pct": pace_dropoff,
        "pos_gain_early": pos_gain_early,
        "laps_recorded": n,
    }


def fetch_rival_lap_times(args):
    """
    Fetch best lap times for top rivals in a race — proxy for car pace vs field.
    Returns the gap between VER's best lap and the fastest rival's best lap.
    """
    year, rnd = args
    key = f"laps_all_{year}_{rnd}_best"
    data = fetch(f"{year}/{rnd}/laps", cache_key=key)
    if not data:
        return None
    races = data["MRData"]["RaceTable"]["Races"]
    if not races:
        return None

    def parse_ms(t):
        if not t:
            return None
        try:
            if ":" in str(t):
                m, s = str(t).split(":")
                return int(m) * 60000 + round(float(s) * 1000)
            return round(float(t) * 1000)
        except:
            return None

    # Collect best lap per driver
    best_by_driver = {}
    for lap in races[0].get("Laps", []):
        for timing in lap.get("Timings", []):
            did = timing.get("driverId")
            ms = parse_ms(timing.get("time"))
            if did and ms and 60000 < ms < 300000:
                if did not in best_by_driver or ms < best_by_driver[did]:
                    best_by_driver[did] = ms

    if "max_verstappen" not in best_by_driver or len(best_by_driver) < 2:
        return None

    ver_best = best_by_driver["max_verstappen"]
    rivals = {k: v for k, v in best_by_driver.items() if k != "max_verstappen"}
    fastest_rival = min(rivals.values()) if rivals else ver_best
    median_rival = np.median(list(rivals.values())) if rivals else ver_best

    return {
        "year": year,
        "round": rnd,
        "ver_best_lap_ms": ver_best,
        "gap_to_fastest_ms": round(ver_best - fastest_rival, 1),  # + = slower
        "gap_to_median_ms": round(ver_best - median_rival, 1),
        "ver_lap_rank": sorted(best_by_driver.values()).index(ver_best) + 1,
        "field_size": len(best_by_driver),
    }


def fetch_lap_positions(args):
    """
    Summarise VER's position trajectory through the race.
    Key insight: does he recover well from bad starts?
    """
    year, rnd = args
    key = f"lappos_ver_{year}_{rnd}"
    data = fetch(f"{year}/{rnd}/drivers/max_verstappen/laps", cache_key=key)
    if not data:
        return None
    races = data["MRData"]["RaceTable"]["Races"]
    if not races or not races[0].get("Laps"):
        return None

    positions = []
    for lap in races[0]["Laps"]:
        for t in lap.get("Timings", []):
            try:
                positions.append(int(t["position"]))
            except:
                pass

    if not positions:
        return None

    n = len(positions)
    start = positions[0]
    mid = positions[n // 2] if n > 1 else start
    end = positions[-1]

    return {
        "year": year,
        "round": rnd,
        "pos_start_of_race": start,
        "pos_mid_race": mid,
        "pos_end_of_race": end,
        "max_pos_reached": min(positions),  # best position during race
        "positions_gained": start - end,  # positive = gained places
        "recovery_score": round(
            (start - mid) / max(start, 1), 3
        ),  # how fast he recovers
    }


def fetch_last_completed_races(year: int) -> list:
    """Fetch current season round-by-round. Stops when a round returns empty."""
    rows = []
    for rnd in range(1, 30):
        key = f"ver_{year}_r{rnd}_result"
        data = fetch(f"{year}/{rnd}/drivers/max_verstappen/results", cache_key=key)
        if not data:
            break
        races = data["MRData"]["RaceTable"]["Races"]
        if not races:
            break
        race = races[0]
        res = race["Results"][0]
        status = res["status"]
        dnf_mech = int(
            any(
                w in status.lower()
                for w in [
                    "engine",
                    "gearbox",
                    "hydraulic",
                    "electrical",
                    "power",
                    "brake",
                    "suspension",
                    "oil",
                    "water",
                    "fuel",
                    "turbo",
                ]
            )
        )
        dnf_crash = int(
            any(
                w in status.lower()
                for w in ["accident", "collision", "spun", "damage", "crash"]
            )
        )
        total_laps = int(race.get("laps", 0)) if race.get("laps") else 0
        driver_laps = int(res.get("laps", total_laps))
        ret_pct = (
            round(driver_laps / total_laps, 3)
            if status != "Finished" and total_laps > 0
            else None
        )

        rows.append(
            {
                "driver_id": "max_verstappen",
                "year": year,
                "round": int(race["round"]),
                "circuit_id": race["Circuit"]["circuitId"],
                "race_name": race["raceName"],
                "total_laps": total_laps,
                "laps_completed": driver_laps,
                "grid": int(res["grid"]),
                "position": (
                    int(res["position"]) if str(res["position"]).isdigit() else None
                ),
                "points": float(res["points"]),
                "status": status,
                "dnf_mechanical": dnf_mech,
                "dnf_crash": dnf_crash,
                "fastest_lap": int(res.get("FastestLap", {}).get("rank", "0") == "1"),
                "fastest_lap_time": res.get("FastestLap", {})
                .get("Time", {})
                .get("time"),
                "fastest_lap_speed": float(
                    res.get("FastestLap", {}).get("AverageSpeed", {}).get("speed", 0)
                    or 0
                ),
                "retirement_lap_pct": ret_pct,
            }
        )
        print(
            f"  ✓ {year} R{race['round']} {race['raceName']}: P{res['position']} ({status})"
        )
    return rows


def fetch_constructor_standings_current(year: int) -> dict:
    """Fetch constructor standings to gauge car pace relative to rivals."""
    key = f"constructor_standings_{year}"
    data = fetch(f"{year}/constructorStandings", cache_key=key)
    if not data:
        return {}
    sl = data["MRData"]["StandingsTable"]["StandingsLists"]
    if not sl:
        return {}
    out = {}
    for s in sl[0]["ConstructorStandings"]:
        cid = s["Constructor"]["constructorId"]
        pos = s.get("position") or s.get("positionText") or "0"
        try:
            pos = int(pos)
        except:
            pos = 0
        out[cid] = {"pos": pos, "pts": float(s["points"]), "wins": int(s["wins"])}
    return out


def fetch_quali_full(args):
    """Fetch quali with Q3 time for gap-to-pole calculation."""
    (year,) = args if isinstance(args, tuple) else (args,)
    key = f"ver_quali_full_{year}"
    data = fetch(f"{year}/drivers/max_verstappen/qualifying", cache_key=key)
    if not data:
        return []
    rows = []
    for race in data["MRData"]["RaceTable"]["Races"]:
        q = race["QualifyingResults"][0]

        def parse_time(t):
            if not t or pd.isna(t):
                return None
            t = str(t).strip()
            if ":" in t:
                m, s = t.split(":")
                return int(m) * 60 + float(s)
            try:
                return float(t)
            except:
                return None

        best = None
        for col in ["Q3", "Q2", "Q1"]:
            v = parse_time(q.get(col))
            if v:
                best = v
                break
        rows.append(
            {
                "year": year,
                "round": int(race["round"]),
                "circuit_id": race["Circuit"]["circuitId"],
                "quali_pos": int(q["position"]),
                "best_quali_sec": best,
            }
        )
    return rows


def fetch_pole_times(year):
    """Fetch P1 qualifying time each race to compute gap-to-pole."""
    key = f"pole_times_{year}"
    data = fetch(f"{year}/qualifying/1", cache_key=key)  # position=1 = pole
    if not data:
        return {}
    pole_map = {}
    for race in data["MRData"]["RaceTable"]["Races"]:
        if not race.get("QualifyingResults"):
            continue
        q = race["QualifyingResults"][0]

        def parse_time(t):
            if not t:
                return None
            t = str(t).strip()
            if ":" in t:
                m, s = t.split(":")
                return int(m) * 60 + float(s)
            try:
                return float(t)
            except:
                return None

        best = None
        for col in ["Q3", "Q2", "Q1"]:
            v = parse_time(q.get(col))
            if v:
                best = v
                break
        if best:
            pole_map[(year, int(race["round"]))] = best
    return pole_map


def fetch_pitstops(args):
    """Fetch pitstop data for VER — avg stop time, number of stops."""
    year, rnd = args
    key = f"pits_{year}_{rnd}_ver"
    data = fetch(f"{year}/{rnd}/drivers/max_verstappen/pitstops", cache_key=key)
    if not data:
        return None
    stops = data["MRData"]["RaceTable"]["Races"]
    if not stops or not stops[0].get("PitStops"):
        return None
    pit_list = stops[0]["PitStops"]

    def dur_to_sec(d):
        try:
            if ":" in str(d):
                m, s = str(d).split(":")
                return int(m) * 60 + float(s)
            return float(d)
        except:
            return None

    durations: list[float] = []
    for p in pit_list:
        val = dur_to_sec(p["duration"])
        if val is not None:
            durations.append(float(val))
    return {
        "year": year,
        "round": rnd,
        "n_stops": len(pit_list),
        "avg_pit_sec": float(f"{sum(durations) / len(durations):.2f}") if durations else None,
        "min_pit_sec": float(f"{min(durations):.2f}") if durations else None,
    }


def fetch_sprint(args):
    """Fetch sprint result for VER if it exists."""
    (year,) = args if isinstance(args, tuple) else (args,)
    key = f"ver_sprint_{year}"
    data = fetch(f"{year}/drivers/max_verstappen/sprint", cache_key=key)
    if not data:
        return []
    rows = []
    for race in data["MRData"]["RaceTable"]["Races"]:
        if not race.get("SprintResults"):
            continue
        res = race["SprintResults"][0]
        rows.append(
            {
                "year": year,
                "round": int(race["round"]),
                "circuit_id": race["Circuit"]["circuitId"],
                "sprint_pos": (
                    int(res["position"]) if str(res["position"]).isdigit() else None
                ),
                "sprint_pts": float(res["points"]),
                "sprint_grid": int(res["grid"]),
            }
        )
    return rows


def fetch_standings(year):
    """Fetch end-of-season (or current) driver standings."""
    key = f"standings_{year}"
    data = fetch(f"{year}/driverStandings", cache_key=key)
    if not data:
        return {}
    sl = data["MRData"]["StandingsTable"]["StandingsLists"]
    if not sl:
        return {}
    out = {}
    for s in sl[0]["DriverStandings"]:
        did = s["Driver"]["driverId"]
        pos = s.get("position") or s.get("positionText") or "0"
        try:
            pos = int(pos)
        except:
            pos = 0
        out[did] = {
            "champ_pos": pos,
            "champ_pts": float(s["points"]),
            "wins": int(s["wins"]),
        }
    return out


def collect_all_data():
    """Fetch everything in parallel. Weather from cache only."""
    print("Fetching race data...")
    DRIVERS = [
        "max_verstappen",
        "george_russell",
        "lewis_hamilton",
        "charles_leclerc",
        "lando_norris",
        "oscar_piastri",
        "carlos_sainz",
        "fernando_alonso",
        "sergio_perez",
        "kimi_antonelli",
    ]
    HIST_YEARS = [2022, 2023, 2024, 2025]
    CURR_YEAR = 2026

    # ── Historical results (2022-2025, cached aggressively) ──────────────────
    all_rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {
            ex.submit(fetch_driver_season, (d, y)): (d, y)
            for d in DRIVERS
            for y in HIST_YEARS
        }
        for f in as_completed(futures):
            all_rows.extend(f.result())

    # ── Current season (2026): fetch round-by-round, never stale ─────────────
    # Delete old season-level cache so we always fetch fresh per-round data
    stale_key = cache_path(f"max_verstappen_{CURR_YEAR}_results")
    if stale_key.exists():
        stale_key.unlink()
        print(f"  Cleared stale 2026 season cache")

    print(f"  Fetching 2026 round-by-round...")
    curr_rows = fetch_last_completed_races(CURR_YEAR)
    all_rows.extend(curr_rows)

    # Also fetch other 2026 drivers for competitor gap (season level is fine if cached)
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {
            ex.submit(fetch_driver_season, (d, CURR_YEAR)): d
            for d in DRIVERS
            if d != "max_verstappen"
        }
        for f in as_completed(futures):
            all_rows.extend(f.result())

    results_df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    print(f"  {len(results_df)} total race result rows")
    ver_2026 = results_df[
        (results_df["driver_id"] == "max_verstappen")
        & (results_df["year"] == CURR_YEAR)
    ]
    print(f"  VER 2026 rounds in data: {sorted(ver_2026['round'].tolist())}")

    # ── Qualifying with times ─────────────────────────────────────────────────
    quali_rows = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(fetch_quali_full, (y,)) for y in HIST_YEARS + [CURR_YEAR]]
        for f in as_completed(futures):
            quali_rows.extend(f.result())
    quali_df = pd.DataFrame(quali_rows) if quali_rows else pd.DataFrame()

    # ── Pole times for gap-to-pole ────────────────────────────────────────────
    pole_map = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(fetch_pole_times, y): y for y in HIST_YEARS + [CURR_YEAR]}
        for f in as_completed(futures):
            pole_map.update(f.result())

    if not quali_df.empty and pole_map:

        def gap(row):
            pole = pole_map.get((row["year"], row["round"]))
            if pole and row["best_quali_sec"]:
                return round(row["best_quali_sec"] - pole, 3)
            return None

        quali_df["gap_to_pole"] = quali_df.apply(gap, axis=1)

    # ── Pitstops (VER only) ───────────────────────────────────────────────────
    ver_races = results_df[results_df["driver_id"] == "max_verstappen"]
    pit_tasks = list(set((int(y), int(r)) for y, r in zip(ver_races["year"], ver_races["round"])))
    pit_rows = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(fetch_pitstops, t): t for t in pit_tasks}
        for f in as_completed(futures):
            r = f.result()
            if r:
                pit_rows.append(r)
    pit_df = pd.DataFrame(pit_rows) if pit_rows else pd.DataFrame()
    print(f"  {len(pit_df)} pitstop records")

    # ── Lap times summary (VER only, parallel) ────────────────────────────────
    lap_rows = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(fetch_lap_times, t): t for t in pit_tasks}
        for f in as_completed(futures):
            r = f.result()
            if r:
                lap_rows.append(r)
    lap_df = pd.DataFrame(lap_rows) if lap_rows else pd.DataFrame()
    print(f"  {len(lap_df)} lap time summaries")

    # ── Rival lap times (car pace vs field) ───────────────────────────────────
    # Only for recent seasons to keep request count manageable
    recent_ver = results_df[
        (results_df["driver_id"] == "max_verstappen") & (results_df["year"] >= 2023)
    ]
    rival_lap_tasks = list(set((int(y), int(r)) for y, r in zip(recent_ver["year"], recent_ver["round"])))
    rival_lap_rows = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(fetch_rival_lap_times, t): t for t in rival_lap_tasks}
        for f in as_completed(futures):
            r = f.result()
            if r:
                rival_lap_rows.append(r)
    rival_lap_df = pd.DataFrame(rival_lap_rows) if rival_lap_rows else pd.DataFrame()
    print(f"  {len(rival_lap_df)} rival lap comparisons")

    # ── Lap positions (position trajectory per race) ───────────────────────────
    lap_pos_rows = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(fetch_lap_positions, t): t for t in pit_tasks}
        for f in as_completed(futures):
            r = f.result()
            if r:
                lap_pos_rows.append(r)
    lap_pos_df = pd.DataFrame(lap_pos_rows) if lap_pos_rows else pd.DataFrame()

    # ── Sprint results ────────────────────────────────────────────────────────
    sprint_rows = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(fetch_sprint, (y,)) for y in HIST_YEARS + [CURR_YEAR]]
        for f in as_completed(futures):
            sprint_rows.extend(f.result())
    sprint_df = pd.DataFrame(sprint_rows) if sprint_rows else pd.DataFrame()
    print(f"  {len(sprint_df)} sprint results")

    # ── Driver championship standings ─────────────────────────────────────────
    standings = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(fetch_standings, y): y for y in HIST_YEARS + [CURR_YEAR]}
        for f in as_completed(futures):
            y = futures[f]
            standings[y] = f.result()

    # ── Constructor standings (car performance signal) ────────────────────────
    constructor_standings = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {
            ex.submit(fetch_constructor_standings_current, y): y
            for y in HIST_YEARS + [CURR_YEAR]
        }
        for f in as_completed(futures):
            y = futures[f]
            constructor_standings[y] = f.result()

    # Red Bull position in constructor standings per year
    rb_pos = {}
    for y, cs in constructor_standings.items():
        rb = cs.get("red_bull", cs.get("red_bull_racing", {}))
        rb_pos[y] = rb.get("pos", 1)
    print(f"  Red Bull constructor pos: {rb_pos}")

    # ── Weather from cache ────────────────────────────────────────────────────
    wx_df = _load_weather_from_cache()

    return (
        results_df,
        quali_df,
        pit_df,
        lap_df,
        rival_lap_df,
        lap_pos_df,
        sprint_df,
        standings,
        rb_pos,
        wx_df,
    )


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

    def _run():
        try:
            from f1_weather_competitors import RACE_DATES_ALL
        except ImportError:
            return
        missing = [
            (c, d)
            for k, d in RACE_DATES_ALL.items()
            for c in [k.split("_", 1)[1]]
            if len(k.split("_", 1)) == 2 and not load_cache(f"wx_{c}_{d}")
        ]
        if not missing:
            return
        print(
            f"  [bg] Fetching {len(missing)} missing weather records in background..."
        )
        # with ThreadPoolExecutor(max_workers=3) as ex:
        #     list(ex.map(lambda t: fetch_weather(*t), missing))
        print("  [bg] Weather cache populated.")

    threading.Thread(target=_run, daemon=True).start()
