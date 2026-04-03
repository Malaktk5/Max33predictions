# MAX 33 · F1 2026 Oracle

**A machine-learning CLI that predicts Max Verstappen's 2026 Formula 1 season — race by race.**

> VER5TAPPEN TRUTHER · in my books he is either #1 or #33 · i do not acknowledge max3

---

## What it does

- Pulls race results (2022–2026) for Max and top competitors from the Jolpica/Ergast API
- Trains **two Random Forest models** on historical race data (position predictor + win probability classifier)
- Predicts finishing position + win/podium probability for every remaining 2026 race
- Runs **10,000 Monte Carlo simulations** to project Max's final championship points range
- Auto-refreshes the model when new race results appear
- Correctly handles **DNFs** — a "Retired" status never counts as a classified finish

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/yourusername/max33-oracle
cd max33-oracle

# 2. Install deps
pip install -r requirements.txt

# 3. Run
python f1_dashboard.py

# Force a full data refresh (clears cache)
python f1_dashboard.py --force-refresh
```

First load takes **5–15 minutes** on a cold cache (fetching years of race data from the API + rate-limit backoffs). After that, cached runs complete in seconds.

---

## File structure

```
max33-oracle/
│
├── f1_dashboard.py          # CLI entrypoint — argparse, output formatting, state
├── f1_api.py                # All API fetching, caching, and data pipeline logic
├── f1_features.py           # Feature engineering — merges DataFrames, computes metrics
├── f1_models.py             # Random Forest training, predictions, Monte Carlo sim
├── f1_weather_competitors.py# Historical race date lookup table for weather fetching
│
├── requirements.txt         # Python dependencies
└── f1_cache/                # Auto-created — cached API responses + trained model pickle
    ├── *.json               # Cached API responses (race results, lap times, quali, etc.)
    └── model_*.pkl          # Pickled trained RandomForest models
```

---

## How the predictions work

### 1. Data collection (`f1_api.py`)
Fetches the following from the **[Jolpica/Ergast F1 API](https://api.jolpi.ca)** (free, no API key):
- **Race results** (2022–2026): finishing position, grid, status, points, lap count
- **Qualifying times**: Q1/Q2/Q3 times, gap to pole position
- **Lap-by-lap times**: average/best lap, pace drop-off (tyre degradation proxy), position recovery
- **Rival lap comparisons**: Max's lap times vs top 5 competitors per race
- **Pit stop data**: number of stops, average stop duration
- **Sprint results**: sprint grid/finish for sprint weekend races
- **Driver standings**: championship points history (2023–2026)
- **Constructor standings**: Red Bull's car performance ranking per season
- **Weather data**: historical race-day conditions from **[Open-Meteo Archive API](https://archive-api.open-meteo.com)** (free, no API key)

All responses are cached locally to `f1_cache/` with smart TTLs:
- Race results/standings → **15 minutes** (refreshes frequently during live season)
- Historical/weather data → **30 days** (never changes)
- Corrupt or empty cache files are automatically deleted and re-fetched

### 2. Feature engineering (`f1_features.py`)
Transforms raw API data into model-ready features:

| Feature | Description |
|---|---|
| `grid` | Starting grid position |
| `quali_pos` | Qualifying position |
| `roll_win_5` | Rolling win rate (last 5 races, exponentially weighted) |
| `roll_pts_5` | Rolling points average |
| `roll_podium_5` | Rolling podium rate |
| `roll_pos_5` | Rolling average finish position |
| `roll_gap_5` | Rolling gap to pole average |
| `circ_win_rate` | Max's historic win rate at this circuit |
| `circ_avg_gap_pole` | Max's average quali gap at this circuit |
| `circ_mech_dnf_rate` | Mechanical DNF probability at this circuit |
| `circ_crash_rate` | Crash/collision DNF probability |
| `circ_avg_stops` | Average pit stop count at this circuit |
| `car_perf` | Red Bull's constructor standing (normalised) |
| `car_trend` | Compound lap pace trend vs 2024/2025 |
| `champ_gap` | Points gap to championship leader |
| `is_street` | Whether it's a street circuit |
| `is_high_speed` | High-speed track flag |
| `is_high_df` | High downforce track flag |
| `overtake_hard` | Low overtaking opportunity flag |
| `wx_temp_c` | Race-day temperature (°C) |
| `wx_rain_mm` | Race-day rainfall (mm) |
| `wx_wind_kmh` | Race-day wind speed (km/h) |

**Recency weighting**: 2026 races → 4.0×, 2025 → 2.5×, 2024 → 1.5×, 2022–2023 → 1.0×

### 3. Model training (`f1_models.py`)
Two separate **scikit-learn Random Forest** models:
- **Position predictor** (regressor): predicts finishing position (1–20)
- **Win predictor** (classifier): predicts probability of winning

Both use `StandardScaler` in a `Pipeline`. Models are pickled to `f1_cache/` and only retrained when new 2026 race results appear (detected via a data hash).

### 4. Prediction output
For each remaining 2026 race:
- Predicted finishing position
- Win probability %
- Podium probability %
- Historical win rate at that circuit

DNFs are displayed as `DNF` (not a classified position) in completed races.

### 5. Championship simulation
Monte Carlo approach (10,000 iterations):
- For each remaining race, randomly sample a finish outcome using the model's predicted probabilities
- Sum the championship points across all remaining races
- Report: mean projection, 10th percentile (worst-case), 50th percentile (median), 90th percentile (best-case)

---

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
scikit-learn>=1.3.0
```

Python 3.9+ recommended.

---

## Notes

- **Rate limits**: The Jolpica API rate-limits aggressive parallel requests. The app automatically backs off with exponential retry (1s → 2s → 4s). First cold-cache run will be slow — this is normal.
- **Cache TTL**: Race results refresh every 15 minutes. Force a full re-fetch anytime with `--force-refresh`.
- **2026 calendar**: Hardcoded in `f1_api.py` (`CALENDAR_2026`). Update each season as the FIA publishes the official schedule.

---

*Built for personal use. Not affiliated with Red Bull Racing, Formula 1, or anyone who actually knows what they're doing.*
