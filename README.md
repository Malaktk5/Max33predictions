# MAX 33 · F1 2026 Oracle

**A machine-learning dashboard that predicts Max Verstappen's 2026 Formula 1 season — race by race.**

> VER5TAPPEN TRUTHER · in my books he is either number #1 or number #33 i do not aknowledge max3

---

## What it does

- Pulls race results (2022–2026) for Max and 6 top competitors from the open Ergast/Jolpica API 

- Trains a Random Forest model on grid position, qualifying, circuit history, rolling form, and weather

- Predicts finishing position + win probability for every remaining 2026 race

- Runs **10,000 Monte Carlo simulations** to project Max's final championship points range

- Auto-refreshes the model when new race results appear and the api gets updated 

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/yourusername/max33-oracle
cd max33-oracle

# 2. Install deps
pip install flask pandas numpy scikit-learn requests

# 3. Run
python f1_dashboard.py
```

Then open **http://localhost:5000** and hit **Refresh**.

The first load takes ~30–60 seconds (downloading seasons of race data). After that, results are cached and it's basically instant

im planning on deploying this app soon too

---

## Files

| File | What it is |
|---|---|
| `f1_dashboard.py` | Main Flask app — data fetching, model training, HTML dashboard |
| `f1_weather_competitors.py` | Lookup table: `year_circuit_id → race date` (used for weather fetching) |
| `f1_cache/` | Auto-created. Stores API responses + trained model. Delete to force a full re-fetch. |

---

## How the predictions work basically

1. **Historical data** — race results, qualifying positions, and weather for every Max race since 2022
2. **Features** — grid position, circuit win rate, rolling 5-race form, weather conditions
3. **Model** — two Random Forests: one predicts finishing position (regression), one predicts win probability (classification)
4. **Recency weighting** — 2025 races count ~4× more than 2022 in training because unfortunatly for us 2025 wasnt as great as 2022/2023
5. **Championship sim** — Monte Carlo: for each remaining race, sample an outcome (win/podium/points/DNF) using the model's probabilities, sum the points, repeat 10,000 times


---

## Requirements

```
flask
pandas
numpy
scikit-learn
requests
```

Python 3.9+ recommended.

---

## Notes

- Data source: [Jolpica/Ergast F1 API](https://api.jolpi.ca) — free, no key needed
- Weather: [Open-Meteo Archive API](https://archive-api.open-meteo.com) — also free, no key needed
- Cache TTL: 6 hours. Force refresh by deleting `f1_cache/`
- The model retrains automatically when new 2026 race results appear

---

*Built for personal use. Not affiliated with Red Bull Racing, Formula 1, or anyone who actually knows what they're doing.*