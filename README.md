# MAX 33 · F1 2026 Oracle

**A machine-learning dashboard that predicts Max Verstappen's 2026 Formula 1 season — race by race.**

> VER5TAPPEN TRUTHER · in my books he is either #1 or #33 · i do not acknowledge max3

---

## What it does

- Pulls race results (2022–2026) for Max and 6 top competitors from the open Ergast/Jolpica API
- Trains a Random Forest model on grid position, qualifying, circuit history, rolling form, and weather
- Predicts finishing position + win probability for every remaining 2026 race
- Runs **10,000 Monte Carlo simulations** to project Max's final championship points range
- Auto-refreshes the model when new race results appear and the API gets updated

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

First load takes ~30–60 seconds (downloading seasons of race data). After that, results are cached and it's basically instant.

---

## File structure

```
max33-oracle/
│
├── f1_dashboard.py            # Flask app + all ML logic (data, features, model, routes)
├── f1_weather_competitors.py  # Race date lookup table for weather fetching
│
├── templates/
│   └── index.html             # HTML page structure (links to CSS + JS)
│
├── static/
│   ├── style.css              # All styles (colors, layout, components)
│   └── app.js                 # All frontend JS (refresh, render, chart)
│
└── f1_cache/                  # Auto-created — API responses + trained model
    ├── data_hash.txt
    └── wx_*.json / *_results.json
```

---

## How the predictions work (basically)

1. **Historical data** — race results, qualifying positions, and weather for every Max race since 2022
2. **Features** — grid position, circuit win rate, rolling 5-race form, weather conditions
3. **Model** — two Random Forests: one predicts finishing position (regression), one predicts win probability (classification)
4. **Recency weighting** — 2025 races count ~4× more than 2022 in training, because unfortunately for us 2025 wasn't as great as 2022/2023
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

Python 3.9+ recommended

---

## Notes

- Data source: [Jolpica/Ergast F1 API](https://api.jolpi.ca) — free, no key needed
- Weather: [Open-Meteo Archive API](https://archive-api.open-meteo.com) — also free, no key needed
- Cache TTL: 6 hours. Force re-fetch by deleting `f1_cache/`
- The model retrains automatically when new 2026 race results appear

---

*Built for personal use. Not affiliated with Red Bull Racing, Formula 1, or anyone who actually knows what they're doing.*
