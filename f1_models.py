# pyre-ignore-all-errors
# type: ignore
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from f1_api import CACHE_DIR, CALENDAR_2026

# ═══════════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════════

FEATS_BASE = [
    # Qualifying
    "grid",
    "quali_pos",
    "pole",
    "gap_to_pole",
    # Circuit historical (weighted)
    "circ_win_rate",
    "circ_podium_rate",
    "circ_avg_finish",
    "circ_dnf_rate",
    "circ_mech_dnf_rate",
    "circ_crash_rate",
    "circ_avg_gap_pole",
    "circ_avg_stops",
    # Rolling form
    "roll_win_5",
    "roll_pts_5",
    "roll_podium_5",
    "roll_pos_5",
    "roll_gap_5",
    # Car / season context
    "car_trend",
    "gap_to_rival",
    "champ_gap",
    "rb_constructor_pos",
    # Lap time features
    "lap_std_ms",
    "pace_dropoff_pct",
    "gap_to_fastest_ms",
    "gap_to_median_ms",
    "ver_lap_rank",
    # Race trajectory
    "recovery_score",
    "positions_gained",
    # Strategy
    "n_stops",
    "retirement_lap_pct",
    # Sprint
    "sprint_pos",
    # Circuit type
    "is_street",
    "is_high_speed",
    "is_high_df",
    "overtake_hard",
    # Fastest lap
    "fastest_lap",
    "fastest_lap_speed",
]
FEATS_WX = ["temp_max", "precipitation", "wind_speed"]


def train(df):
    feats = FEATS_BASE + [f for f in FEATS_WX if f in df.columns]
    feats = [f for f in feats if f in df.columns]
    clean = df.dropna(subset=["position", "win"])
    X = clean[feats].fillna(clean[feats].median()).fillna(0)
    w = clean["recency_w"].fillna(1.0)
    y_pos = clean["position"]
    y_win = clean["win"].astype(int)
    med = clean[feats].median().fillna(0)

    rf_pos = RandomForestRegressor(
        n_estimators=400, max_depth=8, min_samples_leaf=2, random_state=42
    )
    rf_pos.fit(X, y_pos, sample_weight=w)

    rf_win = Pipeline(
        [
            ("sc", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300, max_depth=7, min_samples_leaf=2, random_state=42
                ),
            ),
        ]
    )
    rf_win.fit(X, y_win, rf__sample_weight=w)

    return rf_pos, rf_win, feats, med


def _circuit_profile(df, circ_df_feat, circuit_id):
    row = circ_df_feat[circ_df_feat["circuit_id"] == circuit_id]
    if not row.empty:
        return row.iloc[0].to_dict()
    # fallback: recent overall stats
    recent = df[df["year"] >= df["year"].max() - 1]
    return {
        "circ_win_rate": float(recent["win"].mean()),
        "circ_podium_rate": float(recent["podium"].mean()),
        "circ_avg_finish": float(recent["position"].fillna(20).mean()),
        "circ_dnf_rate": float(recent["dnf"].mean()),
        "circ_mech_dnf_rate": (
            float(recent["dnf_mechanical"].mean())
            if "dnf_mechanical" in recent
            else 0.03
        ),
        "circ_crash_rate": (
            float(recent["dnf_crash"].mean()) if "dnf_crash" in recent else 0.02
        ),
        "circ_avg_gap_pole": (
            float(recent["gap_to_pole"].mean()) if "gap_to_pole" in recent else 0.2
        ),
        "circ_avg_stops": 2.0,
        "circ_races": 0,
    }


def predict_remaining(df, circ_df_feat, rf_pos, rf_win, feats, med):
    completed_rounds = (
        set(df[df["year"] == 2026]["round"].tolist()) if not df.empty else set()
    )

    # Debug: print what 2026 data we actually have
    data_2026 = df[df["year"] == 2026] if not df.empty else pd.DataFrame()
    print(f"  2026 completed rounds in data: {sorted(completed_rounds)}")
    if not data_2026.empty:
        print(f"  2026 circuit IDs from API: {data_2026['circuit_id'].tolist()}")

    # ── Circuit ID aliases — API sometimes returns different IDs ──────────────
    # Map our calendar IDs → what the API actually returns historically
    CID_ALIAS = {
        "albert_park": "albert_park",
        "shanghai": "shanghai",
        "villeneuve": "villeneuve",
        "red_bull_ring": "red_bull_ring",
        "marina_bay": "marina_bay",
        "yas_marina": "yas_marina",
        "americas": "americas",
        "rodriguez": "rodriguez",
        "interlagos": "interlagos",
    }

    STREET = {"monaco", "baku", "marina_bay", "vegas", "jeddah", "miami"}
    HI_SPD = {"monza", "spa", "silverstone", "red_bull_ring", "suzuka"}
    HI_DF = {"monaco", "hungaroring", "marina_bay"}
    OT_HARD = {"monaco", "hungaroring", "marina_bay", "zandvoort", "catalunya"}

    # ── 2026 Red Bull is NOT the fastest car — apply car penalty ─────────────
    # Based on Australia R1: started P20 (crash in quali), finished P6.
    # Mercedes/McLaren/Ferrari are ahead. We penalise gap-to-pole and win rate.
    # This auto-fades as more 2026 races fill completed_rounds and recency kicks in.
    races_2026_done = len(completed_rounds)
    # Penalty scales from full (0 races done) to zero (8+ races done)
    car_penalty = max(0.0, 1.0 - races_2026_done / 8.0)
    # Extra gap-to-pole penalty: ~0.5s when no data, fades to 0 after 8 races
    gap_penalty = 0.55 * car_penalty
    # Win rate suppression factor
    win_suppress = 0.35 * car_penalty  # reduces win prob by up to 35%

    recent5 = df.dropna(subset=["position"]).tail(5)
    roll_win_5 = float(recent5["win"].mean()) * (1 - win_suppress)
    roll_pts_5 = float(recent5["points"].mean()) * (1 - win_suppress * 0.5)
    roll_podium_5 = float(recent5["podium"].mean())
    roll_pos_5 = float(recent5["position"].mean())
    roll_gap_5 = (
        float(recent5["gap_to_pole"].mean()) if "gap_to_pole" in recent5 else 0.15
    )
    roll_gap_5 += gap_penalty
    car_trend = (
        float(df["gap_to_pole"].tail(3).mean()) if "gap_to_pole" in df else 0.2
    ) + gap_penalty
    champ_gap = (
        float(df["champ_gap"].iloc[-1]) if "champ_gap" in df.columns and len(df) else 0
    )
    gap_to_rival = (
        float(df["gap_to_rival"].tail(3).mean()) if "gap_to_rival" in df else 1.0
    )
    gap_to_rival -= car_penalty  # rivals are closer/ahead in 2026

    predictions = []
    for race in CALENDAR_2026:
        rnd = race["round"]
        cid = race["circuit"]

        actual = None
        if rnd in completed_rounds:
            rr = df[(df["year"] == 2026) & (df["round"] == rnd)]
            if not rr.empty:
                pos = rr.iloc[0]["position"]
                status_str = str(rr.iloc[0].get("status", "Finished"))
                valid_finish = status_str == "Finished" or "Lap" in status_str
                actual = int(pos) if valid_finish and pd.notna(pos) else None

        cp = _circuit_profile(df, circ_df_feat, cid)

        quali_pos, grid, gap_pole = 1, 1, car_trend
        sprint_pos_val = np.nan
        n_stops_val = cp.get("circ_avg_stops", 2.0)

        if rnd in completed_rounds:
            q_row = df[(df["year"] == 2026) & (df["round"] == rnd)]
            if not q_row.empty:
                qp = q_row.iloc[0].get("quali_pos", 1)
                gp = q_row.iloc[0].get("grid", 1)
                gtp = q_row.iloc[0].get("gap_to_pole", car_trend)
                sp = q_row.iloc[0].get("sprint_pos", np.nan)
                ns = q_row.iloc[0].get("n_stops", n_stops_val)
                quali_pos = int(qp) if pd.notna(qp) else 1
                grid = int(gp) if pd.notna(gp) else 1
                gap_pole = float(gtp) if pd.notna(gtp) else car_trend
                sprint_pos_val = float(sp) if pd.notna(sp) else np.nan
                n_stops_val = float(ns) if pd.notna(ns) else n_stops_val

        inp = {
            "grid": grid,
            "quali_pos": quali_pos,
            "pole": int(grid == 1),
            "gap_to_pole": gap_pole,
            "circ_win_rate": cp["circ_win_rate"],
            "circ_podium_rate": cp["circ_podium_rate"],
            "circ_avg_finish": cp["circ_avg_finish"],
            "circ_dnf_rate": cp["circ_dnf_rate"],
            "circ_mech_dnf_rate": cp.get("circ_mech_dnf_rate", 0.03),
            "circ_crash_rate": cp.get("circ_crash_rate", 0.02),
            "circ_avg_gap_pole": cp.get("circ_avg_gap_pole", 0.2),
            "circ_avg_stops": n_stops_val,
            "roll_win_5": roll_win_5,
            "roll_pts_5": roll_pts_5,
            "roll_podium_5": roll_podium_5,
            "roll_pos_5": roll_pos_5,
            "roll_gap_5": roll_gap_5,
            "car_trend": car_trend,
            "gap_to_rival": gap_to_rival,
            "champ_gap": champ_gap,
            "rb_constructor_pos": 3.0 * (1 - car_penalty) + 1.0 * car_penalty,
            "n_stops": n_stops_val,
            "sprint_pos": (
                sprint_pos_val
                if pd.notna(sprint_pos_val)
                else float(med.get("sprint_pos", 5.0))
            ),
            "is_street": float(cid in STREET),
            "is_high_speed": float(cid in HI_SPD),
            "is_high_df": float(cid in HI_DF),
            "overtake_hard": float(cid in OT_HARD),
            # Lap features — use circuit historical median or overall median as default
            "lap_std_ms": float(
                med.get(
                    "lap_std_ms",
                    df["lap_std_ms"].median() if "lap_std_ms" in df else 1500,
                )
            ),
            "pace_dropoff_pct": float(
                med.get(
                    "pace_dropoff_pct",
                    (
                        df["pace_dropoff_pct"].median()
                        if "pace_dropoff_pct" in df
                        else 0.5
                    ),
                )
            ),
            "gap_to_fastest_ms": float(
                gap_penalty * 500
                + (df["gap_to_fastest_ms"].median() if "gap_to_fastest_ms" in df else 0)
            ),
            "gap_to_median_ms": float(
                df["gap_to_median_ms"].median() if "gap_to_median_ms" in df else -200
            ),
            "ver_lap_rank": float(
                df["ver_lap_rank"].median() if "ver_lap_rank" in df else 2.0
            ),
            "recovery_score": float(
                df["recovery_score"].median() if "recovery_score" in df else 0.3
            ),
            "positions_gained": float(
                df["positions_gained"].median() if "positions_gained" in df else 1.0
            ),
            "retirement_lap_pct": float(
                df["retirement_lap_pct"].median() if "retirement_lap_pct" in df else 0.5
            ),
            "fastest_lap": float(
                df["fastest_lap"].mean() if "fastest_lap" in df else 0.3
            ),
            "fastest_lap_speed": float(
                df["fastest_lap_speed"].median() if "fastest_lap_speed" in df else 220.0
            ),
        }
        if "temp_max" in feats:
            wx_hist = (
                df[(df["circuit_id"] == cid) & df["temp_max"].notna()]
                if "temp_max" in df.columns
                else pd.DataFrame()
            )
            inp["temp_max"] = (
                float(wx_hist["temp_max"].mean()) if not wx_hist.empty else 25.0
            )
            inp["precipitation"] = (
                float(wx_hist["precipitation"].mean()) if not wx_hist.empty else 0.5
            )
            inp["wind_speed"] = (
                float(wx_hist["wind_speed"].mean()) if not wx_hist.empty else 20.0
            )

        # Only keep features the model was actually trained on
        X_in = pd.DataFrame([inp])[feats].fillna(med).fillna(0)
        pred_pos = float(rf_pos.predict(X_in)[0])
        win_prob = float(rf_win.predict_proba(X_in)[0][1])

        # DNF-adjusted podium prob: high-mech-dnf circuits lower the ceiling
        mech_penalty = cp.get("circ_mech_dnf_rate", 0.03) * 0.5
        pod_prob = min(0.98, win_prob + cp["circ_podium_rate"] * 0.55 - mech_penalty)

        if actual is not None:
            pts = {
                1: 25,
                2: 18,
                3: 15,
                4: 12,
                5: 10,
                6: 8,
                7: 6,
                8: 4,
                9: 2,
                10: 1,
            }.get(actual, 0)
            roll_win_5 = float(roll_win_5) * 0.8 + (1 if actual == 1 else 0) * 0.2
            roll_pts_5 = float(roll_pts_5) * 0.8 + pts * 0.2
            roll_podium_5 = float(roll_podium_5) * 0.8 + (1 if actual <= 3 else 0) * 0.2
            roll_pos_5 = float(roll_pos_5) * 0.8 + actual * 0.2
            roll_gap_5 = float(roll_gap_5) * 0.8 + float(gap_pole) * 0.2
            car_trend = float(car_trend) * 0.7 + float(gap_pole) * 0.3

        # Sprint result for this round if available
        sprint_actual = None
        sprint_pts_actual = 0

        predictions.append(
            {
                "round": rnd,
                "name": race["name"],
                "circuit": cid,
                "date": race["date"],
                "has_sprint": race.get("sprint", False),
                "completed": rnd in completed_rounds,
                "actual": actual,
                "sprint_actual": sprint_actual,
                "sprint_pts": sprint_pts_actual,
                "pred_pos":       float(f"{max(1.0, pred_pos):.1f}"),
                "win_prob":       float(f"{win_prob * 100:.1f}"),
                "podium_prob":    float(f"{pod_prob * 100:.1f}"),
                "win_rate_hist":  float(f"{cp['circ_win_rate'] * 100:.1f}"),
                "is_street":      cid in STREET,
                "mech_risk":      float(f"{cp.get('circ_mech_dnf_rate', 0.03) * 100:.1f}"),
                "crash_risk":     float(f"{cp.get('circ_crash_rate', 0.02) * 100:.1f}"),
                "avg_stops":      float(f"{cp.get('circ_avg_stops', 2.0):.1f}"),
                "circ_races":     int(cp.get("circ_races", 0)),
                "gap_to_pole_avg":float(f"{cp.get('circ_avg_gap_pole', 0.2):.3f}"),
                "car_trend_val":  float(f"{car_trend:.3f}"),
                "champ_gap_val":  float(f"{champ_gap:.0f}"),
            }
        )

    return predictions


def monte_carlo_championship(predictions, n=10000):
    remaining = [p for p in predictions if not p["completed"]]
    pts_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    completed_pts = sum(
        pts_map.get(p["actual"], 0)
        for p in predictions
        if p["completed"] and p["actual"]
    )

    rng = np.random.default_rng(42)
    totals = []

    for _ in range(n):
        season_pts = completed_pts
        for race in remaining:
            wp = min(0.95, race["win_prob"] / 100)
            pp = min(0.95, race["podium_prob"] / 100) - wp
            pp = max(0, pp)
            dnf = (
                race.get("win_rate_hist", 10) / 100 * 0.12
            )  # DNF scales with circuit difficulty
            dnf = min(0.12, max(0.04, dnf))
            tp = max(0, 0.85 - wp - pp - dnf)
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
        "mean":            float(f"{totals.mean():.0f}"),
        "p10":             float(f"{np.percentile(totals, 10):.0f}"),
        "p50":             float(f"{np.percentile(totals, 50):.0f}"),
        "p90":             float(f"{np.percentile(totals, 90):.0f}"),
        "completed_pts":   completed_pts,
        "remaining_races": len(remaining),
        "hist":            np.histogram(totals, bins=30)[0].tolist(),
        "hist_edges":      [float(f"{x:.0f}") for x in np.histogram(totals, bins=30)[1].tolist()],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE — rebuilt on refresh
# ═══════════════════════════════════════════════════════════════════════════════

import pickle

MODEL_PATH = CACHE_DIR / "model.pkl"
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
