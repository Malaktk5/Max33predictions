import pandas as pd
import numpy as np


def build_features(
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
):
    ver = results_df[results_df["driver_id"] == "max_verstappen"].copy()
    if ver.empty:
        return pd.DataFrame(), pd.DataFrame()

    # ── Quali merge ───────────────────────────────────────────────────────────
    if not quali_df.empty:
        ver = ver.merge(
            quali_df[
                [
                    "year",
                    "round",
                    "circuit_id",
                    "quali_pos",
                    "best_quali_sec",
                    "gap_to_pole",
                ]
            ],
            on=["year", "round", "circuit_id"],
            how="left",
        )
    else:
        ver["quali_pos"] = ver["grid"]
        ver["best_quali_sec"] = np.nan
        ver["gap_to_pole"] = np.nan

    # ── Pitstop merge ─────────────────────────────────────────────────────────
    if not pit_df.empty:
        ver = ver.merge(
            pit_df[["year", "round", "n_stops", "avg_pit_sec", "min_pit_sec"]],
            on=["year", "round"],
            how="left",
        )
    else:
        ver["n_stops"] = np.nan
        ver["avg_pit_sec"] = np.nan

    # ── Lap times merge ───────────────────────────────────────────────────────
    if not lap_df.empty:
        ver = ver.merge(
            lap_df[
                [
                    "year",
                    "round",
                    "avg_lap_ms",
                    "best_lap_ms",
                    "lap_std_ms",
                    "pace_dropoff_pct",
                    "pos_gain_early",
                ]
            ],
            on=["year", "round"],
            how="left",
        )
    else:
        for c in [
            "avg_lap_ms",
            "best_lap_ms",
            "lap_std_ms",
            "pace_dropoff_pct",
            "pos_gain_early",
        ]:
            ver[c] = np.nan

    # ── Rival lap times merge ─────────────────────────────────────────────────
    if not rival_lap_df.empty:
        ver = ver.merge(
            rival_lap_df[
                [
                    "year",
                    "round",
                    "gap_to_fastest_ms",
                    "gap_to_median_ms",
                    "ver_lap_rank",
                ]
            ],
            on=["year", "round"],
            how="left",
        )
    else:
        for c in ["gap_to_fastest_ms", "gap_to_median_ms", "ver_lap_rank"]:
            ver[c] = np.nan

    # ── Lap position trajectory merge ─────────────────────────────────────────
    if not lap_pos_df.empty:
        ver = ver.merge(
            lap_pos_df[
                [
                    "year",
                    "round",
                    "pos_start_of_race",
                    "pos_mid_race",
                    "positions_gained",
                    "recovery_score",
                ]
            ],
            on=["year", "round"],
            how="left",
        )
    else:
        for c in [
            "pos_start_of_race",
            "pos_mid_race",
            "positions_gained",
            "recovery_score",
        ]:
            ver[c] = np.nan

    # ── Sprint merge ──────────────────────────────────────────────────────────
    if not sprint_df.empty:
        ver = ver.merge(
            sprint_df[["year", "round", "sprint_pos", "sprint_pts", "sprint_grid"]],
            on=["year", "round"],
            how="left",
        )
    else:
        ver["sprint_pos"] = np.nan

    # ── Weather merge ─────────────────────────────────────────────────────────
    if not wx_df.empty:
        ver = ver.merge(wx_df, on=["year", "circuit_id"], how="left")

    ver = ver.sort_values(["year", "round"]).reset_index(drop=True)

    # ── Basic flags ───────────────────────────────────────────────────────────
    ver["win"] = (ver["position"] == 1).astype(float)
    ver["podium"] = (ver["position"] <= 3).astype(float)
    ver["dnf"] = (ver["status"] != "Finished").astype(float)
    ver["pole"] = (ver["grid"] == 1).astype(float)

    # ── Recency weight — new regs aware ──────────────────────────────────────
    # 2026 is a totally new regulation era — pre-2026 data is much less relevant.
    # 2026 races get weight 1.0, 2025 gets 0.15, 2024 gets 0.08, older gets tiny.
    # As more 2026 races accumulate, pre-2026 data matters less and less.
    races_2026 = (ver["year"] == 2026).sum()
    # Base pre-2026 relevance: starts at 30% (only 2 races known) → fades to 5% at 10 races
    pre_2026_scale = max(0.05, 0.30 - races_2026 * 0.025)

    def reg_weight(row):
        y = row["year"]
        if y == 2026:
            return 1.0
        elif y == 2025:
            return pre_2026_scale
        elif y == 2024:
            return pre_2026_scale * 0.5
        else:
            return pre_2026_scale * 0.2

    ver["recency_w"] = ver.apply(reg_weight, axis=1)

    # ── Constructor position (car pace signal) ────────────────────────────────
    # Red Bull P1 in constructors = dominant car, P3+ = midfield
    ver["rb_constructor_pos"] = ver["year"].map(rb_pos).fillna(1).astype(float)

    # ── Rolling form (last 5 globally) ────────────────────────────────────────
    ver["roll_win_5"] = ver["win"].shift(1).rolling(5, min_periods=1).mean()
    ver["roll_pts_5"] = ver["points"].shift(1).rolling(5, min_periods=1).mean()
    ver["roll_podium_5"] = ver["podium"].shift(1).rolling(5, min_periods=1).mean()
    ver["roll_pos_5"] = ver["position"].shift(1).rolling(5, min_periods=1).mean()
    ver["roll_gap_5"] = ver["gap_to_pole"].shift(1).rolling(5, min_periods=1).mean()

    # ── Car performance trend (rolling 3-race avg gap to pole, recent seasons) ─
    ver["car_trend"] = ver["gap_to_pole"].rolling(3, min_periods=1).mean()

    # ── Competitor gap feature ────────────────────────────────────────────────
    # How many positions ahead of fastest rival per year (recency weighted)
    rivals_df = results_df[results_df["driver_id"] != "max_verstappen"].copy()
    if not rivals_df.empty:
        best_rival = (
            rivals_df.groupby(["year", "round"])["position"]
            .min()
            .reset_index()
            .rename(columns={"position": "best_rival_pos"})
        )
        ver = ver.merge(best_rival, on=["year", "round"], how="left")
        ver["gap_to_rival"] = ver["best_rival_pos"] - ver["position"]
    else:
        ver["gap_to_rival"] = 0

    # ── Championship context ──────────────────────────────────────────────────
    # Points gap to championship leader going into each race
    champ_pts_map = {}
    for y, s in standings.items():
        ver_pts = s.get("max_verstappen", {}).get("champ_pts", 0)
        leader_pts = max((v["champ_pts"] for v in s.values()), default=ver_pts)
        champ_pts_map[int(y)] = ver_pts - leader_pts  # negative = behind leader
    ver["champ_gap"] = ver["year"].map(champ_pts_map).fillna(0)

    # ── Weighted circuit stats ────────────────────────────────────────────────
    circ_rows = []
    for cid, g in ver.groupby("circuit_id"):
        w = g["recency_w"]
        tw = w.sum()
        circ_rows.append(
            {
                "circuit_id": cid,
                "circ_win_rate": float((g["win"] * w).sum() / tw),
                "circ_podium_rate": float((g["podium"] * w).sum() / tw),
                "circ_avg_finish": float((g["position"].fillna(20) * w).sum() / tw),
                "circ_dnf_rate": float((g["dnf"] * w).sum() / tw),
                "circ_mech_dnf_rate": float((g["dnf_mechanical"] * w).sum() / tw),
                "circ_crash_rate": float((g["dnf_crash"] * w).sum() / tw),
                "circ_avg_gap_pole": float(
                    (g["gap_to_pole"].fillna(g["gap_to_pole"].median()) * w).sum() / tw
                ),
                "circ_avg_stops": (
                    float((g["n_stops"].fillna(2) * w).sum() / tw)
                    if "n_stops" in g
                    else 2.0
                ),
                "circ_races": len(g),
            }
        )
    circ_df_feat = pd.DataFrame(circ_rows)
    ver = ver.merge(circ_df_feat, on="circuit_id", how="left")

    # Debug: show what circuit IDs we have and race counts
    print(f"\n  Circuit ID counts in dataset:")
    for cid, cnt in (
        ver.groupby("circuit_id").size().sort_values(ascending=False).head(15).items()
    ):
        print(f"    {cid}: {cnt} races")

    # ── Circuit type features (static knowledge) ──────────────────────────────
    # TRUE street circuits only (temporary barriers, no run-off)
    STREET_CIRCUITS = {"monaco", "baku", "marina_bay", "vegas", "jeddah", "miami"}
    HIGH_SPEED = {"monza", "spa", "silverstone", "red_bull_ring", "suzuka"}
    HIGH_DOWNFORCE = {"monaco", "hungaroring", "marina_bay"}
    OVERTAKE_HARD = {"monaco", "hungaroring", "marina_bay", "zandvoort", "catalunya"}

    ver["is_street"] = [float(cid in STREET_CIRCUITS) for cid in ver["circuit_id"]]
    ver["is_high_speed"] = [float(cid in HIGH_SPEED) for cid in ver["circuit_id"]]
    ver["is_high_df"] = [float(cid in HIGH_DOWNFORCE) for cid in ver["circuit_id"]]
    ver["overtake_hard"] = [float(cid in OVERTAKE_HARD) for cid in ver["circuit_id"]]

    return ver, circ_df_feat
