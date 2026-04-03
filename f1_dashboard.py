"""
VER33 F1 Prediction Dashboard
CLI Entrypoint
"""

import argparse
import time
import warnings

from f1_api import collect_all_data, prefetch_weather_background, CACHE_DIR
from f1_features import build_features
from f1_models import (
    train,
    predict_remaining,
    monte_carlo_championship,
    load_model,
    save_model,
)

from typing import Any, Dict

warnings.filterwarnings("ignore")

STATE: Dict[str, Any] = {
    "predictions": [],
    "championship": {},
    "last_updated": None,
    "loading": False,
    "error": None,
}


def refresh_data():
    STATE["loading"] = True
    STATE["error"] = None
    try:
        print("\n── Refreshing data ──")
        (
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
        ) = collect_all_data()
        df, circ_df_feat = build_features(
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
        if df.empty:
            STATE["error"] = (
                "No data returned from API — check your internet connection."
            )
            print("  No data available.")
            return

        cached = load_model(df)
        if cached:
            rf_pos, rf_win, feats, med = cached
        else:
            rf_pos, rf_win, feats, med = train(df)
            save_model(rf_pos, rf_win, feats, med, df)

        STATE["predictions"] = predict_remaining(
            df, circ_df_feat, rf_pos, rf_win, feats, med
        )
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
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════


def print_predictions(preds):
    print("\n" + "═" * 95)
    print(" 🏁 UPCOMING 2026 RACE PREDICTIONS")
    print("═" * 95)
    print(
        f"{'Rnd':<4} | {'Grand Prix':<20} | {'Status':<8} | {'Actual/Pred':<11} | {'Win %':<6} | {'Podium %':<8} | {'Hist Win %'}"
    )
    print("─" * 95)
    for p in preds:
        status = "DONE" if p["completed"] else "UPCOMING"
        if not p["completed"]:
            print(
                f"{p['round']:<4} | {p['name']:<20} | {status:<8} | P{p['pred_pos']:<10} | {p['win_prob']:>5.1f}% | {p['podium_prob']:>7.1f}% | {p['win_rate_hist']:>8.1f}%"
            )
        else:
            actual = f"P{int(p['actual'])}" if p["actual"] else "DNF"
            print(
                f"{p['round']:<4} | {p['name']:<20} | {status:<8} | {actual:<11} | {'-':>5}  | {'-':>7}  | {'-':>8}"
            )


def print_championship(champ):
    print("\n" + "═" * 60)
    print(" 🏆 CHAMPIONSHIP MONTE CARLO SIMULATION (10k Runs)")
    print("═" * 60)
    print(f"  Current Points:    {champ['completed_pts']}")
    print(f"  Races Remaining:   {champ['remaining_races']}")
    print(f"  Projected Mean:    {champ['mean']} pts")
    print("─" * 60)
    print("  Probable Outcomes:")
    print(f"   - 10th Percentile (Worst-case):   {champ['p10']} pts")
    print(f"   - 50th Percentile (Median):       {champ['p50']} pts")
    print(f"   - 90th Percentile (Best-case):    {champ['p90']} pts")
    print("═" * 60 + "\n")


def main():
    print("\n" + "=" * 60)
    print("  VER33 F1 Dashboard · Terminal Mode")
    print("=" * 60 + "\n")

    refresh_data()

    if STATE["error"]:
        print(f"\n[ERROR] {STATE['error']}")
        return

    if STATE["predictions"]:
        print_predictions(STATE["predictions"])
    if STATE["championship"]:
        print_championship(STATE["championship"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VER33 F1 Oracle CLI")
    parser.add_argument(
        "--force-refresh", action="store_true", help="Force flush cache and refresh"
    )
    args = parser.parse_args()

    if args.force_refresh:
        import shutil

        if CACHE_DIR.exists():
            print("Clearing cache directory...")
            shutil.rmtree(CACHE_DIR)
            CACHE_DIR.mkdir()

    main()
