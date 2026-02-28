"""Run the port logistics optimisation model over the full year, week by week.

Each week's results are saved to src/data/results/week_NN/.
Adjust weights and solver params in the configuration section below.
"""

from pathlib import Path
import json

import pandas as pd
import gurobipy as gp

from data_pipeline import DataPipeline
from model_constraints import PortModel

# ---- Configuration ----
YEAR_START = "2025-01-01"
YEAR_END = "2025-12-31"

W_LOAD = 1.0    # primary:   maximise loaded quantity
W_STAY = 100.0  # secondary: minimise ship stay time
W_MOVE = 10.0   # tertiary:  minimise berth movements

TIME_LIMIT = 120   # seconds per week
MIP_GAP = 0.01     # 1 %

RESULTS_DIR = Path(__file__).resolve().parents[1] / "data" / "results"

# ---- Weekly loop ----
weeks = pd.date_range(YEAR_START, YEAR_END, freq="W-MON")
starts = [pd.Timestamp(YEAR_START)] + list(weeks)
ends = [w - pd.Timedelta(days=1) for w in weeks] + [pd.Timestamp(YEAR_END)]

yearly_summary: list[dict] = []

for week_num, (start, end) in enumerate(zip(starts, ends), start=1):
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    print("=" * 65)
    print(f"WEEK {week_num:02d}  |  {start_str} -> {end_str}")
    print("=" * 65)

    dp = DataPipeline(start_str, end_str)
    data = dp.working_data_compute()

    n_days = len(data["timestamps"])
    n_ships = len(data["ships"])
    if n_days < 7 or n_ships == 0:
        reason = "partial week" if n_days < 7 else "no ships"
        print(f"  Skipping ({reason}, {n_days} days, {n_ships} ships).\n")
        yearly_summary.append(
            {"week": week_num, "start": start_str, "end": end_str,
             "ships": n_ships, "demand": 0, "loaded": 0, "pct": 0,
             "movements": 0, "avg_stay": 0, "status": "skipped"}
        )
        continue

    print(f"  Ships: {n_ships}  |  Days: {len(data['timestamps'])}")

    try:
        model = PortModel(data, w_load=W_LOAD, w_stay=W_STAY, w_move=W_MOVE)
    except gp.GurobiError as exc:
        print(f"  ** Skipped (Gurobi error: {exc}) **\n")
        yearly_summary.append(
            {"week": week_num, "start": start_str, "end": end_str,
             "ships": n_ships, "demand": 0, "loaded": 0, "pct": 0,
             "movements": 0, "avg_stay": 0, "status": "license_limit"}
        )
        continue

    print(f"  Vars: {model.model.NumVars}  |  Constrs: {model.model.NumConstrs}")

    try:
        model.solve(time_limit=TIME_LIMIT, mip_gap=MIP_GAP)
    except gp.GurobiError as exc:
        print(f"  ** Solve failed (Gurobi error: {exc}) **\n")
        yearly_summary.append(
            {"week": week_num, "start": start_str, "end": end_str,
             "ships": n_ships, "demand": 0, "loaded": 0, "pct": 0,
             "movements": 0, "avg_stay": 0, "status": "license_limit"}
        )
        continue

    week_dir = RESULTS_DIR / f"week_{week_num:02d}"
    model.save_results(output_dir=week_dir)

    if model.model.SolCount > 0:
        total_demand = sum(model.ship_demand[s] for s in model.ships)
        total_loaded = sum(
            model.load[t, s].X
            for t in model.timestamp_idx for s in model.ships
        )
        total_moves = sum(
            model.movement[t, s].X
            for t in model.timestamp_idx for s in model.ships
        )
        avg_stay = 0.0
        for s in model.ships:
            etb_d = sum(t * model.etb[t, s].X for t in model.timestamp_idx)
            ets_d = sum(t * model.ets[t, s].X for t in model.timestamp_idx)
            avg_stay += ets_d - etb_d
        avg_stay /= max(n_ships, 1)

        pct = round(total_loaded / total_demand * 100, 1) if total_demand else 0
        print(f"  Loaded: {total_loaded:,.0f} / {total_demand:,.0f} ({pct}%)")
        print(f"  Movements: {int(total_moves)}  |  Avg stay: {avg_stay:.1f}d")

        yearly_summary.append(
            {"week": week_num, "start": start_str, "end": end_str,
             "ships": n_ships, "demand": total_demand,
             "loaded": round(total_loaded, 2), "pct": pct,
             "movements": int(total_moves), "avg_stay": round(avg_stay, 1),
             "status": "optimal" if model.model.MIPGap <= MIP_GAP else "feasible"}
        )
    else:
        print("  ** No feasible solution found **")
        yearly_summary.append(
            {"week": week_num, "start": start_str, "end": end_str,
             "ships": n_ships, "demand": 0, "loaded": 0, "pct": 0,
             "movements": 0, "avg_stay": 0, "status": "infeasible"}
        )
    print()

# ---- Yearly summary CSV ----
summary_df = pd.DataFrame(yearly_summary)
summary_path = RESULTS_DIR / "yearly_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("=" * 65)
print("YEARLY SUMMARY")
print("=" * 65)
print(f"Weeks solved:    {len(summary_df)}")
print(f"Total demand:    {summary_df['demand'].sum():,.0f}")
print(f"Total loaded:    {summary_df['loaded'].sum():,.0f}")
total_d = summary_df["demand"].sum()
if total_d > 0:
    print(f"Overall loaded:  {summary_df['loaded'].sum() / total_d * 100:.1f}%")
print(f"Saved summary to {summary_path}")

# ---- Consolidated ships/load summaries ----
week_dirs = sorted(
    [path for path in RESULTS_DIR.glob("week_*") if path.is_dir()],
    key=lambda path: int(path.name.split("_")[1]),
)

berth_frames: list[pd.DataFrame] = []
etb_frames: list[pd.DataFrame] = []
ets_frames: list[pd.DataFrame] = []
loading_frames: list[pd.DataFrame] = []
ship_results: list[dict] = []

for week_dir in week_dirs:
    at_berth_path = week_dir / "at_berth.csv"
    etb_path = week_dir / "etb.csv"
    ets_path = week_dir / "ets.csv"
    load_path = week_dir / "load.csv"
    objective_path = week_dir / "objective.json"

    required_files = [at_berth_path, etb_path, ets_path, load_path, objective_path]
    if not all(path.exists() for path in required_files):
        continue

    berth_frames.append(pd.read_csv(at_berth_path).drop(columns=["Value"], errors="ignore"))
    etb_frames.append(pd.read_csv(etb_path).drop(columns=["Value"], errors="ignore"))
    ets_frames.append(pd.read_csv(ets_path).drop(columns=["Value"], errors="ignore"))
    loading_frames.append(pd.read_csv(load_path))

    with objective_path.open(encoding="utf-8") as f:
        per_ship = json.load(f).get("per_ship", {})
    for ship, data in per_ship.items():
        row = dict(data)
        row["ship"] = ship
        ship_results.append(row)

if berth_frames and etb_frames and ets_frames and loading_frames and ship_results:
    df_berth_results = pd.concat(berth_frames, ignore_index=True)
    df_etb_results = pd.concat(etb_frames, ignore_index=True)
    df_ets_results = pd.concat(ets_frames, ignore_index=True)
    df_loading_results = pd.concat(loading_frames, ignore_index=True)

    ship_results_df = pd.DataFrame(ship_results).drop(
        columns=["etb_day_idx", "ets_day_idx"], errors="ignore"
    )

    df_ships = ship_results_df.merge(
        df_etb_results.rename(columns={"Ship": "ship", "Date": "etb"}),
        on="ship",
        how="left",
    ).merge(
        df_ets_results.rename(columns={"Ship": "ship", "Date": "ets"}),
        on="ship",
        how="left",
    )
    df_ships = df_ships[
        ["ship", "etb", "ets", "berth", "product", "demand", "loaded", "pct_loaded"]
    ]

    ships_summary_path = RESULTS_DIR / "ships_summary.csv"
    df_ships.to_csv(ships_summary_path, index=False)

    df_load = (
        df_berth_results.merge(
            df_loading_results,
            on=["Date", "Ship"],
            how="left",
        )
        .rename(columns={"Date": "date", "Ship": "ship", "Value": "load"})
        .fillna(0)
        .merge(
            df_ships[["ship", "berth", "product", "demand"]],
            on="ship",
            how="left",
        )
        .sort_values(by="date")
        .reset_index(drop=True)
    )
    df_load["cumulative_load"] = df_load.groupby(["ship", "product"])["load"].cumsum()
    df_load["cumulative_load_pct"] = (df_load["cumulative_load"] / df_load["demand"]).round(4)

    loading_summary_path = RESULTS_DIR / "loading_summary.csv"
    df_load.to_csv(loading_summary_path, index=False)

    print(f"Saved ships summary to {ships_summary_path}")
    print(f"Saved loading summary to {loading_summary_path}")
else:
    print("Skipped ships/loading summaries (missing weekly outputs).")
