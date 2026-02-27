import pandas as pd
import numpy as np
from pathlib import Path

SEED = 42
NUM_FACTORIES = 5
NUM_PRODUCTS = 10
NUM_CLIENTS = 20
MIN_ORDER_QTY = 2000
MAX_SHIPS_PER_WEEK = 20
START_DATE = "2025-01-01"
END_DATE = "2025-12-31"
QUAY_SPLIT = [3, 2]  # berths per quay (must sum to NUM_FACTORIES)


def build_factory_product_map() -> dict[str, list[str]]:
    """Map each factory to its 2 exclusive products."""
    mapping = {}
    for i in range(NUM_FACTORIES):
        factory = f"Factory_{i + 1}"
        product_a = f"Product_{2 * i + 1}"
        product_b = f"Product_{2 * i + 2}"
        mapping[factory] = [product_a, product_b]
    return mapping


def generate_demand_profiles(products: list[str], rng: np.random.Generator) -> dict[str, float]:
    """Assign each product a unique base daily production rate (tons/day)."""
    base_values = np.linspace(500, 15_000, len(products))
    rng.shuffle(base_values)
    return dict(zip(products, base_values))


def generate_production_plan() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    factory_products = build_factory_product_map()
    all_products = [p for prods in factory_products.values() for p in prods]
    base_demand = generate_demand_profiles(all_products, rng)

    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    day_of_year = np.arange(len(dates))
    seasonal_factor = 1.0 + 0.10 * np.sin(2 * np.pi * day_of_year / 365)

    records: list[dict] = []

    for factory, products in factory_products.items():
        for product in products:
            base = base_demand[product]
            noise = rng.normal(loc=1.0, scale=0.15, size=len(dates))
            daily_production = base * noise * seasonal_factor

            is_weekend = np.isin(dates.dayofweek, [5, 6])
            daily_production[is_weekend] *= 0.05

            daily_production = np.clip(daily_production, 0, None).round().astype(int)

            for dt, qty in zip(dates, daily_production):
                records.append(
                    {
                        "Date": dt.strftime("%Y-%m-%d"),
                        "Factory": factory,
                        "Product": product,
                        "Production": qty,
                    }
                )

    df = pd.DataFrame(records)
    df = df.sort_values(["Date", "Factory", "Product"]).reset_index(drop=True)
    return df


def generate_client_orders(production_df: pd.DataFrame) -> pd.DataFrame:
    """Generate week-aligned client orders.

    Every ship's [ArrivalDate, Deadline] falls entirely within a single
    Monday-Sunday week so the weekly optimizer can load 95%+ of demand
    and the model stays within Gurobi license limits.
    """
    rng = np.random.default_rng(SEED + 1)

    factory_products = build_factory_product_map()
    all_products = [p for prods in factory_products.values() for p in prods]
    dates = pd.date_range(START_DATE, END_DATE, freq="D")

    # Daily production per product
    prod_agg = (
        production_df
        .assign(Date=lambda d: pd.to_datetime(d["Date"]))
        .groupby(["Date", "Product"])["Production"]
        .sum()
        .reset_index()
    )
    daily_prod: dict[str, pd.Series] = {}
    for product in all_products:
        series = prod_agg[prod_agg["Product"] == product].set_index("Date")["Production"]
        daily_prod[product] = series.reindex(dates, fill_value=0)

    # Week boundaries (same logic as run_model.py)
    week_mondays = pd.date_range(START_DATE, END_DATE, freq="W-MON")
    w_starts = [pd.Timestamp(START_DATE)] + list(week_mondays)
    w_ends = [m - pd.Timedelta(days=1) for m in week_mondays] + [pd.Timestamp(END_DATE)]

    client_names = [f"Client_{i + 1}" for i in range(NUM_CLIENTS)]
    records: list[dict] = []
    ship_counter = 1
    client_idx = 0

    for ws, we in zip(w_starts, w_ends):
        week_dates = pd.date_range(ws, we, freq="D")
        n_days = len(week_dates)
        if n_days < 3:
            continue

        week_orders: list[dict] = []

        for product in all_products:
            weekly_production = float(daily_prod[product].loc[ws:we].sum())
            if weekly_production < MIN_ORDER_QTY:
                continue

            n_ships = 1 if weekly_production < MIN_ORDER_QTY * 3 else int(rng.integers(1, 3))

            demand_fraction = rng.uniform(0.85, 0.92)
            total_demand = weekly_production * demand_fraction

            if n_ships == 1:
                ship_demands = [int(round(total_demand))]
            else:
                split = rng.uniform(0.35, 0.65)
                ship_demands = [
                    int(round(total_demand * split)),
                    int(round(total_demand * (1 - split))),
                ]

            for demand in ship_demands:
                if demand < MIN_ORDER_QTY:
                    continue

                max_arr_offset = max(0, n_days - 4)
                arr_offset = int(rng.integers(0, max_arr_offset + 1))

                max_window = n_days - 1 - arr_offset
                if max_window >= 3:
                    window = int(rng.integers(3, max_window + 1))
                else:
                    window = max_window

                arr_date = week_dates[arr_offset]
                dl_date = week_dates[arr_offset + window]

                week_orders.append({
                    "Client": client_names[client_idx % NUM_CLIENTS],
                    "Ship": f"Ship_{ship_counter}",
                    "ArrivalDate": arr_date.strftime("%Y-%m-%d"),
                    "Deadline": dl_date.strftime("%Y-%m-%d"),
                    "Product": product,
                    "Quantity": demand,
                })
                ship_counter += 1
                client_idx += 1

        if len(week_orders) > MAX_SHIPS_PER_WEEK:
            week_orders.sort(key=lambda x: x["Quantity"], reverse=True)
            week_orders = week_orders[:MAX_SHIPS_PER_WEEK]

        records.extend(week_orders)

    df = pd.DataFrame(records)
    df = df.sort_values(["ArrivalDate", "Client", "Ship"]).reset_index(drop=True)
    return df


def compute_stock_balance(
    production_df: pd.DataFrame, orders_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute daily cumulative production, cumulative demand, and stock balance per product.

    Demand is attributed to the ship's Deadline date (the day it must be fulfilled).
    A negative Balance means a shortage; positive means surplus stock.
    """
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    all_products = sorted(production_df["Product"].unique())

    # Daily production per product
    daily_prod = (
        production_df
        .assign(Date=pd.to_datetime(production_df["Date"]))
        .groupby(["Date", "Product"])["Production"]
        .sum()
        .reset_index()
    )

    # Daily demand per product, placed on the Deadline date
    daily_demand = (
        orders_df
        .assign(Deadline=pd.to_datetime(orders_df["Deadline"]))
        .groupby(["Deadline", "Product"])["Quantity"]
        .sum()
        .reset_index()
        .rename(columns={"Deadline": "Date", "Quantity": "Demand"})
    )

    rows: list[dict] = []
    for product in all_products:
        prod_series = (
            daily_prod[daily_prod["Product"] == product]
            .set_index("Date")["Production"]
            .reindex(dates, fill_value=0)
        )
        demand_series = (
            daily_demand[daily_demand["Product"] == product]
            .set_index("Date")["Demand"]
            .reindex(dates, fill_value=0)
        )

        cum_prod = prod_series.cumsum()
        cum_demand = demand_series.cumsum()
        balance = cum_prod - cum_demand

        for dt, cp, cd, bal in zip(dates, cum_prod, cum_demand, balance):
            rows.append(
                {
                    "Date": dt.strftime("%Y-%m-%d"),
                    "Product": product,
                    "CumulativeProduction": int(cp),
                    "CumulativeDemand": int(cd),
                    "Balance": int(bal),
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["Date", "Product"]).reset_index(drop=True)
    return df


def generate_infrastructure_specs(
    orders_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate ship_specs, port_specs, and axis_specs aligned with order demands.

    Ship types are derived from per-trip demand percentiles so that every trip
    maps to a SubClass whose DeadWeight >= trip demand.  Berth dimensions,
    crane capacities, and axis capacities are then sized to guarantee that every
    ship type can dock and be loaded within its loading window.

    Returns (ship_specs, port_specs, axis_specs).
    """
    orders = orders_df.copy()
    orders["ArrivalDate"] = pd.to_datetime(orders["ArrivalDate"])
    orders["Deadline"] = pd.to_datetime(orders["Deadline"])
    orders["WindowDays"] = (orders["Deadline"] - orders["ArrivalDate"]).dt.days

    # Per-trip demand (one row = one trip since each ship carries one product)
    trip_demands = orders["Quantity"].values
    min_window = int(orders["WindowDays"].min())
    max_demand = int(trip_demands.max())

    # ---- Ship specs (4 types x 3 subclasses = 12 rows) ----
    # DeadWeight tiers cover the full demand range with headroom
    dw_max = int(np.ceil(max_demand * 1.15 / 1000) * 1000)
    dw_min = 5000
    dead_weights = np.linspace(dw_min, dw_max, 12).astype(int)
    # Round to nearest 1000 for cleanliness
    dead_weights = (np.round(dead_weights / 1000) * 1000).astype(int)

    size_labels = ["small", "medium", "large"]
    type_names = [f"Type_{i}" for i in range(1, 5) for _ in range(3)]
    sub_classes = [
        f"{t}_{s}" for t, s in zip(type_names, size_labels * 4)
    ]

    # Loading rate per type: the largest subclass in each type must be loadable
    # within min_window days.  All subclasses of a type share the same rate.
    type_loading_rates: dict[str, int] = {}
    for idx in range(0, 12, 3):
        t = type_names[idx]
        worst_dw = dead_weights[idx + 2]
        rate = int(np.ceil(worst_dw / min_window / 1000) * 1000)
        type_loading_rates[t] = rate
    loading_rates = [type_loading_rates[t] for t in type_names]

    # Length and Draft scale linearly with DeadWeight
    lengths = np.linspace(90, 260, 12).astype(int)
    drafts = np.round(np.linspace(5.0, 16.5, 12), 1)

    ship_specs = pd.DataFrame(
        {
            "ShipType": type_names,
            "SubClass": sub_classes,
            "LoadingRate": loading_rates,
            "DeadWeight": dead_weights,
            "Length": lengths,
            "Draft": drafts,
        }
    )

    # ---- Port & Axis specs sized per-factory ----
    # Chain: Factory_i -> Axis_i -> Berth_i (positional, 1:1:1).
    # Each berth is sized to fit the largest ship that will actually dock there,
    # based on the max per-trip demand for that factory's products.
    factory_products = build_factory_product_map()
    product_to_factory = {p: f for f, prods in factory_products.items() for p in prods}

    orders_tmp = orders_df.copy()
    orders_tmp["Factory"] = orders_tmp["Product"].map(product_to_factory)
    max_demand_per_factory = orders_tmp.groupby("Factory")["Quantity"].max().to_dict()

    sorted_ship = ship_specs.sort_values("DeadWeight").reset_index(drop=True)
    factories = [f"Factory_{i + 1}" for i in range(NUM_FACTORIES)]

    port_rows: list[dict] = []
    axis_rows: list[dict] = []
    factory_idx = 0

    for qi, (quay, n_berths) in enumerate(zip(["Q1", "Q2"], QUAY_SPLIT)):
        for i in range(n_berths):
            factory = factories[factory_idx]
            fmax = max_demand_per_factory.get(factory, 0)

            eligible = sorted_ship[sorted_ship["DeadWeight"] >= fmax]
            spec = eligible.iloc[0] if len(eligible) > 0 else sorted_ship.iloc[-1]

            ship_rate = int(spec["LoadingRate"])
            nc = 2 if ship_rate > 50_000 else 1
            crane_cap = int(np.ceil(ship_rate / nc / 1000) * 1000)

            letter = chr(ord("A") + i)
            port_rows.append(
                {
                    "Quay": quay,
                    "Berth": f"{quay}{letter}",
                    "MaxDeadWeight": int(spec["DeadWeight"] * 1.05),
                    "MaxLength": int(spec["Length"] + 15),
                    "MaxDraft": round(float(spec["Draft"]) + 0.5, 1),
                    "NumCranes": nc,
                    "CraneCapacity": crane_cap,
                }
            )
            axis_rows.append(
                {
                    "Axis": f"Axis_{factory_idx + 1}",
                    "AxisCapacity": ship_rate,
                    "Quay": quay,
                }
            )
            factory_idx += 1

    port_specs = pd.DataFrame(port_rows)
    axis_specs = pd.DataFrame(axis_rows)

    return ship_specs, port_specs, axis_specs


def _build_chain_mapping(
    port_specs: pd.DataFrame, axis_specs: pd.DataFrame
) -> dict[str, dict]:
    """Build the Factory -> Axis -> Berth 1:1:1 chain.

    Factory_i maps to Axis_i maps to the i-th berth (ordered by QUAY_SPLIT).
    Returns {product: {factory, axis, berth, quay}}.
    """
    factory_products = build_factory_product_map()
    factories = [f"Factory_{i + 1}" for i in range(NUM_FACTORIES)]

    berths: list[str] = []
    axes_list: list[str] = []
    factory_idx = 0
    for qi, (quay, n_berths) in enumerate(zip(["Q1", "Q2"], QUAY_SPLIT)):
        quay_berths = port_specs[port_specs["Quay"] == quay]["Berth"].tolist()
        quay_axes = axis_specs[axis_specs["Quay"] == quay]["Axis"].tolist()
        for i in range(n_berths):
            berths.append(quay_berths[i])
            axes_list.append(quay_axes[i])
            factory_idx += 1

    factory_to_axis = dict(zip(factories, axes_list))
    factory_to_berth = dict(zip(factories, berths))

    chain: dict[str, dict] = {}
    for factory, products in factory_products.items():
        for p in products:
            chain[p] = {
                "Factory": factory,
                "Axis": factory_to_axis[factory],
                "Berth": factory_to_berth[factory],
            }
    return chain


def compute_loading_feasibility(
    orders_df: pd.DataFrame,
    ship_specs: pd.DataFrame,
    port_specs: pd.DataFrame,
    axis_specs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check end-to-end loading feasibility: Factory -> Axis -> Berth -> Ship.

    Returns two DataFrames:
      trip_feasibility  -- one row per order with ship assignment, chain info,
                           and pass/fail flags.
      berth_balance     -- daily (Date, Berth) cumulative capacity vs. demand,
                           analogous to compute_stock_balance.
    """
    chain = _build_chain_mapping(port_specs, axis_specs)
    sorted_specs = ship_specs.sort_values("DeadWeight").reset_index(drop=True)
    port_idx = port_specs.set_index("Berth")
    axis_idx = axis_specs.set_index("Axis")

    orders = orders_df.copy()
    orders["ArrivalDate"] = pd.to_datetime(orders["ArrivalDate"])
    orders["Deadline"] = pd.to_datetime(orders["Deadline"])
    orders["WindowDays"] = (orders["Deadline"] - orders["ArrivalDate"]).dt.days

    # ---- Per-trip feasibility ----
    trip_rows: list[dict] = []
    for _, row in orders.iterrows():
        demand = row["Quantity"]
        product = row["Product"]
        window = row["WindowDays"]

        c = chain[product]
        berth = c["Berth"]
        axis = c["Axis"]

        # Assign smallest ship subclass whose DeadWeight >= demand
        eligible = sorted_specs[sorted_specs["DeadWeight"] >= demand]
        spec = eligible.iloc[0] if len(eligible) > 0 else sorted_specs.iloc[-1]

        ship_dw = int(spec["DeadWeight"])
        ship_len = int(spec["Length"])
        ship_draft = float(spec["Draft"])
        ship_rate = int(spec["LoadingRate"])

        berth_cap = int(port_idx.loc[berth, "NumCranes"] * port_idx.loc[berth, "CraneCapacity"])
        axis_cap = int(axis_idx.loc[axis, "AxisCapacity"])
        effective_rate = min(ship_rate, berth_cap, axis_cap)

        loading_days = int(np.ceil(demand / effective_rate)) if effective_rate > 0 else 999

        fits_dw = ship_dw <= int(port_idx.loc[berth, "MaxDeadWeight"])
        fits_len = ship_len <= int(port_idx.loc[berth, "MaxLength"])
        fits_draft = ship_draft <= float(port_idx.loc[berth, "MaxDraft"])
        ship_can_carry = ship_dw >= demand
        can_load = loading_days <= window

        trip_rows.append(
            {
                "Client": row["Client"],
                "Ship": row["Ship"],
                "ArrivalDate": row["ArrivalDate"].strftime("%Y-%m-%d"),
                "Deadline": row["Deadline"].strftime("%Y-%m-%d"),
                "Product": product,
                "Quantity": demand,
                "WindowDays": window,
                "Factory": c["Factory"],
                "Axis": axis,
                "Berth": berth,
                "ShipSubClass": spec["SubClass"],
                "ShipDeadWeight": ship_dw,
                "EffectiveLoadRate": effective_rate,
                "LoadingDaysNeeded": loading_days,
                "FitsBerth": fits_dw and fits_len and fits_draft,
                "ShipCanCarry": ship_can_carry,
                "CanLoadInTime": can_load,
                "Feasible": (fits_dw and fits_len and fits_draft and ship_can_carry and can_load),
            }
        )

    trip_df = pd.DataFrame(trip_rows)

    # ---- Per-berth daily balance (cumulative capacity vs. demand) ----
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    n_days = len(dates)

    # Effective daily capacity per berth = min(berth_capacity, axis_capacity)
    berth_daily_cap: dict[str, int] = {}
    for _, pr in port_specs.iterrows():
        b = pr["Berth"]
        b_cap = int(pr["NumCranes"] * pr["CraneCapacity"])
        a_cap = int(axis_idx.loc[
            axis_specs.loc[axis_specs["Quay"] == pr["Quay"]].iloc[
                port_specs[port_specs["Quay"] == pr["Quay"]]["Berth"].tolist().index(b)
            ]["Axis"],
            "AxisCapacity",
        ])
        berth_daily_cap[b] = min(b_cap, a_cap)

    # Demand attributed to deadline date per berth
    orders["Berth"] = orders["Product"].map(lambda p: chain[p]["Berth"])
    demand_by_berth = (
        orders.groupby(["Deadline", "Berth"])["Quantity"]
        .sum()
        .reset_index()
        .rename(columns={"Deadline": "Date", "Quantity": "Demand"})
    )

    berth_rows: list[dict] = []
    for berth in sorted(port_specs["Berth"]):
        cap = berth_daily_cap[berth]
        demand_series = (
            demand_by_berth[demand_by_berth["Berth"] == berth]
            .set_index("Date")["Demand"]
            .reindex(dates, fill_value=0)
        )
        cum_cap = cap * np.arange(1, n_days + 1)
        cum_dem = demand_series.cumsum().values

        for dt, cc, cd in zip(dates, cum_cap, cum_dem):
            berth_rows.append(
                {
                    "Date": dt.strftime("%Y-%m-%d"),
                    "Berth": berth,
                    "DailyCapacity": cap,
                    "CumulativeCapacity": int(cc),
                    "CumulativeDemand": int(cd),
                    "Balance": int(cc - cd),
                }
            )

    berth_df = pd.DataFrame(berth_rows)
    berth_df = berth_df.sort_values(["Date", "Berth"]).reset_index(drop=True)

    return trip_df, berth_df


if __name__ == "__main__":
    output_path = Path(__file__).resolve().parent.parent / "data" / "synthetic"
    output_path.mkdir(parents=True, exist_ok=True)

    production_df = generate_production_plan()
    production_file = output_path / "production_plan.csv"
    production_df.to_csv(production_file, index=False)
    print(f"Saved {len(production_df)} rows to {production_file}")

    orders_df = generate_client_orders(production_df)
    orders_file = output_path / "client_orders.csv"
    orders_df.to_csv(orders_file, index=False)
    print(f"Saved {len(orders_df)} rows to {orders_file}")

    balance_df = compute_stock_balance(production_df, orders_df)
    balance_file = output_path / "stock_balance.csv"
    balance_df.to_csv(balance_file, index=False)
    print(f"Saved {len(balance_df)} rows to {balance_file}")

    shortages = balance_df[balance_df["Balance"] < 0]
    if shortages.empty:
        print("No shortages detected -- all demand is covered by production.")
    else:
        n_products = shortages["Product"].nunique()
        worst = shortages.loc[shortages["Balance"].idxmin()]
        print(
            f"Shortages detected for {n_products} product(s). "
            f"Worst: {worst['Product']} on {worst['Date']} "
            f"(balance = {worst['Balance']:,})"
        )

    ship_specs, port_specs, axis_specs = generate_infrastructure_specs(orders_df)

    for name, df in [("ship_specs", ship_specs), ("port_specs", port_specs), ("axis_specs", axis_specs)]:
        path = output_path / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {len(df)} rows to {path}")

    print("\n--- Ship specs ---")
    print(ship_specs.to_string(index=False))
    print("\n--- Port specs ---")
    print(port_specs.to_string(index=False))
    print("\n--- Axis specs ---")
    print(axis_specs.to_string(index=False))

    # ---- Loading feasibility ----
    trip_feas, berth_bal = compute_loading_feasibility(
        orders_df, ship_specs, port_specs, axis_specs
    )

    trip_feas.to_csv(output_path / "trip_feasibility.csv", index=False)
    print(f"\nSaved {len(trip_feas)} rows to {output_path / 'trip_feasibility.csv'}")
    berth_bal.to_csv(output_path / "berth_balance.csv", index=False)
    print(f"Saved {len(berth_bal)} rows to {output_path / 'berth_balance.csv'}")

    # Per-trip summary
    n_infeasible = (~trip_feas["Feasible"]).sum()
    if n_infeasible == 0:
        print("All trips are feasible (ship fits berth, can carry & load in time).")
    else:
        print(f"\n{n_infeasible} infeasible trip(s) detected:")
        for flag in ["FitsBerth", "ShipCanCarry", "CanLoadInTime"]:
            n_fail = (~trip_feas[flag]).sum()
            if n_fail:
                print(f"  - {flag} failed: {n_fail} trips")

    # Berth balance summary
    berth_shortages = berth_bal[berth_bal["Balance"] < 0]
    if berth_shortages.empty:
        print("No berth capacity shortages -- all berths can handle their demand.")
    else:
        for berth in berth_shortages["Berth"].unique():
            worst = berth_shortages[berth_shortages["Berth"] == berth].iloc[-1]
            print(
                f"  Berth {berth}: shortage by {worst['Date']} "
                f"(balance = {worst['Balance']:,})"
            )
