from pathlib import Path

import pandas as pd
import numpy as np

from generate_synthetic_data import build_factory_product_map, NUM_FACTORIES


class DataPipeline:
    """Load synthetic data, filter by date range, and prepare working data for Gurobi."""

    DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic"

    def __init__(self, start: str, end: str) -> None:
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)

        self._load_raw()
        self._filter_and_process()

    # ------------------------------------------------------------------
    # Loading & filtering
    # ------------------------------------------------------------------

    def _load_raw(self) -> None:
        self.df_orders = pd.read_csv(self.DATA_DIR / "client_orders.csv")
        self.df_production = pd.read_csv(self.DATA_DIR / "production_plan.csv")
        self.df_ship_specs = pd.read_csv(self.DATA_DIR / "ship_specs.csv")
        self.df_port = pd.read_csv(self.DATA_DIR / "port_specs.csv")
        self.df_axis = pd.read_csv(self.DATA_DIR / "axis_specs.csv")

    def _filter_and_process(self) -> None:
        self.df_orders["ArrivalDate"] = pd.to_datetime(self.df_orders["ArrivalDate"])
        self.df_orders["Deadline"] = pd.to_datetime(self.df_orders["Deadline"])
        self.df_production["Date"] = pd.to_datetime(self.df_production["Date"])

        # Keep ships whose loading window overlaps [start, end]
        self.df_orders = self.df_orders[
            (self.df_orders["Deadline"] >= self.start)
            & (self.df_orders["ArrivalDate"] <= self.end)
        ].copy()

        # Clamp arrival/deadline to the optimisation window
        self.df_orders["ArrivalDate"] = self.df_orders["ArrivalDate"].clip(lower=self.start)
        self.df_orders["Deadline"] = self.df_orders["Deadline"].clip(upper=self.end)

        self.df_production = self.df_production[
            (self.df_production["Date"] >= self.start)
            & (self.df_production["Date"] <= self.end)
            & (self.df_production["Production"] > 0)
        ].copy()

        self._assign_ship_specs()
        self._build_chain()

    # ------------------------------------------------------------------
    # Ship spec assignment
    # ------------------------------------------------------------------

    def _assign_ship_specs(self) -> None:
        """Assign each ship the smallest SubClass whose DeadWeight >= its demand."""
        sorted_specs = self.df_ship_specs.sort_values("DeadWeight").reset_index(drop=True)

        def _pick_subclass(demand: int) -> str:
            for _, row in sorted_specs.iterrows():
                if demand <= row["DeadWeight"]:
                    return row["SubClass"]
            return sorted_specs.iloc[-1]["SubClass"]

        ship_demand = (
            self.df_orders.groupby("Ship", as_index=False)["Quantity"].sum()
        )
        ship_demand["SubClass"] = ship_demand["Quantity"].apply(_pick_subclass)
        ship_demand = ship_demand.merge(self.df_ship_specs, on="SubClass", how="left")
        self.df_ship_assignments = ship_demand.rename(columns={"Quantity": "Demand"})

    # ------------------------------------------------------------------
    # Fixed chain: Product -> Factory -> Axis -> Berth
    # ------------------------------------------------------------------

    def _build_chain(self) -> None:
        factory_products = build_factory_product_map()
        factories = [f"Factory_{i + 1}" for i in range(NUM_FACTORIES)]
        axes = self.df_axis["Axis"].tolist()
        berths = self.df_port["Berth"].tolist()

        self._product_factory: dict[str, str] = {}
        for factory, products in factory_products.items():
            for p in products:
                self._product_factory[p] = factory

        self._factory_axis = dict(zip(factories, axes))
        self._factory_berth = dict(zip(factories, berths))
        self._axis_berth = dict(zip(axes, berths))

    # ------------------------------------------------------------------
    # Public: working_data_compute
    # ------------------------------------------------------------------

    def working_data_compute(self) -> dict:
        d: dict = {}
        timestamps = list(pd.date_range(self.start, self.end, freq="D"))
        d["timestamps"] = timestamps
        d["timestamp_idx"] = list(range(len(timestamps)))

        ships = self.df_orders["Ship"].unique().tolist()
        d["ships"] = ships
        d["factories"] = sorted(self.df_production["Factory"].unique())
        d["products"] = sorted(
            set(self.df_orders["Product"]) | set(self.df_production["Product"])
        )
        d["quays"] = sorted(self.df_port["Quay"].unique())
        d["berths"] = self.df_port["Berth"].tolist()
        d["berths_anchorage"] = self.df_port["Berth"].tolist() + ["ANCHORAGE"]
        d["axes"] = self.df_axis["Axis"].tolist()

        # --- ship parameters ---
        sa = self.df_ship_assignments.set_index("Ship")
        orders_idx = self.df_orders.set_index("Ship")

        d["ship_demand"] = sa["Demand"].to_dict()
        d["ship_product"] = orders_idx["Product"].to_dict()
        d["ship_load_rate"] = sa["LoadingRate"].to_dict()
        d["ship_length"] = sa["Length"].to_dict()
        d["ship_draft"] = sa["Draft"].to_dict()
        d["ship_dw"] = sa["DeadWeight"].to_dict()

        ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}
        d["ship_laycan"] = {}
        for _, row in self.df_orders.iterrows():
            s = row["Ship"]
            d["ship_laycan"][s] = {
                "ArrivalIdx": ts_to_idx.get(row["ArrivalDate"], 0),
                "DeadlineIdx": ts_to_idx.get(row["Deadline"], len(timestamps) - 1),
            }

        # --- chain parameters ---
        d["product_factory"] = dict(self._product_factory)
        d["factory_axis"] = dict(self._factory_axis)
        d["factory_berth"] = dict(self._factory_berth)
        d["axis_berth"] = dict(self._axis_berth)
        d["berth_quay"] = self.df_port.set_index("Berth")["Quay"].to_dict()

        d["ship_berth"] = {
            s: self._factory_berth[self._product_factory[p]]
            for s, p in d["ship_product"].items()
        }
        d["ship_factory"] = {
            s: self._product_factory[p] for s, p in d["ship_product"].items()
        }
        d["ship_axis"] = {
            s: self._factory_axis[self._product_factory[p]]
            for s, p in d["ship_product"].items()
        }

        # --- infrastructure parameters ---
        port = self.df_port.set_index("Berth")
        port["BerthCapacity"] = port["NumCranes"] * port["CraneCapacity"]
        d["berth_capacity"] = port["BerthCapacity"].to_dict()
        d["berth_length"] = port["MaxLength"].to_dict()
        d["berth_draft"] = port["MaxDraft"].to_dict()
        d["berth_dw"] = port["MaxDeadWeight"].to_dict()
        d["axis_capacity"] = self.df_axis.set_index("Axis")["AxisCapacity"].to_dict()

        # --- production rates keyed by (timestamp_idx, factory, product) ---
        prod_rate: dict[tuple, float] = {}
        for _, row in self.df_production.iterrows():
            t_idx = ts_to_idx.get(row["Date"])
            if t_idx is not None:
                prod_rate[(t_idx, row["Factory"], row["Product"])] = row["Production"]
        d["production_rate"] = prod_rate

        return d
