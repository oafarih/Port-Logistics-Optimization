import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import gurobipy as gp
from gurobipy import GRB


class PortModel:
    """Build and solve the port logistics optimisation model.

    The fixed chain Factory_i -> Axis_i -> Berth_i allows significant
    simplification over a fully-flexible assignment formulation.

    Parameters
    ----------
    data : dict
        working data dictionary produced by ``DataPipeline.working_data_compute()``.
    w_load : float
        Weight for maximising loaded quantity (default 1.0, primary objective).
    w_stay : float
        Penalty weight for ship stay duration (default 100, secondary objective).
    w_move : float
        Penalty weight for ship movements (default 10, tertiary objective).
    """

    def __init__(
        self,
        data: Dict,
        w_load: float = 1.0,
        w_stay: float = 100.0,
        w_move: float = 10.0,
    ) -> None:
        self.data = data
        self.w_load = w_load
        self.w_move = w_move
        self.w_stay = w_stay

        for key, value in data.items():
            setattr(self, key, value)

        self.model = gp.Model("PortLogistics")
        self._add_variables()
        self._add_constraints()
        self._set_objective()
        self.model.update()

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------

    def _add_variables(self) -> None:
        T = self.timestamp_idx
        S = self.ships

        self.at_berth = self.model.addVars(
            T, S, vtype=GRB.BINARY, name="at_berth"
        )
        self.at_anchorage = self.model.addVars(
            T, S, vtype=GRB.BINARY, name="at_anchorage"
        )
        self.load = self.model.addVars(
            T, S, vtype=GRB.CONTINUOUS, lb=0, name="load"
        )
        self.stock = self.model.addVars(
            T, self.factories, self.products,
            vtype=GRB.CONTINUOUS, lb=0, name="stock"
        )
        self.etb = self.model.addVars(
            T, S, vtype=GRB.BINARY, name="etb"
        )
        self.ets = self.model.addVars(
            T, S, vtype=GRB.BINARY, name="ets"
        )
        self.movement = self.model.addVars(
            T, S, vtype=GRB.BINARY, name="movement"
        )

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def _add_constraints(self) -> None:
        self._constr_demand_limit()
        self._constr_ship_position()
        self._constr_one_ship_per_berth()
        self._constr_quay_length()
        self._constr_draft_limit()
        self._constr_loading()
        self._constr_stock_balance()
        self._constr_etb_ets()
        self._constr_movements()

    # 1. A ship cannot load more than its demanded quantity
    def _constr_demand_limit(self) -> None:
        for s in self.ships:
            self.model.addConstr(
                gp.quicksum(self.load[t, s] for t in self.timestamp_idx)
                <= self.ship_demand[s],
                name=f"demand_limit_{s}",
            )

    # 2 & 10.  Ship is at exactly one place during its stay, nowhere outside
    def _constr_ship_position(self) -> None:
        for s in self.ships:
            arr = self.ship_laycan[s]["ArrivalIdx"]
            ddl = self.ship_laycan[s]["DeadlineIdx"]

            for t in self.timestamp_idx:
                # At most one location at any time
                self.model.addConstr(
                    self.at_berth[t, s] + self.at_anchorage[t, s] <= 1,
                    name=f"one_place_{t}_{s}",
                )

            # Between ETB and ETS the ship must be present (berth or anchorage).
            # We enforce: for every t in [arr, ddl], presence >= etb_cumulative - ets_done
            # This is linearised via the ETB/ETS link constraints below.
            # Here we simply forbid the ship from appearing outside [arr, ddl].
            for t in self.timestamp_idx:
                if t < arr or t > ddl:
                    self.model.addConstr(
                        self.at_berth[t, s] == 0,
                        name=f"no_berth_outside_{t}_{s}",
                    )
                    self.model.addConstr(
                        self.at_anchorage[t, s] == 0,
                        name=f"no_anch_outside_{t}_{s}",
                    )

    # 3.  One ship per berth per day
    def _constr_one_ship_per_berth(self) -> None:
        berth_ships: dict[str, list[str]] = {b: [] for b in self.berths}
        for s in self.ships:
            berth_ships[self.ship_berth[s]].append(s)

        for b in self.berths:
            if not berth_ships[b]:
                continue
            for t in self.timestamp_idx:
                self.model.addConstr(
                    gp.quicksum(
                        self.at_berth[t, s] for s in berth_ships[b]
                    ) <= 1,
                    name=f"one_ship_berth_{t}_{b}",
                )

    # 7.  Quay length: sum of berthed ship lengths <= total quay length
    def _constr_quay_length(self) -> None:
        for q in self.quays:
            q_berths = [b for b in self.berths if self.berth_quay[b] == q]
            total_quay_length = sum(self.berth_length[b] for b in q_berths)
            q_ships = [s for s in self.ships if self.ship_berth[s] in q_berths]

            if not q_ships:
                continue
            for t in self.timestamp_idx:
                self.model.addConstr(
                    gp.quicksum(
                        self.ship_length[s] * self.at_berth[t, s]
                        for s in q_ships
                    ) <= total_quay_length,
                    name=f"quay_len_{t}_{q}",
                )

    # 8.  Draft limit: ship cannot berth if its draft exceeds the berth's
    def _constr_draft_limit(self) -> None:
        for s in self.ships:
            b = self.ship_berth[s]
            if self.ship_draft[s] > self.berth_draft[b]:
                for t in self.timestamp_idx:
                    self.model.addConstr(
                        self.at_berth[t, s] == 0,
                        name=f"draft_{t}_{s}",
                    )

    # Loading: can only load when at berth; rate capped by ship, berth, axis
    def _constr_loading(self) -> None:
        for s in self.ships:
            b = self.ship_berth[s]
            a = self.ship_axis[s]
            effective_rate = min(
                self.ship_load_rate[s],
                self.berth_capacity[b],
                self.axis_capacity[a],
            )
            for t in self.timestamp_idx:
                self.model.addConstr(
                    self.load[t, s] <= effective_rate * self.at_berth[t, s],
                    name=f"load_rate_{t}_{s}",
                )

    # Stock balance: stock_t = stock_{t-1} + production - outflow
    def _constr_stock_balance(self) -> None:
        factory_ships: dict[tuple[str, str], list[str]] = {}
        for s in self.ships:
            key = (self.ship_factory[s], self.ship_product[s])
            factory_ships.setdefault(key, []).append(s)

        for t in self.timestamp_idx:
            for f in self.factories:
                for p in self.products:
                    production = self.production_rate.get((t, f, p), 0)
                    outflow = gp.quicksum(
                        self.load[t, s]
                        for s in factory_ships.get((f, p), [])
                    )
                    if t == 0:
                        self.model.addConstr(
                            self.stock[t, f, p] == production - outflow,
                            name=f"stock_init_{t}_{f}_{p}",
                        )
                    else:
                        self.model.addConstr(
                            self.stock[t, f, p]
                            == self.stock[t - 1, f, p] + production - outflow,
                            name=f"stock_bal_{t}_{f}_{p}",
                        )

    # 9.  ETB / ETS: each fires exactly once; ETB before ETS; within laycan
    def _constr_etb_ets(self) -> None:
        T = self.timestamp_idx
        for s in self.ships:
            arr = self.ship_laycan[s]["ArrivalIdx"]
            ddl = self.ship_laycan[s]["DeadlineIdx"]

            # ETB and ETS each fire exactly once
            self.model.addConstr(
                gp.quicksum(self.etb[t, s] for t in T) == 1,
                name=f"etb_once_{s}",
            )
            self.model.addConstr(
                gp.quicksum(self.ets[t, s] for t in T) == 1,
                name=f"ets_once_{s}",
            )

            # ETB cannot fire outside [arr, ddl]
            for t in T:
                if t < arr or t > ddl:
                    self.model.addConstr(self.etb[t, s] == 0, name=f"etb_window_{t}_{s}")
                    self.model.addConstr(self.ets[t, s] == 0, name=f"ets_window_{t}_{s}")

            # Weighted time: ETS day >= ETB day
            t_etb = gp.quicksum(t * self.etb[t, s] for t in T)
            t_ets = gp.quicksum(t * self.ets[t, s] for t in T)
            self.model.addConstr(t_ets >= t_etb, name=f"ets_after_etb_{s}")

            # Ship must be present (berth or anchorage) between ETB and ETS.
            # Modelled via cumulative indicators:
            #   cum_etb[t] = sum_{tau<=t} etb[tau]   (1 after berthing day)
            #   cum_ets[t] = sum_{tau<=t} ets[tau]    (1 after sailing day)
            # present[t] = cum_etb[t] - cum_ets[t-1]  ∈ {0,1}
            # We require: at_berth + at_anchorage >= present  (must be somewhere)
            #             at_berth + at_anchorage <= cum_etb  (not before berthing)
            for t in T:
                cum_etb = gp.quicksum(self.etb[tau, s] for tau in T if tau <= t)
                cum_ets_prev = gp.quicksum(
                    self.ets[tau, s] for tau in T if tau <= t - 1
                ) if t > 0 else 0

                present = cum_etb - cum_ets_prev if t > 0 else cum_etb
                self.model.addConstr(
                    self.at_berth[t, s] + self.at_anchorage[t, s] >= present,
                    name=f"present_lb_{t}_{s}",
                )
                self.model.addConstr(
                    self.at_berth[t, s] + self.at_anchorage[t, s] <= cum_etb,
                    name=f"present_ub_etb_{t}_{s}",
                )
                # Cannot be present after sailing
                cum_ets = gp.quicksum(self.ets[tau, s] for tau in T if tau <= t)
                self.model.addConstr(
                    self.at_berth[t, s] + self.at_anchorage[t, s]
                    <= 1 - cum_ets + gp.quicksum(
                        self.ets[tau, s] for tau in T if tau == t
                    ),
                    name=f"present_ub_ets_{t}_{s}",
                )

    # Movement detection: position change between consecutive days
    def _constr_movements(self) -> None:
        T = self.timestamp_idx
        for s in self.ships:
            for t in T:
                if t == 0:
                    continue
                self.model.addConstr(
                    self.movement[t, s]
                    >= self.at_berth[t, s] - self.at_berth[t - 1, s],
                    name=f"move_to_berth_{t}_{s}",
                )
                self.model.addConstr(
                    self.movement[t, s]
                    >= self.at_berth[t - 1, s] - self.at_berth[t, s],
                    name=f"move_from_berth_{t}_{s}",
                )

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def _set_objective(self) -> None:
        T = self.timestamp_idx
        S = self.ships

        total_loaded = gp.quicksum(self.load[t, s] for t in T for s in S)

        total_movements = gp.quicksum(
            self.movement[t, s] for t in T for s in S
        )

        total_stay = gp.quicksum(
            gp.quicksum(t * self.ets[t, s] for t in T)
            - gp.quicksum(t * self.etb[t, s] for t in T)
            for s in S
        )

        self.model.setObjective(
            self.w_load * total_loaded
            - self.w_move * total_movements
            - self.w_stay * total_stay,
            GRB.MAXIMIZE,
        )

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self, time_limit: float = 300, mip_gap: float = 0.05, **kwargs) -> None:
        self.model.Params.TimeLimit = time_limit
        self.model.Params.MIPGap = mip_gap
        for k, v in kwargs.items():
            setattr(self.model.Params, k, v)
        self.model.optimize()

    def print_summary(self) -> None:
        if self.model.SolCount == 0:
            print("No feasible solution found.")
            return

        total_demand = sum(self.ship_demand[s] for s in self.ships)
        total_loaded = sum(
            self.load[t, s].X
            for t in self.timestamp_idx
            for s in self.ships
        )
        total_moves = sum(
            self.movement[t, s].X
            for t in self.timestamp_idx
            for s in self.ships
        )
        avg_stay = 0.0
        for s in self.ships:
            etb_day = sum(t * self.etb[t, s].X for t in self.timestamp_idx)
            ets_day = sum(t * self.ets[t, s].X for t in self.timestamp_idx)
            avg_stay += ets_day - etb_day
        avg_stay /= max(len(self.ships), 1)

        print(f"Ships:           {len(self.ships)}")
        print(f"Total demand:    {total_demand:,.0f}")
        print(f"Total loaded:    {total_loaded:,.0f}  ({total_loaded / total_demand * 100:.1f}%)")
        print(f"Total movements: {total_moves:.0f}")
        print(f"Avg stay (days): {avg_stay:.1f}")
        print(f"Objective:       {self.model.ObjVal:,.0f}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "data" / "results"

    def save_results(self, output_dir: Optional[Path] = None) -> None:
        """Save solution to *output_dir* (defaults to src/data/results/).

        Creates one CSV per decision-variable group and one objective.json.
        """
        if self.model.SolCount == 0:
            print("No solution to save.")
            return

        out = Path(output_dir) if output_dir else self.DEFAULT_RESULTS_DIR
        out.mkdir(parents=True, exist_ok=True)

        T = self.timestamp_idx
        S = self.ships
        dates = [ts.strftime("%Y-%m-%d") for ts in self.timestamps]

        # --- Variable CSVs ---
        self._save_ts_ship_var(out / "at_berth.csv", self.at_berth, T, S, dates)
        self._save_ts_ship_var(out / "at_anchorage.csv", self.at_anchorage, T, S, dates)
        self._save_ts_ship_var(out / "load.csv", self.load, T, S, dates)
        self._save_ts_ship_var(out / "etb.csv", self.etb, T, S, dates)
        self._save_ts_ship_var(out / "ets.csv", self.ets, T, S, dates)
        self._save_ts_ship_var(out / "movement.csv", self.movement, T, S, dates)

        stock_rows = []
        for t in T:
            for f in self.factories:
                for p in self.products:
                    val = self.stock[t, f, p].X
                    if val != 0:
                        stock_rows.append(
                            {"Date": dates[t], "Factory": f, "Product": p, "Value": round(val, 2)}
                        )
        pd.DataFrame(stock_rows).to_csv(out / "stock.csv", index=False)

        n_files = 7
        print(f"Saved {n_files} variable CSVs to {out}")

        # --- objective.json ---
        total_loaded = sum(self.load[t, s].X for t in T for s in S)
        total_movements = sum(self.movement[t, s].X for t in T for s in S)
        total_stay = sum(
            sum(t * self.ets[t, s].X for t in T)
            - sum(t * self.etb[t, s].X for t in T)
            for s in S
        )
        total_demand = sum(self.ship_demand[s] for s in S)

        per_ship: dict = {}
        for s in S:
            loaded = sum(self.load[t, s].X for t in T)
            etb_day = sum(t * self.etb[t, s].X for t in T)
            ets_day = sum(t * self.ets[t, s].X for t in T)
            per_ship[s] = {
                "demand": self.ship_demand[s],
                "loaded": round(loaded, 2),
                "pct_loaded": round(loaded / self.ship_demand[s] * 100, 1)
                if self.ship_demand[s] > 0
                else 0.0,
                "etb_day_idx": int(round(etb_day)),
                "ets_day_idx": int(round(ets_day)),
                "stay_days": int(round(ets_day - etb_day)),
                "berth": self.ship_berth[s],
                "product": self.ship_product[s],
            }

        objective_data = {
            "solver": {
                "status": self.model.Status,
                "objective_value": self.model.ObjVal,
                "mip_gap": self.model.MIPGap,
                "runtime_seconds": round(self.model.Runtime, 2),
                "num_variables": self.model.NumVars,
                "num_constraints": self.model.NumConstrs,
            },
            "summary": {
                "num_ships": len(S),
                "total_demand": total_demand,
                "total_loaded": round(total_loaded, 2),
                "pct_demand_loaded": round(total_loaded / total_demand * 100, 1)
                if total_demand > 0
                else 0.0,
                "total_movements": int(round(total_movements)),
                "total_stay_days": round(total_stay, 1),
                "avg_stay_days": round(total_stay / max(len(S), 1), 1),
            },
            "weights": {
                "w_load": self.w_load,
                "w_move": self.w_move,
                "w_stay": self.w_stay,
            },
            "per_ship": per_ship,
        }

        obj_path = out / "objective.json"
        with open(obj_path, "w") as f:
            json.dump(objective_data, f, indent=2)
        print(f"Saved objective.json to {out}")

    @staticmethod
    def _save_ts_ship_var(path: Path, var_dict, T, S, dates) -> None:
        rows = []
        for t in T:
            for s in S:
                val = var_dict[t, s].X
                if val != 0:
                    rows.append({"Date": dates[t], "Ship": s, "Value": round(val, 2)})
        pd.DataFrame(rows).to_csv(path, index=False)
