from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FleetSimulationResult:
    policy: str
    metrics: dict[str, float]
    decisions: pd.DataFrame


class FleetOrchestratorAgent:
    def __init__(self, risk_threshold: float = 30.0) -> None:
        self.risk_threshold = risk_threshold

    def rank(self, fleet: pd.DataFrame, day: int) -> pd.DataFrame:
        active = fleet.loc[(~fleet["maintained"]) & (~fleet["failed"])].copy()
        active["predicted_remaining_today"] = active["predicted_rul"] - day
        urgency = -active["predicted_remaining_today"]
        active["risk_score"] = (
            urgency
            + 0.25 * active["pred_uncertainty"]
            + 2.0 * (active["predicted_remaining_today"] <= self.risk_threshold).astype(float)
        )
        return active.sort_values(
            ["risk_score", "predicted_remaining_today", "pred_uncertainty"],
            ascending=[False, True, False],
        )


def simulate_policy(
    latest: pd.DataFrame,
    policy: str,
    slots_per_day: int = 3,
    horizon: int = 160,
    seed: int = 42,
) -> FleetSimulationResult:
    rng = np.random.default_rng(seed)
    fleet = latest[["unit", "cycle", "true_rul", "predicted_rul", "pred_uncertainty"]].copy()
    fleet["maintained"] = False
    fleet["failed"] = False
    fleet["maintenance_day"] = np.nan
    fleet["failure_day"] = np.nan
    decisions: list[dict[str, float | int | str]] = []
    orchestrator = FleetOrchestratorAgent()

    for day in range(horizon + 1):
        due_to_fail = (~fleet["maintained"]) & (~fleet["failed"]) & (fleet["true_rul"] <= day)
        fleet.loc[due_to_fail, "failed"] = True
        fleet.loc[due_to_fail, "failure_day"] = day

        active = fleet.loc[(~fleet["maintained"]) & (~fleet["failed"])].copy()
        if active.empty:
            break

        if policy == "orchestrator":
            ranked = orchestrator.rank(fleet, day)
        elif policy == "shortest_predicted_rul":
            active["predicted_remaining_today"] = active["predicted_rul"] - day
            ranked = active.sort_values(["predicted_remaining_today", "pred_uncertainty"], ascending=[True, False])
        elif policy == "oldest_cycle":
            ranked = active.sort_values(["cycle", "predicted_rul"], ascending=[False, True])
        elif policy == "random":
            ranked = active.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000)))
        else:
            raise ValueError(f"Unknown policy: {policy}")

        selected = ranked.head(slots_per_day)
        for _, row in selected.iterrows():
            unit = int(row["unit"])
            fleet.loc[fleet["unit"] == unit, "maintained"] = True
            fleet.loc[fleet["unit"] == unit, "maintenance_day"] = day
            decisions.append(
                {
                    "policy": policy,
                    "day": day,
                    "unit": unit,
                    "cycle": int(row["cycle"]),
                    "true_rul_at_start": float(row["true_rul"]),
                    "predicted_rul_at_start": float(row["predicted_rul"]),
                    "pred_uncertainty": float(row["pred_uncertainty"]),
                }
            )

    unhandled = (~fleet["maintained"]) & (~fleet["failed"])
    fleet.loc[unhandled, "failure_day"] = horizon + 1
    fleet.loc[unhandled, "failed"] = True

    critical = fleet["true_rul"] <= 30
    critical_success = critical & fleet["maintained"] & (fleet["maintenance_day"] < fleet["true_rul"])
    metrics = {
        "failures_before_maintenance": float((fleet["failed"] & ~fleet["maintained"]).sum()),
        "maintained_before_failure": float((fleet["maintained"] & (fleet["maintenance_day"] < fleet["true_rul"])).sum()),
        "critical_engines": float(critical.sum()),
        "critical_coverage_rate": float(critical_success.sum() / max(1, critical.sum())),
        "mean_maintenance_day": float(fleet.loc[fleet["maintained"], "maintenance_day"].mean()),
        "late_maintenance_count": float((fleet["maintained"] & (fleet["maintenance_day"] >= fleet["true_rul"])).sum()),
    }
    return FleetSimulationResult(policy=policy, metrics=metrics, decisions=pd.DataFrame(decisions))
