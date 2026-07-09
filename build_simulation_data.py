from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mro_simulator.mro_agents import FleetAgentPipeline


ROOT = Path(__file__).resolve().parent


def main() -> None:
    predictions_path = ROOT / "reports" / "test_predictions.csv"
    output_path = ROOT / "ui" / "agent_state.json"
    output_path.parent.mkdir(exist_ok=True)
    latest_predictions = pd.read_csv(predictions_path)
    state = FleetAgentPipeline(latest_predictions, slots_per_round=3, horizon=80).run()
    output_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
