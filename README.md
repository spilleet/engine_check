# NASA C-MAPSS FD001 RUL + Fleet Maintenance Agent

This project trains an RUL predictor on NASA C-MAPSS FD001 and runs a fleet maintenance simulation with limited maintenance slots.

## Quick Start

```bash
cd /Users/idong-ug/Documents/Codex/2026-07-08/nasa-c-mapss-fd001-ai-rul/outputs/cmapss_fd001_agent
python3 run_experiment.py
python3 build_agent_ui.py
python3 realtime_server.py --port 8765
```

Open `http://localhost:8765` to run the agent pipeline UI.

Default data paths:

- `/Users/idong-ug/Downloads/pdm_agent/data/train_FD001.txt`
- `/Users/idong-ug/Downloads/pdm_agent/data/test_FD001.txt`
- `/Users/idong-ug/Downloads/pdm_agent/data/RUL_FD001.txt`

Override them if needed:

```bash
python3 run_experiment.py \
  --train /path/to/train_FD001.txt \
  --test /path/to/test_FD001.txt \
  --rul /path/to/RUL_FD001.txt \
  --slots 3 \
  --horizon 160
```

## What It Does

1. Loads FD001 train/test/RUL files.
2. Builds capped RUL labels for training (`cap=125` by default).
3. Trains a Random Forest RUL model with group-aware validation by engine ID.
4. Evaluates test-set RUL at each engine's latest observed cycle.
5. Builds a round-based agent pipeline:
   - `TelemetryStreamAgent`: streams current engine telemetry frames.
   - `RULPredictorAgent`: predicts RUL from the trained model.
   - `CrisisDetectionAgent`: flags inspection and danger states.
   - `SituationQueryAgent`: summarizes fleet status and top risks.
   - `MaintenanceActionAgent`: assigns limited maintenance slots.
6. Simulates fleet maintenance under limited slots and compares policies:
   - `orchestrator`: predicted RUL priority with uncertainty/risk tie-breakers.
   - `shortest_predicted_rul`: pure predicted RUL priority.
   - `oldest_cycle`: prioritize engines with the longest observed cycles.
   - `random`: random baseline.

## Realtime Operator Workflow

The realtime UI supports a maintenance operator workflow:

- `Critical Queue`: prioritized engines needing immediate maintenance or inspection.
- `Engine Detail`: RUL, uncertainty, status, risk score, and explanation.
- `Approval Panel`: approve, defer, or reject an agent recommendation with a reason.
- `Work Order Log`: generated work orders and audit trail for human decisions.
- `Agent Reasoning Trace`: live stream of crisis detection, situation query, action, and human approval events.

## Outputs

The run creates:

- `reports/metrics.json`
- `reports/fleet_policy_comparison.csv`
- `reports/orchestrator_decisions.csv`
- `reports/test_predictions.csv`
- `reports/top_risk_engines.csv`
- `reports/dashboard.html`
- `reports/policy_comparison.png`
- `reports/rul_prediction_diagnostics.png`
- `reports/maintenance_timeline.png`
- `ui/index.html`
- `ui/app.js`
- `realtime_server.py`
- `ui/agent_state.json` from the legacy precomputed mode
- `artifacts/rul_model.joblib`
