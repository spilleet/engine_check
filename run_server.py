from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from mro_simulator.mro_agents import (
    CrisisDetectionAgent,
    MaintenanceActionAgent,
    SituationQueryAgent,
    MaintenanceDiagnosticianAgent,
    ActionRecommendationAgent,
    MaintenanceReportAgent,
)


ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "ui"


class RealtimeFleetSimulator:
    def __init__(self, predictions: pd.DataFrame, importances: dict[str, float] | None = None) -> None:
        self.frame = predictions.copy()
        self.frame["stream_cycle"] = self.frame["cycle"]
        self.frame["rul"] = self.frame["predicted_rul"]
        self.frame["true_remaining"] = self.frame["true_rul"]
        self.frame["maintained"] = False
        self.frame["under_maintenance"] = False
        self.frame["maintenance_remaining_ticks"] = 0
        self.frame["human_modifier"] = 1.0
        self.frame["pending_supervisor"] = False
        self.frame["risk_score"] = 0.0
        self.frame["status"] = "healthy"
        self.detector = CrisisDetectionAgent()
        self.query = SituationQueryAgent()
        self.action = MaintenanceActionAgent(slots_per_round=3)
        self.diagnostician = MaintenanceDiagnosticianAgent(feature_importances=importances)
        self.recommender = ActionRecommendationAgent()
        self.reporter = MaintenanceReportAgent()
        self.tick = 0
        self.pointer = 0
        self.agent_cost = 0
        self.baseline_cost = 0
        self.protected_failures = 0
        self.missed_failures = 0
        self.maintenance_counts = {int(unit): 0 for unit in self.frame["unit"].unique()}
        self.events: list[dict] = [
            {"time": "00:00", "agent": "system", "message": "실시간 스트림 연결. 엔진 telemetry 수신을 시작합니다."},
            {"time": "00:00", "agent": "objective", "message": f"목표: RUL 급락 감지, 상황 조회, 슬롯 {slots}개 내 정비 조치."},
        ]
        self.work_orders: list[dict] = []
        self.deferred_units: set[int] = set()
        self.last_status = {int(row.unit): "healthy" for row in self.frame.itertuples()}
        self.frame = self.detector.annotate(self.frame)

    def _calculate_restored_rul(self, unit: int, anomalies_count: int) -> float:
        count = self.maintenance_counts.get(unit, 0)
        self.maintenance_counts[unit] = count + 1
        
        base_limit = 125.0
        # 누적 정비 시마다 피로 축적으로 최대 수명 상한 10사이클씩 페널티 (최소 60사이클)
        wear_out_penalty = count * 10.0
        max_possible_rul = max(60.0, base_limit - wear_out_penalty)
        
        # 정비 원인 센서 개수(조치 수준)에 따른 성능 복원율 차등화
        if anomalies_count >= 3:
            restore_rate = 0.96  # Major
        elif anomalies_count == 2:
            restore_rate = 0.80  # Medium
        else:
            restore_rate = 0.65  # Minor
            
        restored = max_possible_rul * restore_rate
        return float(round(restored, 1))

    def advance(self, telemetry_events: int) -> dict:
        # 1. 정비 진행 지연(ticks) 차감 처리
        now_str = self._clock()
        for row in self.frame.itertuples():
            if bool(row.under_maintenance):
                idx = self.frame["unit"] == row.unit
                ticks = int(row.maintenance_remaining_ticks) - 1
                if ticks <= 0:
                    # 정비 완료 처리
                    self.frame.loc[idx, "under_maintenance"] = False
                    self.frame.loc[idx, "maintenance_remaining_ticks"] = 0
                    self.frame.loc[idx, "maintained"] = True
                    self.frame.loc[idx, "human_modifier"] = 1.0
                    self.last_status[int(row.unit)] = "maintained"
                    
                    diag = self.diagnostician.diagnose(self.frame, int(row.unit))
                    rec = self.recommender.recommend(diag)
                    restored_rul = self._calculate_restored_rul(int(row.unit), len(rec["checklist"]))
                    self.frame.loc[idx, "rul"] = restored_rul
                    self.agent_cost += 8000
                    
                    if float(row.true_remaining) <= float(row.rul) + 8:
                        self.protected_failures += 1
                        
                    self.events.append({
                        "time": now_str,
                        "agent": "action_agent",
                        "message": f"엔진 #{int(row.unit)} 정비 완료 및 가동 상태 복귀. 복원 RUL: {restored_rul}."
                    })
                else:
                    self.frame.loc[idx, "maintenance_remaining_ticks"] = ticks

        # 2. 텔레메트리 경과 차감
        touched_units: list[int] = []
        units = self.frame["unit"].astype(int).tolist()
        for _ in range(telemetry_events):
            unit = units[self.pointer % len(units)]
            self.pointer += 1
            touched_units.append(unit)
            idx = self.frame["unit"] == unit
            # 정비 완료된 엔진이거나, 현재 정비 진행 중인 엔진은 수명이 깎이지 않도록 스킵
            if bool(self.frame.loc[idx, "maintained"].iloc[0]) or bool(self.frame.loc[idx, "under_maintenance"].iloc[0]):
                continue
            self.frame.loc[idx, "stream_cycle"] += 1
            self.frame.loc[idx, "rul"] = (self.frame.loc[idx, "rul"] - 1.0).clip(lower=0)
            self.frame.loc[idx, "true_remaining"] = (self.frame.loc[idx, "true_remaining"] - 1.0).clip(lower=0)

        self.tick += 1
        self.frame = self.detector.annotate(self.frame)
        now = self._clock()
        changed = self._status_changes(now)
        actions = self._maybe_act(now)
        self._update_costs()
        summary = self.query.summarize(self.frame)
        top_risks = self.query.top_risks(self.frame)

        if self.tick == 1 or changed or actions:
            self.events.append(
                {
                    "time": now,
                    "agent": "situation_query",
                    "message": f"상황조회: 위험 {summary['danger']}대, 점검요망 {summary['inspect']}대, 최저 예측 RUL {summary['lowest_rul']:.1f}.",
                }
            )

        return self._payload(now, touched_units, summary, top_risks, actions)

    def snapshot(self) -> dict:
        summary = self.query.summarize(self.frame)
        top_risks = self.query.top_risks(self.frame)
        return self._payload("00:00", [], summary, top_risks, [])

    def manual_decision(self, unit: int, decision: str, reason: str = "") -> dict:
        now = self._clock()
        idx = self.frame["unit"] == unit
        if not idx.any():
            raise ValueError(f"Unknown unit: {unit}")
        row = self.frame.loc[idx].iloc[0]
        reason_text = reason.strip() or "사유 미입력"

        if decision == "approve":
            # 1. 1차 실무자 상신 (상급자 결재 대기 상태)
            self.frame.loc[idx, "pending_supervisor"] = True
            
            diag = self.diagnostician.diagnose(self.frame, unit)
            rec = self.recommender.recommend(diag)
            report_md = self.reporter.generate_markdown(
                unit=unit,
                cycle=int(row["stream_cycle"]),
                predicted_rul=float(row["rul"]),
                uncertainty=float(row["pred_uncertainty"]),
                recommendations=rec,
                reason=reason_text
            )
            order = {
                "id": f"WO-REQ-{int(time.time())}-{unit:03d}",
                "time": now,
                "unit": unit,
                "decision": "pending",
                "status": "pending_supervisor",
                "reason": reason_text,
                "predicted_rul": float(row["rul"]),
                "uncertainty": float(row["pred_uncertainty"]),
                "report_md": report_md
            }
            self.work_orders.insert(0, order)
            self.events.append({"time": now, "agent": "human_approval", "message": f"실무자 상신: 엔진 #{unit} 작업지서 상신. 상급자 결재 대기 중. 사유: {reason_text}"})
            
        elif decision == "approve_final":
            # 2. 2차 상급자 최종 승인 -> 정비 지연(under_maintenance) 타이머 개시
            self.frame.loc[idx, "pending_supervisor"] = False
            self.frame.loc[idx, "under_maintenance"] = True
            self.frame.loc[idx, "maintenance_remaining_ticks"] = 3
            
            # 기존 pending 중인 오더를 찾아 결재 완료 상태로 전환
            found = False
            for order in self.work_orders:
                if order["unit"] == unit and order["decision"] == "pending":
                    order["id"] = f"WO-{int(time.time())}-{unit:03d}"
                    order["decision"] = "approved"
                    order["status"] = "under_maintenance"
                    order["time"] = now
                    # 파일 아카이빙 내보내기 수행
                    try:
                        out_dir = ROOT / "reports" / "submitted_orders"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        file_path = out_dir / f"{order['id']}.md"
                        file_path.write_text(order["report_md"], encoding="utf-8")
                    except Exception as exc:
                        print(f"Warning: Could not save supervisor report file: {exc}")
                    found = True
                    break
            
            if not found:
                # 대기열에 없을 경우 즉시 직권 정비 처리
                diag = self.diagnostician.diagnose(self.frame, unit)
                rec = self.recommender.recommend(diag)
                report_md = self.reporter.generate_markdown(
                    unit=unit,
                    cycle=int(row["stream_cycle"]),
                    predicted_rul=float(row["rul"]),
                    uncertainty=float(row["pred_uncertainty"]),
                    recommendations=rec,
                    reason="상급자 직권 정비 승인"
                )
                order = {
                    "id": f"WO-{int(time.time())}-{unit:03d}",
                    "time": now,
                    "unit": unit,
                    "decision": "approved",
                    "status": "under_maintenance",
                    "reason": "상급자 직권 승인",
                    "predicted_rul": float(row["rul"]),
                    "uncertainty": float(row["pred_uncertainty"]),
                    "report_md": report_md
                }
                self.work_orders.insert(0, order)
                
            self.events.append({"time": now, "agent": "human_approval", "message": f"상급자 최종 승인: 엔진 #{unit} 정비 작업 오더 승인 완료 및 입고 개시."})
            
        elif decision == "defer":
            # 3. 보류 처리 -> human_modifier 가중치를 0.5로 깎아 위기 지수 경보 완화
            self.frame.loc[idx, "human_modifier"] = 0.5
            self.frame.loc[idx, "pending_supervisor"] = False
            self.deferred_units.add(unit)
            
            for order in self.work_orders:
                if order["unit"] == unit and order["decision"] == "pending":
                    order["decision"] = "deferred"
                    order["status"] = "monitoring"
                    break
            else:
                order = {
                    "id": f"DF-{int(time.time())}-{unit:03d}",
                    "time": now,
                    "unit": unit,
                    "decision": "deferred",
                    "status": "monitoring",
                    "reason": reason_text,
                    "predicted_rul": float(row["rul"]),
                    "uncertainty": float(row["pred_uncertainty"]),
                }
                self.work_orders.insert(0, order)
                
            self.events.append({"time": now, "agent": "human_approval", "message": f"실무자 보류: 엔진 #{unit} 모니터링 연기. 리스크 50% 감쇄 적용. 사유: {reason_text}"})
            
        elif decision == "reject":
            # 4. 반려 처리
            self.frame.loc[idx, "pending_supervisor"] = False
            for order in self.work_orders:
                if order["unit"] == unit and order["decision"] == "pending":
                    order["decision"] = "rejected"
                    order["status"] = "closed"
                    break
            else:
                order = {
                    "id": f"RJ-{int(time.time())}-{unit:03d}",
                    "time": now,
                    "unit": unit,
                    "decision": "rejected",
                    "status": "closed",
                    "reason": reason_text,
                    "predicted_rul": float(row["rul"]),
                    "uncertainty": float(row["pred_uncertainty"]),
                }
                self.work_orders.insert(0, order)
            self.events.append({"time": now, "agent": "human_approval", "message": f"결재 반려: 엔진 #{unit} 결재 요청을 반려 및 종결함. 사유: {reason_text}"})
            
        else:
            raise ValueError(f"Unknown decision: {decision}")

        self.frame = self.detector.annotate(self.frame) # 피드백 가중치 조율 적용을 위해 재연산
        summary = self.query.summarize(self.frame)
        top_risks = self.query.top_risks(self.frame)
        return self._payload(now, [unit], summary, top_risks, [])

    def _payload(
        self,
        now: str,
        touched_units: list[int],
        summary: dict[str, float],
        top_risks: list[dict],
        actions: list[int],
    ) -> dict:
        visible = self.frame[["unit", "stream_cycle", "rul", "pred_uncertainty", "status", "maintained", "pending_supervisor", "under_maintenance"]].copy()
        return {
            "stream_time": now,
            "tick": self.tick,
            "touched_units": touched_units,
            "summary": summary,
            "engines": visible.to_dict(orient="records"),
            "top_risks": top_risks,
            "actions": actions,
            "cost": {
                "agent": self.agent_cost,
                "baseline": self.baseline_cost,
                "protected_failures": self.protected_failures,
                "missed_failures": self.missed_failures,
            },
            "log": self.events[-12:],
            "work_orders": self.work_orders[:12],
        }

    def _status_changes(self, now: str) -> list[int]:
        changed: list[int] = []
        for row in self.frame.itertuples():
            unit = int(row.unit)
            previous = self.last_status.get(unit, "healthy")
            current = str(row.status)
            if current != previous and current in {"inspect", "danger"}:
                changed.append(unit)
                self.events.append(
                    {
                        "time": now,
                        "agent": "crisis_detector",
                        "message": f"엔진 #{unit} 상태 전이: {previous} → {current}. 예측 RUL {row.rul:.1f}.",
                    }
                )
            self.last_status[unit] = current
        return changed

    def _maybe_act(self, now: str) -> list[int]:
        if self.tick % 5 != 0:
            return []
        # AI 에이전트의 정비 권고 엔진 탐색
        actions = self.action.choose_actions(self.frame)
        units = [int(unit) for unit in actions["unit"].tolist()]
        if units:
            self.events.append(
                {
                    "time": now,
                    "agent": "action_agent",
                    "message": f"[AI 정비 권고] 고장 위험 감지 엔진 #{units}. 실무자 승인 및 상급자 결재가 권장됩니다.",
                }
            )
        return []  # 실제 정비는 집행하지 않음 (인간 최종 승인이 필수)

    def _update_costs(self) -> None:
        active = ~self.frame["maintained"]
        missed_now = int(((self.frame["true_remaining"] <= 0) & active).sum())
        self.missed_failures += missed_now
        self.baseline_cost += int(((self.frame["rul"] < 30) & active).sum()) * 8000 + missed_now * 50000

    def _clock(self) -> str:
        seconds = self.tick
        return f"{seconds // 60:02d}:{seconds % 60:02d}"


class RealtimeRequestHandler(BaseHTTPRequestHandler):
    predictions: pd.DataFrame
    simulator: RealtimeFleetSimulator
    lock: threading.Lock

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/events":
            self._serve_events(parsed.query)
            return
        if parsed.path == "/api/state":
            with self.lock:
                self._send_json({"fleet_size": int(self.predictions["unit"].nunique()), "initial": self.simulator.snapshot()})
            return
        if parsed.path == "/api/diagnose":
            params = parse_qs(parsed.query)
            unit_str = params.get("unit", [""])[0]
            if not unit_str:
                self.send_error(400, "Missing unit parameter")
                return
            try:
                unit = int(unit_str)
                with self.lock:
                    diag = self.simulator.diagnostician.diagnose(self.simulator.frame, unit)
                    rec = self.simulator.recommender.recommend(diag)
                self._send_json({"diagnose": diag, "recommend": rec})
            except Exception as exc:
                self.send_error(400, str(exc))
            return
        self._serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/action":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            unit = int(payload["unit"])
            decision = str(payload["decision"])
            reason = str(payload.get("reason", ""))
            with self.lock:
                result = self.simulator.manual_decision(unit, decision, reason)
            self._send_json(result)
        except Exception as exc:
            self.send_error(400, str(exc))

    def _serve_events(self, query: str) -> None:
        params = parse_qs(query)
        speed = params.get("speed", ["5"])[0]
        batch = {"1": 1, "5": 5, "20": 20}.get(speed, 5)
        delay = {"1": 1.0, "5": 1.0, "20": 0.6}.get(speed, 1.0)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        try:
            self._write_sse("meta", {"fleet_size": int(self.predictions["unit"].nunique()), "speed": speed})
            while True:
                with self.lock:
                    payload = self.simulator.advance(batch)
                self._write_sse("snapshot", payload)
                time.sleep(delay)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _write_sse(self, event: str, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False)
        self.wfile.write(f"event: {event}\n".encode("utf-8"))
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _serve_static(self, path: str) -> None:
        clean = path.strip("/") or "index.html"
        target = (UI_DIR / clean).resolve()
        if not str(target).startswith(str(UI_DIR.resolve())) or not target.exists() or target.is_dir():
            self.send_error(404)
            return
        content_type, _ = mimetypes.guess_type(target)
        self.send_response(200)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.end_headers()
        self.wfile.write(target.read_bytes())

    def _send_json(self, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the realtime C-MAPSS fleet agent UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    import joblib
    args = parse_args()
    predictions = pd.read_csv(ROOT / "reports" / "test_predictions.csv")
    
    # RandomForest 모델 가중치(Feature Importances) 로드
    importances = None
    try:
        model_path = ROOT / "artifacts" / "rul_model.joblib"
        if model_path.exists():
            model_data = joblib.load(model_path)
            model = model_data["model"]
            feature_cols = model_data["feature_columns"]
            importances = dict(zip(feature_cols, model.feature_importances_))
            print("Loaded model feature importances for diagnostics.")
    except Exception as exc:
        print(f"Warning: Could not load feature importances: {exc}")

    RealtimeRequestHandler.predictions = predictions
    RealtimeRequestHandler.simulator = RealtimeFleetSimulator(predictions, importances=importances)
    RealtimeRequestHandler.lock = threading.Lock()
    server = ThreadingHTTPServer((args.host, args.port), RealtimeRequestHandler)
    print(f"Realtime Fleet Commander running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
