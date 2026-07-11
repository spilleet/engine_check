from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
load_dotenv()  # .env 파일에 정의된 API 키 등을 로드

import pandas as pd

# MRO 에이전트 패키지 임포트
from mro_simulator.mro_agents import (
    CrisisDetectionAgent,
    MaintenanceActionAgent,
    SituationQueryAgent,
    MaintenanceDiagnosticianAgent,
    ActionRecommendationAgent,
    MaintenanceReportAgent,
)
from mro_simulator.alert_agent import SmartAlertAgent

# 루트 경로 및 UI 정적 리소스 파일이 들어있는 ui 디렉토리 경로 지정
ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "ui"


class RealtimeFleetSimulator:
    """
    실시간 함대 상태 모니터링 및 정비 시뮬레이션을 수행하는 시뮬레이터 클래스.
    시간의 흐름(tick)에 따라 각 엔진의 RUL을 차감하고, 정비 작업 완료/지연 상태를 관리하며,
    인간 결재 기반 의사결정(HITL)에 따라 리스크 가중치를 보정합니다.
    """
    def __init__(self, predictions: pd.DataFrame, importances: dict[str, float] | None = None) -> None:
        # 예측 데이터프레임을 원본 상태로 복사 후 실시간 상태 관리를 위한 다양한 속성 컬럼들을 동적 생성
        self.frame = predictions.copy()
        
        self.frame["stream_cycle"] = self.frame["cycle"]          # 현재 진행 중인 사이클 누적치
        self.frame["rul"] = self.frame["predicted_rul"]            # 예측 RUL (매 tick 감소)
        self.frame["true_remaining"] = self.frame["true_rul"]      # 실제 잔존 RUL (매 tick 감소)
        self.frame["maintained"] = False                           # 정비 완료 여부
        self.frame["under_maintenance"] = False                    # 현재 정비소 입고(정비 진행) 중 여부
        self.frame["maintenance_remaining_ticks"] = 0              # 정비 완료까지 남은 시간 (3틱 스케줄링)
        self.frame["human_modifier"] = 1.0                         # 실무자 보류 시 리스크 점수를 50% 깎기 위한 피드백 곱연산 가중치
        self.frame["pending_supervisor"] = False                   # 상급자 최종 승인을 대기 중인 상태 표시
        self.frame["risk_score"] = 0.0                             # 에이전트가 계산할 리스크 점수
        self.frame["status"] = "healthy"                           # 엔진 건강 상태 문자열
        
        # 하위 분석 및 리포팅 에이전트 클래스들의 인스턴스 초기화
        self.detector = CrisisDetectionAgent()
        self.query = SituationQueryAgent()
        self.action = MaintenanceActionAgent(slots_per_round=3)
        self.diagnostician = MaintenanceDiagnosticianAgent(feature_importances=importances)
        self.recommender = ActionRecommendationAgent()
        self.reporter = MaintenanceReportAgent()
        
        # SmartAlertAgent 초기화 (환경 변수의 API 키 전달)
        import os
        self.alert_agent = SmartAlertAgent(api_key=os.environ.get("OPENAI_API_KEY"))
        
        self.tick = 0        # 시뮬레이션 경과 시간(초 단위)
        self.pointer = 0     # 텔레메트리 값을 순차 감소시키기 위해 활성 엔진을 가리키는 순환 포인터
        
        # 예지보전 성과 지표 데이터 변수
        self.agent_cost = 0
        self.baseline_cost = 0
        self.protected_failures = 0
        self.missed_failures = 0
        
        # 각 엔진별 정비 이력 횟수를 기록하는 딕셔너리
        self.maintenance_counts = {int(unit): 0 for unit in self.frame["unit"].unique()}
        
        # 실시간 모니터링 로그 기록용 초기 이벤트 설정
        # (기존 버그 수정: 정의되지 않은 slots 변수를 self.action.slots_per_round로 교체)
        self.events: list[dict] = [
            {"time": "00:00", "agent": "system", "message": "실시간 스트림 연결. 엔진 telemetry 수신을 시작합니다."},
            {"time": "00:00", "agent": "objective", "message": f"목표: RUL 급락 감지, 상황 조회, 슬롯 {self.action.slots_per_round}개 내 정비 조치."},
        ]
        self.work_orders: list[dict] = []  # 발행 완료 및 진행 대기 중인 작업 지서 리스트
        self.deferred_units: set[int] = set()  # 관리에 의해 모니터링 보류 처리된 엔진 유닛 셋
        
        # 상태 변화 감지를 위해 직전 상태를 기록해두는 사전
        self.last_status = {int(row.unit): "healthy" for row in self.frame.itertuples()}
        
        # 최초 1회 감지 에이전트 동작으로 리스크 스코어 초기화
        self.frame = self.detector.annotate(self.frame)

    def _calculate_restored_rul(self, unit: int, anomalies_count: int) -> float:
        """
        정비 완료 시 복원될 RUL 값을 복합 물리 원리에 따라 가상 연산합니다.
        - 누적 정비 횟수가 늘어날수록 금속 피로 축적으로 정비 상한치 페널티를 부과합니다.
        - 이상 발생한 센서의 개수(조치 수준)에 따라 정비 복원율을 차등적으로 조정합니다.
        """
        count = self.maintenance_counts.get(unit, 0)
        self.maintenance_counts[unit] = count + 1
        
        base_limit = 125.0
        # 누적 정비 1회당 10사이클씩 최대 수명 상한 하향 조정 (하한선은 60사이클)
        wear_out_penalty = count * 10.0
        max_possible_rul = max(60.0, base_limit - wear_out_penalty)
        
        # 조치된 원인 센서 가이드 개수에 따른 성능 복원율 차등화
        if anomalies_count >= 3:
            restore_rate = 0.96  # 주요(Major) 정비 수준
        elif anomalies_count == 2:
            restore_rate = 0.80  # 중간(Medium) 정비 수준
        else:
            restore_rate = 0.65  # 가벼운(Minor) 예방 정비 수준
            
        restored = max_possible_rul * restore_rate
        return float(round(restored, 1))

    def advance(self, telemetry_events: int) -> dict:
        """
        시뮬레이션 시간을 1틱 진행합니다.
        1. 정비소에 입고된(under_maintenance) 엔진의 정비 남은 시간(ticks)을 차감하고, 완료 시 RUL 복원 적용.
        2. 스트림 유입 속도에 해당하는 `telemetry_events` 만큼의 엔진 텔레메트리 사이클 진행 및 RUL 감소.
        3. 최신 상태 분석(이상 감지, 성과 지표 업데이트)을 반영한 대시보드 갱신용 API 페이로드 반환.
        """
        now_str = self._clock()
        
        # 1. 정비 진행 중인 엔진들에 대한 입고 틱 처리
        for row in self.frame.itertuples():
            if bool(row.under_maintenance):
                idx = self.frame["unit"] == row.unit
                ticks = int(row.maintenance_remaining_ticks) - 1
                if ticks <= 0:
                    # 입고 정비 완료: 정비 상태 해제 및 가동 복귀
                    self.frame.loc[idx, "under_maintenance"] = False
                    self.frame.loc[idx, "maintenance_remaining_ticks"] = 0
                    self.frame.loc[idx, "maintained"] = True
                    self.frame.loc[idx, "human_modifier"] = 1.0  # 보류 상태 리셋
                    self.last_status[int(row.unit)] = "maintained"
                    
                    # 진단 결과를 조회하여 정비 수준 산출 및 RUL 회복 적용
                    diag = self.diagnostician.diagnose(self.frame, int(row.unit))
                    rec = self.recommender.recommend(diag)
                    restored_rul = self._calculate_restored_rul(int(row.unit), len(rec["checklist"]))
                    self.frame.loc[idx, "rul"] = restored_rul
                    self.agent_cost += 8000  # 정비 비용 소모
                    
                    # 고장이 실제로 발생하기 전에(실제 RUL 8 이하일 때) 성공적으로 예방했는지 여부 기록
                    if float(row.true_remaining) <= float(row.rul) + 8:
                        self.protected_failures += 1
                        
                    self.events.append({
                        "time": now_str,
                        "agent": "action_agent",
                        "message": f"엔진 #{int(row.unit)} 정비 완료 및 가동 상태 복귀. 복원 RUL: {restored_rul}."
                    })
                else:
                    # 정비 진행 시간 차감
                    self.frame.loc[idx, "maintenance_remaining_ticks"] = ticks

        # 2. 텔레메트리 경과 및 구동 주기 차감 수행
        touched_units: list[int] = []
        units = self.frame["unit"].astype(int).tolist()
        
        # 주어진 이벤트 빈도수(batch 크기) 만큼 동작 상태 진행
        for _ in range(telemetry_events):
            unit = units[self.pointer % len(units)]
            self.pointer += 1
            touched_units.append(unit)
            idx = self.frame["unit"] == unit
            
            # 이미 정비 완료되었거나, 현재 입고 정비 중인 엔진은 가동 수명이 감소하지 않으므로 스킵
            if bool(self.frame.loc[idx, "maintained"].iloc[0]) or bool(self.frame.loc[idx, "under_maintenance"].iloc[0]):
                continue
            
            # 구동 사이클 1 증가 및 잔존 수명 차감
            self.frame.loc[idx, "stream_cycle"] += 1
            self.frame.loc[idx, "rul"] = (self.frame.loc[idx, "rul"] - 1.0).clip(lower=0)
            self.frame.loc[idx, "true_remaining"] = (self.frame.loc[idx, "true_remaining"] - 1.0).clip(lower=0)

        # 틱 증가 및 상태 이상 다시 판별
        self.tick += 1
        self.frame = self.detector.annotate(self.frame)
        now = self._clock()
        
        # 이전 틱 대비 상태에 변화가 생긴 엔진 추적 및 자동 정비권고 판단
        changed = self._status_changes(now)
        actions = self._maybe_act(now)
        
        # 정비 지연 벌금 및 상태 모니터링 비용 계산
        self._update_costs()
        
        summary = self.query.summarize(self.frame)
        top_risks = self.query.top_risks(self.frame)

        # 상태 변화가 생겼을 때만 추론 로그에 등록하여 로그창 도배 방지
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
        """대시보드 최초 로드 시 전달할 스냅샷 데이터를 반환합니다."""
        summary = self.query.summarize(self.frame)
        top_risks = self.query.top_risks(self.frame)
        return self._payload("00:00", [], summary, top_risks, [])

    def manual_decision(self, unit: int, decision: str, reason: str = "") -> dict:
        """
        인간 관제사 개입(HITL)에 따른 2단계 의사결정을 실시간 반영합니다.
        1. `approve`: 실무자 정비 오더 1차 상신. (상급자 결재 대기 `pending_supervisor` 격상 및 정비 작업 지시서 마크다운 생성)
        2. `approve_final`: 상급자 최종 승인. (입고 대기 `under_maintenance` 상태 활성화 및 3틱 타이머 돌입, 정비 작업 지시서 로컬 파일 저장)
        3. `defer`: 실무자 모니터링 보류. (기여 가중치 `human_modifier`를 0.5로 낮추어 리스크 경보 완화)
        4. `reject`: 결재 반려 및 종결.
        """
        now = self._clock()
        idx = self.frame["unit"] == unit
        if not idx.any():
            raise ValueError(f"Unknown unit: {unit}")
        row = self.frame.loc[idx].iloc[0]
        reason_text = reason.strip() or "사유 미입력"

        if decision == "approve":
            # [1단계] 실무자 결재 요청 상신
            self.frame.loc[idx, "pending_supervisor"] = True
            
            # 센서 데이터 진단 후 정비 오더 체크리스트 및 작업 지시서 초안 생성
            diag = self.diagnostician.diagnose(self.frame, unit)
            rec = self.recommender.recommend(diag)
            report_md = self.reporter.generate_markdown(
                unit=unit,
                cycle=int(row["stream_cycle"]),
                predicted_rul=float(row["rul"]),
                uncertainty=float(row["pred_uncertainty"]),
                recommendations=rec,
                reason=reason_text,
                maintenance_count=self.maintenance_counts.get(unit, 0)
            )
            
            # 결재 대기 중인 오더 레코드 추가
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
            # [2단계] 상급자 최종 승인 완료 -> 입고 및 수리 개시
            self.frame.loc[idx, "pending_supervisor"] = False
            self.frame.loc[idx, "under_maintenance"] = True
            self.frame.loc[idx, "maintenance_remaining_ticks"] = 3
            
            # 결재 대기열의 이전 요청을 찾아 최종 서명된 오더 상태로 격상
            found = False
            for order in self.work_orders:
                if order["unit"] == unit and order["decision"] == "pending":
                    order["id"] = f"WO-{int(time.time())}-{unit:03d}"
                    order["decision"] = "approved"
                    order["status"] = "under_maintenance"
                    order["time"] = now
                    
                    # 작업 지시서 아카이브 디렉토리에 마크다운 형식의 서명 지시서 파일 저장
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
                # 상신 단계를 거치지 않은 경우 즉시 직권 정비 실행 처리
                diag = self.diagnostician.diagnose(self.frame, unit)
                rec = self.recommender.recommend(diag)
                report_md = self.reporter.generate_markdown(
                    unit=unit,
                    cycle=int(row["stream_cycle"]),
                    predicted_rul=float(row["rul"]),
                    uncertainty=float(row["pred_uncertainty"]),
                    recommendations=rec,
                    reason="상급자 직권 정비 승인",
                    maintenance_count=self.maintenance_counts.get(unit, 0)
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
            # [대체조치] 임시로 모니터링 강도 완화 (가중치 0.5 적용)
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
            # [대체조치] 결재 반려 처리
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

        # 가중치 갱신 결과 반영을 위해 이상 판단 재어노테이션
        self.frame = self.detector.annotate(self.frame)
        
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
        """API 및 SSE로 갱신될 데이터 묶음을 구성하여 반환합니다."""
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
            "log": self.events[-12:],          # 최근 추론 로그 12개
            "work_orders": self.work_orders[-12:],  # 최근 발행 지서 12개
        }

    def _status_changes(self, now: str) -> list[int]:
        """직전 틱과 대비하여 엔진 상태가 inspect(점검)나 danger(위험)로 전이되었는지 감시합니다."""
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
                
                # SmartAlertAgent 호출하여 최적의 알람 전송 수단 판별
                alert_desc = self.alert_agent.run_alert_logic(unit, self.tick, current, float(row.rul))
                self.events.append(
                    {
                        "time": now,
                        "agent": "smart_alert_agent",
                        "message": f"[알람 에이전트] {alert_desc}"
                    }
                )
            self.last_status[unit] = current
        return changed

    def _maybe_act(self, now: str) -> list[int]:
        """5틱 간격으로 AI 정비 에이전트가 긴급 엔진의 정비 추천을 진단 추론 로그에 작성합니다."""
        if self.tick % 5 != 0:
            return []
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
        return []  # 권고만 하고 실제 수리는 결재 절차가 필수이므로 여기서는 미집행

    def _update_costs(self) -> None:
        """기존 baseline 수치(RUL 30이하 고장 방치에 다른 벌금)를 업데이트합니다."""
        active = ~self.frame["maintained"]
        # 실제 가동 수명이 다 닳아 0 이하가 된 미보전 고장 판정
        missed_now = int(((self.frame["true_remaining"] <= 0) & active).sum())
        self.missed_failures += missed_now
        self.baseline_cost += int(((self.frame["rul"] < 30) & active).sum()) * 8000 + missed_now * 50000

    def _clock(self) -> str:
        """시뮬레이션 시간 흐름을 시:분 문자열로 변환합니다."""
        seconds = self.tick
        return f"{seconds // 60:02d}:{seconds % 60:02d}"


class RealtimeRequestHandler(BaseHTTPRequestHandler):
    """대시보드 브라우저의 AJAX/Fetch 통신 및 SSE 스트림 연결을 처리하는 HTTP 리퀘스트 핸들러."""
    predictions: pd.DataFrame
    simulator: RealtimeFleetSimulator
    lock: threading.Lock  # 시뮬레이션 상태 갱신 동시성 제어를 위한 스레드 락

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        
        # 1. Server-Sent Events (SSE) 실시간 이벤트 통신 연결부
        if parsed.path == "/api/events":
            self._serve_events(parsed.query)
            return
            
        # 2. 최초 대시보드 로드용 스냅샷 전송
        if parsed.path == "/api/state":
            with self.lock:
                self._send_json({"fleet_size": int(self.predictions["unit"].nunique()), "initial": self.simulator.snapshot()})
            return
            
        # 3. 특정 엔진의 이상 징후 Z-score 진단 및 가이드라인 가공 조회
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
            
        # 4. 정적 파일 호스팅 (index.html, styles.css, app.js 등)
        self._serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        
        # 1. 2단계 의사결정 요청 집행 엔드포인트
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
        """
        SSE 연결을 열어두고 배정된 배속 설정(`speed`)에 맞춰 
        실시간 시뮬레이션을 틱 단위로 한 단계씩 구동하면서 프론트엔드로 푸시합니다.
        """
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
            # 최초 메타 정보 스트림 전송
            self._write_sse("meta", {"fleet_size": int(self.predictions["unit"].nunique()), "speed": speed})
            while True:
                with self.lock:
                    # 지정된 batch 수량만큼 시뮬레이션 진척
                    payload = self.simulator.advance(batch)
                self._write_sse("snapshot", payload)
                time.sleep(delay)
        except (BrokenPipeError, ConnectionResetError):
            return  # 클라이언트가 연결을 끊은 경우 안전하게 스트리밍 루프 탈출

    def _write_sse(self, event: str, payload: dict) -> None:
        """SSE 전송 프로토콜 규격에 맞춰 wfile 버퍼에 작성합니다."""
        data = json.dumps(payload, ensure_ascii=False)
        self.wfile.write(f"event: {event}\n".encode("utf-8"))
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _serve_static(self, path: str) -> None:
        """정적 자원(html, css, js 등)을 찾아 브라우저로 전송합니다. 경로 탐색(directory traversal) 공격 우회를 검사합니다."""
        clean = path.strip("/") or "index.html"
        target = (UI_DIR / clean).resolve()
        
        # 지정된 UI_DIR 범위 바깥의 시스템 경로 호출 차단
        if not str(target).startswith(str(UI_DIR.resolve())) or not target.exists() or target.is_dir():
            self.send_error(404)
            return
            
        content_type, _ = mimetypes.guess_type(target)
        self.send_response(200)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.end_headers()
        self.wfile.write(target.read_bytes())

    def _send_json(self, payload: dict) -> None:
        """JSON 데이터 전송 헬퍼 메서드."""
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        # 터미널 콘솔 로그 출력을 억제하여 불필요한 IO 지연 차단
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the realtime C-MAPSS fleet agent UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    import joblib
    args = parse_args()
    
    # 훈련 시 검증 완료된 RUL 예측치 파일 로드
    predictions = pd.read_csv(ROOT / "reports" / "test_predictions.csv")
    
    # RandomForest 모델 가중치(Feature Importances) 파일 로드
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

    # 핸들러 스키마 주입 및 실시간 멀티스레드 웹 서버 개시
    RealtimeRequestHandler.predictions = predictions
    RealtimeRequestHandler.simulator = RealtimeFleetSimulator(predictions, importances=importances)
    RealtimeRequestHandler.lock = threading.Lock()
    
    server = ThreadingHTTPServer((args.host, args.port), RealtimeRequestHandler)
    print(f"Realtime Fleet Commander running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
