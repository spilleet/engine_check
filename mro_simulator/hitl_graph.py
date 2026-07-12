"""
LangGraph 기반 개별 엔진 Human-in-the-Loop (HITL) 상태 그래프 모듈.

각 엔진이 임계점(RUL<20, Z-score>2.0σ 등)에 도달하면, 해당 엔진만 개별적으로
Ground Hold 상태로 전환하고 LangGraph의 interrupt_before 메커니즘을 통해
인간의 의사결정을 대기합니다.

진정한 HITL 3단계 흐름:
  0단계(검증 게이트): AI 판단 검증 → 정비 진행 or 비행 복귀
  1단계(실무자 상신): 정비 요청 및 승인서 생성 → 1차 상신
  2단계(상급자 승인): 최종 승인 or 반려
"""
from __future__ import annotations

from typing import TypedDict, Optional, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# ---------------------------------------------------------------------------
# 1. 그래프 상태 정의
# ---------------------------------------------------------------------------
class EngineHITLState(TypedDict):
    """개별 엔진의 HITL 워크플로우 상태를 정의합니다."""
    # 엔진 기본 정보 (트리거 시 백엔드에서 주입)
    unit: int
    trigger_type: str        # "rul_danger" | "zscore_anomaly" | "resource_contention"
    trigger_reason: str
    rul: float
    uncertainty: float
    cycle: int
    
    # 진단 결과 (diagnose 노드에서 채워짐)
    anomalies: list[dict]
    recommendations: list[dict]
    
    # 보고서 (generate_report 노드에서 채워짐)
    report_md: str
    
    # 0단계: 인간 검증 게이트 입력
    gate_decision: Optional[str]     # "proceed_maintenance" | "release_to_flight"
    gate_reason: Optional[str]
    
    # 1~2단계: 기존 결재 입력
    approval_decision: Optional[str]  # "approve" | "approve_final" | "defer" | "reject"
    approval_reason: Optional[str]
    
    # Ground Hold 메타데이터
    grounded_at_tick: int
    idle_ticks: int
    idle_cost: float
    
    # 최종 결과
    outcome: Optional[str]    # "maintenance_started" | "released" | "deferred" | "rejected"


# ---------------------------------------------------------------------------
# 2. 그래프 노드 함수 정의
# ---------------------------------------------------------------------------
def diagnose_engine(state: EngineHITLState) -> dict:
    """
    [자동 실행 노드] 
    엔진의 센서 Z-score 진단 및 정비 가이드라인을 도출합니다.
    실제 진단은 시뮬레이터 내부의 에이전트를 호출하여 수행되며,
    그 결과가 트리거 시점에 state에 미리 주입됩니다.
    """
    # 진단 데이터는 트리거 시점에 이미 state에 주입되므로 패스스루
    return {}


def generate_report(state: EngineHITLState) -> dict:
    """
    [자동 실행 노드]
    AI 에이전트 파이프라인(기술 서기 + 재무 플래너 + 품질 에디터)을 가동하여
    정비 요청 및 승인서 마크다운 초안을 생성합니다.
    실제 생성은 시뮬레이터의 MaintenanceReportAgent를 호출하여 수행되며,
    그 결과가 state.report_md에 주입됩니다.
    """
    # 보고서 데이터는 트리거 시점에 이미 state에 주입되므로 패스스루
    return {}


def validate_ground_hold(state: EngineHITLState) -> dict:
    """
    [0단계 HITL 검증 게이트 - 인간 개입 노드]
    이 노드 진입 직전에 interrupt_before가 걸려 그래프 실행이 일시 중단됩니다.
    인간이 gate_decision 값을 update_state로 주입하면 재개됩니다.
    
    - "proceed_maintenance": 정비 진행 → 기존 결재 흐름으로 전환
    - "release_to_flight": 비행 복귀 → 즉시 Ground Hold 해제
    """
    return {"outcome": state.get("gate_decision", "released")}


def apply_final_decision(state: EngineHITLState) -> dict:
    """
    [최종 의사결정 반영 노드]
    인간의 결재 결과를 시뮬레이터 상태에 반영합니다.
    실제 반영은 run_server.py의 manual_decision()이 수행합니다.
    """
    decision = state.get("gate_decision", "release_to_flight")
    if decision == "release_to_flight":
        return {"outcome": "released"}
    return {"outcome": "maintenance_started"}


# ---------------------------------------------------------------------------
# 3. 조건부 라우팅 함수
# ---------------------------------------------------------------------------
def route_after_gate(state: EngineHITLState) -> str:
    """0단계 검증 게이트 이후 분기: 정비 진행 vs 비행 복귀"""
    if state.get("gate_decision") == "proceed_maintenance":
        return "proceed"
    return "release"


# ---------------------------------------------------------------------------
# 4. 그래프 빌드 및 컴파일
# ---------------------------------------------------------------------------
def build_hitl_graph() -> tuple:
    """
    LangGraph StateGraph를 빌드하고 MemorySaver 체크포인터와 함께 컴파일합니다.
    
    그래프 구조:
      [diagnose_engine] → [generate_report] → ((interrupt)) → [validate_ground_hold]
                                                                      │
                                                        ┌─────────────┴──────────────┐
                                                        ▼                            ▼
                                               [apply_final_decision]        [apply_final_decision]
                                               (정비 진행)                   (비행 복귀)
                                                        │                            │
                                                        ▼                            ▼
                                                       END                          END
    
    Returns:
        tuple: (compiled_graph, memory_saver)
    """
    workflow = StateGraph(EngineHITLState)
    
    # 노드 등록
    workflow.add_node("diagnose_engine", diagnose_engine)
    workflow.add_node("generate_report", generate_report)
    workflow.add_node("validate_ground_hold", validate_ground_hold)
    workflow.add_node("apply_final_decision", apply_final_decision)
    
    # 엣지 연결
    workflow.set_entry_point("diagnose_engine")
    workflow.add_edge("diagnose_engine", "generate_report")
    workflow.add_edge("generate_report", "validate_ground_hold")
    
    # 0단계 검증 게이트 이후 조건부 분기
    workflow.add_conditional_edges(
        "validate_ground_hold",
        route_after_gate,
        {
            "proceed": "apply_final_decision",
            "release": "apply_final_decision",
        }
    )
    workflow.add_edge("apply_final_decision", END)
    
    # MemorySaver 체크포인터로 컴파일
    # ★ validate_ground_hold 노드 진입 직전에 interrupt를 걸어 인간 결재 대기
    memory = MemorySaver()
    compiled = workflow.compile(
        checkpointer=memory,
        interrupt_before=["validate_ground_hold"]
    )
    
    return compiled, memory
"""
사용 예시:

    # 1. 그래프 빌드 (서버 시작 시 1회)
    hitl_graph, memory = build_hitl_graph()
    
    # 2. 트리거 발동 시 그래프 기동 (엔진별 독립 thread_id)
    config = {"configurable": {"thread_id": f"engine_{unit}"}}
    hitl_graph.invoke(initial_state, config)
    # → diagnose → generate_report 까지 실행 후 validate_ground_hold 직전에 자동 중단
    
    # 3. 인간 결재 입력 주입 및 재개
    hitl_graph.update_state(config, {"gate_decision": "proceed_maintenance"}, as_node="validate_ground_hold")
    result = list(hitl_graph.stream(None, config))
    # → validate_ground_hold → apply_final_decision → END 까지 실행
"""
