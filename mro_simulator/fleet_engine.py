from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FleetSimulationResult:
    """
    함대 정비 시뮬레이션 결과를 구조화한 데이터 클래스.
    """
    policy: str              # 적용된 시뮬레이션 정책 이름
    metrics: dict[str, float] # 고장수, 예지보전 성공률 등 분석 평가지표 딕셔너리
    decisions: pd.DataFrame   # 각 일자별 정비 결정 이력 데이터프레임


class FleetOrchestratorAgent:
    """
    함대 상태를 감시하고 제한된 정비 리소스를 감안하여, 
    정비 우선순위 가중치 점수를 부여해 지능형 스케줄링을 조율하는 오케스트레이터 에이전트.
    """
    def __init__(self, risk_threshold: float = 30.0) -> None:
        self.risk_threshold = risk_threshold  # 위험 관리가 필요한 RUL 잔여 임계값

    def rank(self, fleet: pd.DataFrame, day: int) -> pd.DataFrame:
        """
        정비되지 않았고 아직 고장나지 않은 유효 엔진들에 대해 리스크 점수를 계산하여
        정비의 시급성 순서로 정렬된 데이터프레임을 반환합니다.
        """
        # 미정비 & 비고장 활성 엔진 마스크 추출
        active = fleet.loc[(~fleet["maintained"]) & (~fleet["failed"])].copy()
        
        # 현재 일자(시뮬레이션 시간 흐름) 기준 보정된 오늘 시점의 예측 RUL
        active["predicted_remaining_today"] = active["predicted_rul"] - day
        
        # 시급성은 잔여 예측 RUL이 작을수록 큽니다.
        urgency = -active["predicted_remaining_today"]
        
        # [우선순위 점수 산출 공식]
        # 리스크 점수 = 시급성 + 0.25 * 예측 불확실성 + 2.0 * (임계치 이하의 위험 엔진 판별 가중치)
        active["risk_score"] = (
            urgency
            + 0.25 * active["pred_uncertainty"]
            + 2.0 * (active["predicted_remaining_today"] <= self.risk_threshold).astype(float)
        )
        
        # 리스크 점수(내림차순), 오늘 예측 RUL(오름차순), 예측 불확실성(내림차순) 기준으로 복합 정렬
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
    """
    특정 유지보수 정책 하에 지정된 기간(horizon) 동안 함대의 운행 및 정비 과정을 가상 시뮬레이션합니다.
    """
    rng = np.random.default_rng(seed)
    
    # 각 엔진의 기본 상태 필드 초기 설정 복사
    fleet = latest[["unit", "cycle", "true_rul", "predicted_rul", "pred_uncertainty"]].copy()
    fleet["maintained"] = False       # 정비 완료 여부
    fleet["failed"] = False           # 고장 발생 여부
    fleet["maintenance_day"] = np.nan # 정비 수행일
    fleet["failure_day"] = np.nan     # 고장 발생일
    
    decisions: list[dict[str, float | int | str]] = []  # 정비 의사결정 기록 보관용 리스트
    orchestrator = FleetOrchestratorAgent()

    # 시뮬레이션 일수 루프 (0일부터 horizon일까지 진행)
    for day in range(horizon + 1):
        # 1. 고장 판정 단계: 정비되지 않았으며 실제 잔존 수명(true_rul)이 오늘 날짜 이하가 된 엔진은 고장으로 판단
        due_to_fail = (~fleet["maintained"]) & (~fleet["failed"]) & (fleet["true_rul"] <= day)
        fleet.loc[due_to_fail, "failed"] = True
        fleet.loc[due_to_fail, "failure_day"] = day

        # 고장도 정비도 되지 않고 돌아가는 활성 엔진 추출
        active = fleet.loc[(~fleet["maintained"]) & (~fleet["failed"])].copy()
        if active.empty:
            break  # 모든 엔진이 처리되었으면 조기 종료

        # 2. 정책에 따른 정비 대상 엔진 정렬
        if policy == "orchestrator":
            # 인공지능 기반 오케스트레이터의 가중 리스크 스코어 순
            ranked = orchestrator.rank(fleet, day)
        elif policy == "shortest_predicted_rul":
            # 예측 RUL이 가장 적게 남은 엔진 순
            active["predicted_remaining_today"] = active["predicted_rul"] - day
            ranked = active.sort_values(["predicted_remaining_today", "pred_uncertainty"], ascending=[True, False])
        elif policy == "oldest_cycle":
            # 누적 구동 사이클이 가장 높은 엔진 순
            ranked = active.sort_values(["cycle", "predicted_rul"], ascending=[False, True])
        elif policy == "random":
            # 가용 리소스를 무작위 배정
            ranked = active.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000)))
        else:
            raise ValueError(f"Unknown policy: {policy}")

        # 3. 일일 정비 슬롯 크기(slots_per_day)만큼 정비 수행
        selected = ranked.head(slots_per_day)
        for _, row in selected.iterrows():
            unit = int(row["unit"])
            fleet.loc[fleet["unit"] == unit, "maintained"] = True
            fleet.loc[fleet["unit"] == unit, "maintenance_day"] = day
            
            # 최종 의사결정 로그 기록
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

    # 4. 시뮬레이션 한계(horizon)를 초과할 때까지 정비 및 고장이 없는 엔진은 강제 고장일 처리(미정비 고장)
    unhandled = (~fleet["maintained"]) & (~fleet["failed"])
    fleet.loc[unhandled, "failure_day"] = horizon + 1
    fleet.loc[unhandled, "failed"] = True

    # 5. 정책 성과지표 산출
    critical = fleet["true_rul"] <= 30  # 실제로 위험 수준(RUL 30 이하)에 이르렀던 대상 엔진 필터링
    # 위험 수준일 때 고장이 발생하기 전에 정비를 정상 완료했는지 여부
    critical_success = critical & fleet["maintained"] & (fleet["maintenance_day"] < fleet["true_rul"])
    
    metrics = {
        # 정비를 받지 못하고 최종 고장난 엔진의 총 개수 (작을수록 좋음)
        "failures_before_maintenance": float((fleet["failed"] & ~fleet["maintained"]).sum()),
        # 고장 이전에 적기에 예지보전이 진행된 성공 건수
        "maintained_before_failure": float((fleet["maintained"] & (fleet["maintenance_day"] < fleet["true_rul"])).sum()),
        # 관리 대상 위험 엔진의 수
        "critical_engines": float(critical.sum()),
        # 임계 수명 내 정비 비율 (Coverage Rate, 클수록 좋음)
        "critical_coverage_rate": float(critical_success.sum() / max(1, critical.sum())),
        # 정비가 실행된 평균 일자
        "mean_maintenance_day": float(fleet.loc[fleet["maintained"], "maintenance_day"].mean()),
        # 고장 시점 이후에 지연 정비가 발생하여 사실상 고장을 예방하지 못한 건수
        "late_maintenance_count": float((fleet["maintained"] & (fleet["maintenance_day"] >= fleet["true_rul"])).sum()),
    }
    return FleetSimulationResult(policy=policy, metrics=metrics, decisions=pd.DataFrame(decisions))
