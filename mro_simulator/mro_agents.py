from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class AgentEvent:
    """
    에이전트 시뮬레이션 중 발생하는 이벤트를 정의하는 데이터 구조 클래스.
    """
    round: int     # 이벤트 발생 라운드 번호
    agent: str     # 이벤트를 발행한 주체 에이전트 이름 (예: crisis_detector, action_agent)
    message: str   # 이벤트 세부 메시지 내용


class TelemetryStreamAgent:
    """
    엔진 텔레메트리 데이터 스트림을 가상으로 시뮬레이션하고 제공하는 에이전트.
    매 라운드마다 엔진의 구동 주기를 증가시키고 예측 수명 및 실제 잔여 수명을 점진적으로 감소시킵니다.
    """
    def __init__(self, latest_predictions: pd.DataFrame, horizon: int = 80) -> None:
        self.latest = latest_predictions.copy()  # RUL 예측값 파일에서 불러온 기본 데이터셋
        self.horizon = horizon                   # 최대 시뮬레이션 주기

    def frame(self, round_no: int, maintained_units: set[int]) -> pd.DataFrame:
        """
        주어진 라운드 정보와 정비 완료된 엔진 식별자 세트를 활용하여,
        현재 시간 흐름에 맞춰 동적으로 계산된 텔레메트리 프레임을 생성합니다.
        """
        frame = self.latest.copy()
        
        # 현재 라운드 수만큼 시뮬레이션 사이클 증가 보정
        frame["stream_cycle"] = frame["cycle"] + round_no - 1
        
        # 예측 잔여 RUL 1씩 감소 처리 (최소값은 0.0 보장)
        frame["rul"] = (frame["predicted_rul"] - round_no + 1).clip(lower=0)
        
        # 정답(Ground Truth) 잔여 수명도 1씩 감소 처리
        frame["true_remaining"] = (frame["true_rul"] - round_no + 1).clip(lower=0)
        
        # 이미 정비가 완료된 엔진인지 판별 마킹
        frame["maintained"] = frame["unit"].isin(maintained_units)
        
        # 정비된 엔진은 임시로 롤백 RUL(예: 125 사이클) 상태로 설정하여 긴급도 분석에서 배제시킴
        frame.loc[frame["maintained"], "rul"] = 125
        return frame


class CrisisDetectionAgent:
    """
    각 엔진의 RUL 예측 신호와 불확실성, 그리고 외부 인간 피드백 가중치를 통합 계산하여,
    엔진의 실시간 건강 상태를 판별하고 리스크 스코어를 주입하는 이상 감지 및 위기 진단 에이전트.
    """
    def __init__(self, danger_threshold: float = 20.0, inspect_threshold: float = 50.0) -> None:
        self.danger_threshold = danger_threshold    # 위험 경보 임계값 (RUL < 20.0)
        self.inspect_threshold = inspect_threshold  # 점검 경보 임계값 (RUL < 50.0)

    def annotate(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        텔레메트리 프레임 데이터의 RUL 및 조건 컬럼을 검사해 
        위험 지수를 레이블링하고 종합 리스크 점수(`risk_score`)를 연산합니다.
        """
        out = frame.copy()
        out["status"] = "healthy"  # 기본 정상 상태 설정
        
        # RUL 임계값에 맞춰 순차적 경보 격상
        out.loc[out["rul"] < self.inspect_threshold, "status"] = "inspect"
        out.loc[out["rul"] < self.danger_threshold, "status"] = "danger"
        out.loc[out["maintained"], "status"] = "maintained"
        
        # 1. 정비 중 상태 예외 처리
        if "under_maintenance" in out.columns:
            out.loc[out["under_maintenance"], "status"] = "under_maintenance"
            
        # 2. 인간 피드백을 통한 가중치 컬럼(human_modifier)이 없는 경우 기본값 1.0으로 초기화
        if "human_modifier" not in out.columns:
            out["human_modifier"] = 1.0
            
        # 리스크 점수 산출 핵심 로직:
        # (점검 임계값 - 예측 RUL) 변위값 + 0.35 * 예측 불확실성 + danger 상태 가중치(4.0)
        base_risk = (
            (self.inspect_threshold - out["rul"]).clip(lower=0)
            + 0.35 * out["pred_uncertainty"]
            + 4.0 * (out["status"] == "danger").astype(float)
        )
        
        # 3. 안전 가드레일: 실제 잔존 수명이 극도로 낮은 엔진(true_remaining < 15)은
        #    사용자의 강제 보류 가중치(`human_modifier` = 0.5)가 적용되더라도 위험 상태를 유지하도록 오프셋 부여
        true_rem = out["true_remaining"] if "true_remaining" in out.columns else out["rul"]
        guardrail = 20.0 * ((out["status"] == "danger") & (true_rem < 15)).astype(float)
        
        # 최종 점수 = 기본 점수 * 보류 팩터(human_modifier) + 가드레일 오프셋
        out["risk_score"] = base_risk * out["human_modifier"] + guardrail
        return out


class SituationQueryAgent:
    """
    전체 함대의 건강 상태 비율을 통합하고, 실시간 위기 모니터링 시 상위 위험도 엔진 목록을 추출해주는 분석 에이전트.
    """
    def summarize(self, frame: pd.DataFrame) -> dict[str, float]:
        """
        정비 완료된 엔진을 제외한 미정비 활성 엔진들의 
        건강 상태 그룹 수량 및 예측 RUL 최저값, 불확실성 평균치 요약을 산출합니다.
        """
        active = frame.loc[~frame["maintained"]]
        return {
            "healthy": int((active["status"] == "healthy").sum()),
            "inspect": int((active["status"] == "inspect").sum()),
            "danger": int((active["status"] == "danger").sum()),
            "maintained": int(frame["maintained"].sum()),
            "lowest_rul": float(active["rul"].min()) if len(active) else 0.0,
            "mean_uncertainty": float(active["pred_uncertainty"].mean()) if len(active) else 0.0,
        }

    def top_risks(self, frame: pd.DataFrame, limit: int = 8) -> list[dict]:
        """
        미정비 엔진 중 리스크 가중치가 크고 RUL이 작게 남은 우선순위 순서대로
        상위 위험 엔진 목록을 JSON 직렬화가 가능한 딕셔너리 리스트로 반환합니다.
        """
        active = frame.loc[~frame["maintained"]].sort_values(["risk_score", "rul"], ascending=[False, True])
        return active.head(limit)[["unit", "stream_cycle", "rul", "true_remaining", "pred_uncertainty", "status", "risk_score"]].to_dict(
            orient="records"
        )


class MaintenanceActionAgent:
    """
    지정된 정비 가용 슬롯 제약 한도에 맞추어, 
    점검 및 위험군 대상 목록 중 정비 우선 배치 대상 엔진을 추천하는 실행 조율 에이전트.
    """
    def __init__(self, slots_per_round: int = 3) -> None:
        self.slots_per_round = slots_per_round  # 라운드당 정비 가용 슬롯 (자원 제한)

    def choose_actions(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        정비되지 않았고 점검/위험 경보가 발생한 엔진에 한해 리스크 스코어가 높은 순서대로
        일일 정비 리소스 한계 내의 정선된 데이터프레임을 반환합니다.
        """
        candidates = frame.loc[(~frame["maintained"]) & (frame["status"].isin(["danger", "inspect"]))].copy()
        ranked = candidates.sort_values(["risk_score", "rul", "pred_uncertainty"], ascending=[False, True, False])
        return ranked.head(self.slots_per_round)


class FleetAgentPipeline:
    """
    시뮬레이션 전체 라운드 루프를 실행하며 에이전트 간의 메시지 상호작용 및
    예지보전 성과 지표(에이전트 정책 비용 대 기준 정책 비용)를 평가하는 통합 파이프라인.
    """
    def __init__(self, latest_predictions: pd.DataFrame, slots_per_round: int = 3, horizon: int = 80) -> None:
        self.stream = TelemetryStreamAgent(latest_predictions, horizon=horizon)
        self.detector = CrisisDetectionAgent()
        self.query = SituationQueryAgent()
        self.action = MaintenanceActionAgent(slots_per_round=slots_per_round)
        self.slots_per_round = slots_per_round
        self.horizon = horizon

    def run(self) -> dict:
        """
        설정된 기간 동안의 에이전트 파이프라인 시뮬레이션을 동기식으로 일괄 구동하고
        대시보드 초기화용 라운드별 시계열 이력을 반환합니다.
        """
        maintained_units: set[int] = set()
        protected_failures = 0          # 방어한 고장 횟수 (실제 고장 발생 전 조치 성공)
        missed_failures = 0             # 방어 실패 고장 횟수 (실제 고장일 이후 조치 혹은 미조치)
        cumulative_agent_cost = 0       # 에이전트 정책 누적 비용 ($8,000 * 정비 수행 횟수)
        cumulative_baseline_cost = 0    # 기존 기준 정책(RUL 30이하 도달 시 일률 조치 + 고장 페널티) 누적 비용
        rounds = []
        events: list[AgentEvent] = [
            AgentEvent(1, "system", "시스템 초기화 완료. 함대 관제 에이전트가 데이터 스트림을 수신합니다."),
            AgentEvent(1, "objective", f"목표: 고장 회피와 정비 비용 최소화. 가용 슬롯: {self.slots_per_round}개/라운드."),
        ]

        # 1라운드부터 horizon 라운드까지 순차적 시뮬레이션
        for round_no in range(1, self.horizon + 1):
            # 텔레메트리 데이터 갱신 및 위기 상태 판별 어노테이션
            frame = self.detector.annotate(self.stream.frame(round_no, maintained_units))
            summary = self.query.summarize(frame)
            top_risks = self.query.top_risks(frame)
            
            # 정비 배정할 엔진 선정
            actions = self.action.choose_actions(frame)
            action_units = [int(unit) for unit in actions["unit"].tolist()]

            # 이상 감지 이벤트 기록 등록
            danger_units = frame.loc[(~frame["maintained"]) & (frame["status"] == "danger"), "unit"].astype(int).tolist()
            if danger_units:
                events.append(AgentEvent(round_no, "crisis_detector", f"위험 엔진 감지: {danger_units[:8]}"))
            
            events.append(
                AgentEvent(
                    round_no,
                    "situation_query",
                    f"상황조회: 위험 {summary['danger']}대, 점검요망 {summary['inspect']}대, 최저 예측 RUL {summary['lowest_rul']:.1f}.",
                )
            )

            # 정비 처방 수행에 따른 비용 산정 및 고장 방어 유무 확인
            for unit in action_units:
                row = frame.loc[frame["unit"] == unit].iloc[0]
                # 실제 잔존 수명이 예측 수명에 근접한 급박한 시점(RUL 오차범위 8 사이클) 이하에서 조치되었다면 성공적인 고장 예방
                if row["true_remaining"] <= row["rul"] + 8:
                    protected_failures += 1
                maintained_units.add(unit)
                
            cumulative_agent_cost += len(action_units) * 8000
            
            # 미정비 상태에서 고장난 개수 계산 및 기준 정책 비용 업데이트
            # (기존 정책: 고장 방치 시 한 대당 $50,000의 높은 페널티 부과)
            missed_now = int(((frame["true_remaining"] <= 0) & (~frame["maintained"])).sum())
            missed_failures += missed_now
            cumulative_baseline_cost += int((frame["rul"] < 30).sum()) * 8000 + missed_now * 50000

            if action_units:
                events.append(AgentEvent(round_no, "action_agent", f"정비 슬롯 배정: 엔진 {action_units}. 예상 비용 ${len(action_units) * 8000:,}."))
            else:
                events.append(AgentEvent(round_no, "action_agent", "정비 슬롯 배정 없음. 위험도 기준 미달."))

            visible = frame[["unit", "stream_cycle", "rul", "pred_uncertainty", "status", "maintained"]].copy()
            # 해당 라운드의 결과 상태 스냅샷 저장
            rounds.append(
                {
                    "round": round_no,
                    "summary": summary,
                    "engines": visible.to_dict(orient="records"),
                    "top_risks": top_risks,
                    "actions": action_units,
                    "cost": {
                        "agent": cumulative_agent_cost,
                        "baseline": cumulative_baseline_cost,
                        "protected_failures": protected_failures,
                        "missed_failures": missed_failures,
                    },
                    # 로그 창 가독성을 위해 최근 3개 라운드 이력만 제한 필터링
                    "log": [event.__dict__ for event in events if max(1, round_no - 3) <= event.round <= round_no],
                }
            )

        return {
            "meta": {
                "fleet_size": int(self.stream.latest["unit"].nunique()),
                "slots_per_round": self.slots_per_round,
                "horizon": self.horizon,
                "danger_threshold": 20,
                "inspect_threshold": 50,
            },
            "rounds": rounds,
        }


class MaintenanceDiagnosticianAgent:
    """
    정비 대상 엔진의 이상 센서를 검출하기 위해 데이터 세트 내의 
    동적 변동 특성($Z$-score)과 AI 모델의 피처 중요도를 결합하여 핵심 이상 원인을 추적하는 진단 에이전트.
    """
    def __init__(self, feature_importances: dict[str, float] | None = None) -> None:
        self.feature_importances = feature_importances or {}  # 훈련 완료된 예측 가중치 정보

    def diagnose(self, frame: pd.DataFrame, unit: int) -> dict:
        """
        정비 대상 엔진의 상태 컬럼을 정상적인 활성 엔진 데이터 분포와 대조 연산하여,
        기여도(Z-score * Importance) 기준 최악의 3대 변위 센서를 산출합니다.
        """
        target_row = frame.loc[frame["unit"] == unit]
        if target_row.empty:
            return {"unit": unit, "anomalies": []}
        
        target_row = target_row.iloc[0]
        
        # 5일 이동평균 롤링 피처 위주로 비교 컬럼 필터링
        cols = [c for c in frame.columns if "roll_mean_5" in c]
        if not cols:
            cols = [c for c in frame.columns if c.startswith("s_") and not ("roll" in c)]
        
        anomalies = []
        for col in cols:
            # 정비 완료되지 않고 동작하는 엔진들을 활성 비교 분포군으로 삼음
            active = frame.loc[~frame["maintained"]]
            if len(active) <= 1:
                active = frame  # 모수가 부족한 경우 전체 엔진을 참조
            mean_val = active[col].mean()
            std_val = active[col].std()
            
            val = target_row[col]
            # 편차 지수 Z-score 계산 (분모 0 방지 epsilon 오차 결합)
            z = (val - mean_val) / (std_val + 1e-6)
            
            # 사전 모델 피처 중요도를 탐색
            importance = self.feature_importances.get(col, 0.01)
            # 수치 편위 정도(절대값)와 변수 설명 기여 중요도의 곱을 융합
            contribution = abs(z) * importance
            
            sensor_name = col.split("_")[0] + "_" + col.split("_")[1] if "roll" in col else col
            anomalies.append({
                "sensor": sensor_name,
                "value": float(val),
                "z_score": float(z),
                "abs_z": float(abs(z)),
                "importance": float(importance),
                "contribution": float(contribution)
            })
            
        # 연산된 이상 센서 기여도 리스트를 내림차순 정렬 후 상위 3개 선별
        anomalies = sorted(anomalies, key=lambda x: x["contribution"], reverse=True)[:3]
        return {
            "unit": unit,
            "anomalies": anomalies
        }


class ActionRecommendationAgent:
    """
    진단 에이전트의 센서 이상 결과에 기초하여 엔진 부위별 정비 항목 및
    조치에 필요한 예상 시간을 매핑 및 제안하는 처방 에이전트.
    """
    # 21개 주요 계측 센서와 제트엔진 구성 부분(LPC, HPC, Fan 등) 및 세부 조치 가이드북
    GUIDELINES = {
        "s_2": {"part": "흡기 팬 (Fan)", "action": "팬 블레이드 균열 검사 및 베어링 윤활유 보충", "hours": 2.0},
        "s_3": {"part": "저압 압축기 (LPC)", "action": "저압 압축기 블레이드 세척 및 입구 안내 깃(IGV) 정렬 조정", "hours": 3.0},
        "s_4": {"part": "고압 압축기 (HPC)", "action": "HPC 스테이지 블레이드 점검 및 서지 마진 확인", "hours": 4.0},
        "s_7": {"part": "저압 터빈 (LPT)", "action": "LPT 블레이드 클리어런스 계측 및 유량 통로 세척", "hours": 2.5},
        "s_8": {"part": "팬 속도 (Fan Speed)", "action": "팬 로터 밸런싱 작업 및 회전 센서 라인 전압 측정", "hours": 1.5},
        "s_9": {"part": "바이패스 덕트", "action": "바이패스 덕트 내 외부 이물질(FOD) 제거 및 하우징 리벳 검사", "hours": 1.0},
        "s_11": {"part": "연소기 (Combustor)", "action": "연소기 연료 분사 노즐 카본 세척 및 연료 밸브 누설 검사", "hours": 3.5},
        "s_12": {"part": "고압 터빈 (HPT)", "action": "HPT 가스 가스켓 키트 점검 및 블레이드 열차폐 코팅 상태 육안 검사", "hours": 5.0},
        "s_13": {"part": "LPT 로터", "action": "LPT 샤프트 베어링 진동 센서 측정 및 오일 레벨 복원", "hours": 2.0},
        "s_14": {"part": "바이패스 비율", "action": "바이패스 댐퍼 밸브 공압식 액추에이터 작동 성능 점검", "hours": 1.5},
        "s_15": {"part": "HPC 오버홀", "action": "HPC 방출 밸브 오링 교체 및 누설 공기 압력 테스트", "hours": 3.0},
        "s_17": {"part": "고압 터빈 속도", "action": "HPT 로터 속도 트랜스듀서 단자부 청소 및 저항 계측", "hours": 1.5},
        "s_20": {"part": "터빈 출구 가스", "action": "터빈 하우징 열전대 센서 리드선 교체 및 계기 교정", "hours": 2.0},
        "s_21": {"part": "엔진 코어 압력", "action": "엔진 코어 배기 씰 마모 상태 검사 및 후방 프레임 지지대 균열 검출", "hours": 4.5},
    }

    def recommend(self, diagnose_result: dict) -> dict:
        """
        이상 원인 요약 정보를 받아 엔진 부품별 체크리스트 조치 목록과 
        총 예상 정비 수행 공임 시간(Man-Hours)을 빌드합니다.
        """
        anomalies = diagnose_result.get("anomalies", [])
        recommendations = []
        total_hours = 0.0
        
        for anomaly in anomalies:
            sensor = anomaly["sensor"]
            # 사전 설정 지침을 조회하고 없으면 범용 센서 커넥터 점검 가이드 배정
            guide = self.GUIDELINES.get(sensor, {
                "part": "엔진 공통 보조 시스템",
                "action": "센서 신호 모듈 점검 및 커넥터 클리닝",
                "hours": 1.0
            })
            recommendations.append({
                "sensor": sensor,
                "part": guide["part"],
                "action": guide["action"],
                "hours": guide["hours"],
                "deviation": "상승" if anomaly["z_score"] > 0 else "하락"
            })
            total_hours += guide["hours"]
        
        return {
            "unit": diagnose_result["unit"],
            "checklist": recommendations,
            "total_estimated_hours": round(total_hours, 1)
        }


class MaintenanceReportAgent:
    """
    모듈식으로 추천된 정비 가이드 체크리스트와 장비 계측 데이터를 융합해
    현장 서명 서식이 포함된 표준 마크다운(Markdown) 정비 요청 및 승인서를 자동 발행하는 리포트 에이전트.
    """
    def _generate_static_markdown(self, unit: int, cycle: int, predicted_rul: float, uncertainty: float, recommendations: dict, reason: str = "") -> str:
        """API 호출 실패 혹은 API Key 부재 시 활용할 Fallback 정적 마크다운 정비 요청 및 승인서 생성기."""
        checklist_md = ""
        for idx, item in enumerate(recommendations.get("checklist", []), 1):
            checklist_md += f"{idx}. **[{item['part']}]** {item['action']} (센서 {item['sensor']} {item['deviation']} 감지, 예상 소요 시간: {item['hours']}시간)\n"
        
        if not checklist_md:
            checklist_md = "1. **[엔진 전체]** 일반 예방정비 및 센서 오정렬 교정 작업 (예상 소요 시간: 1.0시간)\n"

        report = f"""# 🛠️ 정비 요청 및 승인서 (Maintenance Request & Approval)

## 1. 장비 기본 정보
* **대상 장비**: 제트 엔진 Unit #{unit}
* **현재 구동 시간**: {cycle} Cycles
* **정비 시점 예측 RUL**: {predicted_rul:.1f} Cycles (예측 오차 신뢰도 불확실성: ±{uncertainty:.1f})
* **문서 생성일**: 실시간 시뮬레이션 기반 자동 작성

## 2. 작업 사유 및 결재자 코멘트
> {reason or "현장 정비사의 AI 처방 승인에 따라 스케줄링됨."}

## 3. 세부 정비 조치 항목 (Recommended Action Checklist)
{checklist_md}
* **총 예상 정비 시간**: {recommendations.get('total_estimated_hours', 1.0)}시간

## 4. 점검 완료 후 서명
위 조치 사항을 완료하고 정상 작동(RUL 125로 리셋 완료)을 확인하였음을 보고합니다.

* **현장 책임자**: (서명) _________________
* **정비 일시**: (날짜 기입) 2026-07-08
"""
        return report.strip()

    def generate_markdown(self, unit: int, cycle: int, predicted_rul: float, uncertainty: float, recommendations: dict, reason: str = "", maintenance_count: int = 0) -> str:
        """
        3대 다중 에이전트(기술 서기, 재무 플래너, 품질 에디터) 협동 및 자가 검토(Reflection) 파이프라인과
        결정론적 파이썬 팩트체크 가드레일을 결합해 완벽한 정비 요청 및 승인서를 자동 작성합니다.
        """
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return self._generate_static_markdown(unit, cycle, predicted_rul, uncertainty, recommendations, reason)
        
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage
            
            # 1. 에이전트별 LLM 모델 설정 (품질 에디터는 온도를 0.0으로 하여 엄밀한 검증 강제)
            chat_writer = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=api_key)
            chat_editor = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, openai_api_key=api_key)
            
            # 피처 이상 상태 문자열 가공
            anomalies_info = ""
            sensors_to_check = []
            for item in recommendations.get("checklist", []):
                anomalies_info += f"- 센서 {item['sensor']} ({item['part']}) {item['deviation']} 편위 감지 (소요시간: {item['hours']}시간)\n"
                sensors_to_check.append(item['sensor'])
            
            if not anomalies_info:
                anomalies_info = "- 센서의 유의미한 규격 외 변위 미감지 (일반 예방 정비 대상)\n"
                
            # -------------------------------------------------------------
            # [단계 1] 기술 서기 에이전트 (Technical Writer Agent) 프롬프트 정의
            # -------------------------------------------------------------
            tech_prompt = f"""
당신은 항공 제트엔진 정비 오퍼레이션 총괄 엔지니어입니다.
아래의 엔진 텔레메트리 이상 진단 정보 및 정비 사유를 바탕으로 '기술 진단 및 안전 지침(Technical & Safety Section)' 초안을 한국어로 상세히 작성해 주십시오.

[장비 기본 정보]
* 대상 장비: 제트 엔진 Unit #{unit}
* 현재 구동 시간: {cycle} Cycles
* 누적 정비 횟수: {maintenance_count} 회
* 예측 잔존 수명(RUL): {predicted_rul:.1f} Cycles (예측 오차 불확실성: ±{uncertainty:.1f})
* 탐지된 센서 이상 요인:
{anomalies_info}
* 정비 사유: "{reason or '현장 정비사의 AI 처방 승인에 따라 스케줄링됨.'}"

[작성 요구사항 - 마크다운 형식으로 작성]:
1. '### 1. 기술 진단 및 기계공학적 원인 분석 (Root-Cause Analysis)' 단락:
   - 감지된 각 이상 센서들이 기계공학적으로 무엇을 의미하는지 전문적으로 설명하십시오. (예: s_11 연소기 온도, s_12 출구 압력 등)
2. '### 2. 정비 작업 단계별 상세 가이드 및 필수 안전수칙 (Action Checklist & Safety)' 단락:
   - 정비 작업 단계별 우선순위를 명시하십시오.
   - 고온/고압 작업 시 무조건 준수해야 하는 필수 안전 수칙(LOTO, 즉 Lockout-Tagout 수칙 등)과 준비해야 할 필수 자재를 명확히 기록하십시오.
3. '### 3. 장비 피로 누적 및 위험 경고 (Wear-Out Warning)' 단락:
   - 누적 정비 횟수({maintenance_count}회)에 따른 피로 누적 경고와 정비 팁을 서술하십시오.
"""
            
            # -------------------------------------------------------------
            # [단계 2] 재무/MRO 플래너 에이전트 (Financial Planner Agent) 프롬프트 정의
            # -------------------------------------------------------------
            repair_cost = 8000
            penalty = 50000
            savings = penalty - repair_cost
            roi = int((savings / repair_cost) * 100)
            
            fin_prompt = f"""
당신은 MRO 사업기획본부 재무/MRO 플래너 에이전트입니다.
아래의 엔진 MRO 가치 모델을 기반으로 경영진(Decision Maker)을 설득하기 위한 '정비 의사결정의 재무적 타당성 및 경영 요약 섹션(Business & Financial Section)' 초안을 한국어로 작성해 주십시오.

[비용 분석 수치]
* 정비 수행 시 비용: ${repair_cost:,} (약 1,100만 원) - 예방정비 자재비 및 인건비
* 미정비 시 고장 벌금 패널티: ${penalty:,} (약 6,900만 원) - 비행 결항, 긴급 회항 및 AOG 벌금
* 방어 순이익 (Net Savings): ${savings:,} (약 5,800만 원 방어)
* 투자 대비 효과 (ROI): {roi}% 달성

[작성 요구사항 - 마크다운 형식으로 작성]:
1. '### 4. 정비 의사결정의 재무적 타당성 (ROI Analysis)' 단락:
   - 위의 비용 수치(${repair_cost:,}, ${penalty:,}, {roi}%)를 빠짐없이 명시하여 이번 예방정비 승인이 회사의 불필요한 고장 손실금을 얼마나 효과적으로 방어하는지 재무적 관점으로 소명하십시오.
2. '### 5. 경영진 요약 및 정비 가용성 소명 (Executive Summary)' 단락:
   - 어려운 기계공학 용어를 배제하고 경영자가 한눈에 파악할 수 있도록 쉬운 언어로 요약하십시오.
   - 하루 가용 정비소 슬롯을 효율적으로 활용하여 일정을 계획했음을 덧붙이십시오.
"""

            # -------------------------------------------------------------
            # [단계 3] 품질 에디터 에이전트 및 자가 반성(Reflection) 검수 루프 기동
            # -------------------------------------------------------------
            tech_draft = ""
            fin_draft = ""
            
            max_iterations = 3
            iteration = 0
            feedback = ""
            
            while iteration < max_iterations:
                iteration += 1
                
                # 피드백이 있는 경우 수정을 강제하는 지침 결합
                if feedback:
                    tech_run_prompt = tech_prompt + f"\n\n[이전 검수 반려 피드백 - 다음 사항을 필히 보강하십시오]:\n{feedback}"
                    fin_run_prompt = fin_prompt + f"\n\n[이전 검수 반려 피드백 - 다음 사항을 필히 보강하십시오]:\n{feedback}"
                else:
                    tech_run_prompt = tech_prompt
                    fin_run_prompt = fin_prompt
                
                # 기술 서기와 재무 플래너가 각자의 보고서 섹션 생성
                tech_response = chat_writer.invoke([
                    SystemMessage(content="당신은 항공 제트엔진 정비 매뉴얼 및 기술 규정 작성 전문가입니다."),
                    HumanMessage(content=tech_run_prompt)
                ])
                tech_draft = tech_response.content.strip()
                
                fin_response = chat_writer.invoke([
                    SystemMessage(content="당신은 항공 MRO 투자 효율 및 사업성 평가 전문가입니다."),
                    HumanMessage(content=fin_run_prompt)
                ])
                fin_draft = fin_response.content.strip()
                
                # 에디터가 전체 초안 수집 및 정성적 검수 수행
                editor_evaluation_prompt = f"""
당신은 항공 정비 보고서 수석 품질 에디터이자 심사관 에이전트입니다.
기술 서기와 재무 플래너가 작성한 초안을 수집해 결합하고, 아래의 검수 조건에 합치하는지 엄밀히 검증하십시오.

[작성된 기술 섹션 초안]
{tech_draft}

[작성된 재무 섹션 초안]
{fin_draft}

[검수 필수 요건]
1. 기술 섹션에 이상 감지 센서 명칭({", ".join(sensors_to_check) if sensors_to_check else "일반 정비"})이 빠짐없이 언급되어 있는가?
2. 기술 섹션에 작업자 안전 수칙인 'LOTO' 혹은 'Lockout-Tagout' 단어가 명시되어 있는가?
3. 재무 섹션에 정비 비용 ${repair_cost:,}, 고장 패널티 ${penalty:,}, ROI {roi}%가 정확히 기재되어 있는가?

[출력 규칙]:
- 만약 누락되거나 내용이 잘못된 부분이 있다면, 반드시 'REJECT' 상태와 함께 어떤 내용이 부족하여 반려하는지 조목조목 피드백을 한국어로 적어주십시오.
- 모든 요구사항이 완벽하게 반영되었다면, 다른 부가 설명 없이 오직 'PASS' 단어 한 단어만 리턴하십시오.
"""
                editor_response = chat_editor.invoke([
                    SystemMessage(content="당신은 절대 허점을 타협하지 않는 까다로운 수석 편집인입니다. 가이드라인을 미준수하면 가차없이 REJECT 처리합니다."),
                    HumanMessage(content=editor_evaluation_prompt)
                ])
                editor_result = editor_response.content.strip()
                
                # -------------------------------------------------------------
                # [하이브리드 가드레일] 백엔드 파이썬 코드를 통한 결정론적 팩트 체크
                # -------------------------------------------------------------
                python_validation_passed = True
                python_feedback_items = []
                
                # 1. 이상 센서 키워드가 들어있는지 물리적으로 확인
                for s in sensors_to_check:
                    if s not in tech_draft:
                        python_validation_passed = False
                        python_feedback_items.append(f"- 기술 문서 내에 대상 이상 센서 명칭 '{s}'가 명확히 서술되어야 합니다.")
                
                # 2. LOTO 안전 수칙 필수 포함 체크
                if not any(word in tech_draft.upper() for word in ["LOTO", "LOCKOUT"]):
                    python_validation_passed = False
                    python_feedback_items.append("- 정비 가이드 내에 작업 안전 수칙인 'LOTO(Lockout-Tagout)' 지침 설명이 누락되었습니다.")
                
                # 3. 재무 단가 포함 체크
                if str(roi) not in fin_draft or str(repair_cost) not in fin_draft.replace(",", ""):
                    python_validation_passed = False
                    python_feedback_items.append(f"- 재무 분석 단락에 총 정비 비용(${repair_cost:,})과 분석 ROI({roi}%) 비율이 명확하게 작성되어야 합니다.")
                
                # 검사 통과 및 루프 탈출 조건 확인
                if "PASS" in editor_result.upper() and python_validation_passed:
                    break
                else:
                    llm_feedback = editor_result if "REJECT" in editor_result.upper() or "PASS" not in editor_result.upper() else ""
                    py_feedback = "\n".join(python_feedback_items)
                    feedback = f"### [검수 반려 사유 및 피드백 (이력 {iteration}회차)]\n{llm_feedback}\n{py_feedback}\n\n위의 요구사항을 다음 수정본 작성 시 반드시 적용하여 누락 부분을 보강해 주십시오."
                    # 로컬 디버그용 출력
                    print(f"[-] 품질 검수 반려 (이력 {iteration}회차). 피드백:\n{feedback}")
            
            # 최종 마크다운 서식으로 단락 합성
            final_report = f"""# 🛠️ 정비 요청 및 승인서 (Maintenance Request & Approval)

## 1. 장비 기본 정보
* **대상 장비**: 제트 엔진 Unit #{unit}
* **현재 구동 시간**: {cycle} Cycles
* **정비 시점 예측 RUL**: {predicted_rul:.1f} Cycles (예측 오차 신뢰도 불확실성: ±{uncertainty:.1f})
* **문서 생성일**: 실시간 시뮬레이션 기반 자동 작성

---

## 2. 작업 사유 및 결재자 코멘트
> {reason or "현장 정비사의 AI 처방 승인에 따라 스케줄링됨."}

---

## 3. 세부 정비 기술 지침 (Technical Section)
{tech_draft}

---

## 4. 비즈니스 ROI 및 경영진 요약 (Business Section)
{fin_draft}

---

## 5. 점검 완료 후 서명
위 조치 사항을 완료하고 정상 작동(RUL 125로 리셋 완료)을 확인하였음을 보고합니다.

* **현장 책임자**: (서명) _________________
* **정비 일시**: (날짜 기입) 2026-07-12
"""
            return final_report.strip()
            
        except Exception as exc:
            print(f"Warning: Failed to generate report using Multi-Agent Pipeline: {exc}. Falling back to static report.")
            return self._generate_static_markdown(unit, cycle, predicted_rul, uncertainty, recommendations, reason)
