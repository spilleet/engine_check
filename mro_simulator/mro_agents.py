from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class AgentEvent:
    round: int
    agent: str
    message: str


class TelemetryStreamAgent:
    def __init__(self, latest_predictions: pd.DataFrame, horizon: int = 80) -> None:
        self.latest = latest_predictions.copy()
        self.horizon = horizon

    def frame(self, round_no: int, maintained_units: set[int]) -> pd.DataFrame:
        frame = self.latest.copy()
        frame["stream_cycle"] = frame["cycle"] + round_no - 1
        frame["rul"] = (frame["predicted_rul"] - round_no + 1).clip(lower=0)
        frame["true_remaining"] = (frame["true_rul"] - round_no + 1).clip(lower=0)
        frame["maintained"] = frame["unit"].isin(maintained_units)
        frame.loc[frame["maintained"], "rul"] = 125
        return frame


class CrisisDetectionAgent:
    def __init__(self, danger_threshold: float = 20.0, inspect_threshold: float = 50.0) -> None:
        self.danger_threshold = danger_threshold
        self.inspect_threshold = inspect_threshold

    def annotate(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["status"] = "healthy"
        out.loc[out["rul"] < self.inspect_threshold, "status"] = "inspect"
        out.loc[out["rul"] < self.danger_threshold, "status"] = "danger"
        out.loc[out["maintained"], "status"] = "maintained"
        
        # 정비 중 상태 추가
        if "under_maintenance" in out.columns:
            out.loc[out["under_maintenance"], "status"] = "under_maintenance"
            
        if "human_modifier" not in out.columns:
            out["human_modifier"] = 1.0
            
        base_risk = (
            (self.inspect_threshold - out["rul"]).clip(lower=0)
            + 0.35 * out["pred_uncertainty"]
            + 4.0 * (out["status"] == "danger").astype(float)
        )
        
        # 가드레일: 실제로 위험 수준에 이른 엔진은 보류 팩터가 있어도 최소 점수 유지
        true_rem = out["true_remaining"] if "true_remaining" in out.columns else out["rul"]
        guardrail = 20.0 * ((out["status"] == "danger") & (true_rem < 15)).astype(float)
        
        out["risk_score"] = base_risk * out["human_modifier"] + guardrail
        return out


class SituationQueryAgent:
    def summarize(self, frame: pd.DataFrame) -> dict[str, float]:
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
        active = frame.loc[~frame["maintained"]].sort_values(["risk_score", "rul"], ascending=[False, True])
        return active.head(limit)[["unit", "stream_cycle", "rul", "true_remaining", "pred_uncertainty", "status", "risk_score"]].to_dict(
            orient="records"
        )


class MaintenanceActionAgent:
    def __init__(self, slots_per_round: int = 3) -> None:
        self.slots_per_round = slots_per_round

    def choose_actions(self, frame: pd.DataFrame) -> pd.DataFrame:
        candidates = frame.loc[(~frame["maintained"]) & (frame["status"].isin(["danger", "inspect"]))].copy()
        ranked = candidates.sort_values(["risk_score", "rul", "pred_uncertainty"], ascending=[False, True, False])
        return ranked.head(self.slots_per_round)


class FleetAgentPipeline:
    def __init__(self, latest_predictions: pd.DataFrame, slots_per_round: int = 3, horizon: int = 80) -> None:
        self.stream = TelemetryStreamAgent(latest_predictions, horizon=horizon)
        self.detector = CrisisDetectionAgent()
        self.query = SituationQueryAgent()
        self.action = MaintenanceActionAgent(slots_per_round=slots_per_round)
        self.slots_per_round = slots_per_round
        self.horizon = horizon

    def run(self) -> dict:
        maintained_units: set[int] = set()
        protected_failures = 0
        missed_failures = 0
        cumulative_agent_cost = 0
        cumulative_baseline_cost = 0
        rounds = []
        events: list[AgentEvent] = [
            AgentEvent(1, "system", "시스템 초기화 완료. 함대 관제 에이전트가 데이터 스트림을 수신합니다."),
            AgentEvent(1, "objective", f"목표: 고장 회피와 정비 비용 최소화. 가용 슬롯: {self.slots_per_round}개/라운드."),
        ]

        for round_no in range(1, self.horizon + 1):
            frame = self.detector.annotate(self.stream.frame(round_no, maintained_units))
            summary = self.query.summarize(frame)
            top_risks = self.query.top_risks(frame)
            actions = self.action.choose_actions(frame)
            action_units = [int(unit) for unit in actions["unit"].tolist()]

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

            for unit in action_units:
                row = frame.loc[frame["unit"] == unit].iloc[0]
                if row["true_remaining"] <= row["rul"] + 8:
                    protected_failures += 1
                maintained_units.add(unit)
            cumulative_agent_cost += len(action_units) * 8000
            missed_now = int(((frame["true_remaining"] <= 0) & (~frame["maintained"])).sum())
            missed_failures += missed_now
            cumulative_baseline_cost += int((frame["rul"] < 30).sum()) * 8000 + missed_now * 50000

            if action_units:
                events.append(AgentEvent(round_no, "action_agent", f"정비 슬롯 배정: 엔진 {action_units}. 예상 비용 ${len(action_units) * 8000:,}."))
            else:
                events.append(AgentEvent(round_no, "action_agent", "정비 슬롯 배정 없음. 위험도 기준 미달."))

            visible = frame[["unit", "stream_cycle", "rul", "pred_uncertainty", "status", "maintained"]].copy()
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
    def __init__(self, feature_importances: dict[str, float] | None = None) -> None:
        self.feature_importances = feature_importances or {}

    def diagnose(self, frame: pd.DataFrame, unit: int) -> dict:
        target_row = frame.loc[frame["unit"] == unit]
        if target_row.empty:
            return {"unit": unit, "anomalies": []}
        
        target_row = target_row.iloc[0]
        
        # roll_mean_5가 붙은 컬럼 위주로 분석
        cols = [c for c in frame.columns if "roll_mean_5" in c]
        if not cols:
            cols = [c for c in frame.columns if c.startswith("s_") and not ("roll" in c)]
        
        anomalies = []
        for col in cols:
            active = frame.loc[~frame["maintained"]]
            if len(active) <= 1:
                active = frame
            mean_val = active[col].mean()
            std_val = active[col].std()
            
            val = target_row[col]
            z = (val - mean_val) / (std_val + 1e-6)
            
            # 피처 이름 매핑 (z-score 변위와 모델 중요도를 곱함)
            importance = self.feature_importances.get(col, 0.01)
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
            
        # Z-score 단독이 아닌, AI 기여도(contribution = z * importance) 기준 상위 3개 선별
        anomalies = sorted(anomalies, key=lambda x: x["contribution"], reverse=True)[:3]
        return {
            "unit": unit,
            "anomalies": anomalies
        }


class ActionRecommendationAgent:
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
        anomalies = diagnose_result.get("anomalies", [])
        recommendations = []
        total_hours = 0.0
        
        for anomaly in anomalies:
            sensor = anomaly["sensor"]
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
    def generate_markdown(self, unit: int, cycle: int, predicted_rul: float, uncertainty: float, recommendations: dict, reason: str = "") -> str:
        checklist_md = ""
        for idx, item in enumerate(recommendations.get("checklist", []), 1):
            checklist_md += f"{idx}. **[{item['part']}]** {item['action']} (센서 {item['sensor']} {item['deviation']} 감지, 예상 소요 시간: {item['hours']}시간)\n"
        
        if not checklist_md:
            checklist_md = "1. **[엔진 전체]** 일반 예방정비 및 센서 오정렬 교정 작업 (예상 소요 시간: 1.0시간)\n"

        report = f"""# 🛠️ 정비 작업 오더 완료 보고서 (Work Order Report)

## 1. 장비 기본 정보
* **대상 장비**: 제트 엔진 Unit #{unit}
* **현재 구동 시간**: {cycle} Cycles
* **정비 시점 예측 RUL**: {predicted_rul:.1f} Cycles (예측 오차 신뢰도 불확실성: ±{uncertainty:.1f})
* **보고서 생성일**: 실시간 시뮬레이션 기반 자동 작성

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
