import pandas as pd
import numpy as np

# MRO 에이전트 클래스 임포트
from mro_simulator.mro_agents import (
    MaintenanceDiagnosticianAgent,
    ActionRecommendationAgent,
    MaintenanceReportAgent,
)
from mro_simulator.alert_agent import SmartAlertAgent

def test_diagnose_and_recommend():
    """
    에이전트 파이프라인의 핵심인 진단(Diagnose), 처방 추천(Recommend), 
    그리고 작업 지시서 마크다운 생성(Report) 기능을 통합 검증하는 유닛 테스트.
    """
    # 1. 테스트용 임시 가상 데이터프레임 생성 (엔진 3대 분량)
    data = {
        "unit": [1, 2, 3],
        "maintained": [False, False, False],
        "cycle": [10, 10, 10],
        # 특정 센서 계측치의 시계열 이동평균 변동치 모사
        "s_2_roll_mean_5": [10.0, 10.2, 10.1],
        "s_11_roll_mean_5": [50.0, 50.5, 95.0],  # Unit 3의 s_11 센서값이 비정상적으로 높게 이탈
        "s_12_roll_mean_5": [30.0, 28.0, 29.0]
    }
    df = pd.DataFrame(data)

    # 2. Diagnostician 에이전트 검증 (Z-score 편위 분석 및 피처 중요도 결합)
    # 특정 피처의 가중치를 0.5 및 0.1로 사전 정의
    importances = {"s_11_roll_mean_5": 0.5, "s_2_roll_mean_5": 0.1}
    diagnostician = MaintenanceDiagnosticianAgent(feature_importances=importances)
    
    # 3번 엔진(Unit 3)에 대해 진단 수행
    diag_result = diagnostician.diagnose(df, unit=3)

    # 진단 단위가 Unit 3이고 검출된 센서 이상치 건수가 존재하는지 확인
    assert diag_result["unit"] == 3
    assert len(diag_result["anomalies"]) > 0
    # Z-score 편차 가중 평균이 가장 높은 s_11이 최고 기여 요인(첫 번째 인덱스)에 올라왔는지 검증
    assert diag_result["anomalies"][0]["sensor"] == "s_11"

    # 3. ActionRecommendation 에이전트 처방 매핑 검증
    recommender = ActionRecommendationAgent()
    rec_result = recommender.recommend(diag_result)

    # 처방 대상 유닛이 일치하고 조치 항목(checklist)이 채워져 있는지 확인
    assert rec_result["unit"] == 3
    assert len(rec_result["checklist"]) > 0
    # s_11 센서 이탈에 대해 연소기(Combustor) 계통 부품에 대한 세척/검사 처방이 생성되었는지 확인
    assert rec_result["checklist"][0]["sensor"] == "s_11"
    assert rec_result["checklist"][0]["part"] == "연소기 (Combustor)"

    # 4. MaintenanceReport 에이전트의 마크다운 작업 지시서 자동 생성 포맷 검증
    reporter = MaintenanceReportAgent()
    report = reporter.generate_markdown(
        unit=3,
        cycle=10,
        predicted_rul=25.5,
        uncertainty=2.1,
        recommendations=rec_result,
        reason="테스트 결재 사유"
    )

    # 지시서 마크다운 내에 장비 번호, 정비 필요 장비 부품명, 관제사 코멘트가 포함되어 있는지 검증
    assert "Unit #3" in report
    assert "연소기 (Combustor)" in report
    assert "테스트 결재 사유" in report

    # 5. SmartAlertAgent 기능 검증 (Fallback 동작 및 채널 판단 검증)
    alert_agent = SmartAlertAgent(api_key=None) # API Key 없음 -> Fallback 모드 강제 기동
    
    # 케이스 A: 단순 inspect(점검요망) 상태 발생 -> 대시보드 로그 전용 (외부 발신 생략)
    res_inspect = alert_agent.run_alert_logic(unit_id=3, tick=720, status="inspect", rul=45.0)
    assert "대시보드 로그 전용" in res_inspect
    
    # 케이스 B: 주간 근무 시간대 (12:00 -> 틱 720)이고 danger(위험) 상태 발생 -> 슬랙 알림 발신
    res_day_danger = alert_agent.run_alert_logic(unit_id=3, tick=720, status="danger", rul=18.0)
    assert "[Fallback]" in res_day_danger
    assert "슬랙" in res_day_danger
    
    # 케이스 C: 야간 시간대 (02:00 -> 틱 120)이고 danger 상태 발생, 다음 비행까지 여유 있음 -> 슬랙 알람 (전화 깨우지 않음)
    # Unit 3의 다음 비행 예정: 3*7+13%25+2 = 11사이클 (여유)
    res_night_safe = alert_agent.run_alert_logic(unit_id=3, tick=120, status="danger", rul=18.0)
    assert "[Fallback]" in res_night_safe
    assert "슬랙" in res_night_safe
    
    # 케이스 D: 야간 시간대 (02:00 -> 틱 120)이고 danger 상태 발생, 다음 비행 임박 -> 전화 호출 발신
    # Unit 9의 다음 비행 예정: 9*7+13%25+2 = 3사이클 (임박)
    res_night_urgent = alert_agent.run_alert_logic(unit_id=9, tick=120, status="danger", rul=15.0)
    assert "[Fallback]" in res_night_urgent
    assert "전화 호출" in res_night_urgent

if __name__ == "__main__":
    test_diagnose_and_recommend()
    print("All tests passed successfully!")
