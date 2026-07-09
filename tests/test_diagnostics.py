import pandas as pd
import numpy as np
from mro_simulator.mro_agents import (
    MaintenanceDiagnosticianAgent,
    ActionRecommendationAgent,
    MaintenanceReportAgent,
)

def test_diagnose_and_recommend():
    # 1. 테스트용 가짜 데이터프레임 생성 (엔진 3대)
    data = {
        "unit": [1, 2, 3],
        "maintained": [False, False, False],
        "cycle": [10, 10, 10],
        # 변동이 큰 센서 데이터 모사
        "s_2_roll_mean_5": [10.0, 10.2, 10.1],
        "s_11_roll_mean_5": [50.0, 50.5, 95.0], # Unit 3에서 s_11이 비정상적으로 높음
        "s_12_roll_mean_5": [30.0, 28.0, 29.0]
    }
    df = pd.DataFrame(data)

    # 2. Diagnostician 검증 (중요도 가중치 결합)
    importances = {"s_11_roll_mean_5": 0.5, "s_2_roll_mean_5": 0.1}
    diagnostician = MaintenanceDiagnosticianAgent(feature_importances=importances)
    diag_result = diagnostician.diagnose(df, unit=3)

    assert diag_result["unit"] == 3
    assert len(diag_result["anomalies"]) > 0
    # z-score 편차가 가장 큰 s_11이 가장 상위에 올라와야 함
    assert diag_result["anomalies"][0]["sensor"] == "s_11"

    # 3. ActionRecommendation 검증
    recommender = ActionRecommendationAgent()
    rec_result = recommender.recommend(diag_result)

    assert rec_result["unit"] == 3
    assert len(rec_result["checklist"]) > 0
    # s_11에 대응하는 조치 내역 확인
    assert rec_result["checklist"][0]["sensor"] == "s_11"
    assert rec_result["checklist"][0]["part"] == "연소기 (Combustor)"

    # 4. Report 생성 검증
    reporter = MaintenanceReportAgent()
    report = reporter.generate_markdown(
        unit=3,
        cycle=10,
        predicted_rul=25.5,
        uncertainty=2.1,
        recommendations=rec_result,
        reason="테스트 결재 사유"
    )

    assert "Unit #3" in report
    assert "연소기 (Combustor)" in report
    assert "테스트 결재 사유" in report
