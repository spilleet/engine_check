from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# MRO 시뮬레이터 에이전트 파이프라인 임포트
from mro_simulator.mro_agents import FleetAgentPipeline

# 프로젝트 루트 경로 정의
ROOT = Path(__file__).resolve().parent


def main() -> None:
    # 입력과 출력 경로 설정
    # predictions_path: 학습 모델 단계에서 오프라인 생성된 테스트 세트용 잔존 수명(RUL) 예측 파일
    # output_path: 대시보드 프론트엔드가 최초 화면 렌더링에 참조할 시뮬레이션 상태 스냅샷 파일
    predictions_path = ROOT / "reports" / "test_predictions.csv"
    output_path = ROOT / "ui" / "agent_state.json"
    
    # 출력 경로의 부모 폴더(ui)가 없으면 생성
    output_path.parent.mkdir(exist_ok=True)
    
    # 예측된 데이터셋 로드
    latest_predictions = pd.read_csv(predictions_path)
    
    # FleetAgentPipeline을 구성하여 80라운드 동안 매 라운드 3개의 정비 슬롯 제약하에 
    # 정비 배정 시뮬레이션을 가상으로 실행
    state = FleetAgentPipeline(latest_predictions, slots_per_round=3, horizon=80).run()
    
    # 시뮬레이션된 라운드별 에이전트 행정 로그 및 정비 비용 데이터를 JSON 포맷으로 저장
    output_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
