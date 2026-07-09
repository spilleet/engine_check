# NASA C-MAPSS FD001 RUL 예측 및 함대 관제 결과

## 데이터

- 학습 데이터: 20,631행
- 테스트 데이터: 13,096행
- 테스트 엔진 수: 100대
- 사용 피처 수: 51개
- RUL 학습 라벨: 각 엔진의 마지막 사이클 기준 잔여 사이클, 최대 125로 cap 적용

## RUL 예측 모델

Random Forest 기반 `RULPredictorAgent`를 사용했습니다. 센서/운전 조건 원본 피처와 엔진별 rolling mean/std 피처를 함께 사용합니다.

| 평가 구간 | RMSE | MAE | R2 |
|---|---:|---:|---:|
| 엔진 그룹 검증 | 16.15 | 11.52 | 0.850 |
| 테스트 최신 시점 | 18.14 | 12.88 | 0.809 |

## 함대 관제 시뮬레이션

시뮬레이션 조건:

- 정비 슬롯: 하루 3대
- 기간: 160일
- critical 엔진 기준: 시작 시점 true RUL 30 이하
- 비교 정책:
  - `orchestrator`: 예측 RUL, 예측 불확실성, critical threshold를 반영한 위험 우선순위
  - `shortest_predicted_rul`: 예측 RUL이 가장 작은 엔진 우선
  - `oldest_cycle`: 관측 사이클이 가장 긴 엔진 우선
  - `random`: 무작위 우선순위

| 정책 | 고장 전 미정비 | 고장 전 정비 성공 | Critical 커버리지 |
|---|---:|---:|---:|
| orchestrator | 0 | 100 | 1.00 |
| shortest_predicted_rul | 0 | 100 | 1.00 |
| oldest_cycle | 3 | 97 | 0.88 |
| random | 10 | 90 | 0.60 |

## 해석

예측 RUL 기반 정책은 제한된 슬롯에서도 모든 엔진을 고장 전에 정비했습니다. 단순 노후 사이클 기준 정책은 3대, 무작위 정책은 10대가 정비 전 고장으로 처리되었습니다. 이 결과는 FD001처럼 운전 조건이 비교적 일정한 데이터에서도 센서 기반 RUL 예측이 정비 우선순위 결정에 직접적인 개선 효과를 줄 수 있음을 보여줍니다.

생성된 주요 파일:

- `metrics.json`: 전체 모델/시뮬레이션 지표
- `fleet_policy_comparison.csv`: 정책별 비교표
- `orchestrator_decisions.csv`: 정책별 일자/엔진 정비 결정 로그
- `test_predictions.csv`: 테스트 엔진별 실제 RUL, 예측 RUL, 예측 불확실성
- `top_risk_engines.csv`: 예측 RUL 기준 상위 위험 엔진 목록
- `dashboard.html`: 모델 성능, 정책 비교, 정비 순서를 한 화면에서 보는 HTML 대시보드
- `policy_comparison.png`: 정책 비교 시각화
- `rul_prediction_diagnostics.png`: 실제 RUL 대비 예측 RUL 및 오차 분포
- `maintenance_timeline.png`: 오케스트레이터 초기 정비 순서
- `rul_model.joblib`: 학습된 모델 artifact
