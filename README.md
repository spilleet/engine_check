# ✈️ ENGINE CHECK AGENT: 제트엔진 예지보전 & HITL MRO 관제 플랫폼

본 프로젝트는 NASA C-MAPSS 가스 터빈 제트엔진 데이터셋을 활용하여 **기계학습 기반의 잔존 수명(RUL) 예측**을 수행하고, 이를 **다중 에이전트(Multi-Agent)와 인간 결재(HITL, Human-In-The-Loop) 프로세스**와 결합하여 제한된 정비 자원 속에서 예방정비 스케줄링을 최적화하는 실시간 관제 플랫폼입니다.

---

## 🌟 핵심 기능 (Core Features)

### 1. 🤖 머신러닝 기반 RUL 예지 및 불확실성 산출
* **앙상블 예측**: Random Forest Regressor 앙상블 모델을 학습시켜 실시간 가동 제트엔진의 잔존 수명(RUL)을 예측합니다.
* **불확실성($\sigma$) 실시간 연산**: 개별 Decision Tree 예측치들의 표준편차를 기반으로 모델의 예측 **불확실성(Uncertainty)**을 도출하여 의사결정의 신뢰도를 제공합니다.
* **구간 선형 모델(Piecewise Linear) 학습**: 조기 마모 데이터 노이즈 오인 방지를 위해 RUL 상한선을 **125 사이클**로 클리핑하여 예측 정확도(RMSE)를 최적화했습니다.

### 2. 👥 8대 에이전트 기반 MRO 협동 파이프라인
* **TelemetryStreamAgent**: 엔진의 비행 사이클 증가 및 RUL 차감, 정비 복원을 관리하는 시간 제어 장치.
* **CrisisDetectionAgent**: RUL 임계치(Danger: 20, Inspect: 50) 기준 등급 분류 및 리스크 점수(`risk_score`) 연산.
* **SituationQueryAgent**: 함대 건강 요약 통계 및 실시간 상위 위험군 큐(Critical Queue) 도출.
* **MaintenanceActionAgent**: 하루 정비소 수용 한계(슬롯 3개) 내에서 가장 시급한 엔진 우선 추천.
* **MaintenanceDiagnosticianAgent**: 데이터 통계 변위($Z$-score)와 모델 피처 중요도를 결합해 이상 징후 발생 센서(TOP 3) 정밀 진단.
* **ActionRecommendationAgent**: 진단 부품별 조치 지침 및 예상 소요 공임(Man-Hours) 가이드 매핑.
* **MaintenanceReportAgent**: OpenAI API 연동을 통한 지능형 정비 작업 지시서 작성.
* **SmartAlertAgent**: 위기 등급에 따른 현장 알람(슬랙/TTS 전화) 전송 여부 및 발신 수단을 자율 판단하는 가드 에이전트.

### 3. ✍️ OpenAI LLM 기반 지능형 정비 작업 지시서 (Work Order)
* **GPT-4o-mini 실시간 연동**: 상급자 최종 결재 승인 시, 실무자가 즉시 작업에 투입될 수 있는 고품질 한글 작업 지시서를 실시간 생성합니다.
* **실무 맞춤형 지침 구성**:
  * **이상 센서의 기계공학적 원인(Root-Cause)** 해설
  * 작업 소요 시간 단축을 위한 **점검 우선순위(Priority)** 가이드
  * 현장 정비 시 구비해야 할 **준비 자재** 목록
  * 고온/고압 환경에서의 필수 안전 규칙인 **LOTO(Lockout-Tagout) 및 잔압 제거 수칙**
  * 엔진 누적 정비 횟수에 따른 **장비 피로 마모 주의보**
* **자가 안정성(Fallback) 설계**: API 키 미설정 또는 네트워크 차단 시 시스템 마비를 방지하기 위해 정적 가이드라인 템플릿으로 자동 우회 구동됩니다.

### 4. 👥 인간 개입 기반 2단계 결재 워크플로우 (HITL)
* **실무자 모드 (Technician)**: AI의 정비 권고안을 모니터링하고, 현장 소견을 기록하여 **1차 작업지서 상신(Approve)** 또는 **관제 보류(Defer, 리스크 경보 50% 완화)**를 입력합니다.
* **상급자 모드 (Supervisor)**: 상신된 작업지서와 AI의 예비 지침서 초안을 검토 후 **최종 승인(Approve Final)** 또는 **반려(Reject)** 처리합니다.
* **타이머 연동**: 최종 승인이 떨어지는 순간에만 실제 정비소 입고 타이머(3틱)가 돌입하며, 서명이 기재된 작업 지시서가 로컬 마크다운 파일로 아카이빙 저장됩니다.

### 5. ⚡ 실시간 SSE 스트림 반응형 다크모드 대시보드
* **Server-Sent Events(SSE)**: 폴링 없이 1초마다 실시간으로 100대 제트엔진의 텔레메트리 스트림 데이터 수신.
* **그리드 뷰 시각화**: 엔진의 실시간 건강 등급을 색상 카드(건강: 녹색, 점검: 황색, 위험: 적색, 결재대기: 주황)로 표기하며, 정비 중인 엔진은 보라색 펄스 애니메이션이 활성화됩니다.
* **실시간 통계 및 차트**: 전체 함대 요약 통계 배지, 위험도 정렬 뷰, 그리고 현재 클릭한 엔진의 각 센서별 변동 추이($Z$-score) 차트를 실시간으로 갱신 렌더링합니다.

### 6. 🚨 지능형 알람 필터링 및 피로 방지 에이전트 (SmartAlertAgent)
* **알람 피로도(Alert Fatigue) 방어**: 잦은 일반적인 경고(`inspect` 상태) 시에는 현업 작업자 호출(슬랙/전화)을 완전히 차단하고, 오직 웹 대시보드 로그에만 조용히 노출합니다.
* **자율 채널 선정 (LangChain ReAct)**: 치명적인 고장 징후(`danger` 상태) 감지 시에만 가상 비행 스케줄 및 시각 정보를 자율 분석하여 전달을 분기합니다.
  * **주간 근무 및 시간 여유**: 공식 **슬랙 채널(`trigger_emergency_slack`)** 전송 및 기록 보존.
  * **야간 근무 & 비행 임박 (< 5 사이클)**: 잠든 정비사를 물리적으로 깨우기 위한 **자동 TTS 비상 유선 전화 호출(`trigger_tts_voice_call`)** 연결.

### 7. 📊 4대 정비 정책 비교 시뮬레이션
* 제약된 슬롯 하에서 어떤 정책이 정비 비용과 고장률을 최소화하는지 비교 분석을 지원합니다.
  1. `orchestrator` (AI 종합 리스크 기반 배정 - 제안 방식)
  2. `shortest_predicted_rul` (단순 잔여 수명 우선)
  3. `oldest_cycle` (가장 오래 비행한 기체 우선)
  4. `random` (무작위 수리)

---

## 📁 프로젝트 구조 (Project Structure)

```directory
engine-check-dashboard/
│
├── mro_simulator/              # MRO 에이전트 및 시뮬레이션 핵심 로직
│   ├── data_loader.py          # 시계열 이동평균 등 롤링 피처 전처리
│   ├── benchmark_predictor.py  # Random Forest 기반 RUL 예측 및 불확실성 연산
│   ├── fleet_engine.py         # 정책 시뮬레이터 (오프라인 정책 비교)
│   ├── mro_agents.py           # 7대 역할별 MRO 에이전트 파이프라인
│   └── alert_agent.py          # 지능형 알람 필터링 및 비상 채널 제어 에이전트
│
├── ui/                         # 관제 콘솔 웹 프론트엔드 자원
│   ├── index.html              # 메인 대시보드 마크업
│   ├── styles.css              # 다크모드 그리드 디자인 및 모달 스타일
│   ├── app.js                  # SSE 이벤트 수신 및 REST API 제어 스크립트
│   └── agent_state.json        # 오프라인 구동 스냅샷 데이터
│
├── tests/                      # 테스트 스크립트
│   └── test_diagnostics.py     # 유닛 및 통합 테스트
│
├── reports/                    # 분석 산출물 및 생성된 작업지시서 아카이브
│   ├── submitted_orders/       # 최종 결재 완료된 마크다운 작업지시서 (.md)
│   └── policy_comparison.png   # 정책 성과 비교 차트
│
├── artifacts/                  # 학습 완료된 모델 파라미터 파일 (.joblib)
├── .env                        # OpenAI API Key 설정 파일(본인의 api키를 넣을 것)
└── run_server.py               # 실시간 웹 API 및 이벤트 스트림 서버 구동기
```
#워크플로우
<img width="664" height="1550" alt="image" src="https://github.com/user-attachments/assets/7398abbd-12ba-40b7-ad4f-095a73da4bce" />

#에이전트플로우
<img width="1706" height="498" alt="image" src="https://github.com/user-attachments/assets/9622bfc6-99e5-4266-8d40-d33e49eec555" />

---

## 🚀 구동 및 실행 방법 (How to Run)

### 0. 환경 구성 및 의존 라이브러리 설치
```bash
pip install -r requirements.txt
```
* API 호출 키를 사용하려면 프로젝트 루트에 `.env` 파일을 만들고 키를 등록해 주십시오:
  `OPENAI_API_KEY="sk-proj-..."`

### 1. 예측 모델 학습 및 오프라인 시뮬레이션 비교 실행
```bash
python3 train_model.py
```
* Random Forest 모델을 훈련하여 저장하고, 정책 비교 차트(`reports/policy_comparison.png`) 및 분석 데이터(`reports/fleet_policy_comparison.csv`)를 빌드합니다.

### 2. 실시간 웹 관제 대시보드 서버 기동
```bash
python3 run_server.py
```
* 대시보드 서버를 켠 뒤, 크롬 등의 브라우저를 열고 다음 주소에 접속합니다:
  * 👉 **[http://127.0.0.1:8765](http://127.0.0.1:8765)**
* `실시간 스트림 시작`을 누르면 비행 관제가 동작하며 결재 테스트를 시작할 수 있습니다.

### 3. 에이전트 로직 유닛 테스트 실행
```bash
PYTHONPATH=. python3 tests/test_diagnostics.py
```

---
