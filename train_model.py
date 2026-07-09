from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

# MRO 시뮬레이터 모듈 로드
from mro_simulator.data_loader import add_test_rul, add_train_rul, build_features, latest_rows, load_fd001, load_rul
from mro_simulator.fleet_engine import simulate_policy
from mro_simulator.benchmark_predictor import RULPredictorAgent, regression_metrics

# 프로젝트 루트 경로 정의
ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """명령줄 인수(Command-line Arguments) 파서 설정."""
    parser = argparse.ArgumentParser(description="Train FD001 RUL model and simulate fleet maintenance policies.")
    
    # NASA C-MAPSS FD001 기본 원본 파일들의 경로 기본값 지정
    parser.add_argument("--train", default="/Users/idong-ug/Downloads/pdm_agent/data/train_FD001.txt")
    parser.add_argument("--test", default="/Users/idong-ug/Downloads/pdm_agent/data/test_FD001.txt")
    parser.add_argument("--rul", default="/Users/idong-ug/Downloads/pdm_agent/data/RUL_FD001.txt")
    
    # 시뮬레이션 설정 파라미터 (하루 가용 정비 슬롯 수, 평가 기간, RUL 클리핑 상한치)
    parser.add_argument("--slots", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=160)
    parser.add_argument("--rul-cap", type=int, default=125)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # 보고서 이미지 및 모델 파일이 저장될 출력 디렉토리 경로 정의 및 생성
    report_dir = ROOT / "reports"
    artifact_dir = ROOT / "artifacts"
    report_dir.mkdir(exist_ok=True)
    artifact_dir.mkdir(exist_ok=True)

    # 1. 원본 C-MAPSS 텍스트 원천 데이터 파일 로드
    train_raw = load_fd001(args.train)
    test_raw = load_fd001(args.test)
    final_rul = load_rul(args.rul)

    # 2. RUL 잔존수명 타겟 라벨 부여 및 이동평균 롤링 피처 가공
    train_labeled = add_train_rul(train_raw, cap=args.rul_cap)
    test_labeled = add_test_rul(test_raw, final_rul)
    train_features, test_features, feature_cols = build_features(train_labeled, test_labeled)

    # 3. Random Forest 기반 RUL 예측기 에이전트 생성 및 훈련 수행
    agent = RULPredictorAgent()
    validation_metrics = agent.fit(train_features, feature_cols)

    # 4. 각 엔진의 최신 구동 시점(마지막 데이터행) 기준 RUL 예측 및 불확실성 산출
    latest_test = latest_rows(test_features)
    pred, uncertainty = agent.predict_with_uncertainty(latest_test)
    latest_test["predicted_rul"] = pred
    latest_test["pred_uncertainty"] = uncertainty
    
    # 검증 평가 지표 연산
    test_metrics = regression_metrics(latest_test["true_rul"], pred)

    # 5. 예측 결과를 CSV로 저장 및 학습 완료된 모델 객체 파일 저장 (.joblib)
    predictions = latest_test.copy()
    predictions.to_csv(report_dir / "test_predictions.csv", index=False)
    joblib.dump({"model": agent.model, "feature_columns": feature_cols}, artifact_dir / "rul_model.joblib")

    # 6. 각 정책별(AI 조율, 최단 RUL 우선, 최장 운행 우선, 무작위 배정) 정비 시뮬레이션 비교 수행
    policies = ["orchestrator", "shortest_predicted_rul", "oldest_cycle", "random"]
    results = [simulate_policy(latest_test, p, slots_per_day=args.slots, horizon=args.horizon) for p in policies]
    
    # 각 시뮬레이션의 평가지표 병합 및 저장
    comparison = pd.DataFrame([{"policy": r.policy, **r.metrics} for r in results])
    comparison.to_csv(report_dir / "fleet_policy_comparison.csv", index=False)
    
    # 오케스트레이터의 최종 의사결정 프로세스 로그 저장
    pd.concat([r.decisions for r in results], ignore_index=True).to_csv(report_dir / "orchestrator_decisions.csv", index=False)

    # 7. 전체 학습 요약 메트릭 생성 및 JSON 저장
    metrics = {
        "data": {
            "train_rows": int(len(train_raw)),
            "test_rows": int(len(test_raw)),
            "test_units": int(test_raw["unit"].nunique()),
            "features": int(len(feature_cols)),
        },
        "rul_model": {
            "validation": validation_metrics.as_dict(),
            "test_latest_cycle": test_metrics.as_dict(),
        },
        "fleet_simulation": {
            "slots_per_day": args.slots,
            "horizon": args.horizon,
            "policies": comparison.to_dict(orient="records"),
        },
    }
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # 8. 대시보드 리포팅을 위한 이미지 그래프 시각화 및 HTML 정적 페이지 내보내기
    make_policy_chart(comparison, report_dir / "policy_comparison.png")
    make_prediction_diagnostics(predictions, report_dir)
    make_maintenance_timeline(pd.concat([r.decisions for r in results], ignore_index=True), report_dir / "maintenance_timeline.png")
    write_dashboard(metrics, comparison, predictions, report_dir)
    
    # 표준 출력에 메트릭 정보 출력
    print(json.dumps(metrics, indent=2))


def make_policy_chart(comparison: pd.DataFrame, output_path: Path) -> None:
    """
    각 유지보수 정책별 '미보전 고장 건수'와 '위험 상태 예지 커버리지율'을 비교하는
    2열 막대 그래프 시각화 이미지를 생성합니다.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 1열: 정책별 조치 전 발생 고장 엔진 수 (낮을수록 우수)
    comparison.plot.bar(x="policy", y="failures_before_maintenance", ax=axes[0], legend=False, color="#b24745")
    axes[0].set_title("Failures before maintenance")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("engines")
    
    # 2열: 위험 임계 수명 내 예지 정비 대응률 (높을수록 우수)
    comparison.plot.bar(x="policy", y="critical_coverage_rate", ax=axes[1], legend=False, color="#2f7f6f")
    axes[1].set_title("Critical engine coverage")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("rate")
    axes[1].set_ylim(0, 1.05)
    
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_prediction_diagnostics(predictions: pd.DataFrame, report_dir: Path) -> None:
    """
    모델의 RUL 실제값 대 예측값 산점도(Scatter Plot) 및 RUL 오차값 분포 히스토그램을 그려
    모델의 잔존 수명 예측 진단 분석 이미지를 생성합니다.
    """
    scored = predictions.copy()
    scored["error"] = scored["predicted_rul"] - scored["true_rul"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # 1열: 실제값 대 예측값 분산 비교 산점도 (대각선에 밀집할수록 정확)
    axes[0].scatter(scored["true_rul"], scored["predicted_rul"], s=34, color="#2f6f9f", alpha=0.78)
    lo = min(scored["true_rul"].min(), scored["predicted_rul"].min())
    hi = max(scored["true_rul"].max(), scored["predicted_rul"].max())
    axes[0].plot([lo, hi], [lo, hi], color="#444444", linewidth=1, linestyle="--")
    axes[0].set_title("Actual vs predicted RUL")
    axes[0].set_xlabel("actual RUL")
    axes[0].set_ylabel("predicted RUL")

    # 2열: 예측 오차 잔차 분포 히스토그램 (0을 중심으로 좁게 정규분포를 그릴수록 정확)
    axes[1].hist(scored["error"], bins=18, color="#6c7a3d", edgecolor="white")
    axes[1].axvline(0, color="#444444", linewidth=1)
    axes[1].set_title("Prediction error distribution")
    axes[1].set_xlabel("predicted - actual")
    axes[1].set_ylabel("engines")

    fig.tight_layout()
    fig.savefig(report_dir / "rul_prediction_diagnostics.png", dpi=180)
    plt.close(fig)

    # 9. 오늘 시점 잔존 수명 위험 상위 15대 긴급 엔진 선별 저장
    high_risk = scored.sort_values(["predicted_rul", "pred_uncertainty"]).head(15)
    high_risk.to_csv(report_dir / "top_risk_engines.csv", index=False)


def make_maintenance_timeline(decisions: pd.DataFrame, output_path: Path) -> None:
    """
    오케스트레이터 정책에 따라 결정된 초기 45개 엔진의 입고 타임라인 일정을
    가로 바 차트 형태로 렌더링합니다.
    """
    orch = decisions.loc[decisions["policy"] == "orchestrator"].sort_values(["day", "unit"]).copy()
    first = orch.head(45)
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # 잔여 RUL이 임계값(30 사이클) 이하인 초긴급 엔진은 빨간색, 이외 대상은 녹색으로 산점 마킹
    colors = ["#b24745" if rul <= 30 else "#2f7f6f" for rul in first["true_rul_at_start"]]
    ax.barh(first["unit"].astype(str), first["day"], color="#d8d2c1")
    ax.scatter(first["day"], first["unit"].astype(str), s=55, c=colors)
    ax.invert_yaxis()
    ax.set_title("Orchestrator maintenance order, first 45 engines")
    ax.set_xlabel("maintenance day")
    ax.set_ylabel("engine unit")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_dashboard(
    metrics: dict,
    comparison: pd.DataFrame,
    predictions: pd.DataFrame,
    report_dir: Path,
) -> None:
    """
    학습 및 정책 시뮬레이션 지표를 요약하고 시각화 차트 이미지를 포함하는 
    독립형 정적 관제 대시보드 HTML 파일을 생성합니다.
    """
    top_risk = predictions.sort_values(["predicted_rul", "pred_uncertainty"]).head(10)
    
    # 위험 순위 테이블 행 동적 생성
    rows = "\n".join(
        "<tr>"
        f"<td>{int(row.unit)}</td>"
        f"<td>{int(row.cycle)}</td>"
        f"<td>{row.true_rul:.1f}</td>"
        f"<td>{row.predicted_rul:.1f}</td>"
        f"<td>{row.pred_uncertainty:.1f}</td>"
        "</tr>"
        for row in top_risk.itertuples()
    )
    
    # 정책 비교 지표 테이블 행 동적 생성
    policy_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row.policy)}</td>"
        f"<td>{row.failures_before_maintenance:.0f}</td>"
        f"<td>{row.maintained_before_failure:.0f}</td>"
        f"<td>{row.critical_coverage_rate:.2f}</td>"
        "</tr>"
        for row in comparison.itertuples()
    )
    
    validation = metrics["rul_model"]["validation"]
    test = metrics["rul_model"]["test_latest_cycle"]
    
    # HTML 문서 템플릿 정의
    page = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>C-MAPSS FD001 RUL Fleet Agent Dashboard</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #222; background: #f6f7f4; }}
    header {{ padding: 28px 36px 18px; background: #233142; color: white; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    main {{ padding: 24px 36px 40px; max-width: 1180px; margin: 0 auto; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-bottom: 22px; }}
    .metric, section {{ background: white; border: 1px solid #d9ddd2; border-radius: 8px; }}
    .metric {{ padding: 16px; }}
    .metric b {{ display: block; font-size: 24px; margin-top: 6px; }}
    section {{ padding: 18px; margin: 18px 0; }}
    h2 {{ margin: 0 0 14px; font-size: 18px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid #e5e8df; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #d9ddd2; border-radius: 6px; background: white; }}
    .two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    @media (max-width: 800px) {{ .grid, .two {{ grid-template-columns: 1fr; }} main, header {{ padding-left: 18px; padding-right: 18px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>NASA C-MAPSS FD001 RUL Fleet Agent Dashboard</h1>
    <div>RUL 예측 모델과 제한 슬롯 정비 오케스트레이터 결과</div>
  </header>
  <main>
    <div class="grid">
      <div class="metric">Train rows<b>{metrics["data"]["train_rows"]:,}</b></div>
      <div class="metric">Test engines<b>{metrics["data"]["test_units"]}</b></div>
      <div class="metric">Test RMSE<b>{test["rmse"]:.2f}</b></div>
      <div class="metric">Validation R2<b>{validation["r2"]:.3f}</b></div>
    </div>
    <section>
      <h2>정책 비교</h2>
      <table><thead><tr><th>policy</th><th>failures</th><th>maintained</th><th>critical coverage</th></tr></thead><tbody>{policy_rows}</tbody></table>
    </section>
    <div class="two">
      <section><h2>정책 성능</h2><img src="policy_comparison.png" alt="Policy comparison"></section>
      <section><h2>RUL 예측 진단</h2><img src="rul_prediction_diagnostics.png" alt="RUL prediction diagnostics"></section>
    </div>
    <section>
      <h2>오케스트레이터 정비 순서</h2>
      <img src="maintenance_timeline.png" alt="Maintenance timeline">
    </section>
    <section>
      <h2>상위 위험 엔진</h2>
      <table><thead><tr><th>unit</th><th>cycle</th><th>true RUL</th><th>predicted RUL</th><th>uncertainty</th></tr></thead><tbody>{rows}</tbody></table>
    </section>
  </main>
</body>
</html>
"""
    (report_dir / "dashboard.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
