from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit


@dataclass
class RULMetrics:
    """
    RUL(잔존 수명) 예측 모델의 평가 지표를 정의하는 데이터 클래스.
    """
    rmse: float  # 평균 제곱근 오차 (Root Mean Squared Error)
    mae: float   # 평균 절대 오차 (Mean Absolute Error)
    r2: float    # 결정 계수 (R-squared Score)

    def as_dict(self) -> dict[str, float]:
        """지표 객체를 딕셔너리 형태로 반환합니다."""
        return {"rmse": self.rmse, "mae": self.mae, "r2": self.r2}


class RULPredictorAgent:
    """
    RandomForestRegressor를 기반으로 가스 터빈 제트 엔진의 RUL을 예측하는 에이전트 클래스.
    학습 및 개별 예측, 앙상블 불확실성 산출을 수행합니다.
    """
    def __init__(self, random_state: int = 42) -> None:
        # 무작위성을 제어하여 결과를 재현할 수 있도록 난수 시드를 설정한 랜덤 포레스트 회귀 분석 모델 정의
        self.model = RandomForestRegressor(
            n_estimators=250,      # 생성할 의사결정 나무 개수
            min_samples_leaf=3,    # 리프 노드가 되기 위한 최소 샘플 개수 (과적합 방지)
            max_features="sqrt",   # 최적의 분할을 찾기 위해 고려할 무작위 피처 개수 (총 피처 수의 제곱근)
            n_jobs=-1,             # 연산 시 사용 가능한 모든 CPU 코어 활용
            random_state=random_state,
        )
        self.feature_columns: list[str] = []  # 모델 학습 시 사용된 독립 변수(피처) 목록

    def fit(self, df: pd.DataFrame, feature_columns: list[str]) -> RULMetrics:
        """
        주어진 데이터를 사용하여 모델을 학습하고 검증 세트에 대한 평가 결과를 반환합니다.
        가시적인 엔진 단위(unit)가 학습/검증 세트 간에 겹치지 않도록 그룹 분할을 수행합니다.
        """
        self.feature_columns = feature_columns
        groups = df["unit"].to_numpy()  # 엔진 단위를 그룹 키로 사용
        
        # 동일 엔진의 시계열 데이터가 학습과 검증 데이터에 혼재되어 정보가 유출되는 것을 막기 위한 그룹 분할 객체
        split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, valid_idx = next(split.split(df, groups=groups))

        # 데이터 세트 추출
        x_train = df.iloc[train_idx][feature_columns]
        y_train = df.iloc[train_idx]["rul"]
        x_valid = df.iloc[valid_idx][feature_columns]
        y_valid = df.iloc[valid_idx]["rul"]

        # 모델 학습
        self.model.fit(x_train, y_train)
        
        # 검증 세트 예측 수행 및 평가 지표 산출
        pred = self.model.predict(x_valid)
        return regression_metrics(y_valid, pred)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """주어진 입력 데이터프레임에 대하여 모델이 예측한 RUL 값을 반환합니다. 음수 값은 0.0으로 클리핑합니다."""
        return np.maximum(0.0, self.model.predict(df[self.feature_columns]))

    def predict_with_uncertainty(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        랜덤 포레스트 내의 모든 개별 의사결정 나무(estimators)의 예측 편차를 기반으로,
        예측 RUL의 평균값과 함께 모델의 예측 불확실성(표준 편차)을 산출합니다.
        """
        x = df[self.feature_columns].to_numpy()
        # 모든 의사결정 나무의 개별 예측값을 매트릭스로 병합
        tree_preds = np.vstack([tree.predict(x) for tree in self.model.estimators_])
        
        # 앙상블 평균 예측값 (RUL은 0 미만이 될 수 없으므로 최소 0.0 보장)
        mean = np.maximum(0.0, tree_preds.mean(axis=0))
        # 예측치들의 표준 편차를 구하여 예측의 불확실성 지표로 사용
        std = tree_preds.std(axis=0)
        return mean, std


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> RULMetrics:
    """실제 RUL 값과 예측 RUL 값을 비교하여 RMSE, MAE, R2 평가 지표를 계산합니다."""
    mse = mean_squared_error(y_true, y_pred)
    return RULMetrics(
        rmse=float(np.sqrt(mse)),
        mae=float(mean_absolute_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
    )
