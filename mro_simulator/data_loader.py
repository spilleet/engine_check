from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# C-MAPSS 데이터셋 스키마 정의
INDEX_COLUMNS = ["unit", "cycle"]  # 식별자 컬럼 (엔진 ID, 구동 사이클 수)
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]  # 운전 조건 제어 설정값 (3개)
SENSOR_COLUMNS = [f"s_{i}" for i in range(1, 22)]  # 센서 계측값 (21개)
ALL_COLUMNS = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS  # 전체 컬럼 목록


def load_fd001(path: str | Path) -> pd.DataFrame:
    """
    C-MAPSS 텍스트 데이터파일을 공백(white-spaces) 구분자로 파싱하여 로드합니다.
    엔진 식별자(unit) 및 구동 시간(cycle) 기준으로 오름차순 정렬 후 반환합니다.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, names=ALL_COLUMNS)
    return df.sort_values(["unit", "cycle"]).reset_index(drop=True)


def load_rul(path: str | Path) -> pd.Series:
    """기존 테스트 데이터셋의 정답(Ground Truth) 잔존 수명(RUL) 벡터파일을 로드합니다."""
    return pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0].rename("true_rul")


def add_train_rul(df: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    """
    학습용 데이터프레임에 RUL 타겟 라벨을 생성합니다.
    - 엔진 수명 모델링 시, 고장 시점에 인접할 때까지는 고장률이 선형적으로 늘어나지 않고 
      일정 수명 수준을 유지하는 특징이 있습니다. 이를 감안해 상한값(cap)으로 클리핑하는 piecewise 선형 모델을 취합니다.
    """
    out = df.copy()
    # 각 엔진(unit)별로 겪은 최대 수명(max cycle) 구하기
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    # 잔여 사이클을 계산한 뒤, 설정한 상한값(예: 125 사이클)으로 자름
    out["rul"] = (max_cycle - out["cycle"]).clip(upper=cap)
    return out


def add_test_rul(df: pd.DataFrame, final_rul: pd.Series) -> pd.DataFrame:
    """
    테스트용 데이터프레임에 RUL 정답 라벨을 주입합니다.
    - 테스트 셋은 고장 시점 직전이 아닌, 도중에 임의의 구간에서 데이터 계측이 중단됩니다.
    - 따라서 각 엔진의 마지막 계측 시점 잔존 수명(final_rul)에 현재 시점부터 마지막 시점까지의 차이를 더하여 복원합니다.
    """
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    # 각 엔진(unit)과 정답 RUL을 딕셔너리로 매핑
    final_map = dict(zip(sorted(out["unit"].unique()), final_rul.to_numpy()))
    out["final_rul"] = out["unit"].map(final_map)
    # 현재 사이클 기준 정답 RUL = 최종 정답 RUL + (해당 엔진의 최대 계측 사이클 - 현재 사이클)
    out["true_rul"] = out["final_rul"] + (max_cycle - out["cycle"])
    return out


def latest_rows(df: pd.DataFrame) -> pd.DataFrame:
    """각 엔진(unit)의 시계열 이력 중 가장 최근(마지막) 사이클의 데이터 행만 추출하여 반환합니다."""
    idx = df.groupby("unit")["cycle"].idxmax()
    return df.loc[idx].sort_values("unit").reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """
    전체 운전 설정값 및 센서 계측 컬럼 중, 데이터에 변동성이 없는 컬럼(값 종류가 1개뿐인 컬럼)을 제외하고
    모델 예측에 유의미한 변동 피처 목록만 선별합니다.
    """
    candidates = SETTING_COLUMNS + SENSOR_COLUMNS
    variable = [c for c in candidates if df[c].nunique(dropna=False) > 1]
    return variable


def add_rolling_features(df: pd.DataFrame, base_features: list[str], window: int = 5) -> pd.DataFrame:
    """
    시계열 트렌드 및 잡음 제거를 위해 주어진 윈도우 크기(window)만큼의 
    이동평균(rolling mean) 및 이동표준편차(rolling standard deviation) 피처를 생성합니다.
    """
    out = df.sort_values(["unit", "cycle"]).copy()
    grouped = out.groupby("unit", group_keys=False)
    for col in base_features:
        # 이동평균 피처 생성 (초기 데이터 유지를 위해 min_periods=1 적용)
        out[f"{col}_roll_mean_{window}"] = grouped[col].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        # 이동표준편차 피처 생성 (초기 최소 2개 샘플 필요, 부족할 시 fillna(0.0) 처리)
        out[f"{col}_roll_std_{window}"] = (
            grouped[col].rolling(window, min_periods=2).std().reset_index(level=0, drop=True).fillna(0.0)
        )
    return out


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame, window: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    학습 및 테스트 셋에 대해 유효 피처를 선별하고 이동평균/표준편차를 포함한 전체 피처 데이터프레임과 피처 이름 목록을 구성합니다.
    """
    base = feature_columns(train_df)
    train_features = add_rolling_features(train_df, base, window=window)
    test_features = add_rolling_features(test_df, base, window=window)
    
    # 훈련에 사용될 기본 피처 + 평균 피처 + 표준편차 피처 조합
    cols = base + [f"{c}_roll_mean_{window}" for c in base] + [f"{c}_roll_std_{window}" for c in base]
    # 무한값(Infinite)이나 NaN이 없는 유효 데이터 컬럼만 최종 추출
    cols = [c for c in cols if np.isfinite(train_features[c]).all()]
    return train_features, test_features, cols
