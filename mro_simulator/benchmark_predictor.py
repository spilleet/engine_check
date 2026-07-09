from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit


@dataclass
class RULMetrics:
    rmse: float
    mae: float
    r2: float

    def as_dict(self) -> dict[str, float]:
        return {"rmse": self.rmse, "mae": self.mae, "r2": self.r2}


class RULPredictorAgent:
    def __init__(self, random_state: int = 42) -> None:
        self.model = RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        )
        self.feature_columns: list[str] = []

    def fit(self, df: pd.DataFrame, feature_columns: list[str]) -> RULMetrics:
        self.feature_columns = feature_columns
        groups = df["unit"].to_numpy()
        split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, valid_idx = next(split.split(df, groups=groups))

        x_train = df.iloc[train_idx][feature_columns]
        y_train = df.iloc[train_idx]["rul"]
        x_valid = df.iloc[valid_idx][feature_columns]
        y_valid = df.iloc[valid_idx]["rul"]

        self.model.fit(x_train, y_train)
        pred = self.model.predict(x_valid)
        return regression_metrics(y_valid, pred)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.maximum(0.0, self.model.predict(df[self.feature_columns]))

    def predict_with_uncertainty(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = df[self.feature_columns].to_numpy()
        tree_preds = np.vstack([tree.predict(x) for tree in self.model.estimators_])
        mean = np.maximum(0.0, tree_preds.mean(axis=0))
        std = tree_preds.std(axis=0)
        return mean, std


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> RULMetrics:
    mse = mean_squared_error(y_true, y_pred)
    return RULMetrics(
        rmse=float(np.sqrt(mse)),
        mae=float(mean_absolute_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
    )
