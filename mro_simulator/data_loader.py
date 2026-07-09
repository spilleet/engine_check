from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INDEX_COLUMNS = ["unit", "cycle"]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"s_{i}" for i in range(1, 22)]
ALL_COLUMNS = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS


def load_fd001(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=ALL_COLUMNS)
    return df.sort_values(["unit", "cycle"]).reset_index(drop=True)


def load_rul(path: str | Path) -> pd.Series:
    return pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0].rename("true_rul")


def add_train_rul(df: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = (max_cycle - out["cycle"]).clip(upper=cap)
    return out


def add_test_rul(df: pd.DataFrame, final_rul: pd.Series) -> pd.DataFrame:
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    final_map = dict(zip(sorted(out["unit"].unique()), final_rul.to_numpy()))
    out["final_rul"] = out["unit"].map(final_map)
    out["true_rul"] = out["final_rul"] + (max_cycle - out["cycle"])
    return out


def latest_rows(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("unit")["cycle"].idxmax()
    return df.loc[idx].sort_values("unit").reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    candidates = SETTING_COLUMNS + SENSOR_COLUMNS
    variable = [c for c in candidates if df[c].nunique(dropna=False) > 1]
    return variable


def add_rolling_features(df: pd.DataFrame, base_features: list[str], window: int = 5) -> pd.DataFrame:
    out = df.sort_values(["unit", "cycle"]).copy()
    grouped = out.groupby("unit", group_keys=False)
    for col in base_features:
        out[f"{col}_roll_mean_{window}"] = grouped[col].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        out[f"{col}_roll_std_{window}"] = (
            grouped[col].rolling(window, min_periods=2).std().reset_index(level=0, drop=True).fillna(0.0)
        )
    return out


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame, window: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    base = feature_columns(train_df)
    train_features = add_rolling_features(train_df, base, window=window)
    test_features = add_rolling_features(test_df, base, window=window)
    cols = base + [f"{c}_roll_mean_{window}" for c in base] + [f"{c}_roll_std_{window}" for c in base]
    cols = [c for c in cols if np.isfinite(train_features[c]).all()]
    return train_features, test_features, cols
