"""Data loading and preprocessing utilities for Home Credit application data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"


def load_application_data(csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load `application_train.csv` and separate features from the target."""
    data = pd.read_csv(csv_path)
    if TARGET_COLUMN not in data:
        raise ValueError(f"Expected a '{TARGET_COLUMN}' column in {csv_path}.")

    features = data.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore")
    target = data[TARGET_COLUMN].astype("int8")
    return features, target


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create a train-only preprocessing pipeline without data leakage.

    Numeric values use median imputation and scaling. Categorical values use
    most-frequent imputation and one-hot encoding. `handle_unknown='ignore'`
    lets the pipeline safely score categories absent from the training split.
    """
    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
