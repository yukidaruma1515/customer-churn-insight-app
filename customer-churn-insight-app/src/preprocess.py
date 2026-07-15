from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_all: pd.DataFrame
    y_all: pd.Series
    numeric_features: list[str]
    categorical_features: list[str]


def clean_telco_data(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize numeric and target columns in a validated Telco data frame."""
    cleaned = data.copy()
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(cleaned["MonthlyCharges"] * cleaned["tenure"])
    if not set(cleaned["Churn"].dropna().unique()).issubset({0, 1}):
        cleaned["Churn"] = cleaned["Churn"].map({"Yes": 1, "No": 0})
    cleaned["Churn"] = cleaned["Churn"].astype(int)
    cleaned["SeniorCitizen"] = cleaned["SeniorCitizen"].astype(int)
    return cleaned


def split_features(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical model feature names."""
    feature_data = data.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore")
    numeric_features = feature_data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [column for column in feature_data.columns if column not in numeric_features]
    return numeric_features, categorical_features


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Build leakage-safe preprocessing for mixed Telco features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def make_train_test_split(data: pd.DataFrame, random_state: int = 42) -> SplitData:
    """Create a reproducible stratified 75/25 holdout split."""
    numeric_features, categorical_features = split_features(data)
    X = data.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore")
    y = data[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )
    return SplitData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_all=X,
        y_all=y,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def data_quality_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize missingness and dtypes for display."""
    return pd.DataFrame(
        {
            "column": data.columns,
            "missing_count": data.isna().sum().values,
            "missing_rate": data.isna().mean().values,
            "dtype": data.dtypes.astype(str).values,
        }
    )
