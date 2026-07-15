"""Input validation for uploaded Telco churn CSV files."""
from __future__ import annotations

import pandas as pd

from src.load_data import TELCO_COLUMNS

NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
ALLOWED_CATEGORIES = {
    "gender": {"Female", "Male"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "PaymentMethod": {"Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"},
    "Churn": {"Yes", "No", 1, 0},
}


def validate_telco_data(data: pd.DataFrame, min_rows: int = 50) -> list[str]:
    """Return actionable validation errors; an empty list means valid input."""
    errors: list[str] = []
    missing = [column for column in TELCO_COLUMNS if column not in data.columns]
    if missing:
        return [f"必須カラムが不足しています: {', '.join(missing)}。列名を元データ仕様に合わせてください。"]
    if len(data) < min_rows:
        errors.append(f"データ件数が{len(data)}件です。安定した学習のため最低{min_rows}件を用意してください。")
    if data["customerID"].isna().any():
        errors.append("customerIDに欠損があります。顧客を一意に識別できる値を補完してください。")
    duplicate_count = int(data["customerID"].duplicated().sum())
    if duplicate_count:
        errors.append(f"customerIDの重複が{duplicate_count}件あります。重複行を確認・統合してください。")
    for column in NUMERIC_COLUMNS:
        converted = pd.to_numeric(data[column].replace(" ", pd.NA), errors="coerce")
        invalid = int(converted.isna().sum() - data[column].isna().sum())
        if invalid and column != "TotalCharges":
            errors.append(f"{column}に数値へ変換できない値が{invalid}件あります。数値形式へ修正してください。")
    for column, allowed in ALLOWED_CATEGORIES.items():
        invalid_values = set(data[column].dropna().unique()) - allowed
        if invalid_values:
            errors.append(f"{column}に不正な値があります: {sorted(map(str, invalid_values))}。許容値: {sorted(map(str, allowed))}")
    missing_counts = data.isna().sum()
    problematic = missing_counts[missing_counts > 0]
    if not problematic.empty:
        errors.append("欠損値があります: " + ", ".join(f"{col}={count}件" for col, count in problematic.items()) + "。補完または除外してください。")
    return errors
