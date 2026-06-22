from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def load_model(model_path: str | Path) -> dict[str, Any]:
    return joblib.load(model_path)


def add_predictions(data: pd.DataFrame, model_payload: dict[str, Any]) -> pd.DataFrame:
    model = model_payload["model"]
    features = data.drop(columns=["customerID", "Churn"], errors="ignore")
    predicted = data.copy()
    predicted["churn_probability"] = model.predict_proba(features)[:, 1]
    predicted["risk_level"] = pd.cut(
        predicted["churn_probability"],
        bins=[-0.01, 0.40, 0.70, 1.0],
        labels=["Low", "Middle", "High"],
    )
    return predicted.sort_values("churn_probability", ascending=False).reset_index(drop=True)


def save_high_risk_customers(predicted: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predicted[predicted["risk_level"].astype(str) == "High"].to_csv(output_path, index=False)
