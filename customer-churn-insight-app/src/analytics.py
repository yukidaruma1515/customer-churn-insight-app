"""Reusable evaluation and business-analysis functions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score


def threshold_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict:
    """Evaluate binary predictions made by applying an explicit probability threshold."""
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    cm = confusion_matrix(y_true, predicted, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {"precision": precision_score(y_true, predicted, zero_division=0), "recall": recall_score(y_true, predicted, zero_division=0), "f1": f1_score(y_true, predicted, zero_division=0), "confusion_matrix": cm, "high_risk_count": int(predicted.sum()), "captured_churners": int(tp), "missed_churners": int(fn), "tn": int(tn), "fp": int(fp)}


def threshold_curve(y_true: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    """Calculate precision and recall across operational thresholds."""
    return pd.DataFrame([{"threshold": t, **{k: v for k, v in threshold_metrics(y_true, probabilities, t).items() if k in {"precision", "recall", "f1"}}} for t in np.linspace(0.05, 0.95, 19)])


def lift_gain_table(y_true: pd.Series, probabilities: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Build cumulative gain/lift values from customers ranked by churn probability."""
    ranked = pd.DataFrame({"actual": np.asarray(y_true), "probability": probabilities}).sort_values("probability", ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["population_fraction"] = ranked["rank"] / len(ranked)
    total_positive = max(int(ranked["actual"].sum()), 1)
    ranked["cumulative_positive"] = ranked["actual"].cumsum()
    ranked["gain"] = ranked["cumulative_positive"] / total_positive
    ranked["precision"] = ranked["cumulative_positive"] / ranked["rank"]
    base_rate = ranked["actual"].mean()
    ranked["lift"] = ranked["precision"] / base_rate if base_rate else 0.0
    indices = sorted(set([max(0, int(np.ceil(len(ranked) * i / bins)) - 1) for i in range(1, bins + 1)]))
    return ranked.loc[indices, ["population_fraction", "gain", "precision", "lift", "rank", "cumulative_positive"]].reset_index(drop=True)


def top_k_metrics(y_true: pd.Series, probabilities: np.ndarray, fraction: float) -> dict[str, float]:
    """Return precision, recall and lift for the highest-risk population fraction."""
    ranked = pd.DataFrame({"actual": np.asarray(y_true), "probability": probabilities}).sort_values("probability", ascending=False)
    selected = ranked.head(max(1, int(np.ceil(len(ranked) * fraction))))
    precision = float(selected.actual.mean())
    recall = float(selected.actual.sum() / max(ranked.actual.sum(), 1))
    base_rate = float(ranked.actual.mean())
    return {"precision": precision, "recall": recall, "lift": precision / base_rate if base_rate else 0.0}


def calibration_analysis(y_true: pd.Series, probabilities: np.ndarray, bins: int = 10) -> tuple[pd.DataFrame, float]:
    """Return calibration points with bin counts and the Brier score."""
    prob_true, prob_pred = calibration_curve(y_true, probabilities, n_bins=bins, strategy="uniform")
    frame = pd.DataFrame({"actual_churn_rate": prob_true, "mean_predicted_probability": prob_pred})
    bucket = pd.cut(probabilities, bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    counts = pd.Series(bucket).value_counts(sort=False)
    frame["customers"] = counts[counts > 0].to_numpy()
    return frame, float(brier_score_loss(y_true, probabilities))


def segment_metrics(frame: pd.DataFrame, y_true: pd.Series, probabilities: np.ndarray, column: str, threshold: float) -> pd.DataFrame:
    """Compare threshold metrics for each value of a customer segment."""
    rows = []
    work = frame.reset_index(drop=True).copy()
    work["actual"] = np.asarray(y_true)
    work["probability"] = probabilities
    for value, group in work.groupby(column, observed=True, dropna=False):
        metrics = threshold_metrics(group["actual"], group["probability"].to_numpy(), threshold)
        rows.append({"segment": str(value), "customers": len(group), "precision": metrics["precision"], "recall": metrics["recall"], "f1": metrics["f1"]})
    return pd.DataFrame(rows).sort_values("recall")


def add_business_segments(data: pd.DataFrame) -> pd.DataFrame:
    """Add stable tenure and monthly-charge bands used by analyses."""
    result = data.copy()
    result["tenure_band"] = pd.cut(result["tenure"], [-1, 12, 24, 48, np.inf], labels=["0-12か月", "13-24か月", "25-48か月", "49か月以上"])
    result["MonthlyCharges_band"] = pd.cut(result["MonthlyCharges"], [-np.inf, 40, 70, 90, np.inf], labels=["~40", "40-70", "70-90", "90~"])
    return result


def category_churn_summary(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return customer count, churner count and churn rate by category."""
    return data.groupby(column, observed=True)["Churn"].agg(customers="size", churners="sum", churn_rate="mean").reset_index().sort_values("churn_rate", ascending=False)
