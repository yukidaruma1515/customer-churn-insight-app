from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src.preprocess import build_preprocessor, make_train_test_split

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


def build_models(preprocessor, random_state: int = 42) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=220,
                        min_samples_leaf=12,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    if LGBMClassifier is not None:
        models["LightGBM"] = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=240,
                        learning_rate=0.035,
                        num_leaves=24,
                        class_weight="balanced",
                        random_state=random_state,
                        verbosity=-1,
                    ),
                ),
            ]
        )
    return models


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "roc_curve": {"fpr": fpr, "tpr": tpr},
    }


def get_feature_names(model: Pipeline) -> list[str]:
    return list(model.named_steps["preprocessor"].get_feature_names_out())


def get_feature_importance(model: Pipeline) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    feature_names = [name.split("__")[-1] for name in get_feature_names(model)]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        values = np.zeros(len(feature_names))
    return (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def train_and_save(
    data: pd.DataFrame,
    model_path: str | Path,
    preprocessor_path: str | Path,
    metrics_path: str | Path,
    random_state: int = 42,
) -> dict[str, Any]:
    split = make_train_test_split(data, random_state=random_state)
    preprocessor = build_preprocessor(split.numeric_features, split.categorical_features)
    results = {}
    best_model = None
    best_name = ""
    best_score = -1.0

    for name, model in build_models(preprocessor, random_state=random_state).items():
        model.fit(split.X_train, split.y_train)
        metrics = evaluate_model(model, split.X_test, split.y_test)
        results[name] = metrics
        score = 0.50 * metrics["roc_auc"] + 0.30 * metrics["f1"] + 0.20 * metrics["recall"]
        if score > best_score:
            best_score = score
            best_name = name
            best_model = model

    model_path = Path(model_path)
    preprocessor_path = Path(preprocessor_path)
    metrics_path = Path(metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": best_model,
        "model_name": best_name,
        "metrics": results,
        "feature_importance": get_feature_importance(best_model),
        "feature_names": get_feature_names(best_model),
        "numeric_features": split.numeric_features,
        "categorical_features": split.categorical_features,
        "test_data": {"X_test": split.X_test, "y_test": split.y_test},
    }
    joblib.dump(payload, model_path)
    joblib.dump(best_model.named_steps["preprocessor"], preprocessor_path)

    metrics_rows = [
        {
            "model": name,
            "accuracy": values["accuracy"],
            "precision": values["precision"],
            "recall": values["recall"],
            "f1": values["f1"],
            "roc_auc": values["roc_auc"],
        }
        for name, values in results.items()
    ]
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    return payload
