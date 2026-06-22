from __future__ import annotations

import pandas as pd


def global_feature_importance(model_payload: dict) -> pd.DataFrame:
    return model_payload["feature_importance"].copy()


def shap_customer_explanation(model_payload: dict, background: pd.DataFrame, customer_features: pd.DataFrame) -> pd.DataFrame | None:
    try:
        import shap

        model = model_payload["model"]

        def predict_proba(values):
            frame = pd.DataFrame(values, columns=customer_features.columns)
            return model.predict_proba(frame)[:, 1]

        masker_data = background[customer_features.columns].head(100)
        explainer = shap.Explainer(predict_proba, masker_data)
        values = explainer(customer_features)
        shap_values = values.values[0]
        return (
            pd.DataFrame(
                {
                    "feature": customer_features.columns,
                    "value": customer_features.iloc[0].values,
                    "shap_value": shap_values,
                    "impact": abs(shap_values),
                }
            )
            .sort_values("impact", ascending=False)
            .head(8)
        )
    except Exception:
        return None


def rule_based_explanation(customer: pd.Series, feature_importance: pd.DataFrame) -> pd.DataFrame:
    risk_rules = {
        "Contract": customer.get("Contract") == "Month-to-month",
        "MonthlyCharges": customer.get("MonthlyCharges", 0) >= 75,
        "tenure": customer.get("tenure", 0) <= 12,
        "TechSupport": customer.get("TechSupport") == "No",
        "OnlineSecurity": customer.get("OnlineSecurity") == "No",
        "PaymentMethod": customer.get("PaymentMethod") == "Electronic check",
        "InternetService": customer.get("InternetService") == "Fiber optic",
    }
    rows = []
    for feature, active in risk_rules.items():
        if active:
            matched = feature_importance[feature_importance["feature"].str.contains(feature, regex=False)]
            importance = float(matched["importance"].max()) if not matched.empty else 0.0
            rows.append({"feature": feature, "value": customer.get(feature), "risk_reason": "離脱リスクを高める条件に該当", "importance": importance})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).head(8)
