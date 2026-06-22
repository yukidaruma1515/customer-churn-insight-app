from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TELCO_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def generate_sample_telco(n_customers: int = 7043, random_state: int = 42) -> pd.DataFrame:
    """Generate Telco-like data so the app works before the Kaggle CSV is added."""
    rng = np.random.default_rng(random_state)

    tenure = rng.integers(0, 73, n_customers)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n_customers, p=[0.56, 0.21, 0.23])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n_customers, p=[0.34, 0.44, 0.22])
    senior = rng.binomial(1, 0.16, n_customers)
    paperless = rng.choice(["Yes", "No"], n_customers, p=[0.59, 0.41])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n_customers,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    base_charge = rng.normal(38, 11, n_customers)
    base_charge += np.where(internet == "Fiber optic", 38, 0)
    base_charge += np.where(internet == "DSL", 17, 0)
    monthly = np.clip(base_charge + rng.normal(0, 7, n_customers), 18, 119).round(2)
    total = np.where(tenure == 0, 0, monthly * tenure + rng.normal(0, 35, n_customers))
    total = np.clip(total, 0, None).round(2)

    tech_support = np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], n_customers, p=[0.34, 0.66]))
    online_security = np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], n_customers, p=[0.36, 0.64]))

    logit = (
        -1.35
        + 0.95 * (contract == "Month-to-month")
        - 0.85 * (contract == "Two year")
        + 0.55 * (internet == "Fiber optic")
        + 0.42 * (payment == "Electronic check")
        + 0.30 * (paperless == "Yes")
        + 0.38 * senior
        - 0.034 * tenure
        + 0.012 * (monthly - 65)
        + 0.38 * (tech_support == "No")
        + 0.34 * (online_security == "No")
    )
    churn_probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, churn_probability)

    customer_ids = [
        f"{rng.integers(1000, 9999)}-{''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 3))}"
        for _ in range(n_customers)
    ]

    data = pd.DataFrame(
        {
            "customerID": customer_ids,
            "gender": rng.choice(["Female", "Male"], n_customers),
            "SeniorCitizen": senior,
            "Partner": rng.choice(["Yes", "No"], n_customers, p=[0.48, 0.52]),
            "Dependents": rng.choice(["Yes", "No"], n_customers, p=[0.30, 0.70]),
            "tenure": tenure,
            "PhoneService": rng.choice(["Yes", "No"], n_customers, p=[0.90, 0.10]),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n_customers, p=[0.42, 0.48, 0.10]),
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], n_customers)),
            "DeviceProtection": np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], n_customers)),
            "TechSupport": tech_support,
            "StreamingTV": np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], n_customers)),
            "StreamingMovies": np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], n_customers)),
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total.astype(str),
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )
    return data[TELCO_COLUMNS]


def save_sample_telco(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = generate_sample_telco()
    data.to_csv(path, index=False)
    return data


def load_telco_data(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing = [column for column in TELCO_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"必要なカラムが不足しています: {missing}")
    return data[TELCO_COLUMNS].copy()


def ensure_telco_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        return load_telco_data(path)
    return save_sample_telco(path)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    save_sample_telco(root / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
