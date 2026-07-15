from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import auc


def plot_churn_distribution(data: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    counts = data["Churn"].map({1: "Yes", 0: "No"}).value_counts()
    ax.bar(counts.index, counts.values, color=["#d95f59", "#4f86c6"])
    ax.set_title("Churn Distribution")
    ax.set_ylabel("Customers")
    fig.tight_layout()
    return fig


def plot_churn_rate_by_category(data: pd.DataFrame, column: str):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    rates = data.groupby(column, observed=True)["Churn"].mean().sort_values(ascending=False)
    ax.bar(rates.index.astype(str), rates.values, color="#6a994e")
    ax.set_title(f"Churn Rate by {column}")
    ax.set_ylabel("Churn rate")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_numeric_by_churn(data: pd.DataFrame, column: str):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for churn_value, group in data.groupby("Churn"):
        label = "Churn Yes" if churn_value == 1 else "Churn No"
        ax.hist(group[column].dropna(), bins=35, alpha=0.55, label=label)
    ax.set_title(f"{column} Distribution by Churn")
    ax.set_xlabel(column)
    ax.set_ylabel("Customers")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred: No", "Pred: Yes"])
    ax.set_yticks([0, 1], labels=["Actual: No", "Actual: Yes"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=12)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_roc_curves(metrics: dict):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for model_name, values in metrics.items():
        fpr = values["roc_curve"]["fpr"]
        tpr = values["roc_curve"]["tpr"]
        ax.plot(fpr, tpr, label=f"{model_name} AUC={auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_feature_importance(importance: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    top = importance.head(15).sort_values("importance", ascending=True)
    ax.barh(top["feature"], top["importance"], color="#3a7ca5")
    ax.set_title("Top Feature Importance")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_threshold_curve(curve: pd.DataFrame):
    """Plot precision/recall/F1 against the decision threshold."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for metric in ["precision", "recall", "f1"]:
        ax.plot(curve["threshold"], curve[metric], marker="o", label=metric.title())
    ax.set(xlabel="Threshold", ylabel="Score", ylim=(0, 1), title="Precision / Recall by Threshold")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); return fig


def plot_gain_lift(table: pd.DataFrame, metric: str):
    """Plot cumulative gain or lift by targeted population share."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(table["population_fraction"], table[metric], marker="o", label=metric.title())
    if metric == "gain": ax.plot([0, 1], [0, 1], "--", color="#777", label="Random")
    else: ax.axhline(1, linestyle="--", color="#777", label="Random")
    ax.set(xlabel="Targeted population", ylabel=metric.title(), title=f"Cumulative {metric.title()} Chart")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); return fig


def plot_calibration(table: pd.DataFrame):
    """Plot observed churn rate against average predicted probability."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(table["mean_predicted_probability"], table["actual_churn_rate"], marker="o", label="Model")
    ax.plot([0, 1], [0, 1], "--", color="#777", label="Perfect")
    ax.set(xlabel="Mean predicted probability", ylabel="Actual churn rate", title="Calibration Curve", xlim=(0,1), ylim=(0,1))
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); return fig


def plot_crosstab_heatmap(data: pd.DataFrame, row: str, column: str):
    """Plot churn rate and customer counts for a pair of categorical segments."""
    rates = pd.pivot_table(data, index=row, columns=column, values="Churn", aggfunc="mean", observed=True)
    counts = pd.crosstab(data[row], data[column])
    fig, ax = plt.subplots(figsize=(8, 4.5)); image = ax.imshow(rates, cmap="YlOrRd", vmin=0, vmax=max(.5, float(np.nanmax(rates))))
    ax.set_xticks(range(len(rates.columns)), labels=rates.columns.astype(str), rotation=25, ha="right"); ax.set_yticks(range(len(rates.index)), labels=rates.index.astype(str))
    for i in range(len(rates.index)):
        for j in range(len(rates.columns)):
            ax.text(j, i, f"{rates.iloc[i,j]:.1%}\nn={counts.reindex(index=rates.index, columns=rates.columns, fill_value=0).iloc[i,j]}", ha="center", va="center", fontsize=8)
    ax.set(title=f"{row} × {column}: Churn rate / count", xlabel=column, ylabel=row); fig.colorbar(image, ax=ax, label="Churn rate"); fig.tight_layout(); return fig
