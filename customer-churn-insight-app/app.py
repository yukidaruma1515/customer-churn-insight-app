from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.explain import global_feature_importance, rule_based_explanation, shap_customer_explanation
from src.llm_report import build_report_context, generate_report
from src.load_data import ensure_telco_data
from src.predict import add_predictions, save_high_risk_customers
from src.preprocess import clean_telco_data, data_quality_summary
from src.recommend_actions import action_summary, attach_recommendations
from src.train_model import train_and_save
from src.visualize import (
    plot_churn_distribution,
    plot_churn_rate_by_category,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_numeric_by_churn,
    plot_roc_curves,
)


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "churn_model.pkl"
PREPROCESSOR_PATH = ROOT / "models" / "preprocessor.pkl"
METRICS_PATH = ROOT / "outputs" / "model_metrics.csv"
HIGH_RISK_PATH = ROOT / "outputs" / "high_risk_customers.csv"


st.set_page_config(page_title="Customer Churn Insight App", layout="wide")


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    return clean_telco_data(ensure_telco_data(DATA_PATH))


@st.cache_data(show_spinner=False)
def load_uploaded_data(uploaded_file) -> pd.DataFrame:
    return clean_telco_data(pd.read_csv(uploaded_file))


@st.cache_resource(show_spinner=False)
def train_cached(data_hash: int, data: pd.DataFrame):
    _ = data_hash
    return train_and_save(data, MODEL_PATH, PREPROCESSOR_PATH, METRICS_PATH)


st.title("Customer Churn Insight App")
st.caption("顧客離脱リスクを予測し、要因可視化、施策提案、改善レポート生成までつなげるStreamlitアプリです。")

with st.sidebar:
    st.header("Data")
    uploaded_file = st.file_uploader("Telco Customer Churn CSVをアップロード", type=["csv"])
    risk_threshold = st.slider("High risk threshold", 0.50, 0.90, 0.70, 0.05)
    st.caption("APIキーがない場合、LLMレポートはテンプレートで生成されます。")

try:
    data = load_uploaded_data(uploaded_file) if uploaded_file is not None else load_default_data()
except Exception as exc:
    st.error(f"データを読み込めませんでした: {exc}")
    st.stop()

with st.spinner("モデルを学習し、離脱確率を計算しています..."):
    data_hash = hash(pd.util.hash_pandas_object(data, index=True).sum())
    model_payload = train_cached(data_hash, data)
    predicted = add_predictions(data, model_payload)
    predicted["risk_level"] = pd.cut(
        predicted["churn_probability"],
        bins=[-0.01, 0.40, risk_threshold, 1.0],
        labels=["Low", "Middle", "High"],
    )
    enriched = attach_recommendations(predicted)
    save_high_risk_customers(enriched, HIGH_RISK_PATH)

best_metrics = model_payload["metrics"][model_payload["model_name"]]
feature_importance = global_feature_importance(model_payload)
high_risk = enriched[enriched["risk_level"].astype(str) == "High"]
recommended_actions = action_summary(enriched)

overview_cols = st.columns(5)
overview_cols[0].metric("Customers", f"{len(data):,}")
overview_cols[1].metric("Columns", f"{len(data.columns):,}")
overview_cols[2].metric("Churn Rate", f"{data['Churn'].mean():.1%}")
overview_cols[3].metric("High Risk", f"{len(high_risk):,}")
overview_cols[4].metric("Best Model", model_payload["model_name"])

tabs = st.tabs(
    [
        "Overview",
        "EDA",
        "Modeling",
        "Explainability",
        "High Risk Customers",
        "LLM Report",
        "Future Work",
    ]
)

with tabs[0]:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("アプリ概要")
        st.write("Telco Customer Churnデータを使い、顧客ごとの離脱確率、離脱要因、推奨施策を表示します。")
        st.subheader("データ品質")
        st.dataframe(data_quality_summary(data), width="stretch", hide_index=True)
    with right:
        st.subheader("Churn分布")
        st.pyplot(plot_churn_distribution(data), width="stretch")
        st.write(data["Churn"].map({1: "Yes", 0: "No"}).value_counts().rename_axis("Churn").reset_index(name="count"))

    st.subheader("数値特徴量の基本統計量")
    st.dataframe(data.select_dtypes(include="number").describe().T, width="stretch")

with tabs[1]:
    st.subheader("カテゴリ特徴量の分布")
    category = st.selectbox("カテゴリ特徴量", ["Contract", "PaymentMethod", "InternetService", "TechSupport", "OnlineSecurity"])
    cols = st.columns([1, 1])
    with cols[0]:
        st.pyplot(plot_churn_rate_by_category(data, "Contract"), width="stretch")
    with cols[1]:
        st.pyplot(plot_churn_rate_by_category(data, "PaymentMethod"), width="stretch")
    cols2 = st.columns([1, 1])
    with cols2[0]:
        st.pyplot(plot_numeric_by_churn(data, "tenure"), width="stretch")
    with cols2[1]:
        st.pyplot(plot_numeric_by_churn(data, "MonthlyCharges"), width="stretch")
    st.write(data[category].value_counts().rename_axis(category).reset_index(name="count"))

with tabs[2]:
    st.subheader("モデル比較")
    metrics_rows = []
    for name, metrics in model_payload["metrics"].items():
        metrics_rows.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
            }
        )
    metrics_df = pd.DataFrame(metrics_rows).sort_values(["roc_auc", "f1"], ascending=False)
    st.dataframe(metrics_df, width="stretch", hide_index=True)
    st.info("離脱顧客を見逃さないため、AccuracyだけでなくRecall、F1-score、ROC-AUCを重視しています。")
    model_name = st.selectbox("混同行列を表示するモデル", list(model_payload["metrics"].keys()))
    cols = st.columns([0.8, 1.2])
    with cols[0]:
        st.pyplot(plot_confusion_matrix(model_payload["metrics"][model_name]["confusion_matrix"]))
    with cols[1]:
        st.pyplot(plot_roc_curves(model_payload["metrics"]), width="stretch")

with tabs[3]:
    st.subheader("全体の特徴量重要度")
    st.pyplot(plot_feature_importance(feature_importance), width="stretch")
    st.dataframe(feature_importance.head(20), width="stretch", hide_index=True)

    st.subheader("顧客ごとの離脱要因")
    selected_customer = st.selectbox("顧客を選択", enriched["customerID"].head(200).tolist())
    customer = enriched[enriched["customerID"] == selected_customer].iloc[0]
    customer_features = pd.DataFrame([customer.drop(labels=["customerID", "Churn", "churn_probability", "risk_level", "main_risk_factors", "recommended_action"], errors="ignore")])
    shap_result = shap_customer_explanation(model_payload, data.drop(columns=["customerID", "Churn"], errors="ignore"), customer_features)
    if shap_result is not None:
        st.write("SHAPによる個別説明")
        st.dataframe(shap_result, width="stretch", hide_index=True)
    else:
        st.write("SHAPが利用できないため、特徴量重要度とルールベースで代替説明します。")
        st.dataframe(rule_based_explanation(customer, feature_importance), width="stretch", hide_index=True)

with tabs[4]:
    st.subheader("高リスク顧客一覧")
    display_columns = ["customerID", "churn_probability", "risk_level", "main_risk_factors", "recommended_action"]
    st.dataframe(enriched[display_columns].head(100), width="stretch", hide_index=True)
    st.download_button(
        "high_risk_customers.csvをダウンロード",
        high_risk.to_csv(index=False).encode("utf-8"),
        file_name="high_risk_customers.csv",
        mime="text/csv",
    )

with tabs[5]:
    st.subheader("PM向け改善レポート")
    high_risk_trends = [
        f"High Risk顧客数: {len(high_risk):,}",
        f"High Riskの平均tenure: {high_risk['tenure'].mean():.1f}",
        f"High Riskの平均MonthlyCharges: {high_risk['MonthlyCharges'].mean():.1f}",
    ]
    report_context = build_report_context(
        churn_rate=float(data["Churn"].mean()),
        model_name=model_payload["model_name"],
        best_metrics=best_metrics,
        top_features=feature_importance.head(5)["feature"].tolist(),
        high_risk_count=len(high_risk),
        high_risk_trends=high_risk_trends,
        recommended_actions=recommended_actions,
    )
    if st.button("レポートを生成"):
        st.markdown(generate_report(report_context))
    else:
        st.markdown(generate_report(report_context))

with tabs[6]:
    st.subheader("Future Work: Uplift Modelingへの発展")
    st.markdown(
        """
- 現在のアプリでは、離脱予測とルールベースの施策提案を行っています。
- Telco Customer ChurnデータにはTreatment列がないため、厳密なUplift Modelingは実装していません。
- 厳密なUplift Modelingには、キャンペーン配信有無、クーポン提示有無、サポート提案有無などの施策ログが必要です。
- 今後はTreatment列を追加し、T-LearnerやS-Learnerで施策あり・なしの継続確率差を推定します。
- 実サービスでは、接触履歴、問い合わせ履歴、通信品質、NPS、キャンペーン反応などを追加したいです。
"""
    )
