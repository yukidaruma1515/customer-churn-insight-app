"""Streamlit UI for churn prediction, evaluation and retention planning."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.analytics import (add_business_segments, calibration_analysis, category_churn_summary,
    lift_gain_table, segment_metrics, threshold_curve, threshold_metrics, top_k_metrics)
from src.explain import global_feature_importance, rule_based_explanation, shap_customer_explanation
from src.llm_report import build_report_context, generate_report
from src.load_data import ensure_telco_data
from src.predict import add_predictions
from src.preprocess import clean_telco_data, data_quality_summary
from src.recommend_actions import action_rules_table, action_summary, attach_recommendations
from src.train_model import train_and_save
from src.validation import validate_telco_data
from src.visualize import (plot_calibration, plot_churn_distribution, plot_confusion_matrix,
    plot_crosstab_heatmap, plot_feature_importance, plot_gain_lift, plot_numeric_by_churn,
    plot_roc_curves, plot_threshold_curve)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "churn_model.pkl"
PREPROCESSOR_PATH = ROOT / "models" / "preprocessor.pkl"
METRICS_PATH = ROOT / "outputs" / "model_metrics.csv"
st.set_page_config(page_title="顧客解約インサイト", page_icon="📊", layout="wide")


@st.cache_data(show_spinner=False)
def load_default_data() -> tuple[pd.DataFrame, str]:
    """Load repository CSV, or clearly mark generated fallback data."""
    existed = DATA_PATH.exists()
    return clean_telco_data(ensure_telco_data(DATA_PATH)), "CSVデータ" if existed else "動作確認用の合成データ"


@st.cache_data(show_spinner=False)
def load_uploaded_data(contents: bytes) -> pd.DataFrame:
    """Validate and clean user-uploaded CSV bytes."""
    import io
    raw = pd.read_csv(io.BytesIO(contents))
    errors = validate_telco_data(raw)
    if errors:
        raise ValueError("\n".join(f"・{error}" for error in errors))
    return clean_telco_data(raw)


@st.cache_resource(show_spinner=False)
def train_cached(data_digest: str, data: pd.DataFrame):
    """Train once per data content hash during the Streamlit process."""
    _ = data_digest
    return train_and_save(data, MODEL_PATH, PREPROCESSOR_PATH, METRICS_PATH)


def filter_high_risk(data: pd.DataFrame) -> pd.DataFrame:
    """Render high-risk filters and return the filtered frame."""
    result = data.copy()
    pmin, pmax = st.slider("離脱確率", 0.0, 1.0, (float(result.churn_probability.min()), 1.0), .01)
    result = result[result.churn_probability.between(pmin, pmax)]
    filters = [("リスクレベル", "risk_level"), ("契約形態", "Contract"), ("支払い方法", "PaymentMethod"), ("InternetService", "InternetService"), ("TechSupport", "TechSupport"), ("推奨施策", "recommended_action")]
    columns = st.columns(3)
    for index, (label, column) in enumerate(filters):
        options = sorted(result[column].dropna().astype(str).unique())
        selected = columns[index % 3].multiselect(label, options)
        if selected: result = result[result[column].astype(str).isin(selected)]
    tenure = st.slider("利用期間（か月）", 0, int(data.tenure.max()), (0, int(data.tenure.max())))
    charges = st.slider("月額料金", float(data.MonthlyCharges.min()), float(data.MonthlyCharges.max()), (float(data.MonthlyCharges.min()), float(data.MonthlyCharges.max())))
    return result[result.tenure.between(*tenure) & result.MonthlyCharges.between(*charges)]


st.title("📊 Customer Churn Insight App")
st.caption("予測 → 優先順位付け → 施策仮説 → A/Bテスト設計までを一つの画面で確認できます。")
with st.sidebar:
    st.header("分析設定")
    uploaded_file = st.file_uploader("Telco Customer Churn CSV", type=["csv"])
    risk_threshold = st.slider("High Risk閾値", .30, .90, .70, .01)
    remaining_months = st.slider("想定残存月数", 1, 60, 12)
    st.caption("優先対応スコア = 離脱確率 × 月額料金 × 想定残存月数")

try:
    if uploaded_file:
        raw_bytes = uploaded_file.getvalue(); data = load_uploaded_data(raw_bytes); data_source = "アップロードされたCSVデータ"
    else:
        data, data_source = load_default_data(); raw_bytes = data.to_csv(index=False).encode()
except Exception as exc:
    logger.exception("Data loading or validation failed")
    st.error(f"CSVを利用できません。以下を修正して再アップロードしてください。\n\n{exc}"); st.stop()

if "合成" in data_source:
    st.warning("現在は動作確認用の合成データです。結果は実際の通信会社の顧客傾向を示さず、企業判断に直接利用できません。")
else:
    st.success(f"使用データ: {data_source}（{len(data):,}件）")

with st.spinner("モデル評価と予測を準備しています（初回のみ5-fold CVを実行）..."):
    payload = train_cached(hashlib.sha256(raw_bytes).hexdigest(), data)
    enriched = attach_recommendations(add_predictions(data, payload))
    enriched["risk_level"] = pd.cut(enriched.churn_probability, [-.01, .40, risk_threshold, 1], labels=["Low", "Middle", "High"])
    enriched["priority_score"] = enriched.churn_probability * enriched.MonthlyCharges * remaining_months
    enriched = add_business_segments(enriched)
high_risk = enriched[enriched.churn_probability >= risk_threshold].copy()
test = payload["test_data"]; test_probs = payload["metrics"][payload["model_name"]]["y_proba"]
selected_threshold = threshold_metrics(test["y_test"], test_probs, risk_threshold)
importance = global_feature_importance(payload)

for col, label, value in zip(st.columns(5), ["顧客数", "解約率", "High Risk", "捕捉した解約者", "見逃した解約者"], [f"{len(data):,}", f"{data.Churn.mean():.1%}", f"{len(high_risk):,}", f"{selected_threshold['captured_churners']:,}", f"{selected_threshold['missed_churners']:,}"]): col.metric(label, value)

tabs = st.tabs(["概要・EDA", "モデル・閾値", "高度な分析", "セグメント・組合せ", "High Risk Customers", "説明・What-if", "LLMレポート", "施策・A/Bテスト"])
with tabs[0]:
    a,b=st.columns(2)
    with a: st.pyplot(plot_churn_distribution(data), width="stretch")
    with b: st.dataframe(data_quality_summary(data), width="stretch", hide_index=True)
    category=st.selectbox("カテゴリ特徴量", ["Contract","PaymentMethod","InternetService","TechSupport","OnlineSecurity","gender","PaperlessBilling"])
    summary=category_churn_summary(data, category); st.dataframe(summary, width="stretch", hide_index=True)
    st.caption("解約率だけでなく顧客数・解約者数を併記し、少人数カテゴリの過大評価を避けます。")
    st.bar_chart(summary.set_index(category)[["customers","churners"]]); st.bar_chart(summary.set_index(category)["churn_rate"])
    a,b=st.columns(2)
    with a: st.pyplot(plot_numeric_by_churn(data,"tenure"), width="stretch")
    with b: st.pyplot(plot_numeric_by_churn(data,"MonthlyCharges"), width="stretch")

with tabs[1]:
    rows=[]
    for name, values in payload["metrics"].items(): rows.append({"Model":name, **{metric:f"{values['cv'][metric+'_mean']:.3f} ± {values['cv'][metric+'_std']:.3f}" for metric in ["accuracy","precision","recall","f1","roc_auc","pr_auc"]}})
    cv_df=pd.DataFrame(rows); st.dataframe(cv_df, width="stretch", hide_index=True)
    st.download_button("5-fold CV結果CSV", cv_df.to_csv(index=False).encode("utf-8-sig"), "model_cv_metrics.csv", "text/csv")
    st.info("Stratified 5-fold cross-validationの平均 ± 標準偏差です。最終モデルはROC-AUC・F1・Recallの加重スコアで選択しています。")
    c1,c2,c3=st.columns(3); c1.metric("Precision",f"{selected_threshold['precision']:.3f}"); c2.metric("Recall",f"{selected_threshold['recall']:.3f}"); c3.metric("F1",f"{selected_threshold['f1']:.3f}")
    a,b=st.columns(2)
    with a: st.pyplot(plot_confusion_matrix(selected_threshold["confusion_matrix"]))
    with b: st.pyplot(plot_threshold_curve(threshold_curve(test["y_test"], test_probs)), width="stretch")
    st.pyplot(plot_roc_curves(payload["metrics"]), width="stretch")

with tabs[2]:
    gain=lift_gain_table(test["y_test"], test_probs)
    metric_cols=st.columns(6)
    for i,(fraction,label) in enumerate([(.1,"Top10%"),(.2,"Top20%")]):
        values=top_k_metrics(test["y_test"],test_probs,fraction)
        for j,key in enumerate(["precision","recall","lift"]): metric_cols[i*3+j].metric(f"{key.title()}@{label}",f"{values[key]:.2f}" if key=="lift" else f"{values[key]:.1%}")
    top30=top_k_metrics(test["y_test"],test_probs,.3); st.write(f"上位30%への対応で実際の解約者を **{top30['recall']:.1%}** 捕捉できます。")
    a,b=st.columns(2)
    with a: st.pyplot(plot_gain_lift(gain,"gain"),width="stretch")
    with b: st.pyplot(plot_gain_lift(gain,"lift"),width="stretch")
    calibration,brier=calibration_analysis(test["y_test"],test_probs); st.metric("Brier score（低いほど良い）",f"{brier:.4f}")
    a,b=st.columns(2)
    with a: st.pyplot(plot_calibration(calibration),width="stretch")
    with b: st.dataframe(calibration,width="stretch",hide_index=True)

with tabs[3]:
    test_segments=add_business_segments(test["X_test"])
    segment=st.selectbox("評価セグメント",["SeniorCitizen","gender","Contract","InternetService","tenure_band","MonthlyCharges_band"])
    seg=segment_metrics(test_segments,test["y_test"],test_probs,segment,risk_threshold); st.dataframe(seg,width="stretch",hide_index=True)
    low=seg[(seg.customers>=20)&(seg.recall<.40)]
    if not low.empty: st.warning("Recallが40%未満の層があります: "+", ".join(low.segment)+"。対象数と誤分類を確認してください。")
    pairs=[("Contract","tenure_band"),("Contract","MonthlyCharges_band"),("InternetService","MonthlyCharges_band"),("TechSupport","InternetService"),("PaymentMethod","PaperlessBilling")]
    pair=st.selectbox("特徴量の組み合わせ",pairs,format_func=lambda x:f"{x[0]} × {x[1]}"); st.pyplot(plot_crosstab_heatmap(enriched,*pair),width="stretch")

with tabs[4]:
    st.subheader("High Risk顧客のみを表示")
    order=st.radio("並び順",["離脱確率順","優先対応スコア順"],horizontal=True)
    filtered=filter_high_risk(high_risk).sort_values("churn_probability" if order=="離脱確率順" else "priority_score",ascending=False)
    st.caption("優先対応スコアは簡易的な期待損失指標であり、実際の顧客生涯価値（CLV）ではありません。")
    display=["customerID","churn_probability","risk_level","Contract","PaymentMethod","InternetService","TechSupport","tenure","MonthlyCharges","priority_score","recommended_action"]
    st.dataframe(filtered[display].head(100),width="stretch",hide_index=True); st.download_button("絞込結果CSV",filtered.to_csv(index=False).encode("utf-8-sig"),"high_risk_customers.csv","text/csv")

with tabs[5]:
    st.pyplot(plot_feature_importance(importance),width="stretch"); st.caption("SHAPや特徴量重要度は予測理由の説明であり、施策効果を証明するものではありません。")
    customer_id=st.selectbox("顧客",enriched.customerID.head(300).tolist()); customer=enriched[enriched.customerID==customer_id].iloc[0]
    base_features=customer.drop(labels=["customerID","Churn","churn_probability","risk_level","main_risk_factors","recommended_action","priority_score","tenure_band","MonthlyCharges_band"],errors="ignore")
    explanation=shap_customer_explanation(payload,data.drop(columns=["customerID","Churn"]),pd.DataFrame([base_features]))
    st.dataframe(explanation if explanation is not None else rule_based_explanation(customer,importance),width="stretch",hide_index=True)
    st.subheader("What-ifシミュレーション"); scenario=base_features.copy()
    scenario["Contract"]=st.selectbox("契約",[customer.Contract,"One year","Two year"])
    scenario["PaymentMethod"]=st.selectbox("支払い",[customer.PaymentMethod,"Bank transfer (automatic)","Credit card (automatic)"])
    scenario["TechSupport"]=st.selectbox("TechSupport",[customer.TechSupport,"Yes"]); scenario["OnlineSecurity"]=st.selectbox("OnlineSecurity",[customer.OnlineSecurity,"Yes"])
    scenario["MonthlyCharges"]=st.number_input("月額料金",0.0,200.0,float(customer.MonthlyCharges),1.0)
    new_prob=float(payload["model"].predict_proba(pd.DataFrame([scenario]))[0,1]); a,b,c=st.columns(3); a.metric("変更前",f"{customer.churn_probability:.1%}"); b.metric("変更後",f"{new_prob:.1%}"); c.metric("予測差",f"{new_prob-customer.churn_probability:+.1%}")
    st.warning("これは因果効果ではなく、学習済みモデル上の予測変化です。施策効果の確認にはA/Bテストが必要です。")

with tabs[6]:
    context=build_report_context(float(data.Churn.mean()),payload["model_name"],payload["metrics"][payload["model_name"]],importance.head(5).feature.tolist(),len(high_risk),[f"平均tenure: {high_risk.tenure.mean():.1f}",f"平均月額料金: {high_risk.MonthlyCharges.mean():.1f}"],action_summary(enriched))
    if st.button("レポートを生成",type="primary"):
        with st.spinner("生成中..."): st.session_state["llm_report"] = generate_report(context)
    if "llm_report" in st.session_state:
        report,is_template,error=st.session_state["llm_report"]
        if error: st.error(error)
        if is_template: st.info("APIキーがない、またはAPIを利用できないためテンプレートレポートを表示しています。")
        st.markdown(report)
    else: st.info("ボタンを押したときだけレポートを生成します。生成結果はセッション中保持されます。")

with tabs[7]:
    st.subheader("施策ルール（対象条件と根拠）"); st.dataframe(action_rules_table(),width="stretch",hide_index=True)
    st.subheader("A/Bテスト設計例")
    ab=pd.DataFrame([{"実験名":"月額契約High Risk層への長期契約特典","対象顧客":f"Month-to-month かつ予測確率{risk_threshold:.0%}以上","Treatment群":"1年/2年契約への移行割引またはポイント提示","Control群":"現行体験（特典提示なし）","主要KPI":"90日後の解約率","副次KPI":"契約転換率、継続月数、ARPU","Guardrail指標":"問い合わせ率、値引き原価、苦情率","推奨実験期間":"8〜12週間（事前に検出力分析）","仮説":"特典提示により解約率が低下する","成功条件":"統計的・実務的に有意な解約率低下、Guardrail悪化なし"}])
    st.dataframe(ab,width="stretch",hide_index=True); st.warning("相関関係は因果関係を意味しません。A/Bテストで確認するまで、予測モデルから施策の因果効果は断定できません。")
