from __future__ import annotations

import pandas as pd


def identify_risk_factors(customer: pd.Series) -> list[str]:
    factors = []
    if customer.get("Contract") == "Month-to-month":
        factors.append("月額契約で解約しやすい")
    if customer.get("MonthlyCharges", 0) >= 75:
        factors.append("月額料金が高い")
    if customer.get("tenure", 0) <= 12:
        factors.append("利用期間が短い")
    if customer.get("TechSupport") == "No":
        factors.append("TechSupport未加入")
    if customer.get("OnlineSecurity") == "No":
        factors.append("OnlineSecurity未加入")
    if customer.get("PaymentMethod") == "Electronic check":
        factors.append("Electronic check利用")
    if customer.get("InternetService") == "Fiber optic":
        factors.append("Fiber optic利用で料金・品質不満の可能性")
    return factors[:5] or ["複数要因による総合的な離脱リスク"]


def recommend_actions(customer: pd.Series) -> list[str]:
    if customer.get("churn_probability", 0) < 0.40:
        return ["通常フォロー: 利用状況を継続モニタリングする"]

    actions = []
    if customer.get("Contract") == "Month-to-month":
        actions.append("長期契約特典: 1年契約への移行割引やポイント付与を提案する")
    if customer.get("MonthlyCharges", 0) >= 75:
        actions.append("料金プラン見直し: 利用状況に合う低負担プランを提案する")
    if customer.get("tenure", 0) <= 12:
        actions.append("オンボーディング強化: 初期活用サポートや利用定着メールを送る")
    if customer.get("TechSupport") == "No":
        actions.append("サポート加入キャンペーン: 初月無料のTechSupportを提案する")
    if customer.get("OnlineSecurity") == "No":
        actions.append("セキュリティ提案: OnlineSecurityの価値を説明し利用を促す")
    if customer.get("PaymentMethod") == "Electronic check":
        actions.append("支払い方法変更: 自動支払いへの変更特典を提案する")
    if customer.get("InternetService") == "Fiber optic":
        actions.append("満足度確認: 通信品質・料金への不満をヒアリングする")
    return actions[:3] or ["個別フォロー: 利用状況を確認し、解約理由をヒアリングする"]


def attach_recommendations(predicted: pd.DataFrame) -> pd.DataFrame:
    enriched = predicted.copy()
    enriched["main_risk_factors"] = enriched.apply(lambda row: " / ".join(identify_risk_factors(row)), axis=1)
    enriched["recommended_action"] = enriched.apply(lambda row: " / ".join(recommend_actions(row)), axis=1)
    return enriched


def action_summary(enriched: pd.DataFrame) -> list[str]:
    high_risk = enriched[enriched["risk_level"].astype(str) == "High"]
    actions = []
    if (high_risk["Contract"] == "Month-to-month").mean() > 0.5:
        actions.append("高リスク層では月額契約が多いため、長期契約特典のA/Bテストを優先する")
    if (high_risk["PaymentMethod"] == "Electronic check").mean() > 0.4:
        actions.append("Electronic check利用者が多いため、自動支払い変更キャンペーンを検証する")
    if (high_risk["TechSupport"] == "No").mean() > 0.5:
        actions.append("TechSupport未加入者が多いため、サポート加入キャンペーンを実施する")
    if high_risk["MonthlyCharges"].mean() > enriched["MonthlyCharges"].mean():
        actions.append("高リスク層の月額料金が高いため、料金満足度調査とプラン見直し導線を設ける")
    return actions or ["高リスク層の傾向を継続監視し、セグメント別の追加施策を検討する"]
