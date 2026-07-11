from __future__ import annotations

import os
from typing import Any


def build_report_context(
    churn_rate: float,
    model_name: str,
    best_metrics: dict[str, Any],
    top_features: list[str],
    high_risk_count: int,
    high_risk_trends: list[str],
    recommended_actions: list[str],
) -> dict[str, Any]:
    return {
        "churn_rate": churn_rate,
        "model_name": model_name,
        "best_metrics": best_metrics,
        "top_features": top_features,
        "high_risk_count": high_risk_count,
        "high_risk_trends": high_risk_trends,
        "recommended_actions": recommended_actions,
        "ab_tests": [
            "月額契約ユーザーに長期契約特典を提示する群と提示しない群でChurn率を比較する。",
            "TechSupport未加入者に初月無料キャンペーンを提示し、継続率改善を検証する。",
            "Electronic check利用者に自動支払い変更特典を提示し、解約率の変化を見る。",
            "月額料金が高い高リスク層に料金満足度調査とプラン見直し導線を提示し、解約率の変化を見る。",
        ],
    }


def template_report(context: dict[str, Any]) -> str:
    metrics = context["best_metrics"]
    top_features = "\n".join([f"- {feature}" for feature in context["top_features"]])
    trends = "\n".join([f"- {trend}" for trend in context["high_risk_trends"]])
    actions = "\n".join([f"- {action}" for action in context["recommended_actions"]])
    ab_tests = "\n".join([f"- {test}" for test in context["ab_tests"]])
    return f"""
## 1. エグゼクティブサマリー
全体のChurn率は{context['churn_rate']:.1%}です。採用モデルは{context['model_name']}で、ROC-AUCは{metrics['roc_auc']:.3f}、F1-scoreは{metrics['f1']:.3f}、Recallは{metrics['recall']:.3f}でした。高リスク顧客は{context['high_risk_count']:,}人です。

## 2. 主な分析結果
- Accuracy: {metrics['accuracy']:.3f}
- Precision: {metrics['precision']:.3f}
- Recall: {metrics['recall']:.3f}
- F1-score: {metrics['f1']:.3f}
- ROC-AUC: {metrics['roc_auc']:.3f}

## 3. 重要な離脱要因
{top_features}

## 4. 優先対応すべき顧客層
{trends}

## 5. 推奨施策
{actions}

## 6. 次に実施すべきA/Bテスト案
{ab_tests}
""".strip()


def generate_report(context: dict[str, Any]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return template_report(context)
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        prompt = f"""
PMやビジネス職向けに、顧客離脱分析の改善レポートを日本語で作成してください。
以下の見出しを必ず含めてください:
1. エグゼクティブサマリー
2. 主な分析結果
3. 重要な離脱要因
4. 優先対応すべき顧客層
5. 推奨施策
6. 次に実施すべきA/Bテスト案

分析コンテキスト:
{context}
"""
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=0.3,
        )
        return response.output_text
    except Exception:
        return template_report(context)
