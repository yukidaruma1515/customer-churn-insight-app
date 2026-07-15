# Customer Churn Insight App

> 📸 **アプリ画面（スクリーンショット掲載欄）**
> `docs/app-overview.png` を追加し、`![アプリ画面](docs/app-overview.png)` に差し替えてください。

## 1. プロジェクト概要

通信サービスの顧客データから解約確率を推定し、高リスク顧客の優先順位付け、予測理由、施策仮説、A/Bテスト設計までを支援するStreamlitアプリです。単にモデル精度を示すだけでなく、業務で「誰に、なぜ、何を、どう検証するか」へつなげることを目的にしています。

## 2. 背景と解決したい課題

解約者を一律に扱うと、限られた対応工数を有効に配分できません。本アプリは離脱確率、簡易期待損失、顧客属性を組み合わせ、優先対応候補と検証可能な施策案を提示します。

## 3. WebアプリURL

[Streamlit Community Cloud](https://share.streamlit.io/yukidaruma1515/customer-churn-insight-app/main/customer-churn-insight-app/app.py)

## 4. アプリ画面

- 概要・EDA：カテゴリ別の顧客数、解約者数、解約率
- モデル・閾値：5-fold CV、閾値連動の混同行列・Precision・Recall・F1
- 高度な分析：リフト、累積ゲイン、キャリブレーション
- セグメント・組合せ：属性別評価とクロス分析
- High Risk Customers：多条件フィルタ、期待損失順の優先順位
- 説明・What-if：特徴量重要度、SHAP、予測変化シミュレーション

## 5. 使用データ

IBM/Kaggleで広く利用される Telco Customer Churn 形式（7,043行、顧客属性・契約・サービス・料金・解約ラベル）を前提とします。アップロードCSVは必須列、型、カテゴリ、重複、欠損、件数を検証します。

## 6. 実データと合成データ

リポジトリ同梱CSVまたはアップロードCSVを使用します。CSVがない場合のみ動作確認用の合成データを生成し、画面で警告します。**合成データの分析結果は実際の通信会社の傾向を示さず、企業判断に直接利用できません。**

## 7. 分析の流れ

データ検証 → 前処理 → 75/25 holdout + Stratified 5-fold CV → 確率予測 → 閾値評価 → リフト/キャリブレーション → セグメント評価 → 施策仮説/A-Bテスト設計、の順です。

## 8. 使用したモデル

- Logistic Regression（解釈性の高いベースライン）
- Random Forest（非線形・相互作用）
- LightGBM（勾配ブースティング）

クラス不均衡を考慮し、過学習とStreamlit Cloudの計算量を抑える妥当な初期値をコード内にコメントしています。

## 9. モデル評価結果

Accuracy、Precision、Recall、F1-score、ROC-AUC、PR-AUCの5-fold平均と標準偏差を画面とダウンロードCSVで確認できます。データやライブラリ版で数値が変わるため、固定値ではなく実行時の結果を正とします。

## 10. 主な分析結果

上位10%・20%・30%への対応時の解約者捕捉率、Lift、予測確率のBrier scoreと確率帯別実績を確認できます。閾値変更の業務インパクト（対象人数、捕捉、見逃し）も即時表示します。

## 11. 顧客セグメント別の発見

SeniorCitizen、gender、Contract、InternetService、tenure帯、MonthlyCharges帯ごとにPrecision・Recall・F1・対象人数を比較し、十分な件数があるのにRecallが40%未満の層を警告します。

## 12. 高リスク顧客への施策

長期契約特典、料金プラン見直し、オンボーディング、TechSupport、自動支払い変更について、対象条件と仮説根拠を明示します。優先対応スコアは `離脱確率 × 月額料金 × 想定残存月数` の簡易期待損失であり、実際のCLVではありません。

## 13. A/Bテスト案

月額契約かつ高リスクの顧客へ長期契約特典を提示する実験について、Treatment/Control、KPI、Guardrail、期間、仮説、成功条件を定義しています。**相関関係は因果関係を意味せず、A/Bテストで確認するまで施策の因果効果は断定できません。**

## 14. 使用技術

Python 3.11推奨、Streamlit、pandas、scikit-learn、LightGBM、Matplotlib、SHAP、OpenAI API、pytest、GitHub Actions。

## 15. ディレクトリ構成

```text
customer-churn-insight-app/
├── app.py                 # Streamlit UI
├── src/                   # 前処理・学習・分析・説明・施策
├── tests/                 # ユニットテスト
├── data/                  # Telco CSV
├── models/                # 実行時生成（Git除外）
└── outputs/               # 実行時生成（Git除外）
.github/workflows/tests.yml
```

## 16. ローカルでの実行方法

```bash
git clone https://github.com/yukidaruma1515/customer-churn-insight-app.git
cd customer-churn-insight-app/customer-churn-insight-app
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

テストは `pytest -q`。LLMレポートを利用する場合だけ環境変数 `OPENAI_API_KEY` を設定します。秘密情報はコミットしません。キーがなければテンプレート版を表示します。

## 17. 制約と注意点

- 相関関係は因果関係を意味しません。
- SHAPは予測理由の説明であり、施策効果を証明しません。
- What-ifはモデル上の予測変化であり、因果効果ではありません。
- 合成データの結果は実際の企業判断に直接利用できません。
- アップロードデータで再学習する設計であり、本番推論基盤とは異なります。

## 18. 今後の改善点

実運用ではデータドリフト・精度・公平性を監視し、モデルを定期的に再学習する必要があります。加えて、時系列検証、確率校正モデル、コスト最適閾値、A/Bテストの検出力分析、実CLV・施策接触履歴、認証・監査ログの導入が候補です。

## License

[MIT License](LICENSE)
