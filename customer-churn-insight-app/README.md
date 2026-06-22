# Customer Churn Insight App

## 1. アプリ概要

Telco Customer Churnデータを用いて、顧客の離脱リスクを予測し、離脱要因を可視化し、顧客ごとの継続率改善施策を提案するStreamlitアプリです。OpenAI APIキーがある場合はPM向け改善レポートをLLMで生成し、APIキーがない場合はテンプレートでレポートを生成します。

## 2. 背景

通信サービスやサブスクリプション事業では、既存顧客の離脱を防ぐことが売上やLTVに直結します。重要なのは、離脱しそうな顧客を予測するだけでなく、なぜ離脱しそうなのか、どの施策を打つべきかまで意思決定に接続することです。

## 3. 使用データ

Kaggleの「Telco Customer Churn」を想定しています。主なカラムは `customerID`, `tenure`, `Contract`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `Churn` などです。

実CSVがない場合でも、同じカラム構成のサンプルデータを自動生成します。

## 4. 目的

- 顧客ごとの離脱確率を予測する
- AccuracyだけでなくRecall、F1-score、ROC-AUCを重視する
- SHAPまたは特徴量重要度で離脱要因を説明する
- 顧客ごとに改善施策を提案する
- 分析結果をPM・ビジネス職向けレポートに変換する

## 5. 使用技術

- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- Streamlit
- joblib
- SHAP
- LightGBM
- OpenAI API

## 6. ファイル構成

```text
customer-churn-insight-app/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── README.md
├── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── predict.py
│   ├── explain.py
│   ├── recommend_actions.py
│   ├── llm_report.py
│   └── visualize.py
├── models/
│   ├── churn_model.pkl
│   └── preprocessor.pkl
└── outputs/
    ├── model_metrics.csv
    └── high_risk_customers.csv
```

## 7. 実行方法

```bash
cd /Users/nobu-macbook/sophia_puro/customer-churn-insight-app
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

OpenAI APIを使う場合:

```bash
cp .env.example .env
# .envにOPENAI_API_KEYを設定
export OPENAI_API_KEY=your_api_key_here
python3 -m streamlit run app.py
```

## 8. 主な機能

- CSV読み込みとアップロード
- TotalChargesの数値変換
- Churn率、件数、基本統計量の表示
- 契約形態、支払い方法、tenure、MonthlyChargesのEDA
- Logistic Regression、Random Forest、LightGBMの比較
- Accuracy、Precision、Recall、F1-score、ROC-AUC、混同行列、ROC曲線
- 特徴量重要度とSHAPまたは代替説明
- 高リスク顧客一覧
- 顧客ごとの施策提案
- LLMまたはテンプレートによる改善レポート生成
- Uplift Modelingへの発展案

## 9. モデル評価結果

アプリのModelingタブに、各モデルの評価指標が表示されます。離脱顧客を見逃さないことが重要なため、AccuracyだけでなくRecall、F1-score、ROC-AUCを重視して採用モデルを選びます。

## 10. 工夫点

- `customerID` はモデル特徴量から除外しました
- `TotalCharges` は数値型に変換し、欠損を `MonthlyCharges * tenure` で補完しました
- 不均衡データを意識し、Accuracy偏重にしない評価設計にしました
- 予測結果を施策提案とレポート生成につなげました
- APIキーがなくてもテンプレートレポートで動くようにしました

## 11. LLMレポート生成機能

OpenAI APIキーがある場合は、分析結果をもとにPM向け改善レポートを生成します。APIキーがない場合やAPI呼び出しに失敗した場合は、テンプレートベースのレポートを生成します。

レポートには、エグゼクティブサマリー、主な分析結果、重要な離脱要因、優先対応すべき顧客層、推奨施策、A/Bテスト案、今後の発展案、ES・面接用要約文を含めます。

## 12. Uplift Modelingへの発展案

現在のアプリでは離脱予測と施策提案を行います。Telco Customer Churnデータには、キャンペーン配信有無、クーポン提示有無、サポート提案有無などのTreatment列がないため、厳密なUplift Modelingは実装していません。

今後は施策ログを追加し、T-LearnerやS-Learnerを用いて、施策ありの場合と施策なしの場合の継続確率差を推定します。これにより、単に離脱しそうな顧客ではなく、施策によって離脱を防げる顧客を特定できます。

## 13. 今後の改善

- 実データでの閾値最適化
- 顧客セグメント別モデル
- キャンペーン反応ログの追加
- Uplift Modeling
- レポートPDF出力
- 定期再学習

## 14. DeNAインターン応募で説明できるポイント

顧客の離脱リスクを予測するだけでなく、SHAPによる要因可視化、顧客ごとの施策提案、LLMによる改善レポート生成まで実装した。これにより、データ分析を単なる予測精度の比較で終わらせず、プロダクト改善や意思決定に接続することを意識した。
