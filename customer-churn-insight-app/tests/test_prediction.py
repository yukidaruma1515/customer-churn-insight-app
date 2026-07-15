from src.load_data import generate_sample_telco
from src.preprocess import build_preprocessor, clean_telco_data, split_features
from src.train_model import build_models


def test_model_predict_proba_range():
    data = clean_telco_data(generate_sample_telco(120))
    numeric, categorical = split_features(data)
    model = build_models(build_preprocessor(numeric, categorical))["Logistic Regression"]
    features = data.drop(columns=["customerID", "Churn"])
    model.fit(features, data.Churn)
    probabilities = model.predict_proba(features)[:, 1]
    assert len(probabilities) == len(data)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
