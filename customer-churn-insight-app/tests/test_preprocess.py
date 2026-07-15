import pandas as pd

from src.load_data import generate_sample_telco
from src.preprocess import clean_telco_data, make_train_test_split


def test_clean_and_stratified_split():
    data = clean_telco_data(generate_sample_telco(200))
    split = make_train_test_split(data)
    assert data["Churn"].isin([0, 1]).all()
    assert len(split.X_test) == 50
    assert abs(split.y_test.mean() - data.Churn.mean()) < .06
