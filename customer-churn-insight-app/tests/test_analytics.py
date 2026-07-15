import numpy as np
import pandas as pd

from src.analytics import lift_gain_table, threshold_metrics, top_k_metrics


def test_threshold_metrics_apply_probability_cutoff():
    result = threshold_metrics(pd.Series([0, 1, 1, 0]), np.array([.2, .8, .6, .7]), .65)
    assert result["captured_churners"] == 1
    assert result["missed_churners"] == 1
    assert result["high_risk_count"] == 2


def test_lift_and_top_k_rank_highest_probabilities():
    actual = pd.Series([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    probabilities = np.array([.9, .8, .7, .6, .5, .4, .3, .2, .1, .05])
    assert top_k_metrics(actual, probabilities, .2) == {"precision": 1.0, "recall": 1.0, "lift": 5.0}
    table = lift_gain_table(actual, probabilities)
    assert table.iloc[-1]["gain"] == 1.0
