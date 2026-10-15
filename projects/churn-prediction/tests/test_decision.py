import numpy as np
import pytest

from churn_prediction.decision import best_threshold, breakeven_probability, expected_gain, sweep_thresholds


def test_breakeven():
    assert breakeven_probability(cost=60, success_rate=0.25, value=800) == pytest.approx(0.3)
    assert breakeven_probability(60, 0.0, 800) == float("inf")


def test_expected_gain_zero_no_breakeven():
    p = breakeven_probability(60, 0.25, 800)
    assert expected_gain(p, 800, 60, 0.25) == pytest.approx(0.0)


def test_sweep_caso_conhecido():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.6, 0.4, 0.1])
    v = np.full(4, 100.0)
    s = sweep_thresholds(y, p, v, cost=10, success_rate=0.5, thresholds=np.array([0.0, 0.5, 0.95]))
    assert s["targeted"].tolist() == [4, 2, 0]
    assert s["tp"].tolist() == [2, 2, 0]
    # limiar 0.5: 2 churners abordados -> 2*(0.5*100) - 2*10 = 80
    assert s["net_gain"].tolist() == [2 * 50 - 4 * 10, 80.0, 0.0]
    assert s["recall"].tolist()[:2] == [1.0, 1.0]
    assert best_threshold(s) == 0.5


def test_best_threshold_prefere_menos_abordados_no_empate():
    import pandas as pd

    s = pd.DataFrame({"threshold": [0.1, 0.2], "targeted": [10, 5], "net_gain": [3.0, 3.0]})
    assert best_threshold(s) == 0.2
