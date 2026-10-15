import numpy as np
import pytest

from churn_prediction import data
from churn_prediction.explain import contributions, fit_reference
from churn_prediction.model import build_logreg, calibration_table, expected_calibration_error


def test_contribuicoes_somam_exatamente_ao_logit(toy_telco):
    df = toy_telco()
    X, y = df[data.FEATURES], df["Churn"]
    pipe = build_logreg(C=1.0).fit(X, y)
    z_bar = fit_reference(pipe, X)
    for i in (0, 5, 17):
        table, base, logit = contributions(pipe, X.iloc[[i]], z_bar)
        assert base + table["contribution"].sum() == pytest.approx(logit, abs=1e-9)
        p = 1 / (1 + np.exp(-logit))
        assert p == pytest.approx(pipe.predict_proba(X.iloc[[i]])[0, 1], abs=1e-9)
        assert set(table["feature"]) == set(data.FEATURES)  # one-hot agrupado de volta nas variáveis originais


def test_base_e_a_media_do_logit_no_treino(toy_telco):
    df = toy_telco()
    X, y = df[data.FEATURES], df["Churn"]
    pipe = build_logreg().fit(X, y)
    _, base, _ = contributions(pipe, X.iloc[[0]], fit_reference(pipe, X))
    assert base == pytest.approx(pipe.decision_function(X).mean(), abs=1e-9)


def test_ece_calibracao_perfeita_e_enviesada():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.random(20000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.02
    assert expected_calibration_error(y, np.clip(p + 0.2, 0, 1)) > 0.1
    t = calibration_table(y, p)
    assert len(t) == 10 and (t["n"].sum() == 20000)
