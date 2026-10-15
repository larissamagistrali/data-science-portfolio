import pandas as pd
import pytest

from churn_prediction import data


def _raw(toy_telco, total):
    df = toy_telco(len(total))
    df.insert(0, "customerID", [f"id-{i}" for i in range(len(df))])
    df["Churn"] = df["Churn"].map({1: "Yes", 0: "No"})
    df["TotalCharges"] = total
    return df


def test_clean_trata_total_charges_em_branco_e_alvo(toy_telco):
    out = data.clean(_raw(toy_telco, [" ", "10.5", "20"] * 10))
    assert out["TotalCharges"].notna().all()
    assert out["TotalCharges"].iloc[0] == 0.0
    assert set(out["Churn"].unique()) <= {0, 1}
    assert "customerID" not in out.columns
    assert len(out) == 30  # não remove linhas "duplicadas": clientes distintos podem ter perfis idênticos


@pytest.mark.parametrize(
    "row, esperado",
    [
        ({"InternetService": "No", "OnlineSecurity": "Yes", "PhoneService": "Yes", "MultipleLines": "Yes"},
         {"OnlineSecurity": "No internet service", "MultipleLines": "Yes"}),
        ({"InternetService": "DSL", "StreamingTV": "No internet service", "PhoneService": "No", "MultipleLines": "Yes"},
         {"StreamingTV": "No", "MultipleLines": "No phone service"}),
    ],
)
def test_normalize_customer(row, esperado):
    out = data.normalize_customer(row)
    for k, v in esperado.items():
        assert out[k] == v


def test_features_nao_incluem_alvo_nem_total_charges():
    assert "Churn" not in data.FEATURES and "TotalCharges" not in data.FEATURES
