"""Gera um DataFrame sintético mínimo com o schema do Telco (só para testes unitários)."""
import numpy as np
import pandas as pd
import pytest


def _toy_telco(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    internet = rng.choice(["DSL", "Fiber optic", "No"], n)
    addon = lambda: np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], n))  # noqa: E731
    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    df = pd.DataFrame(
        {
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n),
            "tenure": rng.integers(0, 72, n),
            "PhoneService": phone,
            "MultipleLines": np.where(phone == "No", "No phone service", rng.choice(["Yes", "No"], n)),
            "InternetService": internet,
            "OnlineSecurity": addon(), "OnlineBackup": addon(), "DeviceProtection": addon(),
            "TechSupport": addon(), "StreamingTV": addon(), "StreamingMovies": addon(),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "PaymentMethod": rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n),
            "MonthlyCharges": rng.uniform(18, 118, n).round(2),
        }
    )
    logit = -1.0 - 0.04 * df["tenure"] + 1.2 * (df["Contract"] == "Month-to-month") + 0.5 * (df["InternetService"] == "Fiber optic")
    df["Churn"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


@pytest.fixture
def toy_telco():
    return _toy_telco
