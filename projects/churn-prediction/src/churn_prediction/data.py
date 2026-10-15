"""Download e limpeza do Telco Customer Churn (IBM, 7.043 clientes x 21 colunas)."""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
USER_AGENT = "data-science-portfolio (educational; github.com/larissamagistrali)"
PROJECT_DIR = Path(__file__).resolve().parents[2]

TARGET = "Churn"
NUMERIC = ["tenure", "MonthlyCharges"]
BINARY_01 = ["SeniorCitizen"]
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
# TotalCharges fica de fora: é ~ tenure x MonthlyCharges (redundante) e não pode ser digitado de
# forma independente no simulador. A decisão é checada em train.py (ablação em metrics.json).
FEATURES = NUMERIC + BINARY_01 + CATEGORICAL
INTERNET_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


def raw_dir() -> Path:
    return Path(os.environ.get("DSP_RAW_DIR", PROJECT_DIR / "data" / "raw")) / "telco"


def download(dest: Path | None = None) -> Path:
    """Baixa o CSV (via .part) para ``<raw>/Telco-Customer-Churn.csv``; reutiliza se existir."""
    dest = dest or raw_dir()
    dest.mkdir(parents=True, exist_ok=True)
    final = dest / "Telco-Customer-Churn.csv"
    if final.exists():
        return final
    part = final.with_suffix(".csv.part")
    req = urllib.request.Request(URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (URL fixa, https)
        part.write_bytes(resp.read())
    part.rename(final)
    return final


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Limpeza: TotalCharges em branco vira 0 (clientes com tenure 0, ainda sem cobrança),
    Churn vira 0/1 e remove customerID (todos únicos: linhas idênticas nas demais colunas são
    clientes distintos, não duplicatas). TotalCharges é mantido só para a ablação."""
    df = raw.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # " " (em branco) -> NaN
    sem_cobranca = df["TotalCharges"].isna()
    # No dado original os 11 casos em branco têm tenure == 0: ainda não houve cobrança.
    df.loc[sem_cobranca, "TotalCharges"] = 0.0
    df[TARGET] = (df[TARGET].astype(str).str.strip().str.lower() == "yes").astype("int8")
    df["SeniorCitizen"] = df["SeniorCitizen"].astype("int8")
    df = df.drop(columns=["customerID"], errors="ignore").reset_index(drop=True)
    for col in CATEGORICAL:
        df[col] = df[col].astype(str).str.strip()
    return df


def normalize_customer(row: dict) -> dict:
    """Garante coerência entre campos dependentes (usado pelo simulador do app).

    Sem internet, os serviços adicionais valem "No internet service"; sem telefone,
    MultipleLines vale "No phone service" (como no dado original).
    """
    out = dict(row)
    if out.get("InternetService") == "No":
        for col in INTERNET_ADDONS:
            out[col] = "No internet service"
    elif out.get("InternetService") in ("DSL", "Fiber optic"):
        for col in INTERNET_ADDONS:
            if out.get(col) == "No internet service":
                out[col] = "No"
    if out.get("PhoneService") == "No":
        out["MultipleLines"] = "No phone service"
    elif out.get("MultipleLines") == "No phone service":
        out["MultipleLines"] = "No"
    return out
