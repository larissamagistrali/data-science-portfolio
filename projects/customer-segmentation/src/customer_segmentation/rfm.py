"""Construção da tabela RFM (Recência, Frequência, Monetário) por cliente."""
from __future__ import annotations

import hashlib

import pandas as pd

SALT = "dsp-online-retail-ii-v1"  # sal fixo e público: pseudonimização (não é anonimização forte)


def pseudonymize(customer_id: float | int | str) -> str:
    """Hash curto e determinístico do CustomerID (o ID original não é versionado)."""
    raw = f"{SALT}:{int(float(customer_id))}".encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def build_rfm(valid: pd.DataFrame, cancelled: pd.DataFrame | None = None, snapshot: pd.Timestamp | None = None) -> pd.DataFrame:
    """Agrega as linhas de venda válidas por cliente.

    - recency_days: dias entre a última compra e a data de referência (snapshot = último dia do dado + 1);
    - frequency: nº de faturas distintas (compras);
    - monetary: soma de Quantity x Price das compras válidas (GBP);
    - avg_ticket: monetary / frequency;
    - cancel_rate: faturas canceladas / (faturas válidas + canceladas) — usada só no perfil, não no cluster.
    """
    snapshot = snapshot or (valid["InvoiceDate"].max().normalize() + pd.Timedelta(days=1))
    g = valid.groupby("CustomerID")
    rfm = pd.DataFrame(
        {
            "recency_days": (snapshot - g["InvoiceDate"].max()).dt.days.astype("int32"),
            "frequency": g["Invoice"].nunique().astype("int32"),
            "monetary": g["revenue"].sum().round(2),
        }
    )
    rfm["avg_ticket"] = (rfm["monetary"] / rfm["frequency"]).round(2)
    n_cancel = cancelled.groupby("CustomerID")["Invoice"].nunique() if cancelled is not None and len(cancelled) else pd.Series(dtype=float)
    rfm["cancel_rate"] = (n_cancel.reindex(rfm.index).fillna(0) / (rfm["frequency"] + n_cancel.reindex(rfm.index).fillna(0))).round(4)
    rfm = rfm.reset_index()
    rfm.insert(0, "customer", rfm.pop("CustomerID").map(pseudonymize))
    return rfm
