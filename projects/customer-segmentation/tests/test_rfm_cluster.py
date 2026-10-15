import numpy as np
import pandas as pd
import pytest

from customer_segmentation import cluster as C
from customer_segmentation.data import clean
from customer_segmentation.rfm import build_rfm, pseudonymize


def _raw() -> pd.DataFrame:
    rows = [
        # Invoice, StockCode, Quantity, InvoiceDate, Price, CustomerID
        ("1001", "A1", 2, "2011-01-01", 10.0, 1.0),      # válida (20)
        ("1002", "A2", 1, "2011-01-11", 5.0, 1.0),       # válida (5)
        ("1003", "A1", 3, "2011-01-05", 10.0, 2.0),      # válida (30)
        ("C1004", "A1", -1, "2011-01-06", 10.0, 2.0),    # cancelamento
        ("1005", "A1", 1, "2011-01-07", 10.0, None),     # sem cliente
        ("1006", "A1", -2, "2011-01-07", 10.0, 3.0),     # quantidade inválida
        ("1007", "A1", 2, "2011-01-07", 0.0, 3.0),       # preço inválido
        ("1008", "POST", 1, "2011-01-07", 18.0, 3.0),    # não é produto
    ]
    df = pd.DataFrame(rows, columns=["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "CustomerID"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"] = df["CustomerID"].astype("Float64")
    return df


def test_clean_descarta_por_motivo():
    valid, cancelled, stats = clean(_raw())
    assert stats["sem_customer_id"] == 1
    assert stats["cancelamentos"] == 1 and len(cancelled) == 1
    assert stats["quantidade_ou_preco_invalido"] == 2
    assert stats["codigos_nao_produto"] == 1
    assert len(valid) == 3 and stats["clientes"] == 2
    assert (valid["revenue"] > 0).all()


def test_rfm_valores_conhecidos():
    valid, cancelled, _ = clean(_raw())
    rfm = build_rfm(valid, cancelled, snapshot=pd.Timestamp("2011-01-12")).set_index("customer")
    c1, c2 = rfm.loc[pseudonymize(1)], rfm.loc[pseudonymize(2)]
    assert (c1["recency_days"], c1["frequency"], c1["monetary"]) == (1, 2, 25.0)
    assert (c2["recency_days"], c2["frequency"], c2["monetary"]) == (7, 1, 30.0)
    assert c1["cancel_rate"] == 0.0 and c2["cancel_rate"] == 0.5  # 1 cancelada / (1 válida + 1 cancelada)


def test_pseudonimizacao_deterministica_e_nao_expoe_id():
    assert pseudonymize(12346.0) == pseudonymize(12346)
    assert pseudonymize(12346) != pseudonymize(12347)
    assert "12346" not in pseudonymize(12346)


def _blobs(seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[10, 20, 500], [200, 2, 50], [60, 8, 1500]], dtype=float)
    parts = [pd.DataFrame({"recency_days": rng.normal(c[0], 3, 150).clip(1), "frequency": rng.normal(c[1], 0.3, 150).clip(1),
                           "monetary": rng.normal(c[2], 20, 150).clip(1)}) for c in centers]
    df = pd.concat(parts, ignore_index=True)
    df["avg_ticket"] = df["monetary"] / df["frequency"]
    df["cancel_rate"] = 0.0
    df["customer"] = [f"c{i}" for i in range(len(df))]
    return df


def test_select_k_encontra_tres_blocos():
    rfm = _blobs()
    Z, _ = C.transform(rfm)
    table = C.select_k(Z, ks=range(2, 7), sample_size=None)
    assert C.best_k(table, k_min=2, k_max=6) == 3
    assert table["inertia"].is_monotonic_decreasing


def test_nomes_de_segmento_por_regra():
    cents = pd.DataFrame(
        [[-1.0, 1.5, 1.5], [1.2, -0.8, -0.8], [-0.9, -0.5, -0.4], [0.0, 0.4, 0.6], [0.0, 0.0, 0.0], [0.9, 0.4, 0.6]],
        columns=C.RFM_COLS,
    )  # recency_days negativo = compra recente
    names = C.name_segments(cents)
    assert names[0] == "Campeões"
    assert names[1] == "Inativos"
    assert names[2] == "Recentes (baixa frequência)"
    assert names[3] == "Leais"
    assert names[4] == "Intermediários"
    assert names[5] == "Em risco (alto valor)"


def test_nomes_repetidos_ganham_sufixo_por_gasto():
    cents = pd.DataFrame([[1.0, -1.0, -1.0], [1.5, -0.6, -0.9]], columns=C.RFM_COLS)
    names = C.name_segments(cents)
    assert sorted(names.values()) == ["Inativos I", "Inativos II"]
    assert names[1] == "Inativos I"  # maior gasto (-0.9 > -1.0) fica com "I"


def test_quintis_e_comparacao():
    rfm = _blobs()
    q = C.quintile_segments(rfm)
    assert q[["R", "F", "M"]].min().min() == 1 and q[["R", "F", "M"]].max().max() == 5
    assert q["segmento_quintil"].notna().all()
    Z, _ = C.transform(rfm)
    labels = C.fit_kmeans(Z, 3).labels_
    cmp = C.compare_segmentations(Z, labels, pd.Series(labels))
    assert cmp["ari_kmeans_vs_quintis"] == pytest.approx(1.0)


def test_profile_receita_soma_100():
    rfm = _blobs()
    Z, _ = C.transform(rfm)
    labels = C.fit_kmeans(Z, 3).labels_
    prof = C.profile(rfm, labels, {0: "a", 1: "b", 2: "c"})
    assert prof["pct_receita"].sum() == pytest.approx(1.0)
    assert prof["clientes"].sum() == len(rfm)
