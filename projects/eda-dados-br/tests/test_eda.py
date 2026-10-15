import numpy as np
import pandas as pd
import pytest

from eda_dados_br import analysis as A
from eda_dados_br import coleta as C


def test_parse_sgs_converte_data_e_valor():
    df = C.parse_sgs([{"data": "02/01/2024", "valor": "11.75"}, {"data": "01/01/2024", "valor": "11.75"}, {"data": "03/01/2024", "valor": "x"}])
    assert list(df["data"]) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    assert df["valor"].dtype.kind == "f"


def test_parse_sidra_ignora_cabecalho_e_valores_ausentes():
    payload = [
        {"NC": "Nível"},
        {"D1C": "11", "D1N": "Rondônia", "D3N": "2º trimestre 2026", "V": "2.6"},
        {"D1C": "12", "D1N": "Acre", "D3N": "2º trimestre 2026", "V": "..."},
    ]
    df = C.parse_sidra(payload)
    assert len(df) == 1 and df.loc[0, "territorio"] == "Rondônia" and df.loc[0, "valor"] == 2.6


def test_base100_e_yoy_valores_conhecidos():
    s = pd.Series([50.0, 100.0, 150.0], index=pd.date_range("2020-01-01", periods=3, freq="MS"))
    assert A.index_base100(s).tolist() == [100.0, 200.0, 300.0]
    s13 = pd.Series(np.arange(1.0, 15.0), index=pd.date_range("2020-01-01", periods=14, freq="MS"))
    yoy = A.yoy_change(s13)
    assert yoy.iloc[12] == pytest.approx((13 / 1 - 1) * 100)
    assert np.isnan(yoy.iloc[0])


def test_correlacao_defasada_detecta_atraso_conhecido():
    rng = np.random.default_rng(0)
    n = 300
    x = pd.Series(rng.normal(size=n).cumsum(), index=pd.date_range("2000-01-01", periods=n, freq="MS"))
    # y é x atrasado de 3 meses (mais ruído pequeno): o pico deve estar em lag +3
    y = x.shift(3).fillna(0) + rng.normal(scale=0.05, size=n)
    tab = A.lagged_correlation(x, y, max_lag=6)
    best = int(tab.loc[tab["correlacao"].idxmax(), "lag_meses"])
    assert best == 3


def test_ranking_uf_exclui_brasil_e_ordena():
    df = pd.DataFrame({
        "territorio": ["Brasil", "Acre", "Bahia", "Ceará"], "codigo_territorio": [1, 12, 29, 23],
        "periodo": ["2025"] * 4, "valor": [9.0, 3.0, 7.0, 5.0],
    })
    r = A.ranking_uf(df, "2025")
    assert list(r["territorio"]) == ["Acre", "Ceará", "Bahia"]


def test_parse_periodo_sidra():
    assert A.parse_periodo_sidra("julho 2026") == pd.Timestamp("2026-07-01")
    assert A.parse_periodo_sidra("2º trimestre 2026") == pd.Timestamp("2026-04-01")
    assert A.parse_periodo_sidra("1º trimestre 2012") == pd.Timestamp("2012-01-01")
    assert A.parse_periodo_sidra("2025") == pd.Timestamp("2025-01-01")
    with pytest.raises(ValueError):
        A.parse_periodo_sidra("abc")


def test_variacao_mensal_vira_indice_acumulado_antes_da_base100():
    idx = pd.date_range("2024-01-01", periods=3, freq="MS")
    r = pd.Series([1.0, 2.0, -1.0], index=idx)  # % a.m.
    base = A.index_base100(A.comparable_level(r, "% a.m."))
    assert base.round(4).tolist() == [100.0, 102.0, 100.98]
    assert A.comparable_level(r, "% a.a.") is None  # taxa em nível não entra no índice
    assert A.comparable_level(r, "R$/US$").equals(r)  # nível passa direto
