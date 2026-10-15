import numpy as np
import pandas as pd
import pytest

from demand_forecasting import models as M
from demand_forecasting.make_sample import generate


def test_seasonal_naive_repete_mesmo_dia_da_semana():
    idx = pd.date_range("2024-01-01", periods=14, freq="D")
    hist = pd.Series(np.arange(14, dtype=float), index=idx)  # origem = 2024-01-14
    fut = pd.date_range("2024-01-15", periods=7, freq="D")
    out = M.seasonal_naive(hist, fut)
    assert out.tolist() == hist.iloc[-7:].tolist()


def test_metricas_caso_conhecido():
    y = np.array([10.0, 20.0, 0.0])
    yhat = np.array([12.0, 18.0, 5.0])
    assert M.wape(y, yhat) == pytest.approx((2 + 2 + 5) / 30)
    assert M.mape(y, yhat) == pytest.approx((0.2 + 0.1) / 2)  # ignora y = 0 no MAPE


def test_mase_scale():
    hist = pd.Series([1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0])
    # m = 7: |h[7]-h[0]| = |6-1| = 5
    assert M.mase_scale(hist, m=7) == pytest.approx(5.0)


def test_sintetico_fechado_aos_domingos_e_schema():
    df = generate(n_stores=3)
    assert len(df) == 3 * 1095
    assert (df.loc[df["Date"].dt.dayofweek == 6, "Sales"] == 0).all()
    assert {"Store", "Date", "Sales", "Customers", "Open", "Promo", "StateHoliday", "SchoolHoliday", "StoreType", "Assortment", "CompetitionDistance"} <= set(df.columns)


def test_gbm_sem_vazamento_de_vendas_futuras():
    df = generate(n_stores=3)
    origin = pd.Timestamp("2015-01-01")
    stores = [1, 2, 3]
    dates = pd.date_range(origin + pd.Timedelta(days=1), periods=M.HORIZON, freq="D")
    model, levels = M.gbm_fit(df, origin, stores)
    base = M.gbm_predict(model, levels, df, 1, dates)
    tampered = df.copy()
    tampered.loc[tampered["Date"] > origin, "Sales"] = 999_999  # vendas futuras adulteradas
    tampered_pred = M.gbm_predict(model, levels, tampered, 1, dates)
    assert np.allclose(base, tampered_pred)


def test_seasonal_naive_pula_feriado_da_semana_anterior():
    idx = pd.date_range("2024-01-01", periods=14, freq="D")
    hist = pd.Series(np.arange(1, 15, dtype=float), index=idx)  # origem = 2024-01-14
    hist.loc["2024-01-08"] = 0.0  # segunda-feira passada: feriado (loja fechada)
    out = M.seasonal_naive(hist, pd.date_range("2024-01-15", periods=1, freq="D"))
    assert out.tolist() == [1.0]  # usa a segunda-feira de duas semanas atrás, não o zero
