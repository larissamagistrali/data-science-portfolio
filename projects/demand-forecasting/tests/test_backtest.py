import numpy as np
import pandas as pd

from demand_forecasting.backtest import add_intervals


def test_intervalos_usam_so_origens_anteriores():
    rows = []
    rng = np.random.default_rng(0)
    for k, origin in enumerate(pd.to_datetime(["2015-01-01", "2015-04-02", "2015-07-02"])):
        for h in range(1, 29):
            yhat = 100.0
            y = yhat * (1 + rng.normal(0, 0.1 if k < 2 else 5.0))  # última origem com erro enorme
            rows.append({"model": "m", "store": 1, "origin": origin, "h": h, "date": origin + pd.Timedelta(days=h),
                         "y": y, "yhat": yhat, "open": 1, "mase_scale": 1.0})
    bt = add_intervals(pd.DataFrame(rows))
    first = bt[bt["origin"] == pd.Timestamp("2015-01-01")]
    assert first["lo"].isna().all()  # sem calibração na primeira origem
    last = bt[bt["origin"] == pd.Timestamp("2015-07-02")]
    # a calibração vem das origens anteriores (erro ~10%), então a faixa é estreita
    assert (last["hi"] - last["lo"]).mean() < 60


def test_previsao_zero_na_calibracao_nao_estoura_o_intervalo():
    rows = []
    for k, origin in enumerate(pd.to_datetime(["2015-01-01", "2015-04-02"])):
        for h in range(1, 29):
            yhat = 0.0 if (k == 0 and h % 7 == 1) else 100.0  # 4 de 28 dias abertos previstos como zero
            rows.append({"model": "m", "store": 1, "origin": origin, "h": h, "date": origin + pd.Timedelta(days=h),
                         "y": 100.0 + h % 3, "yhat": yhat, "open": 1, "mase_scale": 1.0})
    bt = add_intervals(pd.DataFrame(rows))
    last = bt[bt["origin"] == pd.Timestamp("2015-04-02")]
    assert last["hi"].max() < 200  # sem a proteção, o erro relativo de 100/1 levava a faixa a ~10.000
