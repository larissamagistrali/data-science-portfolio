"""Backtest com origem rolante (sem vazamento) + intervalos empíricos com cobertura medida.

Origens a cada 91 dias em 2015; cada origem prevê H=28 dias à frente. Intervalos de 90%:
quantis 5% e 95% do erro relativo (y - yhat) / yhat, calculados SÓ nas origens anteriores
(calibração), e a cobertura medida na origem seguinte. A primeira origem não tem calibração (NaN).

Uso: PYTHONPATH=projects/demand-forecasting/src python -m demand_forecasting.backtest
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import models as M

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ORIGINS = pd.to_datetime(["2015-01-01", "2015-04-02", "2015-07-02", "2015-10-01"])
Q_LO, Q_HI = 0.05, 0.95


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(DATA_DIR / "sales_sample.parquet")
    stores = sorted(df["Store"].unique().tolist())
    static = df.drop_duplicates("Store").set_index("Store")
    closed_flag = df.pivot(index="Date", columns="Store", values="Open")
    sales = df.pivot(index="Date", columns="Store", values="Sales").asfreq("D").fillna(0.0)
    rows = []
    for k, origin in enumerate(ORIGINS):
        dates = pd.date_range(origin + pd.Timedelta(days=1), periods=M.HORIZON, freq="D")
        gbm, levels = M.gbm_fit(df, origin, stores)
        for s in stores:
            hist = sales[s].loc[:origin]
            preds = {
                "seasonal_naive": M.seasonal_naive(hist, dates),
                "holt_winters": M.holt_winters(hist, dates),
                "gbm": M.gbm_predict(gbm, levels, df, s, dates),
            }
            open_ = closed_flag.loc[dates, s].to_numpy()
            scale = M.mase_scale(hist)
            for name, yhat in preds.items():
                yhat = np.where(open_ == 1, np.nan_to_num(yhat, nan=0.0), 0.0)  # loja fechada = 0 (calendário conhecido)
                y = sales.loc[dates, s].to_numpy()
                rows.append(pd.DataFrame({
                    "model": name, "store": s, "origin": origin, "h": np.arange(1, M.HORIZON + 1), "date": dates,
                    "y": y, "yhat": yhat, "open": open_, "mase_scale": scale,
                }))
        print("origem", origin.date(), "ok")
    bt = pd.concat(rows, ignore_index=True)
    bt = add_intervals(bt)
    return bt, bt


def add_intervals(bt: pd.DataFrame) -> pd.DataFrame:
    bt = bt.copy()
    bt["lo"] = np.nan
    bt["hi"] = np.nan
    origins = sorted(bt["origin"].unique())
    for model in bt["model"].unique():
        for k, origin in enumerate(origins):
            if k == 0:
                continue
            cal = bt[(bt["model"] == model) & (bt["origin"].isin(origins[:k])) & (bt["open"] == 1)]
            # Erro relativo só existe com previsão positiva: dividir por ~0 gera razões de milhares
            # e um quantil de 95% absurdo, que estoura a faixa de todas as previsões.
            cal = cal[cal["yhat"] > 0]
            rel = ((cal["y"] - cal["yhat"]) / cal["yhat"]).to_numpy()
            q_lo, q_hi = np.quantile(rel, [Q_LO, Q_HI])
            m = (bt["model"] == model) & (bt["origin"] == origin)
            yhat = bt.loc[m, "yhat"]
            bt.loc[m, "lo"] = (yhat * (1 + q_lo)).clip(lower=0).to_numpy()
            bt.loc[m, "hi"] = (yhat * (1 + q_hi)).to_numpy()
    return bt


def metrics(bt: pd.DataFrame) -> dict:
    out: dict = {}
    ev = bt[bt["open"] == 1]
    for model, g in ev.groupby("model"):
        y, yhat = g["y"].to_numpy(float), g["yhat"].to_numpy(float)
        mase = (np.abs(g["y"] - g["yhat"]) / g["mase_scale"]).groupby([g["store"], g["origin"]]).mean().mean()
        per_origin = {str(o.date()): M.wape(gg["y"].to_numpy(float), gg["yhat"].to_numpy(float)) for o, gg in g.groupby("origin")}
        cov = {}
        for o, gg in g.dropna(subset=["lo"]).groupby("origin"):
            cov[str(o.date())] = float(((gg["y"] >= gg["lo"]) & (gg["y"] <= gg["hi"])).mean())
        out[model] = {
            "wape": M.wape(y, yhat), "mape": M.mape(y, yhat), "mase": float(mase),
            "wape_por_origem": per_origin,
            "cobertura_90_por_origem": cov,
            "cobertura_90_media": float(np.mean(list(cov.values()))) if cov else None,
        }
    return out


def main() -> None:
    bt, _ = run()
    m = metrics(bt)
    rows_cols = ["model", "store", "origin", "h", "date", "y", "yhat", "lo", "hi", "open"]
    out = bt[rows_cols].copy()
    out["model"] = out["model"].astype("category")
    out["yhat"] = out["yhat"].astype("float32")
    out["lo"] = out["lo"].astype("float32")
    out["hi"] = out["hi"].astype("float32")
    out["y"] = out["y"].astype("float32")
    out.to_parquet(DATA_DIR / "backtest.parquet", compression="zstd", index=False)
    payload = {
        "dados": "SINTÉTICO (schema Rossmann; gerador em src/demand_forecasting/make_sample.py, seed 42)",
        "gerado_em": date.today().isoformat(),
        "horizonte_dias": M.HORIZON,
        "origens": [str(o.date()) for o in ORIGINS],
        "lojas": int(bt["store"].nunique()),
        "intervalo_nominal": 0.90,
        "metricas": m,
        "avaliacao": "apenas dias abertos (Open=1); fechados são previstos como 0",
    }
    (DATA_DIR / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
