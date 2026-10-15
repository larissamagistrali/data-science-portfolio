"""Gera o dado SINTÉTICO com o schema do Rossmann Store Sales (Kaggle, não baixado aqui).

30 lojas x 2013-01-01..2015-12-31 (diário). Componentes: nível por loja, tendência linear suave,
sazonalidade semanal (sábado forte, domingo fechado), sazonalidade anual (verão + dezembro),
promoções em blocos semanais, feriados escolares, feriados nacionais fechados (1/jan e 25/dez),
ruído log-normal. Seed fixa (42). Rotulado como `synthetic` em todo o projeto.

Uso: PYTHONPATH=projects/demand-forecasting/src python -m demand_forecasting.make_sample
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_STORES = 30
START, END = "2013-01-01", "2015-12-31"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def generate(seed: int = SEED, n_stores: int = N_STORES) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, END, freq="D")
    T = len(dates)
    dow = dates.dayofweek.to_numpy()  # 0=seg ... 6=dom
    doy = dates.dayofyear.to_numpy()
    month = dates.month.to_numpy()
    trend = 1 + 0.06 * np.linspace(0, 1, T)
    dow_factor = np.array([1.00, 1.05, 1.02, 1.03, 1.10, 1.28, 0.0])  # domingo fechado
    annual = 1 + 0.10 * np.sin(2 * np.pi * (doy - 30) / 365.25) + 0.08 * (month == 12)
    state_hol = np.where((month == 1) & (dates.day == 1), "a", np.where((month == 12) & (dates.day == 25), "a", "0"))
    rows = []
    for s in range(1, n_stores + 1):
        store_type = rng.choice(list("abcd"))
        assortment = rng.choice(list("abc"), p=[0.5, 0.1, 0.4])
        comp_dist = float(np.clip(np.exp(rng.normal(7.0, 1.0)), 50, 20000))
        level = np.exp(rng.normal(8.0, 0.35))  # vendas base por dia (moeda fictícia)
        # promoções: blocos de 5 dias, ~30% das semanas
        promo = np.zeros(T, dtype=int)
        for w in range(0, T, 7):
            if rng.random() < 0.3:
                start = w + rng.integers(0, 3)
                promo[start : start + 5] = 1
        promo[dow == 6] = 0
        school = (rng.random(T) < 0.03).astype(int)
        closed = (dow == 6) | (state_hol != "0")
        mu = level * trend * dow_factor[dow] * annual * (1 + 0.25 * promo) * (1 + 0.05 * school)
        noise = np.exp(rng.normal(0, 0.08, T))
        sales = np.where(closed, 0, np.round(mu * noise)).astype(int)
        customers = np.where(closed, 0, rng.poisson(np.maximum(sales, 0) / 9.0 + 1)).astype(int)
        rows.append(pd.DataFrame({
            "Store": s, "Date": dates, "Sales": sales, "Customers": customers,
            "Open": (~closed).astype(int), "Promo": promo, "StateHoliday": state_hol,
            "SchoolHoliday": school, "StoreType": store_type, "Assortment": assortment,
            "CompetitionDistance": round(comp_dist, 0),
        }))
    df = pd.concat(rows, ignore_index=True)
    df["Store"] = df["Store"].astype("int16")
    for c in ("StateHoliday", "StoreType", "Assortment"):
        df[c] = df[c].astype("category")
    df["Promo"] = df["Promo"].astype("int8")
    df["Open"] = df["Open"].astype("int8")
    df["SchoolHoliday"] = df["SchoolHoliday"].astype("int8")
    df["Customers"] = df["Customers"].astype("int32")
    df["Sales"] = df["Sales"].astype("int32")
    return df


def main() -> None:
    df = generate()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_DIR / "sales_sample.parquet", compression="zstd", index=False)
    print(df.shape, df.dtypes.to_dict())


if __name__ == "__main__":
    main()
