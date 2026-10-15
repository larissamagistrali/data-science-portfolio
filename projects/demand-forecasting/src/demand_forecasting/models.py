"""Modelos de previsão: seasonal-naive, Holt-Winters (ETS) e gradient boosting com lags seguros.

Todos recebem o histórico até a origem e devolvem previsões para origem+1 .. origem+H.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

HORIZON = 28
MODELS = ("seasonal_naive", "holt_winters", "gbm")
CAL_FEATURES = ["dow", "month", "doy", "dom", "Promo", "SchoolHoliday", "is_holiday", "StoreType", "Assortment", "log_comp"]


def full_calendar(df: pd.DataFrame, store: int) -> pd.Series:
    """Série diária completa (dias sem registro = 0) de Sales de uma loja."""
    s = df.loc[df["Store"] == store].set_index("Date")["Sales"].astype(float)
    return s.asfreq("D", fill_value=0.0)


MAX_WEEKS_BACK = 8


def seasonal_naive(hist: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """Y_t = último valor positivo no mesmo dia da semana (t - 7k, o menor k no histórico).

    Dias com venda zero (loja fechada, feriado) são pulados: o calendário de fechamento é
    conhecido, então copiar o zero de um feriado para um dia aberto seria um erro evitável.
    """
    origin = hist.index[-1]
    out = []
    for t in dates:
        k0 = int(np.ceil((t - origin).days / 7))
        value = np.nan
        for k in range(k0, k0 + MAX_WEEKS_BACK):
            v = hist.get(t - pd.Timedelta(days=7 * k), np.nan)
            if v > 0:
                value = v
                break
        out.append(value)
    return np.array(out, dtype=float)


def holt_winters(hist: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """ETS aditivo com tendência amortecida e sazonalidade semanal (m = 7)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(hist.to_numpy(), trend="add", damped_trend=True, seasonal="add", seasonal_periods=7,
                                   initialization_method="estimated").fit(optimized=True)
        fc = fit.forecast(len(dates))
    return np.maximum(np.asarray(fc, dtype=float), 0.0)


def _features(frame: pd.DataFrame, static: pd.Series, lag_hist: pd.Series) -> pd.DataFrame:
    """Features conhecidas no momento t: calendário, plano de promoções/feriados, estáticas da loja e
    lags de 28 e 364 dias (ambos <= origem para horizonte <= 28)."""
    f = pd.DataFrame(index=frame.index)
    d = frame["Date"]
    f["dow"] = d.dt.dayofweek
    f["month"] = d.dt.month
    f["doy"] = d.dt.dayofyear
    f["dom"] = d.dt.day
    f["Promo"] = frame["Promo"].to_numpy()
    f["SchoolHoliday"] = frame["SchoolHoliday"].to_numpy()
    f["is_holiday"] = (frame["StateHoliday"].astype(str) != "0").astype(int).to_numpy()
    f["StoreType"] = str(static["StoreType"])
    f["Assortment"] = str(static["Assortment"])
    f["log_comp"] = np.log1p(float(static["CompetitionDistance"]))
    f["lag28"] = lag_hist.reindex(d - pd.Timedelta(days=28)).to_numpy()  # lag_hist: série completa da loja
    f["lag364"] = lag_hist.reindex(d - pd.Timedelta(days=364)).to_numpy()
    f["roll28"] = lag_hist.rolling(28, min_periods=7).mean().reindex(d - pd.Timedelta(days=28)).to_numpy()
    return f


GBM_COLS = CAL_FEATURES + ["lag28", "lag364", "roll28"]


def _matrix(X: pd.DataFrame, levels: dict[str, list[str]]) -> pd.DataFrame:
    """Converte categóricas em códigos com níveis fixos (mesmos no treino e na previsão)."""
    X = X.copy()
    for col, lv in levels.items():
        X[col] = pd.Categorical(X[col].astype(str), categories=lv).codes.astype(float)
    return X[GBM_COLS].astype(float)


def _calendar(df: pd.DataFrame, store: int) -> pd.DataFrame:
    """Calendário diário completo de uma loja, indexado por Date (vendas ausentes = 0)."""
    cal = df.loc[df["Store"] == store].set_index("Date").asfreq("D")
    cal.index.name = "Date"
    cal["Sales"] = cal["Sales"].fillna(0.0)
    return cal


def gbm_fit(df: pd.DataFrame, origin: pd.Timestamp, store_ids: list[int]) -> tuple[HistGradientBoostingRegressor, dict]:
    """Modelo global treinado com alvos até a origem; cada linha usa só informação disponível em t."""
    levels = {"StoreType": sorted(df["StoreType"].astype(str).unique()), "Assortment": sorted(df["Assortment"].astype(str).unique())}
    frames = []
    for s in store_ids:
        cal = _calendar(df, s)
        window = cal.loc["2014-01-01": origin].rename_axis("Date").reset_index()
        static = cal.iloc[0][["StoreType", "Assortment", "CompetitionDistance"]]
        X = _features(window, static, cal["Sales"])
        X["y"] = window["Sales"].to_numpy()
        X["Open"] = window["Open"].to_numpy()
        frames.append(X.dropna(subset=["lag28", "lag364", "roll28"]))
    data = pd.concat(frames)
    data = data[data["Open"] == 1]
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=42)
    model.fit(_matrix(data.drop(columns=["y", "Open"]), levels), data["y"].to_numpy())
    return model, levels


def gbm_predict(model, levels: dict, df: pd.DataFrame, store: int, dates: pd.DatetimeIndex) -> np.ndarray:
    cal = _calendar(df, store)
    frame = cal.loc[dates].rename_axis("Date").reset_index()
    static = cal.iloc[0][["StoreType", "Assortment", "CompetitionDistance"]]
    X = _features(frame, static, cal["Sales"])
    X["StoreType"] = str(static["StoreType"])
    X["Assortment"] = str(static["Assortment"])
    return np.maximum(model.predict(_matrix(X, levels)), 0.0)


# --- métricas -------------------------------------------------------------------------------------

def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.abs(y - yhat).sum() / np.abs(y).sum())


def mape(y: np.ndarray, yhat: np.ndarray) -> float:
    m = y > 0
    return float(np.mean(np.abs(y[m] - yhat[m]) / y[m]))


def mase_scale(hist: pd.Series, m: int = 7) -> float:
    """MAE in-sample do naive sazonal (m=7) no histórico: denominador do MASE."""
    d = np.abs(hist.to_numpy()[m:] - hist.to_numpy()[:-m])
    return float(d.mean()) if d.mean() > 0 else np.nan
