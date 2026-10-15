"""Transformações das séries: base 100, variação em 12 meses e correlação defasada."""
from __future__ import annotations

import numpy as np
import pandas as pd


def to_monthly(df: pd.DataFrame, how: str = "last") -> pd.Series:
    """Série mensal (índice = início do mês) a partir de (data, valor); 'last' ou 'mean' por mês."""
    s = df.set_index("data")["valor"].sort_index()
    agg = s.resample("MS").last() if how == "last" else s.resample("MS").mean()
    return agg.dropna()


def index_base100(s: pd.Series, base: pd.Timestamp | None = None) -> pd.Series:
    """Índice base 100 na data-base (padrão: primeiro ponto da série em comum)."""
    s = s.dropna()
    b = base if base is not None else s.index[0]
    return s / s.loc[b] * 100.0


def accumulate_monthly_rate(s: pd.Series) -> pd.Series:
    """Índice acumulado de uma variação mensal em %: produto de (1 + r/100)."""
    return (1.0 + s.dropna() / 100.0).cumprod()


def comparable_level(s: pd.Series, unidade: str) -> pd.Series | None:
    """Nível que admite base 100, ou None.

    Variação mensal (% a.m.) vira índice acumulado: indexar a própria taxa daria números sem
    sentido (0,3% -> 0,6% vira "200"). Níveis (R$/US$) passam direto. Taxas em nível (% a.a., %)
    não entram: base 100 de uma taxa mede a variação da taxa, não se compara a um índice de preços.
    """
    if unidade == "% a.m.":
        return accumulate_monthly_rate(s)
    if unidade.startswith("%"):
        return None
    return s


def yoy_change(s: pd.Series, periods: int = 12) -> pd.Series:
    """Variação percentual sobre o mesmo mês do ano anterior (12 meses)."""
    return (s / s.shift(periods) - 1.0) * 100.0


def lagged_correlation(x: pd.Series, y: pd.Series, max_lag: int = 12) -> pd.DataFrame:
    """Correlação de Pearson entre x_t e y_{t+k} para k em [-max_lag, max_lag], após diferenciar (mudança mensal).

    Diferenciar evita correlações espúrias de séries com tendência. k > 0: y responde com atraso de k meses a x.
    """
    dx, dy = x.diff(), y.diff()
    joined = pd.concat([dx, dy], axis=1, keys=["x", "y"]).dropna()
    rows = []
    for k in range(-max_lag, max_lag + 1):
        shifted = pd.concat([joined["x"], joined["y"].shift(-k)], axis=1).dropna()
        n = len(shifted)
        r = float(np.corrcoef(shifted.iloc[:, 0], shifted.iloc[:, 1])[0, 1]) if n > 3 else np.nan
        rows.append({"lag_meses": k, "correlacao": r, "n": n})
    return pd.DataFrame(rows)


def ranking_uf(df: pd.DataFrame, periodo: str) -> pd.DataFrame:
    """Ranking por UF de um período (rótulo SIDRA, ex.: '2025' ou '1º trimestre 2026'), sem o Brasil."""
    sub = df[(df["periodo"] == periodo) & (df["codigo_territorio"] >= 11) & (df["codigo_territorio"] <= 53)]
    return sub.sort_values("valor", ascending=True).reset_index(drop=True)


MESES_PT = {m: i for i, m in enumerate(
    ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"], start=1)}


def parse_periodo_sidra(texto: str) -> pd.Timestamp:
    """Converte o rótulo SIDRA em data de início do período.

    'julho 2026' -> 2026-07-01; '2º trimestre 2026' -> 2026-04-01; '2025' -> 2025-01-01.
    """
    partes = str(texto).strip().lower().split()
    if len(partes) == 2 and partes[0] in MESES_PT:
        return pd.Timestamp(year=int(partes[1]), month=MESES_PT[partes[0]], day=1)
    if len(partes) == 3 and "trimestre" in partes[1]:
        trimestre = int(partes[0][0])
        return pd.Timestamp(year=int(partes[2]), month=3 * (trimestre - 1) + 1, day=1)
    if len(partes) == 1 and partes[0].isdigit():
        return pd.Timestamp(year=int(partes[0]), month=1, day=1)
    raise ValueError(f"período SIDRA não reconhecido: {texto!r}")
