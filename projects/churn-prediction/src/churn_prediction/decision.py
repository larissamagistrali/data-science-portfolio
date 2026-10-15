"""Decisão por custo de negócio: quem abordar com uma ação de retenção.

Modelo de valor esperado por cliente abordado:  p * s * V - c
- p: probabilidade de churn (calibrada), s: taxa de sucesso da ação (fração dos que iam sair e ficam),
- V: valor do cliente se retido, c: custo da ação. Abordar compensa se p > c / (s * V).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def breakeven_probability(cost: float, success_rate: float, value: float) -> float:
    """Menor probabilidade de churn para a qual abordar o cliente tem valor esperado >= 0."""
    if success_rate <= 0 or value <= 0:
        return float("inf")
    return cost / (success_rate * value)


def expected_gain(p: np.ndarray | float, value: np.ndarray | float, cost: float, success_rate: float) -> np.ndarray | float:
    """Ganho esperado de abordar: p * s * V - c."""
    return np.asarray(p) * success_rate * np.asarray(value) - cost


def sweep_thresholds(
    y: np.ndarray, p: np.ndarray, value: np.ndarray, cost: float, success_rate: float, thresholds: np.ndarray | None = None
) -> pd.DataFrame:
    """Resultado de abordar todos com p >= limiar, para cada limiar (com os rótulos reais).

    Ganho líquido = soma, sobre os abordados, de (y * s * V - c): cada churner abordado é retido com
    probabilidade s (valor esperado s * V); todo abordado custa c. Colunas: threshold, targeted, tp,
    precision, recall, net_gain.
    """
    y, p, value = np.asarray(y, dtype=float), np.asarray(p, dtype=float), np.asarray(value, dtype=float)
    thresholds = np.linspace(0.0, 1.0, 101) if thresholds is None else thresholds
    total_pos = y.sum()
    rows = []
    for t in thresholds:
        sel = p >= t
        n, tp = int(sel.sum()), float(y[sel].sum())
        rows.append(
            {
                "threshold": float(t),
                "targeted": n,
                "tp": int(tp),
                "precision": tp / n if n else float("nan"),
                "recall": tp / total_pos if total_pos else float("nan"),
                "net_gain": float((y[sel] * success_rate * value[sel]).sum() - cost * n),
            }
        )
    return pd.DataFrame(rows)


def best_threshold(sweep: pd.DataFrame) -> float:
    """Limiar que maximiza o ganho líquido (em empate, o menor número de abordados)."""
    best = sweep.sort_values(["net_gain", "targeted"], ascending=[False, True]).iloc[0]
    return float(best["threshold"])
