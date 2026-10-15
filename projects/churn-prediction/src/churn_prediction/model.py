"""Pipelines de modelagem (pré-processamento + estimador) e métricas de probabilidade."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import BINARY_01, CATEGORICAL, NUMERIC

SEED = 42


def build_preprocessor(extra_numeric: tuple[str, ...] = ()) -> ColumnTransformer:
    """Numéricas padronizadas; categóricas em one-hot (binárias viram uma coluna só)."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC + BINARY_01 + list(extra_numeric)),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False), CATEGORICAL),
        ],
        verbose_feature_names_out=True,
    )


def build_logreg(C: float = 1.0, extra_numeric: tuple[str, ...] = ()) -> Pipeline:
    return Pipeline(
        [("prep", build_preprocessor(extra_numeric)), ("clf", LogisticRegression(C=C, max_iter=3000, random_state=SEED))]
    )


def build_hgb(learning_rate: float = 0.05, max_iter: int = 200, max_leaf_nodes: int = 8, l2_regularization: float = 1.0) -> Pipeline:
    clf = HistGradientBoostingClassifier(
        learning_rate=learning_rate, max_iter=max_iter, max_leaf_nodes=max_leaf_nodes,
        l2_regularization=l2_regularization, random_state=SEED,
    )
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """ECE com bins de mesma frequência: média ponderada de |frequência observada - probabilidade média|."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    order = np.argsort(p)
    ece = 0.0
    for idx in np.array_split(order, bins):
        if len(idx):
            ece += len(idx) / len(y) * abs(y[idx].mean() - p[idx].mean())
    return float(ece)


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Probabilidade média prevista x frequência observada, por decil de probabilidade."""
    order = np.argsort(np.asarray(p))
    rows = [
        {"p_mean": float(np.mean(np.asarray(p)[i])), "freq": float(np.mean(np.asarray(y)[i])), "n": int(len(i))}
        for i in np.array_split(order, bins)
        if len(i)
    ]
    return pd.DataFrame(rows)


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
    }
