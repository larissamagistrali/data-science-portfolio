"""Explicação exata da regressão logística: contribuição de cada variável em log-odds.

logit(cliente) = b + sum_k w_k * z_k  (z = variáveis já pré-processadas).
Medimos cada variável contra o cliente "médio" da base de treino (z_bar), então:

    logit(cliente) = logit_medio + sum_variaveis contribuicao_variavel     (identidade exata)

onde contribuicao_variavel = sum_{k em variavel} w_k * (z_k - z_bar_k) e logit_medio = b + sum_k w_k * z_bar_k
é a média, na base de treino, do log-odds do modelo. Não é SHAP nem aproximação: é a decomposição
aditiva do próprio modelo linear. Variáveis categóricas (one-hot) são somadas de volta à variável original.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def _group_of(feature_name: str, original: list[str]) -> str:
    # nomes: "num__tenure", "cat__Contract_Month-to-month"; escolhe a maior coluna original que casa
    base = feature_name.split("__", 1)[1]
    matches = [c for c in original if base == c or base.startswith(f"{c}_")]
    return max(matches, key=len) if matches else base


def fit_reference(pipe: Pipeline, X_train: pd.DataFrame) -> np.ndarray:
    """Vetor médio (z_bar) das variáveis pré-processadas na base de treino."""
    return pipe.named_steps["prep"].transform(X_train).mean(axis=0)


def contributions(pipe: Pipeline, X_row: pd.DataFrame, z_bar: np.ndarray) -> tuple[pd.DataFrame, float, float]:
    """Contribuições (em log-odds) por variável original para uma linha.

    Devolve (tabela ordenada por |contribuição|, logit_medio, logit_do_cliente).
    """
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    z = prep.transform(X_row)[0]
    w = clf.coef_[0]
    names = list(prep.get_feature_names_out())
    original = list(X_row.columns)
    per_col = w * (z - z_bar)
    df = pd.DataFrame({"feature": [_group_of(n, original) for n in names], "contribution": per_col})
    grouped = df.groupby("feature", as_index=False)["contribution"].sum()
    grouped["value"] = grouped["feature"].map(lambda c: X_row.iloc[0][c])
    grouped = grouped.reindex(grouped["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    base = float(clf.intercept_[0] + w @ z_bar)
    logit = float(clf.intercept_[0] + w @ z)
    return grouped, base, logit
