"""Treina, valida e salva os artefatos do projeto de churn.

Uso (na raiz do repo):  PYTHONPATH=projects/churn-prediction/src python -m churn_prediction.train

Saídas em projects/churn-prediction/data/: models.joblib, metrics.json, sample.parquet, test_predictions.parquet.
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate, train_test_split

from . import data as D
from .decision import best_threshold, sweep_thresholds
from .explain import fit_reference
from .model import SEED, build_hgb, build_logreg, calibration_table, probability_metrics

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TEST_SIZE = 0.2
N_SPLITS = 5
MONTHS_OF_VALUE = 12  # valor do cliente = MonthlyCharges x 12 meses (parâmetro do app)
DEFAULT_COST = 60.0  # custo da ação de retenção (US$; valor ilustrativo, ajustável no app)
DEFAULT_SUCCESS = 0.25  # taxa de sucesso da ação (ilustrativa)
SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "neg_brier": "neg_brier_score"}


def _cv_summary(res: dict) -> dict[str, float]:
    out = {}
    for key, name in (("roc_auc", "roc_auc"), ("pr_auc", "pr_auc"), ("neg_brier", "brier")):
        vals = res[f"test_{key}"] * (-1 if key == "neg_brier" else 1)
        out[name] = float(vals.mean())
        out[f"{name}_std"] = float(vals.std(ddof=1))
    return out


def _bootstrap_ci(y: np.ndarray, p: np.ndarray, n: int = 1000) -> dict[str, list[float]]:
    rng = np.random.default_rng(SEED)
    aucs, aps = [], []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
        aps.append(average_precision_score(y[idx], p[idx]))
    return {"roc_auc_ci95": np.percentile(aucs, [2.5, 97.5]).tolist(), "pr_auc_ci95": np.percentile(aps, [2.5, 97.5]).tolist()}


def _calibrated(base, method: str):
    return base if method == "none" else CalibratedClassifierCV(clone(base), method=method, cv=N_SPLITS)


def main() -> None:
    raw = pd.read_csv(D.download())
    df = D.clean(raw)
    X, y = df[D.FEATURES], df[D.TARGET].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    metrics: dict = {
        "dataset": {"linhas": len(df), "colunas_originais": int(raw.shape[1]), "taxa_churn": float(y.mean()),
                    "treino": len(X_tr), "teste": len(X_te), "total_charges_em_branco": int((raw["TotalCharges"].astype(str).str.strip() == "").sum())},
        "gerado_em": date.today().isoformat(),
        "versoes": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__},
    }

    # Ablação: TotalCharges (~ tenure x MonthlyCharges) muda a regressão logística?
    abl = {}
    for name, extra in (("sem_total_charges", ()), ("com_total_charges", ("TotalCharges",))):
        cols = D.FEATURES + list(extra)
        r = cross_validate(build_logreg(1.0, extra), df.loc[X_tr.index, cols], y_tr, cv=cv, scoring="roc_auc")
        abl[name] = {"cv_roc_auc": float(r["test_score"].mean()), "cv_roc_auc_std": float(r["test_score"].std(ddof=1))}
    corr = float(np.corrcoef(df["TotalCharges"], df["tenure"] * df["MonthlyCharges"])[0, 1])
    metrics["ablacao_total_charges"] = {**abl, "corr_total_vs_tenure_x_monthly": corr}

    # Busca em grade (CV estratificada, só no treino).
    grids = {
        "logreg": (build_logreg(), {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}),
        "hgb": (build_hgb(), {"clf__learning_rate": [0.03, 0.06], "clf__max_leaf_nodes": [4, 8, 16],
                              "clf__l2_regularization": [1.0, 10.0], "clf__max_iter": [150, 300]}),
    }
    best, cv_metrics = {}, {}
    for name, (pipe, grid) in grids.items():
        gs = GridSearchCV(pipe, grid, scoring=SCORING, refit="roc_auc", cv=cv, n_jobs=4).fit(X_tr, y_tr)
        best[name] = gs.best_estimator_
        i = gs.best_index_
        cv_metrics[name] = {
            "params": {k.split("__")[1]: v for k, v in gs.best_params_.items()},
            "roc_auc": float(gs.cv_results_["mean_test_roc_auc"][i]), "roc_auc_std": float(gs.cv_results_["std_test_roc_auc"][i]),
            "pr_auc": float(gs.cv_results_["mean_test_pr_auc"][i]), "pr_auc_std": float(gs.cv_results_["std_test_pr_auc"][i]),
            "brier": float(-gs.cv_results_["mean_test_neg_brier"][i]), "brier_std": float(gs.cv_results_["std_test_neg_brier"][i]),
        }
    metrics["cv"] = {"n_splits": N_SPLITS, **cv_metrics}

    # Calibração: compara none / sigmoid (Platt) / isotonic por Brier em CV (só no treino).
    chosen_cal, cal_cv = {}, {}
    for name, base in best.items():
        scores = {}
        for method in ("none", "sigmoid", "isotonic"):
            r = cross_validate(_calibrated(base, method), X_tr, y_tr, cv=cv, scoring="neg_brier_score", n_jobs=4)
            scores[method] = float(-r["test_score"].mean())
        chosen_cal[name] = min(scores, key=scores.get)
        cal_cv[name] = scores
    metrics["calibracao_cv_brier"] = cal_cv
    metrics["calibracao_escolhida"] = chosen_cal

    # Treino final (só treino) + avaliação no teste, com e sem calibração.
    final, test_p = {}, {}
    test_table = {}
    for name, base in best.items():
        per_method = {}
        for method in ("none", "sigmoid", "isotonic"):
            m = _calibrated(base, method)
            m = clone(m).fit(X_tr, y_tr) if method != "none" else base
            per_method[method] = (m, m.predict_proba(X_te)[:, 1])
        test_table[name] = {k: probability_metrics(y_te, v[1]) for k, v in per_method.items()}
        final[name], test_p[name] = per_method[chosen_cal[name]]
    metrics["teste_por_calibracao"] = test_table
    metrics["teste"] = {n: {**probability_metrics(y_te, test_p[n]), **_bootstrap_ci(y_te, test_p[n])} for n in final}
    metrics["teste"]["baseline_prevalencia"] = {"pr_auc": float(y_te.mean())}
    metrics["calibracao_tabela_teste"] = {n: calibration_table(y_te, test_p[n]).round(4).to_dict("records") for n in final}

    # Campeão: regra de 1 erro-padrão (prefere o modelo simples se não for pior que o melhor por mais de 1 EP).
    best_name = max(cv_metrics, key=lambda n: cv_metrics[n]["roc_auc"])
    se = cv_metrics[best_name]["roc_auc_std"] / np.sqrt(N_SPLITS)
    champion = "logreg" if cv_metrics["logreg"]["roc_auc"] >= cv_metrics[best_name]["roc_auc"] - se else best_name
    metrics["campeao"] = {"modelo": champion, "regra": "1 erro-padrão (CV ROC-AUC); em empate técnico, o modelo linear",
                          "melhor_cv": best_name, "erro_padrao_melhor": float(se)}

    # Limiar padrão: escolhido nas predições fora-da-dobra do TREINO (sem espiar o teste).
    oof = cross_val_predict(_calibrated(best[champion], chosen_cal[champion]), X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=4)[:, 1]
    value_tr = X_tr["MonthlyCharges"].to_numpy() * MONTHS_OF_VALUE
    thr = best_threshold(sweep_thresholds(y_tr, oof, value_tr, DEFAULT_COST, DEFAULT_SUCCESS))
    value_te = X_te["MonthlyCharges"].to_numpy() * MONTHS_OF_VALUE
    sweep_te = sweep_thresholds(y_te, test_p[champion], value_te, DEFAULT_COST, DEFAULT_SUCCESS, np.array([0.0, thr, 1.01]))
    todos, escolhido, nenhum = sweep_te.iloc[0], sweep_te.iloc[1], sweep_te.iloc[2]
    metrics["decisao_padrao"] = {
        "custo_acao": DEFAULT_COST, "taxa_sucesso": DEFAULT_SUCCESS, "meses_de_valor": MONTHS_OF_VALUE,
        "limiar": thr, "limiar_escolhido_em": "predições fora-da-dobra do treino",
        "teste": {"abordados": int(escolhido["targeted"]), "abordados_pct": float(escolhido["targeted"] / len(y_te)),
                  "recall": float(escolhido["recall"]), "precisao": float(escolhido["precision"]),
                  "ganho_liquido": float(escolhido["net_gain"]), "ganho_abordar_todos": float(todos["net_gain"]),
                  "ganho_nao_abordar": float(nenhum["net_gain"])},
    }

    # Artefatos.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    base_lr = best["logreg"]
    joblib.dump(
        {"logreg": final["logreg"], "logreg_base": base_lr, "hgb": final["hgb"], "z_bar": fit_reference(base_lr, X_tr),
         "champion": champion, "features": D.FEATURES, "sklearn_version": sklearn.__version__},
        DATA_DIR / "models.joblib", compress=3,
    )
    sample = df.drop(columns=["TotalCharges"]).copy()
    sample["split"] = np.where(sample.index.isin(X_te.index), "teste", "treino")
    for col in D.CATEGORICAL + ["split"]:
        sample[col] = sample[col].astype("category")
    sample.to_parquet(DATA_DIR / "sample.parquet", compression="zstd", index=False)
    pd.DataFrame({"y": y_te.astype("int8"), "p_logreg": test_p["logreg"].astype("float32"), "p_hgb": test_p["hgb"].astype("float32"),
                  "MonthlyCharges": X_te["MonthlyCharges"].to_numpy().astype("float32")}).to_parquet(
        DATA_DIR / "test_predictions.parquet", compression="zstd", index=False)
    (DATA_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("cv", "calibracao_escolhida", "teste", "campeao", "decisao_padrao")}, indent=2, ensure_ascii=False))
    print("python", sys.version.split()[0], "| artefatos em", DATA_DIR)


if __name__ == "__main__":
    main()
