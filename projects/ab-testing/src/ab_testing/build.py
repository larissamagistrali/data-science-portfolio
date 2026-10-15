"""Analisa o Cookie Cats e grava os artefatos versionados.

Uso:  PYTHONPATH=projects/ab-testing/src python -m ab_testing.build
Saídas em projects/ab-testing/data/: cookie_cats.parquet (linhas originais, tipos enxutos) e metrics.json.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import data as D
from . import stats as S

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
A, B = "gate_30", "gate_40"  # A = controle (portão no nível 30), B = tratamento (portão no nível 40)


def analyze_binary(df: pd.DataFrame, col: str) -> dict:
    a, b = df[df["version"] == A][col], df[df["version"] == B][col]
    x1, n1, x2, n2 = int(a.sum()), len(a), int(b.sum()), len(b)
    z = S.two_proportion_ztest(x1, n1, x2, n2)
    return {
        "A": {"n": n1, "conversoes": x1, "taxa": x1 / n1, "ic95": S.wilson_ci(x1, n1)},
        "B": {"n": n2, "conversoes": x2, "taxa": x2 / n2, "ic95": S.wilson_ci(x2, n2)},
        "diff": z["diff"], "diff_relativa": z["diff"] / (x1 / n1),
        "z": z["z"], "p_value": z["p_value"],
        "qui_quadrado": S.chi_square_test(x1, n1, x2, n2),
        "ic95_diff_wald": S.diff_ci_wald(x1, n1, x2, n2),
        "ic95_diff_newcombe": S.diff_ci_newcombe(x1, n1, x2, n2),
        "bootstrap": S.bootstrap_proportion_diff_ci(x1, n1, x2, n2),
        "bayes": S.bayesian_ab(x1, n1, x2, n2),
    }


def analyze_rounds(df: pd.DataFrame) -> dict:
    a = df[df["version"] == A]["sum_gamerounds"].to_numpy()
    b = df[df["version"] == B]["sum_gamerounds"].to_numpy()
    mw = sps.mannwhitneyu(b, a, alternative="two-sided")
    return {
        "A": {"media": float(a.mean()), "mediana": float(np.median(a)), "max": int(a.max())},
        "B": {"media": float(b.mean()), "mediana": float(np.median(b)), "max": int(b.max())},
        "mann_whitney_p": float(mw.pvalue),
        "bootstrap_media": S.bootstrap_diff_ci(a, b, "mean", n_boot=1000),
        "bootstrap_mediana": S.bootstrap_diff_ci(a, b, "median", n_boot=1000),
        "usuarios_zero_rodadas": int((df["sum_gamerounds"] == 0).sum()),
    }


def main() -> None:
    df = D.load_real()
    ret1, ret7, rounds = analyze_binary(df, "retention_1"), analyze_binary(df, "retention_7"), analyze_rounds(df)
    pvals = {"retention_1": ret1["p_value"], "retention_7": ret7["p_value"], "sum_gamerounds (Mann-Whitney)": rounds["mann_whitney_p"]}
    names = list(pvals)
    adjusted = {m: dict(zip(names, S.adjust_pvalues(list(pvals.values()), m).tolist())) for m in ("bonferroni", "holm", "bh")}
    n_a, n_b = int((df["version"] == A).sum()), int((df["version"] == B).sum())
    metrics = {
        "fonte": "Cookie Cats (Tactile Games / DataCamp), espelho público no GitHub",
        "gerado_em": date.today().isoformat(),
        "linhas": len(df),
        "srm": S.srm_test(n_a, n_b),
        "retention_1": ret1,
        "retention_7": ret7,
        "gamerounds": rounds,
        "p_valores": pvals,
        "p_ajustados": adjusted,
        "peeking": S.simulate_peeking(p=0.1, n_per_group=5000, looks=10, n_sims=4000),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["version"] = out["version"].astype("category")
    out.to_parquet(DATA_DIR / "cookie_cats.parquet", compression="zstd", index=False)
    (DATA_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("linhas", "srm", "p_valores", "p_ajustados", "peeking")}, indent=2))
    for k in ("retention_1", "retention_7"):
        r = metrics[k]
        print(k, {kk: r[kk] for kk in ("diff", "p_value", "ic95_diff_newcombe")}, r["A"]["taxa"], r["B"]["taxa"], r["bayes"]["p_b_beats_a"], r["bayes"]["expected_loss_choose_b"])
    print(json.dumps(rounds, indent=1))


if __name__ == "__main__":
    main()
