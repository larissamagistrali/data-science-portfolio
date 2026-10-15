"""Pipeline: Online Retail II -> RFM -> KMeans -> agregados versionados.

Uso:  PYTHONPATH=projects/customer-segmentation/src python -m customer_segmentation.build
Saídas em projects/customer-segmentation/data/: rfm.parquet (um registro por cliente, ID pseudonimizado)
e metrics.json (limpeza, escolha de k, perfil dos clusters, comparação com quintis RFM).
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import cluster as C
from . import data as D
from .rfm import build_rfm

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def main() -> None:
    raw = D.load_raw()
    valid, cancelled, stats = D.clean(raw)
    rfm = build_rfm(valid, cancelled)
    Z, _ = C.transform(rfm)
    table = C.select_k(Z)
    k = C.best_k(table)
    km = C.fit_kmeans(Z, k)
    centroids = pd.DataFrame(km.cluster_centers_, columns=C.RFM_COLS)
    names = C.name_segments(centroids)
    prof = C.profile(rfm, km.labels_, names)
    quint = C.quintile_segments(rfm)
    comparison = C.compare_segmentations(Z, km.labels_, quint["segmento_quintil"])
    quint_profile = (
        rfm.assign(segmento=quint["segmento_quintil"]).groupby("segmento")
        .agg(clientes=("customer", "size"), recencia_mediana_dias=("recency_days", "median"),
             frequencia_mediana=("frequency", "median"), monetario_mediano=("monetary", "median"), receita_total=("monetary", "sum"))
        .assign(pct_receita=lambda d: d["receita_total"] / d["receita_total"].sum()).reset_index()
    )
    cross = pd.crosstab(pd.Series(prof.set_index("cluster")["segmento"].reindex(km.labels_).to_numpy(), name="kmeans"), quint["segmento_quintil"].rename("quintis"))

    metrics = {
        "fonte": "Online Retail II (UCI), CC BY 4.0",
        "gerado_em": date.today().isoformat(),
        "periodo": {"inicio": str(valid["InvoiceDate"].min().date()), "fim": str(valid["InvoiceDate"].max().date()),
                    "data_referencia": str((valid["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)).date())},
        "limpeza": stats,
        "clientes_rfm": len(rfm),
        "k_escolhido": k,
        "regra_k": "maior silhouette entre k=3 e k=8",
        "selecao_k": table.round(4).to_dict("records"),
        "perfil_clusters": prof.round(4).to_dict("records"),
        "centroides_z": centroids.round(4).assign(segmento=pd.Series(names)).to_dict("records"),
        "quintis_perfil": quint_profile.round(4).to_dict("records"),
        "comparacao": comparison,
        "cruzamento_kmeans_quintis": json.loads(cross.to_json(orient="index")),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rfm.to_parquet(DATA_DIR / "rfm.parquet", compression="zstd", index=False)
    (DATA_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k_: metrics[k_] for k_ in ("periodo", "limpeza", "clientes_rfm", "k_escolhido", "comparacao")}, indent=2, ensure_ascii=False))
    print(table.round(4).to_string(index=False))
    print(prof.round(3).to_string(index=False))
    print(quint_profile.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
