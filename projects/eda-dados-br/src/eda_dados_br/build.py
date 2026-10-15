"""Coleta as séries e grava o snapshot versionado (parquet pequeno + meta.json com a data da coleta).

Uso: PYTHONPATH=projects/eda-dados-br/src python -m eda_dados_br.build [--refresh]
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from . import analysis as A
from . import coleta as C

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SIDRA_SPECS = {
    "populacao_uf": dict(table=6579, territory="n3/all", classes="", var=9324, period="all"),
    "desocupacao_uf_trimestre": dict(table=4099, territory="n3/all", classes="", var=4099, period="all"),
}


def build(refresh: bool = False) -> dict:
    raw = C.RAW_DIR
    frames = []
    for s in C.SGS_SERIES:
        df = C.cached(raw / f"sgs_{s.code}.parquet", lambda s=s: C.fetch_sgs(s), refresh)
        df = df.assign(serie=s.nome, codigo=s.code, unidade=s.unidade, fonte=s.fonte_nome)
        frames.append(df)
    sgs = pd.concat(frames, ignore_index=True)
    sgs["data"] = pd.to_datetime(sgs["data"])
    sgs.to_parquet(DATA_DIR / "bcb_sgs.parquet", compression="zstd", index=False)

    ipca_idx = C.cached(raw / "sidra_1737_ipca_indice.parquet", lambda: C.fetch_sidra(1737, "n1/1", "", 2266, "all"), refresh)
    ipca_idx = ipca_idx.assign(data=ipca_idx["periodo"].map(A.parse_periodo_sidra))
    ipca_idx.to_parquet(DATA_DIR / "sidra_ipca_indice.parquet", compression="zstd", index=False)

    for name, spec in SIDRA_SPECS.items():
        classes = spec["classes"]
        df = C.cached(raw / f"sidra_{name}.parquet",
                      lambda spec=spec: C.fetch_sidra(spec["table"], spec["territory"], classes, spec["var"], spec["period"]), refresh)
        df = df.assign(data=df["periodo"].map(A.parse_periodo_sidra))
        df.to_parquet(DATA_DIR / f"sidra_{name}.parquet", compression="zstd", index=False)

    meta = {
        "coletado_em": date.today().isoformat(),
        "user_agent": C.USER_AGENT,
        "fontes": {
            "BCB SGS": "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados (dados abertos do Banco Central do Brasil)",
            "IBGE SIDRA": "https://apisidra.ibge.gov.br (IBGE, dados abertos)",
        },
        "series_sgs": [{"codigo": s.code, "nome": s.nome, "unidade": s.unidade, "linhas": int((sgs["codigo"] == s.code).sum()),
                        "inicio": str(sgs.loc[sgs["codigo"] == s.code, "data"].min().date()),
                        "fim": str(sgs.loc[sgs["codigo"] == s.code, "data"].max().date())} for s in C.SGS_SERIES],
        "sidra": {k: {"tabela": v["table"], "variavel": v["var"]} for k, v in SIDRA_SPECS.items()} | {"ipca_indice": {"tabela": 1737, "variavel": 2266}},
    }
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


if __name__ == "__main__":
    print(json.dumps(build(refresh="--refresh" in sys.argv), indent=2, ensure_ascii=False))
