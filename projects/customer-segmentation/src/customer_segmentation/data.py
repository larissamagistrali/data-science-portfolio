"""Download, leitura e limpeza do Online Retail II (UCI, CC BY 4.0).

O arquivo bruto (zip com um .xlsx de ~45 MB) fica fora do Git, em ``data/raw/`` (ou no diretório
indicado pela variável de ambiente ``DSP_RAW_DIR``). A leitura do .xlsx é lenta (~minutos), então a
primeira execução grava uma cópia em Parquet no mesmo diretório e as seguintes reutilizam essa cópia.
"""
from __future__ import annotations

import os
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
USER_AGENT = "data-science-portfolio (educational; github.com/larissamagistrali)"
PROJECT_DIR = Path(__file__).resolve().parents[2]

# Códigos de estoque que não são produtos (frete, taxas, ajustes manuais, amostras...).
NON_PRODUCT_CODES = {"POST", "D", "M", "DOT", "CRUK", "BANK CHARGES", "ADJUST", "ADJUST2", "S", "B", "AMAZONFEE", "TEST001", "TEST002", "PADS"}


def raw_dir() -> Path:
    base = Path(os.environ.get("DSP_RAW_DIR", PROJECT_DIR / "data" / "raw"))
    return base / "online-retail-ii"


def download(dest: Path | None = None) -> Path:
    """Baixa o zip para ``<raw>/online_retail_ii.zip`` (via .part) e devolve o caminho. Reutiliza se já existir."""
    dest = dest or raw_dir()
    dest.mkdir(parents=True, exist_ok=True)
    final = dest / "online_retail_ii.zip"
    if final.exists():
        return final
    part = final.with_suffix(".zip.part")
    req = urllib.request.Request(URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=600) as resp, open(part, "wb") as fh:  # noqa: S310 (URL fixa, https)
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    part.rename(final)
    return final


def _xlsx_path(dest: Path) -> Path:
    extracted = dest / "extracted"
    xlsx = next(extracted.glob("*.xlsx"), None) if extracted.exists() else None
    if xlsx is None:
        zip_path = download(dest)
        extracted.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():  # só extrai .xlsx com nome simples (evita path traversal)
                if member.endswith(".xlsx") and "/" not in member and ".." not in member:
                    zf.extract(member, extracted)
        xlsx = next(extracted.glob("*.xlsx"))
    return xlsx


def load_raw(dest: Path | None = None) -> pd.DataFrame:
    """Lê as duas planilhas (2009-2010 e 2010-2011) e concatena; usa cópia em Parquet se existir."""
    dest = dest or raw_dir()
    cache = dest / "online_retail_ii.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    sheets = pd.read_excel(_xlsx_path(dest), sheet_name=None, engine="openpyxl", dtype={"Invoice": str, "StockCode": str})
    df = pd.concat(sheets.values(), ignore_index=True)
    df = df.rename(columns={"Customer ID": "CustomerID"})
    for col in ("Invoice", "StockCode"):
        df[col] = df[col].astype(str)
    df["Description"] = df["Description"].astype("string")  # há descrições numéricas/mistas na planilha
    df["CustomerID"] = df["CustomerID"].astype("Float64")
    df.to_parquet(cache, index=False)
    return df


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Aplica as regras de limpeza e devolve (vendas válidas, cancelamentos, contagem de descartes por motivo).

    Regras (na ordem): sem CustomerID; cancelamentos (fatura começa com "C"); quantidade <= 0;
    preço <= 0; códigos que não são produtos (frete, taxas, ajustes). Os cancelamentos são
    devolvidos à parte para a taxa de devolução por cliente (ver ``rfm.build_rfm``).
    """
    stats: dict[str, int] = {"linhas_brutas": len(raw)}
    df = raw.copy()
    df["is_cancel"] = df["Invoice"].str.upper().str.startswith("C")
    sem_cliente = df["CustomerID"].isna()
    stats["sem_customer_id"] = int(sem_cliente.sum())
    df = df[~sem_cliente]
    stats["cancelamentos"] = int(df["is_cancel"].sum())
    cancel = df[df["is_cancel"]]
    df = df[~df["is_cancel"]]
    invalido = (df["Quantity"] <= 0) | (df["Price"] <= 0)
    stats["quantidade_ou_preco_invalido"] = int(invalido.sum())
    df = df[~invalido]
    nao_produto = df["StockCode"].str.upper().isin(NON_PRODUCT_CODES)
    stats["codigos_nao_produto"] = int(nao_produto.sum())
    df = df[~nao_produto].copy()
    df["revenue"] = df["Quantity"] * df["Price"]
    stats["linhas_validas"] = len(df)
    stats["clientes"] = int(df["CustomerID"].nunique())
    return df, cancel.copy(), stats
