"""Coleta das APIs públicas do BCB (SGS) e do IBGE (SIDRA) com cache, retry e User-Agent descritivo.

Não usa a Base dos Dados (exige projeto no Google BigQuery); ver README, seção "Extensões".
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

USER_AGENT = "data-science-portfolio/0.1 (educacional; github.com/larissamagistrali)"
PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_DIR / "data" / "raw"
SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json&dataInicial={start}&dataFinal={end}"
SIDRA_URL = "https://apisidra.ibge.gov.br/values/t/{table}/{territory}/{classes}/v/{var}/p/{period}"


@dataclass(frozen=True)
class SgsSerie:
    code: int
    nome: str
    unidade: str
    fonte_nome: str = "Banco Central do Brasil (SGS)"


# Códigos conferidos na API em 2026-10: Selic meta, IPCA mensal, dólar venda, taxa de desocupação (PNAD).
SGS_SERIES = [
    SgsSerie(432, "Selic meta", "% a.a."),
    SgsSerie(433, "IPCA (variação mensal)", "% a.m."),
    SgsSerie(1, "Dólar comercial (venda)", "R$/US$"),
    SgsSerie(24369, "Taxa de desocupação (PNAD Contínua)", "%"),
]


def _get_json(url: str, retries: int = 3, pause: float = 2.0):
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (APIs oficiais, https)
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"falha ao coletar {url}: {last}")


def parse_sgs(payload: list[dict]) -> pd.DataFrame:
    """Converte a resposta do SGS (data dd/mm/aaaa, valor string) em DataFrame (data, valor)."""
    df = pd.DataFrame(payload)
    if df.empty:
        return pd.DataFrame(columns=["data", "valor"])
    out = pd.DataFrame({"data": pd.to_datetime(df["data"], format="%d/%m/%Y"), "valor": pd.to_numeric(df["valor"], errors="coerce")})
    return out.dropna().sort_values("data").reset_index(drop=True)


def fetch_sgs(serie: SgsSerie, end: date | None = None, window_years: int = 5) -> pd.DataFrame:
    """Coleta a série em janelas de `window_years` anos (a API recusa séries diárias longas: HTTP 406)."""
    end = end or date.today()
    start = date(2010, 1, 1)
    parts = []
    while start <= end:
        stop = min(date(start.year + window_years, 1, 1), end)
        url = SGS_URL.format(code=serie.code, start=start.strftime("%d/%m/%Y"), end=stop.strftime("%d/%m/%Y"))
        parts.append(parse_sgs(_get_json(url)))
        start = date(start.year + window_years, 1, 1)
        time.sleep(0.5)  # gentileza com a API
    return pd.concat(parts, ignore_index=True).drop_duplicates("data").sort_values("data").reset_index(drop=True)


def parse_sidra(payload: list[dict]) -> pd.DataFrame:
    """Converte a resposta do SIDRA (primeira linha = cabeçalho) em DataFrame longo.

    Colunas: territorio (nome), codigo_territorio, periodo (texto do IBGE), valor (float; '...' vira NaN).
    """
    if not payload or len(payload) < 2:
        return pd.DataFrame(columns=["territorio", "codigo_territorio", "periodo", "valor"])
    rows = payload[1:]
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "territorio": df["D1N"],
        "codigo_territorio": df["D1C"].astype(int),
        "periodo": df["D3N"],
        "valor": pd.to_numeric(df["V"], errors="coerce"),
    })
    return out.dropna(subset=["valor"]).reset_index(drop=True)


def fetch_sidra(table: int, territory: str, classes: str, var: int, period: str = "all") -> pd.DataFrame:
    url = SIDRA_URL.format(table=table, territory=territory, classes=classes, var=var, period=period)
    return parse_sidra(_get_json(url))


def cached(path: Path, fetch, refresh: bool = False) -> pd.DataFrame:
    """Cache em data/raw/ (gitignored): reutiliza o arquivo se existir e refresh=False."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = fetch()
    df.to_parquet(path, index=False)
    return df
