"""Cookie Cats: download do espelho público (GitHub) e gerador sintético para testes/uso offline."""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# Espelho público sem login do CSV original do projeto "Mobile Games A/B Testing with Cookie Cats" (DataCamp).
URL = "https://raw.githubusercontent.com/robguilarr/ab_testing_cookie_cats/main/datasets/cookie_cats.csv"
USER_AGENT = "data-science-portfolio (educational; github.com/larissamagistrali)"
PROJECT_DIR = Path(__file__).resolve().parents[2]
COLUMNS = ["userid", "version", "sum_gamerounds", "retention_1", "retention_7"]


def raw_dir() -> Path:
    return Path(os.environ.get("DSP_RAW_DIR", PROJECT_DIR / "data" / "raw")) / "cookie-cats"


def download(dest: Path | None = None) -> Path:
    """Baixa o CSV (via .part); reutiliza se já existir."""
    dest = dest or raw_dir()
    dest.mkdir(parents=True, exist_ok=True)
    final = dest / "cookie_cats.csv"
    if final.exists():
        return final
    part = final.with_suffix(".csv.part")
    req = urllib.request.Request(URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (URL fixa, https)
        part.write_bytes(resp.read())
    part.rename(final)
    return final


def load_real() -> pd.DataFrame:
    return validate(pd.read_csv(download()))


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Confere o schema e tipos; levanta ValueError se algo estiver fora do esperado."""
    if list(df.columns) != COLUMNS:
        raise ValueError(f"colunas inesperadas: {list(df.columns)}")
    if set(df["version"].unique()) != {"gate_30", "gate_40"}:
        raise ValueError("version deve ter exatamente gate_30 e gate_40")
    if df["userid"].duplicated().any():
        raise ValueError("userid duplicado")
    out = df.copy()
    out["userid"] = out["userid"].astype("int64")
    out["sum_gamerounds"] = out["sum_gamerounds"].astype("int32")
    for c in ("retention_1", "retention_7"):
        out[c] = out[c].astype(bool)
    return out


def synthetic(n: int = 20000, seed: int = 42, p1: tuple[float, float] = (0.45, 0.45), p7: tuple[float, float] = (0.19, 0.18)) -> pd.DataFrame:
    """Dado SINTÉTICO com o schema do Cookie Cats (para testes e uso offline; não representa o jogo real).

    p1/p7 = (taxa do gate_30, taxa do gate_40) de retenção em 1 e 7 dias; rodadas ~ log-normal.
    """
    rng = np.random.default_rng(seed)
    version = rng.choice(["gate_30", "gate_40"], n)
    idx = (version == "gate_40").astype(int)
    r1 = rng.random(n) < np.array(p1)[idx]
    r7 = r1 & (rng.random(n) < np.array(p7)[idx] / np.array(p1)[idx])  # retenção em 7d implica ter voltado no 1º dia
    rounds = np.round(rng.lognormal(3.0, 1.3, n)).astype(int)
    return pd.DataFrame({"userid": np.arange(1, n + 1), "version": version, "sum_gamerounds": rounds, "retention_1": r1, "retention_7": r7})
