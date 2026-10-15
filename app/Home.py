"""Página inicial da vitrine de Ciência de Dados."""
import _bootstrap  # noqa: F401

import streamlit as st

from shared import ui

ui.setup_page("Portfólio · Ciência de Dados", "🏠")
st.title("Portfólio de Ciência de Dados")
st.markdown(
    "**Cinco projetos com problema, dados, método e resultado explícitos.** Cada página tem as abas "
    "Problema · Dados · Abordagem · Resultado · Demo interativa. Os números vêm de execuções reais do pipeline "
    "(arquivos `metrics.json` versionados), não de texto digitado."
)
st.info(
    "**Como o repo está organizado:** o pipeline de cada projeto roda localmente (`make data`, `make train`) e grava "
    "amostras e métricas pequenas em `projects/*/data/`. O app só lê esses artefatos; ele não treina nem baixa "
    "bases grandes no servidor."
)
st.caption("Dados: quatro projetos usam dados reais de fonte pública; um usa dados sintéticos com o schema do dado original, rotulado como tal na página.")

ui.project_grid([
    {"icon": "📉", "title": "Churn", "summary": "Quem vai cancelar? Probabilidade calibrada, explicação exata por variável e limiar por custo de retenção.",
     "stack": "scikit-learn · regressão logística · HistGradientBoosting", "page": "pages/1_Churn.py",
     "data_kind": "real", "dataset": "IBM Telco Customer Churn"},
    {"icon": "🧩", "title": "Segmentação de clientes", "summary": "RFM e KMeans com k escolhido por silhouette; segmentos nomeados por regras sobre os centróides.",
     "stack": "pandas · KMeans · RFM", "page": "pages/2_Segmentacao.py",
     "data_kind": "real", "dataset": "Online Retail II (UCI)"},
    {"icon": "🧪", "title": "Testes A/B", "summary": "Retenção no Cookie Cats com IC, p-valor, P(B>A), perda esperada, poder, múltiplas métricas e peeking.",
     "stack": "scipy · statsmodels · Beta-Binomial", "page": "pages/3_AB_Testing.py",
     "data_kind": "real", "dataset": "Cookie Cats"},
    {"icon": "📦", "title": "Previsão de demanda", "summary": "Seasonal-naive, Holt-Winters e gradient boosting com backtest de origem rolante e intervalos de 90% com cobertura medida.",
     "stack": "statsmodels · HistGradientBoosting · backtest", "page": "pages/4_Forecasting.py",
     "data_kind": "synthetic", "dataset": "gerador com schema Rossmann"},
    {"icon": "🇧🇷", "title": "Indicadores do Brasil", "summary": "Selic, IPCA, câmbio e desemprego (BCB) e ranking por UF (IBGE), com correlação defasada.",
     "stack": "APIs SGS e SIDRA · pandas · plotly", "page": "pages/5_EDA_Brasil.py",
     "data_kind": "real", "dataset": "BCB SGS · IBGE SIDRA"},
], columns=2)
