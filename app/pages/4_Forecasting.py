"""Previsão de demanda diária por loja: seasonal-naive, Holt-Winters e GBM, com intervalos de 90%."""
import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from shared import data as D
from shared import theme
from shared import ui

ui.setup_page("Previsão de demanda", "📦")
ui.header(
    "Previsão de demanda",
    "Quanto cada loja vai vender nos próximos dias, e quão confiável é essa previsão?",
    tags=["statsmodels (Holt-Winters)", "HistGradientBoosting", "backtest rolante", "intervalos empíricos"],
    code_path="projects/demand-forecasting",
)
problema, dados, abordagem, resultado, demo = ui.tabs()

PROJ = "demand-forecasting"
D.use_project(PROJ)
metrics = D.read_json(str(D.data_path(PROJ, "metrics.json")))
bt = D.read_parquet(str(D.data_path(PROJ, "backtest.parquet")))
bt["date"] = pd.to_datetime(bt["date"])
bt["origin"] = pd.to_datetime(bt["origin"])
NOMES = {"seasonal_naive": "Seasonal-naive (baseline)", "holt_winters": "Holt-Winters (ETS)", "gbm": "Gradient boosting"}
CORES = theme.color_map(list(NOMES))

with problema:
    st.markdown(
        """
Previsão de demanda orienta compra, estoque e equipe. Erro para cima gera perda por excesso;
erro para baixo gera venda perdida. Por isso o número sozinho não basta: o que importa é
**previsão + intervalo + quanto dá para confiar nele**.

Comparo três modelos com **backtest de origem rolante**: em cada origem, o modelo só vê o passado
e prevê os 28 dias seguintes. Assim a avaliação reflete o uso real, sem vazamento.
"""
    )

with dados:
    ui.data_notice(
        "synthetic",
        "Gerador próprio com o schema do Rossmann Store Sales (Kaggle, não baixado aqui)",
        f"{metrics['lojas']} lojas, 2013-2015, diário. Sazonalidade semanal e anual, promoções, feriados, domingos fechados e ruído. "
        "Os resultados ilustram o pipeline; não representam desempenho em varejo real.",
    )
    st.markdown(
        """
O Rossmann exige credencial do Kaggle, que não tenho aqui. O gerador (`make_sample.py`, seed 42) produz
vendas com a mesma estrutura de colunas, mas os **padrões são os que eu inventei**, então o ranking
dos modelos aqui não deve ser transferido para o dado real sem reavaliação.
Para rodar com o real: `kaggle competitions download -c rossmann-store-sales` (caminho não executado aqui).
"""
    )

with abordagem:
    st.markdown(
        f"""
- **Seasonal-naive:** repete o mesmo dia da semana da última semana observada.
- **Holt-Winters:** suavização exponencial aditiva com tendência amortecida e sazonalidade semanal (m = 7), por loja.
- **Gradient boosting:** um modelo global para todas as lojas, com calendário, plano de promoções e feriados (conhecidos antecipadamente), características da loja e **lags de 28 e 364 dias** (ambos anteriores à origem para qualquer horizonte até 28). Um teste automatizado confirma que alterar vendas futuras não muda a previsão.
- **Métricas:** WAPE (erro absoluto total ÷ venda total), MAPE (dias com venda > 0) e MASE (erro ÷ erro do naive sazonal no histórico).
- **Intervalo de 90%:** quantis 5% e 95% do erro relativo, calculados **apenas nas origens anteriores**. Cobertura medida na origem seguinte.
- **Origens:** {', '.join(metrics['origens'])}. Avaliação só em dias abertos.
"""
    )

with resultado:
    rows = []
    for m in NOMES:
        r = metrics["metricas"][m]
        rows.append({"modelo": NOMES[m], "WAPE": r["wape"], "MAPE": r["mape"], "MASE": r["mase"],
                     "cobertura do IC 90% (média)": r["cobertura_90_media"]})
    ui.metric_row([
        ("WAPE · gradient boosting", f"{metrics['metricas']['gbm']['wape']:.1%}", "Menor é melhor"),
        ("WAPE · Holt-Winters", f"{metrics['metricas']['holt_winters']['wape']:.1%}"),
        ("WAPE · seasonal-naive", f"{metrics['metricas']['seasonal_naive']['wape']:.1%}"),
    ])
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    st.caption("Cobertura nominal: 90%. Medida nas origens seguintes à calibração. Os valores abaixo de 90% indicam intervalos subestimados nessas origens.")
    fig = go.Figure()
    for m in NOMES:
        per = metrics["metricas"][m]["wape_por_origem"]
        fig.add_trace(go.Scatter(x=list(per), y=list(per.values()), mode="lines+markers", name=NOMES[m], line=dict(color=CORES[m])))
    fig.update_layout(title="WAPE por origem de previsão", xaxis_title="Origem", yaxis_title="WAPE", yaxis_tickformat=".0%")
    ui.plot(fig, key="wape_origem")

with demo:
    c1, c2, c3 = st.columns(3)
    stores = sorted(bt["store"].unique().tolist())
    store = c1.selectbox("Loja", stores, index=0)
    origins = sorted(bt["origin"].dt.date.unique().tolist())
    origin = pd.Timestamp(c2.selectbox("Data de origem", origins, index=len(origins) - 1, format_func=str))
    horizon = c3.slider("Horizonte (dias)", 7, 28, 14)
    chosen = st.multiselect("Modelos", list(NOMES), default=["gbm", "seasonal_naive"], format_func=NOMES.get)
    if not chosen:
        st.info("Escolha ao menos um modelo.")
    else:
        sub = bt[(bt["store"] == store) & (bt["origin"] == origin) & (bt["h"] <= horizon)]
        hist = D.read_parquet(str(D.data_path(PROJ, "sales_sample.parquet")))
        hist = hist[(hist["Store"] == store) & (hist["Date"] <= origin) & (hist["Date"] > origin - pd.Timedelta(days=90))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Sales"], mode="lines", name="Histórico (90 dias)", line=dict(color=theme.INK_MUTED, width=1.5)))
        real = sub[sub["model"] == chosen[0]]
        fig.add_trace(go.Scatter(x=real["date"], y=real["y"], mode="lines+markers", name="Real (após origem)", line=dict(color=theme.INK, width=2)))
        for m in chosen:
            g = sub[sub["model"] == m].dropna(subset=["lo"])
            if len(g):
                fig.add_trace(go.Scatter(x=pd.concat([g["date"], g["date"][::-1]]), y=pd.concat([g["hi"], g["lo"][::-1]]),
                                         fill="toself", fillcolor=CORES[m], opacity=0.12, line=dict(width=0),
                                         name=f"IC 90% · {NOMES[m]}", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=sub[sub["model"] == m]["date"], y=sub[sub["model"] == m]["yhat"], mode="lines",
                                     name=NOMES[m], line=dict(color=CORES[m], width=2)))
        fig.add_vline(x=origin, line=dict(dash="dot", color=theme.BASELINE))
        fig.update_layout(title=f"Loja {store} · origem {origin.date()} · horizonte {horizon} dias", xaxis_title="Data", yaxis_title="Vendas")
        ui.plot(fig, key="forecast")
        st.caption("Linha vertical = origem (o modelo só viu o histórico até aqui). Dias fechados (domingos e feriados) são previstos como zero.")
        rows = []
        for m in chosen:
            g = sub[(sub["model"] == m) & (sub["open"] == 1)]
            cov = float(((g["y"] >= g["lo"]) & (g["y"] <= g["hi"])).mean()) if g["lo"].notna().any() else np.nan
            rows.append({"modelo": NOMES[m], "WAPE (trecho)": float(np.abs(g["y"] - g["yhat"]).sum() / max(g["y"].sum(), 1)),
                         "cobertura do IC no trecho": cov})
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
        if origin == origins[0]:
            st.info("A primeira origem não tem calibração prévia, então o intervalo não é calculado nela.")
