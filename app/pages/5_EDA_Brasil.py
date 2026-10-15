"""EDA de indicadores brasileiros: séries do BCB (SGS) e tabelas do IBGE (SIDRA), dados reais."""
import _bootstrap  # noqa: F401

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from shared import data as D
from shared import theme
from shared import ui

ui.setup_page("Indicadores do Brasil", "🇧🇷")
ui.header(
    "Indicadores econômicos do Brasil",
    "Como juros, inflação, câmbio e desemprego se movem, e como os estados se comparam?",
    tags=["API SGS (BCB)", "API SIDRA (IBGE)", "pandas", "correlação defasada"],
    code_path="projects/eda-dados-br",
)
problema, dados, abordagem, resultado, demo = ui.tabs()

PROJ = "eda-dados-br"
D.use_project(PROJ)
from eda_dados_br import analysis as A  # noqa: E402

meta = D.read_json(str(D.data_path(PROJ, "meta.json")))
sgs = D.read_parquet(str(D.data_path(PROJ, "bcb_sgs.parquet")))
sgs["data"] = pd.to_datetime(sgs["data"])
ipca_idx = D.read_parquet(str(D.data_path(PROJ, "sidra_ipca_indice.parquet")))
pop = D.read_parquet(str(D.data_path(PROJ, "sidra_populacao_uf.parquet")))
desoc_uf = D.read_parquet(str(D.data_path(PROJ, "sidra_desocupacao_uf_trimestre.parquet")))
SERIES = sorted(sgs["serie"].unique().tolist())
UNIDADE = sgs.drop_duplicates("serie").set_index("serie")["unidade"].to_dict()
COLOR = theme.color_map(SERIES)

with problema:
    st.markdown(
        """
Juros, inflação, câmbio e desemprego andam juntos, mas com defasagens: o Banco Central mexe na
Selic para conter a inflação, e o efeito aparece meses depois. Esta página explora essas séries
e mostra como elas se relacionam. Também compara estados com dados regionais do IBGE.
"""
    )

with dados:
    ui.data_notice(
        "real",
        "Banco Central do Brasil (SGS) e IBGE (SIDRA), dados abertos sem autenticação",
        f"Coleta em {meta['coletado_em']} com User-Agent descritivo, cache e retry. Snapshot versionado em parquet.",
    )
    st.dataframe(pd.DataFrame(meta["series_sgs"]), hide_index=True, width="stretch")
    st.markdown(
        """
**Séries SIDRA:** IPCA índice geral (tabela 1737, variável 2266, base dez/1993 = 100), população residente
por UF (tabela 6579, variável 9324) e taxa de desocupação por UF e trimestre (tabela 4099).

**Confirmar:** o código SGS 24369 foi rotulado aqui como "desocupação (PNAD Contínua)" pela convenção do BCB;
o rótulo deve ser conferido no catálogo oficial do SGS antes de uso formal.

**Base dos Dados (extensão):** o acesso às mesmas tabelas via BigQuery exige projeto no Google Cloud. Não foi usado aqui.
"""
    )

with abordagem:
    st.markdown(
        """
- **Séries temporais** em frequências diferentes (Selic e câmbio diários; IPCA e desocupação mensais/trimestrais): agrego para o mês (último valor) para comparar.
- **Índice base 100** para comparar níveis com unidades diferentes em **um único eixo**. Variação mensal (IPCA) vira antes
  índice acumulado, Π(1 + r/100); taxas em nível (Selic, desocupação) ficam fora, porque base 100 de uma taxa mede a
  variação da própria taxa e não se compara a um índice de preços.
- **Variação em 12 meses** do IPCA índice: `IPCA(t) / IPCA(t−12) − 1`.
- **Correlação defasada:** diferencio as séries (mudança mês a mês) antes de correlacionar. Sem isso, duas séries com tendência parecem correlacionadas por acaso. Positivo em k: y responde k meses depois de x.
- **Estados:** ranking horizontal, sem mapa (evita geojson pesado).
"""
    )

with resultado:
    selic = sgs[sgs["serie"] == "Selic meta"].set_index("data")["valor"]
    dolar = sgs[sgs["serie"] == "Dólar comercial (venda)"].set_index("data")["valor"]
    ipca_mensal_idx = A.to_monthly(ipca_idx[["data", "valor"]])
    ipca12 = A.yoy_change(ipca_mensal_idx).dropna()
    ui.metric_row([
        ("Selic meta (último)", f"{selic.iloc[-1]:.2f}% a.a.", f"em {selic.index[-1].date()}"),
        ("IPCA em 12 meses (último)", f"{ipca12.iloc[-1]:.2f}%", f"referente a {ipca12.index[-1].strftime('%m/%Y')} · calculado do índice SIDRA"),
        ("Dólar venda (último)", f"R$ {dolar.iloc[-1]:.2f}", f"em {dolar.index[-1].date()}"),
    ])
    st.caption("Os valores acima vêm do snapshot coletado; a página mostra o último dado disponível no momento da coleta.")
    fig = px.line(sgs[sgs["serie"] == "Selic meta"], x="data", y="valor", color="serie", color_discrete_map=COLOR, title="Selic meta (% a.a.)")
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="% a.a.")
    ui.plot(fig, key="selic")

with demo:
    st.subheader("Explorador de séries")
    escolhidas = st.multiselect("Indicadores", SERIES, default=["Dólar comercial (venda)", "IPCA (variação mensal)"])
    ini, fim = st.slider("Período", 2010, 2026, (2019, 2026))
    if not escolhidas:
        st.info("Escolha ao menos um indicador.")
    else:
        sub = sgs[sgs["serie"].isin(escolhidas) & (sgs["data"].dt.year >= ini) & (sgs["data"].dt.year <= fim)]
        fig = px.line(sub, x="data", y="valor", color="serie", facet_col="serie", facet_col_wrap=1, facet_row_spacing=0.12,
                      color_discrete_map=COLOR, category_orders={"serie": [s for s in SERIES if s in escolhidas]},
                      height=max(280, 200 * len(escolhidas)),
                      title="Valores originais (um painel por série, cada um com o próprio eixo)")
        fig.update_yaxes(matches=None, title_text="", showticklabels=True)
        fig.update_xaxes(title_text="")
        fig.for_each_annotation(lambda a: a.update(text=f"{a.text.split('=', 1)[-1]} ({UNIDADE[a.text.split('=', 1)[-1]]})"))
        fig.update_layout(showlegend=False)
        ui.plot(fig, key="explor")
        base, fora, cores = {}, [], {}
        for nome in escolhidas:
            mensal = A.to_monthly(sgs[sgs["serie"] == nome][["data", "valor"]], how="last")
            mensal = mensal[(mensal.index.year >= ini) & (mensal.index.year <= fim)]
            nivel = A.comparable_level(mensal, UNIDADE[nome])
            if nivel is None:
                fora.append(nome)
            elif len(nivel) > 1:
                rotulo = f"{nome.split(' (')[0]} acumulado" if UNIDADE[nome] == "% a.m." else nome
                base[rotulo] = A.index_base100(nivel)
                cores[rotulo] = COLOR[nome]
        if base:
            comp = pd.concat(base, names=["serie", "data"]).reset_index(name="indice")
            fig = px.line(comp, x="data", y="indice", color="serie", color_discrete_map=cores,
                          title="Comparação em base 100 (primeiro mês do período = 100)")
            fig.add_hline(y=100, line=dict(dash="dot", color=theme.BASELINE))
            fig.update_layout(xaxis_title=None, yaxis_title="índice")
            ui.plot(fig, key="base100")
            st.caption("Variação mensal entra como índice acumulado, Π(1 + r/100); níveis (dólar) entram direto.")
        if fora:
            st.caption(f"Fora da base 100: {', '.join(fora)}. Base 100 de uma taxa mede a variação da própria taxa e não se "
                       "compara a um índice de preços; veja essas séries nos painéis acima.")
        if not base:
            st.info("Para a comparação em base 100, inclua uma série de nível (dólar) ou de variação mensal (IPCA).")
        st.markdown("**Correlação defasada entre dois indicadores (mudanças mensais)**")
        c1, c2 = st.columns(2)
        a_nome = c1.selectbox("Indicador X", SERIES, index=0)
        b_nome = c2.selectbox("Indicador Y", SERIES, index=1)
        xa = A.to_monthly(sgs[sgs["serie"] == a_nome][["data", "valor"]])
        yb = A.to_monthly(sgs[sgs["serie"] == b_nome][["data", "valor"]])
        tab = A.lagged_correlation(xa, yb, max_lag=12)
        fig = go.Figure(go.Bar(x=tab["lag_meses"], y=tab["correlacao"], marker_color=theme.CATEGORICAL[0]))
        fig.update_layout(title=f"Correlação entre Δ{a_nome} (t) e Δ{b_nome} (t+k)", xaxis_title="Defasagem k (meses; positivo = Y responde depois de X)",
                          yaxis_title="Correlação", yaxis_range=[-1, 1])
        ui.plot(fig, key="lag")
        best = tab.loc[tab["correlacao"].abs().idxmax()]
        st.caption(f"Maior correlação absoluta em k = {int(best['lag_meses'])} meses: r = {best['correlacao']:.2f} (n = {int(best['n'])}). "
                   "Correlação não é causalidade; a janela curta e a autocorrelação limitam a interpretação.")

        st.subheader("Ranking por UF (IBGE)")
        tab_uf = st.radio("Indicador", ["Taxa de desocupação (trimestre)", "População residente"], horizontal=True)
        if tab_uf.startswith("Taxa"):
            opts = desoc_uf.sort_values("data")["periodo"].unique().tolist()
            per = st.selectbox("Trimestre", opts[::-1], index=0)
            rk = A.ranking_uf(desoc_uf, per)
            unidade = "%"
        else:
            opts = pop.sort_values("data")["periodo"].unique().tolist()
            per = st.selectbox("Ano", opts[::-1], index=0)
            rk = A.ranking_uf(pop, per)
            unidade = "habitantes"
        if rk.empty:
            st.info("Sem dados para o período escolhido.")
        else:
            fig = go.Figure(go.Bar(x=rk["valor"], y=rk["territorio"], orientation="h", marker_color=theme.CATEGORICAL[0],
                                   texttemplate="%{x:,.1f}" if unidade == "%" else "%{x:,.0f}", textposition="outside", cliponaxis=False))
            fig.update_layout(title=f"{tab_uf} · {per} ({unidade})", height=max(400, 22 * len(rk)), xaxis_title=unidade, yaxis_title="",
                              xaxis_range=[0, rk["valor"].max() * 1.15])
            ui.plot(fig, key="uf")
