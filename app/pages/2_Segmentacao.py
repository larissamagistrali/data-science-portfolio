"""Segmentação de clientes: RFM + KMeans sobre o Online Retail II (dados reais, UCI)."""
import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans

from shared import data as D
from shared import theme
from shared import ui

ui.setup_page("Segmentação de clientes", "🧩")
ui.header(
    "Segmentação de clientes (RFM + KMeans)",
    "Quem são os clientes que mais geram receita, quem está sumindo e o que fazer com cada grupo?",
    tags=["RFM", "KMeans", "pandas", "silhouette"],
    code_path="projects/customer-segmentation",
)
problema, dados, abordagem, resultado, demo = ui.tabs()

PROJ = "customer-segmentation"
D.use_project(PROJ)
from customer_segmentation import cluster as C  # noqa: E402

metrics = D.read_json(str(D.data_path(PROJ, "metrics.json")))
rfm = D.read_parquet(str(D.data_path(PROJ, "rfm.parquet")))

ACOES = {
    "Campeões": "Programa de fidelidade e acesso antecipado a lançamentos; pedir indicação. Não desconte: esses clientes já compram.",
    "Leais": "Cross-sell com base no histórico de compras; comunicação recorrente de novidades.",
    "Em risco": "Campanha de reativação com oferta personalizada e contato humano para o maior valor.",
    "Recentes": "Onboarding e incentivo a uma segunda compra no prazo de 30 dias.",
    "Intermediários": "Campanhas de frequência: lembrete de reposição e oferta de pacote.",
    "Hibernando": "Reativação de baixo custo (e-mail/push). Descarte quem não responder após 2 tentativas.",
    "Inativos": "Campanha de win-back de baixo custo; excluir da base de mídia paga após 12 meses sem resposta.",
    "Atenção": "Pesquisa de satisfação e checagem de problemas de entrega ou cancelamento.",
}


def acao_para(segmento: str) -> str:
    for chave, texto in ACOES.items():
        if segmento.startswith(chave):
            return texto
    return "Sem ação pré-definida."


@st.cache_data(show_spinner="Agrupando clientes...")
def run_kmeans(k: int) -> tuple[np.ndarray, dict[int, str], pd.DataFrame]:
    Z, _ = C.transform(rfm)
    km = KMeans(n_clusters=k, n_init=10, random_state=C.SEED).fit(Z)
    cents = pd.DataFrame(km.cluster_centers_, columns=C.RFM_COLS)
    names = C.name_segments(cents)
    return km.labels_, names, C.profile(rfm, km.labels_, names)


with problema:
    st.markdown(
        """
Nem todo cliente vale o mesmo esforço. Um cliente que comprou ontem, com frequência alta e
ticket alto, pede tratamento diferente de quem não compra há um ano. **Segmentar** é separar
a base em grupos com comportamento parecido para orientar ação (e não só descrever).

A abordagem clássica é o **RFM**: **R**ecência (há quantos dias comprou), **F**requência
(em quantas compras) e **M**onetário (quanto gastou). Aqui agrupo com KMeans e comparo com
a segmentação tradicional por quintis.
"""
    )

with dados:
    ui.data_notice(
        "real",
        "Online Retail II (UCI Machine Learning Repository, CC BY 4.0)",
        "Varejo online britânico, 2009-2011. Substitui o Olist (Kaggle, com credencial) como base de exemplo.",
    )
    s = metrics["limpeza"]
    st.markdown(
        f"""
**Limpeza** (de {s['linhas_brutas']:,} linhas brutas):
- {s['sem_customer_id']:,} sem `CustomerID` (vendas de balcão/não identificadas) → descartadas;
- {s['cancelamentos']:,} faturas de cancelamento (prefixo `C`) → usadas só para a taxa de devolução;
- {s['quantidade_ou_preco_invalido']:,} com quantidade ou preço ≤ 0 → descartadas;
- {s['codigos_nao_produto']:,} com códigos de frete, taxas ou ajustes → descartadas.

Resultado: {s['linhas_validas']:,} linhas válidas de {s['clientes']:,} clientes. Período: {metrics['periodo']['inicio']} a {metrics['periodo']['fim']}.
**Versionado:** só o agregado RFM por cliente (`rfm.parquet`), com ID pseudonimizado (hash SHA-256 com sal), sem transações.
"""
    )

with abordagem:
    st.markdown(
        """
1. **RFM por cliente**; monetário = Σ quantidade × preço das compras válidas (libras).
2. **Transformação:** `log(1 + x)` nas três métricas (distribuições muito assimétricas), depois padronização.
3. **Escolha de k** entre 2 e 10 por silhouette e cotovelo. Para uso de negócio restringi a 3 a 8 segmentos.
4. **Nomes por regras sobre os centróides**, não inventados: cada grupo recebe o nome a partir de z-scores de R, F e M.
5. **Comparação:** segmentação por quintis (escores de 1 a 5 em R, F e M, com regras simples) e concordância entre os dois (ARI).
"""
    )

with resultado:
    ui.metric_row([
        ("Clientes segmentados", f"{metrics['clientes_rfm']:,}"),
        ("k escolhido", str(metrics["k_escolhido"]), metrics["regra_k"]),
        ("Silhouette · KMeans", f"{metrics['comparacao']['silhouette_kmeans']:.2f}", "Quanto maior, mais separados os grupos"),
        ("Concordância com quintis (ARI)", f"{metrics['comparacao']['ari_kmeans_vs_quintis']:.2f}", "0 = aleatório, 1 = idêntico"),
    ])
    sel = pd.DataFrame(metrics["selecao_k"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sel["k"], y=sel["inertia"], mode="lines+markers", name="Inércia (cotovelo)"))
    fig.update_layout(title="Inércia por k (cotovelo)", xaxis_title="k", yaxis_title="Inércia")
    ui.plot(fig, key="elbow")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sel["k"], y=sel["silhouette"], mode="lines+markers", name="Silhouette"))
    fig.update_layout(title="Silhouette por k (amostra de 5.000)", xaxis_title="k", yaxis_title="Silhouette")
    ui.plot(fig, key="sil")
    prof = pd.DataFrame(metrics["perfil_clusters"])
    st.markdown("**Perfil dos segmentos (k escolhido)**")
    st.dataframe(prof[["segmento", "clientes", "pct_clientes", "recencia_mediana_dias", "frequencia_mediana", "monetario_mediano", "pct_receita"]].round(3), hide_index=True)
    st.markdown("**Segmentação por quintis RFM (comparação)**")
    st.dataframe(pd.DataFrame(metrics["quintis_perfil"]).round(2), hide_index=True)

with demo:
    k = st.slider("Número de segmentos (k)", 3, 8, int(metrics["k_escolhido"]))
    labels, names, prof_k = run_kmeans(k)
    dfp = rfm.assign(segmento=pd.Series(labels).map(names).to_numpy())
    eixos = {"Recência (dias)": "recency_days", "Frequência (compras)": "frequency", "Monetário (£)": "monetary", "Ticket médio (£)": "avg_ticket"}
    c1, c2 = st.columns(2)
    x_lbl = c1.selectbox("Eixo X", list(eixos), index=1)
    y_lbl = c2.selectbox("Eixo Y", list(eixos), index=2)
    ordem = sorted(names.values())
    cores = theme.color_map(ordem)
    sample = dfp.sample(min(len(dfp), 3000), random_state=0)
    fig = px.scatter(sample, x=eixos[x_lbl], y=eixos[y_lbl], color="segmento", log_x=True, log_y=True,
                     color_discrete_map=cores, category_orders={"segmento": ordem}, hover_data=["customer"], opacity=0.7,
                     labels={eixos[x_lbl]: x_lbl, eixos[y_lbl]: y_lbl})
    fig.update_xaxes(dtick=1)  # escala log: só as décadas (1, 10, 100...), sem as marcas 2 e 5
    fig.update_yaxes(dtick=1)
    fig.update_layout(title="Clientes (amostra de 3.000, escala log)")
    ui.plot(fig, key="scatter")
    st.markdown("**Perfil por segmento e ação sugerida**")
    tabela = prof_k[["segmento", "clientes", "recencia_mediana_dias", "frequencia_mediana", "monetario_mediano", "pct_receita"]].copy()
    tabela["ação sugerida"] = tabela["segmento"].map(acao_para)
    st.dataframe(tabela.round(2), hide_index=True, width="stretch")
    st.caption("Os nomes e as ações são regras de demonstração, não recomendações validadas com dados de campanha. Medir o efeito de cada ação (teste A/B) é o passo seguinte.")
