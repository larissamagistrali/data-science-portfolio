"""Testes A/B: análise do Cookie Cats e calculadoras de significância, tamanho de amostra e peeking."""
import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from shared import data as D
from shared import theme
from shared import ui

ui.setup_page("Testes A/B", "🧪")
ui.header(
    "Testes A/B",
    "A mudança melhorou a retenção de verdade, ou foi sorte da amostra?",
    tags=["estatística frequentista", "Beta-Binomial", "bootstrap", "poder", "múltiplas métricas"],
    code_path="projects/ab-testing",
)
problema, dados, abordagem, resultado, demo = ui.tabs()

PROJ = "ab-testing"
D.use_project(PROJ)
from ab_testing import stats as S  # noqa: E402

metrics = D.read_json(str(D.data_path(PROJ, "metrics.json")))
df = D.read_parquet(str(D.data_path(PROJ, "cookie_cats.parquet")))
A_LBL, B_LBL = "gate_30 (A)", "gate_40 (B)"

with problema:
    st.markdown(
        """
Em um jogo mobile, o portão de progressão (*gate*) ficava no nível 30 ou no nível 40. A pergunta:
mover o portão para o nível 40 **mudou a retenção** dos jogadores? E, se mudou, o efeito é grande
o bastante para importar, ou é só ruído de amostragem?

O foco é a **interpretação**: IC, p-valor, probabilidade bayesiana e o que cada um responde.
"""
    )

with dados:
    ui.data_notice(
        "real",
        "Cookie Cats (Tactile Games; espelho público do CSV em github.com/robguilarr/ab_testing_cookie_cats)",
        f"{metrics['linhas']:,} jogadores. Colunas: userid, version, sum_gamerounds, retention_1, retention_7. "
        "Licença do dado original não verificada no espelho: uso aqui apenas para estudo.",
    )
    st.markdown(
        "A aleatorização e a licença não foram verificadas na fonte original. Antes de decidir algo real, verificar as condições do DataCamp/Tactile."
    )
    fig = go.Figure()
    capped = df["sum_gamerounds"].clip(upper=200)
    edges = np.linspace(0, 200, 61)
    for v, cor in ((A_LBL.split(" ")[0], theme.CATEGORICAL[0]), (B_LBL.split(" ")[0], theme.CATEGORICAL[1])):
        # Linhas em degrau em vez de barras translúcidas sobrepostas: a sobreposição criava uma terceira cor.
        dens, _ = np.histogram(capped[df["version"] == v], bins=edges, density=True)
        fig.add_trace(go.Scatter(x=edges, y=np.append(dens, dens[-1]), mode="lines", line_shape="hv", name=v, line=dict(color=cor, width=2)))
    fig.update_layout(title="Rodadas jogadas (densidade; 200 ou mais agrupadas no último intervalo)", xaxis_title="Rodadas", yaxis_title="Densidade")
    ui.plot(fig, key="hist")

with abordagem:
    st.markdown(
        """
- **Retenção (proporção):** teste z de duas proporções, IC de Wilson por grupo e IC de Newcombe para a diferença. Qui-quadrado como conferência (equivalente sem correção de continuidade).
- **Rodadas (contínua, muito assimétrica):** Mann-Whitney e bootstrap da média e da mediana.
- **Bayesiano:** Beta-Binomial com priori uniforme. P(B > A) e **perda esperada** de escolher cada variante.
- **Verificação de amostra (SRM):** qui-quadrado de aderência da divisão 50/50. Se falhar, algo está quebrado na atribuição, e nenhum resultado deve ser lido antes de investigar.
- **Múltiplas métricas:** Bonferroni, Holm e Benjamini-Hochberg sobre os três p-valores.
"""
    )

with resultado:
    r1, r7 = metrics["retention_1"], metrics["retention_7"]
    srm = metrics["srm"]
    ui.metric_row([
        ("Retenção D1 · A → B", f"{r1['A']['taxa']:.1%} → {r1['B']['taxa']:.1%}", f"Δ = {r1['diff']*100:+.2f} p.p. · p = {r1['p_value']:.3f}"),
        ("Retenção D7 · A → B", f"{r7['A']['taxa']:.1%} → {r7['B']['taxa']:.1%}", f"Δ = {r7['diff']*100:+.2f} p.p. · p = {r7['p_value']:.4f}"),
        ("P(B > A) em D7", f"{r7['bayes']['p_b_beats_a']:.1%}", "Beta-Binomial, priori uniforme"),
    ])
    if srm["p_value"] < 0.01:
        st.error(f"❌ Alerta de SRM: a divisão dos grupos desvia de 50/50 (p = {srm['p_value']:.4f}, A = {srm['share_a']:.2%}). "
                 "Isso pode indicar falha na atribuição. Investigar antes de confiar nos resultados.")
    else:
        st.success(f"✅ Divisão compatível com 50/50 (p = {srm['p_value']:.3f}).")
    rows = []
    for nome, r in (("Retenção D1", r1), ("Retenção D7", r7)):
        rows.append({"métrica": nome, "taxa A": r["A"]["taxa"], "taxa B": r["B"]["taxa"], "Δ (B−A)": r["diff"],
                     "IC95% Δ (Newcombe)": f"[{r['ic95_diff_newcombe'][0]*100:+.2f}, {r['ic95_diff_newcombe'][1]*100:+.2f}] p.p.",
                     "p-valor": r["p_value"], "P(B>A)": r["bayes"]["p_b_beats_a"], "perda se escolher B": r["bayes"]["expected_loss_choose_b"]})
    g = metrics["gamerounds"]
    rows.append({"métrica": "Rodadas (média)", "taxa A": g["A"]["media"], "taxa B": g["B"]["media"], "Δ (B−A)": g["B"]["media"] - g["A"]["media"],
                 "IC95% Δ (Newcombe)": f"bootstrap [{g['bootstrap_media']['ci_low']:+.2f}, {g['bootstrap_media']['ci_high']:+.2f}]",
                 "p-valor": g["mann_whitney_p"], "P(B>A)": np.nan, "perda se escolher B": np.nan})
    st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
    adj = pd.DataFrame(metrics["p_ajustados"]).round(4)
    st.markdown("**P-valores ajustados para múltiplas métricas**")
    st.dataframe(adj, width="stretch")
    st.caption("Com Bonferroni, o D7 continua significativo; D1 e rodadas não passam em nenhum ajuste. "
               "A diferença em D7 é pequena (≈0,8 p.p.) e pode não justificar a mudança sozinha.")
    st.markdown("**Peeking (espiar cedo)**")
    pk = metrics["peeking"]
    st.write(f"Simulação A/A (sem efeito real), {pk['n_sims']:,} repetições: olhar uma vez no fim dá falso positivo em **{pk['fpr_single_look']:.1%}** "
             f"(próximo de α = {pk['alpha']:.0%}). Olhar {pk['looks']} vezes e parar no primeiro p < α dá **{pk['fpr_peeking']:.1%}**.")

with demo:
    st.subheader("(a) Calculadora de significância")
    st.caption("Digite conversões e visitantes de cada variante (ex.: cliques e impressões).")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Variante A (controle)**")
        xa = st.number_input("Conversões A", 0, 10_000_000, 430, key="xa")
        na = st.number_input("Visitantes A", 1, 10_000_000, 1000, key="na")
    with c2:
        st.markdown("**Variante B (tratamento)**")
        xb = st.number_input("Conversões B", 0, 10_000_000, 480, key="xb")
        nb = st.number_input("Visitantes B", 1, 10_000_000, 1000, key="nb")
    alt = st.radio("Hipótese alternativa", ["two-sided", "larger", "smaller"], horizontal=True,
                   format_func={"two-sided": "bilateral (B ≠ A)", "larger": "unilateral (B > A)", "smaller": "unilateral (B < A)"}.get)
    if xa > na or xb > nb:
        st.error("Conversões não podem ser maiores que visitantes.")
    else:
        z = S.two_proportion_ztest(int(xa), int(na), int(xb), int(nb), alternative=alt)
        lo, hi = S.diff_ci_newcombe(int(xa), int(na), int(xb), int(nb))
        bay = S.bayesian_ab(int(xa), int(na), int(xb), int(nb), n_samples=200_000)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Δ (B − A)", f"{z['diff']*100:+.2f} p.p.", f"relativo {z['diff']/z['p1']:+.1%}" if z["p1"] > 0 else None)
        m2.metric("p-valor", f"{z['p_value']:.4f}", "significativo a 5%" if z["p_value"] < 0.05 else "não significativo a 5%")
        m3.metric("P(B > A)", f"{bay['p_b_beats_a']:.1%}", help="Beta-Binomial com priori uniforme")
        m4.metric("Perda esperada se escolher B", f"{bay['expected_loss_choose_b']*100:.3f} p.p.", help="Quanto se perde, em média, se B for pior")
        st.write(f"IC95% da diferença (Newcombe): **[{lo*100:+.2f}, {hi*100:+.2f}] p.p.** · IC credível 95% (Bayes): "
                 f"[{bay['diff_ci_low']*100:+.2f}, {bay['diff_ci_high']*100:+.2f}] p.p.")
        st.caption("Significância não é importância prática: olhe o IC. Um efeito pode ser significativo e pequeno demais para justificar a mudança.")

    st.subheader("(b) Tamanho de amostra e curva de poder")
    p1 = st.slider("Taxa de base do controle", 0.01, 0.95, 0.10, 0.01)
    mde = st.slider("Menor efeito que quero detectar", 0.001, 0.20, 0.02, 0.001, help="Diferença absoluta entre taxas (ex.: 0,02 = de 10% para 12%)")
    alpha = st.select_slider("α", [0.01, 0.05, 0.10], value=0.05)
    poder = st.select_slider("Poder desejado", [0.7, 0.8, 0.9], value=0.8)
    if p1 + mde >= 1:
        st.error("A taxa de base mais o efeito passa de 100%.")
    else:
        n = S.required_sample_size(p1, mde, alpha=alpha, power=poder)
        st.metric("Visitantes por grupo", f"{n:,}", f"{2*n:,} no total")
        ns = np.linspace(max(100, n // 4), n * 2, 60).astype(int)
        curve = S.power_curve(p1, mde, ns, alpha=alpha)
        fig = go.Figure(go.Scatter(x=ns, y=curve, mode="lines", name="Poder", line=dict(color=theme.CATEGORICAL[0])))
        fig.add_hline(y=poder, line=dict(dash="dot", color=theme.BASELINE), annotation_text=f"poder {poder:.0%}")
        fig.add_vline(x=n, line=dict(dash="dot", color=theme.BASELINE), annotation_text=f"n = {n:,}")
        fig.update_layout(title="Poder em função do tamanho do grupo (aproximação normal)", xaxis_title="Visitantes por grupo", yaxis_title="Poder", yaxis_range=[0, 1])
        ui.plot(fig, key="power")
        mdev = S.minimum_detectable_effect(p1, n, alpha=alpha, power=poder)
        st.caption(f"Efeito mínimo detectável com esse n: {mdev*100:.2f} p.p. Amostra menor que isso só detecta efeitos maiores.")

    st.subheader("(c) Peeking: quantas vezes você olhou?")
    looks = st.slider("Número de olhadas até o fim", 1, 20, 10)
    sims = S.simulate_peeking(p=p1, n_per_group=5000, looks=looks, n_sims=2000, seed=7)
    st.write(f"Simulação A/A com taxa {p1:.0%}, 5.000 por grupo: falso positivo com **1 olhada = {sims['fpr_single_look']:.1%}**; "
             f"com **{looks} olhadas e parada no 1º p < 0,05 = {sims['fpr_peeking']:.1%}**.")
    st.warning("⚠️ Espiar e parar no primeiro p < 0,05 infla o erro tipo I. Defina o tamanho da amostra antes, ou use um método de parada sequencial.")
