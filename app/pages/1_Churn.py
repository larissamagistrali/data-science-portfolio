"""Churn: simulador de cliente, explicação por variável e decisão por custo."""
import _bootstrap  # noqa: F401

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve, precision_recall_curve

from shared import data as D
from shared import theme
from shared import ui

ui.setup_page("Churn", "📉")
ui.header(
    "Previsão de churn",
    "Qual cliente de telecom está prestes a cancelar, e vale a pena abordá-lo?",
    tags=["scikit-learn", "regressão logística", "HistGradientBoosting", "calibração"],
    code_path="projects/churn-prediction",
)
problema, dados, abordagem, resultado, demo = ui.tabs()

PROJ = "churn-prediction"
METRICS = D.data_path(PROJ, "metrics.json")
MODELS = D.data_path(PROJ, "models.joblib")
SAMPLE = D.data_path(PROJ, "sample.parquet")
PREDS = D.data_path(PROJ, "test_predictions.parquet")
D.use_project(PROJ)

from churn_prediction import data as CD  # noqa: E402
from churn_prediction.explain import contributions  # noqa: E402
from churn_prediction.decision import breakeven_probability  # noqa: E402
from churn_prediction.model import build_logreg  # noqa: E402

metrics = D.read_json(str(METRICS))
sample = D.read_parquet(str(SAMPLE))
train = sample[sample["split"] == "treino"].drop(columns=["split"])


@st.cache_resource(show_spinner="Carregando o modelo...")
def load_model() -> dict:
    """Carrega o joblib; se a versão da lib não bater, retreina a regressão logística na amostra de treino."""
    try:
        bundle = D.load_joblib(str(MODELS))
        bundle["logreg"].predict_proba(train[CD.FEATURES].head(2))  # teste de compatibilidade
        return {**bundle, "origem": "joblib"}
    except Exception:  # noqa: BLE001 - a página nunca deve quebrar por versão de biblioteca
        X = train[CD.FEATURES]
        model = build_logreg(C=metrics["cv"]["logreg"]["params"]["C"]).fit(X, train[CD.TARGET])
        z_bar = model.named_steps["prep"].transform(X).mean(axis=0)
        return {"logreg": model, "logreg_base": model, "z_bar": z_bar, "origem": "retreinado"}


bundle = load_model()
model = bundle["logreg"]
z_bar = bundle["z_bar"]
cats = {c: sorted(sample[c].astype(str).unique().tolist()) for c in CD.CATEGORICAL}

with problema:
    st.markdown(
        """
Operadoras perdem clientes continuamente, e reter custa menos do que conquistar. O problema
é **escolher quem abordar**: ação de retenção (desconto, contato, benefício) tem custo, e
abordar quem não sairia é dinheiro jogado fora.

Esta página transforma a pergunta em três números: a **probabilidade de churn** de um cliente,
**o que puxa essa probabilidade** para cima ou para baixo, e **se a abordagem compensa** dado
o custo da ação e o valor do cliente.
"""
    )

with dados:
    ui.data_notice(
        "real",
        "IBM Telco Customer Churn (espelho público no GitHub, licença Apache-2.0 no repositório de origem)",
        f"{metrics['dataset']['linhas']:,} clientes, {metrics['dataset']['colunas_originais']} colunas originais. "
        f"Taxa de churn: {metrics['dataset']['taxa_churn']:.1%}. Os dados são de um cenário de demonstração, não de uma operadora real.",
    )
    st.markdown(
        f"""
**Limpeza.** `TotalCharges` tem {metrics['dataset']['total_charges_em_branco']} valores em branco (clientes com tenure 0,
sem cobrança ainda): viram 0. `customerID` é removido. `TotalCharges` **não entra no modelo**: a correlação dele com
`tenure × MonthlyCharges` é {metrics['ablacao_total_charges']['corr_total_vs_tenure_x_monthly']:.4f}, e a ablação
mostrou ganho desprezível ({metrics['ablacao_total_charges']['sem_total_charges']['cv_roc_auc']:.4f} sem contra
{metrics['ablacao_total_charges']['com_total_charges']['cv_roc_auc']:.4f} com).

**Divisão.** Treino/teste estratificados ({metrics['dataset']['treino']} / {metrics['dataset']['teste']}), seed 42. O
teste só é usado para avaliação final.
"""
    )

with abordagem:
    st.markdown(
        """
- **Baseline:** regressão logística com padronização e one-hot. Interpretável e bem calibrada por construção.
- **Comparação:** `HistGradientBoostingClassifier`, com grade de hiperparâmetros por validação cruzada estratificada de 5 dobras no treino.
- **Regra de escolha do campeão:** o modelo linear só perde para o de boosting se a diferença de AUC for maior que um erro-padrão da CV. Assim, não se troca interpretabilidade por ganho de ruído.
- **Calibração:** comparei sem calibração, Platt (sigmoid) e isotônica por Brier na CV do treino. Escolhido por modelo, ver aba Resultado.
- **Limiar:** escolhido por custo de negócio nas predições fora-da-dobra do treino (não no teste).
- **Explicação:** a regressão logística é linear no log-odds, então a contribuição de cada variável é **exata**: `logit = média + soma das contribuições`. Não usei SHAP; a decomposição vem do próprio modelo.
"""
    )

with resultado:
    cv = metrics["cv"]
    te = metrics["teste"]
    ui.metric_row([
        ("ROC-AUC (teste) · logística", f"{te['logreg']['roc_auc']:.3f}", f"IC95% {te['logreg']['roc_auc_ci95'][0]:.3f}–{te['logreg']['roc_auc_ci95'][1]:.3f}"),
        ("ROC-AUC (teste) · boosting", f"{te['hgb']['roc_auc']:.3f}", f"IC95% {te['hgb']['roc_auc_ci95'][0]:.3f}–{te['hgb']['roc_auc_ci95'][1]:.3f}"),
        ("PR-AUC (teste) · campeão", f"{te[metrics['campeao']['modelo']]['pr_auc']:.3f}", f"Prevalência de churn: {metrics['teste']['baseline_prevalencia']['pr_auc']:.3f}"),
        ("Brier (teste) · campeão", f"{te[metrics['campeao']['modelo']]['brier']:.3f}", "Quanto menor, melhor a calibração"),
    ])
    st.caption(f"Campeão: **{metrics['campeao']['modelo']}** ({metrics['campeao']['regra']}). Métricas de CV (5 dobras, treino): "
               f"logística AUC {cv['logreg']['roc_auc']:.3f} ± {cv['logreg']['roc_auc_std']:.3f}; boosting AUC {cv['hgb']['roc_auc']:.3f} ± {cv['hgb']['roc_auc_std']:.3f}.")
    preds = D.read_parquet(str(PREDS))
    fig = go.Figure()
    for col, name in (("p_logreg", "Logística"), ("p_hgb", "Boosting")):
        fpr, tpr, _ = roc_curve(preds["y"], preds[col])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aleatório", line=dict(dash="dot", color=theme.BASELINE)))
    fig.update_layout(title="Curva ROC (teste)", xaxis_title="Falsos positivos", yaxis_title="Verdadeiros positivos")
    ui.plot(fig, key="roc")
    fig = go.Figure()
    for col, name in (("p_logreg", "Logística"), ("p_hgb", "Boosting")):
        pr, rc, _ = precision_recall_curve(preds["y"], preds[col])
        fig.add_trace(go.Scatter(x=rc, y=pr, mode="lines", name=name))
    fig.add_hline(y=float(preds["y"].mean()), line=dict(dash="dot", color=theme.BASELINE), annotation_text="prevalência")
    fig.update_layout(title="Curva precisão × revocação (teste)", xaxis_title="Revocação", yaxis_title="Precisão")
    ui.plot(fig, key="pr")
    cal = pd.DataFrame(metrics["calibracao_tabela_teste"][metrics["campeao"]["modelo"]])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cal["p_mean"], y=cal["freq"], mode="markers+lines", name="Observado por decil"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Calibração perfeita", line=dict(dash="dot", color=theme.BASELINE)))
    fig.update_layout(title="Calibração (teste): probabilidade prevista × frequência observada", xaxis_title="Prevista", yaxis_title="Observada")
    ui.plot(fig, key="cal")
    with st.expander("Ablação do TotalCharges, calibração e limiar"):
        st.json({k: metrics[k] for k in ("ablacao_total_charges", "calibracao_escolhida", "decisao_padrao")})

with demo:
    st.subheader("Simulador de cliente")
    st.caption(f"Modelo em uso: {bundle['origem']}. Probabilidade da regressão logística (campeão).")
    c1, c2, c3 = st.columns(3)
    with c1:
        tenure = st.slider("Meses como cliente (tenure)", 0, 72, 12)
        monthly = st.slider("Mensalidade (R$)", 18.0, 120.0, 70.0, 0.5)
        senior = st.selectbox("Idoso (65+)", [0, 1], format_func=lambda v: "Sim" if v else "Não")
        contract = st.selectbox("Contrato", cats["Contract"])
        payment = st.selectbox("Forma de pagamento", cats["PaymentMethod"])
    with c2:
        internet = st.selectbox("Internet", cats["InternetService"])
        phone = st.selectbox("Telefone fixo", ["Yes", "No"])
        paperless = st.selectbox("Fatura sem papel", ["Yes", "No"])
        partner = st.selectbox("Tem parceiro(a)", ["Yes", "No"])
        dependents = st.selectbox("Tem dependentes", ["Yes", "No"])
    with c3:
        addon_opts = ["Yes", "No"]
        security = st.selectbox("Segurança online", addon_opts, index=1)
        support = st.selectbox("Suporte técnico", addon_opts, index=1)
        backup = st.selectbox("Backup online", addon_opts, index=1)
        protection = st.selectbox("Proteção de dispositivo", addon_opts, index=1)
        streaming = st.selectbox("Streaming de TV", addon_opts, index=1)
    gender = st.selectbox("Gênero (usado só para compatibilidade do modelo)", cats["gender"], index=0)

    st.subheader("Decisão de retenção")
    d1, d2, d3 = st.columns(3)
    threshold = d1.slider("Limiar de abordagem", 0.0, 1.0, 0.32, 0.01, help="Abordar se a probabilidade for maior ou igual a este valor.")
    cost = d2.number_input("Custo da ação por cliente (R$)", min_value=0.0, value=60.0, step=5.0)
    success = d3.slider("Taxa de sucesso da ação", 0.0, 1.0, 0.25, 0.01, help="Fração dos clientes que iriam sair e ficam por causa da ação.")
    months_value = st.slider("Meses de receita considerados como valor do cliente", 1, 36, metrics["decisao_padrao"]["meses_de_valor"])

    row = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone, "MultipleLines": "Yes" if phone == "Yes" else "No phone service", "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection, "TechSupport": support,
        "StreamingTV": streaming, "StreamingMovies": streaming, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly,
    }
    row = CD.normalize_customer(row)
    X = pd.DataFrame([row])[CD.FEATURES]
    for c in CD.CATEGORICAL:
        X[c] = pd.Categorical(X[c].astype(str), categories=cats[c])
    p = float(model.predict_proba(X)[0, 1])
    value = monthly * months_value
    gain = p * success * value - cost
    be = breakeven_probability(cost, success, value)

    m1, m2, m3 = st.columns(3)
    m1.metric("Probabilidade de churn", f"{p:.1%}")
    m2.metric("Ganho esperado de abordar", f"R$ {gain:,.2f}", help="p × sucesso × valor do cliente − custo")
    m3.metric("Probabilidade de equilíbrio", f"{be:.1%}" if np.isfinite(be) else "n/d", help="Abaixo disso, a ação não se paga")
    if p >= threshold and gain > 0:
        st.success("✅ Abordar: a probabilidade passa do limiar e o ganho esperado é positivo.")
    elif p >= threshold:
        st.warning("⚠️ Acima do limiar, mas o ganho esperado é negativo: a ação custa mais do que retém.")
    else:
        st.info("ℹ️ Não abordar: probabilidade abaixo do limiar.")

    contrib, base, logit = contributions(model, X, z_bar)
    contrib = contrib.head(10).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=contrib["contribution"], y=contrib["feature"], orientation="h",
        marker_color=np.where(contrib["contribution"] > 0, theme.CATEGORICAL[1], theme.CATEGORICAL[0]),
        text=[f"{v}" for v in contrib["value"]], textposition="inside", insidetextanchor="middle", textangle=0,
        hovertemplate="%{y} = %{text}<br>contribuição: %{x:+.3f} log-odds<extra></extra>",
    ))
    # Valor que não cabe na barra some (fica no tooltip) em vez de transbordar sobre o eixo em telas estreitas.
    fig.update_layout(title="O que puxa a probabilidade (contribuição em log-odds, exata)", xaxis_title="Laranja: aumenta o risco · Azul: reduz o risco",
                      uniformtext=dict(minsize=10, mode="hide"))
    ui.plot(fig, key="contrib")
    st.caption(
        f"Soma exata: log-odds médio da base = {base:.2f}; log-odds deste cliente = {logit:.2f}. "
        "Mostradas as 10 variáveis de maior impacto absoluto; cada barra é a contribuição dessa variável frente ao cliente médio da base de treino."
    )
