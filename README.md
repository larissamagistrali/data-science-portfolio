# Portfólio de Ciência de Dados

Cinco projetos de ciência de dados com problema, dados, método e resultado explícitos, e um app Streamlit que
serve de vitrine. Cada projeto tem pipeline reproduzível (`make data`, `make train`); o app só lê os artefatos pequenos
versionados em `projects/*/data/`.

**App no ar:** https://SEU-APP.streamlit.app (TODO: preencher após o deploy)

| Projeto | Pergunta que responde | Dados | Stack | Status |
|---|---|---|---|---|
| [churn-prediction](projects/churn-prediction/README.md) | Quem vai cancelar, e vale abordar? | Real: IBM Telco Customer Churn | scikit-learn, calibração, decisão por custo | Concluído |
| [customer-segmentation](projects/customer-segmentation/README.md) | Quais grupos de clientes existem e o que fazer com cada um? | Real: Online Retail II (UCI) | RFM, KMeans, pandas | Concluído |
| [ab-testing](projects/ab-testing/README.md) | A mudança melhorou a retenção, de verdade? | Real: Cookie Cats (espelho público) | scipy, statsmodels, Beta-Binomial | Concluído (SRM detectado no dado) |
| [demand-forecasting](projects/demand-forecasting/README.md) | Quanto cada loja vai vender, e com que intervalo? | **Sintético** (schema Rossmann) | Holt-Winters, GBM, backtest | Concluído |
| [eda-dados-br](projects/eda-dados-br/README.md) | Como juros, inflação, câmbio e desemprego se movem? | Real: APIs BCB (SGS) e IBGE (SIDRA) | pandas, plotly | Concluído |

## Como rodar

```bash
make install      # cria .venv e instala requirements.txt + requirements-dev.txt
make app          # sobe o app em http://localhost:8501
make test         # pytest: teste de fumaça de todas as páginas + testes dos projetos
make data         # baixa/gera os dados (rede necessária; cache em projects/*/data/raw/)
make train        # reexecuta os pipelines e regrava os artefatos em projects/*/data/
make help         # lista todos os alvos, incluindo um por projeto
```

Pré-requisito: Python 3.11+.

## Deploy no Streamlit Community Cloud

- Arquivo principal: `app/Home.py`
- Python 3.11 ou superior; dependências em `requirements.txt` (sem pins, o que o Cloud instala).
- Segredos: não há segredos no repo. Para uso local com segredos, `.streamlit/secrets.toml` (gitignored).
- O app depende só de `projects/*/data/` e de `shared/`; não baixa bases grandes no servidor.

## Estrutura

```
app/                 Home.py e pages/ (uma por projeto; abas Problema · Dados · Abordagem · Resultado · Demo)
shared/              tema dos gráficos, estrutura de página, acesso a dados (DuckDB/Parquet)
projects/<nome>/     src/ (pacote Python), tests/, data/ (amostras e agregados pequenos), README.md
tests/               teste de fumaça: roda Home e cada página com streamlit.testing.AppTest
```

## Sobre a origem dos dados

Quatro projetos usam dados reais de fontes públicas sem autenticação: IBM Telco (churn), Online Retail II (segmentação),
Cookie Cats (A/B) e as APIs do BCB e do IBGE (indicadores). O projeto de previsão de demanda usa um **gerador sintético com o
schema do Rossmann**, porque o dataset original exige credencial do Kaggle. Cada página e cada README indicam a origem e a licença.
Os números de resultado vêm dos arquivos `metrics.json` gerados pelo pipeline, não de texto digitado.
