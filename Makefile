# Atalhos do repositório. `make help` lista os alvos.
PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: help install app test data train data-churn-prediction train-churn-prediction data-customer-segmentation train-customer-segmentation data-ab-testing train-ab-testing data-demand-forecasting train-demand-forecasting data-eda-dados-br
help:            ## Lista os alvos
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-22s %s\n", $$1, $$2}'

install:         ## Cria o venv e instala as dependências do app
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -r requirements.txt -r requirements-dev.txt

app:             ## Sobe o app Streamlit
	$(BIN)/streamlit run app/Home.py

test:            ## Roda os testes (inclui o teste de fumaça do app)
	$(BIN)/pytest -q

# --- Dados e treino (rodam localmente; o app só lê os artefatos versionados em projects/*/data) ---
data:            ## Baixa/gera os dados de todos os projetos
	$(MAKE) data-churn-prediction data-customer-segmentation data-ab-testing data-demand-forecasting data-eda-dados-br

train:           ## Treina/gera métricas de todos os projetos com pipeline de modelagem
	$(MAKE) train-churn-prediction train-customer-segmentation train-ab-testing train-demand-forecasting

data-churn-prediction:  ## Baixa o Telco Customer Churn (IBM) para data/raw/
	PYTHONPATH=projects/churn-prediction/src $(BIN)/python -c "from churn_prediction.data import download; print(download())"

train-churn-prediction:  ## Treina logística vs boosting, calibra, escolhe limiar e grava artefatos
	PYTHONPATH=projects/churn-prediction/src $(BIN)/python -m churn_prediction.train

data-customer-segmentation:  ## Baixa e converte o Online Retail II (UCI) para parquet em data/raw/
	PYTHONPATH=projects/customer-segmentation/src $(BIN)/python -c "from customer_segmentation.data import load_raw; print(len(load_raw()))"

train-customer-segmentation:  ## Limpeza, RFM, escolha de k, KMeans e comparação com quintis
	PYTHONPATH=projects/customer-segmentation/src $(BIN)/python -m customer_segmentation.build

data-ab-testing:  ## Baixa o Cookie Cats (espelho público) para data/raw/
	PYTHONPATH=projects/ab-testing/src $(BIN)/python -c "from ab_testing.data import download; print(download())"

train-ab-testing:  ## Análise do Cookie Cats (testes, bayesiano, múltiplas métricas, peeking)
	PYTHONPATH=projects/ab-testing/src $(BIN)/python -m ab_testing.build

data-demand-forecasting:  ## Gera o dado sintético com schema Rossmann (seed 42)
	PYTHONPATH=projects/demand-forecasting/src $(BIN)/python -m demand_forecasting.make_sample

train-demand-forecasting:  ## Backtest com origem rolante (seasonal-naive, Holt-Winters, GBM) e intervalos
	PYTHONPATH=projects/demand-forecasting/src $(BIN)/python -m demand_forecasting.backtest

data-eda-dados-br:  ## Coleta as APIs do BCB (SGS) e do IBGE (SIDRA) e grava o snapshot
	PYTHONPATH=projects/eda-dados-br/src $(BIN)/python -m eda_dados_br.build
