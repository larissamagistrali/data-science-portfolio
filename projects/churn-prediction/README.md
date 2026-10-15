# Churn: quem vai cancelar e vale a pena abordar?

## Problema
Reter cliente custa menos do que conquistar outro, mas ação de retenção (desconto, contato, benefício) tem custo.
Abordar quem não sairia é desperdício. O projeto estima a **probabilidade de churn** de cada cliente, explica o que
puxa essa probabilidade e decide quem abordar, dado o custo da ação e o valor do cliente.

## Dados
- **IBM Telco Customer Churn**, 7.043 clientes × 21 colunas, taxa de churn de 26,5%. **Dado real**, mas de um
  cenário de demonstração, não de uma operadora de verdade.
- Fonte: espelho do CSV no repositório público `IBM/telco-customer-churn-on-icp4d` (`data/Telco-Customer-Churn.csv`),
  baixado sem autenticação. Licença do repositório de origem: Apache-2.0.
- Limpeza: 11 valores em branco de `TotalCharges` (clientes com tenure 0) viram 0; `customerID` é removido.
- `TotalCharges` **não entra no modelo**: correlação de 0,9996 com `tenure × MonthlyCharges`, e a ablação não
  melhora a AUC (CV 0,8449 sem contra 0,8462 com, diferença menor que um desvio-padrão).
- Divisão estratificada 80/20 (treino 5.634, teste 1.409), seed 42. O teste só entra na avaliação final.

## Abordagem
1. **Baseline:** regressão logística (padronização + one-hot).
2. **Comparação:** `HistGradientBoostingClassifier` com grade pequena, validação cruzada estratificada de 5 dobras no treino.
3. **Campeão:** a logística, pela regra de 1 erro-padrão da CV. O boosting tem AUC de CV um pouco maior (0,8497 contra 0,8450),
   mas a diferença é menor que o erro-padrão (0,005).
4. **Calibração:** sem calibração, Platt (sigmoid) e isotônica, escolhidos por Brier na CV do treino. A logística ficou sem calibração.
5. **Limiar por custo:** abordar se `p × sucesso × valor − custo > 0`. O limiar (0,32) é escolhido nas predições fora-da-dobra
   do treino, não no teste. Custo de R$ 60, sucesso de 25% e 12 meses de valor são **valores ilustrativos**, ajustáveis no app.
6. **Explicação:** a logística é linear no log-odds, então a contribuição de cada variável é **exata**:
   `logit = média da base + soma das contribuições`. Não usei SHAP; a decomposição vem do próprio modelo.

## Resultados
Números copiados de `data/metrics.json`, gerado por `make train-churn-prediction`.

| Modelo (teste) | ROC-AUC | IC95% AUC (bootstrap) | PR-AUC | Brier | ECE |
|---|---|---|---|---|---|
| Logística (campeão) | 0,838 | 0,815–0,859 | 0,625 | 0,140 | 0,022 |
| HistGradientBoosting | 0,848 | 0,827–0,869 | 0,659 | 0,135 | 0,021 |

- Prevalência no teste: 0,265 (linha de base do PR-AUC).
- Limiar 0,32 no teste: aborda 507 clientes (36%), recall 0,72, precisão 0,53.
- Ganho esperado no teste (com os valores ilustrativos): R$ 31,7 mil, contra −R$ 2,9 mil de abordar todos e R$ 0 de não abordar.
  Esse número depende dos parâmetros de custo e sucesso, que não foram validados com dados de campanha.

## Como rodar
```bash
make data-churn-prediction     # baixa o CSV para projects/churn-prediction/data/raw/ (gitignored)
make train-churn-prediction    # treina, valida, calibra e grava data/*.joblib, metrics.json e amostras
make test                      # inclui os testes deste projeto
```
O app lê `data/` (`models.joblib`, `metrics.json`, `sample.parquet`, `test_predictions.parquet`). Se o joblib não carregar
por diferença de versão de biblioteca, a página retreina a logística a partir da amostra de treino e avisa a origem do modelo.

## Limitações
- Dado de demonstração: o desempenho aqui não garante desempenho em operadora real.
- Probabilidades são calibradas no teste, mas o ECE de 0,02 é de uma única divisão; não há intervalo de calibração.
- Custo, taxa de sucesso e valor do cliente são premissas, não medidas. O limiar é ótimo para essas premissas.
- Explicação linear só vale para o campeão linear. O boosting não tem explicação própria aqui.
