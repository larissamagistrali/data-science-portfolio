# Previsão de demanda diária por loja

## Problema
Previsão de demanda orienta compra, estoque e equipe. Erro para cima vira excesso; erro para baixo vira venda perdida.
O projeto compara três modelos com **backtest de origem rolante** e mede o **intervalo de previsão**, não só o ponto.

## Dados
- **SINTÉTICO, com o schema do Rossmann Store Sales** (Kaggle). O dataset real exige credencial do Kaggle, que não está
  disponível aqui. Por isso gerei uma base com as mesmas colunas: `Store`, `Date`, `Sales`, `Customers`, `Open`, `Promo`,
  `StateHoliday`, `SchoolHoliday`, `StoreType`, `Assortment`, `CompetitionDistance`.
- Gerador em `src/demand_forecasting/make_sample.py`, determinístico (seed 42): 30 lojas × 2013-01-01 a 2015-12-31 (32.850 linhas).
  Tem nível por loja, tendência suave, sazonalidade semanal (sábado forte, domingo fechado), anual (verão e dezembro),
  promoções em blocos semanais, feriados escolares, feriados nacionais fechados (1º/jan e 25/dez) e ruído log-normal.
- **Os padrões são os que eu escrevi.** Os resultados ilustram o pipeline; não representam desempenho em varejo real, e a
  ordem dos modelos pode mudar no dado real.
- Caminho para o dado real (**não executado aqui**): `kaggle competitions download -c rossmann-store-sales`, depois
  adaptar a leitura para o CSV original.

## Abordagem
- **Seasonal-naive:** repete o último valor positivo do mesmo dia da semana. Dias fechados e feriados são pulados (até 8 semanas para trás):
  o calendário de fechamento é conhecido, então copiar o zero de um feriado para um dia aberto seria um erro evitável.
- **Holt-Winters (ETS):** aditivo, tendência amortecida, sazonalidade semanal (m = 7), um modelo por loja (statsmodels).
- **Gradient boosting:** modelo global. Features: calendário, promoção, feriados (conhecidos antecipadamente), características da
  loja e lags de 28 e 364 dias. Os lags são escolhidos para ficar antes da origem em qualquer horizonte até 28 dias.
  Teste automatizado: alterar vendas após a origem não muda a previsão.
- **Backtest:** origens em 2015-01-01, 04-02, 07-02 e 10-01; horizonte de 28 dias; avaliação só em dias abertos.
- **Intervalo de 90%:** quantis 5% e 95% do erro relativo `(y − ŷ) / ŷ`, calibrados **apenas nas origens anteriores** e só em previsões
  positivas (dividir por ~0 gera razões de milhares). A primeira origem não tem intervalo.
- **Métricas:** WAPE, MAPE (dias com venda > 0) e MASE (escala = erro do naive sazonal no histórico).

## Resultados
Números de `data/metrics.json`, gerado por `make train-demand-forecasting`. **Dado sintético.**

| Modelo | WAPE | MAPE | MASE | Cobertura do IC 90% (média, 3 origens) |
|---|---|---|---|---|
| Gradient boosting | 7,3% | 7,4% | 0,61 | 90,4% |
| Holt-Winters | 12,1% | 11,8% | 1,00 | 81,1% |
| Seasonal-naive | 14,0% | 14,2% | 1,19 | 89,4% |

- Por origem, WAPE do GBM: 7,4%, 7,1%, 7,5%, 7,3%. Holt-Winters tem 15,9% na primeira origem (sem tendência estável no início).
- Seasonal-naive por origem: WAPE 14,7%, 13,7%, 14,3%, 13,3%; cobertura 86%, 90% e 92%.
- Holt-Winters fica abaixo do nominal nas três origens (75%, 81% e 87%): o intervalo calibrado nas origens anteriores é estreito demais para ele.

## Como rodar
```bash
make data-demand-forecasting     # gera data/sales_sample.parquet (sintético)
make train-demand-forecasting    # backtest, previsões por loja e métricas (~30 s)
make test
```

## Limitações
- Dado sintético: o desempenho absoluto e o ranking entre modelos não se transferem para o Rossmann real sem reavaliação.
- A cobertura é medida em 3 origens; o número tem muita variância para uma conclusão forte.
- Holt-Winters por loja não usa sazonalidade anual; o GBM usa o dia do ano, o que dá vantagem nesse dado.
- Não há previsão de estoque, custo de ruptura nem otimização de pedido; só a previsão de vendas.
- Plano de promoções e de dias abertos é tratado como conhecido no futuro, premissa comum em varejo, mas não verificada.
- O app mostra as previsões do backtest (origens fixas), não previsões novas para datas arbitrárias.
