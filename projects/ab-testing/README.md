# Testes A/B: a retenção mudou de verdade?

## Problema
Em um jogo mobile, o portão de progressão ficava no nível 30 (A) ou no nível 40 (B). A pergunta é se a mudança alterou a
retenção dos jogadores, e se o efeito é grande o bastante para importar. O projeto também cobre o caminho até a decisão:
verificação da amostra, tamanho da amostra, múltiplas métricas e o risco de espiar o resultado cedo.

## Dados
- **Cookie Cats** (Tactile Games, popularizado pelo DataCamp): 90.189 jogadores, colunas `userid`, `version`,
  `sum_gamerounds`, `retention_1`, `retention_7`. **Dado real.**
- Fonte: espelho público do CSV em `robguilarr/ab_testing_cookie_cats` (`datasets/cookie_cats.csv`), baixado sem login.
  Comparei com o espelho `ryanschaub/Mobile-Games-A-B-Testing-with-Cookie-Cats`: arquivos idênticos (2.707.297 bytes). **Licença não verificada**: o repositório
  do espelho não declara licença, e a origem é o material do DataCamp. Uso apenas para estudo.
- Schema validado na leitura (colunas, dois grupos, `userid` único).
- Versionado: `data/cookie_cats.parquet` (450 KB) e `data/metrics.json`.

## Abordagem
- **Retenção (proporção):** teste z de duas proporções (variância combinada sob H0), IC de Wilson por grupo,
  IC de Newcombe para a diferença, qui-quadrado como conferência.
- **Rodadas (contínua, assimétrica):** Mann-Whitney e bootstrap de média e mediana.
- **Bayesiano:** Beta-Binomial com priori uniforme; P(B > A) por Monte Carlo e perda esperada de escolher cada variante.
- **Verificação de amostra (SRM):** qui-quadrado de aderência à divisão 50/50.
- **Múltiplas métricas:** Bonferroni, Holm e Benjamini-Hochberg sobre os três p-valores.
- **Poder e tamanho de amostra:** fórmula normal para duas proporções, MDE por raiz numérica, curva de poder.
- **Peeking:** simulação A/A com 10 olhadas contra uma.
- Módulo `src/ab_testing/stats.py`: os testes em `tests/test_stats.py` comparam com `scipy` e `statsmodels` (z-test, qui-quadrado, Wilson, potência, Holm/Bonferroni/BH, SRM) e com casos conhecidos.

## Resultados
Números de `data/metrics.json`, gerado por `make train-ab-testing`.

| Métrica | A (gate 30) | B (gate 40) | Δ (B−A) | IC95% Δ (Newcombe) | p-valor | p ajustado (Holm) | P(B>A) |
|---|---|---|---|---|---|---|---|
| Retenção D1 | 44,8% | 44,2% | −0,59 p.p. | [−1,24; +0,06] | 0,074 | 0,100 | 0,037 |
| Retenção D7 | 19,0% | 18,2% | −0,82 p.p. | [−1,33; −0,31] | 0,0016 | 0,0047 | 0,0008 |
| Rodadas (média) | 52,5 | 51,3 | −1,2 | bootstrap [−4,1; +0,9] | 0,050 (MW) | 0,100 | n/a |

- **Verificação de amostra:** divisão de 49,56% para A, **SRM com p = 0,0086**. O app mostra esse alerta. Antes de
  concluir qualquer coisa com esses dados, o desvio de atribuição precisa ser investigado.
- Leitura: o gate 40 **reduziu** a retenção em D7 (≈0,8 p.p.), resultado que sobrevive a Holm e Bonferroni. Em D1 e nas rodadas,
  não há evidência suficiente. Com a ressalva do SRM, o resultado não deve ser tomado como definitivo.
- **Peeking (simulação A/A, 4.000 repetições):** uma olhada no fim dá falso positivo em 5,1%; dez olhadas com parada no primeiro p < 0,05 dão 18,5%.

## Como rodar
```bash
make data-ab-testing     # baixa o CSV para data/raw/ (gitignored)
make train-ab-testing    # análise, p-valores ajustados, simulação de peeking, grava metrics.json
make test                # inclui os testes deste projeto
```

## Limitações
- A licença e a aleatorização do dado original não foram verificadas na fonte; o SRM sugere um problema de atribuição.
- Priori uniforme no bayesiano; com amostra grande, a escolha pesa pouco, mas não foi testada outra.
- Métricas de retenção são binárias e agregadas; não há segmentação por país ou plataforma.
- A calculadora de tamanho de amostra usa aproximação normal; para proporções pequenas, a exata pode diferir um pouco.
- Não há intervalo sequencial; a seção de peeking só mostra o problema, não a correção (ex.: alfa gasto de O'Brien-Fleming).
