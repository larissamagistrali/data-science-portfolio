# Indicadores econômicos e regionais do Brasil

## Problema
Juros, inflação, câmbio e desemprego andam juntos, com defasagens. O projeto coleta as séries públicas, compara indicadores
em um eixo só (base 100), calcula variação em 12 meses, mede correlação defasada e compara estados com dados do IBGE.

## Dados
**Dados reais, sem autenticação:**
- **BCB, SGS** (`api.bcb.gov.br/dados/serie/bcdata.sgs.{código}/dados`): Selic meta (432), IPCA mensal (433), dólar comercial
  venda (1) e taxa de desocupação (24369). Desde 2010. Códigos conferidos na API.
- **IBGE, SIDRA** (`apisidra.ibge.gov.br`): IPCA índice geral (tabela 1737, variável 2266), população residente por UF
  (6579 / 9324) e taxa de desocupação por UF e trimestre (4099).
- Coleta com User-Agent descritivo, cache em `data/raw/` (gitignored), retry com pausa, e janelas de 5 anos no SGS
  (a API recusa séries diárias longas com HTTP 406).
- Snapshot versionado: `data/*.parquet` (~350 KB no total) e `data/meta.json` com a **data da coleta** (2026-10-14).
- **Base dos Dados não foi usada**: exige um projeto no Google Cloud (BigQuery). Ficaria como extensão: as mesmas tabelas
  do IBGE e do BCB estão lá, com SQL.
- **Confirmar:** o código SGS 24369 foi rotulado como "taxa de desocupação (PNAD Contínua)" pela convenção do BCB. O rótulo
  não foi conferido no catálogo oficial do SGS.

## Abordagem
- Séries diárias (Selic, dólar) agregadas para o último valor do mês, para comparar com as mensais.
- **Índice base 100** no primeiro mês do período escolhido, num eixo só. Só entram níveis: a variação mensal (IPCA) vira antes
  índice acumulado, Π(1 + r/100), e as taxas em nível (Selic, desocupação) ficam fora, porque base 100 de uma taxa mede a
  variação da própria taxa. Os valores originais aparecem num painel por série, cada um com o próprio eixo.
- Variação em 12 meses do IPCA índice: `IPCA(t) / IPCA(t−12) − 1`.
- **Correlação defasada** entre mudanças mensais (Δ), para k de −12 a +12 meses. Diferenciar evita correlação espúria de séries com tendência.
- Ranking horizontal por UF, sem mapa (evita geojson pesado).

## Resultados
Este projeto é exploratório: não há modelo a treinar nem métrica de desempenho. O app mostra:
- Selic meta, IPCA em 12 meses e dólar no último dado coletado (os valores vêm do snapshot, não são digitados).
- Séries, índice base 100 e correlação defasada entre dois indicadores escolhidos pelo visitante.
- Ranking das UFs por desocupação (trimestre) e por população (ano).

Para a correlação defasada, o maior valor absoluto depende do par escolhido; a página mostra o valor e o número de pares usados.

## Como rodar
```bash
make data-eda-dados-br    # coleta as APIs e grava o snapshot em data/ (alguns segundos; as janelas do SGS fazem poucas requisições)
make test
```
Não há alvo `train-eda-dados-br`: o projeto não tem etapa de treino.

## Limitações
- Dados de 2010 em diante; correlação com 12 meses de defasagem e poucos pontos é instável.
- Correlação não é causalidade; mudanças mensais correlacionadas podem refletir sazonalidade ou choques comuns.
- Taxa de desocupação do SGS (24369) e a da SIDRA (4099) não são a mesma série: não comparar diretamente.
- O snapshot congela a coleta de 2026-10-14. Para atualizar: `make data-eda-dados-br` com `--refresh` (ou apagar `data/raw/`).
- Base dos Dados não integrada (ver acima).
