# Segmentação de clientes com RFM e KMeans

## Problema
Nem todo cliente vale o mesmo esforço. A segmentação separa a base em grupos com comportamento parecido para orientar
ação: quem proteger, quem reativar, quem deixar em paz. O método clássico é o **RFM** (Recência, Frequência, Monetário).

## Dados
- **Online Retail II** (UCI Machine Learning Repository), CC BY 4.0. Varejo online britânico, 2009-2011, 1.067.371 linhas brutas.
- **Substituição consciente:** o projeto inicialmente pensado usaria o Olist, mas ele exige credencial do Kaggle e já é usado
  no repositório de Engenharia de Dados. O Online Retail II é público sem login.
- Download direto de `https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip`, guardado em `data/raw/` (gitignored).
- Limpeza: 243.007 linhas sem `CustomerID` removidas; 18.744 faturas de cancelamento (prefixo `C`) usadas só para a taxa de
  devolução; 71 com quantidade ou preço ≤ 0; 2.662 códigos que não são produtos (frete, taxas, ajustes). Restam 802.887 linhas
  de 5.853 clientes.
- **Versionado:** só o agregado RFM por cliente (`data/rfm.parquet`, 5.853 linhas), com ID pseudonimizado por hash SHA-256
  com sal fixo. Pseudonimização, não anonimização forte.

## Abordagem
1. RFM por cliente: recência em dias até o dia seguinte à última compra; frequência = faturas distintas; monetário = soma de quantidade × preço.
2. `log(1 + x)` nas três métricas e padronização (as distribuições são muito assimétricas).
3. k entre 2 e 10 por silhouette e cotovelo; para uso de negócio, k limitado a 3–8 (escolhido: 4).
4. Nomes de segmento por **regras sobre os centróides** (z-scores de R, F e M), não atribuídos à mão.
5. Comparação com a segmentação por quintis RFM (escores de 1 a 5 e regras simples).

## Resultados
Números de `data/metrics.json`, gerado por `make train-customer-segmentation`.

- k = 4, com silhouette de 0,362 (contra 0,347 em k = 3 e 0,340 em k = 5; o maior entre 3 e 8 é k = 4).
- Segmentos (KMeans):

| Segmento | Clientes | % clientes | Recência mediana (dias) | Frequência mediana | % receita |
|---|---|---|---|---|---|
| Campeões | 1.107 | 18,9% | 15 | 13 | 71,6% |
| Intermediários | 1.471 | 25,1% | 157 | 5 | 18,4% |
| Recentes (baixa frequência) | 1.220 | 20,8% | 23 | 3 | 5,9% |
| Inativos | 2.055 | 35,1% | 400 | 1 | 4,1% |

- Quintis RFM: 6 segmentos; silhouette 0,173 (contra 0,362 do KMeans). Concordância entre as duas segmentações (ARI): 0,43.
- Os 1.107 campeões (19% da base) respondem por 72% da receita.

## Como rodar
```bash
make data-customer-segmentation   # baixa o zip, converte o xlsx para parquet em data/raw/ (~1 min)
make train-customer-segmentation  # limpeza, RFM, escolha de k, KMeans e métricas
```

## Limitações
- Período de 2 anos e varejo britânico de presentes; generalizar para outro setor exige reavaliar as faixas das regras de nomeação.
- As ações sugeridas no app são **regras de demonstração**, não validadas com dados de campanha. O próximo passo natural é um teste A/B por segmento.
- Monetário em libras, sem conversão. Não há custo de aquisição nem margem.
- O `CustomerID` nulo pode esconder clientes reais; as 243 mil linhas descartadas são ~23% do total, o que merece checagem.
- A pseudonimização é fraca: os IDs são inteiros de poucos dígitos e o sal é público, então qualquer pessoa pode recalcular o hash de cada ID possível e reverter a tabela. Não tratar como anonimização.
