# Agrupamento de vinhos com K-Means

Este projeto resolve um problema de **aprendizado não supervisionado**: agrupar
vinhos com composições químicas semelhantes. O número de grupos não é imposto
pelas classes conhecidas da base; ele é escolhido pelo maior **coeficiente de
silhueta** entre `k = 2` e `k = 10`.

## Vídeo de apresentação

A explicação do projeto tem duração de **4 minutos e 39 segundos**.

[▶️ Assistir à apresentação no YouTube](COLE_AQUI_O_LINK_DO_VIDEO)

> Depois de publicar o vídeo, substitua `COLE_AQUI_O_LINK_DO_VIDEO` pela URL do
> YouTube. O GitHub não permite incorporar o player diretamente no README, mas o
> link acima pode apontar para o vídeo ou ser transformado em uma miniatura
> clicável.

## Base de dados utilizada

Foi utilizada a base **Wine**, de Stefan Aeberhard e M. Forina, disponível no
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/109/wine)
(DOI: [10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J)). Ela contém **178
amostras**, sem valores ausentes, descritas por **13 características químicas**:
teor alcoólico, ácido málico, cinzas, alcalinidade das cinzas, magnésio, fenóis
totais, flavonoides, fenóis não flavonoides, proantocianinas, intensidade da cor,
matiz, OD280/OD315 e prolina.

Os vinhos vieram da mesma região da Itália, mas de três cultivares diferentes.
As classes de cultivar existem na base, porém **não entram no treinamento do
K-Means**. Elas são consultadas no final somente para ajudar a interpretar o
resultado e calcular o Adjusted Rand Index (ARI).

## Como executar no Windows (PowerShell)

As dependências ficam isoladas em `.venv`; nada é instalado globalmente.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe analise_kmeans.py
```

Os resultados são gravados em `resultados/`:

- `metricas_por_k.csv`: silhueta e inércia para cada valor de `k`;
- `01_silhueta_por_k.png`: justificativa visual da escolha de `k`;
- `02_clusters_pca.png`: projeção 2D dos clusters por PCA;
- `03_perfis_clusters.png`: perfil químico médio de cada grupo;
- `wine_com_clusters.csv`: base completa com o cluster atribuído;
- `clusters_vs_classes.csv`: comparação posterior com os cultivares;
- `resumo.txt`: principais números da execução.

## Método

1. Carregamento das 178 observações e dos 13 atributos numéricos.
2. Padronização com `StandardScaler` (média 0 e desvio-padrão 1). Isso é
   essencial porque o K-Means usa distância euclidiana e as variáveis estão em
   escalas muito diferentes.
3. Treinamento do K-Means para cada `k` de 2 a 10, com semente 42 e 20
   inicializações para obter um resultado estável.
4. Cálculo do coeficiente de silhueta de cada solução. A métrica varia de -1 a
   1: quanto maior, mais compactos e mais separados estão os grupos.
5. Seleção automática do `k` com maior silhueta e treinamento do modelo final.
6. PCA apenas para projetar os 13 atributos em duas dimensões no gráfico; o
   agrupamento é realizado com os 13 atributos padronizados.

## Resultado obtido

O maior coeficiente de silhueta foi **0,2849**, para **k = 3**. O modelo final
formou clusters com 65, 51 e 62 amostras. O segundo melhor resultado foi
`k = 2`, com silhueta 0,2593; todos os valores de `k` de 4 a 10 também ficaram
abaixo do resultado de `k = 3`.

Os perfis químicos médios ajudam a interpretar os grupos:

- **Cluster 0:** menores valores médios de álcool, intensidade de cor e prolina;
- **Cluster 1:** menores fenóis, flavonoides, matiz e OD280/OD315, com maiores
  ácido málico e intensidade de cor;
- **Cluster 2:** maiores valores médios de álcool, fenóis, flavonoides,
  OD280/OD315 e prolina.

Como verificação posterior, o cruzamento com os cultivares originais produziu
um **Adjusted Rand Index de 0,8975**, indicando forte concordância. Essa
comparação não participou da escolha de `k` nem do treinamento.

## Observação importante

Os números dos clusters (0, 1, 2...) são apenas identificadores e não têm uma
ordem natural. Além disso, K-Means procura grupos aproximadamente esféricos e
usa médias, portanto é sensível à escala e a valores extremos. A silhueta é uma
medida interna de qualidade: ela orienta a escolha de `k`, mas não prova que os
grupos sejam a única interpretação possível do problema.

