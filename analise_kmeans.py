# Alunos: José Guilherme Ferreira Sobrinho e Bruno Henrique Frota Sobral

from __future__ import annotations

import os
from pathlib import Path

BASE_PROJETO = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_PROJETO / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


SEMENTE = 42
VALORES_K = range(2, 11)
PASTA_RESULTADOS = BASE_PROJETO / "resultados"


def carregar_dados() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    wine = load_wine(as_frame=True)
    dados = wine.data.copy()
    classes = wine.target.copy()
    nomes_classes = [f"cultivar_{numero}" for numero in range(1, 4)]
    return dados, classes, nomes_classes


def avaliar_valores_de_k(dados_padronizados) -> pd.DataFrame:
    """Calcula inércia e silhueta para cada k candidato."""
    linhas = []
    for k in VALORES_K:
        modelo = KMeans(n_clusters=k, random_state=SEMENTE, n_init=20)
        grupos = modelo.fit_predict(dados_padronizados)
        linhas.append(
            {
                "k": k,
                "coeficiente_silhueta": silhouette_score(
                    dados_padronizados, grupos
                ),
                "inercia": modelo.inertia_,
            }
        )
    return pd.DataFrame(linhas)


def salvar_grafico_silhueta(avaliacao: pd.DataFrame, melhor_k: int) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(
        avaliacao["k"],
        avaliacao["coeficiente_silhueta"],
        marker="o",
        linewidth=2,
        color="#2563eb",
    )
    melhor = avaliacao.loc[avaliacao["k"] == melhor_k].iloc[0]
    plt.scatter(
        melhor_k,
        melhor["coeficiente_silhueta"],
        s=130,
        color="#dc2626",
        zorder=3,
        label=f"Melhor k = {melhor_k}",
    )
    plt.xticks(list(VALORES_K))
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Coeficiente de silhueta")
    plt.title("Escolha de k pelo coeficiente de silhueta")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PASTA_RESULTADOS / "01_silhueta_por_k.png", dpi=180)
    plt.close()


def salvar_grafico_pca(
    dados_padronizados, grupos, variancia_explicada
) -> pd.DataFrame:
    pca = PCA(n_components=2)
    coordenadas = pca.fit_transform(dados_padronizados)
    projecao = pd.DataFrame(coordenadas, columns=["CP1", "CP2"])
    projecao["cluster"] = grupos

    plt.figure(figsize=(8, 6))
    for cluster in sorted(projecao["cluster"].unique()):
        subconjunto = projecao[projecao["cluster"] == cluster]
        plt.scatter(
            subconjunto["CP1"],
            subconjunto["CP2"],
            s=45,
            alpha=0.8,
            label=f"Cluster {cluster}",
        )
    plt.xlabel(f"Componente principal 1 ({variancia_explicada[0]:.1%})")
    plt.ylabel(f"Componente principal 2 ({variancia_explicada[1]:.1%})")
    plt.title("Clusters K-Means projetados em duas dimensões (PCA)")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PASTA_RESULTADOS / "02_clusters_pca.png", dpi=180)
    plt.close()
    return projecao


def salvar_grafico_perfis(
    dados_padronizados: pd.DataFrame, grupos
) -> pd.DataFrame:
    padronizados = dados_padronizados.copy()
    padronizados["cluster"] = grupos
    perfis = padronizados.groupby("cluster").mean()

    plt.figure(figsize=(13, 6))
    x = range(len(perfis.columns))
    for cluster, linha in perfis.iterrows():
        plt.plot(x, linha, marker="o", linewidth=2, label=f"Cluster {cluster}")
    plt.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    plt.xticks(x, perfis.columns, rotation=55, ha="right")
    plt.ylabel("Média padronizada (z-score)")
    plt.title("Perfil médio das variáveis em cada cluster")
    plt.grid(axis="y", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PASTA_RESULTADOS / "03_perfis_clusters.png", dpi=180)
    plt.close()
    return perfis


def main() -> None:
    PASTA_RESULTADOS.mkdir(exist_ok=True)
    dados, classes_reais, nomes_classes = carregar_dados()

    escalonador = StandardScaler()
    matriz_padronizada = escalonador.fit_transform(dados)
    dados_padronizados = pd.DataFrame(
        matriz_padronizada, columns=dados.columns, index=dados.index
    )

    avaliacao = avaliar_valores_de_k(matriz_padronizada)
    melhor_linha = avaliacao.loc[avaliacao["coeficiente_silhueta"].idxmax()]
    melhor_k = int(melhor_linha["k"])

    modelo_final = KMeans(
        n_clusters=melhor_k, random_state=SEMENTE, n_init=20
    )
    grupos = modelo_final.fit_predict(matriz_padronizada)

    # PCA é usado somente para visualizar os 13 atributos em um plano 2D.
    pca_teste = PCA(n_components=2).fit(matriz_padronizada)
    salvar_grafico_silhueta(avaliacao, melhor_k)
    projecao = salvar_grafico_pca(
        matriz_padronizada, grupos, pca_teste.explained_variance_ratio_
    )
    perfis = salvar_grafico_perfis(dados_padronizados, grupos)

    dados_saida = dados.copy()
    dados_saida["classe_original"] = [nomes_classes[i] for i in classes_reais]
    dados_saida["cluster_kmeans"] = grupos
    dados_saida["CP1"] = projecao["CP1"]
    dados_saida["CP2"] = projecao["CP2"]

    cruzamento = pd.crosstab(
        pd.Series(grupos, name="cluster_kmeans"),
        pd.Series(
            [nomes_classes[i] for i in classes_reais], name="classe_original"
        ),
    )
    ari = adjusted_rand_score(classes_reais, grupos)

    avaliacao.to_csv(
        PASTA_RESULTADOS / "metricas_por_k.csv", index=False
    )
    dados_saida.to_csv(
        PASTA_RESULTADOS / "wine_com_clusters.csv", index=False
    )
    perfis.to_csv(PASTA_RESULTADOS / "perfis_clusters.csv")
    cruzamento.to_csv(PASTA_RESULTADOS / "clusters_vs_classes.csv")

    titulo = "RESULTADO DA ANÁLISE K-MEANS — BASE WINE (UCI)"
    resumo = (
        f"{titulo}\n"
        + "=" * len(titulo)
        + "\n"
        f"Amostras: {len(dados)}\n"
        f"Variáveis utilizadas: {dados.shape[1]}\n"
        f"Faixa de k avaliada: {min(VALORES_K)} a {max(VALORES_K)}\n"
        f"Melhor k: {melhor_k}\n"
        "Coeficiente de silhueta do melhor modelo: "
        f"{melhor_linha['coeficiente_silhueta']:.4f}\n"
        f"Adjusted Rand Index (apenas interpretação): {ari:.4f}\n\n"
        "Quantidade de amostras por cluster:\n"
        f"{pd.Series(grupos).value_counts().sort_index().to_string()}\n\n"
        "Cruzamento com as classes originais (não usadas no treino):\n"
        f"{cruzamento.to_string()}\n"
    )
    (PASTA_RESULTADOS / "resumo.txt").write_text(resumo, encoding="utf-8")
    print(resumo)
    print(f"Arquivos salvos em: {PASTA_RESULTADOS}")


if __name__ == "__main__":
    main()

