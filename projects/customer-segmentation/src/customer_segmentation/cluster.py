"""Escala, escolha de k, KMeans, perfil e nomes de segmento derivados de regras sobre os centróides."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

SEED = 42
RFM_COLS = ["recency_days", "frequency", "monetary"]


def transform(rfm: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """log1p nas três métricas (distribuições muito assimétricas) + padronização."""
    scaler = StandardScaler()
    return scaler.fit_transform(np.log1p(rfm[RFM_COLS].to_numpy(dtype=float))), scaler


def select_k(Z: np.ndarray, ks: range = range(2, 11), sample_size: int | None = 5000) -> pd.DataFrame:
    """Inércia (cotovelo), silhouette e Davies-Bouldin para cada k."""
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
        rows.append(
            {
                "k": k,
                "inertia": float(km.inertia_),
                "silhouette": float(silhouette_score(Z, km.labels_, sample_size=sample_size, random_state=SEED)),
                "davies_bouldin": float(davies_bouldin_score(Z, km.labels_)),
            }
        )
    return pd.DataFrame(rows)


def fit_kmeans(Z: np.ndarray, k: int) -> KMeans:
    return KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)


def best_k(table: pd.DataFrame, k_min: int = 3, k_max: int = 8) -> int:
    """k de maior silhouette dentro de [k_min, k_max] (menos de 3 segmentos não orienta ação de negócio)."""
    sub = table[(table["k"] >= k_min) & (table["k"] <= k_max)]
    return int(sub.loc[sub["silhouette"].idxmax(), "k"])


def name_segments(centroids_z: pd.DataFrame) -> dict[int, str]:
    """Nomes de segmento por regras sobre os centróides padronizados (colunas = RFM_COLS, índice = cluster).

    Em z-score: recência alta = cliente parado há mais tempo, então 'recente' = -z(recency).
    Regras, em ordem (primeira que casar):
      Campeões          recente (rec >= 0.3), frequente (f >= 0.5) e alto gasto (m >= 0.5)
      Em risco (valor)  não recente (rec <= -0.3) mas com f >= 0.3 e m >= 0.3
      Leais             f >= 0.3 e m >= 0.3
      Recentes (baixa frequência)  recente (rec >= 0.3) e f < 0.3
      Inativos          não recente (rec <= -0.3) e baixa frequência (f <= -0.3)
      Intermediários    demais
    Nomes repetidos recebem sufixo I, II... em ordem decrescente de gasto.
    """
    rec = -centroids_z["recency_days"]
    f, m = centroids_z["frequency"], centroids_z["monetary"]
    names: dict[int, str] = {}
    for c in centroids_z.index:
        if rec[c] >= 0.3 and f[c] >= 0.5 and m[c] >= 0.5:
            names[c] = "Campeões"
        elif rec[c] <= -0.3 and f[c] >= 0.3 and m[c] >= 0.3:
            names[c] = "Em risco (alto valor)"
        elif f[c] >= 0.3 and m[c] >= 0.3:
            names[c] = "Leais"
        elif rec[c] >= 0.3 and f[c] < 0.3:
            names[c] = "Recentes (baixa frequência)"
        elif rec[c] <= -0.3 and f[c] <= -0.3:
            names[c] = "Inativos"
        else:
            names[c] = "Intermediários"
    by_name: dict[str, list[int]] = {}
    for c, n in names.items():
        by_name.setdefault(n, []).append(c)
    romans = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]
    for n, clusters in by_name.items():
        if len(clusters) > 1:
            for i, c in enumerate(sorted(clusters, key=lambda c: -m[c])):
                names[c] = f"{n} {romans[i]}"
    return names


def profile(rfm: pd.DataFrame, labels: np.ndarray, names: dict[int, str] | None = None) -> pd.DataFrame:
    """Perfil por cluster: tamanho, medianas de RFM, médias e participação na receita."""
    df = rfm.assign(cluster=labels)
    g = df.groupby("cluster")
    out = pd.DataFrame(
        {
            "clientes": g.size(),
            "pct_clientes": g.size() / len(df),
            "recencia_mediana_dias": g["recency_days"].median(),
            "frequencia_mediana": g["frequency"].median(),
            "monetario_mediano": g["monetary"].median(),
            "ticket_medio": g["avg_ticket"].median(),
            "taxa_cancelamento_media": g["cancel_rate"].mean(),
            "receita_total": g["monetary"].sum(),
        }
    )
    out["pct_receita"] = out["receita_total"] / out["receita_total"].sum()
    if names is not None:
        out.insert(0, "segmento", pd.Series(names))
    return out.reset_index()


# ----- segmentação clássica por quintis RFM (para comparar) -----
def quintile_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """Escores 1-5 de R, F e M por quintil e segmento por regra simples (R e F)."""
    out = pd.DataFrame(index=rfm.index)
    # recência: menor = melhor -> escore maior; rank evita bordas repetidas em F (muitos clientes com 1 compra)
    out["R"] = pd.qcut(rfm["recency_days"].rank(method="first", ascending=False), 5, labels=False).astype(int) + 1
    out["F"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=False).astype(int) + 1
    out["M"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=False).astype(int) + 1

    def rule(r: int, f: int) -> str:
        if r >= 4 and f >= 4:
            return "Campeões"
        if r >= 3 and f >= 3:
            return "Leais"
        if r >= 4 and f <= 2:
            return "Recentes (baixa frequência)"
        if r <= 2 and f >= 3:
            return "Em risco"
        if r <= 2 and f <= 2:
            return "Hibernando"
        return "Atenção"

    out["segmento_quintil"] = [rule(r, f) for r, f in zip(out["R"], out["F"])]
    return out


def compare_segmentations(Z: np.ndarray, kmeans_labels: np.ndarray, quintile_labels: pd.Series) -> dict[str, float]:
    codes = pd.factorize(quintile_labels)[0]
    return {
        "ari_kmeans_vs_quintis": float(adjusted_rand_score(kmeans_labels, codes)),
        "silhouette_kmeans": float(silhouette_score(Z, kmeans_labels, sample_size=5000, random_state=SEED)),
        "silhouette_quintis": float(silhouette_score(Z, codes, sample_size=5000, random_state=SEED)),
        "n_segmentos_quintis": int(len(np.unique(codes))),
    }
