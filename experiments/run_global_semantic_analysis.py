"""
全球规模语义分析（231 国）
=========================
1. 语义多样性度量：区分"量增"与"质变"
2. 语义产业集群：Ward 层次聚类
3. 国家专业化模式分析
4. 贸易战溢出效应可视化
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from scipy.cluster.hierarchy import ward, fcluster
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from research import ResearchContext, BASE_YEAR, END_YEAR
from pipeline import PIPELINE_VERSION

EXPERIMENTS = ROOT / 'experiments'


def load_semantic_phi():
    """加载 TF-IDF 语义 proximity"""
    tfidf_emb = sparse.load_npz(EXPERIMENTS / 'product_embeddings_tfidf.npz')
    sim = cosine_similarity(tfidf_emb)
    np.fill_diagonal(sim, 0)
    return np.clip(sim, 0, 1)


def compute_semantic_diversity(capabilities, sem_phi):
    """语义多样性 = 活跃产品数 × (1 - 平均内部相似度)
    高值 = 产品多且语义分散（质变）
    低值 = 产品少或语义集中（量增）
    """
    active = np.flatnonzero(capabilities)
    n = len(active)
    if n <= 1:
        return 0.0, n
    # 平均内部相似度
    sub_sim = sem_phi[np.ix_(active, active)]
    avg_internal = (sub_sim.sum() - np.trace(sub_sim)) / (n * (n - 1))
    sem_div = n * (1 - avg_internal)
    return float(sem_div), n


def main():
    print("=" * 70)
    print("Global Semantic Analysis (231 countries)")
    print("=" * 70)

    ctx = ResearchContext()
    sem_phi = load_semantic_phi()
    countries = ctx.countries
    products = ctx.products
    n_c = len(countries)

    # 实际数据（2000 和 2023）
    actual_2000 = ctx.actual[0]   # (C, P)
    actual_2023 = ctx.actual[-1]  # (C, P)

    # ================================================================
    # Part 1: 语义多样性分析
    # ================================================================
    print("\n" + "=" * 70)
    print("Part 1: Semantic Diversity (2000 vs 2023)")
    print("=" * 70)

    div_results = []
    for i, c in enumerate(countries):
        caps_00 = actual_2000[i].astype(bool)
        caps_23 = actual_2023[i].astype(bool)
        sd_00, n_00 = compute_semantic_diversity(caps_00, sem_phi)
        sd_23, n_23 = compute_semantic_diversity(caps_23, sem_phi)
        div_results.append({
            'country': c,
            'n_2000': n_00, 'n_2023': n_23,
            'sem_div_2000': sd_00, 'sem_div_2023': sd_23,
            'n_change': n_23 - n_00,
            'sd_change': sd_23 - sd_00,
            'growth_type': 'quality' if sd_23 > sd_00 and n_23 >= n_00
                          else 'loss' if n_23 < n_00
                          else 'quantitative' if n_23 > n_00 and sd_23 <= sd_00
                          else 'stable',
        })

    div_df = pd.DataFrame(div_results)

    # Top 10 by semantic diversity growth
    top_growth = div_df.nlargest(10, 'sd_change')
    print("\n  Top 10 semantic diversity growth:")
    for _, r in top_growth.iterrows():
        print(f"    {r['country']:5s}: div {r['sem_div_2000']:.1f} → {r['sem_div_2023']:.1f} "
              f"(Δ={r['sd_change']:+.1f}, n={r['n_2000']}→{r['n_2023']}, type={r['growth_type']})")

    # Top 10 by semantic diversity loss
    top_loss = div_df.nsmallest(10, 'sd_change')
    print("\n  Top 10 semantic diversity loss:")
    for _, r in top_loss.iterrows():
        print(f"    {r['country']:5s}: div {r['sem_div_2000']:.1f} → {r['sem_div_2023']:.1f} "
              f"(Δ={r['sd_change']:+.1f}, n={r['n_2000']}→{r['n_2023']}, type={r['growth_type']})")

    # Growth type distribution
    type_counts = div_df['growth_type'].value_counts()
    print(f"\n  Growth type distribution:")
    for t, cnt in type_counts.items():
        print(f"    {t:15s}: {cnt:3d} ({cnt/n_c*100:.1f}%)")

    # ================================================================
    # Part 2: 语义产业集群
    # ================================================================
    print("\n" + "=" * 70)
    print("Part 2: Semantic Industry Clusters")
    print("=" * 70)

    # Ward 层次聚类
    # 使用语义距离矩阵
    dist = 1 - sem_phi
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    # 转为 condensed form
    from scipy.spatial.distance import squareform
    condensed = squareform(dist)
    Z = ward(condensed)

    # 10 个集群
    labels = fcluster(Z, t=10, criterion='maxclust')
    print(f"  Products: {len(products)}, Clusters: {max(labels)}")

    cluster_dist = {}
    for k in range(1, max(labels) + 1):
        members = [products[i] for i in range(len(products)) if labels[i] == k]
        cluster_dist[k] = len(members)

    for k in sorted(cluster_dist, key=cluster_dist.get, reverse=True):
        print(f"    Cluster {k:2d}: {cluster_dist[k]:4d} products")

    # 每个集群的国家分布（哪些国家在该集群有 RCA>1）
    print(f"\n  Top countries per cluster (by RCA>1 count in 2023):")
    for k in sorted(cluster_dist, key=cluster_dist.get, reverse=True)[:5]:
        members_idx = [i for i in range(len(products)) if labels[i] == k]
        # 哪些国家在这些产品上有 RCA>1
        country_counts = actual_2023[:, members_idx].sum(axis=1)
        top_countries = np.argsort(-country_counts)[:5]
        member_names = [products[i] for i in members_idx[:5]]
        print(f"    Cluster {k} ({cluster_dist[k]} products, e.g. {member_names}):")
        for ci in top_countries:
            print(f"      {countries[ci]:5s}: {country_counts[ci]} RCAs")

    # ================================================================
    # Part 3: 国家语义特征分析
    # ================================================================
    print("\n" + "=" * 70)
    print("Part 3: Country Semantic Profiles")
    print("=" * 70)

    # 每个国家的"语义中心度"（其能力产品在语义空间中的平均连接度）
    profiles = []
    for i, c in enumerate(countries):
        caps = actual_2023[i].astype(bool)
        if caps.sum() == 0:
            continue
        # 语义中心度：能力产品的平均语义 proximity
        avg_sem = sem_phi[caps][:, caps].mean()
        # 语义独特性：能力产品与非能力产品的平均距离
        non_caps = ~caps
        if non_caps.any():
            sem_unique = 1 - sem_phi[caps][:, non_caps].mean()
        else:
            sem_unique = 0
        profiles.append({
            'country': c,
            'n_products': caps.sum(),
            'avg_sem_proximity': avg_sem,
            'sem_uniqueness': sem_unique,
        })

    prof_df = pd.DataFrame(profiles).sort_values('n_products', ascending=False)

    print("\n  Top 15 countries by product count (2023):")
    print(f"  {'Country':>7} {'N products':>10} {'Avg sem prox':>12} {'Uniqueness':>12}")
    for _, r in prof_df.head(15).iterrows():
        print(f"  {r['country']:>7} {r['n_products']:>10} {r['avg_sem_proximity']:>12.3f} {r['sem_uniqueness']:>12.3f}")

    # ================================================================
    # Save
    # ================================================================
    summary = {
        'pipeline_version': PIPELINE_VERSION,
        'n_countries': n_c,
        'diversity': {
            'top_growth': top_growth[['country', 'sem_div_2000', 'sem_div_2023', 'sd_change', 'growth_type']].to_dict('records'),
            'top_loss': top_loss[['country', 'sem_div_2000', 'sem_div_2023', 'sd_change', 'growth_type']].to_dict('records'),
            'type_distribution': type_counts.to_dict(),
        },
        'clusters': {
            'n_clusters': int(max(labels)),
            'cluster_sizes': {str(k): v for k, v in sorted(cluster_dist.items(), key=lambda x: -x[1])},
        },
        'profiles': prof_df.head(30).to_dict('records'),
    }

    (EXPERIMENTS / 'global_semantic_analysis.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\nSaved: experiments/global_semantic_analysis.json")


if __name__ == '__main__':
    main()
