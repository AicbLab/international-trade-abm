"""
语义多样性 + 产业集群分析
==========================
1. 语义多样性度量：考虑产品间语义距离的有效产品数
2. 语义产业集群：对1242个产品做聚类，分析国家升级路径
3. 封锁溢出可视化：哪些产品受间接影响最大
"""
import sys, json
import numpy as np
from pathlib import Path
from scipy import sparse
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from research import ResearchContext

EXPERIMENTS = ROOT / 'experiments'


def load_semantic_phi():
    tfidf_emb = sparse.load_npz(EXPERIMENTS / 'product_embeddings_tfidf.npz')
    sim = cosine_similarity(tfidf_emb)
    np.fill_diagonal(sim, 0)
    k = 30
    rows, cols, vals = [], [], []
    for i in range(sim.shape[0]):
        top_k = np.argpartition(sim[i], -k)[-k:]
        for j in top_k:
            if sim[i, j] > 0:
                rows.append(i); cols.append(j); vals.append(sim[i, j])
    phi = sparse.csr_matrix((vals, (rows, cols)), shape=sim.shape)
    phi = (phi + phi.T) / 2
    return np.clip(phi.toarray(), 0, 1)


def semantic_diversity(caps, phi_sem):
    """语义多样性：考虑产品间语义距离的有效产品数
    
    对于每个国家的能力向量 caps (P,)，计算：
    sem_div = sum_p caps[p] * (1 - mean_{q!=p} phi_sem[p,q] * caps[q])
    
    值越高表示产品组合越"语义分散"（复杂度更高）
    """
    p = len(caps)
    active = np.flatnonzero(caps)
    if len(active) <= 1:
        return float(len(active))
    
    # 每个活跃产品与其余活跃产品的平均语义相似度
    sub_phi = phi_sem[np.ix_(active, active)]
    np.fill_diagonal(sub_phi, 0)
    mean_sim = sub_phi.mean(axis=1)  # 每个活跃产品的平均语义相似度
    
    # 语义多样性 = 活跃产品数 × (1 - 平均内部相似度)
    n_active = len(active)
    avg_internal_sim = mean_sim.mean()
    sem_div = n_active * (1 - avg_internal_sim)
    return float(sem_div)


def main():
    print("=" * 70)
    print("Semantic Diversity + Industry Cluster Analysis")
    print("=" * 70)

    ctx = ResearchContext()
    products = ctx.products
    actual = ctx.actual
    phi_sem = load_semantic_phi()

    # ================================================================
    # Part 1: 语义多样性 vs 传统多样性
    # ================================================================
    print("\n--- Part 1: Semantic Diversity vs Traditional Diversity ---")
    print(f"\n  {'Country':>4}  {'Trad_2000':>9}  {'Trad_2023':>9}  {'Sem_2000':>9}  {'Sem_2023':>9}  {'SemRatio':>8}")

    sem_div_data = {}
    for i, c in enumerate(ctx.countries):
        caps_2000 = actual[0, i].astype(float)
        caps_2023 = actual[-1, i].astype(float)
        trad_2000 = int(caps_2000.sum())
        trad_2023 = int(caps_2023.sum())
        sem_2000 = semantic_diversity(caps_2000, phi_sem)
        sem_2023 = semantic_diversity(caps_2023, phi_sem)
        ratio = sem_2023 / sem_2000 if sem_2000 > 0 else 0
        sem_div_data[c] = {
            'trad_2000': trad_2000, 'trad_2023': trad_2023,
            'sem_2000': round(sem_2000, 2), 'sem_2023': round(sem_2023, 2),
            'sem_ratio': round(ratio, 3),
        }
        print(f"  {c:>4}  {trad_2000:>9}  {trad_2023:>9}  {sem_2000:>9.2f}  {sem_2023:>9.2f}  {ratio:>8.3f}")

    # ================================================================
    # Part 2: 语义产业集群
    # ================================================================
    print("\n--- Part 2: Semantic Industry Clusters ---")
    
    # 层次聚类
    # 用 1 - phi_sem 作为距离
    dist = 1 - phi_sem
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2  # 确保对称
    dist = np.clip(dist, 0, 1)
    
    # 转为 condensed form
    n = len(products)
    condensed = []
    for i in range(n):
        for j in range(i+1, n):
            condensed.append(dist[i, j])
    condensed = np.array(condensed)
    
    Z = linkage(condensed, method='ward')
    
    # 分成 10 个集群
    labels = fcluster(Z, t=10, criterion='maxclust')
    
    # 统计每个集群的产品数和代表性产品
    cluster_info = {}
    for k in range(1, 11):
        members = np.where(labels == k)[0]
        member_products = [products[i] for i in members]
        
        # 找集群中心（与集群内其他产品平均proximity最高的产品）
        sub_phi = phi_sem[np.ix_(members, members)]
        center_idx = members[sub_phi.mean(axis=1).argmax()]
        center_product = products[center_idx]
        
        # 找集群内所有产品描述的关键词（用TF-IDF top词）
        cluster_info[k] = {
            'n_products': len(members),
            'center': center_product,
            'sample_products': member_products[:8],
        }
        print(f"  Cluster {k:2d}: {len(members):4d} products, center={center_product}, "
              f"sample={member_products[:5]}")

    # 国家在集群间的分布
    print(f"\n  Country cluster distribution (2023):")
    print(f"  {'Country':>4}" + "".join([f"  {'C'+str(k):>5}" for k in range(1, 11)]))
    
    cluster_dist = {}
    for i, c in enumerate(ctx.countries):
        caps = actual[-1, i].astype(bool)
        dist_row = {}
        parts = [f"  {c:>4}"]
        for k in range(1, 11):
            members = np.where(labels == k)[0]
            n_in_cluster = int(caps[members].sum())
            dist_row[f'C{k}'] = n_in_cluster
            parts.append(f"  {n_in_cluster:>5}")
        print("".join(parts))
        cluster_dist[c] = dist_row

    # ================================================================
    # Part 3: 封锁溢出分析
    # ================================================================
    print("\n--- Part 3: Blockade Spillover Analysis ---")
    
    semi_targets = [p for p in ['8542', '8541', '8534', '8536', '8537'] if p in products]
    broad_targets = [p for p in [
        '8471', '8473', '8517', '8523', '8525', '8526', '8528', '8529',
        '8534', '8536', '8537', '8541', '8542',
        '8501', '8502', '8503', '8504',
    ] if p in products]
    
    target_indices = set()
    for p in broad_targets:
        if p in products:
            target_indices.add(products.index(p))
    
    # 每个产品受封锁溢出影响的程度
    spillover_scores = np.zeros(len(products))
    for ti in target_indices:
        spillover_scores = np.maximum(spillover_scores, phi_sem[:, ti])
    
    # 排除目标产品本身
    for ti in target_indices:
        spillover_scores[ti] = 0
    
    # Top-20 受影响最大的非目标产品
    top20_idx = np.argsort(spillover_scores)[::-1][:20]
    print(f"\n  Top-20 products most affected by blockade spillover:")
    print(f"  (broad targets: {broad_targets})")
    for rank, idx in enumerate(top20_idx):
        print(f"    {rank+1:2d}. {products[idx]}  spillover={spillover_scores[idx]:.3f}  "
              f"cluster={labels[idx]}")

    # 按集群统计溢出影响
    print(f"\n  Spillover by cluster:")
    for k in range(1, 11):
        members = np.where(labels == k)[0]
        avg_spill = spillover_scores[members].mean()
        max_spill = spillover_scores[members].max()
        if avg_spill > 0.05:
            print(f"    Cluster {k:2d} ({cluster_info[k]['n_products']} products, "
                  f"center={cluster_info['center']}): "
                  f"avg_spillover={avg_spill:.3f}, max={max_spill:.3f}")

    # ================================================================
    # Save
    # ================================================================
    analysis = {
        'semantic_diversity': sem_div_data,
        'clusters': {str(k): v for k, v in cluster_info.items()},
        'cluster_distribution': cluster_dist,
        'blockade_spillover_top20': [
            {'product': products[idx], 'score': float(spillover_scores[idx]),
             'cluster': int(labels[idx])}
            for idx in top20_idx
        ],
    }
    (EXPERIMENTS / 'semantic_analysis.json').write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\nSaved: experiments/semantic_analysis.json")


if __name__ == '__main__':
    main()
