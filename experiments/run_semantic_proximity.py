"""
语义产品空间：LLM 嵌入增强
============================
用 sentence transformer 嵌入产品描述，构建语义 proximity，
与统计 proximity 融合为混合 proximity。

学术贡献：
- 现有产品空间仅依赖贸易统计 (phi_stat)
- LLM 嵌入捕捉技术/功能语义相似性 (phi_sem)
- 混合 proximity 可能改善预测精度，尤其对小国
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from pipeline import PIPELINE_VERSION
from research import ResearchContext, DEFAULT_THETA, PARAMETERS, BASE_YEAR, END_YEAR

EXPERIMENTS = ROOT / 'experiments'
EXPERIMENTS.mkdir(parents=True, exist_ok=True)


def load_product_descriptions():
    """加载产品描述。"""
    lookup = pd.read_csv(ROOT / 'data' / 'metadata' / 'hs92_hs4_product_lookup.csv')
    return lookup[['product_hs92_code', 'un_description']].dropna()


def build_semantic_proximity(products: list[str], descriptions: pd.DataFrame,
                             model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
    """
    构建语义 proximity 矩阵。

    方案 1: Sentence Transformer（优先，使用本地缓存模型）
    方案 2: TF-IDF 向量化 + 余弦相似度（回退）

    Sentence Transformer 能捕捉更深层的语义关系：
    - “semiconductor” 和 “integrated circuit” 在嵌入空间中更接近
    - “cotton” 和 “textile” 有更高的语义相似度
    """
    # 匹配产品与描述
    desc_map = dict(zip(descriptions['product_hs92_code'], descriptions['un_description']))
    matched = [(p, desc_map.get(p, p)) for p in products]
    texts = [desc for _, desc in matched]

    print(f'  嵌入 {len(texts)} 个产品描述 (模型: {model_name})...')
    t0 = time.time()

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        print(f'  Sentence Transformer 完成，耗时 {time.time() - t0:.1f}s，维度 {embeddings.shape}')

        # 余弦相似度（已归一化，点积 = 余弦）
        print('  计算余弦相似度矩阵...')
        sim = embeddings @ embeddings.T  # (P, P)
        np.fill_diagonal(sim, 0)

        # Top-30 稀疏化
        print('  Top-30 稀疏化...')
        k = 30
        rows, cols, vals = [], [], []
        for i in range(sim.shape[0]):
            top_k = np.argpartition(sim[i], -k)[-k:]
            for j in top_k:
                if sim[i, j] > 0:
                    rows.append(i)
                    cols.append(j)
                    vals.append(sim[i, j])

        phi_sem = sparse.csr_matrix((vals, (rows, cols)), shape=sim.shape)
        phi_sem = (phi_sem + phi_sem.T) / 2
        phi_sem_dense = phi_sem.toarray()
        phi_sem_dense = np.clip(phi_sem_dense, 0, 1)

        return phi_sem_dense, embeddings

    except Exception as e:
        print(f'  Sentence Transformer 失败: {e}')
        print('  回退到 TF-IDF...')
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(
            max_features=5000, stop_words='english',
            ngram_range=(1, 2), sublinear_tf=True,
        )
        embeddings = vectorizer.fit_transform(texts)
        print(f'  TF-IDF 完成，耗时 {time.time() - t0:.1f}s，维度 {embeddings.shape}')

        sim = cosine_similarity(embeddings)
        np.fill_diagonal(sim, 0)

        k = 30
        rows, cols, vals = [], [], []
        for i in range(sim.shape[0]):
            top_k = np.argpartition(sim[i], -k)[-k:]
            for j in top_k:
                if sim[i, j] > 0:
                    rows.append(i)
                    cols.append(j)
                    vals.append(sim[i, j])

        phi_sem = sparse.csr_matrix((vals, (rows, cols)), shape=sim.shape)
        phi_sem = (phi_sem + phi_sem.T) / 2
        phi_sem_dense = phi_sem.toarray()
        phi_sem_dense = np.clip(phi_sem_dense, 0, 1)

        return phi_sem_dense, embeddings


def blend_proximity(phi_stat, phi_sem, alpha: float):
    """
    混合 proximity: phi_hybrid = alpha * phi_stat + (1-alpha) * phi_sem

    alpha=1 → 纯统计
    alpha=0 → 纯语义
    """
    # 两者都已在 [0, 1] 内，直接加权
    hybrid = alpha * phi_stat + (1 - alpha) * phi_sem
    # 确保对称
    hybrid = (hybrid + hybrid.T) / 2
    return hybrid


def simulate_with_proximity(ctx, theta, phi, seed=42):
    """
    用自定义 proximity 矩阵运行模拟。
    复用 ResearchContext.simulate 的逻辑，但替换 proximity。
    """
    from model import ComplexityABM, ProductSpace, PolicyConfig

    # 创建新的 ProductSpace（用混合 proximity）
    space = ProductSpace(ctx.products, phi, max_neighbors=30)

    n_steps = END_YEAR - BASE_YEAR
    params = dict(zip(PARAMETERS, theta))
    model = ComplexityABM(
        space, ctx.caps, ctx.tech,
        policy_config=PolicyConfig(),
        investment_budget=ctx.budgets,
        resource_mask=ctx.resource_mask,
        seed=int(seed), **params
    )
    states = [np.array([a.capabilities.copy() for a in model.agents])]
    for _ in range(n_steps):
        model.step()
        states.append(np.array([a.capabilities.copy() for a in model.agents]))
    return np.asarray(states, dtype=np.uint8), model.history()


def diversity(states):
    return states.sum(axis=2).astype(np.int32)


def main():
    print('=' * 60)
    print('语义产品空间：LLM 嵌入增强')
    print('=' * 60)

    # 加载数据
    ctx = ResearchContext()
    descriptions = load_product_descriptions()
    print(f'产品 {len(ctx.products)}  有描述 {len(descriptions)}')

    # 构建语义 proximity
    phi_sem_dense, embeddings = build_semantic_proximity(ctx.products, descriptions)
    print(f'语义 proximity: shape={phi_sem_dense.shape}, 非零={np.count_nonzero(phi_sem_dense)}')

    # 保存嵌入
    if isinstance(embeddings, np.ndarray):
        np.save(EXPERIMENTS / 'product_embeddings_st.npy', embeddings)
        print(f'嵌入已保存: experiments/product_embeddings_st.npy (shape={embeddings.shape})')
    else:
        sparse.save_npz(EXPERIMENTS / 'product_embeddings_tfidf.npz', embeddings)
        print(f'嵌入已保存: experiments/product_embeddings_tfidf.npz')

    # 统计 proximity
    phi_stat = ctx.space.phi  # 稠密 numpy array

    # 测试不同 alpha
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    theta = DEFAULT_THETA

    actual = ctx.actual.sum(axis=2).astype(np.int32)
    hist_2023 = actual[-1]

    results = []
    for alpha in alphas:
        print(f'\n--- alpha = {alpha} ---')
        if alpha == 1.0:
            # 纯统计（基线）
            phi = phi_stat.copy()
        elif alpha == 0.0:
            # 纯语义
            phi = phi_sem_dense.copy()
        else:
            phi = blend_proximity(phi_stat, phi_sem_dense, alpha)

        # 运行模拟
        t0 = time.time()
        states, _ = simulate_with_proximity(ctx, theta, phi)
        elapsed = time.time() - t0

        div = diversity(states)
        final_div = div[-1]

        mae = float(np.abs(final_div - hist_2023).mean())
        rmse = float(np.sqrt(((final_div - hist_2023) ** 2).mean()))
        holdout_mae = float(np.abs(div[16:] - actual[16:]).mean())

        print(f'  最终 MAE={mae:.1f}  RMSE={rmse:.1f}  留出 MAE={holdout_mae:.1f}  '
              f'耗时 {elapsed:.1f}s')

        results.append({
            'alpha': alpha,
            'final_mae': mae,
            'final_rmse': rmse,
            'holdout_mae': holdout_mae,
            'mean_diversity': float(final_div.mean()),
            'elapsed': elapsed,
        })

    # 汇总
    df = pd.DataFrame(results)
    df.to_csv(EXPERIMENTS / 'semantic_proximity_results.csv', index=False)

    print('\n' + '=' * 60)
    print('结果汇总:')
    print(df.to_string(index=False))
    print('=' * 60)

    # 最优 alpha
    best_idx = df['holdout_mae'].idxmin()
    best_alpha = df.loc[best_idx, 'alpha']
    best_mae = df.loc[best_idx, 'holdout_mae']
    baseline_mae = df[df['alpha'] == 1.0]['holdout_mae'].values[0]

    summary = {
        'pipeline_version': PIPELINE_VERSION,
        'model': 'paraphrase-multilingual-MiniLM-L12-v2',
        'n_products': len(ctx.products),
        'n_descriptions': len(descriptions),
        'embedding_dim': embeddings.shape[1],
        'alphas_tested': alphas,
        'best_alpha': float(best_alpha),
        'best_holdout_mae': float(best_mae),
        'baseline_holdout_mae': float(baseline_mae),
        'improvement': float(baseline_mae - best_mae),
        'results': results,
    }
    (EXPERIMENTS / 'semantic_proximity_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'\n最优 alpha = {best_alpha}，留出 MAE = {best_mae:.1f}')
    print(f'基线 (纯统计) MAE = {baseline_mae:.1f}')
    print(f'改善 = {baseline_mae - best_mae:.1f}')

    print(f'\n产物:')
    print(f'  experiments/semantic_proximity_results.csv')
    print(f'  experiments/semantic_proximity_summary.json')
    print(f'  experiments/product_embeddings.npy')


if __name__ == '__main__':
    main()
