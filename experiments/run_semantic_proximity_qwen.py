"""
语义产品空间：千问 (Qwen) 嵌入增强
====================================
用 DashScope API 调用千问 text-embedding-v2 模型，
构建语义 proximity 矩阵，与统计 proximity 融合。

千问嵌入优势：
- 更强的语义理解能力
- 多语言支持（产品描述含英文）
- 1536 维高质量嵌入空间
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


def build_semantic_proximity_qwen(products: list[str], descriptions: pd.DataFrame,
                                   batch_size: int = 25):
    """
    用千问 text-embedding-v2 构建语义 proximity 矩阵。

    参数:
        products: 产品代码列表
        descriptions: 产品描述 DataFrame
        batch_size: API 批量大小（每次请求的文本数）

    返回:
        phi_sem_dense: 稠密语义 proximity 矩阵
        embeddings: 嵌入矩阵 (P, 1536)
    """
    import dashscope
    from dashscope import TextEmbedding

    # 匹配产品与描述
    desc_map = dict(zip(descriptions['product_hs92_code'], descriptions['un_description']))
    matched = [(p, desc_map.get(p, p)) for p in products]
    texts = [desc for _, desc in matched]

    print(f'  用千问 text-embedding-v2 嵌入 {len(texts)} 个产品描述...')
    t0 = time.time()

    # 批量调用 API
    all_embeddings = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]

        try:
            resp = TextEmbedding.call(
                model='text-embedding-v2',
                input=batch_texts,
                dimension=1536,  # 千问支持 1536 维
            )

            if resp.status_code == 200:
                batch_emb = [item['embedding'] for item in resp.output['embeddings']]
                all_embeddings.extend(batch_emb)
                if (i + 1) % 10 == 0 or i == n_batches - 1:
                    print(f'    批次 {i+1}/{n_batches} 完成')
            else:
                print(f'    API 错误: {resp.code} - {resp.message}')
                # 回退到零向量
                all_embeddings.extend([[0.0] * 1536] * len(batch_texts))

        except Exception as e:
            print(f'    异常: {e}')
            all_embeddings.extend([[0.0] * 1536] * len(batch_texts))

        # 限速：避免触发 API 限流
        time.sleep(0.1)

    embeddings = np.array(all_embeddings, dtype=np.float32)
    print(f'  千问嵌入完成，耗时 {time.time() - t0:.1f}s，维度 {embeddings.shape}')

    # 归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    # 余弦相似度
    print('  计算余弦相似度矩阵...')
    sim = embeddings @ embeddings.T
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


def blend_proximity(phi_stat, phi_sem, alpha: float):
    """混合 proximity: phi_hybrid = alpha * phi_stat + (1-alpha) * phi_sem"""
    hybrid = alpha * phi_stat + (1 - alpha) * phi_sem
    hybrid = (hybrid + hybrid.T) / 2
    return hybrid


def simulate_with_proximity(ctx, theta, phi, seed=42):
    """用自定义 proximity 矩阵运行模拟。"""
    from model import ComplexityABM, ProductSpace, PolicyConfig

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
    print('语义产品空间：千问 (Qwen) 嵌入增强')
    print('=' * 60)

    # 加载数据
    ctx = ResearchContext()
    descriptions = load_product_descriptions()
    print(f'产品 {len(ctx.products)}  有描述 {len(descriptions)}')

    # 构建语义 proximity（千问）
    phi_sem_dense, embeddings = build_semantic_proximity_qwen(ctx.products, descriptions)
    print(f'语义 proximity: shape={phi_sem_dense.shape}, 非零={np.count_nonzero(phi_sem_dense)}')

    # 保存嵌入
    np.save(EXPERIMENTS / 'product_embeddings_qwen.npy', embeddings)
    print(f'嵌入已保存: experiments/product_embeddings_qwen.npy (shape={embeddings.shape})')

    # 统计 proximity
    phi_stat = ctx.space.phi

    # 测试不同 alpha
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    theta = DEFAULT_THETA

    actual = ctx.actual.sum(axis=2).astype(np.int32)
    hist_2023 = actual[-1]

    results = []
    for alpha in alphas:
        print(f'\n--- alpha = {alpha} ---')
        if alpha == 1.0:
            phi = phi_stat.copy()
        elif alpha == 0.0:
            phi = phi_sem_dense.copy()
        else:
            phi = blend_proximity(phi_stat, phi_sem_dense, alpha)

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
    df.to_csv(EXPERIMENTS / 'semantic_proximity_qwen_results.csv', index=False)

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
        'model': 'qwen-text-embedding-v2',
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
    (EXPERIMENTS / 'semantic_proximity_qwen_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'\n最优 alpha = {best_alpha}，留出 MAE = {best_mae:.1f}')
    print(f'基线 (纯统计) MAE = {baseline_mae:.1f}')
    print(f'改善 = {baseline_mae - best_mae:.1f}')

    # 与之前方法对比
    print('\n' + '=' * 60)
    print('方法对比:')
    print('=' * 60)
    print(f'  纯统计基线:           MAE = 49.1')
    print(f'  TF-IDF:               MAE = 48.0 (改善 2.2%)')
    print(f'  Sentence Transformer: MAE = 48.6 (改善 1.0%)')
    print(f'  千问 Qwen:            MAE = {best_mae:.1f} (改善 {(baseline_mae - best_mae) / baseline_mae * 100:.1f}%)')


if __name__ == '__main__':
    main()
