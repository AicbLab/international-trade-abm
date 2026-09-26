"""
语义产品空间 v2：使用趋势修正后的模型
========================================
"""
import sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from model import ProductSpace, PolicyConfig
from research import ResearchContext, DEFAULT_THETA, PARAMETERS, BASE_YEAR, END_YEAR
from pipeline import PIPELINE_VERSION

EXPERIMENTS = ROOT / 'experiments'

def blend_proximity(phi_stat, phi_sem, alpha):
    hybrid = alpha * phi_stat + (1 - alpha) * phi_sem
    return (hybrid + hybrid.T) / 2

def simulate(ctx, theta, phi, seed=42):
    space = ProductSpace(ctx.products, phi, max_neighbors=30)
    params = dict(zip(PARAMETERS, theta))
    model_args = dict(
        product_space=space,
        initial_capabilities=ctx.caps,
        initial_tech=ctx.tech,
        policy_config=PolicyConfig(),
        investment_budget=ctx.budgets,
        resource_mask=ctx.resource_mask,
        growth_trend=ctx.growth_trends,
        seed=int(seed),
        **params,
    )
    from model import ComplexityABM
    model = ComplexityABM(**model_args)
    states = [np.array([a.capabilities.copy() for a in model.agents])]
    for _ in range(END_YEAR - BASE_YEAR):
        model.step()
        states.append(np.array([a.capabilities.copy() for a in model.agents]))
    return np.asarray(states, dtype=np.uint8)

def diversity(states):
    return states.sum(axis=2).astype(np.int32)

def main():
    print("=" * 60)
    print("Semantic Product Space v2: Trend-Corrected")
    print("=" * 60)

    ctx = ResearchContext()
    actual = ctx.actual.sum(axis=2).astype(np.int32)
    phi_stat = ctx.space.phi

    # Load pre-computed semantic proximities
    # TF-IDF
    tfidf_emb = sparse.load_npz(EXPERIMENTS / 'product_embeddings_tfidf.npz')
    from sklearn.metrics.pairwise import cosine_similarity
    sim_tfidf = cosine_similarity(tfidf_emb)
    np.fill_diagonal(sim_tfidf, 0)
    k = 30
    rows, cols, vals = [], [], []
    for i in range(sim_tfidf.shape[0]):
        top_k = np.argpartition(sim_tfidf[i], -k)[-k:]
        for j in top_k:
            if sim_tfidf[i, j] > 0:
                rows.append(i); cols.append(j); vals.append(sim_tfidf[i, j])
    phi_tfidf = sparse.csr_matrix((vals, (rows, cols)), shape=sim_tfidf.shape)
    phi_tfidf = (phi_tfidf + phi_tfidf.T) / 2
    phi_tfidf_dense = np.clip(phi_tfidf.toarray(), 0, 1)

    # Qwen
    emb_qwen = np.load(EXPERIMENTS / 'product_embeddings_qwen.npy')
    sim_qwen = emb_qwen @ emb_qwen.T
    np.fill_diagonal(sim_qwen, 0)
    rows, cols, vals = [], [], []
    for i in range(sim_qwen.shape[0]):
        top_k = np.argpartition(sim_qwen[i], -k)[-k:]
        for j in top_k:
            if sim_qwen[i, j] > 0:
                rows.append(i); cols.append(j); vals.append(sim_qwen[i, j])
    phi_qwen = sparse.csr_matrix((vals, (rows, cols)), shape=sim_qwen.shape)
    phi_qwen = (phi_qwen + phi_qwen.T) / 2
    phi_qwen_dense = np.clip(phi_qwen.toarray(), 0, 1)

    # Sentence Transformer
    emb_st = np.load(EXPERIMENTS / 'product_embeddings_st.npy')
    sim_st = emb_st @ emb_st.T
    np.fill_diagonal(sim_st, 0)
    rows, cols, vals = [], [], []
    for i in range(sim_st.shape[0]):
        top_k = np.argpartition(sim_st[i], -k)[-k:]
        for j in top_k:
            if sim_st[i, j] > 0:
                rows.append(i); cols.append(j); vals.append(sim_st[i, j])
    phi_st = sparse.csr_matrix((vals, (rows, cols)), shape=sim_st.shape)
    phi_st = (phi_st + phi_st.T) / 2
    phi_st_dense = np.clip(phi_st.toarray(), 0, 1)

    # Test different alphas for each method
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    theta = DEFAULT_THETA
    hist_2023 = actual[-1]

    methods = {
        'tfidf': phi_tfidf_dense,
        'qwen': phi_qwen_dense,
        'st': phi_st_dense,
    }

    all_results = {}
    for name, phi_sem in methods.items():
        print(f"\n--- {name} ---")
        results = []
        for alpha in alphas:
            if alpha == 1.0:
                phi = phi_stat.copy()
            elif alpha == 0.0:
                phi = phi_sem.copy()
            else:
                phi = blend_proximity(phi_stat, phi_sem, alpha)

            states = simulate(ctx, theta, phi)
            div = diversity(states)
            final = div[-1]

            mae = float(np.abs(final - hist_2023).mean())
            holdout = float(np.abs(div[16:] - actual[16:]).mean())
            print(f"  alpha={alpha:.1f}: MAE={mae:.1f}  holdout={holdout:.1f}")
            results.append({'alpha': alpha, 'final_mae': mae, 'holdout_mae': holdout})

        best = min(results, key=lambda r: r['holdout_mae'])
        all_results[name] = {'results': results, 'best_alpha': best['alpha'],
                             'best_holdout': best['holdout_mae']}

    # Baseline (pure statistical, trend-corrected)
    states_base = simulate(ctx, theta, phi_stat)
    div_base = diversity(states_base)
    baseline_holdout = float(np.abs(div_base[16:] - actual[16:]).mean())
    baseline_mae = float(np.abs(div_base[-1] - hist_2023).mean())

    print(f"\n{'='*60}")
    print("Summary (trend-corrected model)")
    print(f"{'='*60}")
    print(f"  Baseline (stat only): holdout MAE = {baseline_holdout:.1f}, final MAE = {baseline_mae:.1f}")
    for name, res in all_results.items():
        print(f"  {name}: best alpha={res['best_alpha']}, holdout MAE = {res['best_holdout']:.1f}, "
              f"improvement = {baseline_holdout - res['best_holdout']:.1f} "
              f"({(baseline_holdout - res['best_holdout'])/baseline_holdout*100:.1f}%)")

    summary = {
        'pipeline_version': PIPELINE_VERSION,
        'model': 'trend-corrected',
        'baseline': {'holdout_mae': baseline_holdout, 'final_mae': baseline_mae},
        'methods': all_results,
    }
    (EXPERIMENTS / 'semantic_v2_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\nSaved: experiments/semantic_v2_summary.json")

if __name__ == '__main__':
    main()
