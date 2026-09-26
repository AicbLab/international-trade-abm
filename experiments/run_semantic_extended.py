"""
语义产品空间扩展实验
====================
测试三个语义空间切入点的独立与组合效果：
1. 语义加权模仿（semantic imitation）
2. 语义引导趋势修正（semantic trend）
3. 封锁语义溢出（blockade spillover）

实验设计：2^3 消融（有/无每个机制），加贸易战场景对比
"""
import sys, json, time
import numpy as np
from pathlib import Path
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from model import ComplexityABM, ProductSpace, PolicyConfig
from research import ResearchContext, DEFAULT_THETA, PARAMETERS, BASE_YEAR, END_YEAR
from pipeline import PIPELINE_VERSION
from sklearn.metrics.pairwise import cosine_similarity

EXPERIMENTS = ROOT / 'experiments'


def load_semantic_phi():
    """加载 TF-IDF 语义 proximity（表现最好的方法）"""
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


def blend_phi(phi_stat, phi_sem, alpha):
    """融合统计和语义 proximity"""
    hybrid = alpha * phi_stat + (1 - alpha) * phi_sem
    return (hybrid + hybrid.T) / 2


def simulate(ctx, theta, phi, seed=42, sem_phi=None, policy=None, use_trends=True):
    """运行模拟"""
    space = ProductSpace(ctx.products, phi, max_neighbors=30)
    params = dict(zip(PARAMETERS, theta))
    model = ComplexityABM(
        product_space=space,
        initial_capabilities=ctx.caps,
        initial_tech=ctx.tech,
        policy_config=policy or PolicyConfig(),
        investment_budget=ctx.budgets,
        resource_mask=ctx.resource_mask,
        growth_trend=ctx.growth_trends if use_trends else None,
        semantic_phi=sem_phi,
        seed=int(seed),
        **params,
    )
    states = [np.array([a.capabilities.copy() for a in model.agents])]
    for _ in range(END_YEAR - BASE_YEAR):
        model.step()
        states.append(np.array([a.capabilities.copy() for a in model.agents]))
    return np.asarray(states, dtype=np.uint8)


def diversity(states):
    return states.sum(axis=2).astype(np.int32)


def main():
    print("=" * 70)
    print("Semantic Product Space: Extended Experiments")
    print("=" * 70)

    ctx = ResearchContext()
    actual = ctx.actual.sum(axis=2).astype(np.int32)
    theta = DEFAULT_THETA
    phi_stat = ctx.space.phi

    # 加载语义 proximity（TF-IDF，最优 α=0.8）
    phi_sem = load_semantic_phi()
    phi_blended = blend_phi(phi_stat, phi_sem, 0.8)

    # 贸易战目标
    semi_targets = [p for p in ['8542', '8541', '8534', '8536', '8537'] if p in ctx.products]
    broad_targets = [p for p in [
        '8471', '8473', '8517', '8523', '8525', '8526', '8528', '8529',
        '8534', '8536', '8537', '8541', '8542',
        '8501', '8502', '8503', '8504',
    ] if p in ctx.products]

    chn_idx = ctx.countries.index('CHN')
    results = {}

    # ================================================================
    # Part 1: 消融实验（基线预测力）
    # ================================================================
    print("\n" + "=" * 70)
    print("Part 1: Ablation Study (Baseline Prediction)")
    print("=" * 70)

    configs = {
        # (sem_imitation, sem_trend, phi_type)
        'A_base':        (False, False, 'stat'),
        'B_stat+sem':    (False, False, 'blend'),
        'C_sem_imi':     (True,  False, 'blend'),
        'D_sem_trend':   (False, True,  'blend'),
        'E_both_sem':    (True,  True,  'blend'),
    }

    for name, (sem_imi, sem_trd, phi_type) in configs.items():
        t0 = time.time()
        phi = phi_blended if phi_type == 'blend' else phi_stat
        # 如果不使用语义机制，sem_phi 传 None
        sem = phi_sem if (sem_imi or sem_trd) else None
        states = simulate(ctx, theta, phi, seed=42, sem_phi=sem, use_trends=True)
        div = diversity(states)
        final = div[-1]
        mae = float(np.abs(final - actual[-1]).mean())
        holdout = float(np.abs(div[16:] - actual[16:]).mean())
        elapsed = time.time() - t0

        chn_err = final[chn_idx] - actual[-1, chn_idx]
        results[name] = {
            'sem_imitation': sem_imi, 'sem_trend': sem_trd, 'phi': phi_type,
            'final_mae': mae, 'holdout_mae': holdout, 'chn_err': int(chn_err),
            'time': round(elapsed, 1),
        }
        print(f"  {name:15s}: MAE={mae:.1f}  holdout={holdout:.1f}  CHN_err={chn_err:+d}  ({elapsed:.1f}s)")

    # ================================================================
    # Part 2: 贸易战 + 封锁溢出
    # ================================================================
    print("\n" + "=" * 70)
    print("Part 2: Trade War with Semantic Spillover")
    print("=" * 70)

    war_configs = {
        'no_war':        None,
        'semi_block':    PolicyConfig(blockade_targets=semi_targets, blockade_strength=0.2,
                                      countries=['CHN'], start_step=2018-BASE_YEAR),
        'broad_block':   PolicyConfig(blockade_targets=broad_targets, blockade_strength=0.2,
                                      countries=['CHN'], start_step=2018-BASE_YEAR),
        'broad_spill':   PolicyConfig(blockade_targets=broad_targets, blockade_strength=0.2,
                                      blockade_spillover=0.5,
                                      countries=['CHN'], start_step=2018-BASE_YEAR),
        'strong_spill':  PolicyConfig(blockade_targets=broad_targets, blockade_strength=0.5,
                                      blockade_spillover=0.8,
                                      countries=['CHN'], start_step=2018-BASE_YEAR),
    }

    war_results = {}
    for wname, policy in war_configs.items():
        t0 = time.time()
        states = simulate(ctx, theta, phi_blended, seed=42, sem_phi=phi_sem,
                          policy=policy, use_trends=True)
        div = diversity(states)
        chn_2023 = int(div[-1, chn_idx])
        elapsed = time.time() - t0
        war_results[wname] = {'chn_2023': chn_2023, 'time': round(elapsed, 1)}
        print(f"  {wname:15s}: CHN 2023 = {chn_2023}  ({elapsed:.1f}s)")

    # 计算贸易战效果
    base_chn = war_results['no_war']['chn_2023']
    print(f"\n  Trade war effects on CHN (vs baseline {base_chn}):")
    for wname in war_results:
        if wname != 'no_war':
            effect = war_results[wname]['chn_2023'] - base_chn
            war_results[wname]['effect'] = effect
            print(f"    {wname:15s}: {effect:+d}")

    # ================================================================
    # Part 3: 多 seed 鲁棒性
    # ================================================================
    print("\n" + "=" * 70)
    print("Part 3: Multi-seed Robustness")
    print("=" * 70)

    seeds = [42, 123, 456, 789, 2024]
    robustness = {'stat_no_sem': [], 'blend_no_sem': [], 'blend_sem': []}

    for seed in seeds:
        # 纯统计，无语义机制
        states = simulate(ctx, theta, phi_stat, seed=seed, sem_phi=None, use_trends=True)
        div = diversity(states)
        robustness['stat_no_sem'].append(float(np.abs(div[16:] - actual[16:]).mean()))

        # 融合，无语义机制
        states = simulate(ctx, theta, phi_blended, seed=seed, sem_phi=None, use_trends=True)
        div = diversity(states)
        robustness['blend_no_sem'].append(float(np.abs(div[16:] - actual[16:]).mean()))

        # 融合 + 语义机制
        states = simulate(ctx, theta, phi_blended, seed=seed, sem_phi=phi_sem, use_trends=True)
        div = diversity(states)
        robustness['blend_sem'].append(float(np.abs(div[16:] - actual[16:]).mean()))

    for rname, vals in robustness.items():
        arr = np.array(vals)
        print(f"  {rname:15s}: mean={arr.mean():.2f} ± {arr.std():.2f}  (seeds={seeds})")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    summary = {
        'pipeline_version': PIPELINE_VERSION,
        'ablation': results,
        'trade_war': war_results,
        'robustness': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v)),
                           'values': [float(x) for x in v]}
                       for k, v in robustness.items()},
        'chn_actual': int(actual[-1, chn_idx]),
    }

    (EXPERIMENTS / 'semantic_extended_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\nSaved: experiments/semantic_extended_summary.json")


if __name__ == '__main__':
    main()
