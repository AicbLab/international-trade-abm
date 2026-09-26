"""
综合实验：高贡献迭代
====================
1. 同质 vs 异质预算 A/B 对比
2. 默认参数(低折旧)下的政策实验（放大效果量）
3. 增强政策场景（更强补贴/封锁/多国产关税）
4. 参数敏感性网格（折旧率 × 政策强度）
5. 简单 benchmark 对比（不变假设 / 线性外推）
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from model import PolicyConfig, load_processed, initial_capabilities_from_year, initial_tech_from_eci, ProductSpace
from pipeline import investment_budgets_by_gdp, resource_product_mask, PIPELINE_VERSION
from research import ResearchContext, PARAMETERS, DEFAULT_THETA, BASE_YEAR, END_YEAR

EXPERIMENTS = ROOT / 'experiments'
EXPERIMENTS.mkdir(parents=True, exist_ok=True)

SEED = 42


def simulate_states(ctx, theta, seed=SEED, end_year=END_YEAR, policy=None,
                    heterogeneous=True, resource=True, budget_override=None):
    """运行模拟，返回 states 数组。"""
    return ctx.simulate(theta, seed, end_year=end_year, policy=policy,
                        heterogeneous=heterogeneous, resource=resource,
                        budget_override=budget_override)


def diversity(states):
    """states (T, C, P) → (T, C) 多样性。"""
    return states.sum(axis=2).astype(np.int32)


# ============================================================
# 1. 同质 vs 异质预算 A/B
# ============================================================
def experiment_budget_comparison(ctx):
    print('\n' + '=' * 60)
    print('[实验 1] 同质 vs 异质预算 A/B 对比')
    print('=' * 60)

    theta = DEFAULT_THETA
    states_homo, _ = simulate_states(ctx, theta, heterogeneous=False)
    states_heter, _ = simulate_states(ctx, theta, heterogeneous=True)

    div_homo = diversity(states_homo)
    div_heter = diversity(states_heter)

    # 历史
    actual_div = ctx.actual.sum(axis=2).astype(np.int32)  # (T, C)
    hist_2023 = actual_div[-1]

    final_homo = div_homo[-1]
    final_heter = div_heter[-1]

    mae_homo = float(np.abs(final_homo - hist_2023).mean())
    mae_heter = float(np.abs(final_heter - hist_2023).mean())
    rmse_homo = float(np.sqrt(((final_homo - hist_2023) ** 2).mean()))
    rmse_heter = float(np.sqrt(((final_heter - hist_2023) ** 2).mean()))

    # 留出期
    hold_homo = div_homo[16:] - actual_div[16:]
    hold_heter = div_heter[16:] - actual_div[16:]
    hold_mae_homo = float(np.abs(hold_homo).mean())
    hold_mae_heter = float(np.abs(hold_heter).mean())

    print(f'  最终 MAE:    同质={mae_homo:.1f}  异质={mae_heter:.1f}')
    print(f'  最终 RMSE:   同质={rmse_homo:.1f}  异质={rmse_heter:.1f}')
    print(f'  留出 MAE:    同质={hold_mae_homo:.1f}  异质={hold_mae_heter:.1f}')

    # 逐国对比
    rows = []
    countries = ctx.countries
    for i, c in enumerate(countries):
        rows.append({
            'country': c,
            'historical_2023': int(hist_2023[i]),
            'homogeneous_budget': int(final_homo[i]),
            'heterogeneous_budget': int(final_heter[i]),
            'diff_homo': int(final_homo[i] - hist_2023[i]),
            'diff_heter': int(final_heter[i] - hist_2023[i]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(EXPERIMENTS / 'budget_comparison.csv', index=False)

    result = {
        'homogeneous': {'final_mae': mae_homo, 'final_rmse': rmse_homo,
                        'holdout_mae': hold_mae_homo, 'mean_diversity': float(final_homo.mean())},
        'heterogeneous': {'final_mae': mae_heter, 'final_rmse': rmse_heter,
                          'holdout_mae': hold_mae_heter, 'mean_diversity': float(final_heter.mean())},
    }
    print(f'  平均多样性:  同质={result["homogeneous"]["mean_diversity"]:.1f}  '
          f'异质={result["heterogeneous"]["mean_diversity"]:.1f}')
    return result


# ============================================================
# 2. 默认参数下的政策实验（低折旧 → 更大效果量）
# ============================================================
def experiment_policy_default(ctx, theta=None):
    print('\n' + '=' * 60)
    print('[实验 2] 默认参数下的政策配对实验')
    print('=' * 60)

    if theta is None:
        theta = DEFAULT_THETA
    products = ctx.products

    results = {}

    # A: 定向补贴 CHN 半导体
    targets_a = [p for p in ['8542'] if p in products]
    if targets_a:
        pol = PolicyConfig(subsidy_targets=targets_a, subsidy_strength=5.0,
                           countries=['CHN'], start_step=1)
        st_base, _ = simulate_states(ctx, theta)
        st_pol, _ = simulate_states(ctx, theta, policy=pol)
        d = int(diversity(st_pol)[-1][ctx.countries.index('CHN')]) - \
            int(diversity(st_base)[-1][ctx.countries.index('CHN')])
        results['A_subsidy_CHN_strong'] = {'diff_CHN': d, 'strength': 5.0, 'start': 1}
        print(f'  A 补贴(CHN,×5,step1): CHN 差异 = {d:+d}')

    # B: 技术封锁 CHN（强封锁 + 早启动）
    targets_b = [p for p in ['8542', '8471', '8541', '8473'] if p in products]
    if targets_b:
        pol = PolicyConfig(blockade_targets=targets_b, blockade_strength=0.5,
                           countries=['CHN'], start_step=1)
        st_base, _ = simulate_states(ctx, theta)
        st_pol, _ = simulate_states(ctx, theta, policy=pol)
        d = int(diversity(st_pol)[-1][ctx.countries.index('CHN')]) - \
            int(diversity(st_base)[-1][ctx.countries.index('CHN')])
        results['B_blockade_CHN_strong'] = {'diff_CHN': d, 'strength': 0.5,
                                             'targets': len(targets_b), 'start': 1}
        print(f'  B 封锁(CHN,0.5,{len(targets_b)}品,step1): CHN 差异 = {d:+d}')

    # C: 关税 ASEAN（多产品 + 多国家）
    targets_c = [p for p in ['6109', '6204', '6110', '6104', '6203', '6103']
                 if p in products]
    if targets_c:
        pol = PolicyConfig(tariff_targets=targets_c, tariff_rate=0.5,
                           tariff_elasticity=1.5,
                           countries=['VNM', 'IDN', 'BGD', 'KHM', 'MMR'],
                           start_step=1)
        st_base, _ = simulate_states(ctx, theta)
        st_pol, _ = simulate_states(ctx, theta, policy=pol)
        div_base = diversity(st_base)[-1]
        div_pol = diversity(st_pol)[-1]
        country_diffs = {}
        for c in ['VNM', 'IDN', 'BGD', 'KHM', 'MMR']:
            if c in ctx.countries:
                idx = ctx.countries.index(c)
                d = int(div_pol[idx]) - int(div_base[idx])
                country_diffs[c] = d
        results['C_tariff_ASEAN_strong'] = {'country_diffs': country_diffs,
                                             'rate': 0.5, 'elasticity': 1.5,
                                             'targets': len(targets_c), 'start': 1}
        print(f'  C 关税(ASEAN,0.5,{len(targets_c)}品,step1): {country_diffs}')

    # D: 全面封锁 CHN（极端场景 — 所有高技术产品）
    high_tech = [p for p in products if p[:2] in ('84', '85', '90', '38')]
    if len(high_tech) >= 10:
        pol = PolicyConfig(blockade_targets=high_tech[:50], blockade_strength=0.3,
                           countries=['CHN'], start_step=1)
        st_base, _ = simulate_states(ctx, theta)
        st_pol, _ = simulate_states(ctx, theta, policy=pol)
        d = int(diversity(st_pol)[-1][ctx.countries.index('CHN')]) - \
            int(diversity(st_base)[-1][ctx.countries.index('CHN')])
        results['D_full_blockade_CHN'] = {'diff_CHN': d, 'n_targets': min(50, len(high_tech)),
                                           'strength': 0.3, 'start': 1}
        print(f'  D 全面封锁(CHN,0.3,{min(50, len(high_tech))}品): CHN 差异 = {d:+d}')

    return results


# ============================================================
# 3. 参数敏感性网格
# ============================================================
def experiment_sensitivity_grid(ctx):
    print('\n' + '=' * 60)
    print('[实验 3] 参数敏感性：折旧率 × 封锁强度')
    print('=' * 60)

    products = ctx.products
    blockade_targets = [p for p in ['8542', '8471', '8541', '8473'] if p in products]

    dep_values = [0.001, 0.005, 0.01, 0.02, 0.04]
    blk_values = [0.0, 0.1, 0.2, 0.3, 0.5]

    rows = []
    for dep in dep_values:
        for blk in blk_values:
            theta = np.array([0.05, 0.1, dep])
            # baseline
            st_base, _ = simulate_states(ctx, theta)
            div_base = int(diversity(st_base)[-1][ctx.countries.index('CHN')])
            # blockade
            if blk > 0:
                pol = PolicyConfig(blockade_targets=blockade_targets,
                                   blockade_strength=blk,
                                   countries=['CHN'], start_step=1)
                st_pol, _ = simulate_states(ctx, theta, policy=pol)
                div_pol = int(diversity(st_pol)[-1][ctx.countries.index('CHN')])
            else:
                div_pol = div_base
            rows.append({
                'depreciation': dep,
                'blockade_strength': blk,
                'chn_baseline_div': div_base,
                'chn_policy_div': div_pol,
                'chn_diff': div_pol - div_base,
            })

    df = pd.DataFrame(rows)
    df.to_csv(EXPERIMENTS / 'sensitivity_grid.csv', index=False)

    print('  封锁效果（CHN 多样性差异）:')
    print(df[df['blockade_strength'] > 0][['depreciation', 'blockade_strength', 'chn_diff']]
           .to_string(index=False))
    return df


# ============================================================
# 4. 简单 Benchmark
# ============================================================
def experiment_benchmarks(ctx):
    print('\n' + '=' * 60)
    print('[实验 4] 简单 Benchmark 对比')
    print('=' * 60)

    actual = ctx.actual.sum(axis=2).astype(np.int32)  # (T, C)
    hist_2023 = actual[-1]
    hist_2000 = actual[0]

    # Benchmark 1: 不变假设（预测 = 2000 年值）
    naive = hist_2000
    mae_naive = float(np.abs(naive - hist_2023).mean())

    # Benchmark 2: 线性外推（2000→2005 趋势 → 外推到 2023）
    hist_2005 = actual[5]
    trend = (hist_2005 - hist_2000).astype(float)
    linear = (hist_2000 + trend * (23 / 5)).astype(np.int32)
    mae_linear = float(np.abs(linear - hist_2023).mean())

    # Benchmark 3: 全球均值（所有国家预测为全球平均）
    global_mean = int(hist_2023.mean())
    mae_global = float(np.abs(np.full(len(hist_2023), global_mean) - hist_2023).mean())

    # 模型（默认参数）
    states, _ = simulate_states(ctx, DEFAULT_THETA)
    model_final = diversity(states)[-1]
    mae_model = float(np.abs(model_final - hist_2023).mean())

    print(f'  不变假设 MAE:    {mae_naive:.1f}')
    print(f'  线性外推 MAE:    {mae_linear:.1f}')
    print(f'  全球均值 MAE:    {mae_global:.1f}')
    print(f'  ABM (默认) MAE:  {mae_model:.1f}')

    result = {
        'naive_persistence': mae_naive,
        'linear_extrapolation': mae_linear,
        'global_mean': mae_global,
        'abm_default': mae_model,
    }
    return result


# ============================================================
# 5. 逐国时间序列对比（用于绘图）
# ============================================================
def experiment_timeseries(ctx):
    print('\n' + '=' * 60)
    print('[实验 5] 逐国时间序列（模型 vs 历史）')
    print('=' * 60)

    states, _ = simulate_states(ctx, DEFAULT_THETA)
    div = diversity(states)  # (T, C)
    actual = ctx.actual.sum(axis=2).astype(np.int32)  # (T, C)

    rows = []
    for t in range(div.shape[0]):
        for i, c in enumerate(ctx.countries):
            rows.append({
                'year': BASE_YEAR + t,
                'country': c,
                'model_diversity': int(div[t, i]),
                'historical_diversity': int(actual[t, i]),
            })
    df = pd.DataFrame(rows)
    df.to_csv(EXPERIMENTS / 'timeseries_model_vs_history.csv', index=False)
    print(f'  已保存 {len(df)} 行时间序列数据')
    return df


# ============================================================
# Main
# ============================================================
def main():
    print('=' * 60)
    print('综合实验：高贡献迭代')
    print('=' * 60)

    ctx = ResearchContext()
    print(f'国家 {len(ctx.countries)}  产品 {len(ctx.products)}')

    all_results = {}

    all_results['budget_comparison'] = experiment_budget_comparison(ctx)
    all_results['policy_default'] = experiment_policy_default(ctx)
    all_results['sensitivity_grid'] = 'saved to sensitivity_grid.csv'
    experiment_sensitivity_grid(ctx)
    all_results['benchmarks'] = experiment_benchmarks(ctx)
    experiment_timeseries(ctx)

    # 汇总
    summary = {
        'pipeline_version': PIPELINE_VERSION,
        'default_theta': {p: float(v) for p, v in zip(PARAMETERS, DEFAULT_THETA)},
        'seed': SEED,
        'budget_comparison': all_results['budget_comparison'],
        'benchmarks': all_results['benchmarks'],
        'policy_default': {k: {kk: (vv if not isinstance(vv, dict) else vv)
                                for kk, vv in v.items()}
                           for k, v in all_results['policy_default'].items()},
    }
    (EXPERIMENTS / 'full_analysis_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8',
    )

    print('\n' + '=' * 60)
    print('全部实验完成。产物:')
    for f in ['budget_comparison.csv', 'sensitivity_grid.csv',
              'timeseries_model_vs_history.csv', 'full_analysis_summary.json']:
        print(f'  experiments/{f}')
    print('=' * 60)


if __name__ == '__main__':
    main()
