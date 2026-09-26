"""
政策配对实验
=============
三组配对实验（baseline vs 政策干预），使用校准参数或默认参数。
每组使用相同 seed，仅政策参数不同。

实验 A: 定向补贴 — CHN 半导体 (HS92 8542)
实验 B: 技术封锁 — CHN 半导体+计算机 (8542, 8471)
实验 C: 关税保护 — ASEAN 劳动密集型 (6109, 6204, 6110)
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

from model import PolicyConfig, load_processed, ProductSpace
from pipeline import (investment_budgets_by_gdp, resource_product_mask,
                      PIPELINE_VERSION)
from research import (ResearchContext, PARAMETERS, DEFAULT_THETA,
                      BASE_YEAR, END_YEAR)

EXPERIMENTS = ROOT / 'experiments'
EXPERIMENTS.mkdir(parents=True, exist_ok=True)

SEED = 42


def load_theta():
    """尝试加载校准参数，失败则回退默认。"""
    summary_path = EXPERIMENTS / 'calibration_summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        theta = np.array([summary['theta_star'][p] for p in PARAMETERS])
        print(f'  使用校准参数: {dict(zip(PARAMETERS, theta))}')
        return theta
    print(f'  校准文件不存在，使用默认参数: {dict(zip(PARAMETERS, DEFAULT_THETA))}')
    return DEFAULT_THETA


def run_paired(ctx, theta, policy, label, start_step=6):
    """运行配对实验：baseline + 政策，返回对比 DataFrame。"""
    policy_cfg = policy
    policy_cfg.start_step = start_step

    # baseline（无政策）
    states_base, hist_base = ctx.simulate(theta, seed=SEED)
    # 政策
    states_policy, hist_policy = ctx.simulate(
        theta, seed=SEED, policy=policy_cfg)

    # 逐国对比（最终多样性差异）
    countries = ctx.countries
    div_base = states_base[-1].sum(axis=1).astype(np.int32)
    div_policy = states_policy[-1].sum(axis=1).astype(np.int32)

    rows = []
    for i, iso3 in enumerate(countries):
        rows.append({
            'experiment': label,
            'country': iso3,
            'baseline_diversity': int(div_base[i]),
            'policy_diversity': int(div_policy[i]),
            'diff': int(div_policy[i]) - int(div_base[i]),
        })

    # 逐时间步对比（均值）
    ts_rows = []
    for step in range(states_base.shape[0]):
        ts_rows.append({
            'experiment': label,
            'step': step,
            'year': BASE_YEAR + step,
            'baseline_mean_div': float(states_base[step].sum(axis=1).astype(np.int32).mean()),
            'policy_mean_div': float(states_policy[step].sum(axis=1).astype(np.int32).mean()),
        })

    return pd.DataFrame(rows), pd.DataFrame(ts_rows)


def main():
    print('=' * 60)
    print('政策配对实验')
    print('=' * 60)

    theta = load_theta()

    print('\n[1/5] 加载实验上下文 ...')
    ctx = ResearchContext()
    products = ctx.products
    print(f'  国家 {len(ctx.countries)}  产品 {len(products)}')

    # 检查目标产品是否存在
    all_experiments = []

    # ---- 实验 A: 定向补贴 CHN 半导体 ----
    print('\n[2/5] 实验 A: 定向补贴 CHN → 8542 (集成电路) ...')
    subsidy_targets = [p for p in ['8542'] if p in products]
    if not subsidy_targets:
        print('  警告: 8542 不在产品列表中，跳过')
    else:
        policy_a = PolicyConfig(
            subsidy_targets=subsidy_targets,
            subsidy_strength=3.0,
            countries=['CHN'],
            start_step=6,
        )
        df_a, ts_a = run_paired(ctx, theta, policy_a, 'A_subsidy_CHN_8542')
        all_experiments.append(('A', df_a, ts_a))
        print(f'  CHN 多样性差异: {df_a[df_a.country=="CHN"]["diff"].values[0]:+d}')

    # ---- 实验 B: 技术封锁 CHN ----
    print('\n[3/5] 实验 B: 技术封锁 CHN → 8542, 8471 ...')
    blockade_targets = [p for p in ['8542', '8471'] if p in products]
    if not blockade_targets:
        print('  警告: 目标产品不在列表中，跳过')
    else:
        policy_b = PolicyConfig(
            blockade_targets=blockade_targets,
            blockade_strength=0.3,
            countries=['CHN'],
            start_step=6,
        )
        df_b, ts_b = run_paired(ctx, theta, policy_b, 'B_blockade_CHN')
        all_experiments.append(('B', df_b, ts_b))
        if 'CHN' in ctx.countries:
            chn_diff = df_b[df_b.country == 'CHN']['diff'].values[0]
            print(f'  CHN 多样性差异: {chn_diff:+d}')

    # ---- 实验 C: 关税 ASEAN ----
    print('\n[4/5] 实验 C: 关税 VNM/IDN/BGD → 劳动密集型产品 ...')
    tariff_targets = [p for p in ['6109', '6204', '6110', '6104']
                      if p in products]
    if not tariff_targets:
        print('  警告: 目标产品不在列表中，跳过')
    else:
        policy_c = PolicyConfig(
            tariff_targets=tariff_targets,
            tariff_rate=0.25,
            tariff_elasticity=1.0,
            countries=['VNM', 'IDN', 'BGD'],
            start_step=6,
        )
        df_c, ts_c = run_paired(ctx, theta, policy_c, 'C_tariff_ASEAN')
        all_experiments.append(('C', df_c, ts_c))
        for c in ['VNM', 'IDN', 'BGD']:
            if c in ctx.countries:
                d = df_c[df_c.country == c]['diff'].values[0]
                print(f'  {c} 多样性差异: {d:+d}')

    # ---- 保存结果 ----
    print('\n[5/5] 保存结果 ...')
    country_dfs = [e[1] for e in all_experiments]
    ts_dfs = [e[2] for e in all_experiments]

    if country_dfs:
        pd.concat(country_dfs, ignore_index=True).to_csv(
            EXPERIMENTS / 'policy_country_results.csv', index=False)
    if ts_dfs:
        pd.concat(ts_dfs, ignore_index=True).to_csv(
            EXPERIMENTS / 'policy_timeseries_results.csv', index=False)

    # 汇总
    summary = {
        'pipeline_version': PIPELINE_VERSION,
        'theta': {p: float(v) for p, v in zip(PARAMETERS, theta)},
        'seed': SEED,
        'policy_start_step': 6,
        'experiments_run': [e[0] for e in all_experiments],
        'subsidy': {'targets': subsidy_targets if subsidy_targets else [],
                    'strength': 3.0, 'country': 'CHN'},
        'blockade': {'targets': blockade_targets if blockade_targets else [],
                     'strength': 0.3, 'country': 'CHN'},
        'tariff': {'targets': tariff_targets if tariff_targets else [],
                   'rate': 0.25, 'elasticity': 1.0,
                   'countries': ['VNM', 'IDN', 'BGD']},
    }
    (EXPERIMENTS / 'policy_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'\n产物:')
    print(f'  {EXPERIMENTS / "policy_country_results.csv"}')
    print(f'  {EXPERIMENTS / "policy_timeseries_results.csv"}')
    print(f'  {EXPERIMENTS / "policy_summary.json"}')
    print('\n完成。')


if __name__ == '__main__':
    main()
