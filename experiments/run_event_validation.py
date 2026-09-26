"""
真实事件验证：2018 中美贸易战
================================
分析实际数据中 CHN 在贸易战前后的变化，
与模型预测做定量对照。

贸易战时间线：
- 2018.03: 美国宣布对钢铝加征关税
- 2018.07: 第一轮 340 亿美元关税生效 (List 1)
- 2018.09: 第二轮 2000 亿美元关税生效 (List 3)
- 2019.05: List 3 税率从 10% 升至 25%
- 2019.09: List 4A 生效

目标产品 (HS92):
- List 1/2: 8471, 8542, 8541, 8473, 8504, 8517 等 (机械/电子/半导体)
- List 3: 覆盖更广，包括家具、纺织、化工等
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

from model import ComplexityABM, ProductSpace, PolicyConfig, load_processed
from research import ResearchContext, DEFAULT_THETA, PARAMETERS, BASE_YEAR, END_YEAR
from pipeline import PIPELINE_VERSION

EXPERIMENTS = ROOT / 'experiments'
EXPERIMENTS.mkdir(parents=True, exist_ok=True)


def analyze_actual_data(ctx):
    """分析实际数据中 CHN 在贸易战前后的变化。"""
    print('\n' + '=' * 60)
    print('[分析 1] 实际数据：CHN 贸易战前后变化')
    print('=' * 60)

    actual = ctx.actual  # (T, C, P) uint8
    countries = ctx.countries
    products = ctx.products
    chn_idx = countries.index('CHN')

    # CHN 多样性时间序列
    div_chn = actual[:, chn_idx, :].sum(axis=1).astype(int)  # (T,)

    print(f'\nCHN 多样性时间序列:')
    for t in range(len(div_chn)):
        year = BASE_YEAR + t
        marker = ' ← 贸易战开始' if year == 2018 else ''
        print(f'  {year}: {div_chn[t]:4d}{marker}')

    # 贸易战前后对比
    div_2017 = div_chn[2017 - BASE_YEAR]
    div_2018 = div_chn[2018 - BASE_YEAR]
    div_2019 = div_chn[2019 - BASE_YEAR]
    div_2020 = div_chn[2020 - BASE_YEAR]
    div_2023 = div_chn[2023 - BASE_YEAR]

    print(f'\n贸易战前后:')
    print(f'  2017: {div_2017}')
    print(f'  2018: {div_2018} (变化: {div_2018 - div_2017:+d})')
    print(f'  2019: {div_2019} (变化: {div_2019 - div_2018:+d})')
    print(f'  2020: {div_2020} (变化: {div_2020 - div_2019:+d})')
    print(f'  2023: {div_2023} (vs 2017: {div_2023 - div_2017:+d})')

    # 找出 CHN 丢失的产品
    caps_2017 = actual[2017 - BASE_YEAR, chn_idx, :].astype(bool)
    caps_2023 = actual[2023 - BASE_YEAR, chn_idx, :].astype(bool)

    lost = caps_2017 & ~caps_2023
    gained = ~caps_2017 & caps_2023

    lost_products = [products[i] for i in range(len(products)) if lost[i]]
    gained_products = [products[i] for i in range(len(products)) if gained[i]]

    print(f'\nCHN 产品变化 (2017→2023):')
    print(f'  丢失: {lost.sum()} 个产品')
    print(f'  新增: {gained.sum()} 个产品')
    print(f'  净变化: {gained.sum() - lost.sum():+d}')

    if len(lost_products) > 0:
        print(f'\n  丢失产品列表: {lost_products[:20]}{"..." if len(lost_products) > 20 else ""}')
    if len(gained_products) > 0:
        print(f'\n  新增产品列表: {gained_products[:20]}{"..." if len(gained_products) > 20 else ""}')

    return {
        'div_2017': int(div_2017),
        'div_2023': int(div_2023),
        'lost': int(lost.sum()),
        'gained': int(gained.sum()),
        'lost_products': lost_products,
        'gained_products': gained_products,
    }


def identify_trade_war_targets(products):
    """识别贸易战目标产品在模型中的映射。"""
    print('\n' + '=' * 60)
    print('[分析 2] 贸易战目标产品映射')
    print('=' * 60)

    # 美国贸易战 List 1/2 目标产品 (HS92 四位码)
    # 来源: USTR Section 301 tariff lists
    list1_targets = [
        '8471', '8473', '8479', '8501', '8502', '8503', '8504',
        '8511', '8512', '8513', '8516', '8517', '8518', '8519',
        '8521', '8522', '8523', '8525', '8526', '8527', '8528',
        '8529', '8531', '8534', '8535', '8536', '8537', '8538',
        '8539', '8540', '8541', '8542', '8543', '8544', '8545',
        '8546', '8547', '8548',
    ]

    # 半导体核心产品
    semiconductor_targets = ['8542', '8541', '8534', '8536', '8537']

    # 找出在模型产品集中的目标
    found_targets = [p for p in list1_targets if p in products]
    found_semi = [p for p in semiconductor_targets if p in products]

    print(f'\nList 1/2 目标: {len(list1_targets)} 个 HS 码')
    print(f'  在模型中找到: {len(found_targets)} 个')
    print(f'  半导体核心: {len(found_semi)} 个 → {found_semi}')

    # 加载产品描述
    lookup = pd.read_csv(ROOT / 'data' / 'metadata' / 'hs92_hs4_product_lookup.csv')
    desc_map = dict(zip(lookup['product_hs92_code'], lookup['un_description']))

    print(f'\n半导体目标产品描述:')
    for p in found_semi:
        desc = desc_map.get(p, 'N/A')
        print(f'  {p}: {desc[:80]}')

    return found_targets, found_semi


def simulate_trade_war(ctx, targets, start_year=2018, blockade_strength=0.2):
    """模拟贸易战场景。"""
    print('\n' + '=' * 60)
    print(f'[模拟] 贸易战场景: {len(targets)} 目标产品, 起始 {start_year}, 封锁强度 {blockade_strength}')
    print('=' * 60)

    theta = DEFAULT_THETA
    start_step = start_year - BASE_YEAR

    # 基线（无政策）
    from research import ResearchContext
    states_base, _ = ctx.simulate(theta, seed=42)

    # 贸易战（对 CHN 技术封锁）
    policy = PolicyConfig(
        blockade_targets=targets,
        blockade_strength=blockade_strength,
        countries=['CHN'],
        start_step=start_step,
    )
    states_war, _ = ctx.simulate(theta, seed=42, policy=policy)

    # 分析结果
    countries = ctx.countries
    chn_idx = countries.index('CHN')

    div_base = states_base.sum(axis=2).astype(np.int32)
    div_war = states_war.sum(axis=2).astype(np.int32)
    actual = ctx.actual.sum(axis=2).astype(np.int32)

    print(f'\nCHN 多样性对比 (2017-2023):')
    print(f'  年份   实际   基线   贸易战  贸易战-基线')
    for year in range(2017, 2024):
        t = year - BASE_YEAR
        actual_val = actual[t, chn_idx]
        base_val = div_base[t, chn_idx]
        war_val = div_war[t, chn_idx]
        diff = war_val - base_val
        marker = ' ←' if year == 2018 else ''
        print(f'  {year}  {actual_val:5d}  {base_val:5d}  {war_val:5d}  {diff:+4d}{marker}')

    # 2023 对比
    actual_2023 = actual[-1, chn_idx]
    base_2023 = div_base[-1, chn_idx]
    war_2023 = div_war[-1, chn_idx]

    print(f'\n2023 年结果:')
    print(f'  实际: {actual_2023}')
    print(f'  基线: {base_2023} (误差: {base_2023 - actual_2023:+d})')
    print(f'  贸易战: {war_2023} (误差: {war_2023 - actual_2023:+d})')
    print(f'  贸易战效果: {war_2023 - base_2023:+d}')

    # 逐产品分析
    caps_base_2023 = states_base[-1, chn_idx, :].astype(bool)
    caps_war_2023 = states_war[-1, chn_idx, :].astype(bool)
    products = ctx.products

    lost_by_war = caps_base_2023 & ~caps_war_2023
    lost_products = [products[i] for i in range(len(products)) if lost_by_war[i]]

    print(f'\n贸易战导致丢失的产品: {lost_by_war.sum()} 个')
    if len(lost_products) > 0:
        print(f'  {lost_products[:20]}{"..." if len(lost_products) > 20 else ""}')

    return {
        'actual_2023': int(actual_2023),
        'baseline_2023': int(base_2023),
        'trade_war_2023': int(war_2023),
        'war_effect': int(war_2023 - base_2023),
        'baseline_error': int(base_2023 - actual_2023),
        'war_error': int(war_2023 - actual_2023),
        'lost_products': lost_products,
    }


def simulate_counterfactual(ctx, targets, start_year=2018):
    """反事实分析：如果没有贸易战，CHN 会怎样？"""
    print('\n' + '=' * 60)
    print('[反事实] 如果没有贸易战')
    print('=' * 60)

    theta = DEFAULT_THETA
    states_base, _ = ctx.simulate(theta, seed=42)

    div_base = states_base.sum(axis=2).astype(np.int32)
    actual = ctx.actual.sum(axis=2).astype(np.int32)
    chn_idx = ctx.countries.index('CHN')

    # 贸易战期间 (2018-2023) 的实际变化
    actual_change = actual[2023 - BASE_YEAR, chn_idx] - actual[2017 - BASE_YEAR, chn_idx]
    # 基线预测的变化
    base_change = div_base[2023 - BASE_YEAR, chn_idx] - div_base[2017 - BASE_YEAR, chn_idx]

    print(f'\nCHN 2017→2023 变化:')
    print(f'  实际: {actual_change:+d}')
    print(f'  基线预测: {base_change:+d}')
    print(f'  差异: {actual_change - base_change:+d}')

    if actual_change < base_change:
        print(f'\n  解读: 实际增长低于模型预测 {base_change - actual_change} 个产品')
        print(f'  这可能反映了贸易战的抑制效果')
    else:
        print(f'\n  解读: 实际增长高于模型预测')

    return {
        'actual_change': int(actual_change),
        'baseline_change': int(base_change),
        'gap': int(actual_change - base_change),
    }


def main():
    print('=' * 60)
    print('真实事件验证：2018 中美贸易战')
    print('=' * 60)

    ctx = ResearchContext()
    print(f'国家 {len(ctx.countries)}  产品 {len(ctx.products)}')

    # 分析实际数据
    actual_analysis = analyze_actual_data(ctx)

    # 识别目标产品
    targets, semi_targets = identify_trade_war_targets(ctx.products)

    if len(targets) == 0:
        print('\n警告: 没有找到贸易战目标产品！')
        return

    # 模拟贸易战（使用半导体目标）
    use_targets = semi_targets if len(semi_targets) > 0 else targets[:10]
    war_result = simulate_trade_war(ctx, use_targets, start_year=2018, blockade_strength=0.2)

    # 反事实分析
    counterfactual = simulate_counterfactual(ctx, use_targets)

    # 汇总
    summary = {
        'pipeline_version': PIPELINE_VERSION,
        'event': '2018 US-China Trade War',
        'actual_analysis': {
            'div_2017': actual_analysis['div_2017'],
            'div_2023': actual_analysis['div_2023'],
            'lost': actual_analysis['lost'],
            'gained': actual_analysis['gained'],
        },
        'trade_war_simulation': war_result,
        'counterfactual': counterfactual,
        'targets_used': use_targets,
        'start_year': 2018,
        'blockade_strength': 0.2,
    }

    (EXPERIMENTS / 'trade_war_validation.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8',
    )

    print('\n' + '=' * 60)
    print('产物: experiments/trade_war_validation.json')
    print('=' * 60)


if __name__ == '__main__':
    main()
