"""
真实事件验证 v2：贸易战 vs 产业政策
======================================
核心发现：模型预测 CHN 2017→2023 减少 14 产品，实际增加 54 产品。
差距 68 产品 = "中国制造 2025" 产业政策效果 - 贸易战抑制效果

本脚本测试：
1. 纯贸易战（封锁）能否解释实际变化
2. 产业政策（补贴）能否弥补差距
3. 贸易战 + 产业政策组合能否匹配实际
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

from model import PolicyConfig
from research import ResearchContext, DEFAULT_THETA, BASE_YEAR, END_YEAR
from pipeline import PIPELINE_VERSION

EXPERIMENTS = ROOT / 'experiments'
EXPERIMENTS.mkdir(parents=True, exist_ok=True)


def diversity(states):
    return states.sum(axis=2).astype(np.int32)


def main():
    print('=' * 60)
    print('真实事件验证 v2：贸易战 vs 产业政策')
    print('=' * 60)

    ctx = ResearchContext()
    countries = ctx.countries
    products = ctx.products
    chn_idx = countries.index('CHN')
    theta = DEFAULT_THETA

    # 实际数据
    actual = ctx.actual.sum(axis=2).astype(np.int32)
    actual_2017 = actual[2017 - BASE_YEAR, chn_idx]
    actual_2023 = actual[2023 - BASE_YEAR, chn_idx]
    actual_change = actual_2023 - actual_2017

    print(f'\n实际数据: CHN 2017={actual_2017}, 2023={actual_2023}, 变化={actual_change:+d}')

    # 半导体目标产品
    semi_targets = ['8542', '8541', '8534', '8536', '8537']
    semi_targets = [p for p in semi_targets if p in products]

    # 中国制造 2025 目标领域 (HS92 映射)
    # 来源: MIC2025 十大领域 → HS 码近似映射
    mic2025_targets = [
        # 新一代信息技术
        '8471', '8473', '8517', '8523', '8525', '8526', '8528', '8529', '8534',
        '8536', '8537', '8541', '8542',
        # 高端数控机床
        '8456', '8457', '8458', '8459', '8460',
        # 航空航天
        '8801', '8802', '8803', '8804', '8805',
        # 海洋工程
        '8901', '8902', '8903', '8904', '8905',
        # 先进轨道交通
        '8601', '8602', '8603', '8604', '8605', '8606', '8607', '8608', '8609',
        # 节能与新能源汽车
        '8701', '8702', '8703', '8704', '8705', '8706', '8707', '8708', '8709',
        # 电力装备
        '8501', '8502', '8503', '8504',
        # 新材料
        '7208', '7209', '7210', '7211', '7212',
        # 生物医药
        '3001', '3002', '3003', '3004', '3005', '3006',
    ]
    mic2025_targets = [p for p in mic2025_targets if p in products]

    print(f'\n目标产品:')
    print(f'  半导体 (贸易战目标): {len(semi_targets)} 个 → {semi_targets}')
    print(f'  中国制造2025 (产业政策目标): {len(mic2025_targets)} 个')

    # ============================================================
    # 场景 1: 基线（无政策）
    # ============================================================
    print('\n--- 场景 1: 基线（无政策） ---')
    states_base, _ = ctx.simulate(theta, seed=42)
    div_base = diversity(states_base)
    base_2017 = div_base[2017 - BASE_YEAR, chn_idx]
    base_2023 = div_base[2023 - BASE_YEAR, chn_idx]
    base_change = base_2023 - base_2017
    print(f'  CHN: {base_2017} → {base_2023}, 变化={base_change:+d}')

    # ============================================================
    # 场景 2: 纯贸易战（封锁半导体）
    # ============================================================
    print('\n--- 场景 2: 纯贸易战（封锁半导体, 2018起） ---')
    policy_war = PolicyConfig(
        blockade_targets=semi_targets,
        blockade_strength=0.2,
        countries=['CHN'],
        start_step=2018 - BASE_YEAR,
    )
    states_war, _ = ctx.simulate(theta, seed=42, policy=policy_war)
    div_war = diversity(states_war)
    war_2023 = div_war[2023 - BASE_YEAR, chn_idx]
    war_effect = war_2023 - base_2023
    print(f'  CHN 2023: {war_2023}, 贸易战效果: {war_effect:+d}')

    # ============================================================
    # 场景 3: 纯产业政策（补贴中国制造2025）
    # ============================================================
    print('\n--- 场景 3: 纯产业政策（补贴中国制造2025, 2015起） ---')
    policy_mic = PolicyConfig(
        subsidy_targets=mic2025_targets,
        subsidy_strength=3.0,
        countries=['CHN'],
        start_step=2015 - BASE_YEAR,
    )
    states_mic, _ = ctx.simulate(theta, seed=42, policy=policy_mic)
    div_mic = diversity(states_mic)
    mic_2023 = div_mic[2023 - BASE_YEAR, chn_idx]
    mic_effect = mic_2023 - base_2023
    print(f'  CHN 2023: {mic_2023}, 产业政策效果: {mic_effect:+d}')

    # ============================================================
    # 场景 4: 贸易战 + 产业政策
    # ============================================================
    print('\n--- 场景 4: 贸易战 + 产业政策 ---')
    policy_both = PolicyConfig(
        subsidy_targets=mic2025_targets,
        subsidy_strength=3.0,
        blockade_targets=semi_targets,
        blockade_strength=0.2,
        countries=['CHN'],
        start_step=2015 - BASE_YEAR,
    )
    states_both, _ = ctx.simulate(theta, seed=42, policy=policy_both)
    div_both = diversity(states_both)
    both_2023 = div_both[2023 - BASE_YEAR, chn_idx]
    print(f'  CHN 2023: {both_2023}')

    # ============================================================
    # 汇总对比
    # ============================================================
    print('\n' + '=' * 60)
    print('汇总对比')
    print('=' * 60)
    print(f'\nCHN 2017→2023 变化:')
    print(f'  实际:          {actual_change:+d}')
    print(f'  基线预测:      {base_change:+d}')
    print(f'  差距:          {actual_change - base_change:+d}')
    print(f'\n政策效果:')
    print(f'  贸易战 (封锁): {war_effect:+d}')
    print(f'  产业政策 (补贴): {mic_effect:+d}')
    print(f'  组合:          {both_2023 - base_2023:+d}')

    # 逐产品分析
    caps_actual_2017 = ctx.actual[2017 - BASE_YEAR, chn_idx, :].astype(bool)
    caps_actual_2023 = ctx.actual[2023 - BASE_YEAR, chn_idx, :].astype(bool)
    caps_base_2023 = states_base[-1, chn_idx, :].astype(bool)
    caps_mic_2023 = states_mic[-1, chn_idx, :].astype(bool)

    # 实际新增但模型未预测到的产品
    actual_gained = ~caps_actual_2017 & caps_actual_2023
    model_missed = actual_gained & ~caps_base_2023
    mic_covered = actual_gained & caps_mic_2023 & ~caps_base_2023

    print(f'\n逐产品分析:')
    print(f'  实际新增: {actual_gained.sum()} 个')
    print(f'  模型未预测到: {model_missed.sum()} 个')
    print(f'  产业政策覆盖: {mic_covered.sum()} 个')

    # 汇总
    summary = {
        'pipeline_version': PIPELINE_VERSION,
        'event': '2018 US-China Trade War + Made in China 2025',
        'actual': {
            'chn_2017': int(actual_2017),
            'chn_2023': int(actual_2023),
            'change': int(actual_change),
        },
        'baseline': {
            'chn_2017': int(base_2017),
            'chn_2023': int(base_2023),
            'change': int(base_change),
        },
        'gap': int(actual_change - base_change),
        'scenarios': {
            'trade_war_only': {'chn_2023': int(war_2023), 'effect': int(war_effect)},
            'mic2025_only': {'chn_2023': int(mic_2023), 'effect': int(mic_effect)},
            'both': {'chn_2023': int(both_2023), 'effect': int(both_2023 - base_2023)},
        },
        'product_analysis': {
            'actual_gained': int(actual_gained.sum()),
            'model_missed': int(model_missed.sum()),
            'mic_covered': int(mic_covered.sum()),
        },
    }

    (EXPERIMENTS / 'trade_war_validation_v2.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    print('\n' + '=' * 60)
    print('产物: experiments/trade_war_validation_v2.json')
    print('=' * 60)


if __name__ == '__main__':
    main()
