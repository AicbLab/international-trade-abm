"""
降维 ABC-SMC 校准 v2
=====================
将目标从 144 维降到 ~15 维：
  - 每训练年：mean_div, median_div, p25_div, p75_div, max_div, eci_mean (6 维 × 3 年 = 18 维)
增加粒子到 200，世代到 10。
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

from calibration import abc_smc
from research import ResearchContext, PARAMETERS, BOUNDS, DEFAULT_THETA, BASE_YEAR, TRAIN_YEARS
from pipeline import PIPELINE_VERSION

EXPERIMENTS = ROOT / 'experiments'
EXPERIMENTS.mkdir(parents=True, exist_ok=True)

PARTICLES = 200
GENERATIONS = 10
REPEATS = 2
SEED = 42


def low_dim_summary(states):
    """低维 summary：每年 6 个统计量 × 3 年 = 18 维。"""
    from research import regional_eci
    parts = []
    for year in TRAIN_YEARS:
        m = states[year - BASE_YEAR]
        div = m.sum(axis=1).astype(float)
        eci = regional_eci(m)
        parts.extend([
            div.mean(),
            np.median(div),
            np.percentile(div, 25),
            np.percentile(div, 75),
            div.max(),
            np.abs(eci).mean(),
        ])
    return np.array(parts)


def main():
    print('=' * 60)
    print('降维 ABC-SMC 校准 v2')
    print('=' * 60)

    ctx = ResearchContext()
    print(f'国家 {len(ctx.countries)}  产品 {len(ctx.products)}')

    # 计算低维目标
    target = low_dim_summary(ctx.actual)
    print(f'目标维度: {len(target)}')
    print(f'目标值: {np.round(target, 2)}')

    # 尺度：每个维度的 max(abs(target), 10%)
    scales = np.maximum(np.abs(target) * 0.1, 5.0)

    def simulator(theta, seed):
        states, _ = ctx.simulate(theta, seed, end_year=max(TRAIN_YEARS))
        return low_dim_summary(states)

    # ABC-SMC
    print(f'\n运行 ABC-SMC（{PARTICLES} 粒子 × {GENERATIONS} 代 × {REPEATS} 重复）...')
    t0 = time.time()
    result = abc_smc(
        simulator=simulator,
        bounds=BOUNDS,
        target=target,
        scales=scales,
        particles=PARTICLES,
        generations=GENERATIONS,
        seed=SEED,
        repeats=REPEATS,
        max_attempts=2000,
        output=EXPERIMENTS / 'calibration_v2_result.json',
    )
    elapsed = time.time() - t0
    print(f'完成 = {result["complete"]}  耗时 {elapsed:.1f}s')

    # 诊断
    print('\n逐代诊断:')
    for g in result['diagnostics']:
        print(f'  代 {g["generation"]}: ε={g["epsilon"]:.4f}  ESS={g["ess"]:.1f}  尝试={g["attempts"]}')

    # 后验参数
    theta_star = np.average(result['particles'], axis=0, weights=result['weights'])
    print(f'\n后验参数（加权均值）:')
    for name, val in zip(PARAMETERS, theta_star):
        print(f'  {name} = {val:.4f}')

    # 留出验证
    states_cal, _ = ctx.simulate(theta_star, seed=SEED)
    states_def, _ = ctx.simulate(DEFAULT_THETA, seed=SEED)

    actual_div = ctx.actual.sum(axis=2).astype(np.int32)
    div_cal = states_cal.sum(axis=2).astype(np.int32)
    div_def = states_def.sum(axis=2).astype(np.int32)

    hold_cal = np.abs(div_cal[16:] - actual_div[16:]).mean()
    hold_def = np.abs(div_def[16:] - actual_div[16:]).mean()
    final_cal = np.abs(div_cal[-1] - actual_div[-1]).mean()
    final_def = np.abs(div_def[-1] - actual_div[-1]).mean()

    print(f'\n留出验证:')
    print(f'  校准: holdout_mae={hold_cal:.1f}  final_mae={final_cal:.1f}')
    print(f'  默认: holdout_mae={hold_def:.1f}  final_mae={final_def:.1f}')

    summary = {
        'pipeline_version': PIPELINE_VERSION,
        'particles': PARTICLES,
        'generations': GENERATIONS,
        'repeats': REPEATS,
        'seed': SEED,
        'target_dim': len(target),
        'complete': result['complete'],
        'theta_star': {n: float(v) for n, v in zip(PARAMETERS, theta_star)},
        'theta_default': {n: float(v) for n, v in zip(PARAMETERS, DEFAULT_THETA)},
        'calibrated': {'holdout_mae': float(hold_cal), 'final_mae': float(final_cal)},
        'default': {'holdout_mae': float(hold_def), 'final_mae': float(final_def)},
        'epsilon_trajectory': [g['epsilon'] for g in result['diagnostics']],
    }
    (EXPERIMENTS / 'calibration_v2_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'\n产物: experiments/calibration_v2_result.json')
    print(f'      experiments/calibration_v2_summary.json')


if __name__ == '__main__':
    main()
