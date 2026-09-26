"""
ABC-SMC 校准实验
================
1. 加载 ResearchContext（数据 + 产品空间 + 协变量）
2. 运行加权 ABC-SMC 匹配训练期 (2005/2010/2015) summary statistics
3. 保存校准结果
4. 用后验参数运行完整 24 步 baseline
5. 评估留出期 (2016-2023) 指标
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
from research import ResearchContext, PARAMETERS, BOUNDS, DEFAULT_THETA, BASE_YEAR, END_YEAR
from pipeline import PIPELINE_VERSION

EXPERIMENTS = ROOT / 'experiments'
EXPERIMENTS.mkdir(parents=True, exist_ok=True)

# ---------- 校准参数 ----------
PARTICLES = 50
GENERATIONS = 5
REPEATS = 2
SEED = 42


def main():
    print('=' * 60)
    print('ABC-SMC 校准实验')
    print('=' * 60)

    # ---- 加载上下文 ----
    print('\n[1/5] 加载实验上下文 ...')
    t0 = time.time()
    ctx = ResearchContext()
    print(f'      国家 {len(ctx.countries)}  产品 {len(ctx.products)}')
    print(f'      目标维度 {len(ctx.target)}  训练年份 2005/2010/2015')
    print(f'      加载耗时 {time.time()-t0:.1f}s')

    # ---- ABC-SMC 校准 ----
    print(f'\n[2/5] 运行 ABC-SMC（{PARTICLES} 粒子 × {GENERATIONS} 代 × {REPEATS} 重复）...')
    t0 = time.time()
    result = abc_smc(
        simulator=ctx.simulator,
        bounds=BOUNDS,
        target=ctx.target,
        scales=ctx.scales,
        particles=PARTICLES,
        generations=GENERATIONS,
        seed=SEED,
        repeats=REPEATS,
        max_attempts=500,
        output=EXPERIMENTS / 'calibration_result.json',
    )
    elapsed = time.time() - t0
    print(f'      完成 = {result["complete"]}  耗时 {elapsed:.1f}s')

    # ---- 诊断输出 ----
    print('\n[3/5] 逐代诊断:')
    diag_rows = []
    for g in result['diagnostics']:
        print(f'      代 {g["generation"]}: ε={g["epsilon"]:.4f}  '
              f'ESS={g["ess"]:.1f}  尝试={g["attempts"]}')
        diag_rows.append(g)
    pd.DataFrame(diag_rows).to_csv(EXPERIMENTS / 'calibration_history.csv', index=False)

    # ---- 后验参数 ----
    theta_star = np.average(result['particles'], axis=0, weights=result['weights'])
    print(f'\n[4/5] 后验参数（加权均值）:')
    for name, val in zip(PARAMETERS, theta_star):
        print(f'      {name} = {val:.4f}')
    print(f'      默认参数: {dict(zip(PARAMETERS, DEFAULT_THETA))}')

    # ---- 留出验证 ----
    print(f'\n[5/5] 留出验证 (2016-2023) ...')

    # 校准参数
    states_cal, hist_cal = ctx.simulate(theta_star, seed=SEED)
    metrics_cal = ctx.metrics(states_cal)

    # 默认参数
    states_def, hist_def = ctx.simulate(DEFAULT_THETA, seed=SEED)
    metrics_def = ctx.metrics(states_def)

    print('\n      指标对比:')
    print(f'      {"指标":<30s} {"校准":>10s} {"默认":>10s}')
    print(f'      {"-"*52}')
    for key in metrics_cal:
        v_cal = metrics_cal[key]
        v_def = metrics_def[key]
        better = '✓' if abs(v_cal) < abs(v_def) else ''
        print(f'      {key:<30s} {v_cal:>10.2f} {v_def:>10.2f}  {better}')

    # ---- 保存汇总 ----
    summary = {
        'pipeline_version': PIPELINE_VERSION,
        'particles': PARTICLES,
        'generations': GENERATIONS,
        'repeats': REPEATS,
        'seed': SEED,
        'complete': result['complete'],
        'theta_star': {name: float(val) for name, val in zip(PARAMETERS, theta_star)},
        'theta_default': {name: float(val) for name, val in zip(PARAMETERS, DEFAULT_THETA)},
        'metrics_calibrated': metrics_cal,
        'metrics_default': metrics_def,
        'n_generations_run': len(result['diagnostics']),
    }
    (EXPERIMENTS / 'calibration_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    print(f'\n产物:')
    print(f'  {EXPERIMENTS / "calibration_result.json"}')
    print(f'  {EXPERIMENTS / "calibration_history.csv"}')
    print(f'  {EXPERIMENTS / "calibration_summary.json"}')
    print('\n完成。')


if __name__ == '__main__':
    main()
