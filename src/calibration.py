"""
加权 ABC-SMC 校准模块
======================
实现 Toni et al. (2010) 风格的自适应 ABC 序列蒙特卡洛，
用于匹配 ABM 输出与历史数据的 summary statistics。

核心函数
--------
- importance_weights : 基于上一代混合核密度计算重要性权重
- abc_smc            : 主校准循环
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np


def importance_weights(
    candidates: np.ndarray,
    previous_particles: np.ndarray,
    previous_weights: np.ndarray,
    kernel_std: np.ndarray,
) -> np.ndarray:
    """ABC-SMC 重要性权重（均匀先验下）。

    对于均匀先验，w(x) ∝ 1 / Σ_j w_j · K_σ(x − θ_j)。
    返回归一化权重（和为 1）。
    """
    candidates = np.asarray(candidates, dtype=float)
    previous_particles = np.asarray(previous_particles, dtype=float)
    previous_weights = np.asarray(previous_weights, dtype=float)
    kernel_std = np.asarray(kernel_std, dtype=float)

    safe_std = np.maximum(kernel_std, 1e-15)
    diff = candidates[:, None, :] - previous_particles[None, :, :]
    log_pdf = -0.5 * np.sum((diff / safe_std) ** 2, axis=2)
    log_norm = -np.sum(np.log(safe_std * np.sqrt(2 * np.pi)))
    pdf = np.exp(log_pdf + log_norm)  # (N, M)

    mixture = pdf @ previous_weights  # (N,)
    w = np.where(mixture > 0, 1.0 / mixture, 0.0)
    total = w.sum()
    if total > 0:
        w /= total
    else:
        w = np.full(len(candidates), 1.0 / len(candidates))
    return w


def abc_smc(
    simulator: Callable,
    bounds: np.ndarray,
    target: np.ndarray,
    scales: np.ndarray,
    particles: int,
    generations: int,
    seed: int = 0,
    repeats: int = 1,
    max_attempts: int = 100,
    output: Path | None = None,
) -> dict:
    """加权 ABC-SMC 校准。

    Parameters
    ----------
    simulator   : (theta, seed) → 1-D array，与 target 同维
    bounds      : (D, 2)  参数下界/上界
    target      : (K,)    目标 summary statistics
    scales      : (K,)    距离归一化尺度（必须全部 > 0）
    particles   : 每代粒子数
    generations : 最大世代数
    seed        : 随机种子
    repeats     : 每个候选模拟次数（取均值降噪）
    max_attempts: 每代最大候选批次（每批 particles 个候选）
    output      : 若提供路径，保存诊断 JSON

    Returns
    -------
    dict with keys: particles, weights, complete, diagnostics
    """
    scales = np.asarray(scales, dtype=float)
    if (scales <= 0).any() or not np.isfinite(scales).all():
        raise ValueError('scales 必须全部为正')

    rng = np.random.default_rng(int(seed))
    bounds = np.asarray(bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    dim = len(lo)
    target = np.asarray(target, dtype=float)

    def _dist(sim_out):
        return float(np.linalg.norm((sim_out - target) / scales))

    diagnostics = []
    prev_particles = None
    prev_weights = None
    prev_kernel_std = None
    epsilon = np.inf
    complete = False

    for gen in range(generations):
        accepted_list = []
        dist_list = []
        attempts = 0

        for _ in range(max_attempts):
            attempts += 1

            # ---- 提议 1 个候选 ----
            if prev_particles is None:
                cand = rng.random(dim) * (hi - lo) + lo
            else:
                idx = rng.choice(particles, p=prev_weights)
                cand = np.clip(
                    prev_particles[idx]
                    + rng.normal(size=dim) * prev_kernel_std,
                    lo, hi,
                )

            # ---- 模拟 ----
            sim_sum = np.zeros(len(target))
            for _ in range(repeats):
                sim_sum += simulator(
                    cand, int(rng.integers(0, 2**31))
                )
            sim_mean = sim_sum / repeats
            d = _dist(sim_mean)

            if gen == 0 or d <= epsilon:
                accepted_list.append(cand)
                dist_list.append(d)

            if len(accepted_list) >= particles:
                break

        # ---- 判断是否耗尽 ----
        if len(accepted_list) < particles:
            ess_prev = (float(1.0 / (prev_weights ** 2).sum())
                        if prev_weights is not None else float(particles))
            diagnostics.append({
                'generation': gen,
                'epsilon': float(epsilon),
                'ess': ess_prev,
                'attempts': attempts,
                'accepted': len(accepted_list),
                'n_candidates': attempts,
            })
            complete = False
            break

        # 取前 particles 个
        acc_arr = np.array(accepted_list[:particles])
        dist_arr = np.array(dist_list[:particles])

        # ---- 加权 ----
        if gen == 0:
            acc_w = np.full(particles, 1.0 / particles)
        else:
            acc_w = importance_weights(
                acc_arr, prev_particles, prev_weights, prev_kernel_std,
            )

        # 新 epsilon = 接受粒子中最大距离（保证单调递减）
        epsilon = float(dist_arr.max())

        # ---- 核带宽 = 2 × 加权标准差 ----
        wm = np.average(acc_arr, axis=0, weights=acc_w)
        wv = np.average((acc_arr - wm) ** 2, axis=0, weights=acc_w)
        prev_kernel_std = 2.0 * np.sqrt(np.maximum(wv, 1e-12))

        ess = float(1.0 / np.sum(acc_w ** 2))
        diagnostics.append({
            'generation': gen,
            'epsilon': float(epsilon),
            'ess': ess,
            'attempts': attempts,
            'accepted': particles,
            'n_candidates': attempts,
        })

        prev_particles = acc_arr.copy()
        prev_weights = acc_w.copy()
        complete = True

    result = {
        'particles': (prev_particles
                      if prev_particles is not None
                      else np.empty((0, dim))),
        'weights': (prev_weights
                    if prev_weights is not None
                    else np.empty(0)),
        'complete': complete,
        'diagnostics': diagnostics,
    }

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            'particles': result['particles'].tolist(),
            'weights': result['weights'].tolist(),
            'complete': result['complete'],
            'diagnostics': diagnostics,
        }
        output.write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    return result
