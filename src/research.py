"""统一实验口径：2000初始化，2005/2010/2015训练，2016–2023留出。"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from model import (ComplexityABM, ProductSpace, PolicyConfig, load_processed,
                   initial_capabilities_from_year, initial_tech_from_eci)
from pipeline import investment_budgets_by_gdp, resource_product_mask, PIPELINE_VERSION

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = ['learning_rate', 'imitation_rate', 'depreciation_rate']
BOUNDS = np.array([[.01, 1.], [0., .5], [0., .04]])
TRAIN_YEARS = [2005, 2010, 2015]
BASE_YEAR = 2000
END_YEAR = 2023
DEFAULT_THETA = np.array([.05, .1, .005])


def regional_eci(matrix):
    """同一24经济体口径的区域ECI，不等同官方全球ECI。

    对称化的国家转移矩阵第二特征向量；符号向多样性正相关方向固定。
    空行输出0，谱退化输出0，避免将任意特征基误作可识别复杂度。
    """
    m = np.asarray(matrix, dtype=float)
    diversity = m.sum(axis=1)
    active = diversity > 0
    result = np.zeros(len(m))
    if active.sum() < 3:
        return result
    a = m[active]
    d = diversity[active]
    ubiquity = a.sum(axis=0)
    w = np.divide(a, ubiquity, out=np.zeros_like(a), where=ubiquity > 0)
    s = (w @ a.T) / np.sqrt(d[:, None] * d[None, :])
    values, vectors = np.linalg.eigh(s)
    if values[-1] - values[-2] < 1e-9 or values[-2] - values[-3] < 1e-9:
        return result
    v = vectors[:, -2] / np.sqrt(d)
    if v.std() < 1e-12:
        return result
    v = (v - v.mean()) / v.std()
    sign = np.dot(v, d - d.mean())
    if sign < 0 or (abs(sign) < 1e-12 and v[np.argmax(np.abs(v))] < 0):
        v = -v
    result[active] = v
    return result


def summary_vector(states):
    """拼接逐国多样性和区域ECI，避免只拟合均值掩盖国家间偏差。"""
    parts = []
    for year in TRAIN_YEARS:
        m = states[year - BASE_YEAR]
        parts.extend([m.sum(axis=1), regional_eci(m)])
    return np.concatenate(parts).astype(float)


class ResearchContext:
    def __init__(self):
        summary = json.loads((ROOT / 'data/processed/pipeline_summary.json').read_text(encoding='utf-8'))
        if summary.get('pipeline_version') != PIPELINE_VERSION:
            raise ValueError('旧版产品空间无效，请重新运行管线')
        countries, products, long, phi = load_processed()
        self.countries = sorted(countries)
        self.products = products
        self.space = ProductSpace(products, phi, max_neighbors=30)
        self.caps = initial_capabilities_from_year(long, self.countries, products, BASE_YEAR)
        self.tech = initial_tech_from_eci(self.countries, BASE_YEAR)
        self.budgets = investment_budgets_by_gdp(self.countries)
        self.resource_mask = resource_product_mask(products)
        self.actual = np.array([
            list(initial_capabilities_from_year(long, self.countries, products, year).values())
            for year in range(BASE_YEAR, END_YEAR + 1)], dtype=np.uint8)
        self.target = summary_vector(self.actual)
        self.scales = np.concatenate([
            np.r_[np.maximum(self.actual[y-BASE_YEAR].sum(axis=1), 50), np.full(len(countries), 2.)]
            for y in TRAIN_YEARS])
        # 计算国家增长趋势（年均多样性变化）
        self.growth_trends = self._compute_growth_trends()

    def _compute_growth_trends(self) -> dict[str, float]:
        """从历史数据计算每个国家的年均多样性变化。"""
        div = self.actual.sum(axis=2).astype(float)  # (T, C)
        n_years = div.shape[0] - 1
        trends = {}
        for i, c in enumerate(self.countries):
            change = div[-1, i] - div[0, i]
            trends[c] = float(change / n_years)
        return trends

    def simulate(self, theta, seed, end_year=END_YEAR, policy=None,
                 heterogeneous=True, resource=True, budget_override=None,
                 use_trends=True):
        theta = np.asarray(theta)
        budgets = self.budgets if heterogeneous else 5.
        if budget_override is not None:
            budgets = budget_override
        trends = self.growth_trends if use_trends else None
        model = ComplexityABM(self.space, self.caps, self.tech,
                              policy_config=policy or PolicyConfig(),
                              investment_budget=budgets,
                              resource_mask=self.resource_mask if resource else None,
                              growth_trend=trends,
                              seed=int(seed), **dict(zip(PARAMETERS, theta)))
        states = [np.array([a.capabilities.copy() for a in model.agents])]
        for _ in range(end_year - BASE_YEAR):
            model.step()
            states.append(np.array([a.capabilities.copy() for a in model.agents]))
        return np.asarray(states, dtype=np.uint8), model.history()

    def simulator(self, theta, seed):
        states, _ = self.simulate(theta, seed, end_year=max(TRAIN_YEARS))
        return summary_vector(states)

    def metrics(self, states):
        actual = self.actual.sum(axis=2).astype(float)
        simulated = states.sum(axis=2).astype(float)
        hold = simulated[16:] - actual[16:]
        end = simulated[-1] - actual[-1]
        eci_diff = regional_eci(states[-1]) - regional_eci(self.actual[-1])
        return dict(holdout_mae=float(np.abs(hold).mean()),
                    holdout_rmse=float(np.sqrt((hold**2).mean())),
                    final_mae=float(np.abs(end).mean()), final_rmse=float(np.sqrt((end**2).mean())),
                    final_regional_eci_mae=float(np.abs(eci_diff).mean()),
                    final_mean_diversity=float(simulated[-1].mean()),
                    n_overestimated=int((end > 0).sum()))


def input_fingerprint():
    """为校准/政策产物绑定代码、数据和协变量的SHA256。"""
    paths = ['src/model.py', 'src/pipeline.py', 'src/research.py', 'src/calibration.py',
             'data/processed/M_matrix.parquet', 'data/product_space/proximity_full.parquet',
             'data/processed/country_budget_inputs.csv']
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
