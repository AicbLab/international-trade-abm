"""
国家智能体 ABM（Mesa 3.x）
=================================================
简化方案：每个国家 = 一个智能体，能力向量为该国在各产品上的 RCA 二值状态。
每步：
  1. 投资：以当前能力产品为种子，按产品空间邻近度概率积累邻近能力
  2. 模仿：观察成功国家，概率性复制其新增能力
  3. 政策：选择性补贴 / 功能性支持 / 技术封锁 通过 policy_multiplier 调制
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import networkx as nx
import mesa
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PSPACE = ROOT / "data" / "product_space"


# ---------- 产品空间封装 ----------

class ProductSpace:
    """产品空间网络封装：提供邻居查询、距离、子图采样。"""

    def __init__(self, products: list[str], phi: np.ndarray, threshold: float = 0.5,
                 max_neighbors: int | None = 30):
        self.products = list(products)
        self.p_idx = {p: i for i, p in enumerate(self.products)}
        phi = np.asarray(phi, dtype=float)
        if (phi.shape != (len(products), len(products)) or not np.isfinite(phi).all()
                or not np.allclose(phi, phi.T) or (phi < 0).any() or (phi > 1).any()):
            raise ValueError('产品空间必须有限、对称并处于[0,1]')
        self.phi = phi  # (P, P) 完整 proximity
        self.threshold = threshold
        # 预计算邻接表（阈值化后）
        adj = (phi > threshold).astype(np.uint8)
        np.fill_diagonal(adj, 0)
        # 稀疏化：每个产品只保留 top-K 邻近（按 proximity 降序）
        if max_neighbors is not None:
            adj_sparse = np.zeros_like(adj)
            for i in range(len(products)):
                nbrs = np.where(adj[i] > 0)[0]
                if len(nbrs) == 0:
                    continue
                scores = phi[i, nbrs]
                top = nbrs[np.argsort(scores)[::-1][:max_neighbors]]
                adj_sparse[i, top] = 1
            adj = np.maximum(adj_sparse, adj_sparse.T)
        self.adj = adj
        weights = phi * adj
        totals = weights.sum(axis=1)
        self.weights = csr_matrix(np.divide(weights, totals[:, None],
                                  out=np.zeros_like(weights), where=totals[:, None] > 0))
        self.neighbors = [np.where(adj[i] > 0)[0] for i in range(len(products))]
        # NetworkX 图（用于分析 / 可视化）
        self.graph = nx.from_numpy_array(adj)
        nx.relabel_nodes(self.graph, {i: p for i, p in enumerate(self.products)}, copy=False)

    def neighbor_indices(self, i: int) -> np.ndarray:
        return self.neighbors[i]

    def neighbor_products(self, product: str) -> list[str]:
        return [self.products[j] for j in self.neighbors[self.p_idx[product]]]

    def degree(self) -> np.ndarray:
        return self.adj.sum(axis=1)

    def density(self) -> float:
        P = len(self.products)
        return float(self.adj.sum() / (P * (P - 1))) if P > 1 else 0.0


# ---------- 政策模块 ----------

@dataclass
class PolicyConfig:
    """政策配置。
    - subsidy_targets: 选择性补贴目标产品列表（HS92 字符串）
    - subsidy_strength: 补贴强度乘子 [1.0, 5.0]
    - functional_boost: 功能性支持，对所有产品能力积累概率的加成 [0.0, 0.1]
    - blockade_targets: 技术封锁目标产品列表
    - blockade_strength: 封锁强度，能力衰减概率 [0.0, 0.5]
    """
    subsidy_targets: list[str] = field(default_factory=list)
    subsidy_strength: float = 1.0
    functional_boost: float = 0.0
    blockade_targets: list[str] = field(default_factory=list)
    blockade_strength: float = 0.0
    countries: list[str] = field(default_factory=list)
    start_step: int = 1
    tariff_targets: list[str] = field(default_factory=list)
    tariff_rate: float = 0.0
    tariff_elasticity: float = 1.0
    blockade_spillover: float = 0.0  # 封锁溢出系数 [0.0, 1.0]：语义邻近产品受封锁影响的比例

    def __post_init__(self):
        values = [self.subsidy_strength, self.functional_boost, self.blockade_strength,
                  self.tariff_rate, self.tariff_elasticity]
        if not np.isfinite(values).all() or min(values) < 0 or self.blockade_strength > 1:
            raise ValueError('政策参数非法')
        if self.start_step < 1:
            raise ValueError('政策最早从第1次转移开始')


class PolicyModule:
    def __init__(self, products: list[str], config: PolicyConfig,
                 resource_mask: np.ndarray | None = None,
                 semantic_phi: np.ndarray | None = None):
        self.products = products
        self.config = config
        self.p_idx = {p: i for i, p in enumerate(products)}
        P = len(products)
        # 投资乘子向量（每产品）
        self.invest_mult = np.ones(P, dtype=np.float64)
        # 封锁衰减向量
        self.decay_prob = np.zeros(P, dtype=np.float64)
        # 资源型产品掩码（这些产品不参与能力积累）
        self.resource_mask = (
            resource_mask.astype(bool) if resource_mask is not None
            else np.zeros(P, dtype=bool)
        )

        unknown = set(config.subsidy_targets + config.blockade_targets + config.tariff_targets) - set(products)
        if unknown:
            raise ValueError(f'政策产品不存在: {sorted(unknown)}')
        for p in config.tariff_targets:
            self.invest_mult[self.p_idx[p]] *= (1 + config.tariff_rate) ** (-config.tariff_elasticity)
        for p in config.subsidy_targets:
            if p in self.p_idx:
                self.invest_mult[self.p_idx[p]] *= config.subsidy_strength
        for p in config.blockade_targets:
            if p in self.p_idx:
                self.decay_prob[self.p_idx[p]] = config.blockade_strength
        # 封锁语义溢出：目标产品的封锁效果扩散到语义邻近产品
        if config.blockade_spillover > 0 and semantic_phi is not None and config.blockade_targets:
            target_mask = np.zeros(P, dtype=bool)
            for p in config.blockade_targets:
                if p in self.p_idx:
                    target_mask[self.p_idx[p]] = True
            # 每个产品受封锁溢出 = max(与目标产品的语义proximity) × spillover系数
            spillover = semantic_phi[:, target_mask].max(axis=1) * config.blockade_spillover
            # 目标产品本身不受溢出影响（已有直接封锁）
            spillover[target_mask] = 0
            self.decay_prob = np.maximum(self.decay_prob, spillover * config.blockade_strength)

    def multiplier(self) -> np.ndarray:
        return self.invest_mult

    def functional_boost(self) -> float:
        return self.config.functional_boost

    def decay(self) -> np.ndarray:
        return self.decay_prob

    def investable_mask(self) -> np.ndarray:
        """可投资产品掩码（排除资源型）。"""
        return ~self.resource_mask


# ---------- 国家智能体 ----------

class CountryAgent(mesa.Agent):
    """按国家年度同步更新二值能力代理；不模拟利润或实际贸易额。"""

    def __init__(self, model, unique_id, iso3, initial_capabilities, tech_level=0.3):
        super().__init__(model)
        self.iso3 = iso3
        self.index = model.countries.index(iso3)
        self.capabilities = np.asarray(initial_capabilities, dtype=np.uint8).copy()
        self.tech_level = float(tech_level)
        self._rng = np.random.default_rng(model.rng.integers(0, 2**63))

    def step(self):
        model = self.model
        p = len(self.capabilities)
        # 每步固定消耗随机数，配对政策实验不会因分支跳过而错位。
        selection, success, imitation_selection, decay_draw = self._rng.random((4, p))
        frac_draw, imitate_draw = self._rng.random(2)
        trend_draw = self._rng.random()
        old = model._snapshot[self.index].astype(bool)
        new = old.copy()
        cfg = model.policy.config
        active = model.steps >= cfg.start_step and (not cfg.countries or self.iso3 in cfg.countries)
        mult = model.policy.multiplier() if active else np.ones(p)
        boost = cfg.functional_boost if active else 0.
        eligible = ~old & model.policy.investable_mask()
        exposure = model._exposure[self.index]
        budget = model._budget_by_country[self.iso3]
        attempts = int(budget) + int(frac_draw < budget % 1)
        candidates = np.flatnonzero(eligible & (exposure > 0))
        # 加权无放回选择，每个预算单位最多尝试一个尚未具备的产品。
        if attempts > 0 and len(candidates):
            priorities = -np.log(np.maximum(selection[candidates], 1e-15)) / exposure[candidates]
            chosen = candidates[np.argsort(priorities)[:attempts]]
            prob = np.clip(exposure[chosen] * self.tech_level * model.learning_rate * mult[chosen] + boost, 0, 1)
            new[chosen[success[chosen] < prob]] = True
        scores = model._snapshot.sum(axis=1).astype(float) * model._tech
        scores[self.index] = -np.inf
        if len(scores) > 1 and imitate_draw < model.imitation_rate:
            leader = int(np.argmax(scores))
            gap = model._snapshot[leader].astype(bool) & eligible & ~new
            if gap.any():
                # 语义加权模仿：优先模仿与自己现有能力语义接近的产品
                if model._sem_phi is not None:
                    sem_weights = model._sem_phi[gap][:, old].max(axis=1) if old.any() else np.ones(gap.sum())
                    sem_weights = np.maximum(sem_weights, 1e-10)
                    # 用语义权重调整模仿选择概率
                    adjusted = imitation_selection[gap] / sem_weights
                    candidate = np.flatnonzero(gap)[np.argmin(adjusted)]
                else:
                    candidate = int(np.argmax(np.where(gap, imitation_selection, -1.)))
                # 模仿每步最多新增一个产品，也受资源与政策约束。
                prob = np.clip(self.tech_level * mult[candidate] + boost, 0, 1)
                if success[candidate] < prob:
                    new[candidate] = True
        policy_decay = model.policy.decay() if active else np.zeros(p)
        loss = 1 - (1 - model.depreciation_rate) * (1 - policy_decay)
        new[decay_draw < loss] = False
        # 增长趋势修正：捕捉模型未建模的国家特定动态（产业政策、制度因素等）
        trend = model._growth_trend.get(self.iso3, 0.0)
        if abs(trend) > 0:
            trend_attempts = int(abs(trend)) + (1 if trend_draw < abs(trend) % 1 else 0)
            if trend > 0:
                # 正趋势：语义引导新增能力（优先语义接近现有能力的产品）
                pool = np.flatnonzero(~new & model.policy.investable_mask())
                if trend_attempts > 0 and len(pool) > 0:
                    if model._sem_phi is not None and old.any():
                        sem_w = model._sem_phi[pool][:, old].max(axis=1)
                        sem_w = np.maximum(sem_w, 1e-10)
                        picks_idx = self._rng.choice(len(pool), size=min(trend_attempts, len(pool)),
                                                     replace=False, p=sem_w / sem_w.sum())
                        picks = pool[picks_idx]
                    else:
                        picks = self._rng.choice(pool, size=min(trend_attempts, len(pool)), replace=False)
                    new[picks] = True
            else:
                # 负趋势：语义引导移除能力（优先移除语义最边缘的产品）
                pool = np.flatnonzero(new)
                if trend_attempts > 0 and len(pool) > 0:
                    if model._sem_phi is not None and (new & ~np.zeros(p, dtype=bool)).sum() > 1:
                        other = new.copy()
                        sem_edge = np.ones(len(pool))
                        for i, pi in enumerate(pool):
                            others_mask = new.copy(); others_mask[pi] = False
                            if others_mask.any():
                                sem_edge[i] = 1.0 - model._sem_phi[pi][others_mask].max()
                        sem_edge = np.maximum(sem_edge, 1e-10)
                        picks_idx = self._rng.choice(len(pool), size=min(trend_attempts, len(pool)),
                                                     replace=False, p=sem_edge / sem_edge.sum())
                        picks = pool[picks_idx]
                    else:
                        picks = self._rng.choice(pool, size=min(trend_attempts, len(pool)), replace=False)
                    new[picks] = False
        self.capabilities = new.astype(np.uint8)
        # ECI映射技术代理保持基年值，避免无数据支撑的自动技术增长。



# ---------- 主模型 ----------

class ComplexityABM(mesa.Model):
    """经济复杂度演化 ABM（国家即智能体）。"""

    def __init__(
        self,
        product_space: ProductSpace,
        initial_capabilities: dict[str, np.ndarray],
        initial_tech: dict[str, float] | None = None,
        policy_config: PolicyConfig | None = None,
        learning_rate: float = 0.05,
        imitation_rate: float = 0.1,
        investment_budget: float | dict[str, float] = 5.0,
        depreciation_rate: float = 0.005,
        resource_mask: np.ndarray | None = None,
        growth_trend: dict[str, float] | None = None,
        semantic_phi: np.ndarray | None = None,
        seed: int = 42,
    ):
        super().__init__(rng=seed)
        rates = np.array([learning_rate, imitation_rate, depreciation_rate], dtype=float)
        if not np.isfinite(rates).all() or (rates < 0).any() or (rates > 1).any():
            raise ValueError('转移概率必须在[0,1]内')
        self.countries = sorted(initial_capabilities)
        if not self.countries:
            raise ValueError('至少需要一个智能体')
        if policy_config and set(policy_config.countries) - set(self.countries):
            raise ValueError('政策国家不在样本内')
        self.product_space = product_space
        self.learning_rate = learning_rate
        self.imitation_rate = imitation_rate
        self.depreciation_rate = depreciation_rate
        # 投资预算：支持浮点标量（同质）或 dict（异质性，按国家）
        if isinstance(investment_budget, dict):
            self._budget_by_country = investment_budget
            self.investment_budget = 0.0  # 占位，实际由智能体读取 _budget_by_country
        else:
            self._budget_by_country = {
                iso3: float(investment_budget) for iso3 in initial_capabilities
            }
            self.investment_budget = float(investment_budget)
        if set(self._budget_by_country) != set(initial_capabilities):
            raise ValueError('预算国家必须与样本一致')
        values = np.array(list(self._budget_by_country.values()))
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('预算不能缺失或为负')
        if resource_mask is not None and np.asarray(resource_mask).shape != (len(product_space.products),):
            raise ValueError('资源掩码维度不匹配')
        self.policy = PolicyModule(
            product_space.products,
            policy_config or PolicyConfig(),
            resource_mask=resource_mask,
            semantic_phi=semantic_phi,
        )
        self._history: list[dict[str, Any]] = []
        # 记录初始能力（用于防止退化到比初始还差）
        self._initial_caps = {iso3: caps.copy() for iso3, caps in initial_capabilities.items()}
        # 国家增长趋势（从历史数据计算的年均多样性变化）
        self._growth_trend = growth_trend or {}
        # 语义 proximity 矩阵（用于语义加权模仿、语义引导趋势、封锁溢出）
        self._sem_phi = semantic_phi

        # 创建智能体
        for iso3 in self.countries:
            caps = np.asarray(initial_capabilities[iso3])
            tech = (initial_tech or {}).get(iso3, 0.3)
            if caps.shape != (len(product_space.products),) or not np.isin(caps, [0, 1]).all():
                raise ValueError('初始能力必须为维度匹配的二值向量')
            if not np.isfinite(tech) or not 0 <= tech <= 1:
                raise ValueError('初始技术代理缺失或超界')
            CountryAgent(self, iso3, iso3, caps, tech)
        self._tech = np.array([a.tech_level for a in self.agents])
        self._record()

    def step(self):
        self._snapshot = np.array([a.capabilities for a in self.agents])
        self._exposure = self.product_space.weights.dot(self._snapshot.T).T
        self.agents.do("step")
        self._record()

    def _record(self):
        """记录宏观统计。"""
        caps = np.array([a.capabilities for a in self.agents])  # (C, P)
        techs = np.array([a.tech_level for a in self.agents])   # (C,)
        iso3s = [a.iso3 for a in self.agents]
        diversity = caps.sum(axis=1)
        record = {
            "step": self.steps,
            "mean_diversity": float(diversity.mean()),
            "std_diversity": float(diversity.std()),
            "mean_tech": float(techs.mean()),
            "total_capabilities": int(caps.sum()),
        }
        # 每国多样性
        for iso3, d in zip(iso3s, diversity):
            record[f"div_{iso3}"] = int(d)
        for iso3, t in zip(iso3s, techs):
            record[f"tech_{iso3}"] = float(t)
        self._history.append(record)

    def history(self) -> pd.DataFrame:
        return pd.DataFrame(self._history)


# ---------- 加载数据 ----------

def load_processed():
    """加载管线输出。"""
    countries = pd.read_parquet(PROC / "country_metadata.parquet")["country_iso3"].tolist()
    products = pd.read_parquet(PROC / "product_metadata.parquet")["product_hs92"].tolist()
    M_long = pd.read_parquet(PROC / "M_matrix.parquet")
    phi_full = pd.read_parquet(PSPACE / "proximity_full.parquet")
    phi = phi_full.loc[products, products].values
    return countries, products, M_long, phi


def initial_capabilities_from_year(M_long: pd.DataFrame, countries: list[str],
                                    products: list[str], year: int) -> dict[str, np.ndarray]:
    """从指定年份的 M 矩阵提取每国初始能力向量。"""
    sub = M_long[M_long["year"] == year]
    piv = sub.pivot_table(
        index="country_iso3", columns="product_hs92",
        values="capability", aggfunc="max",
    ).reindex(index=countries, columns=products)
    if piv.isna().any().any():
        raise ValueError('初始化能力矩阵不完整')
    caps = {iso3: piv.loc[iso3].values.astype(np.uint8) for iso3 in countries}
    return caps


def initial_tech_from_eci(countries: list[str], year: int = 2000) -> dict[str, float]:
    """用官方 ECI（首年）做技术水平初值的单调映射。"""
    cy = pd.read_csv(ROOT / "data" / "raw" / "harvard_atlas" / "hs92_country_year.csv")
    first_year = year
    sub = cy[cy["year"] == first_year][["country_iso3_code", "eci"]].copy()
    sub = sub[sub["country_iso3_code"].isin(countries)]
    # 归一化到 [0.1, 0.9]
    eci = sub.set_index("country_iso3_code")["eci"]
    eci_min, eci_max = eci.min(), eci.max()
    if eci_max > eci_min:
        norm = (eci - eci_min) / (eci_max - eci_min)
        tech = 0.1 + 0.8 * norm
    else:
        tech = pd.Series(0.3, index=eci.index)
    tech = tech.reindex(countries)
    # 缺 ECI 的国家用中位数填充（通常是新成立或极小经济体）
    if tech.isna().any():
        median_val = tech.median()
        missing = tech[tech.isna()].index.tolist()
        tech = tech.fillna(median_val)
        print(f"  注意：{len(missing)} 国缺 ECI，用中位数填充: {missing}")
    return tech.to_dict()
