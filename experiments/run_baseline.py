"""
运行 baseline 实验并生成可视化 + 方法文档。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import (
    ComplexityABM, ProductSpace, PolicyConfig,
    load_processed, initial_capabilities_from_year, initial_tech_from_eci,
)
import sys
sys.path.insert(0, str(ROOT / "src"))
from pipeline import resource_product_mask, investment_budgets_by_gdp

DOCS = ROOT / "docs"
FIGS = DOCS / "figures"
DOCS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)


# ---------- 主流程 ----------

def run_baseline(years: int = 24, seed: int = 42,
                 learning_rate: float = 0.05,
                 imitation_rate: float = 0.1,
                 investment_budget: float | dict[str, float] = 5.0,
                 depreciation_rate: float = 0.005,
                 max_neighbors: int = 30,
                 use_resource_mask: bool = True,
                 use_gdp_budget: bool = True) -> pd.DataFrame:
    print("加载管线输出 ...")
    countries, products, M_long, phi = load_processed()
    print(f"  国家 {len(countries)}  产品 {len(products)}")

    space = ProductSpace(products, phi, threshold=0.5, max_neighbors=max_neighbors)
    print(f"  产品空间密度：{space.density():.4f}  平均度：{space.degree().mean():.1f}")

    init_caps = initial_capabilities_from_year(M_long, countries, products, year=2000)
    init_tech = initial_tech_from_eci(countries)

    # 资源型产品掩码
    res_mask = resource_product_mask(products) if use_resource_mask else None
    print(f"  资源型产品数：{res_mask.sum() if res_mask is not None else 0}")

    # 投资预算：同质 vs GDP 异质性
    if use_gdp_budget:
        budgets = investment_budgets_by_gdp(countries, budget_min=1.0, budget_max=15.0)
        print(f"  投资预算：GDP 异质性缩放 [1, 15]")
    else:
        budgets = investment_budget
        print(f"  投资预算：同质 = {investment_budget}")

    model = ComplexityABM(
        product_space=space,
        initial_capabilities=init_caps,
        initial_tech=init_tech,
        policy_config=PolicyConfig(),
        learning_rate=learning_rate,
        imitation_rate=imitation_rate,
        investment_budget=budgets,
        depreciation_rate=depreciation_rate,
        resource_mask=res_mask,
        seed=seed,
    )

    print(f"运行 {years} 步 baseline ...")
    for _ in range(years):
        model.step()

    hist = model.history()
    hist.to_parquet(ROOT / "experiments" / "baseline_history.parquet", index=False)
    print(f"  完成。末步平均多样性 {hist['mean_diversity'].iloc[-1]:.1f}")
    return hist


def plot_evolution(hist: pd.DataFrame, countries: list[str]):
    """绘制平均多样性、总能力、代表性国家多样性演化。"""
    steps = hist["step"].values

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, hist["mean_diversity"], lw=2, label="mean diversity (all)")
    ax.fill_between(
        steps,
        hist["mean_diversity"] - hist["std_diversity"],
        hist["mean_diversity"] + hist["std_diversity"],
        alpha=0.2, label="±1 std",
    )
    ax.set_xlabel("step")
    ax.set_ylabel("diversity (# products with RCA>1)")
    ax.set_title("Baseline ABM: mean export diversity over time")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "baseline_mean_diversity.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, hist["total_capabilities"], lw=2, color="tab:green")
    ax.set_xlabel("step")
    ax.set_ylabel("total capabilities (sum over countries)")
    ax.set_title("Baseline ABM: total capabilities")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "baseline_total_capabilities.png", dpi=140)
    plt.close(fig)

    # 代表性国家
    reps = ["CHN", "JPN", "KOR", "VNM", "IND", "IDN"]
    reps = [c for c in reps if c in countries]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c in reps:
        col = f"div_{c}"
        if col in hist.columns:
            ax.plot(steps, hist[col], lw=1.5, label=c)
    ax.set_xlabel("step")
    ax.set_ylabel("diversity")
    ax.set_title("Selected countries' export diversity (baseline)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "baseline_selected_countries.png", dpi=140)
    plt.close(fig)

    # 技术水平演化
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, hist["mean_tech"], lw=2, color="tab:purple")
    ax.set_xlabel("step")
    ax.set_ylabel("mean tech_level")
    ax.set_title("Baseline ABM: mean technology level")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "baseline_mean_tech.png", dpi=140)
    plt.close(fig)


def compare_to_history(hist: pd.DataFrame, M_long: pd.DataFrame,
                       countries: list[str]) -> pd.DataFrame:
    """对比模型末步与历史 2023 年的国家多样性。"""
    hist_year = M_long[M_long["year"] == 2023].groupby("country_iso3")["capability"].sum()
    model_last = hist.iloc[-1]
    rows = []
    for c in countries:
        rows.append({
            "country_iso3": c,
            "historical_2023_diversity": int(hist_year.get(c, 0)),
            "model_final_diversity": int(model_last.get(f"div_{c}", 0)),
        })
    df = pd.DataFrame(rows)
    df["diff"] = df["model_final_diversity"] - df["historical_2023_diversity"]
    df = df.sort_values("historical_2023_diversity", ascending=False)
    df.to_csv(ROOT / "experiments" / "baseline_vs_history.csv", index=False)
    return df


def write_method_doc(summary: dict):
    """沉淀方法文档到 docs/。"""
    doc = DOCS / "国家智能体ABM方法说明.md"
    content = f"""# 国家智能体 ABM 方法说明

> 基于已下载的 Harvard HS92 四位码数据，采用"国家即智能体"的简化方案。
> 每个国家智能体的能力向量 = 该国在各产品上的 RCA 二值状态（RCA > 1）。

## 1. 数据范围

- **国家子集**：25 个亚太经济体（ASEAN10 + China + JP + KR + TW + HK +
  IN/PAK/BD/LK + AU/NZ + PNG/FJI/MNG）
- **产品层级**：HS92 四位码（标准 4 位数字，剔除 `XXXX` 等非标准代码）
- **时间窗口**：2000–2023（初始化用 2000 年）
- **数据来源**：Harvard Atlas HS92 `hs92_country_product_year_4.csv`

## 2. 关键计算

### 2.1 RCA（显示性比较优势）

$$
RCA_{{c,p,t}} = \\frac{{X_{{c,p,t}} / X_{{c,*,t}}}}{{X_{{*,p,t}} / X_{{*,*,t}}}}
$$

能力二值化：$M_{{c,p,t}} = 1$ iff $RCA_{{c,p,t}} > 1$。

### 2.2 产品空间 proximity

$$
\\varphi(i, j) = \\min\\bigl(P(RCA_i \\mid RCA_j),\\; P(RCA_j \\mid RCA_i)\\bigr)
$$

使用全球（亚太子集）M 矩阵的"至少一年有 RCA"作为国家能力，
通过矩阵运算 `cooc = M.T @ M` 一次计算所有产品对的共现。
阈值 0.5 以上连边。

### 2.3 智能体步进步机制

每步对每个国家智能体：

1. **投资**：对每个有能力的产品 $s$，采样其邻近产品 $j$，以概率
   $P = \\varphi(s,j) \\cdot \\text{{tech}} \\cdot \\text{{lr}} \\cdot \\mu_j + b$
   积累新能力（$\\mu_j$ 为政策乘子，$b$ 为功能性支持）。
2. **模仿**：以 `imitation_rate` 概率观察利润最高（能力数 × tech）的国家，
   按其相对成功度复制部分能力。
3. **政策衰减**：技术封锁目标产品以 `blockade_strength` 概率丢失能力。
4. **技术水平演化**：$\\text{{tech}} \\mathrel{{+}}= 0.002 \\ln(1 + \\text{{diversity}}) - 0.001$。

## 3. Baseline 参数

| 参数 | 值 |
|---|---|
| 学习率 `learning_rate` | {summary.get("learning_rate", 0.05)} |
| 模仿率 `imitation_rate` | {summary.get("imitation_rate", 0.1)} |
| 模拟步数 | {summary.get("n_steps", 24)} |
| 初始化年份 | 2000 |
| 技术水平初值 | 由 2000 年官方 ECI 线性归一化到 [0.1, 0.9] |
| 政策 | 无（neutral baseline） |

## 4. 输出

- `experiments/baseline_history.parquet`：每步宏观统计 + 每国多样性/tech
- `experiments/baseline_vs_history.csv`：模型末步 vs 历史 2023 年国家多样性对比
- `docs/figures/baseline_*.png`：演化曲线
- `docs/国家智能体ABM结果分析.md`：结果解读（运行后生成）

## 5. 已知局限

1. **国家即智能体** 抹平了企业异质性，无法模拟企业进入/退出/多产品结构。
2. **能力积累** 仅依赖产品空间邻近度，未建模研发成本、FDI、人力资本。
3. **模仿机制** 高度简化，未区分地理/文化距离。
4. **政策模块** 仅影响概率乘子，未建模财政约束、一般均衡效应。
5. **校准** 未做 ABC-SMC，参数为经验值；后续可用历史矩匹配精调。

## 6. 复现

```bash
python experiments/run_baseline.py
```

运行环境：Python 3.14、pandas 3.x、numpy 2.x、networkx 3.x、mesa 3.x、matplotlib 3.x。
"""
    doc.write_text(content, encoding="utf-8")
    print(f"方法文档：{doc}")
    return doc


def main():
    hist = run_baseline(
        years=24, seed=42,
        learning_rate=0.05, imitation_rate=0.1,
        depreciation_rate=0.005,
        max_neighbors=30,
        use_resource_mask=True,
        use_gdp_budget=True,
    )

    print("加载历史 M 矩阵用于对比 ...")
    countries, products, M_long, _ = load_processed()

    print("绘制演化图 ...")
    plot_evolution(hist, countries)

    print("对比历史 ...")
    compare_to_history(hist, M_long, countries)

    print("写方法文档 ...")
    write_method_doc({
        "learning_rate": 0.05,
        "imitation_rate": 0.1,
        "n_steps": 24,
    })

    print("完成。")


if __name__ == "__main__":
    main()
