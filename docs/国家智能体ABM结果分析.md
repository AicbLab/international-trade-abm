# 国家智能体 ABM · 实验结果分析

> 运行环境：Python 3.14 · mesa 3.5.1 · pandas 3.0.3 · numpy 2.4.3 · networkx 3.6.1
> 运行命令：`python experiments/run_calibration.py && python experiments/run_policy_experiments.py`

## 1. 实验设置

| 项 | 值 |
|---|---|
| 国家子集 | 24 个亚太经济体 |
| 产品层级 | HS92 四位码 |
| 产品数 | 1242 |
| 初始化年份 | 2000 |
| 模拟步数 | 24（对应 2000 → 2024） |
| 投资预算 | GDP 异质缩放 [1, 15] |
| 资源掩码 | HS25/26 + 2701-2709（保守矿产代理） |
| 产品空间稀疏化 | top-30 邻近 |
| 技术水平 | 固定（ECI 初值，不自动增长） |
| 折旧 | 对所有能力一视同仁（不保护初始能力） |

## 2. ABC-SMC 校准结果

### 2.1 后验参数

| 参数 | 默认值 | ABC-SMC 校准值 | 先验范围 |
|---|---|---|---|
| learning_rate | 0.05 | **0.459** | [0.01, 1.0] |
| imitation_rate | 0.1 | **0.249** | [0.0, 0.5] |
| depreciation_rate | 0.005 | **0.015** | [0.0, 0.04] |

**观察**：校准参数显著高于默认值。更高的学习率和模仿率意味着能力积累更快，
更高的折旧率意味着能力丢失也更快。这反映了模型在训练期内对"快速饱和"模式的拟合。

### 2.2 校准收敛

| 代 | epsilon | ESS | 尝试次数 |
|---:|---:|---:|---:|
| 0 | 3.3646 | 50.0 | 50 |
| 1 | 3.3408 | 47.5 | 54 |
| 2 | 3.3241 | 49.4 | 63 |
| 3 | 3.3120 | 49.3 | 56 |
| 4 | 3.2940 | 49.2 | 64 |

**观察**：epsilon 从 3.36 下降到 3.29（约 2% 收缩），ESS 稳定在 47-50。
在 144 维目标空间中，50 粒子的后验收缩有限——这是"维数灾难"的典型表现。
增加粒子数到 200+ 可能改善收敛，但计算成本显著增加。

### 2.3 留出验证（2016–2023）

| 指标 | 校准参数 | 默认参数 | 更优 |
|---|---:|---:|---|
| 留出 MAE | 53.66 | **49.10** | 默认 |
| 留出 RMSE | 65.22 | **59.76** | 默认 |
| 最终 MAE | 63.12 | **57.96** | 默认 |
| 最终 RMSE | 76.81 | **69.60** | 默认 |
| 区域 ECI MAE | 0.38 | **0.35** | 默认 |
| 最终平均多样性 | 169.1 | 182.9 | — |
| 高估国家数 | 9 | **8** | 默认 |

**结论**：默认参数在留出期表现优于校准参数。
校准参数可能过拟合了训练期（2005/2010/2015）的 summary statistics，
导致更高的学习/模仿率使模型在留出期过度积累能力。

**后续建议**：
- 增加粒子数（200-500）和世代数（10-20）以改善后验近似
- 使用正则化（如增大 scales）防止过拟合训练期
- 考虑_leave-one-year-out_ 交叉验证替代固定训练/留出生

## 3. 政策配对实验结果

### 3.1 实验 A：定向补贴 CHN → 8542（集成电路）

| 项 | 值 |
|---|---|
| 政策 | 补贴强度 ×3.0，从 step 6 开始 |
| CHN 多样性差异 | **+1** |

**解读**：补贴使 CHN 在 8542（集成电路）上的投资优先级提高 3 倍。
在高折旧率（0.015）下，额外获取的能力在 24 步净效果仅为 +1。
这说明在高折旧环境中，补贴政策的效果被快速折旧抵消。

### 3.2 实验 B：技术封锁 CHN → 8542, 8471

| 项 | 值 |
|---|---|
| 政策 | 封锁强度 0.3（每步 30% 概率丢失），从 step 6 开始 |
| CHN 多样性差异 | **-2** |

**解读**：封锁使目标产品（半导体 + 计算机）每步有额外 30% 概率丢失能力。
19 步（step 6–24）累积效果使 CHN 净损失 2 个产品。
效果有限因为：(1) CHN 可能原本在这些产品上无 RCA；(2) 高折旧使"自然丢失"已较多。

### 3.3 实验 C：关税保护 VNM/IDN/BGD → 劳动密集型产品

| 国家 | 多样性差异 |
|---|---|
| VNM | +0 |
| IDN | +0 |
| BGD | +0 |

**解读**：关税（25% 从价税，弹性 1.0）降低了目标产品（T恤、女装等）的投资乘子
约 21%。但由于这些产品在国家能力边缘（exposure 低），
关税对整体多样性无显著影响。

### 3.4 政策实验小结

| 实验 | 效果 | 原因 |
|---|---|---|
| 补贴 | 微弱正面 (+1) | 高折旧抵消补贴收益 |
| 封锁 | 微弱负面 (-2) | 目标产品可能本就不在 RCA 边界 |
| 关税 | 无效果 (0) | 目标产品处于能力边缘，关税无作用点 |

**关键洞察**：在 ABC-SMC 校准的高折旧参数下，政策效果被快速折旧稀释。
使用默认参数（折旧 0.005）可能产生更显著的政策差异。

## 4. 回归测试覆盖

共 22 个测试全部通过：

| 模块 | 测试数 | 覆盖内容 |
|---|---:|---|
| test_research.py (Pipeline) | 5 | proximity 对称性、uint8 溢出、资源代理保守性、主键唯一、全球参照 |
| test_research.py (Model) | 11 | 种子复现、初始状态、稀疏图对称、预算约束、折旧、资源掩码、政策作用域、异质预算、关税乘子、政策复现 |
| test_calibration.py | 6 | ABC-SMC 复现+加权、重要性权重、无效 scale 拒绝、耗尽标记、训练摘要隔离、ECI 有限性 |

## 5. 产物清单

```
data/processed/
  rca_matrix.parquet          — 连续 RCA（C × P × Y）
  M_matrix.parquet            — 二值能力矩阵
  country_metadata.parquet    — 国家元数据
  product_metadata.parquet    — 产品元数据
  country_budget_inputs.csv   — GDP/出口协变量
  resource_proxy.csv          — 资源型产品标记

data/product_space/
  proximity.parquet           — 阈值化邻接（1242 × 1242）
  proximity_full.parquet      — 完整 proximity

experiments/
  calibration_result.json     — ABC-SMC 后验粒子与权重
  calibration_history.csv     — 逐代 epsilon、ESS
  calibration_summary.json    — 后验参数 + 留出期指标
  policy_country_results.csv  — 逐国政策效果
  policy_timeseries_results.csv — 逐时间步政策对比
  policy_summary.json         — 政策实验配置

docs/
  国家智能体ABM方法说明.md    — 方法文档
  国家智能体ABM结果分析.md    — 本文档
  figures/baseline_*.png      — 基线演化曲线（4 张）
```

## 6. 复现

```bash
# 完整复现流程
python scripts/prepare_covariates.py   # 协变量
python -m pytest tests/ -v             # 回归测试
python experiments/run_calibration.py  # ABC-SMC 校准
python experiments/run_policy_experiments.py  # 政策实验
python experiments/run_baseline.py     # 基线 + 可视化
```
