# 国家智能体 ABM 方法说明

> 基于已下载的 Harvard HS92 四位码数据，采用"国家即智能体"的简化方案。
> 每个国家智能体的能力向量 = 该国在各产品上的 RCA 二值状态（RCA > 1）。

## 1. 数据范围

- **国家子集**：24 个亚太经济体（ASEAN10 + China + JP + KR + TW + HK +
  IN/PAK/BD/LK + AU/NZ + PNG/FJI/MNG）
- **产品层级**：HS92 四位码（标准 4 位数字，剔除 `XXXX` 等非标准代码）
- **时间窗口**：2000–2023（初始化用 2000 年）
- **数据来源**：Harvard Atlas HS92 `hs92_country_product_year_4.csv`

## 2. 关键计算

### 2.1 RCA（显示性比较优势）

$$
RCA_{c,p,t} = \frac{X_{c,p,t} / X_{c,*,t}}{X_{*,p,t} / X_{*,*,t}}
$$

能力二值化：$M_{c,p,t} = 1$ iff $RCA_{c,p,t} > 1$。

RCA 的世界参照使用**全球** 232 国数据（而非亚太子集），避免区域子集抬高阈值。

### 2.2 产品空间 proximity

$$
\varphi(i, j) = \min\bigl(P(RCA_i \mid RCA_j),\; P(RCA_j \mid RCA_i)\bigr)
$$

使用全球（亚太子集）M 矩阵的"基年至少一年有 RCA"作为国家能力，
通过矩阵运算 `cooc = M.T @ M` 一次计算所有产品对的共现。
阈值 0.5 以上连边，再稀疏化为 top-30 邻近。

### 2.3 智能体步进机制

每步对每个国家智能体：

1. **投资**：以当前能力产品为种子，按产品空间邻近度加权无放回选择
   尚未具备的产品，以概率
   $P = \varphi(s,j) \cdot \text{tech} \cdot \text{lr} \cdot \mu_j + b$
   积累新能力（$\mu_j$ 为政策乘子，$b$ 为功能性支持）。
   每步投资次数受预算约束（浮点预算按概率拆分为整数次尝试）。
2. **模仿**：以 `imitation_rate` 概率观察利润最高（能力数 × tech）的国家，
   按其相对成功度复制部分能力（每步最多新增一个产品，受资源掩码约束）。
3. **折旧**：每个能力以 `depreciation_rate` 概率丢失（对所有能力一视同仁，
   不保护初始能力）。
4. **政策衰减**：技术封锁目标产品以 `blockade_strength` 概率额外丢失能力。

**注意**：技术水平在模拟中保持基年值不变（ECI 映射仅用于初值），
不建模自动技术增长。

### 2.4 异质性投资预算

投资预算按各国 2000 年经济规模缩放：

- 有 WDI GDP 数据的 23 国：用 $\ln(\text{GDP})$ 归一化到 $[0, 1]$
- 缺 GDP 的 TWN：用 $\ln(\text{出口额})$ 归一化（明确标记为代理，非 GDP）
- 映射到预算区间 $[1, 15]$：$\text{budget}_c = 1 + 14 \times \text{scale}_c$

数据来源：World Bank WDI `NY.GDP.MKTP.CD`（2000 年），
缺值回退规则记录在 `data/metadata/covariates_manifest.json`。

### 2.5 保守资源分类

资源型产品代理仅覆盖 HS 章节 25（盐、硫）、26（矿石）和 2701–2709（煤、石油、天然气）。
不排除化学品（28-38）、棉纺（52）或食品（01-24）。
资源型产品不参与能力积累（投资与模仿均跳过）。

## 3. Baseline 参数

| 参数 | 默认值 | ABC-SMC 校准值 |
|---|---|---|
| 学习率 `learning_rate` | 0.05 | 0.459 |
| 模仿率 `imitation_rate` | 0.1 | 0.249 |
| 折旧率 `depreciation_rate` | 0.005 | 0.015 |
| 投资预算 `investment_budget` | GDP 异质 [1, 15] | 同左 |
| 产品空间稀疏化 `max_neighbors` | 30 | 同左 |
| 模拟步数 | 24 | 同左 |
| 初始化年份 | 2000 | 同左 |
| 技术水平初值 | ECI 归一化 [0.1, 0.9] | 同左 |
| 政策 | 无（neutral baseline） | 同左 |

## 4. ABC-SMC 校准

### 4.1 方法

采用加权 ABC 序列蒙特卡洛（Toni et al. 2010），均匀先验：

| 参数 | 下界 | 上界 |
|---|---|---|
| learning_rate | 0.01 | 1.0 |
| imitation_rate | 0.0 | 0.5 |
| depreciation_rate | 0.0 | 0.04 |

**目标统计**：训练年份（2005/2010/2015）的逐国多样性 + 区域 ECI，
共 $3 \times 2 \times 24 = 144$ 维。

**距离度量**：归一化 L2 距离，尺度为各维度的 $\max(\text{diversity}, 50)$ 或 2.0（ECI）。

### 4.2 运行配置

- 粒子数：50
- 世代数：5
- 重复模拟：2（降噪）
- 随机种子：42

### 4.3 训练/留出分离

- **训练期**：2000–2015（初始化 + 15 步），匹配 summary statistics
- **留出期**：2016–2023（8 步），仅用于评估泛化

## 5. 政策实验

三组配对实验（baseline vs 政策干预），使用相同 seed 和参数，仅政策不同：

| 实验 | 政策类型 | 目标国家 | 目标产品 | 参数 |
|---|---|---|---|---|
| A | 定向补贴 | CHN | 8542（集成电路） | strength=3.0, start=step 6 |
| B | 技术封锁 | CHN | 8542, 8471 | strength=0.3, start=step 6 |
| C | 关税 | VNM, IDN, BGD | 6109, 6204, 6110, 6104 | rate=0.25, elasticity=1.0 |

## 6. 输出

### 基线实验
- `experiments/baseline_history.parquet`：每步宏观统计 + 每国多样性/tech
- `experiments/baseline_vs_history.csv`：模型末步 vs 历史 2023 对比
- `docs/figures/baseline_*.png`：演化曲线

### 校准实验
- `experiments/calibration_result.json`：后验粒子与权重
- `experiments/calibration_history.csv`：逐代 epsilon、ESS
- `experiments/calibration_summary.json`：后验参数 + 留出期指标

### 政策实验
- `experiments/policy_country_results.csv`：逐国多样性差异
- `experiments/policy_timeseries_results.csv`：逐时间步对比
- `experiments/policy_summary.json`：实验配置

## 7. 已知局限

1. **国家即智能体** 抹平了企业异质性，无法模拟企业进入/退出/多产品结构。
2. **能力积累** 仅依赖产品空间邻近度，未建模研发成本、FDI、人力资本。
3. **模仿机制** 高度简化，未区分地理/文化距离。
4. **政策模块** 仅影响概率乘子，未建模财政约束、一般均衡效应。
5. **RCA 直接等同于能力**，对转口贸易/加工贸易/资源型 RCA 有系统性偏差。
6. **ABC-SMC 校准** 受限于 50 粒子和 144 维目标，后验收缩有限；
   默认参数在留出期表现略优于校准参数（可能过拟合训练期）。
7. **折旧对所有能力一视同仁**（不保护初始能力），
   模型对"失去比较优势"的历史事件敏感。

## 8. 复现

```bash
# 1. 数据管线（已运行，产物在 data/processed/）
python src/pipeline.py

# 2. 协变量准备
python scripts/prepare_covariates.py

# 3. 回归测试
python -m pytest tests/ -v

# 4. ABC-SMC 校准
python experiments/run_calibration.py

# 5. 政策实验
python experiments/run_policy_experiments.py

# 6. 基线实验（含可视化）
python experiments/run_baseline.py
```

运行环境：Python 3.14、pandas 3.x、numpy 2.x、networkx 3.x、mesa 3.x、matplotlib 3.x、scipy 1.x。
