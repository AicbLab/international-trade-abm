# 基于智能体建模的国际贸易经济复杂度演化研究

Research Proposal

*Agent-Based Modeling of Economic Complexity Evolution in International Trade*

- 研究方向：计算经济学 / 国际贸易 / 经济复杂度
- 方法：Agent-Based Modeling (ABM) + 产品空间网络 + 演化博弈
- 日期：2026 年 9 月

## 目录

1. [研究背景与问题提出](#一研究背景与问题提出)
2. [文献综述与研究空白](#二文献综述与研究空白)
3. [研究问题与假说](#三研究问题与假说)
4. [模型设计](#四模型设计)
5. [数据来源与处理](#五数据来源与处理)
6. [技术实现方案](#六技术实现方案)
7. [实验设计](#七实验设计)
8. [模型验证策略](#八模型验证策略)
9. [预期贡献与创新点](#九预期贡献与创新点)
10. [可行性评估](#十可行性评估)
11. [时间线与工作量](#十一时间线与工作量)
12. [目标期刊与发表策略](#十二目标期刊与发表策略)
13. [风险与应对](#十三风险与应对)
14. [参考文献](#十四参考文献)

## 一、研究背景与问题提出

### 1.1 现实背景

当前全球贸易格局正经历深刻变革：中美贸易摩擦持续升级、技术封锁与出口管制并行、区域贸易协定重组全球价值链。各国纷纷通过产业政策寻求出口结构升级和经济复杂度跃迁。然而，传统计量方法难以回答一个核心问题：**在异质性企业互动、政策干预与网络外部性交织下，一国出口复杂度是如何"涌现"出来的？**

### 1.2 理论背景

经济复杂度理论（Hausmann & Hidalgo, 2011）揭示了一个重要规律：一国出口产品的多样性和复杂性与其经济增长高度相关。该理论的核心概念包括：

- **产品空间（Product Space）**：所有产品构成的网络，产品间的"距离"由共出口关系决定
- **经济复杂度指数（ECI）**：衡量一国出口篮子的复杂程度
- **产品复杂度指数（PCI）**：衡量单个产品的技术门槛
- **生产能力（Productive Capabilities）**：企业生产特定产品所需的知识、技术、制度等要素组合

### 1.3 方法论困境

> **核心矛盾**
>
> 经济复杂度理论本质上是动态演化理论，但现有研究方法几乎全是静态计量。Chudziak (2025, JEIC) 明确指出：*"一些现象只有在更高复杂度的系统中才可能涌现"*，传统方法"过度简化了对数据矩的校准"，无法揭示反馈循环和涌现机制。

这正是 ABM 方法的用武之地：通过模拟异质性企业的微观决策，观察宏观复杂度指标的涌现过程。

## 二、文献综述与研究空白

### 2.1 经济复杂度研究现状

| 研究方向 | 代表文献 | 方法 | 局限 |
| --- | --- | --- | --- |
| 复杂度与增长 | Hausmann et al. (2007); Hidalgo et al. (2007) | 面板回归、工具变量 | 只能回答"相关于什么"，不能回答"如何演化" |
| 产品空间网络 | Hidalgo et al. (2007); Neffke et al. (2011) | 网络分析、可视化 | 描述性的，不解释动态过程 |
| 产业政策评估 | Harvard Growth Lab (2024); SSRN (2024) | DID、合成控制法 | 无法做反事实模拟 |
| 区域复杂度动态 | Multiple (2022-2024) | 系统综述、面板 ARDL | 忽略企业异质性 |

### 2.2 ABM 在经济领域的应用

| 方向 | 代表文献 | 与本研究的关系 |
| --- | --- | --- |
| 经济复杂度 ABM（呼吁） | **Chudziak (2025, JEIC)** | 直接先驱，提出方向但未实现 |
| 全球贸易 ABM 校准 | Li et al. (2025, JASSS) | 提供校准方法论（ABC-SMC） |
| 创新与扩散 ABM | Ponta et al. (2025); 多篇 | 提供技术扩散机制设计参考 |
| LLM + ABM | Wu et al. (2026, MALLES); Nature HSSC (2024) | 前沿方法，可用于增强企业决策真实性 |
| 供应链金融 ABM | Multiple (2024) | 提供企业异质性建模参考 |

### 2.3 明确的研究空白

> **三个无人区**
>
> 1. **没有人在 ABM 中实现过完整的经济复杂度演化过程**——企业如何积累能力、如何在产品空间中移动、如何涌现出国家层面的复杂度
> 2. **产业政策对复杂度演化的影响**只有计量研究，缺乏"如果...会怎样"的反事实模拟
> 3. **企业异质性**在贸易复杂度研究中被完全忽略——现有研究把国家当作同质整体

## 三、研究问题与假说

### 3.1 核心研究问题

> **总问题**
>
> **产业政策如何通过异质性企业的互动行为影响一国出口经济复杂度的演化路径？**

### 3.2 子问题

1. **涌现机制**：异质性企业的能力积累与产品选择行为，如何在宏观层面涌现出经济复杂度的演化轨迹？
2. **政策效应**：不同类型的产业政策（选择性补贴 vs. 功能性支持 vs. 贸易保护）对复杂度跃迁路径有何差异化影响？
3. **网络效应**：产品空间网络的结构特征（密度、聚类、核心-边缘）如何调节政策效果？
4. **外部冲击**：面对关税冲击或技术封锁，不同复杂度水平的经济体表现出怎样的韧性差异？

### 3.3 研究假说

| 编号 | 假说 | 理论依据 |
| --- | --- | --- |
| H1 | 企业能力积累的异质性是复杂度非线性跃迁的必要条件 | 演化经济学；能力基础理论 |
| H2 | 针对"邻近高复杂度产品"的定向补贴比普惠补贴更有效推动复杂度升级 | 产品空间理论（Hausmann 2007） |
| H3 | 产品空间网络密度适中的经济体，政策干预效果最好（倒 U 型关系） | 网络科学；复杂度阈值理论 |
| H4 | 初始复杂度较高的经济体对贸易冲击的韧性更强，但恢复路径依赖历史锁定效应 | 路径依赖理论；韧性经济学 |

## 四、模型设计

### 4.1 模型总体架构

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        环 境 层 (Environment)                        │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ 产品空间网络  │  │  政策模块     │  │  市场模块     │              │
│  │ (Product     │  │  (Policy     │  │  (Market     │              │
│  │  Space Net)  │  │   Module)    │  │   Module)    │              │
│  │              │  │              │  │              │              │
│  │ · 节点=产品  │  │ · 补贴规则   │  │ · 需求形成   │              │
│  │ · 边=proximity│ │ · 关税设定   │  │ · 价格机制   │              │
│  │ · 距离矩阵   │  │ · 出口管制   │  │ · 竞争动态   │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│                      智 能 体 层 (Agent Layer)                       │
│                                                                     │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐     │
│  │ Firm 1  │ │ Firm 2  │ │ Firm 3  │ │  ...    │ │ Firm N  │     │
│  │         │ │         │ │         │ │         │ │         │     │
│  │ 能力向量 │ │ 能力向量 │ │ 能力向量 │ │         │ │ 能力向量 │     │
│  │ 技术水平 │ │ 技术水平 │ │ 技术水平 │ │         │ │ 技术水平 │     │
│  │ 规模    │ │ 规模    │ │ 规模    │ │         │ │ 规模    │     │
│  │ 学习率  │ │ 学习率  │ │ 学习率  │ │         │ │ 学习率  │     │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘     │
│                                                                     │
│  每个企业的行为规则：                                                │
│  ① 生产决策 → 选择能生产且利润最高的产品                              │
│  ② 能力投资 → 以概率 p 积累邻近产品的生产能力                         │
│  ③ 模仿学习 → 观察成功企业，尝试进入相似领域                           │
│  ④ 退出/进入 → 持续亏损则退出，高利润吸引新企业                       │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│                      涌 现 层 (Emergence Layer)                      │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ 出口结构     │  │ 复杂度指标   │  │ 网络演化     │              │
│  │              │  │              │  │              │              │
│  │ · 产品分布   │  │ · ECI 时序   │  │ · 产品空间   │              │
│  │ · 出口集中度 │  │ · PCI 变化   │  │   结构变化   │              │
│  │ · 市场分布   │  │ · 复杂度增速 │  │ · 新边出现   │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 企业智能体详细设计

#### 属性定义

| 属性 | 类型 | 说明 | 初始化 |
| --- | --- | --- | --- |
| `capabilities` | Binary Vector (P×1) | 能生产的产品集合，1=有能力，0=无能力 | 随机分配，稀疏（平均 3-5 个产品） |
| `tech_level` | Float [0, 1] | 技术水平，影响生产效率和能力积累速度 | 均匀分布 U(0.1, 0.5) |
| `scale` | Float > 0 | 企业规模，影响产能和市场影响力 | 对数正态分布 |
| `profit_history` | List[Float] | 历史利润序列 | 空列表 |
| `learning_rate` | Float [0, 1] | 学习能力，影响能力积累概率 | 均匀分布 U(0.01, 0.1) |
| `age` | Integer | 企业年龄（存活时间步数） | 0 |

#### 行为规则（每时间步执行）

```python
def step(self):
    # 1. 生产决策：在有能力生产的产品中选择预期利润最高的
    feasible = [p for p in products if self.capabilities[p] == 1]
    if not feasible:
        self.try_acquire_capability()  # 尝试获取第一个能力
        return

    best_product = max(feasible, key=lambda p: self.expected_profit(p))
    self.produce(best_product)

    # 2. 能力投资：以概率 p 尝试积累邻近产品的能力
    self.invest_in_capabilities()

    # 3. 模仿学习：观察利润最高的企业，尝试进入相似领域
    role_model = max(all_firms, key=lambda f: f.avg_profit())
    self.imitate(role_model)

    # 4. 更新状态
    self.age += 1
    self.update_profit_history()
```

#### 能力积累机制

```python
def invest_in_capabilities(self):
    """基于产品空间邻近性，以概率积累新能力"""
    current_products = [p for p in products if self.capabilities[p] == 1]

    for cp in current_products:
        # 获取产品空间中的邻近产品
        neighbors = product_space.neighbors(cp, max_distance=0.5)

        for neighbor in neighbors:
            if self.capabilities[neighbor] == 0:
                # 积累概率 = f(技术水平, 学习率, 距离, 政策补贴)
                prob = (self.tech_level * self.learning_rate
                        * (1 - product_space.distance(cp, neighbor))
                        * self.policy_multiplier(neighbor))

                if random() < prob:
                    self.capabilities[neighbor] = 1
                    self.tech_level += small_bonus  # 多样化带来技术提升
```

### 4.3 产品空间网络构建

产品空间网络基于 UN COMTRADE 数据构建，遵循 Hidalgo et al. (2007) 的经典方法：

1. **显示性比较优势（RCA）**：计算每个国家在每个产品上的 RCA
2. **共出口概率（Proximity）**：`φ(i,j) = min{P(RCA_i|RCA_j), P(RCA_j|RCA_i)}`
3. **网络构建**：以产品为节点，φ > 阈值（通常 0.5）的产品对之间连边
4. **产品复杂度**：通过迭代法计算 PCI（与 ECI 互为特征向量）

### 4.4 政策模块设计

| 政策类型 | 实现方式 | 参数 |
| --- | --- | --- |
| 选择性补贴 | 对目标产品的生产能力积累概率乘以补贴系数 | 目标产品列表、补贴强度 [1.0, 5.0] |
| 功能性支持 | 对所有企业的技术水平提升给予加成 | 提升幅度 [0.01, 0.1] |
| 贸易保护（关税） | 提高进口产品价格，间接提高国内企业利润 | 关税率 [0%, 50%]、覆盖产品范围 |
| 出口管制/技术封锁 | 移除特定产品的能力或切断与特定产品的边 | 管制产品列表、管制强度 |

## 五、数据来源与处理

### 5.1 核心数据源

| 数据源 | 内容 | 用途 | 获取方式 |
| --- | --- | --- | --- |
| [UN COMTRADE](https://comtrade.un.org/) | 双边贸易流量（HS 6位码） | 构建出口矩阵、计算 RCA 和产品空间 | API 免费下载 |
| [Harvard Atlas](https://atlas.hks.harvard.edu/data-downloads/) | 预计算的 ECI/PCI、产品空间距离矩阵 | 模型校准、验证基准 | 直接下载 |
| [OEC](https://oec.world/) | 经济复杂度可视化数据 | 快速探索、结果展示 | 网站 + API |
| [Nature Sci Data (2022)](https://www.nature.com/articles/s41597-022-01732-5) | 整合数据库（1995-2020） | 一站式获取所有复杂度指标 | Zenodo 下载 |
| [WTO Data Lab](https://datalab.wto.org/resource-library) | 贸易政策数据（关税、NTM） | 政策模块参数校准 | 免费下载 |

### 5.2 数据处理流程

```text
# 1. 数据下载与清洗
COMTRADE → 选择国家子集（如 ASEAN + China）
         → 选择产品层级（HS 4位码，约 1200 种）
         → 选择时间范围（2000-2023）
         → 清洗：去除缺失值、价格异常

# 2. 构建出口矩阵 M[c, p, t]
#    c = 国家, p = 产品, t = 年份
#    M[c,p,t] = 1 if RCA[c,p,t] > 1, else 0

# 3. 计算产品空间网络（使用基年或滑动窗口）
for each pair (i, j) in products:
    proximity[i,j] = min(P(RCA_i|RCA_j), P(RCA_j|RCA_i))
product_network = NetworkX graph with proximity > 0.5

# 4. 计算复杂度指标
ECI, PCI = eigenvalue_method(M)  # 或 method_of_reflections

# 5. 提取校准目标
target_stats = {
    'ECI_distribution': ECI.std(),
    'export_diversity': M.sum(axis=1).mean(),
    'network_density': product_network.density(),
    'capability_accumulation_rate': ...
}
```

### 5.3 数据规模选择

> **计算量权衡**
>
> 全量数据（200+ 国家 × 5000+ 产品）计算量过大。建议选择子集：
>
> - **推荐方案**：20-30 个亚太国家 × 500-800 种产品（HS 4位码聚合）
> - **企业数量**：每国 50-200 家代表性企业（总计 1000-6000 智能体）
> - **时间步**：模拟 200-500 步（对应 20-50 年）

## 六、技术实现方案

### 6.1 技术栈

| 组件 | 工具 | 版本 | 用途 |
| --- | --- | --- | --- |
| ABM 框架 | **Mesa** | 3.x | 智能体建模核心框架 |
| 网络分析 | **NetworkX** | 3.x | 产品空间网络构建与分析 |
| 复杂度计算 | **pyecid** / 自实现 | - | ECI、PCI、RCA 计算 |
| 数据处理 | Pandas + NumPy | 2.x / 2.x | 数据清洗与矩阵运算 |
| 可视化 | Matplotlib + Plotly | 3.x / 5.x | 静态图 + 交互式可视化 |
| 校准 | ABC-SMC (自实现) | - | 模型参数校准 |
| 并行计算 | Multiprocessing / Dask | - | 批量实验运行 |

### 6.2 项目结构

```text
economic-complexity-abm/
├── data/
│   ├── raw/                  # 原始 COMTRADE 数据
│   ├── processed/            # 清洗后的出口矩阵
│   └── product_space/        # 产品空间网络（邻接矩阵、距离矩阵）
├── src/
│   ├── agents/
│   │   ├── firm.py           # 企业智能体类
│   │   └── government.py     # 政府智能体（可选）
│   ├── model/
│   │   ├── complexity_model.py  # Mesa 主模型
│   │   ├── product_space.py     # 产品空间网络封装
│   │   └── policy.py            # 政策模块
│   ├── analysis/
│   │   ├── metrics.py           # 复杂度指标计算
│   │   ├── visualization.py     # 可视化工具
│   │   └── calibration.py       # ABC-SMC 校准
│   └── utils/
│       ├── data_loader.py       # 数据加载
│       └── config.py            # 配置管理
├── experiments/
│   ├── baseline.py              # 基准情景
│   ├── subsidy_experiment.py    # 补贴政策实验
│   ├── tariff_experiment.py     # 关税冲击实验
│   └── blockade_experiment.py   # 技术封锁实验
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_product_space_viz.ipynb
│   ├── 03_model_validation.ipynb
│   └── 04_results_analysis.ipynb
├── tests/
├── requirements.txt
├── README.md
└── paper/                       # 论文草稿
    ├── figures/
    └── manuscript.tex
```

### 6.3 核心代码框架

```python
# src/model/complexity_model.py
import mesa
import networkx as nx

class EconomicComplexityModel(mesa.Model):
    """经济复杂度演化 ABM 主模型"""

    def __init__(self, n_firms=100, product_space_graph=None,
                 policy_config=None, initial_capabilities=None):
        super().__init__()
        self.num_firms = n_firms
        self.product_space = product_space_graph  # NetworkX graph
        self.products = list(product_space_graph.nodes)
        self.policy = PolicyModule(policy_config)

        # 创建企业智能体
        self.firms = mesa.AgentSet(self)
        for i in range(n_firms):
            firm = FirmAgent(i, self)
            self.firms.add(firm)

        # 数据收集器
        self.datacollector = mesa.DataCollector(
            self,
            model_reporters={
                "ECI": self.compute_eci,
                "AvgProfit": self.compute_avg_profit,
                "ExportDiversity": self.compute_export_diversity,
                "ActiveFirms": lambda m: len([f for f in m.firms if f.active]),
            }
        )

    def step(self):
        """单步模拟"""
        # 所有企业同时行动
        self.firms.do("step")

        # 市场出清（价格调整）
        self.market_clearing()

        # 企业进入/退出
        self.entry_exit()

        # 收集数据
        self.datacollector.collect(self)


# src/agents/firm.py
class FirmAgent(mesa.Agent):
    """异质性企业智能体"""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.capabilities = self._init_capabilities()
        self.tech_level = np.random.uniform(0.1, 0.5)
        self.scale = np.random.lognormal(0, 1)
        self.learning_rate = np.random.uniform(0.01, 0.1)
        self.profit_history = []
        self.age = 0
        self.active = True

    def step(self):
        if not self.active:
            return

        # 1. 生产决策
        feasible = [p for p in self.model.products
                    if self.capabilities[p]]
        if feasible:
            best = max(feasible, key=self.expected_profit)
            profit = self.produce(best)
            self.profit_history.append(profit)

        # 2. 能力投资
        self.invest_in_capabilities()

        # 3. 模仿学习
        self.imitate_successful_firms()

        self.age += 1

    def invest_in_capabilities(self):
        current = [p for p in self.model.products if self.capabilities[p]]
        for cp in current:
            neighbors = list(self.model.product_space.neighbors(cp))
            for nb in neighbors:
                if not self.capabilities[nb]:
                    dist = nx.shortest_path_length(
                        self.model.product_space, cp, nb)
                    prob = (self.tech_level * self.learning_rate
                            * (1.0 / (1 + dist))
                            * self.model.policy.get_multiplier(nb))
                    if np.random.random() < prob:
                        self.capabilities[nb] = True
```

## 七、实验设计

### 7.1 实验矩阵

| 实验编号 | 情景名称 | 政策设置 | 目的 |
| --- | --- | --- | --- |
| E0 | 基准情景（Laissez-faire） | 无任何政策干预 | 观察自然演化路径 |
| E1 | 选择性补贴 — 高复杂度产品 | 对 PCI 排名前 20% 的产品给予能力积累加成（×2.0） | 测试"瞄准高端"策略 |
| E2 | 选择性补贴 — 邻近产品 | 对当前出口产品空间中距离 < 0.3 的产品给予加成 | 测试"渐进升级"策略 |
| E3 | 功能性支持 | 所有企业技术水平每步 +0.005 | 测试"提升基础能力"策略 |
| E4 | 贸易保护 | 对进口竞争产品征收 20% 关税 | 测试"进口替代"策略 |
| E5 | 关税冲击 | 第 100 步突然对所有出口产品征收 30% 关税 | 测试贸易冲击下的韧性 |
| E6 | 技术封锁 | 第 100 步移除特定高复杂度产品的能力 | 测试技术脱钩的影响 |
| E7 | 组合政策 | E2 + E3（邻近补贴 + 功能支持） | 测试政策协同效应 |

### 7.2 实验参数

| 参数 | 基准值 | 敏感性范围 |
| --- | --- | --- |
| 企业数量 N | 500 | [100, 1000, 2000] |
| 产品数量 P | 200 | [100, 500, 800] |
| 模拟步数 T | 300 | [100, 500] |
| 初始平均能力数 | 5 | [3, 8, 15] |
| 学习率范围 | [0.01, 0.1] | [0.005, 0.2] |
| 重复次数 | 50 | - |

### 7.3 输出指标

- **宏观指标**：ECI 时序、出口多样性、HHI 集中度、Gini 系数
- **微观指标**：企业规模分布、能力分布、利润分布、存活率
- **网络指标**：产品空间密度变化、新出现的产品-国家组合、核心产品占比
- **政策指标**：补贴效率（每单位补贴带来的 ECI 增量）、 welfare 变化

## 八、模型验证策略

### 8.1 三层验证框架

#### 第一层：模式匹配（Pattern Matching）

| 目标模式 | 真实数据特征 | 模型应复现 |
| --- | --- | --- |
| 企业规模分布 | 对数正态 / Pareto 分布 | 模拟的企业规模分布与真实分布的 KS 检验 p > 0.05 |
| 出口多样性分布 | 右偏分布，少数国家出口大量产品 | 模拟的多样性分布与真实分布匹配 |
| ECI-人均GDP 正相关 | 相关系数 ~0.6-0.8 | 模拟中 ECI 高的经济体"增长"更快 |
| 产品空间聚类 | 存在明显的产品簇（如机械类、化工类） | 模拟中企业能力也呈现聚类特征 |

#### 第二层：统计矩匹配（Moment Matching）

- ECI 分布的均值、方差、偏度
- 产品出口量的 Pareto 指数
- 企业存活率的经验分布

#### 第三层：历史反事实（Historical Counterfactual）

选择韩国（1960-2000 成功跃迁）和阿根廷（停滞案例）作为校准对象：

- 用真实产业政策参数运行模型
- 检验模型能否复现韩国的复杂度跃迁轨迹
- 检验模型能否复现阿根廷的锁定效应

### 8.2 校准方法

采用 Li et al. (2025, JASSS) 提出的 ABC-SMC 框架：

1. 定义先验分布（参数范围）
2. 运行模型，提取模拟统计量
3. 计算模拟统计量与真实数据的距离
4. 保留距离小于阈值 θ 的参数集
5. 用保留的参数集生成下一代先验
6. 重复直到收敛

## 九、预期贡献与创新点

### 9.1 方法创新

> **首次实现**
>
> **首次在 ABM 中实现完整的经济复杂度动态演化模拟**——从企业能力积累到产品空间跳跃，再到国家层面复杂度涌现的完整链条。

### 9.2 理论贡献

- 揭示"企业异质性决策 → 国家复杂度涌现"的**微观机制**
- 建立产品空间网络结构与政策效果之间的**因果关系**（而非仅相关性）
- 提出复杂度演化的**路径依赖与锁定**的形式化理论

### 9.3 政策价值

- 为产业政策提供**"计算实验室"**——在模拟中测试不同政策组合的效果
- 回答"什么样的产业政策最有效"这一核心政策问题
- 为发展中国家的出口升级策略提供定量依据

## 十、可行性评估

| 维度 | 评分 | 依据 |
| --- | --- | --- |
| 文献空白 | ★★★★★ | Chudziak (2025) 呼吁但无人实现，几乎无直接竞争 |
| 数据可得 | ★★★★★ | COMTRADE、Harvard Atlas、OEC 全部免费开放 |
| 技术可行 | ★★★★☆ | Mesa + NetworkX 生态成熟，Python 工具链完备 |
| 发表潜力 | ★★★★★ | 方法创新 + 政策相关，可投多个顶刊 |
| 工作量 | ★★★★☆ | 4-6 个月可完成，可控 |

## 十一、时间线与工作量

| 阶段 | 工作内容 |
| --- | --- |
| 阶段 1 | 文献综述（2-3 周）：经济复杂度 + ABM 方法论 |
| 阶段 2 | 数据准备（1-2 周）：下载 COMTRADE、构建产品空间网络 |
| 阶段 3 | 模型开发（4-6 周）：Mesa 框架实现、企业行为规则 |
| 阶段 4 | 校准验证（2-3 周）：ABC-SMC 校准、模式匹配 |
| 阶段 5 | 实验运行（2-3 周）：政策实验、敏感性分析 |
| 阶段 6 | 论文撰写（4-6 周）：结果分析、可视化、写作 |

**总计：15-23 周（约 4-6 个月）**

## 十二、目标期刊与发表策略

| 优先级 | 期刊 | IF | 匹配度 | 策略 |
| --- | --- | --- | --- | --- |
| 首选 | Journal of Economic Interaction and Coordination | 2.8 | ★★★★★ | Chudziak (2025) 发表于此刊，直接对话 |
| 首选 | Journal of Economic Dynamics and Control | 3.2 | ★★★★★ | 顶刊，偏好方法严谨的动态模型 |
| 备选 | JASSS | 3.5 | ★★★★★ | ABM 领域旗舰期刊 |
| 备选 | Computational Economics | 2.3 | ★★★★★ | 门槛较低，适合快速发表 |
| 冲刺 | Research Policy | 7.9 | ★★★★☆ | 强调产业政策贡献 |
| 冲刺 | Journal of International Economics | 4.5 | ★★★★☆ | 国贸顶刊，需强调贸易理论贡献 |

## 十三、风险与应对

| 风险 | 概率 | 影响 | 应对策略 |
| --- | --- | --- | --- |
| 模型无法复现真实复杂度分布 | 中 | 高 | 调整能力积累规则；引入更多异质性；参考 MALLES 的 LLM 增强方法 |
| 校准不收敛 | 中 | 中 | 简化模型；减少参数维度；使用遗传算法替代 ABC |
| 计算量过大 | 低 | 中 | 减小国家/产品子集；使用 Julia 重写核心循环；并行化 |
| 审稿人质疑 ABM 的外部有效性 | 高 | 高 | 强化验证环节；提供历史反事实证据；与计量结果交叉验证 |
| Chudziak 或其他人抢先发表 | 低 | 极高 | 加速进度；强调差异化（如加入 LLM、聚焦特定政策问题） |

## 十四、参考文献

1. Chudziak, S. (2025). Studying economic complexity with agent-based models: Advances, challenges and future perspectives. *Journal of Economic Interaction and Coordination*, 20(2). [DOI: 10.1007/s11403-024-00428-w](https://link.springer.com/article/10.1007/s11403-024-00428-w)
2. Li, K., Ge, J., Lomax, N., & Polhill, J.G. (2025). Calibrating a Global Trade Agent-Based Model with an HPC-ABC-SMC Framework. *JASSS*, 29(3). [Link](https://www.jasss.org/29/3/5.html)
3. Hausmann, R., Hwang, J., & Rodrik, D. (2007). What You Export Matters. *Journal of Economic Growth*, 12(1), 1-25.
4. Hidalgo, C.A., Klinger, B., Barabási, A.L., & Hausmann, R. (2007). The Product Space Conditions the Development of Nations. *Science*, 317(5837), 482-487.
5. Wu, Y., Liu, Y., & Deng, X. (2026). MALLES: A Multi-agent LLMs-based Economic Sandbox with Consumer Preference Alignment. *arXiv*. [Link](https://arxiv.org/html/2603.17694v1)
6. Mashkova, A. & Bakhtizin, A. (2025). Agent based modeling of resilience of key economies to sanctions pressure. *The Journal of New Economics*, 67, 12-24.
7. Ponta, L. et al. (2025). The Complexity of Innovation Strategy: An Agent-Based Approach. In: *Computational Social Systems*. Springer.
8. Harvard Growth Lab (2024). Innovation Policies Under Economic Complexity. Working Paper No. 234.
9. Box-Steffensmeier, D. et al. (2022). An integrated database for economic complexity. *Scientific Data*, 9. [DOI: 10.1038/s41597-022-01732-5](https://www.nature.com/articles/s41597-022-01732-5)
10. Kazil, J., Masad, D., & Fraser, A. (2020). Utilizing Python for Agent-Based Modeling: The Mesa Framework. *Social Computing and Social Media*.
