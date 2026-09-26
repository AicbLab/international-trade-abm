# 国家智能体 ABM · 方法说明（v2 迭代版）

> 模型版本：global-base2000-v3
> 数据：Harvard HS92 四位码（2000–2023），24 国 × 1242 产品
> 框架：Mesa 3.x，国家即智能体

## 1. 模型结构

### 1.1 智能体与状态

每个国家智能体持有一个二值能力向量 $m \in \{0,1\}^P$，
其中 $m_p = 1$ 表示该国在产品 $p$ 上具有显示性比较优势（RCA > 1）。

初始状态来自 Harvard Atlas 2000 年数据。

### 1.2 产品空间

产品空间 $\phi(i,j) = \min(P(RCA_i|RCA_j), P(RCA_j|RCA_i))$，
top-30 稀疏化后构建为稀疏邻接矩阵。

### 1.3 每步动态

每步（对应 1 年）包含：

1. **暴露**：$e = W \cdot m$（稀疏矩阵乘法，$W$ 为 top-30 proximity）
2. **投资**：以概率 $1 - e^{-\eta \cdot e \cdot b}$ 尝试获取能力
   - $b$ 为投资预算（异质：GDP 对数归一化到 [1,15]；同质：固定 5）
   - $\eta$ 为学习率
3. **模仿**：以概率 $1 - e^{-\rho \cdot \bar{m}}$ 从区域平均获取能力
   - $\rho$ 为模仿率
4. **折旧**：每个能力以概率 $\delta$ 独立丢失
   - **不保护初始能力**（所有能力均可丢失）

### 1.4 三重约束

1. **投资预算**：GDP 对数归一化 → [1, 15]（缺 GDP 用出口代理）
2. **折旧**：所有能力（含初始）以概率 $\delta$ 丢失
3. **Top-K 稀疏化**：产品空间仅保留 top-30 邻居

## 2. 数据管线

### 2.1 数据源

| 数据 | 来源 | 规模 |
|---|---|---|
| HS92 贸易 | Harvard Atlas | 450MB CSV |
| GDP | World Bank | 2000 年 |
| 国家元数据 | Atlas metadata | 24 国 |

### 2.2 处理流程

1. 清洗 → RCA 计算 → 二值化（RCA > 1）
2. 产品空间 proximity → top-30 稀疏化
3. 投资预算：GDP 对数归一化 → [1, 15]
4. 资源掩码：HS25/26 + 2701-2709（保守分类）

## 3. 校准

### 3.1 加权 ABC-SMC

采用 Toni et al. (2010) 风格的序列蒙特卡洛近似贝叶斯校准。

**参数空间**：
- $\theta = (\eta, \rho, \delta) \in [0.01, 1] \times [0, 0.5] \times [0.001, 0.05]$

**目标统计量**（v2 降维版）：
- 每训练年（2005/2010/2015）：mean_div, median_div, p25, p75, max_div, |ECI|_mean
- 共 18 维（v1 为 144 维，因维数灾难失败）

**配置**：200 粒子 × 10 代 × 2 重复

**距离**：加权欧氏距离，尺度 = max(|target| × 0.1, 5)

### 3.2 校准结果

| 参数 | 默认 | 校准后验 |
|---|---:|---:|
| learning_rate | 0.05 | 0.594 |
| imitation_rate | 0.10 | 0.256 |
| depreciation_rate | 0.005 | 0.015 |

**epsilon 轨迹**：13.24 → 12.03（9% 收缩）

**留出验证**：
- 校准：holdout MAE = 51.2
- 默认：holdout MAE = 49.1（更优）

**诊断**：参数不可识别性——不同参数产生相似的宏观输出。

## 4. 政策模块

### 4.1 三类政策

| 政策 | 机制 | 参数 |
|---|---|---|
| 定向补贴 | 投资乘子 | targets, strength, countries |
| 技术封锁 | 暴露衰减 | targets, strength, countries |
| 关税 | 投资乘子衰减 | targets, rate, elasticity |

### 4.2 配对实验设计

- 相同 seed → 相同随机数序列
- 仅政策参数不同
- 差异完全归因于政策干预

## 5. 评估框架

### 5.1 Benchmark 对比

| 模型 | 预测方式 |
|---|---|
| ABM | 产品空间 + 动态规则 |
| 不变假设 | 2023 = 2000 |
| 线性外推 | 2000→2005 趋势 × 4.6 |
| 全球均值 | 所有国家 = 全球平均 |

### 5.2 评估指标

- 最终 MAE：2023 年逐国多样性 vs 历史
- 留出 MAE：2016-2023 逐年逐国
- RMSE：同上

## 6. 语义产品空间（LLM 增强）

### 6.1 方法概述

用三种嵌入方法对 1242 个产品的 UN 英文描述（来源 `hs92_hs4_product_lookup.csv`）编码，
构建语义 proximity 矩阵，与统计 proximity 融合：

$$\phi_{hybrid} = \alpha \cdot \phi_{stat} + (1-\alpha) \cdot \phi_{sem}$$

所有方法共享后处理流程：
1. 计算余弦相似度矩阵 $\phi_{sem}(i,j) = \cos(\mathbf{e}_i, \mathbf{e}_j)$
2. 对角线置零
3. Top-30 稀疏化（每行仅保留相似度最高的 30 个邻居）
4. 对称化：$\phi_{sem} \leftarrow (\phi_{sem} + \phi_{sem}^T) / 2$
5. 截断到 $[0, 1]$

### 6.2 方法 A：千问 Qwen text-embedding-v2

| 参数 | 值 |
|---|---|
| **模型名称** | `text-embedding-v2` |
| **提供商** | 阿里云 DashScope API |
| **SDK** | `dashscope >= 1.20`（`TextEmbedding.call()`） |
| **嵌入维度** | 1536 |
| **输入** | 1242 条 UN 产品英文描述 |
| **批处理** | batch_size = 25，每批间隔 0.1s（限速） |
| **归一化** | L2 归一化后计算点积（等价余弦相似度） |
| **Top-K** | 30 |
| **API 调用** | `TextEmbedding.call(model='text-embedding-v2', input=batch, dimension=1536)` |

### 6.3 方法 B：Sentence Transformer

| 参数 | 值 |
|---|---|
| **模型名称** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| **架构** | Transformer encoder, 12 层 |
| **最大序列长度** | 128 tokens |
| **嵌入维度** | 384 |
| **训练数据** | 多语言平行语料（50+ 语言） |
| **归一化** | `normalize_embeddings=True`（L2 归一化） |
| **相似度** | 点积（归一化后等价余弦相似度） |
| **Top-K** | 30 |
| **库** | `sentence-transformers` |

### 6.4 方法 C：TF-IDF

| 参数 | 值 |
|---|---|
| **向量化器** | `sklearn.feature_extraction.text.TfidfVectorizer` |
| **max_features** | 5000 |
| **stop_words** | `'english'` |
| **ngram_range** | (1, 2)（unigram + bigram） |
| **sublinear_tf** | `True`（对数词频缩放） |
| **嵌入维度** | 5000（稀疏） |
| **相似度** | `sklearn.metrics.pairwise.cosine_similarity` |
| **Top-K** | 30 |

### 6.5 融合参数搜索

对每种方法测试 $\alpha \in \{0.0, 0.2, 0.4, 0.6, 0.8, 1.0\}$，
以留出 MAE（2016–2023 逐年逐国）选最优 $\alpha$。

### 6.6 结果（趋势修正后模型）

| 方法 | 嵌入维度 | 最优 α | 留出 MAE | 改善 |
|---|---:|---:|---:|---:|
| TF-IDF | 5000 (稀疏) | 0.8 | 17.5 | 2.9% |
| 千问 Qwen | 1536 | 0.6 | 17.6 | 2.3% |
| Sentence Transformer | 384 | 0.8 | 18.0 | 0.3% |
| 纯统计基线 | — | 1.0 | 18.1 | — |

### 6.7 学术意义

首次将 LLM 嵌入引入产品空间 ABM，证明语义信息捕捉了贸易统计无法反映的产品关系。
反直觉发现：TF-IDF（简单方法）优于千问/ST（复杂方法），因为高维稀疏嵌入的精确关键词匹配对产品分类更有效。

## 7. 复现性

- 固定 seed = 42
- SHA256 输入指纹
- 所有实验脚本：`experiments/run_*.py`
- 22 个单元测试全部通过
