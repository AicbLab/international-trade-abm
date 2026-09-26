"""
数据管线：清洗 → RCA → 产品空间网络
=================================================
输入：data/raw/harvard_atlas/hs92_country_product_year_4.csv
输出：
  data/processed/rca_matrix.parquet      连续 RCA
  data/processed/M_matrix.parquet        二值能力矩阵 (RCA > 1)
  data/processed/country_metadata.parquet 国家元数据
  data/processed/product_metadata.parquet产品元数据
  data/product_space/proximity.parquet   产品空间邻接矩阵（阈值 0.5）
  data/product_space/proximity_full.parquet 完整 proximity（用于分析）
  data/processed/apex_countries.csv      选定亚太国家清单
  data/processed/pipeline_summary.json   管线摘要
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "harvard_atlas"
PROC = ROOT / "data" / "processed"
PSPACE = ROOT / "data" / "product_space"

# 亚太国家 ISO3（25 个经济体：ASEAN10 + China + JP + KR + TW + HK + SG +
# AU + NZ + IN + PK + BD + LK）
# 说明：SG 已在 ASEAN 中，此处仅列出一次；选择标准是亚太活跃贸易体，
# 兼顾多样性（发达/发展中/资源型/制造型）。
ASIA_PACIFIC_ISO3 = [
    # ASEAN
    "BRN", "IDN", "MYS", "PHL", "SGP", "THA", "VNM", "KHM", "LAO", "MMR",
    # 东北亚
    "CHN", "JPN", "KOR", "TWN", "HKG",
    # 南亚
    "IND", "PAK", "BGD", "LKA",
    # 大洋洲
    "AUS", "NZL",
    # 其他亚太活跃经济体
    "PNG", "FJI", "MNG",
]

PROXIMITY_THRESHOLD = 0.5
RCA_THRESHOLD = 1.0
STUDY_YEARS = range(2000, 2024)

BASE_YEAR = 2000
PIPELINE_VERSION = 'global-base2000-v4'


def resource_product_mask(products: list[str]) -> np.ndarray:
    """保守的矿产/初级燃料代理，不是官方 SITC 对照或全资源分类。

    仅选 HS25、26 和 2701–2709；不把化学品、棉纺、食品整体排除。
    """
    return np.array([p[:2] in {'25', '26'} or '2701' <= p <= '2709'
                     for p in products], dtype=bool)


def investment_budgets_by_gdp(countries: list[str], budget_min: float = 1.0,
                               budget_max: float = 15.0) -> dict[str, float]:
    """读取已核实的规模预算表；缺GDP的回退口径由表内逐国记录。"""
    if not 0 <= budget_min <= budget_max:
        raise ValueError('预算上下界不合法')
    table = pd.read_csv(PROC / 'country_budget_inputs.csv').set_index('country_iso3')
    norms = table.loc[countries, 'scale_normalized']
    if not np.isfinite(norms).all() or not norms.between(0, 1).all():
        raise ValueError('规模输入缺失或超界，请先运行 scripts/prepare_covariates.py')
    return (budget_min + (budget_max - budget_min) * norms).to_dict()


# ---------- IO ----------

def load_raw_main_table() -> pd.DataFrame:
    """流式读取主表（~450MB），仅保留必要列。"""
    cols = [
        "country_id", "country_iso3_code",
        "product_id", "product_hs92_code",
        "year", "export_value",
    ]
    print("[1/7] 读取主表 ...")
    t0 = time.time()
    df = pd.read_csv(
        RAW / "hs92_country_product_year_4.csv",
        usecols=cols,
        dtype={
            "country_id": "int32",
            "country_iso3_code": "category",
            "product_id": "int32",
            "product_hs92_code": "str",
            "year": "int16",
            "export_value": "float64",
        },
    )
    print(f"      读取完成：{len(df):,} 行，耗时 {time.time()-t0:.1f}s")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """清洗规则（保留全球数据，用于 RCA 的世界参照）：
    - 仅保留标准 HS4（4 位数字）；剔除 XXXX 等非标准代码
    - 剔除出口额缺失或 ≤0 的记录（不视为零贸易，直接丢弃）
    - 剔除出口额负值（异常）
    - 限定研究年份 2000–2023
    注：亚太子集裁剪在 build_matrices() 中完成，以保证 RCA 分母为全球。
    """
    print("[2/7] 清洗 ...")
    n0 = len(df)

    # 研究年份
    df = df[df["year"].isin(set(STUDY_YEARS))]

    # 标准 HS4：4 位数字
    hs = df["product_hs92_code"]
    valid_hs = hs.str.match(r"^\d{4}$")
    df = df[valid_hs]

    df = df[df['country_iso3_code'] != 'ANS'].copy()
    keys = ['country_iso3_code', 'product_hs92_code', 'year']
    if df.duplicated(keys).any():
        raise ValueError('国家-产品-年份主键重复，不能自动求和')
    if not np.isfinite(df['export_value']).all() or (df['export_value'] < 0).any():
        raise ValueError('出口额存在非有限值或负值，须明确处理后才能建模')

    # 重置索引
    df = df.reset_index(drop=True)

    print(f"      清洗前 {n0:>12,} 行")
    print(f"      清洗后 {len(df):>12,} 行")
    print(f"      国家数 {df['country_iso3_code'].nunique()}（全球）")
    print(f"      产品数 {df['product_hs92_code'].nunique()}")
    print(f"      年份跨度 {df['year'].min()}–{df['year'].max()}")
    return df


def build_matrices(df: pd.DataFrame):
    """构造 RCA / M 矩阵。

    关键：RCA 的分母必须是**全球**出口结构，而不是亚太子集。
    因此先用全量 df 计算世界出口总额，再裁剪到亚太子集输出 M/rca。
    """
    print("[3/7] 构建出口额 / RCA / M 矩阵 ...")

    # 1. 全球有序列表（用于计算世界参照）
    all_countries = sorted(df["country_iso3_code"].unique())
    base = df[df['year'] == BASE_YEAR]
    if base.empty:
        raise ValueError('缺少初始化年份')
    products = sorted(base.loc[base['export_value'] > 0, 'product_hs92_code'].unique())
    years = sorted(df["year"].unique())
    c_all_idx = {c: i for i, c in enumerate(all_countries)}
    p_idx = {p: i for i, p in enumerate(products)}
    print(f"      全球国家 {len(all_countries)}  产品 {len(products)}  年份 {len(years)}")

    # 2. 构建全球 3D 出口额数组 X_all[C_all, P, Y]
    X_all = np.zeros((len(all_countries), len(products), len(years)), dtype=np.float64)
    for yi, y in enumerate(years):
        sub = df[df["year"] == y]
        if sub.empty:
            continue
        piv = sub.pivot_table(
            index="country_iso3_code",
            columns="product_hs92_code",
            values="export_value",
            aggfunc="sum",
        )
        # 对齐到有序列表；fillna(0) 替换已有的 NaN（国家-产品无出口记录）
        piv = piv.reindex(index=all_countries, columns=products).fillna(0.0)
        X_all[:, :, yi] = piv.values

    # 3. 世界出口：每个产品-年份的全球出口总和
    X_world_p_y = X_all.sum(axis=0)  # (P, Y)
    X_world_total_y = X_world_p_y.sum(axis=0)  # (Y,)

    # 4. 计算全球 RCA：每个国家（含亚太外）的 RCA
    X_country_all_y = X_all.sum(axis=1)  # (C_all, Y)
    active = X_country_all_y > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        share_cp_y = X_all / X_country_all_y[:, None, :]
        share_wp_y = X_world_p_y / X_world_total_y[None, :]
        rca_all = share_cp_y / share_wp_y[None, :, :]
    rca_all = np.nan_to_num(rca_all, nan=0.0, posinf=0.0, neginf=0.0)

    # 5. 使用全球所有国家（不再裁剪到亚太子集）
    countries = all_countries  # 保持有序

    X = X_all
    rca = rca_all
    M = (rca > RCA_THRESHOLD).astype(np.uint8)
    base_index = years.index(BASE_YEAR)
    M_base_global = (rca_all[active[:, base_index], :, base_index] > RCA_THRESHOLD).astype(np.uint8)

    c_idx = {c: i for i, c in enumerate(countries)}

    print(f"      全球：C={len(countries)}, P={len(products)}, Y={len(years)}")
    # 诊断
    M_any = M.sum(axis=(0, 2))  # 每产品有多少国家-年有 RCA>1
    print(f"      有 RCA>1 的产品数（至少一次）：{(M_any > 0).sum()}")
    div = M.sum(axis=1)  # (C, Y)
    print(f"      平均出口多样性（首年）：{div[:, 0].mean():.1f}")
    print(f"      平均出口多样性（末年）：{div[:, -1].mean():.1f}")

    return {
        "countries": countries, "products": products, "years": years,
        "X": X, "rca": rca, "M": M,
        "c_idx": c_idx, "p_idx": p_idx,
        "M_base_global": M_base_global,
        "n_global_base": int(active[:, base_index].sum()),
        "missing_cell_assumption": '有效国家年度内缺省产品视为零；非有限值和缺年不填零',
    }


def build_proximity(M_global: np.ndarray) -> np.ndarray:
    """基于全球二值矩阵 M[C, P]（基年或多年平均）计算产品空间 proximity。

    φ(i, j) = min( P(RCA_i | RCA_j), P(RCA_j | RCA_i) )
    其中 P(RCA_i | RCA_j) = (#国家同时在 i 和 j 有 RCA) / (#国家在 j 有 RCA)

    使用矩阵运算：
        cooc = M.T @ M          # (P, P) 共现次数
        n_p  = M.sum(axis=0)    # (P,) 每种产品有 RCA 的国家数
        P_i_given_j = cooc / n_p[:, None]
        phi = min(P_i_given_j, P_i_given_j.T)
    """
    print("[4/7] 构建产品空间 proximity ...")
    # 使用多年 M 矩阵的"至少一年有 RCA"作为国家能力（更稳健）
    # M_global shape: (C, P)
    m = np.asarray(M_global, dtype=np.float64)
    if m.ndim != 2 or not np.isin(m, [0, 1]).all():
        raise ValueError('proximity 输入必须是二值二维矩阵')
    cooc = m.T @ m
    n_p = m.sum(axis=0)
    denominator = np.maximum(n_p[:, None], n_p[None, :])
    phi = np.divide(cooc, denominator, out=np.zeros_like(cooc), where=denominator > 0)
    # 对角线置 0（自身不算邻近）
    np.fill_diagonal(phi, 0.0)
    # 清理 NaN
    phi = np.nan_to_num(phi, nan=0.0)
    print(f"      产品数 {phi.shape[0]}，边数 (φ>0.5) {(phi > PROXIMITY_THRESHOLD).sum() // 2:,}")
    return phi


def save_outputs(matrices, phi, df_clean):
    """保存处理结果。"""
    print("[5/7] 保存处理结果 ...")
    PROC.mkdir(parents=True, exist_ok=True)
    PSPACE.mkdir(parents=True, exist_ok=True)

    countries = matrices["countries"]
    products = matrices["products"]
    years = matrices["years"]

    # 将 3D 数组展平为 long-form，便于下游消费
    def to_long(arr, value_name):
        idx = pd.MultiIndex.from_product(
            [countries, products, years],
            names=["country_iso3", "product_hs92", "year"],
        )
        return pd.Series(arr.ravel(), index=idx, name=value_name).reset_index()

    rca_long = to_long(matrices["rca"], "rca")
    M_long = to_long(matrices["M"], "capability")

    rca_long.to_parquet(PROC / "rca_matrix.parquet", index=False)
    M_long.to_parquet(PROC / "M_matrix.parquet", index=False)

    # 国家 / 产品元数据
    pd.DataFrame({"country_iso3": countries}).to_parquet(
        PROC / "country_metadata.parquet", index=False)
    pd.DataFrame({"product_hs92": products}).to_parquet(
        PROC / "product_metadata.parquet", index=False)

    # 产品空间
    phi_df = pd.DataFrame(phi, index=products, columns=products)
    phi_df.to_parquet(PSPACE / "proximity_full.parquet")
    # 阈值化邻接
    adj = (phi > PROXIMITY_THRESHOLD).astype(np.uint8)
    np.fill_diagonal(adj, 0)
    adj_df = pd.DataFrame(adj, index=products, columns=products)
    adj_df.to_parquet(PSPACE / "proximity.parquet")

    # 国家清单
    pd.DataFrame({
        "country_iso3": countries,
        "in_asia_pacific": [c in ASIA_PACIFIC_ISO3 for c in countries],
    }).to_csv(PROC / "apex_countries.csv", index=False)

    # 管线摘要
    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "network_base_year": BASE_YEAR,
        "n_global_base": matrices['n_global_base'],
        "missing_cell_assumption": matrices['missing_cell_assumption'],
        "countries": countries,
        "n_countries": len(countries),
        "n_products": len(products),
        "years": [int(y) for y in years],
        "n_years": len(years),
        "rca_threshold": RCA_THRESHOLD,
        "proximity_threshold": PROXIMITY_THRESHOLD,
        "n_edges_thresholded": int(adj.sum() // 2),
        "density_thresholded": float(adj.sum() / (len(products) * (len(products) - 1))),
        "clean_rows": int(len(df_clean)),
    }
    with open(PROC / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def report_global_capability(matrices):
    """基于全球数据构建产品空间（更稳健，避免亚太子集偏差）。
    这里仅用于对比；主输出使用亚太子集。"""
    # 为简化，本版本仅使用亚太子集构建 proximity。
    # 如需全球 proximity，可在此扩展。
    pass


def main():
    print("=" * 60)
    print("数据管线：清洗 → RCA → 产品空间")
    print("=" * 60)

    df_raw = load_raw_main_table()
    df_clean = clean(df_raw)

    matrices = build_matrices(df_clean)

    # 产品空间：使用亚太子集 M 矩阵的"至少一年有 RCA"
    # M[c, p, y] → M_global[c, p] = 1 if any year has RCA>1
    phi = build_proximity(matrices['M_base_global'])

    summary = save_outputs(matrices, phi, df_clean)

    print()
    print("[6/7] 关键统计")
    print(f"      国家数：{summary['n_countries']}")
    print(f"      产品数：{summary['n_products']}")
    print(f"      年份数：{summary['n_years']}")
    print(f"      产品空间边数（φ>0.5）：{summary['n_edges_thresholded']:,}")
    print(f"      产品空间密度：{summary['density_thresholded']:.4f}")

    # 能力统计
    M = matrices["M"]
    diversity = M.sum(axis=1)  # (C, Y) 每国每年有 RCA 的产品数
    print(f"      平均出口多样性（首年）：{diversity[:, 0].mean():.1f}")
    print(f"      平均出口多样性（末年）：{diversity[:, -1].mean():.1f}")

    print()
    print("[7/7] 完成。输出目录：")
    print(f"      {PROC}")
    print(f"      {PSPACE}")


if __name__ == "__main__":
    main()
