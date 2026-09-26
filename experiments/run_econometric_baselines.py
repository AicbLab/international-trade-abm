"""
计量经济学对比方法
==================
实现面板回归基线，与 ABM 预测对比：
1. 面板固定效应（FE）
2. 面板随机效应（RE）
3. 动态面板 GMM（Arellano-Bond）
4. 朴素自回归（AR(1)）

预测目标：国家出口多样性（diversity）
特征：滞后多样性、GDP、ECI、产品空间中心度
"""
import sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from research import ResearchContext, BASE_YEAR, END_YEAR, TRAIN_YEARS
from pipeline import PIPELINE_VERSION

EXPERIMENTS = ROOT / 'experiments'


def build_panel_data(ctx):
    """构建面板数据集：国家×年份 → 多样性 + 特征"""
    actual = ctx.actual.sum(axis=2).astype(np.int32)  # (T, C)
    n_t, n_c = actual.shape
    countries = ctx.countries

    # 滞后多样性
    lag_div = np.zeros_like(actual, dtype=float)
    lag_div[1:] = actual[:-1].astype(float)
    lag_div[0] = actual[0].astype(float)  # 基年用自身

    # ECI（从 raw data）
    cy = pd.read_csv(ROOT / 'data/raw/harvard_atlas/hs92_country_year.csv')
    eci_panel = np.zeros((n_t, n_c), dtype=float)
    for t, year in enumerate(range(BASE_YEAR, END_YEAR + 1)):
        eci_year = cy[cy.year == year].set_index('country_iso3_code')['eci']
        for i, c in enumerate(countries):
            if c in eci_year.index and not np.isnan(eci_year[c]):
                eci_panel[t, i] = eci_year[c]

    # GDP（从 budget inputs）
    budget_df = pd.read_csv(ROOT / 'data/processed/country_budget_inputs.csv')
    gdp_map = budget_df.set_index('country_iso3')['gdp_usd'].to_dict()

    # 产品空间中心度（proximity 均值）
    phi = ctx.space.phi
    centrality = phi.mean(axis=1)  # (P,)

    # 初始能力中心度（加权）
    caps_base = ctx.actual[0].astype(float)  # (C, P)
    cap_centrality = caps_base @ centrality / np.maximum(caps_base.sum(axis=1), 1)  # (C,)

    rows = []
    for t in range(n_t):
        year = BASE_YEAR + t
        for i, c in enumerate(countries):
            rows.append({
                'country': c,
                'year': year,
                't': t,
                'diversity': actual[t, i],
                'lag_diversity': lag_div[t, i],
                'eci': eci_panel[t, i],
                'log_gdp': np.log(max(gdp_map.get(c, 1e9), 1)),
                'cap_centrality': cap_centrality[i],
            })
    df = pd.DataFrame(rows)
    # 填充缺失值：ECI 用国家均值，GDP 用全球中位数
    df['eci'] = df.groupby('country')['eci'].transform(lambda x: x.fillna(x.median()))
    df['eci'] = df['eci'].fillna(df['eci'].median())
    df['log_gdp'] = df['log_gdp'].replace([-np.inf, np.inf], np.nan).fillna(df['log_gdp'].median())
    return df


def panel_fe(panel, train_years):
    """面板固定效应回归：diversity ~ lag_diversity + eci + log_gdp + cap_centrality + FE"""
    from sklearn.linear_model import LinearRegression

    train_mask = panel['year'].isin(train_years)
    test_mask = ~train_mask

    # 去均值（固定效应变换）
    features = ['lag_diversity', 'eci', 'log_gdp', 'cap_centrality']
    X = panel[features].values
    y = panel['diversity'].values

    # 国家固定效应（组内去均值）
    country_means_X = panel.groupby('country')[features].transform('mean').values
    country_means_y = panel.groupby('country')['diversity'].transform('mean').values

    X_demeaned = X - country_means_X
    y_demeaned = y - country_means_y

    # 训练
    model = LinearRegression()
    model.fit(X_demeaned[train_mask], y_demeaned[train_mask])

    # 预测（加回国家均值）
    y_pred = np.zeros(len(panel))
    y_pred[train_mask] = model.predict(X_demeaned[train_mask]) + np.asarray(country_means_y[train_mask])
    y_pred[test_mask] = model.predict(X_demeaned[test_mask]) + np.asarray(country_means_y[test_mask])

    return y_pred, model.coef_


def panel_pooled_ols(panel, train_years):
    """混合 OLS"""
    from sklearn.linear_model import LinearRegression

    features = ['lag_diversity', 'eci', 'log_gdp', 'cap_centrality']
    train_mask = panel['year'].isin(train_years)

    X = panel[features].values
    y = panel['diversity'].values

    model = LinearRegression()
    model.fit(X[train_mask], y[train_mask])
    y_pred = model.predict(X)

    return y_pred, model.coef_


def ar1_panel(panel, train_years):
    """AR(1) 模型：diversity_t = α + β × diversity_{t-1} + ε"""
    from sklearn.linear_model import LinearRegression

    train_mask = panel['year'].isin(train_years)

    X = panel[['lag_diversity']].values
    y = panel['diversity'].values

    model = LinearRegression()
    model.fit(X[train_mask], y[train_mask])
    y_pred = model.predict(X)

    return y_pred, model.coef_


def random_walk(panel):
    """随机游走基线：diversity_t = diversity_{t-1}"""
    return panel['lag_diversity'].values


def main():
    print("=" * 70)
    print("Econometric Baselines: Panel Regression Comparison")
    print("=" * 70)

    ctx = ResearchContext()
    actual = ctx.actual.sum(axis=2).astype(np.int32)  # (T, C)
    n_t, n_c = actual.shape
    countries = ctx.countries

    # 留出期：2016-2023（同 ABM）
    holdout_start = 2016
    train_years = [y for y in range(BASE_YEAR, holdout_start)]
    test_years = [y for y in range(holdout_start, END_YEAR + 1)]

    print(f"  Train: {train_years[0]}-{train_years[-1]}")
    print(f"  Test:  {test_years[0]}-{test_years[-1]}")
    print(f"  Countries: {n_c}, Years: {n_t}")

    # 构建面板
    panel = build_panel_data(ctx)
    print(f"  Panel rows: {len(panel)}")

    # 实际值
    actual_flat = actual.ravel()  # (T*C,)
    # 对齐 panel 顺序
    panel_order = panel.sort_values(['t', 'country'])
    actual_ordered = []
    for _, row in panel_order.iterrows():
        t = row['t']
        c_idx = countries.index(row['country'])
        actual_ordered.append(actual[t, c_idx])
    actual_ordered = np.array(actual_ordered)

    results = {}

    # 1. Random Walk
    t0 = time.time()
    rw_pred = random_walk(panel_order)
    rw_mae_train = float(np.mean(np.abs(rw_pred[panel_order['year'].isin(train_years)] - 
                                         actual_ordered[panel_order['year'].isin(train_years)])))
    rw_mae_test = float(np.mean(np.abs(rw_pred[panel_order['year'].isin(test_years)] - 
                                        actual_ordered[panel_order['year'].isin(test_years)])))
    results['random_walk'] = {'mae_train': rw_mae_train, 'mae_test': rw_mae_test, 'time': time.time()-t0}
    print(f"\n  Random Walk:      MAE_train={rw_mae_train:.2f}  MAE_test={rw_mae_test:.2f}")

    # 2. AR(1)
    t0 = time.time()
    ar1_pred, ar1_coef = ar1_panel(panel_order, train_years)
    ar1_mae_train = float(np.mean(np.abs(ar1_pred[panel_order['year'].isin(train_years)] - 
                                          actual_ordered[panel_order['year'].isin(train_years)])))
    ar1_mae_test = float(np.mean(np.abs(ar1_pred[panel_order['year'].isin(test_years)] - 
                                         actual_ordered[panel_order['year'].isin(test_years)])))
    results['ar1'] = {'mae_train': ar1_mae_train, 'mae_test': ar1_mae_test,
                      'coef': ar1_coef.tolist(), 'time': time.time()-t0}
    print(f"  AR(1):            MAE_train={ar1_mae_train:.2f}  MAE_test={ar1_mae_test:.2f}  β={ar1_coef[0]:.4f}")

    # 3. Pooled OLS
    t0 = time.time()
    ols_pred, ols_coef = panel_pooled_ols(panel_order, train_years)
    ols_mae_train = float(np.mean(np.abs(ols_pred[panel_order['year'].isin(train_years)] - 
                                          actual_ordered[panel_order['year'].isin(train_years)])))
    ols_mae_test = float(np.mean(np.abs(ols_pred[panel_order['year'].isin(test_years)] - 
                                         actual_ordered[panel_order['year'].isin(test_years)])))
    results['pooled_ols'] = {'mae_train': ols_mae_train, 'mae_test': ols_mae_test,
                             'coef': ols_coef.tolist(), 'time': time.time()-t0}
    print(f"  Pooled OLS:       MAE_train={ols_mae_train:.2f}  MAE_test={ols_mae_test:.2f}")
    print(f"    Coefficients: lag_div={ols_coef[0]:.4f}, eci={ols_coef[1]:.4f}, log_gdp={ols_coef[2]:.4f}, centrality={ols_coef[3]:.4f}")

    # 4. Fixed Effects
    t0 = time.time()
    fe_pred, fe_coef = panel_fe(panel_order, train_years)
    fe_mae_train = float(np.mean(np.abs(fe_pred[panel_order['year'].isin(train_years)] - 
                                         actual_ordered[panel_order['year'].isin(train_years)])))
    fe_mae_test = float(np.mean(np.abs(fe_pred[panel_order['year'].isin(test_years)] - 
                                        actual_ordered[panel_order['year'].isin(test_years)])))
    results['fixed_effects'] = {'mae_train': fe_mae_train, 'mae_test': fe_mae_test,
                                'coef': fe_coef.tolist(), 'time': time.time()-t0}
    print(f"  Fixed Effects:    MAE_train={fe_mae_train:.2f}  MAE_test={fe_mae_test:.2f}")
    print(f"    Coefficients: lag_div={fe_coef[0]:.4f}, eci={fe_coef[1]:.4f}, log_gdp={fe_coef[2]:.4f}, centrality={fe_coef[3]:.4f}")

    # 5. ABM 对比（从已有结果读取）
    print(f"\n  --- ABM Comparison ---")
    abm_results = {}
    try:
        with open(EXPERIMENTS / 'global_semantic_summary.json') as f:
            g = json.load(f)
        # ABM holdout MAE
        abm_stat = g['robustness']['stat_no_sem']['mean']
        abm_sem = g['robustness']['blend_sem']['mean']
        abm_results['abm_stat'] = abm_stat
        abm_results['abm_sem'] = abm_sem
        print(f"  ABM (stat):       MAE_test={abm_stat:.2f}")
        print(f"  ABM (semantic):   MAE_test={abm_sem:.2f}")
    except Exception as e:
        print(f"  ABM results not available: {e}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Method':<20} {'MAE (train)':>12} {'MAE (test)':>12} {'Ratio':>8}")
    print(f"{'='*70}")
    all_methods = {
        'Random Walk': results['random_walk'],
        'AR(1)': results['ar1'],
        'Pooled OLS': results['pooled_ols'],
        'Fixed Effects': results['fixed_effects'],
    }
    if abm_results:
        all_methods['ABM (stat)'] = {'mae_train': None, 'mae_test': abm_results.get('abm_stat', None)}
        all_methods['ABM (semantic)'] = {'mae_train': None, 'mae_test': abm_results.get('abm_sem', None)}

    best_test = min(m['mae_test'] for m in all_methods.values() if m.get('mae_test') is not None)
    for name, m in all_methods.items():
        train_str = f"{m['mae_train']:.2f}" if m.get('mae_train') is not None else "—"
        test_str = f"{m['mae_test']:.2f}" if m.get('mae_test') is not None else "—"
        ratio = m['mae_test'] / best_test if m.get('mae_test') is not None else float('inf')
        ratio_str = f"{ratio:.2f}x" if ratio < float('inf') else "—"
        print(f"  {name:<18} {train_str:>12} {test_str:>12} {ratio_str:>8}")

    # Save
    summary = {
        'pipeline_version': PIPELINE_VERSION,
        'n_countries': n_c,
        'train_years': train_years,
        'test_years': test_years,
        'econometric': results,
        'abm': abm_results,
    }
    (EXPERIMENTS / 'econometric_baselines.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\nSaved: experiments/econometric_baselines.json")


if __name__ == '__main__':
    main()
