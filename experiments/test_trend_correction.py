"""测试增长趋势修正效果。"""
import sys, numpy as np
sys.path.insert(0, 'src')
from research import ResearchContext, DEFAULT_THETA, BASE_YEAR

ctx = ResearchContext()
actual = ctx.actual.sum(axis=2).astype(np.int32)

print("Growth trends:")
for i, c in enumerate(ctx.countries):
    t = ctx.growth_trends[c]
    print(f"  {c}: {t:+.2f}/yr")

print("\nCorrected model vs actual (2023):")
print(f"  {'Country':>4}  {'Actual':>6}  {'NoTrend':>7}  {'Trend':>6}  {'Err(no)':>8}  {'Err(yes)':>8}")

states_no, _ = ctx.simulate(DEFAULT_THETA, seed=42, use_trends=False)
states_yes, _ = ctx.simulate(DEFAULT_THETA, seed=42, use_trends=True)

div_no = states_no.sum(axis=2).astype(np.int32)[-1]
div_yes = states_yes.sum(axis=2).astype(np.int32)[-1]
div_actual = actual[-1]

mae_no = 0
mae_yes = 0
for i, c in enumerate(ctx.countries):
    e_no = div_no[i] - div_actual[i]
    e_yes = div_yes[i] - div_actual[i]
    mae_no += abs(e_no)
    mae_yes += abs(e_yes)
    print(f"  {c:>4}  {div_actual[i]:6d}  {div_no[i]:7d}  {div_yes[i]:6d}  {e_no:+8d}  {e_yes:+8d}")

print(f"\n  Mean MAE:  no_trend={mae_no/len(ctx.countries):.1f}  trend={mae_yes/len(ctx.countries):.1f}")
print(f"  Improvement: {(mae_no - mae_yes)/mae_no*100:.1f}%")

# CHN specific
chn = ctx.countries.index('CHN')
print(f"\n  CHN: actual={div_actual[chn]}, no_trend={div_no[chn]}, trend={div_yes[chn]}")
print(f"  CHN error: no_trend={div_no[chn]-div_actual[chn]:+d}, trend={div_yes[chn]-div_actual[chn]:+d}")
