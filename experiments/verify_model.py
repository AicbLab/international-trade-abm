"""Verify trend-corrected model performance."""
import sys, numpy as np
sys.path.insert(0, 'src')
from research import ResearchContext, DEFAULT_THETA

ctx = ResearchContext()
actual = ctx.actual.sum(axis=2).astype(np.int32)

# Growth trends
print("Growth trends (selected):")
for c in ['CHN', 'VNM', 'HKG', 'JPN', 'SGP']:
    print(f"  {c}: {ctx.growth_trends[c]:+.2f}/yr")

# Model performance
states, _ = ctx.simulate(DEFAULT_THETA, seed=42)
div = states.sum(axis=2).astype(np.int32)

mae = np.mean(np.abs(div[-1] - actual[-1]))
print(f"\nOverall MAE (2023): {mae:.1f}")

chn_i = ctx.countries.index('CHN')
print(f"CHN: pred={div[-1,chn_i]} actual={actual[-1,chn_i]} err={div[-1,chn_i]-actual[-1,chn_i]:+d}")

# Per-country errors
print("\nPer-country errors (2023):")
errs = div[-1] - actual[-1]
for i, c in enumerate(ctx.countries):
    print(f"  {c}: pred={div[-1,i]} actual={actual[-1,i]} err={errs[i]:+d}")
