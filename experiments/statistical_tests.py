"""Statistical significance tests for semantic extensions."""
import numpy as np
from scipy import stats

# Multi-seed results (5 seeds)
stat_no_sem = [18.052, 18.016, 17.068, 19.396, 17.646]
blend_no_sem = [17.521, 17.385, 16.771, 18.646, 17.615]
blend_sem = [17.021, 16.880, 17.583, 18.406, 17.688]

print("=" * 60)
print("Statistical Significance Tests")
print("=" * 60)

# 1. stat_no_sem vs blend_sem (full semantic extension)
t1, p1 = stats.ttest_rel(stat_no_sem, blend_sem)
diff1 = np.mean(stat_no_sem) - np.mean(blend_sem)
print(f"\n1. Pure statistical vs Full semantic:")
print(f"   Mean MAE: {np.mean(stat_no_sem):.3f} vs {np.mean(blend_sem):.3f}")
print(f"   Diff: {diff1:.3f} ({diff1/np.mean(stat_no_sem)*100:.1f}%)")
print(f"   Paired t-test: t={t1:.3f}, p={p1:.4f} {'*' if p1<0.05 else '(not significant)'}")

# 2. blend_no_sem vs blend_sem (semantic mechanisms on top of blended phi)
t2, p2 = stats.ttest_rel(blend_no_sem, blend_sem)
diff2 = np.mean(blend_no_sem) - np.mean(blend_sem)
print(f"\n2. Blend phi (no sem mech) vs Blend phi + sem mechanisms:")
print(f"   Mean MAE: {np.mean(blend_no_sem):.3f} vs {np.mean(blend_sem):.3f}")
print(f"   Diff: {diff2:.3f} ({diff2/np.mean(blend_no_sem)*100:.1f}%)")
print(f"   Paired t-test: t={t2:.3f}, p={p2:.4f} {'*' if p2<0.05 else '(not significant)'}")

# 3. Non-parametric Wilcoxon
print(f"\n3. Wilcoxon signed-rank tests:")
w1, pw1 = stats.wilcoxon(stat_no_sem, blend_sem)
print(f"   stat vs sem: W={w1}, p={pw1:.4f}")
w2, pw2 = stats.wilcoxon(blend_no_sem, blend_sem)
print(f"   blend vs sem: W={w2}, p={pw2:.4f}")

# 4. Effect size (Cohen's d)
def cohens_d(x, y):
    return (np.mean(x) - np.mean(y)) / np.sqrt((np.std(x)**2 + np.std(y)**2) / 2)

d1 = cohens_d(stat_no_sem, blend_sem)
d2 = cohens_d(blend_no_sem, blend_sem)
print(f"\n4. Effect sizes (Cohen's d):")
print(f"   stat vs sem: d={abs(d1):.3f} ({'large' if abs(d1)>0.8 else 'medium' if abs(d1)>0.5 else 'small'})")
print(f"   blend vs sem: d={abs(d2):.3f} ({'large' if abs(d2)>0.8 else 'medium' if abs(d2)>0.5 else 'small'})")

# 5. Bootstrap confidence interval
print(f"\n5. Bootstrap 95% CI for mean improvement (10000 resamples):")
rng = np.random.default_rng(42)
n = len(stat_no_sem)
boot_diffs = []
for _ in range(10000):
    idx = rng.choice(n, n, replace=True)
    boot_diffs.append(np.mean(np.array(stat_no_sem)[idx]) - np.mean(np.array(blend_sem)[idx]))
boot_diffs = np.array(boot_diffs)
ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
print(f"   stat vs sem: {ci_low:.3f} to {ci_high:.3f}")
print(f"   (CI excludes 0: {'YES' if ci_low > 0 else 'NO'})")

boot_diffs2 = []
for _ in range(10000):
    idx = rng.choice(n, n, replace=True)
    boot_diffs2.append(np.mean(np.array(blend_no_sem)[idx]) - np.mean(np.array(blend_sem)[idx]))
boot_diffs2 = np.array(boot_diffs2)
ci_low2, ci_high2 = np.percentile(boot_diffs2, [2.5, 97.5])
print(f"   blend vs sem: {ci_low2:.3f} to {ci_high2:.3f}")
print(f"   (CI excludes 0: {'YES' if ci_low2 > 0 else 'NO'})")
