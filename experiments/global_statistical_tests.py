"""Global scale (231 countries) statistical significance tests with 30 seeds."""
import json
import numpy as np
from scipy import stats

ROOT = __import__('pathlib').Path(__file__).resolve().parents[1]
with open(ROOT / 'experiments' / 'global_semantic_summary.json') as f:
    g = json.load(f)

stat_no_sem = g['robustness']['stat_no_sem']['values']
blend_no_sem = g['robustness']['blend_no_sem']['values']
blend_sem = g['robustness']['blend_sem']['values']

print("=" * 60)
print(f"Global Scale Statistical Tests (n={len(stat_no_sem)} seeds, {g['n_countries']} countries)")
print("=" * 60)

# 1. Paired t-tests
t1, p1 = stats.ttest_rel(stat_no_sem, blend_sem)
diff1 = np.mean(stat_no_sem) - np.mean(blend_sem)
print(f"\n1. Pure statistical vs Full semantic:")
print(f"   Mean MAE: {np.mean(stat_no_sem):.3f} vs {np.mean(blend_sem):.3f}")
print(f"   Diff: {diff1:.3f} ({diff1/np.mean(stat_no_sem)*100:.1f}%)")
print(f"   Paired t-test: t={t1:.3f}, p={p1:.6f} {'***' if p1<0.001 else '**' if p1<0.01 else '*' if p1<0.05 else '(n.s.)'}")

t2, p2 = stats.ttest_rel(blend_no_sem, blend_sem)
diff2 = np.mean(blend_no_sem) - np.mean(blend_sem)
print(f"\n2. Blend phi (no sem mech) vs Blend phi + sem mechanisms:")
print(f"   Mean MAE: {np.mean(blend_no_sem):.3f} vs {np.mean(blend_sem):.3f}")
print(f"   Diff: {diff2:.3f} ({diff2/np.mean(blend_no_sem)*100:.1f}%)")
print(f"   Paired t-test: t={t2:.3f}, p={p2:.6f} {'***' if p2<0.001 else '**' if p2<0.01 else '*' if p2<0.05 else '(n.s.)'}")

t3, p3 = stats.ttest_rel(stat_no_sem, blend_no_sem)
diff3 = np.mean(stat_no_sem) - np.mean(blend_no_sem)
print(f"\n3. Pure statistical vs Blend phi (proximity fusion only):")
print(f"   Mean MAE: {np.mean(stat_no_sem):.3f} vs {np.mean(blend_no_sem):.3f}")
print(f"   Diff: {diff3:.3f} ({diff3/np.mean(stat_no_sem)*100:.1f}%)")
print(f"   Paired t-test: t={t3:.3f}, p={p3:.6f} {'***' if p3<0.001 else '**' if p3<0.01 else '*' if p3<0.05 else '(n.s.)'}")

# 2. Wilcoxon
print(f"\n4. Wilcoxon signed-rank tests:")
w1, pw1 = stats.wilcoxon(stat_no_sem, blend_sem)
print(f"   stat vs sem: W={w1}, p={pw1:.6f} {'***' if pw1<0.001 else '**' if pw1<0.01 else '*' if pw1<0.05 else '(n.s.)'}")
w2, pw2 = stats.wilcoxon(blend_no_sem, blend_sem)
print(f"   blend vs sem: W={w2}, p={pw2:.6f} {'***' if pw2<0.001 else '**' if pw2<0.01 else '*' if pw2<0.05 else '(n.s.)'}")

# 3. Effect sizes
def cohens_d(x, y):
    return (np.mean(x) - np.mean(y)) / np.sqrt((np.std(x)**2 + np.std(y)**2) / 2)

d1 = cohens_d(stat_no_sem, blend_sem)
d2 = cohens_d(blend_no_sem, blend_sem)
d3 = cohens_d(stat_no_sem, blend_no_sem)
print(f"\n5. Effect sizes (Cohen's d):")
print(f"   stat vs sem: d={abs(d1):.3f} ({'large' if abs(d1)>0.8 else 'medium' if abs(d1)>0.5 else 'small'})")
print(f"   blend vs sem: d={abs(d2):.3f} ({'large' if abs(d2)>0.8 else 'medium' if abs(d2)>0.5 else 'small'})")
print(f"   stat vs blend: d={abs(d3):.3f} ({'large' if abs(d3)>0.8 else 'medium' if abs(d3)>0.5 else 'small'})")

# 4. Bootstrap CI
print(f"\n6. Bootstrap 95% CI for mean improvement (10000 resamples):")
rng = np.random.default_rng(42)
n = len(stat_no_sem)

boot1 = []
for _ in range(10000):
    idx = rng.choice(n, n, replace=True)
    boot1.append(np.mean(np.array(stat_no_sem)[idx]) - np.mean(np.array(blend_sem)[idx]))
boot1 = np.array(boot1)
ci1 = np.percentile(boot1, [2.5, 97.5])
print(f"   stat vs sem: {ci1[0]:.3f} to {ci1[1]:.3f} (excludes 0: {'YES' if ci1[0]>0 else 'NO'})")

boot2 = []
for _ in range(10000):
    idx = rng.choice(n, n, replace=True)
    boot2.append(np.mean(np.array(blend_no_sem)[idx]) - np.mean(np.array(blend_sem)[idx]))
boot2 = np.array(boot2)
ci2 = np.percentile(boot2, [2.5, 97.5])
print(f"   blend vs sem: {ci2[0]:.3f} to {ci2[1]:.3f} (excludes 0: {'YES' if ci2[0]>0 else 'NO'})")

boot3 = []
for _ in range(10000):
    idx = rng.choice(n, n, replace=True)
    boot3.append(np.mean(np.array(stat_no_sem)[idx]) - np.mean(np.array(blend_no_sem)[idx]))
boot3 = np.array(boot3)
ci3 = np.percentile(boot3, [2.5, 97.5])
print(f"   stat vs blend: {ci3[0]:.3f} to {ci3[1]:.3f} (excludes 0: {'YES' if ci3[0]>0 else 'NO'})")

# 5. Direction consistency
all_stat = np.array(stat_no_sem)
all_sem = np.array(blend_sem)
all_blend = np.array(blend_no_sem)
print(f"\n7. Direction consistency (30 seeds):")
print(f"   stat > sem (semantic helps): {(all_stat > all_sem).sum()}/30 = {(all_stat > all_sem).mean()*100:.0f}%")
print(f"   blend > sem (sem mech helps): {(all_blend > all_sem).sum()}/30 = {(all_blend > all_sem).mean()*100:.0f}%")
print(f"   stat > blend (fusion helps): {(all_stat > all_blend).sum()}/30 = {(all_stat > all_blend).mean()*100:.0f}%")

# 6. Comparison with 24-country results
print(f"\n8. Comparison: 24 countries vs 231 countries")
print(f"   24 countries (5 seeds):")
print(f"     stat: 18.04 ± 0.77 | blend: 17.59 ± 0.61 | sem: 17.52 ± 0.54")
print(f"   231 countries (30 seeds):")
print(f"     stat: {np.mean(stat_no_sem):.2f} ± {np.std(stat_no_sem):.2f} | blend: {np.mean(blend_no_sem):.2f} ± {np.std(blend_no_sem):.2f} | sem: {np.mean(blend_sem):.2f} ± {np.std(blend_sem):.2f}")
print(f"   Improvement from scaling: {(18.04 - np.mean(stat_no_sem))/18.04*100:.1f}% (stat baseline)")
