"""查看 baseline 结果。"""
import numpy as np
import pandas as pd

hist = pd.read_parquet("experiments/baseline_history.parquet")
comp = pd.read_csv("experiments/baseline_vs_history.csv")

print("=== 演化序列（每 4 步） ===")
print(hist.iloc[::4][["step", "mean_diversity", "std_diversity", "mean_tech", "total_capabilities"]].to_string(index=False))

print()
print("=== 模型末步 vs 历史 2023 多样性（全部 24 国） ===")
print(comp.to_string(index=False))

print()
print("=== 差异统计 ===")
diff = comp["diff"]
print(f"MAE: {diff.abs().mean():.1f}")
print(f"RMSE: {np.sqrt((diff ** 2).mean()):.1f}")
print(f"模型高估（diff>0）：{(diff > 0).sum()}/{len(diff)} 国家")
print(f"模型低估（diff<0）：{(diff < 0).sum()}/{len(diff)} 国家")
print(f"平均相对误差：{(diff.abs() / comp['historical_2023_diversity'].clip(lower=1)).mean():.2%}")
