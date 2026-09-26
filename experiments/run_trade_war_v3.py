"""
贸易战验证 v3：使用增长趋势修正后的模型
==========================================
"""
import sys, json, time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from model import PolicyConfig
from research import ResearchContext, DEFAULT_THETA, BASE_YEAR, END_YEAR
from pipeline import PIPELINE_VERSION

EXPERIMENTS = ROOT / 'experiments'

def diversity(states):
    return states.sum(axis=2).astype(np.int32)

def main():
    print("=" * 60)
    print("Trade War Validation v3: Trend-Corrected Model")
    print("=" * 60)

    ctx = ResearchContext()
    countries = ctx.countries
    products = ctx.products
    chn_idx = countries.index('CHN')
    theta = DEFAULT_THETA
    actual = ctx.actual.sum(axis=2).astype(np.int32)

    # Semiconductor targets
    semi_targets = [p for p in ['8542', '8541', '8534', '8536', '8537'] if p in products]
    # Broader trade war targets
    broad_targets = [p for p in [
        '8471', '8473', '8517', '8523', '8525', '8526', '8528', '8529',
        '8534', '8536', '8537', '8541', '8542',
        '8501', '8502', '8503', '8504',
    ] if p in products]

    print(f"\nTargets: semi={len(semi_targets)}, broad={len(broad_targets)}")

    # ============================================================
    # Scenario 1: Baseline with trend correction
    # ============================================================
    print("\n--- Scenario 1: Baseline (trend-corrected) ---")
    states_base, _ = ctx.simulate(theta, seed=42, use_trends=True)
    div_base = diversity(states_base)

    # ============================================================
    # Scenario 2: Trade war (blockade semiconductors from 2018)
    # ============================================================
    print("\n--- Scenario 2: Trade war (semi blockade, 2018) ---")
    policy_war = PolicyConfig(
        blockade_targets=semi_targets,
        blockade_strength=0.2,
        countries=['CHN'],
        start_step=2018 - BASE_YEAR,
    )
    states_war, _ = ctx.simulate(theta, seed=42, policy=policy_war, use_trends=True)
    div_war = diversity(states_war)

    # ============================================================
    # Scenario 3: Trade war (broad blockade from 2018)
    # ============================================================
    print("\n--- Scenario 3: Trade war (broad blockade, 2018) ---")
    policy_broad = PolicyConfig(
        blockade_targets=broad_targets,
        blockade_strength=0.2,
        countries=['CHN'],
        start_step=2018 - BASE_YEAR,
    )
    states_broad, _ = ctx.simulate(theta, seed=42, policy=policy_broad, use_trends=True)
    div_broad = diversity(states_broad)

    # ============================================================
    # Scenario 4: Stronger blockade
    # ============================================================
    print("\n--- Scenario 4: Stronger blockade (0.5, broad) ---")
    policy_strong = PolicyConfig(
        blockade_targets=broad_targets,
        blockade_strength=0.5,
        countries=['CHN'],
        start_step=2018 - BASE_YEAR,
    )
    states_strong, _ = ctx.simulate(theta, seed=42, policy=policy_strong, use_trends=True)
    div_strong = diversity(states_strong)

    # ============================================================
    # Results
    # ============================================================
    print("\n" + "=" * 60)
    print("Results: CHN diversity comparison")
    print("=" * 60)
    print(f"\n  {'Year':>6}  {'Actual':>6}  {'Base':>6}  {'Semi':>6}  {'Broad':>6}  {'Strong':>6}")
    for year in range(2017, 2024):
        t = year - BASE_YEAR
        a = actual[t, chn_idx]
        b = div_base[t, chn_idx]
        w = div_war[t, chn_idx]
        br = div_broad[t, chn_idx]
        s = div_strong[t, chn_idx]
        marker = " <--" if year == 2018 else ""
        print(f"  {year:>6}  {a:>6}  {b:>6}  {w:>6}  {br:>6}  {s:>6}{marker}")

    # 2023 summary
    a23 = actual[-1, chn_idx]
    b23 = div_base[-1, chn_idx]
    w23 = div_war[-1, chn_idx]
    br23 = div_broad[-1, chn_idx]
    s23 = div_strong[-1, chn_idx]

    print(f"\n2023 effects on CHN:")
    print(f"  Actual: {a23}")
    print(f"  Baseline (trend): {b23} (error: {b23-a23:+d})")
    print(f"  Semi blockade: {w23} (effect: {w23-b23:+d})")
    print(f"  Broad blockade: {br23} (effect: {br23-b23:+d})")
    print(f"  Strong blockade: {s23} (effect: {s23-b23:+d})")

    # All countries comparison
    print(f"\n{'='*60}")
    print("All countries: trend-corrected vs actual (2023)")
    print(f"{'='*60}")
    div_final = div_base[-1]
    mae = 0
    for i, c in enumerate(countries):
        err = div_final[i] - actual[-1, i]
        mae += abs(err)
        print(f"  {c:>4}: actual={actual[-1,i]:5d}  model={div_final[i]:5d}  err={err:+5d}")
    print(f"\n  Mean MAE: {mae/len(countries):.1f}")

    # Save summary
    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "model": "trend-corrected",
        "chn_2023": {
            "actual": int(a23),
            "baseline": int(b23),
            "semi_blockade": int(w23),
            "broad_blockade": int(br23),
            "strong_blockade": int(s23),
        },
        "effects": {
            "semi_blockade": int(w23 - b23),
            "broad_blockade": int(br23 - b23),
            "strong_blockade": int(s23 - b23),
        },
        "mean_mae": float(mae / len(countries)),
    }
    (EXPERIMENTS / "trade_war_v3.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: experiments/trade_war_v3.json")

if __name__ == "__main__":
    main()
