#!/usr/bin/env python3
from pathlib import Path

import pandas as pd

golden_path = Path("data/processed/final/golden_improved.parquet")
df = pd.read_parquet(golden_path)

print("Phase 3 Golden Metrics:")
print(f"Count: {len(df)}")
print(f"Mean ML score: {df['improved_composite_score'].mean():.3f}")
print(f"Top ML score: {df['improved_composite_score'].max():.3f}")
print(f"Mean PM: {df['pm_total'].mean():.0f} mas/yr")
if "neowise_parallax_mas" in df.columns:
    sig_plx = df["neowise_parallax_mas"].dropna()
    print(f"Mean NEOWISE parallax (non-null): {sig_plx.mean():.1f} mas ({len(sig_plx)} sources)")
else:
    print("No neowise_parallax_mas column found")

if len(df) < 100:
    raise ValueError(f"Expected >= 100 golden candidates, got {len(df)}")
print("Metrics validation passed")
