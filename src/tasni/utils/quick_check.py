#!/usr/bin/env python3

import pandas as pd

from tasni.core.config import OUTPUT_DIR

df = pd.read_csv(str(OUTPUT_DIR / "golden_targets.csv"))
print(f"Targets: {len(df)}")
print(f"Columns: {list(df.columns)[:10]}")
print(f"Has prime_score: {'prime_score' in df.columns}")
print(df.head(3))
