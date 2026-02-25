import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("/home/server/tasni/src")))
from tasni.analysis.background_blend_analysis import analytical_blend_limit

df = pd.read_csv("/home/server/tasni/data/processed/final/golden_improved.csv")
row = df[df["designation"].str.contains("J193547")].iloc[0]

w1_target = float(row["w1mpro"])
w2_target = float(row["w2mpro"])
pm_masyr = float(row["pm_total"])
obs_w1 = float(row["trend_w1"]) * 1000
obs_w2 = float(row["trend_w2"]) * 1000

print("\\begin{deluxetable}{lccc}")
print(
    "\\tablecaption{Maximum Blend Contamination vs. Background Brightness for J193547.43+601201.5\\label{tab:blend_j193547}}"
)
print("\\tabletypesize{\\scriptsize}")
print("\\tablehead{")
print(
    "  \\colhead{Background $W2$ (mag)} & \\colhead{Max Fade W1 (mmag yr$^{-1}$)} & \\colhead{Max Fade W2 (mmag yr$^{-1}$)} & \\colhead{\\% of Observed W1 Fade}"
)
print("}")
print("\\startdata")
for w2_bg in np.arange(14.0, 17.5, 0.5):
    res = analytical_blend_limit(w1_target, w2_target, pm_masyr, w2_background=w2_bg)
    w1_fade = res["max_fade_rate_w1_mmag_yr"]
    w2_fade = res["max_fade_rate_w2_mmag_yr"]
    pct = (w1_fade / obs_w1) * 100
    print(f"{w2_bg:.1f} & {w1_fade:.1f} & {w2_fade:.1f} & {pct:.1f}\\% \\\\")
print("\\enddata")
print(
    "\\tablecomments{Maximum apparent fading rates produced by a stationary background source exactly centered on the J193547 position at $t=0$. The target's observed fade rate is $25.8$ mmag yr$^{-1}$ in W1. At the CatWISE completeness limit ($W2 \\approx 15.0$), blending can account for up to $\\sim$82\\% of the observed W1 fade if perfectly aligned. However, if the background source is 1 mag fainter ($W2=16.0$), the maximum contribution drops to $\\approx$33\\%, requiring both a highly specific spatial alignment and a bright, unresolved background source to solely explain the observed trend.}"
)
print("\\end{deluxetable}")
