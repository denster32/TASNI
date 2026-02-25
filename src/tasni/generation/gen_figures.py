import matplotlib
import numpy as np
import pandas as pd

from tasni.core.config import OUTPUT_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv(str(OUTPUT_DIR / "golden_targets.csv"))
out_dir = OUTPUT_DIR / "figures"
out_dir.mkdir(exist_ok=True)

print("Generating figures...")

# Figure 1: Sky distribution
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection="aitoff")
ra_rad = np.radians(df["ra"].values)
dec_rad = np.radians(df["dec"].values)
scatter = ax.scatter(ra_rad - np.pi, dec_rad, c=df["prime_score"], cmap="plasma", s=60, alpha=0.8)
plt.colorbar(scatter, ax=ax, shrink=0.6, label="prime_score")
ax.set_title("Sky Distribution of TASNI Thermal Anomaly Candidates")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(out_dir / "fig1_sky_distribution.png"), dpi=300, bbox_inches="tight")
plt.savefig(str(out_dir / "fig1_sky_distribution.pdf"))
plt.close()
print("  Fig 1: Sky distribution")

# Figure 2: WISE colors
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax1 = axes[0]
ax1.scatter(
    df["w1_w2_color"],
    df["w2_w3"] if "w2_w3" in df.columns else df["w2mpro"] - df["w3mpro"],
    c=df["prime_score"],
    cmap="plasma",
    s=40,
    alpha=0.7,
)
ax1.set_xlabel("W1 - W2 (mag)")
ax1.set_ylabel("W2 - W3 (mag)")
ax1.set_title("WISE Color-Color Diagram")
ax1.grid(True, alpha=0.3)
ax2 = axes[1]
ax2.hist(df["w1_w2_color"], bins=25, edgecolor="black", alpha=0.7, color="steelblue")
ax2.set_xlabel("W1 - W2 (mag)")
ax2.set_ylabel("Count")
ax2.set_title("W1 - W2 Distribution")
ax2.axvline(
    df["w1_w2_color"].mean(),
    color="red",
    linestyle="--",
    label=f'Mean: {df["w1_w2_color"].mean():.2f}',
)
ax2.legend()
plt.tight_layout()
plt.savefig(str(out_dir / "fig2_wise_colors.png"), dpi=300, bbox_inches="tight")
plt.savefig(str(out_dir / "fig2_wise_colors.pdf"))
plt.close()
print("  Fig 2: WISE colors")

# Figure 3: Temperature
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df["T_eff_K"], bins=30, edgecolor="black", alpha=0.7, color="steelblue")
ax.set_xlabel("Effective Temperature (K)")
ax.set_ylabel("Count")
ax.set_title("Temperature Distribution")
ax.axvline(300, color="red", linestyle="--", label="300 K")
ax.axvline(
    df["T_eff_K"].mean(), color="green", linestyle="--", label=f'Mean: {df["T_eff_K"].mean():.0f} K'
)
ax.legend()
plt.tight_layout()
plt.savefig(str(out_dir / "fig3_temperature.png"), dpi=300, bbox_inches="tight")
plt.savefig(str(out_dir / "fig3_temperature.pdf"))
plt.close()
print("  Fig 3: Temperature")

# Figure 4: Score distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax1 = axes[0]
ax1.hist(df["prime_score"], bins=25, edgecolor="black", alpha=0.7)
ax1.set_xlabel("prime_score")
ax1.set_ylabel("Count")
ax1.set_title("Score Distribution")
ax2 = axes[1]
ax2.scatter(df["w1_w2_color"], df["prime_score"], c="steelblue", s=30, alpha=0.7)
ax2.set_xlabel("W1 - W2 (mag)")
ax2.set_ylabel("prime_score")
plt.tight_layout()
plt.savefig(str(out_dir / "fig4_score_distribution.png"), dpi=300, bbox_inches="tight")
plt.savefig(str(out_dir / "fig4_score_distribution.pdf"))
plt.close()
print("  Fig 4: Score distribution")

print(f"\nFigures saved to {out_dir}")
