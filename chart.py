import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Professional Seaborn styling
sns.set_style("whitegrid")
sns.set_context("talk")  # presentation-ready text sizes

# Seed for reproducibility
rng = np.random.default_rng(42)

# Generate realistic synthetic data for customer spending behavior
# Segments: Value, Mid, Premium, VIP
# Use log-normal-like right-skew typical of purchase amounts
segments = [
    {"name": "Value", "mu": 3.2, "sigma": 0.55, "n": 400},     # lower spend
    {"name": "Mid", "mu": 3.8, "sigma": 0.50, "n": 400},       # mid spend
    {"name": "Premium", "mu": 4.3, "sigma": 0.45, "n": 350},   # higher spend
    {"name": "VIP", "mu": 4.8, "sigma": 0.40, "n": 250},       # highest spend
]

records = []
for seg in segments:
    # Base purchase amounts via log-normal
    base = rng.lognormal(mean=seg["mu"], sigma=seg["sigma"], size=seg["n"])

    # Add realistic noise and clamp to minimum of $1
    noise = rng.normal(0, 5, size=seg["n"])
    amounts = np.clip(base + noise, 1, None)

    # Introduce occasional high-ticket outliers per segment
    outlier_count = max(1, seg["n"] // 50)
    outliers = rng.lognormal(mean=seg["mu"] + 1.1, sigma=seg["sigma"] + 0.3, size=outlier_count)
    amounts[:outlier_count] = amounts[:outlier_count] + outliers

    for a in amounts:
        records.append({"Segment": seg["name"], "Purchase Amount ($)": a})

df = pd.DataFrame(records)

# Optional winsorized cap for extreme tails to avoid visual domination
# Keep true outliers visible but cap truly extreme values
upper_cap = df.groupby("Segment")["Purchase Amount ($)"].quantile(0.995)
df["Cap"] = df["Segment"].map(upper_cap)
df["Purchase Amount ($)"] = np.where(
    df["Purchase Amount ($)"] > df["Cap"], df["Cap"], df["Purchase Amount ($)"]
)
df = df.drop(columns=["Cap"])

# Order segments from lower to higher value
order = ["Value", "Mid", "Premium", "VIP"]

# Color palette suitable for executives: muted yet distinct
palette = sns.color_palette("Blues", n_colors=len(order))

# Create square figure so that with dpi=64 it saves as 512x512
plt.figure(figsize=(8, 8))

# Boxplot emphasizing distribution and outliers
ax = sns.boxplot(
    data=df,
    x="Segment",
    y="Purchase Amount ($)",
    order=order,
    palette=palette,
    width=0.6,
    showcaps=True,
    boxprops={"linewidth": 1.5},
    whiskerprops={"linewidth": 1.5},
    medianprops={"linewidth": 2, "color": "#222222"},
    flierprops={"marker": "o", "markersize": 4, "alpha": 0.5}
)

# Titles and labels for executive audiences
ax.set_title("Distribution of Purchase Amounts by Customer Segment", pad=16, weight="bold")
ax.set_xlabel("Customer Segment")
ax.set_ylabel("Purchase Amount ($)")

# Ticks formatting for currency-like values
y_ticks = ax.get_yticks()
ax.set_yticklabels([f"${int(t):,}" if t >= 1 else f"${t:.2f}" for t in y_ticks])

# Subtle grid and clean axes
sns.despine(offset=5, trim=True)

# Tight layout and exact pixel export: 8in * 64dpi = 512px
plt.tight_layout()
plt.savefig("chart.png", dpi=64, bbox_inches="tight")
plt.close()
