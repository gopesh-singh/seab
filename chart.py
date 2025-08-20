import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Set Seaborn style and context
sns.set_style("whitegrid")
sns.set_context("talk")

# Generate synthetic customer engagement data
n_samples = 300
data = pd.DataFrame({
    "email_open_rate": np.clip(np.random.normal(0.5, 0.1, n_samples), 0, 1),
    "click_through_rate": np.clip(np.random.normal(0.2, 0.05, n_samples), 0, 1),
    "time_on_site": np.random.normal(120, 30, n_samples),  # in seconds
    "pages_per_visit": np.random.normal(5, 1.5, n_samples),
    "bounce_rate": np.clip(np.random.normal(0.4, 0.1, n_samples), 0, 1),
    "conversion_rate": np.clip(np.random.normal(0.05, 0.02, n_samples), 0, 1),
})

# Compute correlation matrix
corr = data.corr()

# Create figure and heatmap
plt.figure(figsize=(8, 8))  # 8 inches * 64 dpi = 512 pixels
heatmap = sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)

# Add title
plt.title("Customer Engagement Correlation Matrix", fontsize=16)

# Save figure with required dimensions
plt.savefig("chart.png", dpi=64, bbox_inches="tight")
