import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_PATH   = Path("DATA_processed_severity.csv")
OUTPUT_CSV   = INPUT_PATH.parent / "final_processed_data.csv"
OUTPUT_CHART = INPUT_PATH.parent / "Y_class_distribution.png"
THRESHOLD_Q  = 0.80   # 80th percentile per country

# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=True)
df = df.sort_values(["country", "date"]).reset_index(drop=True)

print(f"Loaded : {df.shape[0]:,} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 2. Y_class: per-country 80th-percentile threshold
# ─────────────────────────────────────────────
# Compute threshold per country and merge back — avoids groupby.apply
# dropping the group key column (pandas 2.x behaviour).
thresholds = (
    df.groupby("country")["Y_severity"]
    .quantile(THRESHOLD_Q)
    .rename("_threshold")
    .reset_index()
)
df = df.merge(thresholds, on="country", how="left")

# Label:  1 = High Risk (≥ threshold),  0 = Normal (< threshold)
df["Y_class"] = (df["Y_severity"] >= df["_threshold"]).astype(int)
df = df.drop(columns=["_threshold"])

# ─────────────────────────────────────────────
# 3. REPORT
# ─────────────────────────────────────────────
threshold_records = []
for country, grp in df.groupby("country"):
    t  = grp.merge(thresholds, on="country")["_threshold"].iloc[0]
    n1 = (grp["Y_class"] == 1).sum()
    n0 = (grp["Y_class"] == 0).sum()
    threshold_records.append({
        "country"    : country,
        "threshold"  : round(grp.merge(
                         df.groupby("country")["Y_severity"]
                         .quantile(THRESHOLD_Q).rename("t").reset_index(),
                         on="country")["t"].iloc[0], 4),
        "High Risk 1": n1,
        "Normal 0"   : n0,
        "pct_high %"  : round(n1 / len(grp) * 100, 1),
    })

# Cleaner: recompute thresholds inline
threshold_records = []
for country, grp in df.groupby("country"):
    thresh = grp["Y_severity"].quantile(THRESHOLD_Q)
    n1 = (grp["Y_class"] == 1).sum()
    n0 = (grp["Y_class"] == 0).sum()
    threshold_records.append({
        "country"    : country,
        "threshold"  : round(thresh, 4),
        "Normal (0)" : n0,
        "High Risk (1)": n1,
        "High Risk %" : round(n1 / len(grp) * 100, 1),
    })

report_df = pd.DataFrame(threshold_records).set_index("country")

print("\n" + "=" * 55)
print(f"Y_class DISTRIBUTION  (threshold = P{int(THRESHOLD_Q*100)} per country)")
print("=" * 55)
print(report_df.to_string())

# ─────────────────────────────────────────────
# 4. BAR CHART
# ─────────────────────────────────────────────
countries = report_df.index.tolist()
n = len(countries)

COLORS = {"Normal (0)": "#4C9BE8", "High Risk (1)": "#E85C5C"}

fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.8), sharey=False)
if n == 1:
    axes = [axes]

for ax, country in zip(axes, countries):
    row    = report_df.loc[country]
    labels = ["Normal (0)", "High Risk (1)"]
    vals   = [row["Normal (0)"], row["High Risk (1)"]]
    max_v  = max(vals)

    bars = ax.bar(
        labels, vals,
        color=[COLORS[l] for l in labels],
        width=0.52, alpha=0.88,
        edgecolor="white", linewidth=1.4
    )

    # Count labels above bars
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_v * 0.018,
            f"{v:,}",
            ha="center", va="bottom",
            fontsize=11.5, fontweight="bold", color="#222"
        )

    # % labels inside bars
    pct_h = row["High Risk %"]
    for i, (bar, pct) in enumerate(zip(bars, [100 - pct_h, pct_h])):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{pct:.1f}%",
            ha="center", va="center",
            fontsize=10.5, color="white", fontweight="bold"
        )

    ax.set_title(
        f"{country}\nP{int(THRESHOLD_Q*100)} threshold: {row['threshold']:.4f}",
        fontsize=13, fontweight="bold", pad=10
    )
    ax.set_ylabel("Number of Days", fontsize=10.5)
    ax.set_ylim(0, max_v * 1.16)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=9)

fig.suptitle(
    "Y_class Distribution per Country\n0 = Normal  |  1 = High Risk",
    fontsize=15, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig(OUTPUT_CHART, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nChart saved → {OUTPUT_CHART.resolve()}")

# ─────────────────────────────────────────────
# 5. SAVE
# ─────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
print(f"CSV saved  → {OUTPUT_CSV.resolve()}")
print(f"Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Columns    : {df.columns.tolist()}")
