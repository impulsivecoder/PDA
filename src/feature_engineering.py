import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats.mstats import winsorize

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_PATH   = Path("DATA_processed.csv")                               # <- update if needed
OUTPUT_PATH  = INPUT_PATH.parent / (INPUT_PATH.stem + "_features.csv")
WINDOW       = 30    # rolling window size (days)
WINSOR_LIMIT = 0.01  # clip bottom and top 1% per country

# ─────────────────────────────────────────────
# 1. LOAD & SORT
# ─────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=True)
df = df.sort_values(["country", "date"]).reset_index(drop=True)

print(f"Loaded  : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Countries: {df['country'].unique().tolist()}\n")

# ─────────────────────────────────────────────
# 2. HELPER: ROLLING Z-SCORE
# ─────────────────────────────────────────────
def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Compute a rolling z-score using a sliding window.
    Each value is normalised relative to the local mean and std
    of the preceding `window` days — capturing seasonal norms
    instead of the global distribution.
    min_periods=1 avoids NaNs at series start (except row 0 where std=NaN).
    """
    roll_mean = series.rolling(window=window, min_periods=1).mean()
    roll_std  = series.rolling(window=window, min_periods=1).std()
    return (series - roll_mean) / roll_std.replace(0, np.nan)

# ─────────────────────────────────────────────
# 3. PER-COUNTRY FEATURE ENGINEERING
# ─────────────────────────────────────────────
results = []

for country, group in df.groupby("country"):
    g = group.copy()

    # ── Step A: Winsorize — clip outliers to the 1st and 99th percentile ──
    # This suppresses extreme spikes before computing z-scores so that
    # a single anomalous day doesn't distort the rolling statistics.
    g["daily_count_w"] = winsorize(
        g["daily_count"], limits=[WINSOR_LIMIT, WINSOR_LIMIT]
    )
    g["collisions_w"] = winsorize(
        g["Amount of collisions happen each day"], limits=[WINSOR_LIMIT, WINSOR_LIMIT]
    )

    # ── Step B: Rolling z-score on the winsorized series ──
    # fires_z      → normalised daily_count relative to its 30-day local window
    # collisions_z → normalised collision count relative to its 30-day local window
    g["fires_z"]      = rolling_zscore(g["daily_count_w"],  WINDOW)
    g["collisions_z"] = rolling_zscore(g["collisions_w"],   WINDOW)

    results.append(g)

df_out = (
    pd.concat(results)
    .sort_values(["country", "date"])
    .reset_index(drop=True)
    # Drop intermediate winsorized helpers — not needed in final output
    .drop(columns=["daily_count_w", "collisions_w"])
)

# ─────────────────────────────────────────────
# 4. REPORT
# ─────────────────────────────────────────────
print("=" * 60)
print("FEATURE ENGINEERING REPORT")
print("=" * 60)

print(f"\nOutput shape : {df_out.shape[0]:,} rows × {df_out.shape[1]} columns")
print(f"New columns  : fires_z, collisions_z")

print("\n── NaN counts in new columns (expected: 1 per country) ──")
print(df_out[["fires_z", "collisions_z"]].isna().sum().to_string())

print("\n── Descriptive stats ──")
print(df_out[["fires_z", "collisions_z"]].describe().round(4).to_string())

print("\n── Per-country stats ──")
for col in ["fires_z", "collisions_z"]:
    print(f"\n  {col}:")
    print(
        df_out.groupby("country")[col]
        .agg(["mean", "std", "min", "max"])
        .round(4)
        .to_string()
    )

# ─────────────────────────────────────────────
# 5. SAVE
# ─────────────────────────────────────────────
df_out.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved → {OUTPUT_PATH.resolve()}")
