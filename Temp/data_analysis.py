import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

# Set input path — output will be saved to the same folder automatically
INPUT_PATH  = Path("DATA.csv")                                          # <- update if needed
OUTPUT_PATH = INPUT_PATH.parent / (INPUT_PATH.stem + "_processed.csv") # same folder, new name

df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

print("=" * 60)
print("STEP 1 — RAW DATA LOADED")
print("=" * 60)
print(f"Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}\n")

# ─────────────────────────────────────────────
# 2. CONVERT 'date' COLUMN TO DATETIME
# ─────────────────────────────────────────────
df["date"] = pd.to_datetime(df["date"], dayfirst=True)

print("=" * 60)
print("STEP 2 — 'date' COLUMN CONVERTED TO DATETIME")
print("=" * 60)
print(f"Date range : {df['date'].min().date()}  →  {df['date'].max().date()}")
print(f"dtype      : {df['date'].dtype}\n")

# ─────────────────────────────────────────────
# 3. SORT BY country (alphabetical) THEN date (chronological)
# ─────────────────────────────────────────────
df = df.sort_values(by=["country", "date"]).reset_index(drop=True)

print("=" * 60)
print("STEP 3 — SORTED: country (A→Z) then date (oldest→newest)")
print("=" * 60)
print(df[["date", "country"]].head(6).to_string(index=True))
print("...\n")

# ─────────────────────────────────────────────
# 4. FILL MISSING VALUES WITH PER-COUNTRY MEAN
# ─────────────────────────────────────────────

# Identify numeric columns eligible for mean-fill
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Count NaNs before filling
nan_before = df[numeric_cols].isna().sum()
total_nan_before = nan_before.sum()

# Fill each country's NaNs with that country's column mean
df[numeric_cols] = (
    df.groupby("country")[numeric_cols]
    .transform(lambda x: x.fillna(x.mean()))
)

# Count NaNs after filling
nan_after = df[numeric_cols].isna().sum()
total_nan_after = nan_after.sum()

print("=" * 60)
print("STEP 4 — MISSING VALUES FILLED WITH PER-COUNTRY MEAN")
print("=" * 60)
print(f"Total NaNs BEFORE : {total_nan_before:,}")
print(f"Total NaNs AFTER  : {total_nan_after:,}")

if nan_before[nan_before > 0].empty:
    print("  → No missing values were found in numeric columns.")
else:
    print("\nPer-column breakdown (only columns that had NaNs):")
    summary = pd.DataFrame({
        "Before": nan_before[nan_before > 0],
        "After" : nan_after[nan_before > 0]
    })
    print(summary.to_string())
print()

# ─────────────────────────────────────────────
# 5. DATASET STRUCTURE REPORT
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 5 — DATASET STRUCTURE REPORT")
print("=" * 60)

print("\n── Column dtypes ──")
print(df.dtypes.to_string())

print("\n── Numeric summary ──")
print(df[numeric_cols].describe().round(3).to_string())

print("\n── Row count per country ──")
country_counts = (
    df.groupby("country")
      .size()
      .reset_index(name="row_count")
      .sort_values("row_count", ascending=False)
)
country_counts.index = range(1, len(country_counts) + 1)
print(country_counts.to_string())

print(f"\nTotal countries : {df['country'].nunique()}")
print(f"Total rows      : {len(df):,}")
print(f"Date range      : {df['date'].min().date()}  →  {df['date'].max().date()}")

# ─────────────────────────────────────────────
# 6. SAVE PROCESSED DATA — same folder as input
# ─────────────────────────────────────────────
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nProcessed file saved → {OUTPUT_PATH.resolve()}")